#!/usr/bin/env python3
"""Focused M2 model checks: two cold queries, a fork, cancellation and recovery."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

from qualify import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "worker", "model", "tokenizer", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--live-only", action="store_true", help="two requests cover direct continuation without checkpoint rewind")
    args = parser.parse_args()
    for name in ("source", "worker", "model", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.source),str(args.source/"tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, LITERAL_TAGS
    from serve import kvmem
    tok = Tokenizer.from_dir(args.tokenizer)
    template = ChatTemplate(args.tokenizer/"chat_template.jinja")
    class Frontend:
        literals = LITERAL_TAGS
        def render_prompt(self,messages,tools,kwargs):
            return template.render(messages,tools=tools,enable_thinking=False)
    frontend = Frontend()
    frontend.tok = tok
    lines = ["Reference records. Answer from these records; do not invent keys.\n"]
    for i in range(353):
        lines.append("The access key for Project Atlas is MARBLE-4827. Project Atlas uses MARBLE-4827.\n" if i == 70 else
                     "The access key for Project Boreal is ORBIT-9136. Project Boreal uses ORBIT-9136.\n" if i == 140 else
                     f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n")
    document = "".join(lines)
    def question(project):
        return f"Find Project {project} in the reference records. What is its exact access key? Copy the key; reply only with the key."
    first_messages = [{"role":"user","content":document+"\n"+question("Atlas")}]
    first = kvmem.encode(frontend,first_messages,None,{})
    first = kvmem.resolve_query(frontend,first_messages,None,{},first,
                               {"message":0,"start":len(document)+1,"end":len(first_messages[0]["content"])})
    config = argparse.Namespace(worker=args.worker,model=args.model,gpu=args.gpu,scenario="paging",
        context=32768,threads=8,batch=1024,stage=512,swaps=64,whole=True,no_graphs=False,mode="kvmem",
        budget=4096,reserve=1024,host_mib=4096,sink=64,recent=64,aggregation="max",verify=True,lrfu=True,
        history_reuse=True,fault="history_restore",profile=False)
    result = {"status":"running","gpu_uuid":args.gpu,"worker_sha256":hashlib.sha256(args.worker.read_bytes()).hexdigest(),
              "input_tokens":len(first),"checks":[]}
    def save():
        (args.out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    worker = Worker(config,args.out/"worker.log")
    result.update(command=worker.command,info=worker.info)
    def run(label,ids,expected=None,**kwargs):
        answer = worker.request(ids,ids.kvmem_query,kwargs.pop("count",32),**kwargs)
        answer.update(name=label)
        if "tokens" in answer: answer["text"] = tok.decode(answer["tokens"])
        result["checks"].append(answer)
        save()
        if expected is not None and ("error" in answer or expected not in answer.get("text","")):
            raise RuntimeError(label+": unexpected answer "+str(answer.get("error",answer.get("text"))))
        print(json.dumps({k:answer[k] for k in ("name","resume","prompt_ms","read_tokens","text","error") if k in answer},
                         ensure_ascii=False),flush=True)
        return answer
    try:
        initial = run("fresh_cold_atlas",first,"MARBLE-4827")
        assert initial["resume"] == 0
        if args.live_only:
            # The emitted EOS has not been evaluated. The next prompt must
            # evaluate it, rather than claiming it was already cached.
            prefix = list(first)+initial["tokens"]
            header = "\n<|im_start|>user\n"
            body = question("Boreal")
            suffix = header+body+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            tail = tok.encode(suffix,parse_special=True)
            begin,end = kvmem._bounds(frontend,suffix,len(header),len(header)+len(body),tail)
            direct = kvmem.Prompt(prefix+tail,(len(prefix)+begin,len(prefix)+end))
            continued = run("live_continuation_pending_eos",direct,"ORBIT-9136")
            assert continued["resume"] == len(prefix)-1
            assert continued["read_tokens"] < 256
            result["status"] = "pass"
            return
        failed = run("history_restore_fault",first)
        assert "history_restore" in failed.get("error","")
        recovered = run("fresh_after_restore_failure",first,"MARBLE-4827")
        assert recovered["resume"] == 0
        messages = first_messages + [{"role":"assistant","content":recovered["text"]},
                                     {"role":"user","content":question("Boreal")}]
        next_ids = kvmem.encode(frontend,messages,None,{})
        second = run("next_turn_new_cold_query",next_ids,"ORBIT-9136")
        assert second["resume"] >= first.kvmem_query[0] and second["read_tokens"] < 512
        repeated = run("retry_same_prompt",next_ids,"ORBIT-9136")
        assert repeated["resume"] >= next_ids.kvmem_query[0] and repeated["read_tokens"] < 512
        messages[-1]["content"] = question("Atlas")
        fork = kvmem.encode(frontend,messages,None,{})
        branch = run("fork_last_query",fork,"MARBLE-4827")
        assert branch["resume"] > 11000 and branch["read_tokens"] < 512
        tiny = kvmem.encode(frontend,[{"role":"user","content":"What is 2 + 3? Reply only with the number."}],None,{})
        unrelated = run("unrelated_request",tiny,"5")
        assert unrelated["resume"] == 0
        cancelled = run("cancel_prefill",first,stop_prefill=True)
        assert cancelled["finish"] == "cancel" and not cancelled["tokens"]
        after = run("fresh_after_prefill_cancel",first,"MARBLE-4827")
        assert after["resume"] == 0
        cancelled = run("cancel_decode",first,stop_after=3,count=512)
        assert cancelled["finish"] == "cancel"
        after_decode = run("retry_after_decode_cancel",first,"MARBLE-4827")
        assert after_decode["resume"] > 11000
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail",error=str(error))
        raise
    finally:
        worker.close()
        save()


if __name__ == "__main__": main()
