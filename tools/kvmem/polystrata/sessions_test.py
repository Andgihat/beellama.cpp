#!/usr/bin/env python3
"""Focused retained-history checks for RAM and optional shared SSD backing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from qualify import Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","worker","model","tokenizer","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--scenario",choices=("switch","quota","fault-park","fault-enter",
                                               "ssd-switch","fault-disk-read","fault-disk-write","ssd-capacity"),
                        default="switch")
    parser.add_argument("--mtp-drafts",type=int,choices=range(4),default=0)
    args = parser.parse_args()
    for name in ("source","worker","model","tokenizer","out"):
        setattr(args,name,getattr(args,name).resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path[:0] = [str(args.source),str(args.source/"tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate,LITERAL_TAGS
    from serve import kvmem
    tok = Tokenizer.from_dir(args.tokenizer)
    template = ChatTemplate(args.tokenizer/"chat_template.jinja")
    class Frontend:
        literals = LITERAL_TAGS
        def render_prompt(self,messages,tools,kwargs):
            return template.render(messages,tools=tools,enable_thinking=False)
    frontend = Frontend()
    frontend.tok = tok
    fixtures = {}
    for name,key in (("A","MARBLE-4827"),("B","ONYX-9136"),("C","LANTERN-7052")):
        document = f"Reference set {name}. Answer only from this set's records.\n"+"".join(
            f"The access key for Project Atlas is {key}. Project Atlas uses {key}.\n" if i == 70 else
            f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n"
            for i in range(257))
        question = "Find Project Atlas in the reference records. What is its exact access key? Reply only with the key."
        messages = [{"role":"user","content":document+"\n"+question}]
        ids = kvmem.encode(frontend,messages,None,{})
        ids = kvmem.resolve_query(frontend,messages,None,{},ids,
                                  {"message":0,"start":len(document)+1,"end":len(messages[0]["content"])})
        fixtures[name] = (ids,key)
    short_document = "The access key for Project Atlas is QUARTZ-2468.\n"
    short_question = "What is the exact access key for Project Atlas? Reply only with the key."
    short_messages = [{"role":"user","content":short_document+short_question}]
    short_ids = kvmem.encode(frontend,short_messages,None,{})
    short_ids = kvmem.resolve_query(frontend,short_messages,None,{},short_ids,
                                    {"message":0,"start":len(short_document),"end":len(short_messages[0]["content"])})
    fixtures["S"] = (short_ids,"QUARTZ-2468")
    ssd = args.scenario.startswith("ssd-") or args.scenario.startswith("fault-disk-")
    disk_path = (args.out/"native cache with spaces").resolve()
    if ssd: disk_path.mkdir(parents=True,exist_ok=True)
    fault = {"fault-park":"session_park","fault-enter":"session_enter",
             "fault-disk-read":"disk_read","fault-disk-write":"disk_write"}.get(args.scenario)
    config = argparse.Namespace(worker=args.worker,model=args.model,gpu=args.gpu,scenario="paging",
        context=32768,threads=8,batch=1024,stage=512,swaps=64,whole=True,no_graphs=False,mode="kvmem",
        budget=4096,reserve=1024,
        host_mib=128 if ssd else (224 if args.scenario == "quota" else 4096),
        disk_mib=16 if args.scenario == "ssd-capacity" else (512 if ssd else 0),disk_path=disk_path,
        sink=64,recent=64,aggregation="max",verify=True,lrfu=True,history_reuse=True,
        sessions=4 if args.scenario == "quota" else 2,session_metrics=True,fault=fault,profile=False,mtp_drafts=args.mtp_drafts)
    result = {"status":"running","scenario":args.scenario,"gpu_uuid":args.gpu,
              "worker_sha256":hashlib.sha256(args.worker.read_bytes()).hexdigest(),"checks":[]}
    def save():
        (args.out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    log = args.out/"worker.log"
    worker = Worker(config,log)
    result.update(command=worker.command,info=worker.info)
    def run(name,label,reuse=None,error=None):
        ids,key = fixtures[name]
        answer = worker.request(ids,ids.kvmem_query,32)
        answer.update(name=label,document=name,input_tokens=len(ids))
        if "tokens" in answer: answer["text"] = tok.decode(answer["tokens"])
        result["checks"].append(answer)
        save()
        if error:
            assert error in answer.get("error", ""), answer
        else:
            assert "error" not in answer and key in answer["text"], answer
            if reuse is True: assert answer["resume"] >= ids.kvmem_query[0] and answer["read_tokens"] < 256, answer
            if reuse is False: assert answer["resume"] == 0, answer
        print(json.dumps({k:answer[k] for k in ("name","resume","read_tokens","text","error") if k in answer},ensure_ascii=False),flush=True)
        return answer
    try:
        if args.scenario == "ssd-capacity":
            run("A","oversized_history",error="KVMem request exceeds Host payload capacity")
            run("S","short_request_after_rejection",False)
        elif args.scenario in ("fault-disk-write","fault-disk-read"):
            stage = "write" if args.scenario == "fault-disk-write" else "read"
            run("A",f"disk_{stage}_failure",error=f"disk_{stage}")
            run("A",f"fresh_A_after_{stage}_failure",False)
            run("B","fresh_B",False)
            run("A",f"cached_A_after_{stage}_recovery",True)
        else:
            run("A","fresh_A",False)
        if args.scenario == "fault-park":
            run("B","park_failure",error="session_park")
            run("B","fresh_B_after_failure",False)
            run("A","fresh_A_after_failure",False)
            run("B","cached_B_after_recovery",True)
        elif args.scenario not in ("ssd-capacity","fault-disk-write","fault-disk-read"):
            run("B","fresh_B",False)
            if args.scenario == "fault-enter":
                run("A","enter_failure",error="session_enter")
                run("B","unaffected_cached_B",True)
                run("A","fresh_A_after_enter_failure",False)
            elif args.scenario == "quota":
                run("A","A_after_shared_RAM_eviction",False)
            else:
                run("A","restored_A",True)
                run("C","fresh_C",False)
                run("A","recent_A_retained",True)
                run("B","old_B_evicted",False)
        if args.mtp_drafts and args.scenario != "ssd-capacity":
            ids,_ = fixtures["A"]
            cancelled = worker.request(ids,ids.kvmem_query,128,stop_after=1)
            cancelled.update(name="mtp_generation_cancel",document="A",input_tokens=len(ids))
            result["checks"].append(cancelled)
            assert cancelled.get("finish") == "cancel", cancelled
            run("A","cached_A_after_MTP_cancel",True)
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail",error=str(error))
        raise
    finally:
        worker.close()
        pattern = r"KVMEM_SESSIONS parked=(\d+) evictions=(\d+) restores=(\d+) skipped=(\d+) host_bytes=(\d+) host_limit=(\d+) session=(\d+) incarnation=(\d+)"
        counters = [dict(zip(("parked","evictions","restores","skipped","host_bytes","host_limit","session","incarnation"),map(int,values)))
                    for values in re.findall(pattern,log.read_text(encoding="utf-8"))]
        completed = [r for r in result["checks"] if "error" not in r]
        if len(counters) == len(completed):
            for run_result,metrics in zip(completed,counters): run_result["sessions"] = metrics
        result["counters"] = counters
        storage_pattern = (r"KVMEM_STORAGE host_bytes=(\d+) host_limit=(\d+) disk_bytes=(\d+) disk_limit=(\d+) "
                           r"read_bytes=(\d+) written_bytes=(\d+) errors=(\d+)")
        storage = [dict(zip(("host_bytes","host_limit","disk_bytes","disk_limit","read_bytes","written_bytes","errors"),map(int,values)))
                   for values in re.findall(storage_pattern,log.read_text(encoding="utf-8"))]
        result["storage"] = storage
        if result["status"] == "pass":
            if len(counters) != len(completed) or any(m["parked"] >= config.sessions or m["host_bytes"] > m["host_limit"] for m in counters):
                result.update(status="fail",error="retained count/shared quota metrics failed")
            if args.scenario == "quota" and counters[-1]["evictions"] < 2:
                result.update(status="fail",error="RAM pressure did not evict idle histories")
            if ssd and (len(storage) != len(completed) or
                        any(m["host_bytes"] > m["host_limit"] or m["disk_bytes"] > m["disk_limit"] for m in storage)):
                result.update(status="fail",error="SSD shared quota metrics failed")
            if ssd and args.scenario != "ssd-capacity" and storage and not (storage[-1]["read_bytes"] and storage[-1]["written_bytes"]):
                result.update(status="fail",error="SSD scenario performed no physical read/write")
            if list(disk_path.glob("kvmem-spill-*")):
                result.update(status="fail",error="ephemeral SSD file remained after worker shutdown")
        save()
    if result["status"] != "pass": raise RuntimeError(result["error"])


if __name__ == "__main__": main()
