#!/usr/bin/env python3
"""Two HTTP turns prove managed reuse survives real template rendering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","worker","model","tokenizer","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--port",type=int,default=18281)
    parser.add_argument("--retained-sessions",type=int,choices=range(1,17),default=1)
    parser.add_argument("--mtp-drafts",type=int,choices=range(4),default=0)
    parser.add_argument("--host-mib",type=int,default=4096)
    parser.add_argument("--disk-mib",type=int,default=0)
    parser.add_argument("--launcher",type=Path,help="server.py wrapper to test; defaults to this source tree")
    args = parser.parse_args()
    for name in ("source","worker","model","tokenizer","out"):
        setattr(args,name,getattr(args,name).resolve())
    if args.launcher: args.launcher=args.launcher.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    disk_path = (args.out/"native cache with spaces").resolve()
    if args.disk_mib: disk_path.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,str(args.launcher or Path(__file__).with_name("server.py")),"--source",str(args.source),
               "--worker",str(args.worker),"--model",str(args.model),"--tokenizer",str(args.tokenizer),
               "--gpu",args.gpu,"--port",str(args.port),"--context","32768","--budget","4096",
               "--reserve","1024","--prefill","1024","--max-tokens","32","--history-reuse",
               "--host-mib",str(args.host_mib),
               "--retained-sessions",str(args.retained_sessions),
               "--mtp-drafts",str(args.mtp_drafts),
               "--","--threads","8","--stage-mib","512","--tail-skip","0"]
    if args.disk_mib:
        at = command.index("--")
        command[at:at] = ["--disk-path",str(disk_path),"--disk-mib",str(args.disk_mib)]
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,payload=None):
        data = None if payload is None else json.dumps(payload,ensure_ascii=False).encode("utf-8")
        with opener.open(urllib.request.Request(base+path,data=data,headers={"Content-Type":"application/json"}),timeout=180) as response:
            return json.load(response)
    result = {"status":"running","command":command,"gpu_uuid":args.gpu}
    if args.disk_mib: os.environ["KVMEM_POLY_STORAGE_METRICS"] = "1"
    with (args.out/"server.log").open("w",encoding="utf-8") as log:
        proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                creationflags=0x08000000 if os.name == "nt" else 0)
        try:
            deadline = time.monotonic()+180
            while True:
                if proc.poll() is not None: raise RuntimeError("frontend exited: "+str(proc.returncode))
                try:
                    result["health"] = request("/health")
                    break
                except (OSError,urllib.error.URLError):
                    if time.monotonic() > deadline: raise TimeoutError("frontend health timeout")
                    time.sleep(.25)
            result["props"] = request("/props")
            assert result["props"]["kvmem"]["history_reuse"]
            assert result["props"]["kvmem"]["retained_sessions"] == args.retained_sessions
            assert result["props"]["kvmem"]["max_active_requests"] == 1
            assert result["props"]["kvmem"]["shared_host_quota"]
            assert result["props"]["kvmem"].get("mtp",False) == bool(args.mtp_drafts)
            assert result["props"]["kvmem"].get("ssd",False) == bool(args.disk_mib)
            assert result["props"]["kvmem"].get("disk_mib",0) == args.disk_mib
            assert result["props"]["kvmem"].get("ssd_persistence") == ("process" if args.disk_mib else "none")
            document = "参考资料：\n"+"".join(
                "The access key for Project Atlas is MARBLE-4827. Project Atlas uses MARBLE-4827.\n" if i == 70 else
                f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n"
                for i in range(353))
            question = "Find Project Atlas in the reference records. What is its exact access key? Reply only with the key."
            messages = [{"role":"user","content":document+"\n"+question}]
            payload = {"model":"kvmem-polystrata-qwen35","messages":messages,"max_tokens":32,"temperature":0,
                       "chat_template_kwargs":{"enable_thinking":False},
                       "kvmem_query":{"message":0,"start":len(document)+1,"end":len(messages[0]["content"])}}
            first = request("/v1/chat/completions",payload)
            answer = first["choices"][0]["message"]["content"].strip()
            assert answer == "MARBLE-4827", first
            if args.retained_sessions > 1:
                other_document = "Different reference set.\n"+document.replace("MARBLE-4827","ONYX-9136")
                other_messages = [{"role":"user","content":other_document+"\n"+question}]
                other_payload = dict(payload, messages=other_messages,
                    kvmem_query={"message":0,"start":len(other_document)+1,"end":len(other_messages[0]["content"])})
                other = request("/v1/chat/completions",other_payload)
                assert other["choices"][0]["message"]["content"].strip() == "ONYX-9136", other
                result["interleaved"] = other
            messages += [{"role":"assistant","content":answer},
                         {"role":"user","content":"请根据之前的资料，再只输出 Project Atlas 的访问密钥。"}]
            payload.pop("kvmem_query")
            second = request("/v1/chat/completions",payload)
            assert second["choices"][0]["message"]["content"].strip() == "MARBLE-4827", second
            cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
            assert cached > first["usage"]["prompt_tokens"]-256, second
            result.update(status="pass",first=first,second=second,cached_tokens=cached)
        except Exception as error:
            result.update(status="fail",error=str(error))
            raise
        finally:
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],capture_output=True,check=True)
                else: proc.terminate()
                proc.wait(timeout=30)
            if args.disk_mib:
                for _ in range(100):
                    if not list(disk_path.glob("kvmem-spill-*")): break
                    time.sleep(.05)
                assert not list(disk_path.glob("kvmem-spill-*")), "ephemeral SSD file remained after server shutdown"
            (args.out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"cached_tokens":result["cached_tokens"]}),flush=True)


if __name__ == "__main__": main()
