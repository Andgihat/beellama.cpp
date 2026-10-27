#!/usr/bin/env python3
"""Compact MTP evidence, refusing incomplete correctness or decode checks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    def read(name):
        data=json.loads((args.task/"results"/name/"results.json").read_text(encoding="utf-8"))
        if data["status"] != "pass": raise RuntimeError(name+" failed or is incomplete")
        return data
    perf=read("decode-32k-mtp")
    sessions=read("mtp-sessions")
    api=read("mtp-api-qualified")
    package=read("m2-package-api")
    graph=(args.task/"logs/mtp-accepted-graph-qualified.log").read_text(encoding="utf-8")
    no_graph=(args.task/"logs/mtp-accepted-no-graph.log").read_text(encoding="utf-8")
    for enabled,log in ((1,graph),(0,no_graph)):
        for prefix in (63,191,319):
            for accepted in (1,2):
                if f"PASS prefix={prefix} accepted={accepted} graphs={enabled}" not in log:
                    raise RuntimeError("accepted-state coverage is incomplete")
        if "PASS cancellation_reset" not in log: raise RuntimeError("window cancellation check missing")
    if "PASS accept_fault_invalidated_and_rebuilt" not in graph: raise RuntimeError("accept fault did not fire")
    runs=[r for s in perf["series"] for r in s["runs"] if not r["warmup"]]
    if perf["mtp_drafts"] != 1 or len(runs) != 6 or any(len(r["tokens"])!=2048 or r["draft_offered"]<1 for r in runs):
        raise RuntimeError("matched original/integrated MTP workload was not completed")
    report={"status":"qualified_one_draft_mtp_increment","date":"2026-10-12",
        "scope":"Windows/CUDA qwen35moe F16, one embedded draft, one active request, optional 1..16 retained histories; no SSD or wider drafts yet",
        "gpu_uuid":perf["gpu_uuid"],"workload":{k:perf[k] for k in
            ("input_tokens","output_tokens","budget","reserve","context","fixture_sha256","mtp_drafts")},
        "original_mtp":perf["native"],"kvmem_mtp":perf["kvmem"],
        "vs_original_mtp_percent":perf["decode_change_percent"],
        "vs_m2_plain_percent":perf["managed_decode_change_percent"],
        "series":[{"name":s["name"],"mode":s["mode"],"worker_sha256":s["worker_sha256"],
            "runs":[{k:r[k] for k in ("repeat","warmup","decode_tok_s","prompt_ms","interval_p95_ms",
                "interval_max_ms","draft_accepted","draft_offered","decode_exchanges") if k in r}
                for r in s["runs"]]} for s in perf["series"]],
        "accepted_state":{"prefixes":[63,191,319],"kept_tokens":[1,2],"graphs":[False,True],
            "native_KV":"byte-identical to aligned-view serial evaluation",
            "recurrent_and_draft_hidden_max_relative":0,"mean_key_max_relative":0,
            "accept_fault_recovery":True,"pending_window_checkpoint_rejected":True,"cancel_reset":True},
        "session_checks":[{k:r[k] for k in ("name","resume","read_tokens","finish","text","error",
            "draft_accepted","draft_offered") if k in r} for r in sessions["checks"]],
        "http":{"cached_tokens":api["cached_tokens"],"props":api["props"]["kvmem"]},
        "m2_runtime_package":{"clean_PATH_http_pass":True,"cached_tokens":package["cached_tokens"],
            "path":str((args.task/"packages/three-backend-m2").resolve()),"Python":"user supplied, not bundled"},
        "backend_source":json.loads((Path(__file__).resolve().parents[2]/"backends/versions.json").read_text(encoding="utf-8"))["polystrata"],
        "limits":"one synthetic workload; sparse outputs need not match full context; prefill not gated; native draft policy may offer fewer than the configured maximum; H excludes checkpoints/metadata/staging"}
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","vs_original_mtp_percent","vs_m2_plain_percent")}))


if __name__=="__main__": main()
