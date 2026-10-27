#!/usr/bin/env python3
"""Export a compact retained-session report from qualified raw evidence only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    def read(name):
        result = json.loads((args.results/name/"results.json").read_text(encoding="utf-8"))
        if result["status"] != "pass": raise RuntimeError(name+" did not pass")
        return result
    perf = read("decode-32k-sessions")
    api = read("sessions-api")
    checks = {name:read("sessions-"+name) for name in ("switch-clean","quota","fault-park","fault-enter")}
    if not perf["decode_5_percent_gate"] or not perf["managed_decode_5_percent_gate"] or perf["stall_review_needed"]:
        raise RuntimeError("decode qualification failed")
    runs = [r for s in perf["series"] for r in s["runs"] if not r["warmup"]]
    if len(runs) != 3 or any(len(r["tokens"]) != 2048 or r["decode_exchanges"] < 1 or r["resume"] < 32000 for r in runs):
        raise RuntimeError("parked-history decode workload is incomplete")
    report = {"status":"qualified_retained_conversation_increment","date":"2026-10-12",
        "scope":"opt-in 1..16 retained histories, one GPU-active request; shared native Host KV quota and LRU",
        "gpu_uuid":perf["gpu_uuid"], "workload":{k:perf[k] for k in
            ("input_tokens","output_tokens","budget","reserve","context","fixture_sha256")},
        "original":perf["native"],"retained_sessions":perf["kvmem"],
        "decode_change_percent":perf["decode_change_percent"],
        "managed_decode_change_percent":perf["managed_decode_change_percent"],
        "decode_runs":[{k:r[k] for k in ("repeat","resume","read_tokens","prompt_ms","decode_tok_s",
            "interval_p95_ms","interval_max_ms","decode_exchanges","sessions")} for r in runs],
        "checks":{name:{"worker_sha256":result["worker_sha256"],"checks":[{k:r[k] for k in
            ("name","resume","read_tokens","text","error","sessions") if k in r} for r in result["checks"]]}
            for name,result in checks.items()},
        "http":{"status":api["status"],"cached_tokens":api["cached_tokens"],"props":api["props"]["kvmem"]},
        "performance_binary_sha256":perf["series"][0]["worker_sha256"],
        "backend_source":json.loads((Path(__file__).resolve().parents[2]/"backends/versions.json").read_text(encoding="utf-8"))["polystrata"],
        "limits":"one synthetic decode workload, saved identical original baseline; prefill not gated; H excludes recurrent checkpoints, metadata and model RAM; no concurrency, MTP, vision or SSD"}
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"decode":perf["kvmem"]["decode_tok_s"],
                      "vs_original_percent":perf["decode_change_percent"],
                      "vs_current_history_percent":perf["managed_decode_change_percent"],
                      "http_cached_tokens":api["cached_tokens"]}))


if __name__ == "__main__": main()
