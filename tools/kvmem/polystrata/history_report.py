#!/usr/bin/env python3
"""Export compact M2 evidence only from completed decode and history checks."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    def read(name):
        result = json.loads((args.results/name/"results.json").read_text(encoding="utf-8"))
        if result["status"] != "pass": raise RuntimeError(name+" did not pass")
        return result
    m1 = read("decode-32k")
    m2 = read("decode-32k-history")
    history = read("history-verified")
    live = read("history-live")
    api = read("history-api-qualified")
    if not m1["decode_5_percent_gate"] or not m2["decode_5_percent_gate"]:
        raise RuntimeError("decode regression gate failed")
    if any(m1[k] != m2[k] for k in ("gpu_uuid","fixture_sha256","input_tokens","output_tokens","budget","reserve","context")):
        raise RuntimeError("M1/M2 performance workload differs")
    def summarize(result,mode):
        runs = [run for series in result["series"] if series["mode"] == mode
                for run in series["runs"] if not run["warmup"]]
        if len(runs) != 3 or any(len(r["tokens"]) != 2048 for r in runs):
            raise RuntimeError("performance series is incomplete")
        return {"median_decode_tok_s":statistics.median(r["decode_tok_s"] for r in runs),
                "decode_tok_s_samples":[r["decode_tok_s"] for r in runs],
                "median_prompt_ms":statistics.median(r["prompt_ms"] for r in runs),
                "median_interval_p95_ms":statistics.median(r["interval_p95_ms"] for r in runs),
                "worst_interval_ms":max(r["interval_max_ms"] for r in runs),
                "resume_tokens":[r["resume"] for r in runs],"read_tokens":[r["read_tokens"] for r in runs],
                "decode_exchanges":[r.get("decode_exchanges",0) for r in runs],
                "binary_hashes":sorted({s["worker_sha256"] for s in result["series"] if s["mode"] == mode})}
    report = {"status":"qualified_current_conversation_increment", "date":"2026-10-12",
              "gpu_uuid":m1["gpu_uuid"], "workload":{k:m1[k] for k in
                ("input_tokens","output_tokens","budget","reserve","context","fixture_sha256")},
              "native":summarize(m1,"native"),"m1":summarize(m1,"kvmem"),"m2":summarize(m2,"kvmem"),
              "m1_decode_change_percent":m1["decode_change_percent"],
              "m2_decode_change_percent":m2["decode_change_percent"],
              "scope":"one current conversation, opt-in, Windows/CUDA qwen35moe F16; no parked sessions or concurrency",
              "checks":[{k:r[k] for k in ("name","resume","read_tokens","prompt_ms","finish","error","text") if k in r}
                        for r in history["checks"]],
              "http":{"status":api["status"],"cached_tokens":api["cached_tokens"],"props":api["props"]["kvmem"]},
              "live_continuation":[{k:r[k] for k in ("name","resume","read_tokens","text") if k in r}
                                   for r in live["checks"]],
              "backend_source":json.loads((Path(__file__).resolve().parents[2]/"backends/versions.json").read_text(encoding="utf-8"))["polystrata"],
              "limits":"synthetic workload; token counts match, sparse outputs need not match; prefill not gated; client intervals include OS/pipe delay"}
    if any(n < 1 for n in report["m1"]["decode_exchanges"]+report["m2"]["decode_exchanges"]):
        raise RuntimeError("generation-time paging was not exercised")
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"m1_decode":report["m1"]["median_decode_tok_s"],
                      "m2_decode":report["m2"]["median_decode_tok_s"],"http_cached_tokens":api["cached_tokens"]}))


if __name__ == "__main__": main()
