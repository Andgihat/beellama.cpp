#!/usr/bin/env python3
"""Summarize saved A/B evidence; never treat a successful run as a speed gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        result = json.loads((args.results / name / "results.json").read_text(encoding="utf-8"))
        if result["status"] != "pass":
            raise RuntimeError(f"{name}: a failed or incomplete run cannot be summarized as accepted")
        return result
    originals = [read("perf-original-matched"), read("perf-original-repeat")]
    native = read("perf-integrated-native")
    candidate = read("perf-m1-matched")
    paging = read("perf-paging-system")
    user = read("perf-paging-user")
    user_original = read("perf-original-user")
    metrics = ("prompt_ms", "decode_ms", "wall_ms", "prefill_tok_s", "decode_tok_s")
    def medians(runs):
        assert all(len(run["tokens"]) == 128 and run["resume"] == 0 for run in runs)
        return {key: statistics.median(run[key] for run in runs) for key in metrics}
    def compare(base, test):
        return {"prefill_change_percent": (test["prefill_tok_s"]/base["prefill_tok_s"]-1)*100,
                "decode_change_percent": (test["decode_tok_s"]/base["decode_tok_s"]-1)*100,
                "prompt_time_change_percent": (test["prompt_ms"]/base["prompt_ms"]-1)*100,
                "wall_time_change_percent": (test["wall_ms"]/base["wall_ms"]-1)*100}
    report = {"gpu_uuid":candidate["gpu_uuid"], "machine":candidate["machine"],
              "warmups_per_case":2, "original_repeats_per_case":10, "candidate_repeats_per_case":5,
              "output_tokens":128, "matched":[], "paging":[],
              "worker_sha256":candidate["worker_sha256"], "settings":candidate["command"],
              "m1_status":"qualifying_explicit_query_ranges",
              "long_query_interface_authorized":"2026-10-12",
              "matched_candidate_series":"perf-m1-matched"}
    for index, case in enumerate(candidate["cases"]):
        names = [*originals, native, candidate]
        fixtures = [json.loads((args.results / folder / (case["name"]+".json")).read_text())["ids"]
                    for folder in ("perf-original-matched", "perf-original-repeat", "perf-integrated-native", "perf-m1-matched")]
        assert all(ids == fixtures[0] for ids in fixtures)
        assert all(run["gpu_uuid"] == candidate["gpu_uuid"] and run["cases"][index]["input_tokens"] == case["input_tokens"] for run in names)
        original_runs = [run for result in originals for run in result["cases"][index]["runs"]]
        base = medians(original_runs)
        test = medians(case["runs"])
        entry = {"input_tokens":case["input_tokens"], "original":base, "kvmem":test,
                 "integrated_native":medians(native["cases"][index]["runs"]), **compare(base, test),
                 "original_ranges_ms":{key:[min(r[key] for r in original_runs), max(r[key] for r in original_runs)]
                                        for key in ("prompt_ms", "decode_ms")},
                 "greedy_original_distinct_outputs":len({tuple(r["tokens"]) for r in original_runs})}
        report["matched"].append(entry)
    for layout, result, base_result in (("system", paging, originals[1]), ("user", user, user_original)):
        case = result["cases"][0]
        baseline_case = next(c for c in base_result["cases"] if c["input_tokens"] == case["input_tokens"])
        base, test = medians(baseline_case["runs"]), medians(case["runs"])
        report["paging"].append({"layout":layout, "input_tokens":case["input_tokens"],
                                  "query_tokens":case["query"][1]-case["query"][0],
                                  "native":base, "kvmem":test, **compare(base,test),
                                  "read_tokens":sorted({r["read_tokens"] for r in case["runs"]}),
                                  "logical_context":result["context"], "physical_token_capacity":result["budget"]+result["reserve"],
                                  "expert_cache_info":result["info"]})
    report["matched_5_percent_gate"] = all(e["prefill_change_percent"] >= -5 and e["decode_change_percent"] >= -5 for e in report["matched"])
    range_originals = [read("range-original-before"), read("range-original-after")]
    range_native_runs = [run for result in range_originals for run in result["cases"][0]["runs"]]
    base = medians(range_native_runs)
    report["explicit_query"] = []
    reference_ids = json.loads((args.results / "range-original-before/prompt_8192.json").read_text())["ids"]
    assert json.loads((args.results / "range-original-after/prompt_8192.json").read_text())["ids"] == reference_ids
    assert json.loads((args.results / "perf-paging-user/prompt_8192.json").read_text())["ids"] == reference_ids
    report["explicit_query_default_mode_preserved"] = True
    for series in ("range-practical", "range-tight"):
        result = read(series)
        case = result["cases"][0]
        fixture = json.loads((args.results / series / "prompt_8192.json").read_text())
        assert fixture["ids"] == reference_ids
        assert result["gpu_uuid"] == report["gpu_uuid"]
        assert result["query_mode"] == "explicit-tail"
        test = medians(case["runs"])
        entry = {"series":series, "native":base, "kvmem":test, **compare(base,test),
                 "input_tokens":case["input_tokens"], "query_tokens":case["query"][1]-case["query"][0],
                 "budget":result["budget"], "reserve":result["reserve"], "context":result["context"],
                 "physical_capacity":result["budget"]+result["reserve"], "read_tokens":sorted({r["read_tokens"] for r in case["runs"]}),
                 "settings":result["command"], "worker_sha256":result["worker_sha256"],
                 "original_ranges_ms":{key:[min(r[key] for r in range_native_runs),max(r[key] for r in range_native_runs)]
                                       for key in ("prompt_ms","decode_ms")}}
        report["explicit_query"].append(entry)
    practical = report["explicit_query"][0]
    report["explicit_query_5_percent_gate"] = practical["prefill_change_percent"] >= -5 and practical["decode_change_percent"] >= -5
    report["teacher_forced"] = read("teacher-forced")
    report["qualification"] = {name:read(name)["status"] for name in
        ("fault-final-prepare", "fault-final-spill", "fault-final-restore", "fault-final-publish",
         "fault-prepare-stress", "paging-diagnostic", "paging-eager-mean", "host-admission", "api-verified",
         "api-query-range-final", "range-fault-prepare", "range-fault-spill", "range-fault-restore", "range-fault-publish",
         "range-eager-mean", "range-practical-cold")}
    if report["matched_5_percent_gate"] and report["explicit_query_5_percent_gate"] and report["teacher_forced"]["five_percent_gate"]:
        report["m1_status"] = "complete_with_qualified_configuration_limits"
    else:
        report["m1_status"] = "performance_gate_failed"
    report["performance_scope"] = "5% gates apply to matched non-paging and explicit-query B4096/R8192 at 9.3k input; forced paging at B2048/R4096 remains slower"
    report["unconfirmed_allocation_failure"] = {"evidence":"fault-prepare/results.json",
        "initial_error":"ERR bad allocation on the recovery request", "repeat_recoveries_passed":5,
        "root_cause":"not established; retained, not attributed to legacy behavior"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"matched_gate":report["matched_5_percent_gate"], "m1_status":report["m1_status"],
                      "matched_changes":[{k:e[k] for k in ("input_tokens", "prefill_change_percent", "decode_change_percent")} for e in report["matched"]]}, indent=2))


if __name__ == "__main__":
    main()
