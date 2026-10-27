#!/usr/bin/env python3
"""One long-context decode comparison, including reserve exhaustion.

Uses the original unmodified worker and the managed worker in A/B/A order.
Prefill is reported, not gated. Client token intervals include pipe/scheduling
latency; whole-decode throughput uses the worker's inclusive timer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys

from qualify import Worker


def percentile(values, p):
    ordered = sorted(values)
    at = (len(ordered)-1)*p
    lo = int(at)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(at-lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "integrated", "model", "tokenizer", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--output", type=int, default=2048)
    parser.add_argument("--history-reuse", action="store_true")
    parser.add_argument("--mtp-drafts", type=int, choices=range(4), default=0)
    parser.add_argument("--host-mib", type=int, default=4096)
    parser.add_argument("--disk-mib", type=int, default=0)
    parser.add_argument("--baseline", type=Path, help="reuse today's identical original series for the M2-only check")
    parser.add_argument("--retained-sessions", type=int, choices=range(1,17), default=1,
                        help="if >1, interleave an unrelated short conversation before each measured long request")
    parser.add_argument("--reference", type=Path, help="also gate decode against the preceding managed increment")
    parser.add_argument("--screen", action="store_true",
                        help="run only one warmup and one measured managed candidate; do not apply comparison gates")
    args = parser.parse_args()
    if args.screen and (args.baseline or args.reference or args.history_reuse or args.disk_mib or args.retained_sessions > 1):
        parser.error("--screen is a managed RAM-only candidate check and cannot be combined with baseline/reference/history/SSD/sessions")
    if args.history_reuse and not args.baseline:
        parser.error("--history-reuse requires the saved original --baseline")
    if args.retained_sessions > 1 and not args.history_reuse:
        parser.error("multiple retained sessions require --history-reuse")
    if args.disk_mib < 0 or (args.disk_mib and not args.baseline):
        parser.error("SSD performance runs require nonnegative --disk-mib and a saved original --baseline")
    for name in ("source", "original", "integrated", "model", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    disk_path = (args.out / "native cache with spaces").resolve()
    if args.disk_mib: disk_path.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.source / "tools"))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, LITERAL_TAGS
    from serve import kvmem

    tok = Tokenizer.from_dir(args.tokenizer)
    template = ChatTemplate(args.tokenizer / "chat_template.jinja")
    class Frontend:
        literals = LITERAL_TAGS
        def render_prompt(self, messages, tools, kwargs):
            return template.render(messages, tools=tools, enable_thinking=False)
    frontend = Frontend()
    frontend.tok = tok
    lines = ["Reference records. Answer from these records; do not invent keys.\n"]
    for i in range(10000):
        lines.append("The access key for Project Atlas is MARBLE-4827. Project Atlas uses MARBLE-4827.\n"
                     if i == 70 else
                     f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n")
        if i % 32 == 0 and len(tok.encode("".join(lines))) >= args.length:
            break
    document = "".join(lines)
    question = ("Find Project Atlas in the reference records. Start your answer with its exact access key. "
                "Then write a detailed numbered list of 300 observations about these records. "
                "Discuss their organization, serial numbers and inspection dates in full sentences. "
                "Write at least 4000 words; keep going through every numbered item. Do not summarize or stop early.")
    messages = [{"role":"user", "content":document+"\n"+question}]
    ids = kvmem.encode(frontend, messages, None, {})
    selected = kvmem.resolve_query(frontend, messages, None, {}, ids,
                                  {"message":0,"start":len(document)+1,"end":len(messages[0]["content"])})
    assert selected == ids
    unrelated_messages = [{"role":"user","content":"Unrelated conversation. Reply with only the word ISOLATED."}]
    unrelated_ids = kvmem.encode(frontend, unrelated_messages, None, {})
    fixture = {"ids":list(ids), "query":selected.kvmem_query, "expected":"MARBLE-4827"}
    fixture_json = json.dumps(fixture, ensure_ascii=False)
    (args.out / "fixture.json").write_text(fixture_json, encoding="utf-8")
    result = {"status":"running", "gpu_uuid":args.gpu, "input_tokens":len(ids),
              "mtp_drafts":args.mtp_drafts,
              "host_mib":args.host_mib, "disk_mib":args.disk_mib,
              "output_tokens":args.output, "budget":4096, "reserve":1024, "context":40960,
              "fixture_sha256":hashlib.sha256(fixture_json.encode()).hexdigest(), "series":[],
              "scope":"warm server A/B/A, no prefix reuse; inclusive decode; prefill not gated"}
    if args.screen:
        result["scope"] = "managed candidate screen: one warmup plus one measured decode; prefill reported, not gated"
    baseline = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("mtp_drafts",0) != args.mtp_drafts:
            raise ValueError("saved original baseline uses a different MTP setting")
        if (not args.history_reuse and not args.disk_mib) or baseline["status"] != "pass" or any(
                baseline[key] != result[key] for key in ("gpu_uuid","input_tokens","output_tokens","budget","reserve","context","fixture_sha256")):
            raise ValueError("managed increment requires a passing identical original workload")
        original_hashes = {series["worker_sha256"] for series in baseline["series"] if series["mode"] == "native"}
        if original_hashes != {hashlib.sha256(args.original.read_bytes()).hexdigest()}:
            raise ValueError("original binary differs from the saved baseline")
        result.update(baseline=str(args.baseline.resolve()),history_reuse=args.history_reuse,
                      scope=("SSD-backed fresh request; same original baseline and fixed-length decode; prefill not gated"
                             if args.disk_mib else
                             "M2 current-prefix reuse; same original baseline and fixed-length decode; prefill not gated"))
    if args.retained_sessions > 1:
        result.update(retained_sessions=args.retained_sessions,
                      scope="M2 parked-prefix restore after unrelated short conversation; inclusive long decode; prefill not gated")
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference["status"] != "pass" or any(reference[key] != result[key] for key in
                ("gpu_uuid","input_tokens","output_tokens","budget","reserve","context","fixture_sha256")):
            raise ValueError("managed reference must be a passing identical workload")
        result["reference"] = str(args.reference.resolve())
    def save():
        (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    os.environ["KVMEM_POLY_DECODE_METRICS"] = "1"
    try:
        if args.screen:
            plan = (("kvmem-screen", "kvmem", args.integrated, 1),)
        elif baseline:
            plan = (("kvmem-ssd" if args.disk_mib else "kvmem-history", "kvmem", args.integrated, 3),)
        else:
            plan = (("original-before", "native", args.original, 2),
                    ("kvmem", "kvmem", args.integrated, 3),
                    ("original-after", "native", args.original, 1))
        for label, mode, worker_path, count in plan:
            config = argparse.Namespace(worker=worker_path, model=args.model, gpu=args.gpu,
                scenario="performance", context=40960, threads=8, batch=1024, stage=512, swaps=64,
                whole=True, no_graphs=False, mode=mode, budget=4096, reserve=1024, host_mib=args.host_mib,
                disk_mib=args.disk_mib if mode == "kvmem" else 0,disk_path=disk_path,
                sink=64, recent=64, aggregation="max", verify=False, lrfu=True, fault=None, profile=False,
                history_reuse=args.history_reuse, sessions=args.retained_sessions,
                session_metrics=args.retained_sessions > 1, mtp_drafts=args.mtp_drafts)
            log = args.out / (label+".log")
            worker = Worker(config, log)
            series = {"name":label, "mode":mode, "command":worker.command, "info":worker.info,
                      "worker_sha256":hashlib.sha256(worker_path.read_bytes()).hexdigest(), "runs":[]}
            completed = []
            series["interleaved"] = []
            result["series"].append(series)
            try:
                for repeat in range(-1, count):
                    if args.retained_sessions > 1 and repeat >= 0:
                        unrelated = worker.request(unrelated_ids, (0,len(unrelated_ids)), 32)
                        if "error" in unrelated:
                            raise RuntimeError(unrelated["error"])
                        unrelated.update(before_repeat=repeat, text=tok.decode(unrelated["tokens"]))
                        completed.append(unrelated)
                        series["interleaved"].append(unrelated)
                    run = worker.request(ids, selected.kvmem_query, args.output)
                    if "error" in run:
                        raise RuntimeError(run["error"])
                    text = tok.decode(run["tokens"])
                    if len(run["tokens"]) != args.output or (not args.history_reuse and run["resume"] != 0):
                        raise RuntimeError("fixed-length fresh-request workload was not completed")
                    if args.history_reuse and repeat >= 0 and (run["resume"] < selected.kvmem_query[0] or run["read_tokens"] >= 512):
                        raise RuntimeError("managed prefix was not reused up to the query")
                    if "MARBLE-4827" not in text[:200]:
                        raise RuntimeError("cold-record answer missing from the first 200 characters: "+text[:200])
                    if args.mtp_drafts and run["draft_offered"] < 1:
                        raise RuntimeError("MTP measurement offered no draft tokens")
                    intervals = [b-a for a,b in zip(run["token_arrival_ms"],run["token_arrival_ms"][1:])]
                    run.update(repeat=repeat, warmup=repeat < 0, text_head=text[:240],
                               decode_tok_s=(args.output-1)*1000/run["decode_ms"],
                               interval_p95_ms=percentile(intervals,.95), interval_p99_ms=percentile(intervals,.99),
                               interval_max_ms=max(intervals), expert_hit_rate=run["expert_hits"]/max(1,run["expert_total"]))
                    series["runs"].append(run)
                    completed.append(run)
                    save()
                    print(json.dumps({"series":label, **{k:run[k] for k in
                        ("repeat","prompt_ms","decode_ms","decode_tok_s","interval_p95_ms","interval_max_ms","text_head")}},
                        ensure_ascii=False), flush=True)
            finally:
                worker.close()
            if mode == "kvmem":
                metrics = re.findall(r"KVMEM_DECODE produced=(\d+) exchanges=(\d+) restored_bytes=(\d+) ms=([\d.]+)",
                                     log.read_text(encoding="utf-8"))
                if len(metrics) != len(completed):
                    raise RuntimeError("missing generation-only paging counters")
                for run, (produced, exchanges, restored, _) in zip(completed, metrics):
                    run.update(decode_exchanges=int(exchanges), decode_restored_bytes=int(restored))
                    if "before_repeat" not in run and (int(produced) != args.output or
                            (args.output > result["reserve"] and int(exchanges) < 1)):
                        raise RuntimeError("generation did not exercise reserve exhaustion")
                if args.retained_sessions > 1:
                    session_metrics = re.findall(r"KVMEM_SESSIONS parked=(\d+) evictions=(\d+) restores=(\d+) skipped=(\d+) host_bytes=(\d+) host_limit=(\d+) session=(\d+) incarnation=(\d+)",
                                                 log.read_text(encoding="utf-8"))
                    if len(session_metrics) != len(completed):
                        raise RuntimeError("missing retained-session counters")
                    for run, values in zip(completed, session_metrics):
                        run["sessions"] = dict(zip(("parked","evictions","restores","skipped","host_bytes","host_limit","session","incarnation"),map(int,values)))
                        if run["sessions"]["host_bytes"] > run["sessions"]["host_limit"]:
                            raise RuntimeError("shared host quota exceeded")
                    if any(run["sessions"]["restores"] < run["repeat"]+1 for run in series["runs"] if not run["warmup"]):
                        raise RuntimeError("long requests did not restore a parked history")
                if args.disk_mib:
                    decode_io = re.findall(r"KVMEM_DECODE_IO read_bytes=(\d+) written_bytes=(\d+)",
                                           log.read_text(encoding="utf-8"))
                    if len(decode_io) != len(completed):
                        raise RuntimeError("missing generation-only SSD I/O counters")
                    for run, values in zip(completed,decode_io):
                        run["decode_io"] = dict(zip(("read_bytes","written_bytes"),map(int,values)))
                    storage = re.findall(r"KVMEM_STORAGE host_bytes=(\d+) host_limit=(\d+) disk_bytes=(\d+) disk_limit=(\d+) read_bytes=(\d+) written_bytes=(\d+) errors=(\d+)",
                                         log.read_text(encoding="utf-8"))
                    if len(storage) != len(completed):
                        raise RuntimeError("missing SSD quota counters")
                    for run, values in zip(completed,storage):
                        run["storage"] = dict(zip(("host_bytes","host_limit","disk_bytes","disk_limit","read_bytes","written_bytes","errors"),map(int,values)))
                        if run["storage"]["host_bytes"] > run["storage"]["host_limit"] or run["storage"]["disk_bytes"] > run["storage"]["disk_limit"]:
                            raise RuntimeError("SSD shared quota exceeded")
                    measured = [run for run in series["runs"] if not run["warmup"]]
                    result["ssd_decode_io_exercised"] = any(run["decode_io"]["read_bytes"] or run["decode_io"]["written_bytes"] for run in measured)
                    if not result["ssd_decode_io_exercised"]:
                        raise RuntimeError("measured decode performed no SSD I/O")
            save()
        for mode in (("kvmem",) if args.screen else ("native", "kvmem")):
            if mode == "native" and baseline:
                result[mode] = baseline[mode]
                continue
            runs = [run for series in result["series"] if series["mode"] == mode
                    for run in series["runs"] if not run["warmup"]]
            result[mode] = {k:statistics.median(run[k] for run in runs) for k in
                ("decode_tok_s","prompt_ms","decode_ms","interval_p95_ms","interval_p99_ms","interval_max_ms","expert_hit_rate")}
            result[mode]["worst_interval_ms"] = max(run["interval_max_ms"] for run in runs)
        if args.screen:
            result["status"] = "pass"
        else:
            result["decode_change_percent"] = (result["kvmem"]["decode_tok_s"]/result["native"]["decode_tok_s"]-1)*100
            result["decode_5_percent_gate"] = result["decode_change_percent"] >= -5
            if reference:
                result["managed_decode_change_percent"] = (result["kvmem"]["decode_tok_s"]/reference["kvmem"]["decode_tok_s"]-1)*100
                result["managed_decode_5_percent_gate"] = result["managed_decode_change_percent"] >= -5
            native_worst = result["native"].get("worst_interval_ms", result["native"]["interval_max_ms"])
            result["stall_review_needed"] = result["kvmem"]["worst_interval_ms"] > max(250,2*native_worst)
            result["status"] = "pass" if result["decode_5_percent_gate"] and result.get("managed_decode_5_percent_gate",True) and not result["stall_review_needed"] else "review"
        print(json.dumps({k:v for k,v in result.items() if k != "series"}, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        result.update(status="fail", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
