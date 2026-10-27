#!/usr/bin/env python3
"""Same-token decode audit using the unmodified upstream --follow CLI.

Windows limits command-line length: use the 2.3k fixture, not the 9.3k case.
The alternating fresh-process runs are a cold expert-cache audit, complementing
the warmed streaming-server throughput measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "integrated", "model", "fixture", "reference", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for name in ("original", "integrated", "model", "fixture", "reference", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    case = next(c for c in reference["cases"] if c["input_tokens"] == len(fixture["ids"]))
    tape = case["runs"][0]["tokens"]
    assert len(tape) == 128
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    result = {"gpu_uuid":args.gpu, "input_tokens":len(fixture["ids"]), "output_tokens":len(tape),
              "output_sha256":hashlib.sha256(json.dumps(tape).encode()).hexdigest(),
              "scope":"alternating fresh processes; same prompt and forced output IDs; cold expert cache",
              "runs":[], "status":"running"}
    def save():
        (args.out/"results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        for repeat in range(args.repeats):
            for mode, worker in (("native",args.original),("kvmem",args.integrated)):
                command = [str(worker),"--model",str(args.model),"--max-context","16384","--threads","8",
                           "--prompt-batch","1024","--stage-mib","512","--window","1","--no-draft",
                           "--lookup","0","--tail-skip","0","--expert-swaps","64","--prompt-cache","0",
                           "--conversation-cache-mib","0","--conversation-cache-slots","0","--kv-whole",
                           "--tokens",",".join(map(str,fixture["ids"])),"--follow",",".join(map(str,tape)),"--max-new","128"]
                if mode == "kvmem":
                    command += ["--kvmem-budget","12288","--kvmem-gen-reserve","4096","--kvmem-host-mib","4096",
                                "--kvmem-sink-tokens","64","--kvmem-recent-tokens","64","--kvmem-score-agg","max",
                                "--kvmem-query-begin",str(fixture["query"][0]),"--kvmem-query-end",str(fixture["query"][1])]
                if os.name == "nt" and len(subprocess.list2cmdline(command)) >= 32767:
                    raise ValueError("fixture exceeds Windows command-line limit; use the 2.3k fixture")
                started = time.perf_counter()
                proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180,
                                      creationflags=0x08000000 if os.name == "nt" else 0)
                (args.out/f"{mode}-{repeat}.log").write_text(proc.stderr, encoding="utf-8")
                if proc.returncode:
                    raise RuntimeError(f"{mode} failed: {proc.returncode}; inspect its log")
                generated = [int(t) for t in proc.stdout.strip().split(",")]
                if generated != tape:
                    raise RuntimeError("--follow did not produce the exact reference token sequence")
                matched = re.search(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s",proc.stderr)
                if not matched or int(matched[1]) != len(fixture["ids"]) or int(matched[3]) != len(tape):
                    raise RuntimeError("CLI statistics do not match the fixed workload")
                run = {"mode":mode,"repeat":repeat,"prefill_tok_s":float(matched[2]),"decode_tok_s":float(matched[4]),
                       "wall_ms_including_load":(time.perf_counter()-started)*1000,"same_tokens":True,
                       "worker_sha256":hashlib.sha256(worker.read_bytes()).hexdigest()}
                result["runs"].append(run)
                save()
                print(json.dumps(run),flush=True)
        for mode in ("native","kvmem"):
            result[mode] = {key:statistics.median(r[key] for r in result["runs"] if r["mode"]==mode)
                            for key in ("prefill_tok_s","decode_tok_s")}
        result["change_percent"] = {key:(result["kvmem"][key]/result["native"][key]-1)*100 for key in result["native"]}
        result["five_percent_gate"] = all(value >= -5 for value in result["change_percent"].values())
        result["status"] = "pass"
    except Exception as error:
        result.update(status="fail",error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
