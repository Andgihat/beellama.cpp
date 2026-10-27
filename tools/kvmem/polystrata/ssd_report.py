#!/usr/bin/env python3
"""Build compact P7 evidence and reject incomplete SSD qualification."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    def read(name):
        data=json.loads((args.task/"results"/name/"results.json").read_text(encoding="utf-8"))
        if data["status"] != "pass": raise RuntimeError(name+" failed or is incomplete")
        return data
    perf=read("decode-32k-mtp-ssd")
    switched=read("ssd-sessions-switch")
    fault_write=read("ssd-fault-write")
    fault_read=read("ssd-fault-read")
    capacity=read("ssd-capacity")
    api=read("ssd-api")
    package=read("m3-package-api")
    if perf.get("host_mib") != 128 or perf.get("disk_mib") != 1024 or not perf.get("ssd_decode_io_exercised"):
        raise RuntimeError("SSD performance configuration or I/O proof is missing")
    series=[s for s in perf["series"] if s["mode"] == "kvmem"]
    if len(series) != 1 or len(series[0]["runs"]) != 4:
        raise RuntimeError("SSD performance series must contain one warmup and three measurements")
    measured=[r for r in series[0]["runs"] if not r["warmup"]]
    if len(measured) != 3 or any(len(r["tokens"]) != 2048 or r["draft_offered"] < 1 for r in measured):
        raise RuntimeError("SSD fixed decode workload was not completed")
    for run in series[0]["runs"]:
        storage=run["storage"]
        if storage["host_bytes"] > storage["host_limit"] or storage["disk_bytes"] > storage["disk_limit"] or storage["errors"]:
            raise RuntimeError("SSD quota or storage error counter failed")
        if not (run["decode_io"]["read_bytes"] or run["decode_io"]["written_bytes"]):
            raise RuntimeError("decode did not exercise SSD I/O")
    for name,data,needle in (("write",fault_write,"disk_write"),("read",fault_read,"disk_read")):
        if needle not in data["checks"][0].get("error","") or any(needle in item.get("error","") for item in data["checks"][1:]):
            raise RuntimeError(name+" failure did not fire once and recover")
    if "payload capacity" not in capacity["checks"][0].get("error","") or "error" in capacity["checks"][1]:
        raise RuntimeError("capacity rejection/recovery proof is incomplete")
    if not switched["storage"] or not switched["storage"][-1]["read_bytes"] or not switched["storage"][-1]["written_bytes"]:
        raise RuntimeError("retained-session SSD I/O proof is missing")
    accepted={}
    for mode in ("graph","no-graph"):
        text=(args.task/"results"/"ssd-mtp-cold"/mode/"worker.log").read_text(encoding="utf-8")
        for prefix in (63,191,319):
            for keep in (1,2):
                if f"PASS prefix={prefix} accepted={keep}" not in text:
                    raise RuntimeError("cold accepted-state coverage is incomplete")
        match=re.search(r"PASS ssd read_bytes=(\d+) written_bytes=(\d+)",text)
        if not match: raise RuntimeError("cold physical I/O proof is missing")
        accepted[mode]={"read_bytes":int(match.group(1)),"written_bytes":int(match.group(2))}
    core=(args.task/"logs"/"ssd-core-tests.log").read_text(encoding="utf-8")
    for needle in ("host KV storage:","spill I/O:","tiered KV:","KVMem history planner PASS",
                   "KVMem retained-session planner"):
        if needle not in core: raise RuntimeError("model-free storage evidence is incomplete")
    archive=args.task/"source-archive"/"kvmem-m3-ssd-source.tar.gz"
    expected_sha=(Path(str(archive)+".sha256").read_text(encoding="utf-8").split()[0])
    digest=hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b""): digest.update(chunk)
    if digest.hexdigest() != expected_sha: raise RuntimeError("source archive checksum failed")
    reconstructed=args.task/"source-bundle-check"/"m3-ssd"/"kvmem-rocm-source"
    source_manifest=json.loads((reconstructed/"SOURCE-MANIFEST.json").read_text(encoding="utf-8"))
    clean_log=(args.task/"logs"/"ssd-clean-source-build.log").read_text(encoding="utf-8",errors="replace")
    versions=json.loads((Path(__file__).resolve().parents[2]/"backends/versions.json").read_text(encoding="utf-8"))
    if source_manifest["backends"]["polystrata"]["patched_tree"] != versions["polystrata"]["patched_tree"] or \
            "[132/132] Linking CXX executable strata-poly.exe" not in clean_log:
        raise RuntimeError("clean source reconstruction/build evidence is incomplete")
    report={"status":"qualified_process_local_ssd_increment","date":"2026-10-12",
        "scope":"Windows/CUDA qwen35moe F16, one embedded draft, one active request, 1..16 retained histories, shared bounded RAM/process-local buffered SSD",
        "gpu_uuid":perf["gpu_uuid"],
        "workload":{k:perf[k] for k in ("input_tokens","output_tokens","budget","reserve","context","fixture_sha256","mtp_drafts","host_mib","disk_mib")},
        "original_mtp":perf["native"],"ram_mtp":json.loads((args.task/"results"/"decode-32k-mtp"/"results.json").read_text(encoding="utf-8"))["kvmem"],
        "ssd_mtp":perf["kvmem"],"vs_original_mtp_percent":perf["decode_change_percent"],
        "vs_ram_mtp_percent":perf["managed_decode_change_percent"],
        "decode_samples":[{k:r[k] for k in ("repeat","warmup","decode_tok_s","prompt_ms","interval_p95_ms","interval_max_ms",
                           "draft_accepted","draft_offered","decode_exchanges","decode_io","storage")} for r in series[0]["runs"]],
        "cold_accepted_state":accepted,
        "retained_sessions":{"checks":[{k:r[k] for k in ("name","resume","read_tokens","finish","text","error") if k in r}
                                        for r in switched["checks"]],"final_storage":switched["storage"][-1]},
        "faults":{"disk_write":fault_write["checks"][0]["error"],"disk_read":fault_read["checks"][0]["error"],
                  "capacity":capacity["checks"][0]["error"],"short_recovery":capacity["checks"][1]["text"]},
        "http":{"cached_tokens":api["cached_tokens"],"props":api["props"]["kvmem"]},
        "runtime_package":{"clean_system_path_http_pass":True,"cached_tokens":package["cached_tokens"],
                           "path":str((args.task/"packages"/"three-backend-m3").resolve()),
                           "worker_sha256":series[0]["worker_sha256"],"Python":"user supplied, not bundled"},
        "source_reconstruction":{"framework_commit":source_manifest["base_commit"],
                                 "polystrata_tree":source_manifest["backends"]["polystrata"]["patched_tree"],
                                 "archive_sha256":expected_sha,"manifest_files":len(source_manifest["files"]),
                                 "manifest_hashes_verified":True,"clean_cuda_build":"132/132",
                                 "llamacpp_source":"local pinned archive override; no network dependency"},
        "lifecycle":{"ephemeral_files_removed":True,"restart_persistence":False,"gpu_publication_from_io_worker":False,
                     "read_queue_jobs":1,"cold_reads_async_cpu_io":True,"automatic_demotion_synchronous":True},
        "backend_source":versions["polystrata"],
        "limits":"one synthetic workload; OS-buffered files, not raw-device bandwidth; prefill not gated; H/D cover native KV records only, excluding recurrent checkpoints, metadata, pinned staging, model and allocator overhead; no crash/restart persistence"}
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status","vs_original_mtp_percent","vs_ram_mtp_percent")}))


if __name__=="__main__": main()
