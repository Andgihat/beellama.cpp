#!/usr/bin/env python3
"""Exercise the actual pinned frontend and native worker on a loopback test port."""
from __future__ import annotations

import argparse
import concurrent.futures
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
    for name in ("source", "worker", "model", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, default=18280)
    args = parser.parse_args()
    for name in ("source", "worker", "model", "tokenizer", "out"):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).with_name("server.py")),
               "--source", str(args.source), "--worker", str(args.worker),
               "--model", str(args.model), "--tokenizer", str(args.tokenizer),
               "--gpu", args.gpu, "--port", str(args.port), "--context", "32768",
               "--budget", "2048", "--reserve", "4096", "--max-tokens", "64",
               "--", "--threads", "8", "--stage-mib", "512", "--tail-skip", "0"]
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return opener.open(urllib.request.Request(base + path, data=data,
                           headers={"Content-Type": "application/json"}), timeout=180)
    result = {"command": command, "gpu_uuid": args.gpu, "checks": [], "status": "running"}
    with (args.out / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=0x08000000 if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"frontend exited: {proc.returncode}")
                try:
                    with request("/health") as response:
                        result["health"] = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("frontend did not become healthy")
                    time.sleep(0.25)
            for endpoint in ("/v1/models", "/props"):
                with request(endpoint) as response:
                    result[endpoint] = json.load(response)
            assert result["/props"]["backend"] == "polystrata", result["/props"]
            assert result["/props"]["kvmem"]["enabled"]
            assert "last_user_unicode_characters" in result["/props"]["kvmem"]["query_ranges"]
            payload = {"model": "kvmem-polystrata-qwen35", "messages": [
                       {"role": "user", "content": "请只输出数字：7乘以8等于多少？"}],
                       "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            with request("/v1/chat/completions", payload) as response:
                complete = json.load(response)
            assert complete["choices"][0]["message"]["content"].strip() == "56", complete
            result["checks"].append({"nonstream_unicode": complete})
            payload.update(stream=True)
            chunks = []
            with request("/v1/chat/completions", payload) as response:
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].decode("utf-8").strip()
                    if data == "[DONE]":
                        break
                    chunks.append(json.loads(data))
            answer = "".join(chunk["choices"][0].get("delta", {}).get("content", "") for chunk in chunks)
            assert answer.strip() == "56", chunks
            result["checks"].append({"stream_unicode": chunks})
            payload.update(stream=False)
            payload["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
            try:
                with request("/v1/chat/completions", payload) as response:
                    raise AssertionError("vision was accepted: " + response.read().decode())
            except urllib.error.HTTPError as error:
                assert 400 <= error.code < 500, error.code
                result["checks"].append({"vision_rejected": error.code})
            payload["messages"][0]["content"] = "What is 2 + 3? Reply only with the number."
            with request("/v1/chat/completions", payload) as response:
                recovery = json.load(response)
            assert recovery["choices"][0]["message"]["content"].strip() == "5", recovery
            result["checks"].append({"recovery_after_rejection": recovery})
            document = "资料：Reference records. Answer from these records.\n" + "".join(
                "The access key for Project Atlas is MARBLE-4827. Project Atlas uses MARBLE-4827.\n"
                if i == 70 else f"Record {i}: Project Cedar-{i} has inspection date 2031-04-19 and serial REF-{i:05d}.\n"
                for i in range(353))
            question = "Find Project Atlas in the reference records. What is its exact access key? " \
                       "Copy the access key from its record. Reply only with that key, without explanation."
            payload["messages"][0]["content"] = document + "\n" + question
            payload["kvmem_query"] = {"message":0,"start":len(document)+1,"end":len(document)+1+len(question)}
            payload["max_tokens"] = 48
            with request("/v1/chat/completions", payload) as response:
                cold = json.load(response)
            assert cold["choices"][0]["message"]["content"].strip() == "MARBLE-4827", cold
            result["checks"].append({"explicit_range_cold_answer":cold,"range":dict(payload["kvmem_query"])})
            payload["stream"] = True
            cold_chunks = []
            with request("/v1/chat/completions", payload) as response:
                for line in response:
                    if line.startswith(b"data: "):
                        data = line[6:].decode("utf-8").strip()
                        if data == "[DONE]":
                            break
                        cold_chunks.append(json.loads(data))
            assert "".join(c["choices"][0].get("delta", {}).get("content", "") for c in cold_chunks).strip() == "MARBLE-4827"
            result["checks"].append({"explicit_range_cold_stream":cold_chunks})
            payload["stream"] = False
            for spec in ({"message":0,"start":True,"end":2}, {"message":0,"start":3,"end":3},
                         {"message":1,"start":0,"end":2}, {"tokens":[0,2]}, None):
                payload["kvmem_query"] = spec
                try:
                    with request("/v1/chat/completions", payload) as response:
                        raise AssertionError("an invalid explicit range was accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400, error.code
            result["checks"].append({"invalid_explicit_ranges_rejected":5})
            sys.path[:0] = [str(args.source), str(args.source / "tools")]
            from strata_tokenizer import Tokenizer
            from serve.frontend import ChatTemplate, LITERAL_TAGS
            from serve import kvmem
            template = ChatTemplate(args.tokenizer / "chat_template.jinja")
            class Frontend:
                literals = LITERAL_TAGS
                tok = Tokenizer.from_dir(args.tokenizer)
                def render_prompt(self, messages, tools, kwargs):
                    return template.render(messages, tools=tools, enable_thinking=False)
            frontend = Frontend()
            message = "文档不是查询。\nWhat is 2 + 3? Reply only with the number."
            messages = [{"role":"user", "content":message}]
            ids = kvmem.encode(frontend, messages, None, {})
            selected = kvmem.resolve_query(frontend, messages, None, {}, ids,
                                           {"message":0,"start":len("文档不是查询。\n"),"end":len(message)})
            payload.update(messages=messages, kvmem_query={"tokens":list(selected.kvmem_query)}, max_tokens=32)
            with request("/v1/chat/completions", payload) as response:
                token_range = json.load(response)
            assert token_range["choices"][0]["message"]["content"].strip() == "5", token_range
            assert token_range["usage"]["prompt_tokens"] == len(ids), token_range
            result["checks"].append({"explicit_token_range":token_range,"range":list(selected.kvmem_query)})
            def concurrent_range(question, answer):
                message = "上下文不是查询。\n" + question
                body = {"model":"kvmem-polystrata-qwen35", "messages":[{"role":"user","content":message}],
                        "max_tokens":32, "temperature":0, "chat_template_kwargs":{"enable_thinking":False},
                        "kvmem_query":{"message":0,"start":len("上下文不是查询。\n"),"end":len(message)}}
                with request("/v1/chat/completions", body) as response:
                    value = json.load(response)
                assert value["choices"][0]["message"]["content"].strip() == answer, value
                return value
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(concurrent_range, "What is 2 + 3? Reply only with the number.", "5")
                second = pool.submit(concurrent_range, "请只输出数字：7乘以8等于多少？", "56")
                result["checks"].append({"fifo_request_local_ranges":[first.result(), second.result()]})
            result["status"] = "pass"
        except Exception as error:
            result.update(status="fail", error=str(error))
            raise
        finally:
            # Windows process termination is not a console Ctrl+C; terminate the
            # entire owned test process tree so no native GPU child is orphaned.
            if proc.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=True)
                else:
                    proc.terminate()
                proc.wait(timeout=30)
            (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"])}))


if __name__ == "__main__":
    main()
