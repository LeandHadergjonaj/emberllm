#!/usr/bin/env python3
"""compare_llamacpp.py — measured, reproducible emberllm vs llama.cpp comparison.

Runs the same prefill/decode benchmarks on both engines, on the same machine,
with the same models (identical source weights, each engine's own converter and
Q8_0 quantizer), and writes every raw number to bench/results/ as JSON.

Methodology (deliberately mirrored between engines):
  * prefill ("ppN"):  time to process an N-token prompt as a batch, tok/s.
      - emberllm:   `ember bench --pp N`   (1 warmup + 5 timed reps, mean +/- sd)
      - llama.cpp:  `llama-bench -p N`     (warmup + 5 timed reps, mean +/- sd)
  * decode ("tgN @ dD"): time to generate N tokens one at a time, starting from
    a D-token context — decode speed depends on context depth, so it is pinned.
      - emberllm:   `ember bench --pp D --tg N` (decode timed separately from prefill)
      - llama.cpp:  `llama-bench -n N -d D`
  * both engines see synthetic token streams (no sampling cost in the timing;
    llama-bench times eval only, ember bench times forward+argmax-free decode).

Usage:
  python3 bench/compare_llamacpp.py --llama-cpu <dir/llama-bench> \
      [--llama-metal <dir/llama-bench>] [--out bench/results] [--quick]

The script only *reads* models and *runs* binaries; build/convert steps are
documented in notebooks/benchmark_vs_llamacpp.ipynb and the README.
"""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, quant, ember model, gguf model).  All four models come from the same
# upstream weights; each engine's own converter+quantizer produced its file.
MODELS = [
    ("qwen3-0.6b",   "Q8_0", "models/qwen3-0.6b-q8.ember",   "models/gguf/qwen3-0.6b-q8_0.gguf"),
    ("qwen2.5-0.5b", "Q8_0", "models/qwen2.5-0.5b-q8.ember", "models/gguf/qwen2.5-0.5b-q8_0.gguf"),
    ("smollm2-135m", "Q8_0", "models/smollm2-135m-q8.ember", "models/gguf/smollm2-135m-q8_0.gguf"),
    ("stories110M",  "Q8_0", "models/stories110M-q8.ember",  "models/gguf/stories110M-q8_0.gguf"),
    # Q4_0 included deliberately: emberllm's Q4_0 kernel is NOT SIMD-optimized
    # and llama.cpp repacks Q4_0 for aarch64 — an expected, honest loss.
    ("qwen3-0.6b",   "Q4_0", "models/qwen3-0.6b-q4.ember",   "models/gguf/qwen3-0.6b-q4_0.gguf"),
]

THREADS = [1, 2, 4, 6, 8]
PP_SIZES = [128, 512]      # prefill batch sizes
TG_N = 128                 # decode length
TG_DEPTHS = [128, 512]     # context depth at which decode is timed
REPS = 5


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def capture_env(llama_cpu, llama_metal):
    def out(cmd):
        try:
            return sh(cmd).stdout.strip()
        except Exception as e:  # noqa: BLE001 - env capture is best-effort
            return f"<unavailable: {e}>"

    env = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "cpu": out(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cores_total": out(["sysctl", "-n", "hw.ncpu"]),
        "cores_performance": out(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
        "cores_efficiency": out(["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
        "mem_bytes": out(["sysctl", "-n", "hw.memsize"]),
        "macos": out(["sw_vers", "-productVersion"]),
        "power": out(["pmset", "-g", "batt"]),
        "thermal": out(["pmset", "-g", "therm"]),
        "cc_version": out(["cc", "--version"]).splitlines()[0] if out(["cc", "--version"]) else "",
        "ember_commit": out(["git", "-C", REPO, "rev-parse", "HEAD"]),
        "ember_dirty": bool(out(["git", "-C", REPO, "status", "--porcelain", "--", "src", "Makefile"])),
        "ember_cflags": "-std=c11 -O3 -ffast-math -march=native -funroll-loops -pthread (Makefile default)",
        "llama_cpu_bin": llama_cpu,
        "llama_metal_bin": llama_metal,
    }
    for label, b in (("llama_cpu", llama_cpu), ("llama_metal", llama_metal)):
        if not b:
            continue
        d = os.path.dirname(os.path.dirname(os.path.abspath(b)))  # build dir -> repo
        env[label + "_commit"] = out(["git", "-C", d, "rev-parse", "HEAD"])
        env[label + "_describe"] = out(["git", "-C", d, "describe", "--tags", "--always"])
    return env


def warm_cache(path):
    """Read the model file once so first-touch page faults don't skew run 1.
    Both engines also do an untimed warmup rep; this just equalizes cold starts."""
    with open(path, "rb") as f:
        while f.read(1 << 24):
            pass


def run_ember(model, threads, pp, tg, reps):
    cmd = [os.path.join(REPO, "ember"), "bench", os.path.join(REPO, model),
           "--pp", str(pp), "--tg", str(tg), "--threads", str(threads), "--reps", str(reps)]
    r = sh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ember bench failed: {' '.join(cmd)}\n{r.stderr}")
    text = r.stdout
    m_pp = re.search(r"prefill \(pp(\d+)\):\s+([\d.]+) tok/s\s+\(\+/- ([\d.]+)\)", text)
    m_tg = re.search(r"decode  \(tg(\d+)\):\s+([\d.]+) tok/s\s+\(\+/- ([\d.]+)\)", text)
    if not (m_pp and m_tg):
        raise RuntimeError(f"could not parse ember bench output:\n{text}")
    return {
        "cmd": " ".join(cmd), "stdout": text,
        "pp": {"n": int(m_pp.group(1)), "mean": float(m_pp.group(2)), "stddev": float(m_pp.group(3))},
        "tg": {"n": int(m_tg.group(1)), "depth": pp, "mean": float(m_tg.group(2)), "stddev": float(m_tg.group(3))},
    }


def run_llama_bench(bench_bin, gguf, threads_list, pp_list, tg_n, depths, reps, ngl):
    """One llama-bench invocation; returns its parsed -o json rows."""
    cmd = [bench_bin, "-m", os.path.join(REPO, gguf), "-r", str(reps), "-o", "json",
           "-t", ",".join(map(str, threads_list)), "-ngl", str(ngl)]
    if pp_list:
        cmd += ["-p", ",".join(map(str, pp_list)), "-n", "0"]
    else:
        cmd += ["-p", "0", "-n", str(tg_n), "-d", ",".join(map(str, depths))]
    r = sh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {' '.join(cmd)}\n{r.stderr}")
    rows = json.loads(r.stdout)
    for row in rows:
        row["_cmd"] = " ".join(cmd)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpu", required=True, help="path to CPU-only llama-bench")
    ap.add_argument("--llama-metal", default=None, help="path to Metal llama-bench (optional context rows)")
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    ap.add_argument("--quick", action="store_true", help="2 reps, 1 thread count — smoke test of the harness only")
    args = ap.parse_args()

    threads = [1] if args.quick else THREADS
    reps = 2 if args.quick else REPS
    os.makedirs(args.out, exist_ok=True)

    env = capture_env(args.llama_cpu, args.llama_metal)
    with open(os.path.join(args.out, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(f"env: {env['cpu']}, {env['cores_performance']}P+{env['cores_efficiency']}E cores, macOS {env['macos']}")

    results = []   # flat rows: engine/model/quant/threads/test/n/depth/mean/stddev
    raw = []       # full command output for auditability

    def add(engine, model, quant, t, test, n, depth, mean, sd, src):
        results.append({"engine": engine, "model": model, "quant": quant, "threads": t,
                        "test": test, "n": n, "depth": depth,
                        "tok_per_sec_mean": round(mean, 2), "tok_per_sec_stddev": round(sd, 2)})
        print(f"  {engine:16s} {model:13s} {quant} t={t} {test}{n}"
              f"{'@d%d' % depth if depth else '':8s} {mean:9.2f} tok/s (+/- {sd:.2f})")
        raw.append({"engine": engine, "model": model, "quant": quant, "src": src})

    for name, quant, ember_path, gguf_path in MODELS:
        for p in (ember_path, gguf_path):
            if not os.path.exists(os.path.join(REPO, p)):
                print(f"SKIP {name} {quant}: missing {p}", file=sys.stderr)
                break
        else:
            print(f"\n== {name} ({quant}) ==")
            # --- emberllm: one bench run per (threads, depth-config) ---
            warm_cache(os.path.join(REPO, ember_path))
            for t in threads:
                a = run_ember(ember_path, t, 512, TG_N, reps)   # -> pp512 + tg128@d512
                b = run_ember(ember_path, t, 128, TG_N, reps)   # -> pp128 + tg128@d128
                add("emberllm", name, quant, t, "pp", 512, 0, a["pp"]["mean"], a["pp"]["stddev"], a)
                add("emberllm", name, quant, t, "tg", TG_N, 512, a["tg"]["mean"], a["tg"]["stddev"], a)
                add("emberllm", name, quant, t, "pp", 128, 0, b["pp"]["mean"], b["pp"]["stddev"], b)
                add("emberllm", name, quant, t, "tg", TG_N, 128, b["tg"]["mean"], b["tg"]["stddev"], b)
            # --- llama.cpp CPU: one invocation for all thread counts per test kind ---
            warm_cache(os.path.join(REPO, gguf_path))
            for rows, kind in (
                (run_llama_bench(args.llama_cpu, gguf_path, threads, PP_SIZES, TG_N, [], reps, 0), "pp"),
                (run_llama_bench(args.llama_cpu, gguf_path, threads, [], TG_N, TG_DEPTHS, reps, 0), "tg"),
            ):
                for row in rows:
                    n = row["n_prompt"] if kind == "pp" else row["n_gen"]
                    add("llama.cpp-cpu", name, quant, row["n_threads"], kind, n,
                        row.get("n_depth", 0), row["avg_ts"], row["stddev_ts"], row)
            # --- llama.cpp Metal (GPU): context rows, one thread config ---
            if args.llama_metal:
                for rows, kind in (
                    (run_llama_bench(args.llama_metal, gguf_path, [6], PP_SIZES, TG_N, [], reps, 99), "pp"),
                    (run_llama_bench(args.llama_metal, gguf_path, [6], [], TG_N, TG_DEPTHS, reps, 99), "tg"),
                ):
                    for row in rows:
                        n = row["n_prompt"] if kind == "pp" else row["n_gen"]
                        add("llama.cpp-metal", name, quant, row["n_threads"], kind, n,
                            row.get("n_depth", 0), row["avg_ts"], row["stddev_ts"], row)

    # Drift check: re-run the very first config after the whole sweep. If the
    # machine's thermal/background state changed materially, this exposes it.
    name, quant, ember_path, _ = MODELS[0]
    d = run_ember(ember_path, threads[0], 512, TG_N, reps)
    add("emberllm-driftcheck", name, quant, threads[0], "pp", 512, 0, d["pp"]["mean"], d["pp"]["stddev"], d)
    add("emberllm-driftcheck", name, quant, threads[0], "tg", TG_N, 512, d["tg"]["mean"], d["tg"]["stddev"], d)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"env": env, "results": results}, f, indent=2)
    with open(os.path.join(args.out, "raw.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nwrote {len(results)} rows -> {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
