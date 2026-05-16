"""Run all 15 ERM smoke tests (1 epoch, 3 parallel workers by default).

Usage:
    python scripts/smoke_erm.py                       # 3 workers default (one per backbone)
    python scripts/smoke_erm.py --workers 15          # fully parallel (OOM risk on single GPU)
    python scripts/smoke_erm.py --workers 1           # sequential
    python scripts/smoke_erm.py --data-root /home/ak562fx/ins-tuke/Data
    python scripts/smoke_erm.py --batch-size 2        # reduce VRAM further
"""
import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = ["wav2vec2", "hubert", "wavlm"]
HELD_OUTS = [0, 1, 2, 3, 4]

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def run_combo(model, held_out, log_dir, ckpt_dir, data_root, batch_size, num_workers):
    tag = f"erm_{model}_held{held_out}"
    out_path = log_dir / f"{tag}.out"
    err_path = log_dir / f"{tag}.err"

    log(f"[{tag}] start")

    cmd = [
        sys.executable, "scripts/train.py",
        "method=erm",
        f"model={model}",
        f"held_out_domain={held_out}",
        "epochs=1",
        f"data_root={data_root}",
        f"checkpoint_dir={ckpt_dir}",
        "wandb_offline=true",
        f"batch_size={batch_size}",
        f"num_workers={num_workers}",
    ]

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.Popen(
        cmd, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    stdout, stderr = proc.communicate()

    out_path.write_text(stdout)
    err_path.write_text(stderr)

    if proc.returncode == 0:
        status = "PASS"
    elif "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
        status = "FAIL(OOM)"
    else:
        status = f"FAIL(exit {proc.returncode})"

    ok = proc.returncode == 0
    log(f"[{tag}] {status}")

    return tag, ok, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/ak562fx/ins-tuke/Data")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel workers (default 3 — one per backbone; use 1 for sequential)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size per combo (default 4 — low VRAM)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers (default 4 — matches SLURM)")
    args = parser.parse_args()

    workers = min(args.workers, len(MODELS) * len(HELD_OUTS))

    if workers > 1:
        print(f"WARNING: --workers {workers} > 1; GPU contention is your responsibility")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_dir = ROOT / "logs" / f"{ts}_erm_smoke"
    ckpt_dir = ROOT / "checkpoints" / f"{ts}_erm_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    combos = [(m, h) for h in HELD_OUTS for m in MODELS]

    print(f"Smoke run: {len(combos)} combos, {workers} workers, batch_size={args.batch_size}, num_workers={args.num_workers} → {log_dir}")
    print("=" * 60)

    results = []
    seen_tags = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_combo, m, h, log_dir, ckpt_dir, args.data_root, args.batch_size, args.num_workers): (m, h)
            for m, h in combos
        }
        for fut in as_completed(futures):
            try:
                tag, ok, status = fut.result()
                results.append((tag, ok, status))
                seen_tags.add(tag)
            except Exception as e:
                m, h = futures[fut]
                tag = f"erm_{m}_held{h}"
                results.append((tag, False, f"ERROR({repr(e)})"))
                seen_tags.add(tag)

    expected = {f"erm_{m}_held{h}" for h in HELD_OUTS for m in MODELS}
    for tag in sorted(expected - seen_tags):
        results.append((tag, False, "ERROR(missing)"))

    results.sort(key=lambda x: x[0])

    print("\n" + "=" * 60)
    print(f"Results ({log_dir.name}):")
    failed = []
    for tag, ok, status in results:
        print(f"  {status:20s}  {tag}")
        if not ok:
            failed.append(tag)

    print()
    if any(s == "FAIL(OOM)" for _, _, s in results):
        print("HINT: OOM detected — rerun with --workers 1 or reduce --batch-size")
        print()

    if failed:
        print(f"FAILED: {len(failed)}/{len(combos)}")
        sys.exit(1)
    else:
        print(f"ALL {len(combos)}/{len(combos)} PASSED")


if __name__ == "__main__":
    main()
