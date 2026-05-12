"""Run all 12 ERM smoke tests (1 epoch) in parallel.

Usage:
    python scripts/smoke_erm.py                   # 4 workers default
    python scripts/smoke_erm.py --workers 12      # all at once
    python scripts/smoke_erm.py --data-root /home/ak562fx/ins-tuke/Data
"""
import argparse
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = ["wav2vec2", "hubert", "wavlm"]
HELD_OUTS = [0, 1, 2, 3]

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def run_combo(model, held_out, log_dir, ckpt_dir, data_root):
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
    ]

    proc = subprocess.Popen(
        cmd, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    out_path.write_text(stdout)
    err_path.write_text(stderr)

    ok = proc.returncode == 0
    status = "PASS" if ok else f"FAIL(exit {proc.returncode})"
    log(f"[{tag}] {status}")

    return tag, ok, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/ak562fx/ins-tuke/Data")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers (default 4, max 12)")
    args = parser.parse_args()

    workers = min(args.workers, 12)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_dir = ROOT / "logs" / f"{ts}_erm_smoke"
    ckpt_dir = ROOT / "checkpoints" / f"{ts}_erm_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    combos = [(m, h) for h in HELD_OUTS for m in MODELS]

    print(f"Smoke run: {len(combos)} combos, {workers} workers → {log_dir}")
    print("=" * 60)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_combo, m, h, log_dir, ckpt_dir, args.data_root): (m, h)
            for m, h in combos
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])

    print("\n" + "=" * 60)
    print(f"Results ({log_dir.name}):")
    failed = []
    for tag, ok, rc in results:
        status = "PASS" if ok else f"FAIL(exit {rc})"
        print(f"  {status:16s}  {tag}")
        if not ok:
            failed.append(tag)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(combos)}")
        sys.exit(1)
    else:
        print(f"ALL {len(combos)}/{len(combos)} PASSED")


if __name__ == "__main__":
    main()
