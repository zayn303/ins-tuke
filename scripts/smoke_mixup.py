"""Run all 12 DIFL smoke tests (1 epoch).

Usage:
    python scripts/smoke_mixup.py
    python scripts/smoke_mixup.py --workers 1
    python scripts/smoke_mixup.py --data-root /home/ak562fx/ins-tuke/Data
    python scripts/smoke_mixup.py --batch-size 2
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
HELD_OUTS = [0, 1, 3, 4]

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def run_combo(model, held_out, log_dir, ckpt_dir, data_root, batch_size, num_workers):
    tag = f"mixup_{model}_held{held_out}"
    cmd = [
        sys.executable, "scripts/train.py",
        "method=mixup",
        f"model={model}",
        f"held_out_domain={held_out}",
        "epochs=1",
        "unfreeze_top_n_layers=0",
        "mixup_alpha=0.2",
        f"data_root={data_root}",
        f"checkpoint_dir={ckpt_dir}",
        "wandb_offline=true",
        f"batch_size={batch_size}",
        f"num_workers={num_workers}",
    ]
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    log(f"[{tag}] start")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = proc.communicate()
    (log_dir / f"{tag}.out").write_text(stdout)
    (log_dir / f"{tag}.err").write_text(stderr)
    if proc.returncode == 0:
        status = "PASS"
    elif "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
        status = "FAIL(OOM)"
    else:
        status = f"FAIL(exit {proc.returncode})"
    log(f"[{tag}] {status}")
    return tag, proc.returncode == 0, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/ak562fx/ins-tuke/Data")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    workers = min(args.workers, len(MODELS) * len(HELD_OUTS))
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_dir = ROOT / "logs" / f"{ts}_mixup_smoke"
    ckpt_dir = ROOT / "checkpoints" / f"{ts}_mixup_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    combos = [(m, h) for h in HELD_OUTS for m in MODELS]
    print(f"Smoke run: {len(combos)} combos, {workers} workers → {log_dir}")
    print("=" * 60)
    results = []
    seen_tags = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_combo, m, h, log_dir, ckpt_dir, args.data_root, args.batch_size, args.num_workers): (m, h) for m, h in combos}
        for fut in as_completed(futures):
            try:
                tag, ok, status = fut.result()
            except Exception as e:
                m, h = futures[fut]
                tag = f"mixup_{m}_held{h}"
                ok, status = False, f"ERROR({repr(e)})"
            results.append((tag, ok, status))
            seen_tags.add(tag)
    for tag in sorted({f"mixup_{m}_held{h}" for h in HELD_OUTS for m in MODELS} - seen_tags):
        results.append((tag, False, "ERROR(missing)"))
    results.sort(key=lambda x: x[0])
    print("\n" + "=" * 60)
    failed = []
    for tag, ok, status in results:
        print(f"  {status:20s}  {tag}")
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
