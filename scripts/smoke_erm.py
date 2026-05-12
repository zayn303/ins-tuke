"""Run all 12 ERM smoke tests (1 epoch) sequentially on login node.

Usage:
    python scripts/smoke_erm.py
    python scripts/smoke_erm.py --data-root /home/ak562fx/ins-tuke/Data
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = ["wav2vec2", "hubert", "wavlm"]
HELD_OUTS = [0, 1, 2, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/ak562fx/ins-tuke/Data")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_dir = ROOT / "logs" / f"{ts}_erm_smoke"
    ckpt_dir = ROOT / "checkpoints" / f"{ts}_erm_smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    combos = [(m, h) for h in HELD_OUTS for m in MODELS]
    results = []

    print(f"Smoke run: {len(combos)} combos → {log_dir}")
    print("=" * 60)

    for model, held_out in combos:
        tag = f"erm_{model}_held{held_out}"
        out_path = log_dir / f"{tag}.out"
        err_path = log_dir / f"{tag}.err"

        print(f"\n[{tag}] starting ...")

        cmd = [
            sys.executable, "scripts/train.py",
            "method=erm",
            f"model={model}",
            f"held_out_domain={held_out}",
            "epochs=1",
            f"data_root={args.data_root}",
            f"checkpoint_dir={ckpt_dir}",
            "wandb_offline=true",
        ]

        with open(out_path, "w") as fout, open(err_path, "w") as ferr:
            proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate()

        out_path.write_text(stdout)
        err_path.write_text(stderr)

        # stream last few lines to terminal
        for line in stdout.splitlines()[-5:]:
            print(f"  {line}")
        if stderr.strip():
            for line in stderr.splitlines()[-3:]:
                print(f"  ERR: {line}")

        ok = proc.returncode == 0
        results.append((tag, ok, proc.returncode))
        print(f"[{tag}] {'PASS' if ok else 'FAIL'} (exit {proc.returncode})")

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
