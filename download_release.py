"""Download the inference caches (and optionally the checkpoints) from the GitHub release.

The caches hold the logits, penultimate embeddings and correctness of both trained
ResNet-18 models on the validation split, the clean test set and all of CIFAR-10-C.
With them, `eval_shift.py --use_cache` reproduces every number in the paper on a CPU
in a few minutes, without CIFAR-10-C (2.9 GB) or 2 x 201 epochs of training.

Usage:
    python download_release.py                  # caches -> ./demo_output/
    python download_release.py --models         # also the two model.pth files -> ./checkpoints/
"""
import argparse
import os
import sys
import urllib.request

BASE = 'https://github.com/TaegwonL/DS340W-Group2-Midterm/releases/download/v1.0/'
CACHES = {'cache_baseline.npz': 877842264, 'cache_fmfp.npz': 877842264}
MODELS = {'baseline_model.pth': 44768075, 'fmfp_model.pth': 44769671}


def fetch(name, size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print('already have', path)
        return
    tmp = path + '.part'
    with urllib.request.urlopen(BASE + name) as r, open(tmp, 'wb') as f:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            sys.stdout.write('\r{}  {:6.1f} / {:.1f} MB'.format(name, done / 1e6, size / 1e6))
            sys.stdout.flush()
    print()
    if os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise SystemExit('{}: expected {} bytes, got a different size; run again'.format(name, size))
    os.replace(tmp, path)
    print('wrote', path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='./demo_output', help='folder for the caches')
    p.add_argument('--models', action='store_true', help='also download the two checkpoints')
    p.add_argument('--models_out', default='./checkpoints')
    args = p.parse_args()
    for name, size in CACHES.items():
        fetch(name, size, args.out)
    if args.models:
        for name, size in MODELS.items():
            fetch(name, size, args.models_out)


if __name__ == '__main__':
    main()
