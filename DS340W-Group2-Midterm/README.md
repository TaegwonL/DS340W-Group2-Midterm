# DS 340W Group 2 — Midterm Code Repository

This repository collects code and documentation for our midterm project, **Reliable AI Failure Prediction Under Changing Data**.

## Research direction
We are studying whether failure prediction remains useful when test inputs differ from training data. The proposed comparison connects three parent papers:

1. **FMFP**: confidence-based failure prediction and training with flat minima (Paper 1).
2. **Risk Advisor**: an independent failure-risk estimator (Paper 2).
3. **Temporal quality degradation**: motivation to test changes after deployment (Paper 3).

CIFAR-10-C corruption severity is planned as a **controlled data-shift proxy**, not a simulation of actual elapsed time.

## What is included in this snapshot
This snapshot contains **Parent Paper 1 (FMFP)** source code and a CIFAR-10 data-splitting script. It does **not** yet include the full integrated Risk Advisor / CIFAR-10-C experiments or verified final comparison results.

- `main_fmfp.py`, `train_fmfp.py`: FMFP training entrypoint and training routine.
- `main_base.py`, `train_base.py`: baseline and comparison training code.
- `model/`: model definitions, including ResNet18.
- `utils/`: preprocessing, metrics, SAM and other utilities.
- `export_cifar10_splits.py`: reproduces a 45,000 train / 5,000 validation / 10,000 test CIFAR-10 split (NumPy seed 42).
- `framework.png`: FMFP framework diagram from the source repository.

## Environment and usage
Dependencies are listed in `requirements.txt` without pinned versions; compatibility with every new version is **not verified**. The original FMFP reference used Python 3.9.7 / PyTorch 1.10.0. Run commands from this repository's root on a machine with a supported CUDA GPU and PyTorch installation.

```bash
python export_cifar10_splits.py
python main_fmfp.py --data cifar10 --model resnet18 --method fmfp --gpu 0 --run 1
```

**Important:** FMFP uses stochastic weight averaging starting after epoch 120. A 1-epoch smoke test checks that code runs; it does not reproduce final FMFP performance. The baseline script `main_base.py` currently imports the Unix-only `resource` module and may need a Windows compatibility fix before it runs on Windows.

The data splitting script creates `split_data/` locally; resulting datasets should not be committed. The training scripts download CIFAR-10 through torchvision when necessary.

## Original work and attribution
FMFP was introduced in *Rethinking Confidence Calibration for Failure Prediction* (Zhu et al., ECCV 2022). The FMFP training/model/utilities files and framework figure derive from the authors' public code: https://github.com/Impression2805/FMFP . See the included original MIT `LICENSE` and `FMFP_ORIGINAL_README.md` for the original project description. This repository also includes student-written data export and a Windows import adjustment. Do not interpret this repository as evidence that the paper's full experiments were reproduced.

## Dataset / next steps
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Obtain and validate CIFAR-10-C and integrated evaluation files before claiming corruption-robustness results.
- Add verified runs, seed values, evaluation tables, and figures once the team finishes testing.
