from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10

# Run this script from the FMFP project folder.
# It reproduces the same split used in utils/data.py:
# - original CIFAR-10 training set: 50,000
# - validation: first 10% after NumPy shuffle with seed 42 = 5,000
# - training: remaining 45,000
# - official CIFAR-10 test set: 10,000

project = Path.cwd()

# Try to locate the CIFAR-10 files already downloaded by the project.
found = list(project.rglob("cifar-10-batches-py"))
if found:
    cifar_root = found[0].parent
    print(f"Found existing CIFAR-10 data at: {cifar_root}")
    download = False
else:
    # Fallback location. torchvision will download only if needed.
    cifar_root = project / "data" / "cifar10_data"
    cifar_root.mkdir(parents=True, exist_ok=True)
    print(f"CIFAR-10 folder not found. Using: {cifar_root}")
    download = True

train_full = CIFAR10(root=str(cifar_root), train=True, download=download)
test_full = CIFAR10(root=str(cifar_root), train=False, download=download)

# Reproduce utils/data.py exactly.
num_train = len(train_full)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))

np.random.seed(42)
np.random.shuffle(indices)

train_idx = np.array(indices[split:], dtype=np.int64)
val_idx = np.array(indices[:split], dtype=np.int64)

x_all = np.asarray(train_full.data, dtype=np.uint8)
y_all = np.asarray(train_full.targets, dtype=np.int64)

x_train = x_all[train_idx]
y_train = y_all[train_idx]

x_val = x_all[val_idx]
y_val = y_all[val_idx]

x_test = np.asarray(test_full.data, dtype=np.uint8)
y_test = np.asarray(test_full.targets, dtype=np.int64)
test_idx = np.arange(len(test_full), dtype=np.int64)

out = project / "split_data"
out.mkdir(exist_ok=True)

train_file = out / "Train_CIFAR10_45000.npz"
val_file = out / "Validation_CIFAR10_5000.npz"
test_file = out / "Test_CIFAR10_10000.npz"

np.savez_compressed(
    train_file,
    images=x_train,
    labels=y_train,
    original_indices=train_idx,
)
np.savez_compressed(
    val_file,
    images=x_val,
    labels=y_val,
    original_indices=val_idx,
)
np.savez_compressed(
    test_file,
    images=x_test,
    labels=y_test,
    original_indices=test_idx,
)

print("\nCreated 3 dataset files:")
for path, x, y in [
    (train_file, x_train, y_train),
    (val_file, x_val, y_val),
    (test_file, x_test, y_test),
]:
    mb = path.stat().st_size / (1024 * 1024)
    print(f"{path.name}: images={x.shape}, labels={y.shape}, size={mb:.1f} MB")

print("\nSplit complete.")
print("Train: 45,000 / Validation: 5,000 / Test: 10,000")
