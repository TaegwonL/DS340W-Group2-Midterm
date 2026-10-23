import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

MEAN = [0.491, 0.482, 0.447]
STD = [0.247, 0.243, 0.262]
PER_SEVERITY = 10000


class CorruptedCIFAR(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # np.asarray pulls one image out of the memory-mapped file
        return self.transform(Image.fromarray(np.asarray(self.images[i]))), int(self.labels[i]), i


def open_corruption(root, name):
    """Memory-map one corruption file (50,000 images = 5 severities x 10,000)."""
    path = os.path.join(root, name + '.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode='r')


def load_labels(root):
    # labels.npy repeats the CIFAR-10 test labels once per severity
    return np.load(os.path.join(root, 'labels.npy'))[:PER_SEVERITY]


def severity_slice(images, labels, severity, limit=None):
    lo = (severity - 1) * PER_SEVERITY
    n = PER_SEVERITY if limit is None else min(limit, PER_SEVERITY)
    return images[lo:lo + n], labels[:n]


def get_loader(images, labels, batch_size=128, num_workers=0):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return DataLoader(CorruptedCIFAR(images, labels, tf), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)
