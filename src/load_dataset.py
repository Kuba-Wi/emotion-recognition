import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Tuple
import zipfile
from abc import ABC, abstractmethod
import torchvision.transforms.functional as F


class TransformBase(ABC):
    @abstractmethod
    def __call__(self, sample: Tuple[Image.Image, int]) -> Tuple[torch.Tensor, int]:
        pass


class Transform(TransformBase):
    def __call__(self, sample: Tuple[Image.Image, int]) -> Tuple[torch.Tensor, int]:
        image = sample[0]
        label = sample[1]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        return transform(image), label


class TransformToRGB(TransformBase):
    def __call__(self, sample):
        image, label = sample
        image = image.convert('RGB')

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return transform(image), label


class ImageDataset(Dataset):
    def __init__(self, dataset_path: str, images_dirs: List[str], transform: TransformBase = None):
        self.transform = transform
        self.images_paths = []

        for i, dir_name in enumerate(images_dirs):
            dir_path = os.path.join(dataset_path, dir_name)
            for img_name in os.listdir(dir_path):
                self.images_paths.append((os.path.join(dir_path, img_name), i))

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        sample = (Image.open(self.images_paths[idx][0]),
                  self.images_paths[idx][1])

        if self.transform:
            sample = self.transform(sample)

        return sample


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def extract_zip(zip_path: str, destination_path: str) -> None:
    # Check if the ZIP file exists
    if not os.path.exists(zip_path):
        print(f"ZIP file '{zip_path}' does not exist.")
        return

    # Create the destination path if it does not exist
    os.makedirs(destination_path, exist_ok=True)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

    print(f"ZIP file has been extracted to: {destination_path}")


def remove_empty_images(dataset_path, threshold=10):
    removed_files = []

    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                # Open the image and calculate the total pixel value
                image = Image.open(file_path)
                total_pixel_value = sum(image.convert("L").point(lambda p: p > 0).getdata())

                # Check if the image is empty based on the threshold
                if total_pixel_value < threshold:
                    removed_files.append(file_path)
                    os.remove(file_path)
                    print(f"Removed empty image: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return removed_files


if __name__ == "__main__":
    extract_zip("data/FER-2013.zip", "data/original")

    removed_files_train = remove_empty_images("data/train")
    print(f"Removed {len(removed_files_train)} empty images.")
    removed_files_test = remove_empty_images("data/train")
    print(f"Removed {len(removed_files_test)} empty images.")
