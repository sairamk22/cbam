import traceback
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableSequence,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Memory
from numpy import ndarray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import Resize
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

ROOT = Path(".")
CACHE = ROOT / "__CACHE__"
SIZE = (128, 128)
SIZE = (224, 224)
SIZED_CACHE = CACHE / ("224x224" if SIZE == (224, 224) else "128x128")
TRAIN_IMAGES = SIZED_CACHE / "x_train.npz"
TRAIN_LABELS = SIZED_CACHE / "y_train.npz"
TEST_IMAGES = SIZED_CACHE / "x_test.npz"
TEST_LABELS = SIZED_CACHE / "y_test.npz"


DIC_PATH = "./VOCdevkit/VOC2012/ImageSets/Main"
# fmt: off
OBJECT_CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
LABELS = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv/monitor",
    255: "void",
}
ROOT = Path(".")
DATA = ROOT / "VOCdevkit/VOC2012"
SPLITS_PATH = DATA / "ImageSets/Main"
SEG_PATH = DATA / "ImageSets/Segmentation"
CATEGORY_FNAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
IMGS = DATA / "JPEGImages"
SEGS = DATA / "SegmentationClass"
# fmt: on


def inspect_data(
    source: Literal["Main", "Segmentation"],
    split: Literal["train", "val"] = "train",
    multiclass: bool = False,
    n_sample_per_class: int = 2,
    seed: int = None,
) -> None:
    """Sample some random images from either the "Main" or "Segmentation" folder and plot them.

    Parameters
    ----------
    source: Literal["Main", "Segmentation"]
        Which folder to simple images from.

    split: Literal["train", "val"]
        Whether to sample from training or validation splits.

    multiclass: bool = False
        If True, plot only images with multiple classes. If False, plot images with only a single
        class. Currently only works when `source="Segmentation"`.

    seed: Optional[int] = None
        For reproducible selection of random images. Default is `None`, which means a different
        seed each time.

    Notes
    -----
    Note that when running this function many samples are not usable for classification and our
    purposes. For example, with `seed=3` the second last image labelled `tvmonitor` instead is
    mostly three young women. We need to decide how to handle such multi-class images. Either
    we exclude them, use the "largest" segmentation label for the class label, or use them only
    in validation. Either way, we must DOCUMENT this decision.

    You can see how we might limit to single class images in this function when the option
    `multiclass` is True/False. You can also see how we could set aside `multiclass` images
    specifically for testing using similar logic.

    Note also in the `multiclass=True` case we use the "largest" class as the overall label. This is
    something to think about for training.
    """
    if seed is not None:
        np.random.seed(seed)
    samples: List[Tuple[str, Path]] = []

    if source == "Main":
        cat_imgs: List[Tuple[str, Path]] = []  # categories and imgs
        ftemplate = f"{{category}}_{split}.txt"
        for category in CATEGORY_FNAMES:
            splitsfile = SPLITS_PATH / ftemplate.format(category=category)
            with open(splitsfile, "r") as file:
                splits = [line.split() for line in file.readlines()]
                paths = [IMGS / f"{split[0]}.jpg" for split in splits]
                indicators = [split[1] for split in splits]
                for indicator, path in zip(indicators, paths):
                    # lines are of form "2008_000023  1" or "2008_000023  -1"
                    # where -1 is if class is not present in that file
                    if str(indicator) == "1":
                        cat_imgs.append((category, path))
            np.random.shuffle(cat_imgs)  # type: ignore
            for i in range(n_sample_per_class):
                img = np.asarray(Image.open(cat_imgs[i][1]))
                samples.append((cat_imgs[i][0], img))
    else:
        all_segs = sorted(
            SEGS.rglob("*.png")
        )  # these are the segmentation masks with class labels
        np.random.shuffle(all_segs)  # type: ignore
        count = 0
        for seg in all_segs:
            mask = np.asarray(
                Image.open(seg), dtype=np.uint8
            ).copy()  # labels are in [0, 255]
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html for "void" info
            mask[
                mask == 255
            ] = 0  # remove "void" class, which is 255 according to PASCAL documentation
            labels, counts = np.unique(mask, return_counts=True)
            if not multiclass:
                if len(labels) > 2:  # multiple classes
                    continue
                img = np.asarray(
                    Image.open(IMGS / f"{seg.stem}.jpg")
                )  # get image from mask name
                category = LABELS[labels[1]]
            else:
                if len(labels) < 3:  # want more than just one class and background
                    continue
                img = np.asarray(
                    Image.open(IMGS / f"{seg.stem}.jpg")
                )  # get image from mask name
                # background (0) usually largest, exclude it
                labels, counts = labels[1:], counts[1:]
                idx = np.argsort(counts)
                cat1 = LABELS[labels[idx[0]]]
                cat2 = LABELS[labels[idx[1]]]
                category = f"{cat1}, {cat2}"  # two most common categories to include in plot title
            samples.append((category, img))
            count += 1
            if count == 20 * n_sample_per_class:
                break

    # layout our images in a grid, with title info to say what we are plotting
    fig, axes = plt.subplots(
        nrows=2 * n_sample_per_class, ncols=10
    )  # 20 categories total
    ax: plt.Axes
    for (category, img), ax in zip(samples, axes.flat):
        ax.imshow(img)
        ax.set_title(category, fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    mc = (
        "May include multiple classes"
        if source == "Main"
        else f"{'Multiclass' if multiclass else 'Single-class'} images"
    )
    suptitle = f"{source} images - {mc}"
    fig.suptitle(suptitle)
    fig.set_size_inches(w=14, h=10)
    fig.tight_layout()
    plt.show()


def load_image(path: Path) -> Tensor:
    try:
        return Resize(SIZE, antialias=True)(read_image(str(path)))
    except:
        traceback.print_exc()


def load_train_images() -> Any:
    if TRAIN_IMAGES.exists() and TRAIN_LABELS.exists():
        print("Found preprocessed training data. Loading...")
        images = np.load(TRAIN_IMAGES)["data"]
        labels = np.load(TRAIN_LABELS)["data"]
        return images, labels

    paths, labels = [], []
    ftemplate = "{category}_train.txt"
    for category_id, category in enumerate(CATEGORY_FNAMES):
        splitsfile = SPLITS_PATH / ftemplate.format(category=category)
        with open(splitsfile, "r") as file:
            splits = [line.split() for line in file.readlines()]
            image_paths = [IMGS / f"{split[0]}.jpg" for split in splits]
            indicators = [split[1] for split in splits]
            for indicator, path in zip(indicators, image_paths):
                # lines are of form "2008_000023  1" or "2008_000023  -1"
                # where -1 is if class is not present in that file
                if str(indicator) == "1":
                    paths.append(path)
                    labels.append(torch.LongTensor([category_id]))
    images = process_map(
        load_image, paths, chunksize=5, desc="Loading and resizing train images"
    )

    labels = torch.cat(labels).numpy()  # long ndarray of shape (n_samples,)
    images = torch.stack(
        images
    ).numpy()  # uint8 ndarray of shape (n_samples, 3, 224, 224)
    np.savez_compressed(TRAIN_IMAGES, data=images)
    print(f"Saved resized train images to {TRAIN_IMAGES}")
    np.savez_compressed(TRAIN_LABELS, data=labels)
    print(f"Saved train image labels to {TRAIN_LABELS}")
    return images, labels


def load_val_images() -> Any:
    if TEST_IMAGES.exists() and TEST_LABELS.exists():
        print("Found preprocessed test data. Loading...")
        images = np.load(TEST_IMAGES)["data"]
        labels = np.load(TEST_LABELS)["data"]
        return images, labels

    paths, labels = [], []
    ftemplate = "{category}_val.txt"
    for category_id, category in enumerate(CATEGORY_FNAMES):
        splitsfile = SPLITS_PATH / ftemplate.format(category=category)
        with open(splitsfile, "r") as file:
            splits = [line.split() for line in file.readlines()]
            image_paths = [IMGS / f"{split[0]}.jpg" for split in splits]
            indicators = [split[1] for split in splits]
            for indicator, path in zip(indicators, image_paths):
                # lines are of form "2008_000023  1" or "2008_000023  -1"
                # where -1 is if class is not present in that file
                if str(indicator) == "1":
                    paths.append(path)
                    labels.append(torch.LongTensor([category_id]))
    images = process_map(
        load_image, paths, chunksize=5, desc="Loading and resizing validation images"
    )

    images = torch.stack(
        images
    ).numpy()  # uint8 ndarray of shape (n_samples, 3, 224, 224)
    labels = torch.cat(labels).numpy()  # long ndarray of shape (n_samples,)
    np.savez_compressed(TEST_IMAGES, data=images)
    print(f"Saved resized test images to {TEST_IMAGES}")
    np.savez_compressed(TEST_LABELS, data=labels)
    print(f"Saved test image labels to {TEST_LABELS}")
    return images, labels


class MainDataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        size: Tuple[int, int] = SIZE,
        classes: List[int] = None,
    ):

        self.size = size
        self.phase = split
        self.samples: List[Tuple[str, Path]] = []  # categories and imgs
        loader = load_val_images if self.phase == "val" else load_train_images
        self.images, self.labels = loader()
        self.images = torch.from_numpy(self.images)
        # https://pytorch.org/hub/pytorch_vision_resnet/
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.images = self.images.float() / 255.0
        self.images = transforms.Normalize(mean=mean, std=std)(self.images)
        self.labels = torch.from_numpy(self.labels)
        if classes is not None:
            idx = torch.zeros_like(self.labels, dtype=torch.bool)
            for i, cls in enumerate(classes):
                ix = self.labels == cls
                idx |= ix
            self.images = self.images[idx]
            self.labels = self.labels[idx]
            self.count_classes()
            idxs = []
            for cls in classes:
                idxs.append(self.labels == cls)
            for i, idx in enumerate(idxs):
                self.labels[idx] = i
            assert len(self.labels) == len(self.images)
            # self.images = self.images[self.la]

    def __getitem__(self, i: int) -> Tuple[ndarray, str]:
        # return self.images[i].float() / 255.0, self.labels[i]
        return self.images[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.images)

    def count_classes(self) -> None:
        uniqs, count = np.unique(self.labels.numpy(), return_counts=True)
        for unq, cnt in zip(uniqs, count):
            print(f"class {unq} ({OBJECT_CATEGORIES[unq]}): {cnt}")


class Segmentations(Dataset):
    def __init__(
        self, size: Tuple[int, int] = (224, 224), max_images: int = 20
    ) -> None:
        self.samples: List[Tuple[str, ndarray, ndarray]] = []
        self.size = (224, 224)
        all_segs = sorted(
            SEGS.rglob("*.png")
        )  # these are the segmentation masks with class labels
        np.random.shuffle(all_segs)  # type: ignore
        for seg in all_segs:
            mask = np.asarray(
                Image.open(seg), dtype=np.uint8
            ).copy()  # labels are in [0, 255]
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html for "void" info
            mask[
                mask == 255
            ] = 0  # remove "void" class, which is 255 according to PASCAL documentation
            labels = np.unique(mask, return_counts=True)[0]

            if len(labels) > 2:  # multiple classes
                continue
            img = np.asarray(
                Image.open(IMGS / f"{seg.stem}.jpg")
            )  # get image from mask name
            img = cv2.resize(img, self.size)

            seg_img = np.asarray(Image.open(seg))
            seg_img = cv2.resize(seg_img, self.size)
            category = LABELS[labels[1]]
            self.samples.append((category, img, seg_img))
            if len(self.samples) >= max_images:
                break

    def __getitem__(self, index: int) -> Tuple[str, ndarray, ndarray]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


# This method loads all the segmentations of the specified classes
class SegmentationsforSameClass(Dataset):
    def __init__(
        self, size: Tuple[int, int] = (224, 224), classes=[1, 4, 14, 19]
    ) -> None:
        self.samples: List[Tuple[str, ndarray, ndarray]] = []
        self.size = (224, 224)
        all_segs = sorted(
            SEGS.rglob("*.png")
        )  # these are the segmentation masks with class labels
        np.random.shuffle(all_segs)  # type: ignore
        for seg in all_segs:
            mask = np.asarray(
                Image.open(seg), dtype=np.uint8
            ).copy()  # labels are in [0, 255]
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html for "void" info
            mask[
                mask == 255
            ] = 0  # remove "void" class, which is 255 according to PASCAL documentation
            labels = np.unique(mask, return_counts=True)[0]

            if len(labels) > 2:  # multiple classes
                continue
            img = np.asarray(
                Image.open(IMGS / f"{seg.stem}.jpg")
            )  # get image from mask name
            img = cv2.resize(img, self.size)

            seg_img = np.asarray(Image.open(seg))
            seg_img = cv2.resize(seg_img, self.size)
            # category = LABELS[labels[1]]
            for id, cls in enumerate(classes):
                if classes[id] == labels[1]:
                    category = LABELS[labels[1]]
                    self.samples.append((category, img, seg_img))

            # if len(self.samples) >= 50:
            #    break

    def __getitem__(self, index: int) -> Tuple[str, ndarray, ndarray]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


# This method loads all the segmentation masks irrespective of multi-class or not
class SegmentationsWithoutAmbiguity(Dataset):
    def __init__(
        self,
        size: tuple = (224, 224),
        multiclass=False,
    ):
        self.samples = []
        self.size = size
        all_segs = sorted(
            SEGS.rglob("*.png")
        )  # these are the segmentation masks with class labels
        np.random.shuffle(all_segs)  # type: ignore
        for seg in all_segs:
            mask = np.asarray(
                Image.open(seg), dtype=np.uint8
            ).copy()  # labels are in [0, 255]
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html for "void" info
            mask[
                mask == 255
            ] = 0  # remove "void" class, which is 255 according to PASCAL documentation
            labels, counts = np.unique(mask, return_counts=True)
            if not multiclass:
                if len(labels) > 2:  # multiple classes
                    continue
                seg_path = SEGS / f"{seg.stem}.jpg"
                category = labels[1]
            else:
                if len(labels) < 3:  # want more than just one class and background
                    continue
                seg_path = SEGS / f"{seg.stem}.jpg"

                # background (0) usually largest, exclude it
                labels, counts = labels[1:], counts[1:]
                idx = np.argsort(counts)
                category = labels[idx[0]]
            self.samples.append((category, seg_path))


# This method loads all the samples irrespective of multi-class or not
class DatasetWithoutAmbiguity(Dataset):
    def __init__(
        self,
        size: tuple = (224, 224),
        multiclass=False,
    ):
        self.samples = []
        self.size = size
        all_segs = sorted(
            SEGS.rglob("*.png")
        )  # these are the segmentation masks with class labels
        np.random.shuffle(all_segs)  # type: ignore
        for seg in all_segs:
            mask = np.asarray(
                Image.open(seg), dtype=np.uint8
            ).copy()  # labels are in [0, 255]
            # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html for "void" info
            mask[
                mask == 255
            ] = 0  # remove "void" class, which is 255 according to PASCAL documentation
            labels, counts = np.unique(mask, return_counts=True)
            if not multiclass:
                if len(labels) > 2:  # multiple classes
                    continue
                img_path = IMGS / f"{seg.stem}.png"
                category = labels[1]
            else:
                if len(labels) < 3:  # want more than just one class and background
                    continue
                img_path = IMGS / f"{seg.stem}.jpg"

                # background (0) usually largest, exclude it
                labels, counts = labels[1:], counts[1:]
                idx = np.argsort(counts)
                category = labels[idx[0]]
            self.samples.append((category, img_path))

    def __getitem__(self, index):
        category_id, image_path = self.samples[index]
        image = np.asarray(Image.open(image_path))
        image = cv2.resize(image, self.size)
        label = torch.LongTensor([category_id])
        return image, label

    def __len__(self) -> int:
        return len(self.samples)
