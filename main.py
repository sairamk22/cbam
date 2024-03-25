import random
import sys
from argparse import ArgumentParser
from math import sqrt
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy import signal
from scipy.stats import pearsonr
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score, mean_squared_error
from torch.functional import norm
from torch.utils.data import DataLoader

from loader import MainDataset, SegmentationsforSameClass
from model import CbamMaps

# from ResNet import ResidualNet
# This part of the code is implemented based on the official Lightning Module's github repository example https://rb.gy/s78ahn
# Code to Plot the histogram distribution of three major metrics iou,mae and correlation
# https://www.machinelearningplus.com/plots/matplotlib-histogram-python-examples/
def histogram_of_metrics(iou_scores, mae_scores, corr_scores):
    # Normalize plot
    kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)
    # Plot
    plt.hist(iou_scores, **kwargs, color="g", label="IoU")
    plt.hist(mae_scores, **kwargs, color="b", label="MAE")
    plt.hist(corr_scores, **kwargs, color="r", label="Correlation")
    plt.gca().set(title=" Histogram of Metrics")
    plt.xlim(-1, 1)
    plt.legend()


def visualize(model: LightningModule, layer=0):
    PREDICTIONS_LABELS = {
        0: "aeroplane",
        1: "boat",
        2: "motorbike",
        3: "train",
    }
    model.to("cpu")
    with torch.no_grad():
        feature_map_idx = 0
        iou_scores = []
        mae_scores = []
        dice_scores = []
        corr_scores = []
        ssim_scores = []
        segs_dataset = SegmentationsforSameClass()
        #print("Number of Segmentations", len(segs_dataset))
        segs_dataset = torch.utils.data.DataLoader(
            segs_dataset, batch_size=1, num_workers=4
        )
        model.to("cpu")
        feature_map_idx = layer
        index = 0
        for batch in segs_dataset:
            category, images_np, seg = batch
            images = images_np.permute(0, 3, 1, 2).float()
            output, feature_maps, channel_attn_maps, spatial_attn_maps = model.forward(
                images.to("cpu")
            )
            # layout our images in a grid, with title info to say what we are plotting
            image_ids = np.arange(len(output))
            # image_ids=len(output)
            fig, axes = plt.subplots(
                nrows=len(image_ids), ncols=3
            )  # 20 categories total
            samples = []
            for image_idx in image_ids:
                predicted_class = np.argmax((output.detach().numpy())[image_idx])
                prediction = PREDICTIONS_LABELS[int(predicted_class)]
                feature_maps_spatial_attn = (
                    spatial_attn_maps[feature_map_idx][image_idx]
                    * feature_maps[feature_map_idx][image_idx]
                )
                all_maps_attn = (
                    channel_attn_maps[feature_map_idx][image_idx]
                    .unsqueeze(1)
                    .unsqueeze(2)
                    * spatial_attn_maps[feature_map_idx][image_idx]
                )
                fmap_attn_combined = (
                    all_maps_attn * feature_maps[feature_map_idx][image_idx]
                )
                spatial_maps = (
                    spatial_attn_maps[feature_map_idx][image_idx].detach().numpy()
                )
                # interpolating attention maps for deeper attention layers
                if feature_map_idx >= 1:
                    all_maps_attn = all_maps_attn.unsqueeze(0)
                    all_maps_attn = torch.nn.functional.interpolate(
                        all_maps_attn, size=(224, 224), mode="bilinear"
                    )
                    all_maps_attn = all_maps_attn.squeeze(0)
                seg_a = seg[image_idx].numpy()
                combined_maps = all_maps_attn[feature_map_idx].detach()
                data_min = np.min(seg_a, keepdims=True)
                data_max = np.max(seg_a, keepdims=True)
                # Normalizing the values to set them on  scale of 0-1
                a = (seg_a - data_min) / (data_max - data_min)
                b = all_maps_attn[feature_map_idx].detach().numpy()
                # The code replaces combined attention maps with the spatial attention map
                # b=spatial_maps.squeeze()
                print(index)
                print("Prediction :", prediction, " , Original :", category[image_idx])
                # IoU Score Median threshold
                threshold_a = np.median(a)
                threshold_b = np.median(b)
                # Referred from https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
                seg_a = np.where(a > threshold_a, 255, 0)
                seg_b = np.where(b > threshold_b, 1, 0)
                intersection = numpy.logical_and(seg_a, seg_b)
                union = numpy.logical_or(seg_a, seg_b)
                iou_score = numpy.sum(intersection) / numpy.sum(union)
                # print("IoU Score: %.3f"% iou_score)
                iou_scores.append(iou_score)
                ax = (0, 1)  # W,H axes of each image
                intersection_sum = numpy.sum(intersection)
                mask_sum = numpy.sum(seg_a) + np.sum(seg_b)
                if int(mask_sum) == 0:
                    dice = 1.0
                else:
                    dice = 2.0 * intersection_sum / mask_sum
                # print("Dice: %.3f"% dice)
                dice_scores.append(dice)
                mae = np.mean(np.absolute(a - b))
                # print("mae: %.3f"% mae)
                mae_scores.append(mae)
                # SSIM
                ssim_const = ssim(a, b, data_range=a.max() - a.min())
                # print("ssim: %.3f"% ssim_const)
                ssim_scores.append(ssim_const)
                # Correlation
                corr, _ = pearsonr(a.flat, b.flat)
                corr_scores.append(corr)
                # print('Pearsons correlation: %.3f' % corr)
                samples.extend(
                    [
                        # ("Original",images_np[image_idx]),
                        # Code to fix the segmentations that looked like outline
                        (
                            f"{index}" + "  Segmentation Map",
                            seg[image_idx].float().numpy() * 255,
                        ),
                        ("Original", images_np[image_idx]),
                        # ("Feature Map",feature_maps[feature_map_idx][image_idx][feature_map_idx].detach().numpy()),
                        (
                            "Channel and Spatial Map",
                            all_maps_attn[feature_map_idx].detach().numpy().squeeze(),
                        ),
                        # ("Spatial Map",spatial_maps.squeeze()),
                        # ("channel and Spatial Map * fmap",fmap_attn_combined[feature_map_idx].detach().numpy())
                    ]
                )
                print(
                    f"IoU: {iou_scores[index]:.3f}, Dice: {dice_scores[index]:.3f},MAE: {mae_scores[index]:.3f},Pearson Correlation: {corr_scores[index]:.3f},SSIM: {ssim_scores[index]:.3f}"
                )
                index += 1
            ax: plt.Axes
            for (category, img), ax in zip(samples, axes.flat):
                if "Segmentation" in category:
                    ax.imshow(img, vmin=0, vmax=255)
                else:
                    ax.imshow(img)
                ax.set_title(category, fontsize=16)
                # ax = axes.ravel()
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            fig.set_size_inches(w=40, h=40)
            fig.tight_layout()
            plt.show()
        #   if(index>=140):
        #     break
    histogram_of_metrics(iou_scores, mae_scores, corr_scores)
    # break


def train() -> Tuple[LightningModule, Trainer]:
    source = "Main"
    CLASSES = [0, 3, 13, 18]
    # dataset = DatasetWithoutAmbiguity(multiclass=False)
    dataset = MainDataset("train", classes=CLASSES)  # aeroplane, boat, motorbike, train
    """
    class 0 (background): 327
    class 1 (aeroplane): 268
    class 2 (bicycle): 395
    class 3 (bird): 260
    class 4 (boat): 365
    class 5 (bottle): 213
    class 6 (bus): 590
    class 7 (car): 539
    class 8 (cat): 566
    class 9 (chair): 151
    class 10 (cow): 269
    class 11 (diningtable): 632
    class 12 (dog): 237
    class 13 (horse): 265
    class 14 (motorbike): 1994
    class 15 (person): 269
    class 16 (potted plant): 171
    class 17 (sheep): 257
    class 18 (sofa): 273
    class 19 (train): 290
    """
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.manual_seed(4)
    )
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=True, batch_size=32, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, num_workers=4, shuffle=False, batch_size=32, drop_last=True
    )
    print("No. of Train Samples %d" % len(train_dataset))
    print("No. of Val Samples %d" % len(val_dataset))
    model = CbamMaps()
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    # Code to enable checkpoint callback
    # checkpoint_callback = LightningModule.load_from_checkpoint("/content/gdrive/MyDrive/CBAM_Final_Model_checkpoints/lightning_logs/version_52/checkpoints/epoch=1-step=56.ckpt")
    # checkpoint_callback = ModelCheckpoint(
    # save_top_k=1,
    # verbose=True,
    # mode='max',
    # )
    args = parser.parse_args()
    # Manually loading a checkpoint
    # checkpoint = torch.load(
    #     "/content/gdrive/MyDrive/CBAM_Final_Model_checkpoints/lightning_logs/version_95/checkpoints/epoch=49-step=1400.ckpt"
    # )
    # model.load_state_dict(checkpoint['state_dict'],strict=False)
    # Uncomment this line and comment line 78  to run on colab
    # args, unknown = parser.parse_known_args()
    trainer: Trainer = Trainer.from_argparse_args(
        args,
        gpus=1,
        enable_checkpointing=True,
        # callbacks=[checkpoint_callback],
        # limit_train_batches=1.0,
        # resume_from_checkpoint='/content/gdrive/MyDrive/CBAM_Final_Model_checkpoints/lightning_logs/version_59/checkpoints/epoch=55-step=1568.ckpt',
        min_epochs=1,
        max_epochs=1,
        auto_lr_find=True,
        default_root_dir="/content/gdrive/MyDrive/CBAM_Final_Model_checkpoints",
    )
    # Code to find a learning rate
    # results = trainer.tuner.lr_find(model,train_loader)
    # results.plot(suggest=True, show=True)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(ckpt_path="best", dataloaders=val_loader)
    # results=trainer.predict(model,dataloaders=val_loader)
    # Code to pass the layer number in visualize method there are 8 layers indexed from 0-7
    layer = 0
    # Visualizes the attention maps,segmentation maps and original maps in a grid and prints their individual correlation scores
    visualize(model, layer)
    return model, trainer


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    train()
