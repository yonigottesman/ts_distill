import argparse
import os

from tqdm.auto import tqdm

import wandb

os.environ["KERAS_BACKEND"] = "torch"
from functools import partial
from pathlib import Path

import keras
import torch
from keras.src.backend.torch.core import get_device
from keras.src.trainers.data_adapters import data_adapter_utils
from monai.data import decollate_batch
from monai.data.utils import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)


class BTCVDataset(torch.utils.data.Dataset):
    # BTCV is a small dataset so I load it to memory and cache the deterministic transforms results.
    # For larger datasets process the data and save to disk.

    def __init__(self, data, deterministic_transform, transform):
        self.transform = transform
        self.data = [deterministic_transform(d) for d in tqdm(data, desc="Loading data")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def collate_fn(batch):
    batch = list_data_collate(batch)
    return batch["image"], batch["label"]


def get_btcv_data_loaders(
    raw_data_dir,
    train_labels_dir,
    validation_labels_dir,
    patch_samples_per_image=4,
    patch_size=(96, 96, 96),
):
    deterministic_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=float("nan")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    train_transforms = Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=patch_samples_per_image,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ]
    )

    val_names = [
        "img0035.nii.gz",
        "img0036.nii.gz",
        "img0037.nii.gz",
        "img0038.nii.gz",
        "img0039.nii.gz",
        "img0040.nii.gz",
    ]

    val_files = [
        {"image": i.as_posix(), "label": (validation_labels_dir / i.name.replace("img", "label")).as_posix()}
        for i in raw_data_dir.glob("*.nii.gz")
        if i.name in val_names
    ]
    train_files = [
        {"image": i.as_posix(), "label": (train_labels_dir / i.name.replace("img", "label")).as_posix()}
        for i in raw_data_dir.glob("*.nii.gz")
        if i.name not in val_names
    ]

    ds_train = BTCVDataset(train_files, deterministic_transform, train_transforms)
    ds_val = BTCVDataset(val_files, deterministic_transform, Compose([]))

    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


class CustomTorchWrapper(keras.layers.TorchModuleWrapper):
    def compute_output_spec(self, inputs_spec, ouptut_channels=14):
        h, w, d = inputs_spec.shape[2:]  # Get spatial dimensions from input
        output_shape = (None, ouptut_channels, h, w, d)
        return keras.KerasTensor(shape=output_shape, dtype="float32")


class SlidingWindowValidationModel(keras.models.Model):
    def __init__(self, patch_size, sliding_window_batch_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.sliding_window_batch_size = sliding_window_batch_size

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        x, y = x.to(get_device()), y.to(get_device())
        y_pred = sliding_window_inference(
            x, self.patch_size, self.sliding_window_batch_size, partial(self, training=False)
        )
        return self.compute_metrics(x, y, y_pred, sample_weight)


def get_model(patch_size=(96, 96, 96), pretrained_weights=None):
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=True,
    )
    if pretrained_weights:
        model.load_from(torch.load(pretrained_weights))
        print("Using pretrained self-supervied Swin UNETR backbone weights !")
    inputs = keras.layers.Input(shape=(1, *patch_size))
    x = CustomTorchWrapper(model)(inputs)
    k_model = SlidingWindowValidationModel(patch_size, 4, inputs, x)
    # bug in TorchModuleWrapper so i have to do this manually https://github.com/keras-team/keras/issues/20726
    k_model = k_model.half()
    return k_model


class DiceCELossKeras(torch.nn.Module):
    def __init__(self, to_onehot_y, softmax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax)

    def forward(self, y_true, y_pred):
        return self.l(y_pred, y_true)


class MonaiDiceMetricKeras(keras.metrics.Metric):
    def __init__(self, include_background=True, reduction="mean", get_not_nans=False, *args, **kwargs):
        super().__init__(name="monai_dice", *args, **kwargs)
        self.m = DiceMetric(include_background=include_background, reduction=reduction, get_not_nans=get_not_nans)

        self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=14)

    def reset_state(self):
        self.m.reset()

    def update_state(self, y_true, y_preds, sample_weight=None):
        if torch.is_grad_enabled() or y_true.device == torch.device("meta"):
            # dont compute for train set or keras build stage
            return
        y_true_list = decollate_batch(y_true)
        y_true_convert = [self.post_label(val_label_tensor) for val_label_tensor in y_true_list]
        y_preds_list = decollate_batch(y_preds)
        y_preds_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in y_preds_list]

        self.m(y_pred=y_preds_convert, y=y_true_convert)

    def result(self):
        if self.m.get_buffer() is None:
            return 0.0
        return self.m.aggregate().item()


class CustomWandbLogger(keras.callbacks.Callback):
    def __init__(self, log_batch=False):
        super().__init__()
        self.log_batch = log_batch

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        wandb.log({"epoch": epoch}, commit=False)
        wandb.log(logs, commit=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root directory of the data.")
    parser.add_argument("--validation_labels", type=str, required=True, help="Directory of validation labels.")
    parser.add_argument("--train_labels", required=True, type=str, help="Directory of training labels.")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="name of run in wandb")
    parser.add_argument("--pretrained_weights", type=str, help="Path to the pretrained weights .pt file")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    wandb.init(project="ts_distill", name=args.wandb_run_name)

    train_dl, val_dl = get_btcv_data_loaders(
        Path(args.data_dir),
        train_labels_dir=Path(args.train_labels),
        validation_labels_dir=Path(args.validation_labels),
    )
    model = get_model(pretrained_weights=args.pretrained_weights)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss=DiceCELossKeras(to_onehot_y=True, softmax=True),
        metrics=[MonaiDiceMetricKeras()],
        run_eagerly=False,
    )

    model.fit(
        train_dl,
        validation_data=val_dl,
        epochs=1500,
        validation_freq=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "model.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor="val_monai_dice",
                mode="max",
            ),
            CustomWandbLogger(),
        ],
    )
