import os

os.environ["KERAS_BACKEND"] = "torch"
from functools import partial
from pathlib import Path

import keras
import torch
from keras.layers import TorchModuleWrapper
from keras.src.backend.torch.core import get_device
from keras.src.trainers.data_adapters import data_adapter_utils
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
)
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
    EnsureTyped,
    Lambda,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)


def get_btcv_data_loaders(
    patch_samples_per_image=4,
    patch_size=(96, 96, 96),
    raw_data_dir=Path("RawData/Training"),
):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=float("nan")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            EnsureTyped(keys=["image", "label"], device=get_device(), track_meta=False),
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
            Lambda(lambda d: (d["image"], d["label"])),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=float("nan")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=get_device(), track_meta=True),
            Lambda(lambda d: (d["image"], d["label"])),
        ]
    )
    image_root = raw_data_dir / "img"
    label_root = raw_data_dir / "label"

    files = [
        {"image": i.as_posix(), "label": (label_root / i.name.replace("img", "label")).as_posix()}
        for i in image_root.glob("*.nii.gz")
    ]

    val_names = [
        "img0035.nii.gz",
        "img0036.nii.gz",
        "img0037.nii.gz",
        "img0038.nii.gz",
        "img0039.nii.gz",
        "img0040.nii.gz",
    ]

    val_files = [i for i in files if i["image"].split("/")[-1] in val_names]
    train_files = [i for i in files if i["image"].split("/")[-1] not in val_names]
    assert len(val_files) + len(train_files) == len(files)

    # super specific to BTCV small set. For any other data we cannot cache.
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
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
        y_pred = sliding_window_inference(
            x.to(get_device()), self.patch_size, self.sliding_window_batch_size, partial(self, training=False)
        )
        return self.compute_metrics(x, y, y_pred, sample_weight)


def get_model(patch_size=(96, 96, 96)):
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=True,
    )
    inputs = keras.layers.Input(shape=(1, *patch_size))
    x = CustomTorchWrapper(model)(inputs)
    k_model = SlidingWindowValidationModel(patch_size, 4, inputs, x)
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

    def reset_states(self):
        self.m.reset()

    def update_state(self, y_true, y_preds, sample_weight=None):
        y_true_list = decollate_batch(y_true)
        y_true_convert = [self.post_label(val_label_tensor) for val_label_tensor in y_true_list]
        y_preds_list = decollate_batch(y_preds)
        y_preds_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in y_preds_list]

        self.m(y_pred=y_preds_convert, y=y_true_convert)

    def result(self):
        return self.m.aggregate().item()


if __name__ == "__main__":
    train_dl, val_dl = get_btcv_data_loaders()
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=DiceCELossKeras(to_onehot_y=True, softmax=True),
        metrics=[MonaiDiceMetricKeras()],
        run_eagerly=False,
    )
    model.fit(
        train_dl,
        validation_data=val_dl,
        epochs=1000,
        validation_freq=10,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "model.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor="val_monai_dice",
                mode="max",
            )
        ],
    )
