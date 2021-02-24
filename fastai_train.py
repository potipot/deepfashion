import os
from omegaconf import OmegaConf

from icevision.all import *
import icedata

from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from fastai.distributed import *
from fastai.learner import AvgLoss
from fastai.torch_core import find_bs

from icevision_detector import *


@patch
def accumulate(self:AvgLoss, learn):
    #bs = find_bs(learn.yb)
    bs = find_bs(learn.xb)
    self.total += learn.to_detach(learn.loss.mean())*bs
    self.count += bs


@patch
def create_batch(self:DataLoader, b):
    return efficientdet.dataloaders.build_train_batch(b)


@call_parse
def main(
        path: Param("Training dataset path", str) = './datasets',
        bs: Param("Batch size", int) = 4,
        model_name: Param("Architecture backbone", str) = 'tf_efficientdet_d1',
        img_size: Param("Image size", int) = 512,
        num_workers: Param("Number of workers to use", int) = 4,
        num_classes: Param("Number of classes in dataset", int) = 14,
        project: Param("name of the wandb project", str) = 'deepfashion-presentation',
        run_name: Param("wandb run name", str) = 'presentation',
):
    (train_records, valid_records), class_map = deepfashion_dataset(path, autofix=True)

    aug_tfms = tfms.A.aug_tfms(
        size=img_size,
        shift_scale_rotate=tfms.A.ShiftScaleRotate(rotate_limit=(-15, 15)),
    )
    aug_tfms.append(tfms.A.Normalize())


    train_tfms = tfms.A.Adapter(aug_tfms)
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    train_dl = efficientdet.train_dl(train_ds, batch_size=bs, num_workers=num_workers, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=bs, num_workers=num_workers, shuffle=False)

    model = efficientdet.model(model_name=model_name, num_classes=num_classes, img_size=img_size)
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    learn = efficientdet.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

    rank = os.environ.copy().get('RANK', 0)
    if rank == '0':
        wandb.init(project=project, name=run_name, mode='offline')
        cbs = [WandbCallback()]
    else: cbs = []

    with learn.distrib_ctx():
        learn.fit(100, 1e-4, cbs=cbs)







