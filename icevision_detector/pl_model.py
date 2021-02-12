from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch import nn
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from icevision import COCOMetric, ClassMap, Metric
from icevision.models import efficientdet
from .metrics import SimpleConfusionMatrix

__all__ = ['BaseModel', 'EffDetModel']


@dataclass
class TimmConfig:
    opt:str = 'fusedmomentum'
    weight_decay:float = 4e-5
    lr: float = 0.01
    momentum: float = 0.9

    opt_eps: float = 1e-3
    # opt_betas
    # opt_args

    epochs: int = 300
    lr_noise: tuple = (0.4, 0.9)
    sched: str = 'cosine'
    min_lr: float = 1e-5
    decay_rate: float = 0.1
    warmup_lr: float = 1e-4
    warmup_epochs: int = 5
    cooldown_epochs: int = 10

    lr_cycle_limit:int = 1
    lr_cycle_mul: float = 1.0
    lr_noise_pct: float = 0.67
    lr_noise_std: float = 1.0
    seed: int = 42


class BaseModel(efficientdet.lightning.ModelAdapter):
    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super(BaseModel, self).__init__(model=model, metrics=metrics)
        self.n_train_dls, self.n_test_dls, self.n_valid_dls = None, None, None

    def configure_optimizers(self):
        optimizer = create_optimizer(self.timm_config, self.model)
        lr_scheduler, num_epochs = create_scheduler(self.timm_config, optimizer)
        return [optimizer], [lr_scheduler]

    def setup(self, stage_name):
        def _get_num_of_dataloaders(dataloader):
            n: int
            dl = getattr(self, dataloader)
            assert(callable(dl))
            result = dl()
            if isinstance(result, list): n = len(self.val_dataloader.dataloader)
            elif isinstance(result, torch.utils.data.dataloader.DataLoader): n = 1
            elif result is None: n = 0
            else: raise RuntimeError
            return n
        self.n_train_dls = _get_num_of_dataloaders('train_dataloader')
        self.n_valid_dls = _get_num_of_dataloaders('val_dataloader')
        self.n_test_dls = _get_num_of_dataloaders('test_dataloader')

    def freeze_to_head(self, train_class_head=True, train_bbox_head=False):
        """
        Freezes the model up to the head part.
        Parameters control whether to train labels classifier and bbox regressor.
        """
        self.freeze()
        for param in self.model.model.box_net.parameters():
             param.requires_grad = train_bbox_head
        for param in self.model.model.class_net.parameters():
            param.requires_grad = train_class_head
        self.train()


class EffDetModel(BaseModel):
    def __init__(self, num_classes: int, img_size: int, model_name: Optional[str] = "tf_efficientdet_lite0", **timm_args):
        model = efficientdet.model(model_name=model_name, num_classes=num_classes, img_size=img_size, pretrained=True)
        # TODO: change this once pl-mAP is merged: https://github.com/PyTorchLightning/pytorch-lightning/pull/4564
        metrics = [COCOMetric(print_summary=True), SimpleConfusionMatrix()]
        self.timm_config = TimmConfig(**timm_args)
        super().__init__(model=model, metrics=metrics)

    def validation_step(self, batch, batch_idx, dataset_idx: int = 0):
        # execute validation on batch
        (xb, yb), records = batch

        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(raw_preds["detections"], 0)
            val_losses = {f'val_{key}': value for key, value in raw_preds.items() if 'loss' in key}
            loss = efficientdet.loss_fn(raw_preds, yb)

        for metric in self.metrics[dataset_idx]:
            metric.accumulate(records=records, preds=preds)

        # logging losses in step
        self.log_dict(val_losses)

    def validation_epoch_end(self, epoch_output):
        # deprecated?? trainer.evaluation_loop.py @ 210 - for pl factory Metrics in self.evaluation_callback_metrics
        # epoch_output is a list of step_outputs per dataloader shape: [0..n_dls, 0..n_batches]
        for dataset_idx in range(len(self.metrics)):
            self.finalize_metrics(dataset_idx)

        # the len of this must be kept as len of dataloaders, otherwise pl ignores idx
        # return epoch_output

    def test_step(self, batch, batch_idx, dataset_idx=0):
        return self.validation_step(batch, batch_idx, dataset_idx)

    def test_epoch_end(self, *args, **kwargs):
        self.validation_epoch_end(*args, **kwargs)

    def setup(self, stage_name):
        super(EffDetModel, self).setup(stage_name=stage_name)

        # create separate metrics for each dataloader
        self.metrics = [
            [deepcopy(metric) for metric in self.metrics]
            for _ in range(self.n_valid_dls if stage_name == 'train' else self.n_test_dls)
        ]
        # self.pl_metrics = nn.ModuleList(
        #     [nn.ModuleList([deepcopy(pl_metric) for pl_metric in self.pl_metrics])
        #      for _ in range(self.n_valid_dls)]
        # )

    def finalize_metrics(self, dataset_idx: int = 0) -> None:
        for metric in self.metrics[dataset_idx]:
            metric_logs = metric.finalize()
            log = getattr(metric, 'log', None)
            if callable(log):
                log(self.logger)
            else:
                for k, v in metric_logs.items():
                    # TODO: metric logging with forced dataset_idx
                    self.log(f"{metric.name}/{k}/dl_idx_{dataset_idx}", v)
