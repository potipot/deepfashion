from icevision.all import *
import icedata

from omegaconf import OmegaConf

from icevision_detector import *

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint



params = OmegaConf.create({
    'project': 'deepfashion',
    'run_name': 'presentation',
    'img_size': 512,
    'num_classes': 14,
    'bs': 4,
    'num_workers': 4,
    'path': './datasets',
    'model_name': 'tf_efficientdet_d1'
})


(train_records, valid_records), class_map = deepfashion_dataset(params.path, autofix=True)



aug_tfms = tfms.A.aug_tfms(
    size=params.img_size,
    shift_scale_rotate=tfms.A.ShiftScaleRotate(rotate_limit=(-15, 15)),
)
aug_tfms.append(tfms.A.Normalize())


train_tfms = tfms.A.Adapter(aug_tfms)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(params.img_size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

train_dl = efficientdet.train_dl(train_ds, batch_size=params.bs, num_workers=params.num_workers, shuffle=True)
valid_dl = efficientdet.valid_dl(valid_ds, batch_size=params.bs, num_workers=params.num_workers, shuffle=False)



light_model = EffDetModel(
    num_classes=params.num_classes, img_size=params.img_size, model_name=params.model_name,
    lr=0.12, warmup_epochs=2)



wandb_logger = WandbLogger(project=params.project, name=params.run_name)
lr_monitor = LearningRateMonitor(log_momentum=True)

checkpoint_callback = ModelCheckpoint(
    verbose=True,
    monitor='COCOMetric/AP (IoU=0.50) area=all/dl_idx_0',
    mode='max'
)


trainer = pl.Trainer(
    gpus=1,
    max_epochs=3,
    sync_batchnorm=True,  # from Ross'es training config
    weights_summary='top', # print top-level modules summary
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoint_callback],
    amp_level='O2',  # mixed precision
    precision=16,
)

light_model = EffDetModel(
    num_classes=params.num_classes, img_size=params.img_size, model_name=params.model_name,
    lr=0.12, warmup_epochs=2)


trainer.fit(light_model, train_dl, valid_dl)


# trainer = pl.Trainer(
#     gpus=2,
#     accelerator='ddp',
#     logger=wandb_logger,
#     callbacks=[lr_monitor, checkpoint_callback],
#     max_epochs=300, # caveat
#     sync_batchnorm=True,  # from Ross'es training config
#     weights_summary='full',
#     amp_level='O2',
#     precision=16,
# )
#
#
#
# trainer.fit(light_model, train_dl, valid_dl)





