from icevision.all import *
import icedata

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from fastai.vision.all import *
from fastai.callback.wandb import *

import wandb

from icevision_detector import *


path: Param("Training dataset path", str) = Path.home()/'Datasets/image/deepfashion/'
bs: Param("Batch size", int) = 6
log: Param("Log to wandb", bool) = True
num_workers: Param("Number of workers to use", int) = 4
resume: Param("Link to pretrained model", str) = None
name: Param('experiment name', str) = 'd3'


def deepfashion_dataset(
        root_dir:Union[Path, str],
        mask: bool = False,
        autofix: bool = True,
        cache_records: bool = True
) -> Tuple[tuple, ClassMap]:

    if isinstance(root_dir, str): root_dir = Path(root_dir)
    coco_train = icedata.coco.parser(
        img_dir=root_dir / 'train',
        annotations_file=root_dir / 'train/deepfashion2.json',
        mask=mask)

    coco_valid = icedata.coco.parser(
        img_dir=root_dir / 'validation',
        annotations_file=root_dir / 'validation/deepfashion2.json',
        mask=mask)

    train_records, *_ = coco_train.parse(data_splitter=SingleSplitSplitter(), autofix=autofix,
                                         cache_filepath=root_dir / 'train_cache' if cache_records else None)
    valid_records, *_ = coco_valid.parse(data_splitter=SingleSplitSplitter(), autofix=autofix,
                                         cache_filepath=root_dir / 'valid_cache' if cache_records else None)

    assert(coco_train.class_map==coco_valid.class_map), f"ClassMap for train and valid differ: {coco_train.class_map=}!={coco_valid.class_map=}"
    return (train_records, valid_records), coco_train.class_map


(train_records, valid_records), class_map = deepfashion_dataset(path, autofix=True)


size = 512
num_classes = len(class_map)
aug_tfms = tfms.A.aug_tfms(
    size=size,
    shift_scale_rotate=tfms.A.ShiftScaleRotate(rotate_limit=(-15, 15)),
    pad=partial(tfms.A.PadIfNeeded, border_mode=0)
)
aug_tfms.append(tfms.A.Normalize())

train_tfms = tfms.A.Adapter(aug_tfms)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

metrics = [COCOMetric(print_summary=True)]


train_dl = efficientdet.train_dl(train_ds, batch_size=bs, num_workers=num_workers, shuffle=True)
valid_dl = efficientdet.valid_dl(valid_ds, batch_size=bs, num_workers=num_workers, shuffle=False)

batch, samples = first(train_dl)
show_samples(samples[:6], class_map=class_map, ncols=3)


checkpoint_callback = ModelCheckpoint(
    verbose=True,
    monitor='COCOMetric/AP (IoU=0.50:0.95) area=all/dl_idx_0',
    mode='max')

lr_monitor = LearningRateMonitor(log_momentum=True)
wandb_logger = WandbLogger(project='deepfashion', offline=not log, name=name)


# In[10]:


# light_model = EffDetModel(
#     num_classes=num_classes, img_size=size, model_name='tf_efficientdet_d3',
#     lr=0.05, warmup_epochs=0,
# )
# light_model.freeze_to_head()
light_model = EffDetModel.load_from_checkpoint(checkpoint_path='checkpoints/d3_frozen.ckpt', num_classes=num_classes,
                                               img_size=size, model_name="tf_efficientdet_d3",
                                               lr=0.05, warmup_epochs=1)

# In[11]:


trainer = pl.Trainer(
    # num_sanity_val_steps=100,  # 1000 is enough to validate on all COCO @bs=8
    limit_train_batches=1000,
    limit_val_batches=100,
    gpus=1,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoint_callback],
    max_epochs=100,
    sync_batchnorm=True,
    weights_summary='full',
    amp_level='O2', precision=16,
)

trainer.fit(light_model, train_dl, valid_dl)





