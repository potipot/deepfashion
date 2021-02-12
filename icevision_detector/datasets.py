import icedata
from icevision.all import *


def coco_dataset(
        coco_root:Union[Path, str],
        mask: bool = False,
        autofix: bool = True,
        cache_records: bool = True
) -> Tuple[tuple, ClassMap]:

    if isinstance(coco_root, str): coco_root = Path(coco_root)
    coco_train = icedata.coco.parser(
        img_dir=coco_root / 'train2017',
        annotations_file=coco_root / 'annotations/instances_train2017.json',
        mask=mask)

    coco_valid = icedata.coco.parser(
        img_dir=coco_root / 'val2017',
        annotations_file=coco_root / 'annotations/instances_val2017.json',
        mask=mask)

    train_records, *_ = coco_train.parse(data_splitter=SingleSplitSplitter(), autofix=autofix,
                                         cache_filepath=coco_root / 'train_cache' if cache_records else None)
    valid_records, *_ = coco_valid.parse(data_splitter=SingleSplitSplitter(), autofix=autofix,
                                         cache_filepath=coco_root / 'valid_cache' if cache_records else None)

    assert(coco_train.class_map==coco_valid.class_map), \
        f"ClassMap for train and valid differ: {coco_train.class_map=}!={coco_valid.class_map=}"
    return (train_records, valid_records), coco_train.class_map
