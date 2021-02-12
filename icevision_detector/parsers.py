from icevision.all import *
from fastai.vision.all import Pipeline, Image

__all__ = ['TxtParser', 'test_pipeline']


def get_txt_annotations(txt_file_path):
    with open(txt_file_path) as text_file:
        data = text_file.readlines()

    bboxes, labels = [], []
    for line in data:
        bbox, label = line.strip().split('|')
        bbox = [float(px) for px in bbox.split(',')]
        bboxes.append(bbox)
        labels.append(label)
    return {'bboxes': bboxes, 'labels': labels}


def get_image_annotations(image_file_path):
    return get_txt_annotations(image_file_path.with_suffix('.txt'))


def get_test_annotations(image_file_path):
    def _poly2bbox(polygon: np.ndarray):
        r = polygon[:, 0].max()
        l = polygon[:, 0].min()

        t = polygon[:, 1].max()
        b = polygon[:, 1].min()
        return l, b, r, t

    def _write_line(bbox, label):
        return ','.join([str(x) for x in bbox]) + '|' + label + '\n'

    def _clip_w_h(polygon, w, h):
        return np.stack([polygon[:, 0].clip(0, w), polygon[:, 1].clip(0, h)], axis=1)

    bboxes = []
    labels = []

    label_file = image_file_path.with_suffix('.json')
    if label_file.exists():
        with open(label_file) as json_file:
            data = json.load(json_file)

        w = data['imageWidth']
        h = data['imageHeight']

        for shape in data['shapes']:
            polygon = np.array(shape['points'])
            polygon = _clip_w_h(polygon, w, h)
            bboxes.append(_poly2bbox(polygon))
            labels.append(shape['label'])

    with open(image_file_path.with_suffix('.txt'), mode='w') as outfile:
        for bbox, label in zip(bboxes, labels):
            outfile.write(_write_line(bbox, label))
    return {'bboxes': bboxes, 'labels': labels}


def remove_unknown_labels(label_instance, class_map):
    bboxes, labels = label_instance.bboxes, label_instance.labels
    bboxes_filtered, labels_filtered = [], []
    unknown_labels = set()
    for bbox, label in zip(bboxes, labels):
        if label in class_map.id2class:
            bboxes_filtered.append(bbox)
            labels_filtered.append(label)
        else:
            unknown_labels.add(label)
    print(f'unknown labels: {unknown_labels}, ignoring.')
#     bboxes, labels = zip(*[(bbox, label) for bbox, label in zip(bboxes, labels) if label in class_map.id2class])
    label_instance.bboxes, label_instance.labels = bboxes_filtered, labels_filtered
    return label_instance


def maybe_rename_labels(label_instance):
    test_json_to_21_gen_classes = {
        'monety': 'gotowka',
        'banknoty': 'gotowka',
        'ostre_przedmioty': 'ostry_przedmiot',
        'papierosy': 'paczka_papierosow',
        'woda_w_butelce': 'plyn',
        'wodka': 'plyn',
        'wino': 'plyn',
        'nozyczki': 'ostry_przedmiot',
        'noz': 'ostry_przedmiot',
        'powerbank': 'baterie',
        'baterie_alkaliczne': 'baterie',
        'wiertarka_akumulatorowa': 'narzedzia_akumulatorowe',
        'kosmetyki_w_plynie': 'plyn',
    }

    test_json_to_10_classes = {
        'ostry_przedmiot': 'ostre_przedmioty',
        'nozyczki': 'ostre_przedmioty',
        'noz': 'ostre_przedmioty',
        'woda_w_butelce': 'plyn',
        'wodka': 'plyn',
        'wino': 'plyn',
        'kosmetyki_w_plynie': 'plyn',
        'paczka_papierosow': 'papierosy',

    }
    label_instance.labels = [test_json_to_10_classes.get(label, label) for label in label_instance.labels]

    return label_instance


class Label:
    def __init__(self, bboxes, labels):
        self.bboxes, self.labels = bboxes, labels

    @classmethod
    def create(cls, _dict):
        return cls(bboxes=_dict['bboxes'], labels=_dict['labels'])


class TxtParser(parsers.Parser, parsers.FilepathMixin, parsers.LabelsMixin, parsers.BBoxesMixin):
    pipeline = Pipeline([get_image_annotations, Label.create])

    def __init__(self, path, class_map, folders=None, pipeline=None):
        if pipeline: self.pipeline = pipeline
        self.path = path
        self.items = get_image_files(path, folders=folders)
        self.path2label = {str(item): self.pipeline(item) for item in self.items}
        # self.class_map = class_map
        super().__init__(class_map=class_map)

    def __iter__(self): return iter(self.items)
    def __len__(self): return len(self.items)
    def image_width_height(self, o) -> Tuple[int, int]: return Image.open(o).size
    def imageid(self, o) -> Hashable: return o
    def filepath(self, o) -> Union[str, Path]: return o
    def bboxes(self, o) -> List[BBox]: return [BBox(*bbox) for bbox in self.path2label[str(o)].bboxes]
    def labels(self, o) -> List[int]:
        return self.path2label[str(o)].labels

    @staticmethod
    def list_labels(x):
        try:
            bboxes, labels = x
        except ValueError as e:
            print(e)
            labels = x
        return labels


def test_pipeline(class_map):
    return Pipeline([get_test_annotations,
                         Label.create,
                         maybe_rename_labels,
                         partial(remove_unknown_labels, class_map=class_map)])