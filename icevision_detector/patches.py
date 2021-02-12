from typing import Tuple
from fastcore.foundation import patch
from icevision import parsers, FilepathRecordMixin
from torchvision import io


@patch
def image_width_height(self:parsers.COCOBaseParser, o) -> Tuple[int, int]:
    return self._info['width'], self._info['height']


@patch
def _load(self:FilepathRecordMixin):
    torch_img = io.read_image(str(self.filepath))
    c, self.height, self.width = torch_img.shape
    if c == 1: torch_img = torch_img.float().mean(dim=0).repeat((3,1,1)).byte()
    # CHW to HWC
    self.img = torch_img.permute(1,2,0).numpy()
    # TODO, HACK: is it correct to overwrite height and width here?
    return


