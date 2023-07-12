import numpy as np

class Bbox:
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2):
        return cls(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_mask(cls, mask):
        if not np.any(mask):
            return None  # No mask found

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return cls.from_xyxy(xmin, ymin, xmax, ymax)

    def get_xywh(self):
        return [self.x, self.y, self.w, self.h]

    def get_xyxy(self):
        return [self.x, self.y, self.x2, self.y2]

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h


def seg_to_bbox(seg):
    pass
