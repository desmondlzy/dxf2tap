"""What the plate looks like to the lead planner."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Plate:
    """Two fields on one Y-up grid: room is the distance to the nearest metal, width the local part thickness."""

    room: np.ndarray
    width: np.ndarray
    origin_mm: tuple[float, float]
    resolution_mm: float

    def room_at(self, points):
        return self._read(self.room, points, outside=-1.0)

    def width_at(self, points):
        return self._read(self.width, points, outside=0.0)

    def _read(self, field, points, outside):
        """Bilinear lookup; off the grid reads as outside."""
        points = np.asarray(points, dtype=float)
        column = (points[..., 0] - self.origin_mm[0]) / self.resolution_mm
        row = (points[..., 1] - self.origin_mm[1]) / self.resolution_mm
        rows, columns = field.shape
        on = (row >= 0) & (column >= 0) & (row <= rows - 1) & (column <= columns - 1)

        left = np.clip(np.floor(column).astype(int), 0, columns - 2)
        low = np.clip(np.floor(row).astype(int), 0, rows - 2)
        across = np.clip(column - left, 0.0, 1.0)
        up = np.clip(row - low, 0.0, 1.0)
        lower = field[low, left] * (1 - across) + field[low, left + 1] * across
        upper = field[low + 1, left] * (1 - across) + field[low + 1, left + 1] * across
        return np.where(on, lower * (1 - up) + upper * up, outside)


def survey_plate(parts, resolution_mm=0.25, pad_mm=30.0):
    """Paint the parts into a pixel mask and build the two fields the lead planner reads. pad_mm must exceed the furthest a lead can reach."""
    points = np.vstack([c for outer, holes in parts for c in [outer, *holes]])
    origin = (float(points[:, 0].min() - pad_mm), float(points[:, 1].min() - pad_mm))
    span_x = float(points[:, 0].max() + pad_mm) - origin[0]
    span_y = float(points[:, 1].max() + pad_mm) - origin[1]
    shape = (int(np.ceil(span_y / resolution_mm)) + 1,
             int(np.ceil(span_x / resolution_mm)) + 1)

    def to_px(pts):
        local = (np.asarray(pts, dtype=float) - np.asarray(origin)) / resolution_mm
        return np.round(local).astype(np.int32)

    # Outers filled, then holes punched, so a hole always wins.
    metal = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(metal, [to_px(outer) for outer, _ in parts], 1)
    holes = [to_px(hole) for _, holes in parts for hole in holes]
    if holes:
        cv2.fillPoly(metal, holes, 0)
    metal = metal.astype(bool)

    return Plate(
        room=(ndimage.distance_transform_edt(~metal) * resolution_mm).astype(np.float32),
        width=(2 * ndimage.distance_transform_edt(metal) * resolution_mm).astype(
            np.float32
        ),
        origin_mm=origin,
        resolution_mm=resolution_mm,
    )
