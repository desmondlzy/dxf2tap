"""The post's records: a lead, a cut, and the machine settings."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lead:
    """A tangent arc onto the contour (center set), a straight (center None), or a straight run to bend and then the arc. free_end is the pierce or the torch-off point."""

    free_end: tuple[float, float]
    center: tuple[float, float] | None
    radius: float
    sweep_deg: float = 90.0
    bend: tuple[float, float] | None = None

    @property
    def is_arc(self):
        return self.center is not None

    @property
    def length_mm(self):
        """Everything the torch travels with the arc lit."""
        if not self.is_arc:
            return self.radius
        run = 0.0 if self.bend is None else math.dist(self.free_end, self.bend)
        return self.radius * math.radians(self.sweep_deg) + run


@dataclass(frozen=True)
class Cut:
    """One contour, fully decided: the open path the torch walks, its leads, feeds and THC."""

    part: str
    path: np.ndarray
    lead_in: Lead | None
    lead_out: Lead | None
    thc: bool
    feed_mm_min: float
    lead_feed_mm_min: float

    @property
    def is_hole(self):
        """Counter-clockwise means a hole, since the material was put on the right of travel."""
        x, y = self.path[:, 0], self.path[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0

    @property
    def pierce(self):
        """Where the torch fires: the lead-in's free end, else the seam."""
        if self.lead_in is not None:
            return self.lead_in.free_end
        return (float(self.path[0][0]), float(self.path[0][1]))


@dataclass(frozen=True)
class Machine:
    """Settings read off a sample program by read_machine, plus the kerf, which no program records. preamble is the sample's header, written back verbatim."""

    preamble: tuple[str, ...]
    feed_mm_min: float
    sense_offset_mm: tuple[float, float]
    retract_mm: float = 20.0
    kerf_mm: float = 0.6

    small_feed_fraction: float = 0.6
    lead_feed_fraction: float = 0.9
    small_contour_mm: float = 25.0
