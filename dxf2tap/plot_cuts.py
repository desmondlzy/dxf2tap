import matplotlib

matplotlib.use("Agg")  # no display 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString
from shapely.ops import unary_union

KERF = "0.78"
CONTOUR = "#12355b"
LEAD_IN = "#1a8a3c"
LEAD_OUT = "#d1651a"
PIERCE = "#c02020"
RAPID = "0.62"
DPI = 200


def plot_cuts(cuts, figure_path, kerf_mm, title=""):
    """Draw the plan"""
    figure, axis = plt.subplots(figsize=(10, 10))

    swath = unary_union(
        [LineString(_whole_path(cut)).buffer(kerf_mm / 2, quad_segs=16) for cut in cuts]
    )
    for piece in getattr(swath, "geoms", [swath]):
        axis.add_patch(PathPatch(_compound(piece), facecolor=KERF,
                                 edgecolor="none", zorder=1))

    here = None
    for order, cut in enumerate(cuts, start=1):
        pierce = np.asarray(cut.pierce)
        if here is not None:
            axis.plot(*zip(here, pierce), color=RAPID, lw=0.7, ls=(0, (2, 3)),
                      zorder=2)
        axis.plot(cut.path[:, 0], cut.path[:, 1], color=CONTOUR, lw=1.2, zorder=4)
        _arrow(axis, cut.path)
        for lead, color, ends in ((cut.lead_in, LEAD_IN, "in"),
                                   (cut.lead_out, LEAD_OUT, "out")):
            if lead is None:
                continue
            seam = cut.path[0] if ends == "in" else cut.path[-1]
            axis.plot(*zip(*_lead_points(lead, seam, ends)), color=color,
                      lw=1.8, zorder=5)
        axis.add_patch(plt.Circle(pierce, kerf_mm, facecolor="none",
                                  edgecolor=PIERCE, lw=1.1, zorder=6))
        axis.annotate(str(order), pierce, textcoords="offset points",
                      xytext=(5, 5), fontsize=8, color=PIERCE, zorder=7)
        here = tuple(cut.lead_out.free_end if cut.lead_out else cut.path[-1])

    everything = np.vstack([_whole_path(cut) for cut in cuts])
    pad = 3 * kerf_mm + 2.0
    axis.set_xlim(everything[:, 0].min() - pad, everything[:, 0].max() + pad)
    axis.set_ylim(everything[:, 1].min() - pad, everything[:, 1].max() + pad)
    axis.set_aspect("equal")
    axis.grid(color="0.85", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_title(f"{title}    {len(cuts)} cuts, kerf {kerf_mm:g} mm",
                   fontsize=11, loc="left")
    axis.legend(handles=[
        Patch(facecolor=KERF, label="metal removed"),
        Line2D([], [], color=CONTOUR, lw=1.4, label="cut, in travel direction"),
        Line2D([], [], color=LEAD_IN, lw=1.8, label="lead-in"),
        Line2D([], [], color=LEAD_OUT, lw=1.8, label="lead-out"),
        Line2D([], [], color=PIERCE, lw=0, marker="o", mfc="none", ms=7,
               label="pierce"),
        Line2D([], [], color=RAPID, lw=0.9, ls=(0, (2, 3)), label="travel"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncols=6, fontsize=8,
       frameon=False)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=DPI)
    plt.close(figure)
    return len(cuts)


def _whole_path(cut):
    """Every millimeter the arc is lit, leads included."""
    pieces = []
    if cut.lead_in is not None:
        pieces.append(np.array(_lead_points(cut.lead_in, cut.path[0], "in")))
    pieces.append(cut.path)
    if cut.lead_out is not None:
        pieces.append(np.array(_lead_points(cut.lead_out, cut.path[-1], "out")))
    return np.vstack(pieces)


def _lead_points(lead, seam, ends):
    """A lead as a polyline, running from its free end to the seam or back."""
    if not lead.is_arc:
        pair = [tuple(lead.free_end), tuple(seam)]
        return pair if ends == "in" else pair[::-1]

    center = np.asarray(lead.center)
    start = np.arctan2(*(np.asarray(seam) - center)[::-1])
    turn = np.radians(lead.sweep_deg) * (1 if ends == "out" else -1)
    angle = np.linspace(start, start + turn, 48)
    arc = center + lead.radius * np.column_stack([np.cos(angle), np.sin(angle)])
    if ends == "in":
        arc = arc[::-1]
    points = [tuple(p) for p in arc]

    if lead.bend is not None:
        points = ([tuple(lead.free_end)] + points if ends == "in"
                  else points + [tuple(lead.free_end)])
    return points


def _arrow(axis, path):
    """One arrowhead a quarter of the way round."""
    if len(path) < 8:
        return
    at = len(path) // 4
    axis.annotate("", xy=path[at + 1], xytext=path[at], zorder=5,
                  arrowprops=dict(arrowstyle="-|>", color=CONTOUR,
                                  lw=0.9, shrinkA=0, shrinkB=0,
                                  mutation_scale=11))


def _compound(polygon):
    """A shapely polygon with holes as one matplotlib path."""
    vertices, codes = [], []
    for loop in [polygon.exterior, *polygon.interiors]:
        points = np.asarray(loop.coords)
        vertices.extend(points)
        codes.extend([MplPath.MOVETO]
                     + [MplPath.LINETO] * (len(points) - 2)
                     + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(vertices), codes)
