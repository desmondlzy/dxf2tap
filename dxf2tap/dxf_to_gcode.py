from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from .emit_swiftcut import emit_swiftcut
from .plan_lead import plan_lead
from .plate import survey_plate
from .plot_cuts import plot_cuts
from .read_dxf import read_dxf
from .types import Cut


def dxf_to_gcode(
    dxf_path,
    output_path,
    machine,
    plot_path=None,
    lead_mm=3.0,
    min_lead_mm=0.5,
    overburn_mm=0.5,
    clearance_mm=1.5,
    plate_resolution_mm=0.25,
    name=None,
    date=None,
):
    """Convert a DXF into a plasma program"""
    source = Path(dxf_path)
    drawn = read_dxf(source)
    if not drawn:
        raise ValueError(f"no closed contours in {source.name}")

    kerf = machine.kerf_mm
    parts, dropped = _compensate(drawn, kerf)
    if not parts:
        raise ValueError(f"nothing survives a {kerf:g} mm kerf")

    stem = source.stem
    width = len(str(len(parts)))
    names = ([stem] if len(parts) == 1
             else [f"{stem}-{i + 1:0{width}d}" for i in range(len(parts))])

    plate = survey_plate(
        parts, plate_resolution_mm,
        pad_mm=max(30.0, 3 * lead_mm, 3 * clearance_mm),
    )

    by_part = []
    for label, (outer, holes) in zip(names, parts):
        group = []
        seams = []
        for contour, hole in [(h, True) for h in holes] + [(outer, False)]:
            points = _orient(contour, hole)
            # A hole has no lead-out, so its overburn must outlast the arc's lag.
            overburn = max(overburn_mm, kerf) if hole else overburn_mm
            seam, entry, leaving = plan_lead(
                points, plate,
                lead_mm=lead_mm, min_lead_mm=min_lead_mm,
                overburn_mm=overburn, clearance_mm=clearance_mm,
                avoid=tuple(seams), lead_out=not hole,
            )
            path = _cut_path(points, seam, overburn)
            seams.append((float(path[0][0]), float(path[0][1])))
            _, span = cv2.minEnclosingCircle(np.asarray(points, dtype=np.float32))
            small = 2 * span < machine.small_contour_mm
            feed = machine.feed_mm_min * (machine.small_feed_fraction if small else 1.0)
            group.append(Cut(
                part=label,
                path=path,
                lead_in=entry,
                lead_out=leaving,
                thc=not small,
                feed_mm_min=feed,
                lead_feed_mm_min=feed * machine.lead_feed_fraction,
            ))
        by_part.append(group)

    cuts = _order(by_part)
    label = name or stem
    Path(output_path).write_text(emit_swiftcut(cuts, machine, name=label, date=date))
    if plot_path is not None:
        plot_cuts(cuts, plot_path, kerf, title=label)

    leads = [lead for cut in cuts for lead in (cut.lead_in, cut.lead_out)
             if lead is not None]
    return {
        "contours": len(cuts),
        "holes": sum(cut.is_hole for cut in cuts),
        "dropped": dropped,
        "arc_leads": sum(lead.is_arc for lead in leads),
        "straight_leads": sum(not lead.is_arc for lead in leads),
        # A hole has no lead-out by design; only a wanted lead that is missing counts.
        "leadless": sum(cut.lead_in is None for cut in cuts)
        + sum(cut.lead_out is None and not cut.is_hole for cut in cuts),
        "min_lead_mm": min((lead.radius for lead in leads), default=0.0),
        "cut_length_mm": sum(float(np.hypot(*np.diff(cut.path, axis=0).T).sum())
                             for cut in cuts) + sum(lead.length_mm for lead in leads),
    }


def _compensate(parts, kerf):
    """Offset each part half a kerf into the waste through one GEOS buffer, dropping loops narrower than the kerf before and after. Returns the parts and the count dropped."""
    kept, dropped = [], 0
    for outer, holes in parts:
        part = _drop_narrow((outer, holes), 2 * kerf)
        if part is None:
            dropped += 1 + len(holes)
            continue
        dropped += len(holes) - len(part[1])
        if not kerf:
            kept.append(part)
            continue
        grown = Polygon(part[0], part[1]).buffer(
            kerf / 2, join_style="mitre", quad_segs=64, mitre_limit=8.0
        )
        for piece in getattr(grown, "geoms", [grown]):
            if piece.is_empty:
                continue
            loop = lambda ring: np.asarray(ring.coords, dtype=float)[:-1, :2]
            piece = _drop_narrow((loop(piece.exterior),
                                  [loop(hole) for hole in piece.interiors]), kerf)
            if piece is not None:
                kept.append(piece)
    return kept, dropped


def _drop_narrow(part, min_diameter_mm):
    """Drop loops whose inscribed circle is under min_diameter_mm; None if the outer itself is lost."""
    outer, holes = part
    if min_diameter_mm <= 0:
        return part

    def survives(points):
        return not Polygon(points).buffer(-min_diameter_mm / 2).is_empty

    if not survives(outer):
        return None
    return outer, [hole for hole in holes if survives(hole)]


def _orient(points, hole):
    """Wind the contour so the material is on the right of travel: outer clockwise, hole counter-clockwise."""
    points = np.asarray(points, dtype=float)
    x, y = points[:, 0], points[:, 1]
    counter_clockwise = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0
    return points if counter_clockwise == hole else points[::-1].copy()


def _cut_path(points, seam_mm, overburn_mm):
    """Unroll the closed contour into the open path the torch walks: from the seam, round the loop, and overburn_mm past it."""
    closed = np.vstack([points, points[:1]])
    steps = np.hypot(*np.diff(closed, axis=0).T)
    stations = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(stations[-1])
    if total <= 0:
        return np.asarray(points, dtype=float)

    overburn = min(overburn_mm, total / 4)
    start = float(np.mod(seam_mm, total))
    wanted = np.concatenate([
        [start],
        stations[(stations > start) & (stations < total)],
        stations[:-1] + total,
    ])
    wanted = wanted[wanted <= start + total + overburn]
    wanted = np.append(wanted, start + total + overburn)
    walked = np.mod(wanted, total)
    return np.stack([
        np.interp(walked, stations, closed[:, 0]),
        np.interp(walked, stations, closed[:, 1]),
    ], axis=1)


def _order(by_part, start=(0.0, 0.0)):
    """Each part's holes before its boundary, and parts nearest-neighbor by pierce point."""
    remaining = [group for group in by_part if group]
    here = np.asarray(start, dtype=float)
    ordered = []
    while remaining:
        distances = [
            float(np.hypot(*(np.asarray(group[0].pierce) - here))) for group in remaining
        ]
        group = remaining.pop(int(np.argmin(distances)))
        ordered.extend(group)
        here = np.asarray(group[-1].path[-1], dtype=float)
    return ordered
