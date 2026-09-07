from pathlib import Path

import cv2
import ezdxf
import ezdxf.path
import numpy as np

COINCIDENT_MM = 1e-7

CURVE_TYPES = frozenset(
    {"LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "SPLINE", "HELIX"}
)


def read_dxf(path, tolerance_mm=0.02, join_tolerance_mm=0.01):
    """Read a DXF into parts, each an (outer, holes) pair of (N, 2) arrays, chaining loose entities into closed profiles and deciding holes by nesting. An open profile raises."""
    doc = ezdxf.readfile(str(path))

    loops = []
    open_chains = []
    for entity in _flat_entities(doc.modelspace()):
        if entity.dxftype() not in CURVE_TYPES:
            continue
        curve = ezdxf.path.make_path(entity)
        vertices = np.array(
            [(v.x, v.y) for v in curve.flattening(tolerance_mm)], dtype=float
        )
        if len(vertices) >= 2:
            step = np.hypot(*np.diff(vertices, axis=0).T)
            vertices = np.vstack([vertices[0], vertices[1:][step > COINCIDENT_MM]])
        if len(vertices) < 2:
            continue
        if _close(vertices[0], vertices[-1], join_tolerance_mm):
            if len(vertices) > 3:
                loops.append(vertices[:-1])
        else:
            open_chains.append(vertices)

    chained, unclosed = _chain(open_chains, join_tolerance_mm)
    if unclosed:
        raise ValueError(
            f"{unclosed} open contour(s) in {Path(path).name} -- plasma cuts closed "
            "profiles only"
        )
    return _split(loops + chained)


def _flat_entities(layout):
    """Yield geometry, exploding block references."""
    for entity in layout:
        if entity.dxftype() == "INSERT":
            yield from _flat_entities(entity.virtual_entities())
        else:
            yield entity


def _close(a, b, tolerance):
    return bool(np.hypot(*(np.asarray(a) - np.asarray(b))) <= tolerance)


def _chain(pieces, tolerance):
    """Walk open pieces end to end into loops; returns the loops and how many never closed."""
    remaining = list(pieces)
    loops, unclosed = [], 0
    while remaining:
        chain = remaining.pop(0)
        while not _close(chain[0], chain[-1], tolerance):
            for index, piece in enumerate(remaining):
                if _close(chain[-1], piece[0], tolerance):
                    chain = np.vstack([chain, piece[1:]])
                elif _close(chain[-1], piece[-1], tolerance):
                    chain = np.vstack([chain, piece[::-1][1:]])
                else:
                    continue
                remaining.pop(index)
                break
            else:
                unclosed += 1
                chain = None
                break
        if chain is not None and len(chain) > 3:
            loops.append(chain[:-1])
    return loops, unclosed


def _split(contours):
    """Group contours into (outer, holes) parts by nesting depth; an island inside a hole is its own part."""
    n = len(contours)
    polys = [np.asarray(c, dtype=np.float32) for c in contours]
    areas = np.array([abs(cv2.contourArea(p)) for p in polys])
    boxes = [cv2.boundingRect(p) for p in polys]

    def inside(child, holder):
        """Containment by majority vote over the child's vertices."""
        cx, cy, cw, ch = boxes[child]
        hx, hy, hw, hh = boxes[holder]
        if cx < hx or cy < hy or cx + cw > hx + hw or cy + ch > hy + hh:
            return False
        pts = polys[child]
        votes = [cv2.pointPolygonTest(polys[holder], tuple(map(float, p)), False)
                 for p in pts[::max(1, len(pts) // 16)]]
        return sum(1 for v in votes if v > 0) * 2 > len(votes)

    # The smallest enclosing contour is the parent.
    parent = np.full(n, -1, dtype=int)
    for child in range(n):
        for holder in range(n):
            if holder == child or areas[holder] <= areas[child] or not inside(child, holder):
                continue
            if parent[child] == -1 or areas[holder] < areas[parent[child]]:
                parent[child] = holder

    depth = []
    for index in range(n):
        steps, walk = 0, parent[index]
        while walk != -1:
            steps, walk = steps + 1, parent[walk]
        depth.append(steps)

    order = [i for i in range(n) if depth[i] % 2 == 0]
    slot = {index: position for position, index in enumerate(order)}
    parts = [(contours[i], []) for i in order]
    for index in range(n):
        if depth[index] % 2 == 1:
            parts[slot[int(parent[index])]][1].append(contours[index])
    return parts
