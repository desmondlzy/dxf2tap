import numpy as np

from .types import Lead

SAMPLE_MM = 0.2  # how finely a lead is checked against the plate
CANDIDATE_SEAMS = 180
SHRINK = 0.8  # radius ladder step when the wanted lead does not fit
SWEEPS_DEG = (90.0, 135.0, 180.0)  # arc sweeps, gentlest first
ANGLES_DEG = (45.0, 60.0, 90.0)  # straight leads, measured off the contour
COMFORT = 2.0  # pierce clearance in rules past which more stops helping

# Ranking of leads that already passed every hard constraint.
WEIGHTS = {
    "radius": 1.00,  # a bigger arc is gentler
    "sweep": 0.50,  # 90 degrees unless a wrap buys something
    "flatness": 0.60,  # seam on a straight run, not a corner
    "thickness": 0.50,  # put the divots where there is metal
    "clearance": 0.40,  # comfort beyond the minimum
    "separation": 0.25,  # keep a part's seams apart
    "run": 0.25,  # a straight run is travel and heat, so earn it
}


def plan_lead(
    points,
    plate,
    lead_mm=3.0,
    min_lead_mm=0.5,
    overburn_mm=0.5,
    clearance_mm=1.5,
    avoid=(),
    lead_out=True,
):
    """Choose the seam and leads for one contour: the largest tangential arc that clears the plate, on the flattest and thickest run, with the pierce the clearest point of the whole lead. Returns the seam as arclength plus the lead-in and lead-out, None where none fits."""
    closed = np.vstack([points, points[:1]])
    steps = np.hypot(*np.diff(closed, axis=0).T)
    stations = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(stations[-1])
    if total <= 0:
        return 0.0, None, None

    overburn = min(overburn_mm, total / 4)
    span = min(0.5, total / 32)
    turn_span = min(2.0, total / 8)
    probe_mm = 3 * clearance_mm
    slack = plate.resolution_mm  # the ramp test forgives one cell of raster error
    pierce_mm = clearance_mm + 2 * plate.resolution_mm  # the raster flatters room near an edge

    def at(distance):
        wrapped = np.mod(np.atleast_1d(np.asarray(distance, dtype=float)), total)
        return np.stack([np.interp(wrapped, stations, closed[:, 0]),
                         np.interp(wrapped, stations, closed[:, 1])], axis=-1)

    def tangent(distance):
        step = at(np.asarray(distance) + span) - at(np.asarray(distance) - span)
        length = np.linalg.norm(step, axis=-1, keepdims=True)
        return step / np.where(length > 0, length, 1.0)

    def normal(heading):
        return np.stack([-heading[..., 1], heading[..., 0]], axis=-1)

    def leads(distance, radius, sweep, departing):
        """Every arc lead at one radius and sweep, vectorized over seams, plus the shortest straight run beyond the arc that gets the pierce its clearance."""
        point, heading = at(distance), tangent(distance)
        into = normal(heading)
        center = point + radius * into
        sense = 1 if departing else -1
        count = max(6, int(np.ceil(radius * sweep / SAMPLE_MM)))
        phi = np.linspace(0.0, sweep, count) * sense
        spoke = point - center
        turn = np.stack([np.cos(phi), np.sin(phi)], axis=0)
        samples = np.stack([
            center[:, 0, None] + turn[0] * spoke[:, 0, None]
            - turn[1] * spoke[:, 1, None],
            center[:, 1, None] + turn[1] * spoke[:, 0, None]
            + turn[0] * spoke[:, 1, None],
        ], axis=-1)
        room = plate.room_at(samples)
        needed = np.minimum(clearance_mm, radius * (1 - np.cos(phi)) / 2)
        legal = np.all(room >= needed - slack, axis=1)

        spin = sense * sweep
        along = sense * np.stack([
            heading[:, 0] * np.cos(spin) - heading[:, 1] * np.sin(spin),
            heading[:, 0] * np.sin(spin) + heading[:, 1] * np.cos(spin),
        ], axis=1)
        bend = samples[:, -1, :]
        span = max(lead_mm, 2 * clearance_mm)
        reach = np.arange(0.0, span + SAMPLE_MM, SAMPLE_MM)
        walk = bend[:, None, :] + reach[None, :, None] * along[:, None, :]
        outside = plate.room_at(walk)

        held = float(needed[-1])
        clear = np.cumprod(outside >= held - slack, axis=1).astype(bool)
        peak = np.maximum.accumulate(
            np.concatenate([room.max(axis=1)[:, None], outside], axis=1), axis=1
        )[:, 1:]
        good = clear & (outside >= pierce_mm) & (outside >= peak - slack)

        stop = np.argmax(good, axis=1)
        legal &= good.any(axis=1)
        rows = np.arange(len(stop))
        return (legal, walk[rows, stop, :], center, outside[rows, stop],
                bend, reach[stop])

    def straights(distance, length, angle, departing):
        """Straight leads at an angle off the contour, the fallback when no arc fits."""
        point, heading = at(distance), tangent(distance)
        into = normal(heading)
        sense = 1 if departing else -1
        away = (np.sin(angle) * into
                + sense * np.cos(angle) * heading)
        count = max(4, int(np.ceil(length / SAMPLE_MM)))
        reach = np.linspace(0.0, length, count)
        samples = point[:, None, :] + reach[None, :, None] * away[:, None, :]
        room = plate.room_at(samples)
        needed = np.minimum(clearance_mm, reach * np.sin(angle) / 2)
        legal = np.all(room >= needed - slack, axis=1)
        legal &= room[:, -1] >= pierce_mm
        legal &= room[:, -1] >= room.max(axis=1) - slack
        return (legal, samples[:, -1, :], None, room[:, -1],
                None, np.zeros(len(point)))

    seams = np.arange(CANDIDATE_SEAMS) * (total / CANDIDATE_SEAMS)
    here = at(seams)

    ahead, behind = tangent(seams + turn_span), tangent(seams - turn_span)
    flatness = -np.abs(np.arctan2(
        ahead[:, 0] * behind[:, 1] - ahead[:, 1] * behind[:, 0],
        ahead[:, 0] * behind[:, 0] + ahead[:, 1] * behind[:, 1],
    ))

    inward = normal(tangent(seams))
    depth = np.linspace(0.0, probe_mm, 12)
    thickness = plate.width_at(
        here[:, None, :] - depth[None, :, None] * inward[:, None, :]
    ).max(axis=1)

    if len(avoid):
        gaps = np.linalg.norm(here[:, None, :] - np.asarray(avoid)[None], axis=2)
        separation = np.minimum(gaps.min(axis=1) / max(2 * lead_mm, 1e-9), 1.0)
    else:
        separation = np.ones(len(seams))

    static = (WEIGHTS["flatness"] * _unit(flatness)
              + WEIGHTS["thickness"] * _unit(thickness)
              + WEIGHTS["separation"] * separation)

    sizes = []
    size = lead_mm
    while size >= min_lead_mm:
        sizes.append(size)
        size *= SHRINK
    sizes.append(min_lead_mm)

    for build in (leads, straights):
        arc = build is leads
        shapes = SWEEPS_DEG if arc else ANGLES_DEG
        wanted = shapes[0]
        best = None
        entry_only = None
        for shape_deg in shapes:
            shape = np.radians(shape_deg)
            for size in sizes:
                ok_in, end_in, hub_in, room_in, bend_in, run_in = build(
                    seams, size, shape, False
                )
                if lead_out:
                    ok_out, end_out, hub_out, room_out, bend_out, run_out = build(
                        seams + overburn, size, shape, True
                    )
                    workable = ok_in & ok_out
                    spare = np.minimum(room_in, room_out)
                    strayed = np.maximum(run_in, run_out)
                else:
                    workable, spare, strayed = ok_in, room_in, run_in
                comfort = np.minimum(spare / (COMFORT * clearance_mm), 1.0)
                thrift = 1.0 - np.minimum(strayed / lead_mm, 1.0)
                score = (static
                         + WEIGHTS["radius"] * min(size / lead_mm, 1.0)
                         + WEIGHTS["sweep"] * min(wanted / shape_deg, 1.0)
                         + WEIGHTS["clearance"] * comfort
                         + WEIGHTS["run"] * thrift)

                def shape_at(index, end, hubs, bends, runs):
                    return _lead(end[index],
                                 hubs[index] if arc else None,
                                 bends[index] if arc else None,
                                 float(runs[index]), size, shape_deg, arc)

                if workable.any():
                    pick = int(np.argmax(np.where(workable, score, -np.inf)))
                    if best is None or score[pick] > best[0]:
                        best = (
                            float(score[pick]), float(seams[pick]),
                            shape_at(pick, end_in, hub_in, bend_in, run_in),
                            shape_at(pick, end_out, hub_out, bend_out, run_out)
                            if lead_out else None,
                        )
                if entry_only is None and ok_in.any():
                    pick = int(np.argmax(np.where(ok_in, score, -np.inf)))
                    entry_only = (float(seams[pick]),
                                  shape_at(pick, end_in, hub_in, bend_in, run_in))
        if best is not None:
            return best[1], best[2], best[3]
        if entry_only is not None:
            return entry_only[0], entry_only[1], None

    return float(seams[int(np.argmax(static))]), None, None


def _lead(end, center, bend, run, size, shape_deg, arc):
    """One candidate as a Lead; a zero run drops the bend so a plain arc stays plain."""
    return Lead(
        free_end=(float(end[0]), float(end[1])),
        center=(float(center[0]), float(center[1])) if arc else None,
        radius=size,
        sweep_deg=shape_deg if arc else 0.0,
        bend=(float(bend[0]), float(bend[1]))
        if arc and run > 1e-9 else None,
    )


def _unit(values):
    """Rescale to [0, 1] across the candidates, or all ones if they tie."""
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)
