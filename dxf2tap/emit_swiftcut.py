import re
from datetime import date as _date

import numpy as np

FILENAME = re.compile(r"^\s*\(Filename:")
DATE = re.compile(r"^\s*\(Date:")


def emit_swiftcut(cuts, machine, name="part", date=None):
    """Format the cuts as a SwiftCut .tap program, under the sample's header and in its dialect. pure formatting."""
    stamp = date if date is not None else _date.today().strftime("%d.%m.%Y")
    offset = np.asarray(machine.sense_offset_mm, dtype=float)

    out = []
    for raw in machine.preamble:
        if FILENAME.match(raw):
            out.append(f" (Filename: {name})")
        elif DATE.match(raw):
            out.append(f" (Date: {stamp})")
        else:
            out.append(raw)

    state = {"motion": None, "feed": machine.feed_mm_min}
    part = None

    def move(code, point, center=None, feed=None, here=None):
        """One motion block, with G-code and feed emitted modally."""
        word = "" if state["motion"] == code else f" {code}"
        text = f"{word} X{_n(point[0])} Y{_n(point[1])}"
        if center is not None:
            text += f" I{_n(center[0] - here[0])} J{_n(center[1] - here[1])}"
        if feed is not None and feed != state["feed"]:
            text += f" F{_n(feed)}"
            state["feed"] = feed
        state["motion"] = code
        out.append(text if text.startswith(" ") else " " + text)

    for cut in cuts:
        # Rapid to the sensor position, reference the plate, then climb to the pierce.
        pierce = np.asarray(cut.pierce, dtype=float)
        move("G00", pierce + offset)
        if cut.part != part:
            part = cut.part
            out.append(f" (Part: {part})")
        out.append(f"#130={1 if cut.thc else 0} (THC {'On' if cut.thc else 'Off'})")
        out.append(" M12 (reference soft sense)")
        out.append(f" Z#100 X{_n(pierce[0])} Y{_n(pierce[1])}")
        out.append(" M03")
        out.append(" G04 P#110")
        out.append(" G00 Z#120")
        state["motion"] = "G00"

        here = pierce
        if cut.lead_in is not None:
            entry = cut.lead_in
            if entry.bend is not None:
                move("G01", entry.bend, feed=cut.lead_feed_mm_min)
                here = np.asarray(entry.bend, dtype=float)
            code = "G03" if entry.is_arc else "G01"
            move(code, cut.path[0], entry.center, cut.lead_feed_mm_min, here)
        here = cut.path[0]

        for point in cut.path[1:]:
            move("G01", point, feed=cut.feed_mm_min)
            here = point

        # Height control off before leaving the plate, or it dives the torch.
        if cut.thc:
            out.append("#130=0")
        if cut.lead_out is not None:
            leaving = cut.lead_out
            code = "G03" if leaving.is_arc else "G01"
            corner = leaving.bend if leaving.bend is not None else leaving.free_end
            move(code, corner, leaving.center, cut.lead_feed_mm_min, here)
            if leaving.bend is not None:
                move("G01", leaving.free_end, feed=cut.lead_feed_mm_min)

        out.append(" M05")
        out.append(f" G00 Z{_n(machine.retract_mm)}")
        state["motion"] = "G00"

    out.append(" M30")
    return "\n".join(out) + "\n"


def _n(value):
    """Three decimals with trailing zeros stripped, the machine's own format."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0", "-") else text
