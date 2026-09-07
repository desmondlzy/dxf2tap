import re
from pathlib import Path

from .types import Machine

REQUIRED_VARIABLES = (100, 110, 120, 130)  # pierce height, delay, cut height, THC
XY = r"X(-?\d+\.?\d*)\s+Y(-?\d+\.?\d*)"


def read_machine(sample, kerf_mm=0.6):
    """Read the machine's settings off a program it has run: the header up to the first rapid, verbatim, plus the feed and the sensor offset. Refuses a file missing the variables the emitted blocks refer to."""
    path = Path(sample)
    lines = path.read_text().splitlines()

    preamble = []
    first_rapid = None
    for index, raw in enumerate(lines):
        if re.match(r"G0?0\s+X", _bare(raw)):
            first_rapid = index
            break
        preamble.append(raw)
    if first_rapid is None:
        raise ValueError(f"{path.name}: no rapid move -- is this a cut program?")

    header = "\n".join(_bare(line) for line in preamble)
    feeds = re.findall(r"\bF(\d+\.?\d*)", header)
    if not feeds:
        raise ValueError(f"{path.name}: no feed (F word) in the header")
    defined = {int(n) for n in re.findall(r"#(\d+)\s*=", header)}
    missing = [n for n in REQUIRED_VARIABLES if n not in defined]
    if missing:
        wanted = ", ".join(f"#{n}" for n in missing)
        raise ValueError(f"{path.name}: header does not set {wanted}")

    # The sensor offset is the first rapid relative to the first pierce.
    rapid = re.search(XY, _bare(lines[first_rapid]))
    pierce = None
    for raw in lines[first_rapid + 1:first_rapid + 8]:
        pierce = re.search(r"Z#100\s+" + XY, _bare(raw))
        if pierce:
            break
    if rapid is None or pierce is None:
        raise ValueError(f"{path.name}: no 'Z#100 X Y' pierce follows the first rapid")
    offset = (float(rapid.group(1)) - float(pierce.group(1)),
              float(rapid.group(2)) - float(pierce.group(2)))

    retract = re.search(r"\bM0?5\b[^\n]*\n\s*G0?0\s+Z(\d+\.?\d*)",
                        "\n".join(_bare(line) for line in lines))

    return Machine(
        preamble=tuple(preamble),
        feed_mm_min=float(feeds[-1]),
        sense_offset_mm=offset,
        retract_mm=float(retract.group(1)) if retract else 20.0,
        kerf_mm=kerf_mm,
    )


def _bare(line):
    """The line without comments or margins."""
    return re.sub(r"\(.*?\)", "", line).strip()
