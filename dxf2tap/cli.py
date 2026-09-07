"""Command line. tyro builds it from convert's signature, so the annotations stay."""

import sys
from pathlib import Path
from typing import Annotated

import tyro

from .dxf_to_gcode import dxf_to_gcode
from .read_machine import read_machine


def convert(
    input_dxf: Path,
    input_tap: Path,
    output: Annotated[Path, tyro.conf.arg(aliases=["-o"])],
    kerf_mm: float = 0.6,
    lead_mm: float = 3.0,
    clearance_mm: float = 1.5,
    plot: bool = True,
) -> int:
    """Convert a DXF into a SwiftCut plasma program + choose leading sequences.

    Args:
        input_dxf: the drawing, closed contours of true part geometry.
        input_tap: a program the machine has run; its header is copied.
        output: where to write the program; the plot goes beside it as .png.
        kerf_mm: width of the slot the torch removes; 0 if the DXF is already offset.
        lead_mm: wanted lead radius.
        clearance_mm: Metal a pierce must have around it.
        plot: Draw the cutting plan; --no-plot to skip.
    """
    machine = read_machine(input_tap, kerf_mm=kerf_mm)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = output.with_suffix(".png") if plot else None
    report = dxf_to_gcode(
        input_dxf, output, machine, plot_path=figure,
        lead_mm=lead_mm, clearance_mm=clearance_mm, name=output.stem,
    )

    print(f"wrote {output}")
    if figure is not None:
        print(f"wrote {figure}")
    dropped = f", {report['dropped']} dropped as narrower than the kerf" if report["dropped"] else ""
    print(f"  {report['contours']} contours ({report['holes']} holes){dropped}")
    print(f"  {report['arc_leads']} arc leads, {report['straight_leads']} straight,"
          f" {report['leadless']} pierced on the line; smallest {report['min_lead_mm']:.2f} mm")
    print(f"  {report['cut_length_mm']:.0f} mm of cut at {kerf_mm:g} mm kerf")

    if report["leadless"]:
        print(f"  warning: {report['leadless']} contour(s) pierced on the line",
              file=sys.stderr)


def main():
    tyro.cli(convert)


if __name__ == "__main__":
    main()
