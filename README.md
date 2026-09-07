# dxf2tap

A small and quick code generator for turning dxf drawings into the gcode files (.tap).

The generator parses the dxf file, reads the sample gcode for header, converts the dxf to gcode commands with the lead-in and lead-out automatically selected. A plot for the cutting plan will be generated alongside.

The project is developed for my personal use that involves the plasma cutter at ETH Student Project House Metal Studio.
Your mileage may vary on a different machine.

# Usage

Install the python environment (managed using uv here).
```
uv sync
```

Then, prepare your dxf file, and a .tap sample file from which we would read the header for the new gcode. For the generated gcode to work properly on your target machine, it is recommended to get the sample gcode by using the CAM software known to work with your machine.

```
uv run dxf2tap --input-tap data/sample.tap --input-dxf data/drawing.dxf -o output/target.tap
```

The program generates `output/target.tap` and its plot beside it as `output/target.png`. `uv run dxf2tap --help` lists the few knobs: `--kerf-mm` (0.6 by default, 0 for a DXF that already carries the offset), `--lead-mm`, `--clearance-mm`, and `--no-plot`.

# License

MIT, see [LICENSE](LICENSE).
