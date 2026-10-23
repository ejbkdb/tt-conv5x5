![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# 7x7x128 Hardwired Convolution Engine

An int8 7×7×128 convolution for [Tiny Tapeout](https://tinytapeout.com). The 6,272 weights are synthesized into logic, so the filter is fixed at tapeout. The host streams one input window in, one byte per clock, and reads back an exact signed 32-bit sum.

- [Datasheet: how it works, pinout, and how to test](docs/info.md)

## Layout

| Path | What it is |
|---|---|
| `src/project.v` | Top module `tt_um_ejborcherding_conv7x7`: address counter, pipeline, MAC, framing, readout |
| `src/weights_rom.v` | **Generated.** The 7×7×128 weight ROM. Do not edit by hand |
| `tools/gen_weights.py` | Writes `src/weights_rom.v` and `test/weights.npy` from your weights |
| `test/test.py` | Cocotb tests that check the chip against a NumPy reference |
| `test/weights.npy` | **Generated.** The same weights, used by the tests |
| `info.yaml` | Tiny Tapeout project config (3x2 tiles, 25 MHz) |

## Putting in your own filter

The repo ships with random weights (seed 1). To use real ones, write a Python function that returns an integer array of shape `(7, 7, 128)` indexed `[ky, kx, c]` with values in `[-128, 127]`:

```python
# my_filters.py
import numpy as np

def make_filter():
    w = ...                      # your trained / designed filter
    return np.clip(np.round(w), -128, 127).astype(np.int8)
```

Then regenerate:

```bash
python tools/gen_weights.py --func my_filters:make_filter
```

Other sources:

```bash
python tools/gen_weights.py --seed 42             # different random weights
python tools/gen_weights.py --npy my_filter.npy   # a saved array
python tools/gen_weights.py --func my_filters:make_filter --chw   # (128, 7, 7) PyTorch order
```

You can also `from tools.gen_weights import write_rom` and call `write_rom(weights)` from your own script.

Always commit `src/weights_rom.v` and `test/weights.npy` together. The tests compare one against the other.

## Running the tests

```bash
pip install -r test/requirements.txt   # also needs iverilog
cd test
make
```

Eight tests cover random windows, worst-case sums, ROM ordering, gaps in VALID, back-to-back windows, restarting mid-window, and the error flag. They take about 45 seconds.

## Size

Synthesized against the Sky130 HD cell library with random weights: about 6,700 cells and 0.039 mm² of cell area, of which the ROM is most. A 3x2 slot (about 0.108 mm²) leaves comfortable room for routing. A 2x2 slot might fit and would cost less; try it if the 3x2 build passes easily. Real trained weights usually shrink the ROM, because random data is close to the worst case.

The GitHub Actions `gds` workflow runs the real place-and-route. It is the final check that the design fits and meets timing at 25 MHz (`CLOCK_PERIOD` 40 ns in `src/config.json`).

## Submitting

1. Push this folder to a public GitHub repo and enable GitHub Actions and Pages (see the [Tiny Tapeout FAQ](https://tinytapeout.com/faq/)).
2. Make sure the `gds`, `docs` and `test` workflows pass.
3. Submit the repo on [app.tinytapeout.com](https://app.tinytapeout.com) for the shuttle you want.

The module name includes `ejborcherding`. Tiny Tapeout asks for your GitHub username there, so rename `tt_um_ejborcherding_conv7x7` in `src/project.v`, `test/tb.v` and `info.yaml` if yours differs.
