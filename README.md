![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# 7x7 Conv, 1→128 Channels, Hardwired Weights

A 7×7 int8 convolution with 1 input channel and 128 output channels, for [Tiny Tapeout](https://tinytapeout.com). The 128 filters (6,272 weights) are synthesized into logic, so they're fixed at tapeout. The host streams a grayscale image in, one column of 7 pixels at a time. For every output pixel the chip returns 128 exact signed 21-bit sums. A 1048×1048 image produces a 1048×1048×128 output.

- [Datasheet: how it works, pinout, and how to test](docs/info.md)

## Layout

| Path | What it is |
|---|---|
| `src/project.v` | Top module `tt_um_ejborcherding_conv7x7`: column buffer, 7×7 window, MAC engine, result handshake |
| `src/weights_rom.v` | **Generated.** The 128×7×7 weight ROM. Do not edit by hand |
| `tools/gen_weights.py` | Writes `src/weights_rom.v` and `test/weights.npy` from your weights |
| `test/test.py` | Cocotb tests, including a host model that tiles an image, checked against a NumPy reference |
| `test/weights.npy` | **Generated.** The same weights, shape (128, 7, 7), used by the tests |
| `info.yaml` | Tiny Tapeout project config (3x2 tiles, 25 MHz) |

## Putting in your own filters

The repo ships with random weights (seed 1). To use real ones, write a Python function that returns an integer array of shape `(128, 7, 7)` indexed `[f, ky, kx]` with values in `[-128, 127]`:

```python
# my_filters.py
import numpy as np

def make_filter():
    w = ...                      # your trained / designed filters
    return np.clip(np.round(w), -128, 127).astype(np.int8)
```

Then regenerate:

```bash
python tools/gen_weights.py --func my_filters:make_filter
```

The generator also accepts PyTorch's `nn.Conv2d(1, 128, 7).weight` layout `(128, 1, 7, 7)` and the TensorFlow/Keras layout `(7, 7, 1, 128)`, and converts them automatically. Other sources:

```bash
python tools/gen_weights.py --seed 42             # different random weights
python tools/gen_weights.py --npy my_filter.npy   # a saved array
```

You can also `from tools.gen_weights import write_rom` and call `write_rom(weights)` from your own script.

Always commit `src/weights_rom.v` and `test/weights.npy` together. The tests compare one against the other.

## Running the tests

```bash
pip install -r test/requirements.txt   # also needs iverilog
cd test
make
```

Ten tests cover single windows, worst-case sums, ROM ordering, sliding along a row band, a small image end to end with "same" padding, a slow host, backpressure, restarting with START, and the error flag. They take about 90 seconds.

## Size

The previous 128→1 version came to about 6,700 Sky130 cells and 0.039 mm² of cell area. This version adds a 49-byte window and a 7-byte column buffer, about 450 flip-flops, plus the window read mux. A generic Yosys synthesis grows from about 8,200 to 9,900 gates. Expect roughly 0.05–0.06 mm² in Sky130, which is about half of a 3x2 slot (about 0.108 mm²). A 2x2 slot is now unlikely to fit.

The GitHub Actions `gds` workflow runs the real place-and-route. It is the final check that the design fits and meets timing at 25 MHz (`CLOCK_PERIOD` 40 ns in `src/config.json`).

## Submitting

1. Push this folder to a public GitHub repo and enable GitHub Actions and Pages (see the [Tiny Tapeout FAQ](https://tinytapeout.com/faq/)).
2. Make sure the `gds`, `docs` and `test` workflows pass.
3. Submit the repo on [app.tinytapeout.com](https://app.tinytapeout.com) for the shuttle you want.

The module name includes `ejborcherding`. Tiny Tapeout asks for your GitHub username there, so rename `tt_um_ejborcherding_conv7x7` in `src/project.v`, `test/tb.v` and `info.yaml` if yours differs.
