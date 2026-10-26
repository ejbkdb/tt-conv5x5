![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# 5x5 Conv, 1→64 Channels, Hardwired Weights

A 5×5 int8 convolution with 1 input channel and 64 output channels, for [Tiny Tapeout](https://tinytapeout.com). The 64 filters (1,600 weights) are synthesized into logic, so they're fixed at tapeout. The host streams a grayscale image in, one column of 5 pixels at a time. For every output pixel the chip returns 64 exact signed 20-bit sums. A 1048×1048 image produces a 1048×1048×64 output.

- [Datasheet: how it works, pinout, and how to test](docs/info.md)

## Layout

| Path | What it is |
|---|---|
| `src/project.v` | Top module `tt_um_ejborcherding_conv7x7`: column buffer, pixel and weight rings, MAC engine, result handshake |
| `src/weights_rom.v` | **Generated.** The weight ROM: one 200-bit word (25 weights) per filter. Do not edit by hand |
| `tools/gen_weights.py` | Writes `src/weights_rom.v` and `test/weights.npy` from your weights |
| `test/test.py` | Cocotb tests, including a host model that tiles an image, checked against a NumPy reference |
| `test/weights.npy` | **Generated.** The same weights, shape (64, 5, 5), used by the tests |
| `info.yaml` | Tiny Tapeout project config (3x2 tiles, 25 MHz) |

## Putting in your own filters

The repo ships with random weights (seed 1). To use real ones, write a Python function that returns an integer array of shape `(64, 5, 5)` indexed `[f, ky, kx]` with values in `[-128, 127]`:

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

The generator also accepts PyTorch's `nn.Conv2d(1, 64, 5).weight` layout `(64, 1, 5, 5)` and the TensorFlow/Keras layout `(5, 5, 1, 64)`, and converts them automatically. Other sources:

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

Ten tests cover single windows, worst-case sums, ROM ordering, sliding along a row band, a small image end to end with "same" padding, a slow host, backpressure, restarting with START, and the error flag. They take about 15 seconds.

## Size

A 7×7×128 version of this design reached 83% utilization of a 3x2 slot after timing repair and did not route (hundreds of thousands of detailed-routing violations). A 7×7×64 version reached 65% utilization but needed about 132% of the available routing tracks. This 5×5×64 version reaches only 42% utilization and 87% global-routing demand, but detailed routing still stalls at about 28,500 violations. The limit was wiring, not area: the ROM (one mux tree on an 11-bit address) and the 25:1 window mux each sent hundreds of wires into one spot.

The current datapath removes both. The window and the current filter's weights are two register rings that rotate past a fixed multiplier, and the ROM returns a whole filter per 6-bit address. Results, timing and pins are unchanged.

With that datapath the design routes in a 3x2 tile:

| | |
|---|---|
| Standard cells | 6,145 (532 flip-flops), 44% utilization |
| Global-route demand | 70.5% of available tracks |
| Setup slack at 25 MHz | +20.3 ns (worst corner) |
| Hold slack | +0.005 ns (fast corner, after the SDC's 0.25 ns clock uncertainty) |
| DRC / LVS | 0 / 0 |
| Antenna | 1 net |
| Max-slew warnings | 1,924, mostly at the slow corner |
| TT precheck | passes |
| Gate-level tests | 10/10 pass on the routed netlist |

Routing needs two settings in the editable part of `src/config.json`:
- **Hold margins set to 0.** With the defaults (0.1 ns at placement), the flow packs about 650 hold buffers next to the I/O pins, which all sit on one short stretch of the top edge. That corner then can't route. The hold checks still include the 0.25 ns clock uncertainty from the Tiny Tapeout SDC.
- **Wire repair after global routing**, with repaired wires capped at 250 µm.

The GitHub Actions `gds` workflow runs the same place-and-route and is the final check (`CLOCK_PERIOD` 40 ns in `src/config.json`).

## Submitting

1. Push this folder to a public GitHub repo and enable GitHub Actions and Pages (see the [Tiny Tapeout FAQ](https://tinytapeout.com/faq/)).
2. Make sure the `gds`, `docs` and `test` workflows pass.
3. Submit the repo on [app.tinytapeout.com](https://app.tinytapeout.com) for the shuttle you want.

The module name includes `ejborcherding`. Tiny Tapeout asks for your GitHub username there, so rename `tt_um_ejborcherding_conv7x7` in `src/project.v`, `test/tb.v` and `info.yaml` if yours differs. (The name still says 7x7 from the original design; the kernel is now 5×5.)
