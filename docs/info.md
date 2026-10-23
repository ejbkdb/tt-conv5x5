## How it works

This chip computes one output pixel of a 7×7×128 convolution layer with **int8 weights hardwired into the silicon**. The 6,272 weights are not stored in memory. They are synthesized into a block of combinational logic (a ROM made of gates), so the filter is fixed forever at tapeout.

For each output pixel the host streams the 6,272 activations of one 7×7×128 input window, one signed byte per clock. The chip multiplies each activation by the matching weight and accumulates the result:

```
result = sum over ky in 0..6, kx in 0..6, c in 0..127 of  w[ky][kx][c] * x[ky][kx][c]
```

Activations must arrive in the same order as the ROM: **address = (ky·7 + kx)·128 + c**, so the channel index changes fastest. That is the natural order for a channels-last (HWC) image: send all 128 channels of the top-left pixel, then the next pixel to the right, and so on row by row.

The datapath is a two-stage pipeline:

1. The address counter selects a weight from the ROM. The weight and the incoming activation are registered.
2. A signed 8×8 multiplier feeds a 28-bit accumulator. The worst-case sum (6272 × 16384) fits in 28 bits, so the result is always exact.

The final sum is sign-extended to 32 bits and held in a result register, which the host reads one byte at a time.

## How to test

**Pins**

| Pin | Name | Direction | Function |
|---|---|---|---|
| `ui[7:0]` | ACT | in | Signed 8-bit activation |
| `uio[0]` | VALID | in | ACT is accepted on the rising clock edge while VALID is 1 |
| `uio[1]` | START | in | Together with VALID, marks element 0 of a new window |
| `uio[3:2]` | BYTE_SEL | in | Which byte of the 32-bit result appears on `uo` (0 = least significant) |
| `uio[5]` | BUSY | out | A window is being received or is still in the pipeline |
| `uio[6]` | ERR | out | Sticky: VALID was asserted outside a window. Cleared by START |
| `uio[7]` | DONE | out | The result register holds a completed window. Cleared by START |
| `uo[7:0]` | RESULT | out | Byte BYTE_SEL of the signed 32-bit result |

**Computing one output pixel**

1. Reset the design (`rst_n` low for a few cycles).
2. Drive the first activation on ACT with VALID=1 and START=1 for one clock.
3. Drive the remaining 6,271 activations in ROM order with VALID=1 and START=0. VALID may drop for any number of cycles between activations, so the host does not have to keep up with the clock.
4. Lower VALID and wait for DONE, which rises within two clocks of the last activation.
5. Set BYTE_SEL to 0, 1, 2, 3 and read `uo` each time. Assemble the four bytes into a signed 32-bit integer, least significant byte first. BYTE_SEL is combinational, so no clock is needed between reads.

Asserting START at any time abandons the current window and begins a new one. Windows can run back to back with no idle cycles: the previous result stays readable until the next window completes. Extra activations after the 6,272nd are ignored and set ERR.

**Throughput**: one multiply-accumulate per clock, so 6,272 clocks per output pixel. At 25 MHz that is about 4,000 output pixels per second if the host can supply a byte every clock. The RP2040's PIO can do this; plain bit-banging from Python will be much slower but still works.

**Reference check**: `test/weights.npy` holds the exact weights in the ROM. `numpy.dot(weights.reshape(-1), window.reshape(-1))` gives the expected result for a (7, 7, 128) window.

## External hardware

None required. The Tiny Tapeout demo board's RP2040 supplies the clock, streams activations on the input and bidirectional pins, and reads the result.
