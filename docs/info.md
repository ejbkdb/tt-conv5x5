## How it works

This chip is one 5×5 convolution layer with **1 input channel and 64 output channels**. Its 64 filters of 5×5 int8 weights (1,600 in total) are **hardwired into the silicon**. They are synthesized into a block of combinational logic (a ROM made of gates), so the filters are fixed forever at tapeout.

For every output pixel the chip computes 64 sums, one per filter f:

```
y[f] = sum over ky in 0..4, kx in 0..4 of  w[f][ky][kx] * x[ky][kx]
```

A grayscale image of H×W pixels becomes an H×W×64 output ("same" padding), e.g. 1048×1048 → 1048×1048×64.

**Streaming the image.** The chip holds a 5×5 window of pixels. The host sends the image as **columns of 5 pixels, each column top to bottom**. Each full column shifts into the window from the right, and the leftmost column drops out. The host processes the image one **row band** (5 image rows) at a time:

- The first byte of a band carries START. That resets the window, so the first 5 columns only fill it.
- From the 5th column on, every column completes a window and starts one output pixel. Sliding one pixel to the right costs only 5 input bytes.
- For output row `oy` of a "same" convolution, the band is rows `oy-2 .. oy+2` of the image, zero-padded by 2 pixels on every side. That is `W + 4` columns, giving W output pixels.

**Compute.** For each window the engine walks filter 0..63 and, within each filter, all 25 taps in order `ky*5 + kx`, one multiply-accumulate per clock. The weight ROM is laid out the same way (address `f*25 + ky*5 + kx`). The datapath is a two-stage pipeline:

1. The counters select a weight from the ROM and a pixel from the window. Both are registered.
2. A signed 8×8 multiplier feeds a 20-bit accumulator. The worst-case sum (25 × 16,384) fits in 20 bits, so every result is exact.

After each filter's 25th tap the sum goes into a result register, which the host reads one byte at a time and then acknowledges. That is one result every 25 clocks, and 64 results per pixel. While the engine works on a window, the next column can already be loaded into the column buffer. When the last tap of a pixel is issued, the column moves into the window in the same clock and the next pixel starts, so pixels run back to back with no idle cycles.

**Backpressure.** If the host hasn't acknowledged the previous result when a new one is ready, the engine waits. Results are never lost or overwritten. A slow host just slows the chip down.

## How to test

**Pins**

| Pin | Name | Direction | Function |
|---|---|---|---|
| `ui[7:0]` | PIX | in | Signed 8-bit pixel |
| `uio[0]` | VALID | in | PIX is accepted on the rising clock edge while VALID is 1 |
| `uio[1]` | START | in | Together with VALID, marks the top pixel of the first column of a new row band |
| `uio[3:2]` | BYTE_SEL | in | 0, 1, 2: byte of the sign-extended 24-bit result on `uo` (0 = least significant). 3: the filter index (0..63) of that result |
| `uio[4]` | ACK | in | High for one clock after reading a result; frees the result register |
| `uio[5]` | IN_RDY | out | The column buffer has room. Only send PIX while IN_RDY is 1 |
| `uio[6]` | ERR | out | Sticky until reset: PIX was sent while IN_RDY was 0 (the byte is dropped), or ACK was sent with no result waiting |
| `uio[7]` | OUT_VALID | out | The result register holds an unread result |
| `uo[7:0]` | RESULT | out | The byte selected by BYTE_SEL |

All inputs are sampled on the rising edge of the clock. Change them while the clock is low.

**Computing one row band**

1. Reset the design (`rst_n` low for a few cycles).
2. Send the band's first pixel with VALID=1 and START=1. Send every following pixel with VALID=1 and START=0, column by column, each column top to bottom. Before each pixel, check that IN_RDY is 1. It drops while the column buffer is full and the engine is still busy with the previous pixel.
3. Whenever OUT_VALID is 1: read BYTE_SEL=0, 1, 2 to assemble a signed 24-bit result (least significant byte first), optionally read BYTE_SEL=3 for its filter index, then pulse ACK for one clock. BYTE_SEL is combinational, so no clock is needed between reads.
4. Results arrive in order: filter 0..63 for the first pixel of the band, then filter 0..63 for the next, and so on. A band of `n` columns gives `n - 4` pixels.
5. Start the next band with START at any time; it doesn't need to wait for the previous band to finish computing.

Steps 2 and 3 run at the same time. The host has to keep reading results, or the engine stalls and IN_RDY stays low. Asserting START also discards a partly received column or a partly loaded window.

**Throughput**: one multiply-accumulate per clock, so 1,600 clocks per output pixel (64 filters × 25 taps). At 25 MHz that is about 15,600 output pixels per second, or about 1.2 minutes of compute for a 1048×1048 image, if the host keeps up. Input is 5 bytes per pixel. Output is 64 results per pixel, one every 25 clocks, so reading out at full speed means about 1,000,000 results per second.

**Reference check**: `test/weights.npy` holds the exact weights in the ROM, shape (64, 5, 5) indexed `[f, ky, kx]`. `weights.reshape(64, 25) @ window.reshape(25)` gives the 64 expected results for one 5×5 window. `test/test.py` contains a host model (`image_stream`) that turns a whole image into the byte stream described above.

## External hardware

None required. The Tiny Tapeout demo board's RP2040 supplies the clock, streams pixels on the input and bidirectional pins, and reads the results.
