# SPDX-FileCopyrightText: © 2026 EJ Borcherding
# SPDX-License-Identifier: Apache-2.0
#
# Tests for tt_um_ejborcherding_conv7x7: a 7x7 convolution with 1 input channel
# and 128 output channels. Checks the chip against a NumPy reference using
# test/weights.npy, which tools/gen_weights.py writes alongside
# src/weights_rom.v. Only top-level pins are used, so the same tests also run
# on the gate-level netlist.

import os
import random

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, First, RisingEdge, Timer

NF, K = 128, 7
VALID, START, ACK = 0x01, 0x02, 0x10
IN_RDY, ERR, OUT_VALID = 0x20, 0x40, 0x80

W = np.load(os.path.join(os.path.dirname(__file__), "weights.npy")).astype(np.int64)
assert W.shape == (NF, K, K), W.shape  # w[f, ky, kx]; ROM address f*49 + ky*7 + kx
W_FLAT = W.reshape(NF, K * K)


# ------------------------------------------------------------ reference model

def ref_window(x):
    """The 128 filter outputs for one 7x7 window."""
    return W_FLAT @ np.asarray(x, dtype=np.int64).reshape(K * K)


def ref_band(band):
    """Outputs for a (7, n) row band: one pixel per window position, shape (n-6, 128)."""
    band = np.asarray(band)
    return np.array([ref_window(band[:, ox:ox + K]) for ox in range(band.shape[1] - K + 1)])


def ref_image(img):
    """'Same' convolution of an (H, W) image with zero padding: shape (H, W, 128)."""
    img = np.asarray(img, dtype=np.int64)
    p = np.pad(img, K // 2)
    out = np.zeros(img.shape + (NF,), dtype=np.int64)
    for ky in range(K):
        for kx in range(K):
            win = p[ky:ky + img.shape[0], kx:kx + img.shape[1]]
            out += win[:, :, None] * W[None, None, :, ky, kx]
    return out


# ------------------------------------------------------------ host side model

def band_stream(band):
    """Bytes for a (7, n) row band: columns left to right, each column top to
    bottom, with START on the very first byte. Returns [(byte, start), ...]."""
    band = np.asarray(band)
    assert band.shape[0] == K
    return [(int(band[ky, kx]), kx == 0 and ky == 0)
            for kx in range(band.shape[1]) for ky in range(K)]


def image_stream(img):
    """Bytes for a whole image with 'same' padding: one row band per output row."""
    p = np.pad(np.asarray(img), K // 2)
    return [b for oy in range(img.shape[0]) for b in band_stream(p[oy:oy + K])]


def flags(dut):
    return int(dut.uio_out.value) & 0xE0


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)


async def read_result(dut):
    """Read the result register through BYTE_SEL: returns (filter index, value)."""
    b = []
    for sel in range(4):
        dut.uio_in.value = sel << 2
        await Timer(1, unit="ns")
        b.append(int(dut.uo_out.value))
    value = b[0] | b[1] << 8 | b[2] << 16
    if value & 0x800000:
        value -= 1 << 24
    return b[3], value


async def run_host(dut, stream, n_results, read=True, feed_gap=0.0, ack_gap=0.0):
    """Act as the host. Feed `stream` while IN_RDY is high, and read and ACK
    every result until `n_results` have arrived. Inputs change on falling edges;
    the chip samples on rising edges. feed_gap / ack_gap are the per-cycle odds
    of the host skipping a cycle, to model a slow host. With no gaps, idle
    stretches are skipped by waiting on IN_RDY / OUT_VALID edges.
    Returns [(filter, value), ...] in arrival order."""
    results = []
    i = 0
    while i < len(stream) or len(results) < n_results:
        fl = flags(dut)
        can_feed = i < len(stream) and fl & IN_RDY
        has_out = read and fl & OUT_VALID
        if not (can_feed or has_out) and not (feed_gap or ack_gap):
            dut.uio_in.value = 0
            timeout = Timer(2, unit="ms")
            fired = await First(RisingEdge(dut.in_rdy), RisingEdge(dut.out_valid), timeout)
            assert fired is not timeout, (
                f"host stuck: fed {i}/{len(stream)} bytes, read {len(results)}/{n_results} results, "
                f"flags {flags(dut):#04x}")
            await FallingEdge(dut.clk)
            continue

        drive = 0
        if has_out and not (ack_gap and random.random() < ack_gap):
            results.append(await read_result(dut))
            drive |= ACK
        if can_feed and not (feed_gap and random.random() < feed_gap):
            byte, start = stream[i]
            i += 1
            dut.ui_in.value = byte & 0xFF
            drive |= VALID | (START if start else 0)
        dut.uio_in.value = drive
        await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    return results


def check(results, expected):
    """Compare host results with expected outputs of shape (..., 128)."""
    expected = np.asarray(expected).reshape(-1, NF)
    assert len(results) == expected.size, f"got {len(results)} results, expected {expected.size}"
    filts = [f for f, _ in results]
    assert filts == list(range(NF)) * len(expected), "results not in filter order 0..127 per pixel"
    got = np.array([v for _, v in results]).reshape(-1, NF)
    bad = np.argwhere(got != expected)
    if bad.size:
        p, f = bad[0]
        raise AssertionError(f"{len(bad)} mismatches; first at pixel {p} filter {f}: "
                             f"got {got[p, f]}, expected {expected[p, f]}")


async def expect_idle(dut):
    """No stray results and no error once all work is done."""
    await ClockCycles(dut.clk, 3 * K * K)
    await FallingEdge(dut.clk)
    assert flags(dut) == IN_RDY, f"flags after the run: {flags(dut):#04x}"


def rand_pixels(rng, shape):
    return rng.integers(-128, 128, size=shape)


# ---------------------------------------------------------------------- tests

@cocotb.test()
async def test_reset_state(dut):
    await setup(dut)
    assert int(dut.uio_oe.value) == 0xE0
    assert flags(dut) == IN_RDY, f"flags after reset: {flags(dut):#04x}"


@cocotb.test()
async def test_single_window(dut):
    await setup(dut)
    x = rand_pixels(np.random.default_rng(10), (K, K))
    results = await run_host(dut, band_stream(x), NF)
    check(results, ref_window(x))
    await expect_idle(dut)


@cocotb.test()
async def test_extremes(dut):
    """Largest-magnitude sums, to check the accumulator width and sign handling.
    Each case is its own START-framed window, sent back to back."""
    await setup(dut)
    cases = [
        np.where(W[0] < 0, -128, 127),  # max positive for filter 0
        np.where(W[0] < 0, 127, -128),  # max negative for filter 0
        np.full((K, K), -128),
        np.full((K, K), 127),
        np.zeros((K, K), dtype=np.int64),
    ]
    stream = [b for x in cases for b in band_stream(x)]
    results = await run_host(dut, stream, NF * len(cases))
    check(results, [ref_window(x) for x in cases])
    await expect_idle(dut)


@cocotb.test()
async def test_unit_vectors(dut):
    """A single 1 at tap (ky, kx) must return w[f, ky, kx] for every filter
    (checks ROM order and where each byte lands in the window)."""
    await setup(dut)
    taps = [(0, 0), (0, 6), (1, 0), (3, 3), (6, 0), (6, 6)]
    windows = []
    for ky, kx in taps:
        x = np.zeros((K, K), dtype=np.int64)
        x[ky, kx] = 1
        windows.append(x)
    stream = [b for x in windows for b in band_stream(x)]
    results = await run_host(dut, stream, NF * len(taps))
    check(results, [W[:, ky, kx] for ky, kx in taps])


@cocotb.test()
async def test_sliding_band(dut):
    """After the first 7 columns, every further column of 7 bytes yields one pixel."""
    await setup(dut)
    band = rand_pixels(np.random.default_rng(11), (K, 11))
    stream = band_stream(band)
    assert len(stream) == 7 * 11
    results = await run_host(dut, stream, NF * 5)
    check(results, ref_band(band))
    await expect_idle(dut)


@cocotb.test()
async def test_image(dut):
    """End to end: a small image with 'same' padding, one row band per output row."""
    await setup(dut)
    img = rand_pixels(np.random.default_rng(12), (2, 4))
    expected = ref_image(img)
    results = await run_host(dut, image_stream(img), expected.size // NF * NF)
    check(results, expected)
    await expect_idle(dut)


@cocotb.test()
async def test_slow_host(dut):
    """A host that misses cycles on both sides. Reading slower than one result
    per 49 clocks stalls the engine, which must not lose or corrupt results."""
    await setup(dut)
    random.seed(3)
    band = rand_pixels(np.random.default_rng(13), (K, 8))
    results = await run_host(dut, band_stream(band), NF * 2, feed_gap=0.5, ack_gap=0.985)
    check(results, ref_band(band))
    await expect_idle(dut)


@cocotb.test()
async def test_backpressure(dut):
    """With nobody reading, the engine parks with one result waiting and the
    next column held in the buffer. Reading then resumes it with nothing lost."""
    await setup(dut)
    band = rand_pixels(np.random.default_rng(14), (K, 8))
    await run_host(dut, band_stream(band), 0, read=False)
    await ClockCycles(dut.clk, 3000)
    await FallingEdge(dut.clk)
    assert flags(dut) == OUT_VALID, f"flags while stalled: {flags(dut):#04x}"
    filt, _ = await read_result(dut)
    assert filt == 0
    results = await run_host(dut, [], NF * 2)
    check(results, ref_band(band))
    await expect_idle(dut)


@cocotb.test()
async def test_restart(dut):
    """START discards a partial column and a partly loaded window, and can
    begin a new band right after the previous band's last pixel."""
    await setup(dut)
    rng = np.random.default_rng(15)
    junk = band_stream(rand_pixels(rng, (K, 4)))
    band = rand_pixels(rng, (K, 8))
    window = rand_pixels(rng, (K, K))
    stream = junk[:3] + junk[:7 * 4] + band_stream(band) + band_stream(window)
    results = await run_host(dut, stream, NF * 3)
    check(results, np.concatenate([ref_band(band), ref_window(window)[None]]))
    await expect_idle(dut)


@cocotb.test()
async def test_error_flag(dut):
    """ERR is set by an ACK with no result, and by PIX while IN_RDY is low (the
    byte is dropped). Only reset clears it."""
    await setup(dut)
    dut.uio_in.value = ACK
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await FallingEdge(dut.clk)
    assert flags(dut) & ERR, "ERR not set by an ACK with no result"

    await setup(dut)
    assert not flags(dut) & ERR, "reset did not clear ERR"
    band = rand_pixels(np.random.default_rng(16), (K, 8))
    await run_host(dut, band_stream(band), 0, read=False)
    assert not flags(dut) & IN_RDY
    dut.ui_in.value = 0x55
    dut.uio_in.value = VALID
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await FallingEdge(dut.clk)
    assert flags(dut) & ERR, "ERR not set by PIX while IN_RDY was low"
    results = await run_host(dut, [], NF * 2)
    check(results, ref_band(band))  # the extra byte was dropped
