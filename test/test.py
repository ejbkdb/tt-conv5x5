# SPDX-FileCopyrightText: © 2026 EJ Borcherding
# SPDX-License-Identifier: Apache-2.0
#
# Tests for tt_um_ejborcherding_conv7x7. Checks the chip against a NumPy
# reference using test/weights.npy, which tools/gen_weights.py writes alongside
# src/weights_rom.v. Only top-level pins are used, so the same tests also run
# on the gate-level netlist.

import os
import random

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer

N = 7 * 7 * 128
VALID, START = 0x01, 0x02
BUSY, ERR, DONE = 0x20, 0x40, 0x80

WEIGHTS = np.load(os.path.join(os.path.dirname(__file__), "weights.npy")).astype(np.int64)
W_FLAT = WEIGHTS.reshape(-1)  # ROM order: (ky*7 + kx)*128 + c
assert W_FLAT.size == N


def reference(x):
    return int(np.dot(W_FLAT, np.asarray(x, dtype=np.int64)))


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
    """Read the signed 32-bit result through BYTE_SEL, with VALID low."""
    value = 0
    for sel in range(4):
        dut.uio_in.value = sel << 2
        await Timer(1, unit="ns")
        value |= int(dut.uo_out.value) << (8 * sel)
    dut.uio_in.value = 0
    return value - (1 << 32) if value & (1 << 31) else value


async def feed(dut, x, gap_prob=0.0, on_index=None):
    """Stream one window. Inputs change on falling edges; the chip samples on rising edges."""
    for i, v in enumerate(x):
        while gap_prob and random.random() < gap_prob:
            dut.uio_in.value = 0
            await FallingEdge(dut.clk)
        dut.ui_in.value = int(v) & 0xFF
        dut.uio_in.value = VALID | (START if i == 0 else 0)
        await FallingEdge(dut.clk)
        if on_index is not None:
            await on_index(i)
    dut.uio_in.value = 0


async def finish(dut):
    """Wait for the pipeline to drain and return the result."""
    for _ in range(4):
        await FallingEdge(dut.clk)
    assert flags(dut) & DONE, "DONE not set after a full window"
    assert not flags(dut) & BUSY, "BUSY still set after the window drained"
    assert not flags(dut) & ERR, "ERR set during a well-formed window"
    return await read_result(dut)


def rand_window(rng):
    return rng.integers(-128, 128, size=N)


@cocotb.test()
async def test_reset_state(dut):
    await setup(dut)
    assert int(dut.uio_oe.value) == 0xE0
    assert flags(dut) == 0, f"flags after reset: {flags(dut):#04x}"
    assert await read_result(dut) == 0


@cocotb.test()
async def test_random_window(dut):
    await setup(dut)
    rng = np.random.default_rng(10)
    for _ in range(2):
        x = rand_window(rng)
        await feed(dut, x)
        got = await finish(dut)
        assert got == reference(x), f"got {got}, expected {reference(x)}"


@cocotb.test()
async def test_extremes(dut):
    """Largest-magnitude sums, to check the accumulator width and sign handling."""
    await setup(dut)
    cases = {
        "max positive": np.where(W_FLAT < 0, -128, 127),
        "max negative": np.where(W_FLAT < 0, 127, -128),
        "all -128": np.full(N, -128),
        "all zero": np.zeros(N, dtype=np.int64),
    }
    for name, x in cases.items():
        await feed(dut, x)
        got = await finish(dut)
        assert got == reference(x), f"{name}: got {got}, expected {reference(x)}"


@cocotb.test()
async def test_unit_vectors(dut):
    """A single 1 at address i must return exactly weight i (checks ROM ordering)."""
    await setup(dut)
    for addr in [0, 1, 127, 128, 7 * 128, N - 1]:
        x = np.zeros(N, dtype=np.int64)
        x[addr] = 1
        await feed(dut, x)
        got = await finish(dut)
        assert got == int(W_FLAT[addr]), f"addr {addr}: got {got}, expected {W_FLAT[addr]}"


@cocotb.test()
async def test_gaps(dut):
    """VALID may drop for any number of cycles mid-window."""
    await setup(dut)
    random.seed(3)
    x = rand_window(np.random.default_rng(11))
    await feed(dut, x, gap_prob=0.3)
    got = await finish(dut)
    assert got == reference(x)


@cocotb.test()
async def test_back_to_back(dut):
    """Window B starts the cycle after window A ends; A's result is readable during B."""
    await setup(dut)
    rng = np.random.default_rng(12)
    a, b = rand_window(rng), rand_window(rng)
    await feed(dut, a)
    seen = {}

    async def peek(i):
        # Read A's result a few cycles into B by changing only BYTE_SEL.
        if 4 <= i < 8:
            sel = i - 4
            keep = int(dut.uio_in.value) & 0x03
            dut.uio_in.value = keep | (sel << 2)
            await Timer(1, unit="ns")
            seen[sel] = int(dut.uo_out.value)
            seen["flags"] = flags(dut)

    await feed(dut, b, on_index=peek)
    a_val = sum(seen[s] << (8 * s) for s in range(4))
    a_val = a_val - (1 << 32) if a_val & (1 << 31) else a_val
    assert a_val == reference(a), f"window A read during B: got {a_val}, expected {reference(a)}"
    assert seen["flags"] & BUSY
    got = await finish(dut)
    assert got == reference(b)


@cocotb.test()
async def test_restart(dut):
    """START in the middle of a window abandons it and begins a new one."""
    await setup(dut)
    rng = np.random.default_rng(13)
    await feed(dut, rand_window(rng)[:1000])
    assert flags(dut) & BUSY and not flags(dut) & DONE
    x = rand_window(rng)
    await feed(dut, x)
    got = await finish(dut)
    assert got == reference(x)


@cocotb.test()
async def test_error_flag(dut):
    """VALID without START outside a window sets ERR and is ignored; START clears it."""
    await setup(dut)
    dut.ui_in.value = 0x55
    dut.uio_in.value = VALID
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await FallingEdge(dut.clk)
    assert flags(dut) & ERR, "ERR not set by a stray VALID"
    assert not flags(dut) & BUSY, "stray VALID must not start a window"

    x = rand_window(np.random.default_rng(14))
    await feed(dut, x)
    got = await finish(dut)  # also checks ERR was cleared by START
    assert got == reference(x)

    # extra samples after the window has finished are rejected too
    dut.uio_in.value = VALID
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    await FallingEdge(dut.clk)
    assert flags(dut) & ERR
    assert await read_result(dut) == reference(x), "stray VALID changed the result"
