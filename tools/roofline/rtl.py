"""Parse RTL simulation output to extract cycle counts.

The rvv_core_mini_highmem_axi_sim binary (built in tests/verilator_sim/)
prints to stderr:

    Total simulation cycles: <N>

when the simulation halts normally.  This module parses that output and
also provides helpers to read MCYCLE values written into the ELF's symbol
table (once per-layer instrumentation is added to run_mobilenet.cc).

Usage
-----
  from tools.roofline.rtl import parse_sim_output, SimResult

  result = parse_sim_output("/tmp/sim_cycles_out.txt")
  if result:
      print(f"Total RTL cycles: {result.total_cycles:,}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SimResult:
    """Parsed RTL simulation results."""

    total_cycles: int                       # from "Total simulation cycles: N"
    layer_cycles: List[int] = field(default_factory=list)  # per-layer (if instrumented)
    source_file: str = ""
    extra: dict = field(default_factory=dict)  # invoke_cycles, framework_cycles (cocotb)

    @property
    def total_ms_at_1ghz(self) -> float:
        return self.total_cycles / 1e6  # cycles / (1000 cycles/µs) = µs → /1000 = ms


def parse_sim_output(path: str) -> Optional["SimResult"]:
    """Parse simulation or cocotb test output to extract cycle counts.

    Recognises two output formats:

    1. rvv_core_mini_highmem_axi_sim (C++ standalone sim):
         "Total simulation cycles: <N>"

    2. cocotb_run_mobilenet_v1.py (Python/cocotb test):
         "ROOFLINE_DATA invoke_cycles=<N>"
         "ROOFLINE_DATA framework_cycles=<N>"
         "ROOFLINE_DATA total_cycles=<N>"
         "Total number of execution cycles: <N>"

    Returns None if no recognisable output is found (sim still running).
    """
    text = Path(path).read_text(errors="replace")

    # --- C++ standalone sim ---
    m = re.search(r"Total simulation cycles:\s*(\d+)", text)
    if m:
        total = int(m.group(1))
        # Also look for per-layer cycle markers
        layer_cycles: List[int] = []
        for lm in re.finditer(r"Layer\s+(\d+)\s+cycles:\s*(\d+)", text):
            idx, cyc = int(lm.group(1)), int(lm.group(2))
            while len(layer_cycles) <= idx:
                layer_cycles.append(0)
            layer_cycles[idx] = cyc
        return SimResult(total_cycles=total, layer_cycles=layer_cycles, source_file=path)

    # --- cocotb test with ROOFLINE_DATA markers ---
    total_m = re.search(r"ROOFLINE_DATA total_cycles=(\d+)", text)
    if not total_m:
        total_m = re.search(r"Total number of execution cycles:\s*(\d+)", text)
    if total_m:
        total = int(total_m.group(1))
        invoke_m = re.search(r"ROOFLINE_DATA invoke_cycles=(\d+)", text)
        framework_m = re.search(r"ROOFLINE_DATA framework_cycles=(\d+)", text)
        extra: dict = {}
        if invoke_m:
            extra["invoke_cycles"] = int(invoke_m.group(1))
        if framework_m:
            extra["framework_cycles"] = int(framework_m.group(1))
        return SimResult(
            total_cycles=total,
            layer_cycles=[],
            source_file=path,
            extra=extra,
        )

    return None


def find_latest_sim_output() -> Optional[str]:
    """Return the most recently modified sim output file under /tmp."""
    candidates = list(Path("/tmp").glob("sim_cycles*.txt"))
    candidates += list(Path("/tmp").glob("highmem_sim*.txt"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# ── How to get per-layer cycles ───────────────────────────────────────────────
#
# To add per-layer timing to run_mobilenet.cc:
#
# 1. Add these globals (with .data section alignment for testbench visibility):
#
#    extern "C" {
#      uint32_t inference_cycles_lo = 0;
#      uint32_t inference_cycles_hi = 0;
#      uint32_t layer_cycles[8] = {};   // up to 8 layers
#      uint32_t num_layers_timed = 0;
#    }
#
# 2. Add an inline helper:
#
#    static inline uint64_t rdcycle64() {
#      uint32_t lo, hi;
#      asm volatile(
#        "1: rdcycleh %1\n"
#        "   rdcycle  %0\n"
#        "   rdcycleh t0\n"
#        "   bne %1, t0, 1b\n"
#        : "=r"(lo), "=r"(hi) : : "t0");
#      return ((uint64_t)hi << 32) | lo;
#    }
#
# 3. Wrap each Invoke with:
#
#    uint64_t t0 = rdcycle64();
#    interpreter.Invoke();
#    uint64_t t1 = rdcycle64();
#    inference_cycles_lo = (uint32_t)(t1 - t0);
#    inference_cycles_hi = (uint32_t)((t1 - t0) >> 32);
#
# 4. The C++ testbench (core_mini_axi_tb.cc) reads these symbols after halt:
#
#    auto lo = tb.ReadWordSync("inference_cycles_lo");
#    auto hi = tb.ReadWordSync("inference_cycles_hi");
#    uint64_t total = ((uint64_t)*hi << 32) | *lo;
#    fprintf(stderr, "Total simulation cycles: %llu\n", total);
#
# This gives the pure-inference cycle count (excluding ELF load / framework
# init), which can be compared to the roofline layer sum.
