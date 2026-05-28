"""CoralNPU hardware parameter definitions for roofline analysis.

All parameters sourced from RTL:
  - hdl/chisel/src/coralnpu/Parameters.scala
  - hdl/verilog/rvv/inc/rvv_backend_define.svh
  - hdl/chisel/src/coralnpu/BUILD (per-target overrides)
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class CoralNPUConfig:
    """Hardware parameters for one CoralNPU configuration.

    Two configs are defined below:
      CORE_MINI_AXI      – default lsuDataBits=128
      CORE_MINI_HIGHMEM  – highmem TCM layout, same compute parameters

    Compute throughput formulas
    ===========================
    For a vwmacc (widening multiply-accumulate) instruction:

      peak_macs_per_cycle = num_mul_units * (vlen_bits / sew_accumulator_bits)

    where sew_accumulator_bits = 2 * sew_input_bits  (widening doubles width).

    The dispatch width (3 µops/cycle) can limit throughput when instructions
    have no data dependencies.  With a dependency chain (accumulation loop),
    throughput is limited by the latency × lmul of each instruction.

    Both are modelled:
      - ``peak_throughput_macs_cycle`` – fully-pipelined / independent-issue
      - ``chained_throughput_macs_cycle`` – RAW-limited accumulation chain
    """

    name: str

    # ── Vector ISA (rvv_backend_define.svh) ──────────────────────────────────
    vlen_bits: int          # VLEN
    num_mul_units: int      # NUM_MUL
    num_alu_units: int      # NUM_ALU
    num_lsu_units: int      # NUM_LSU

    # ── Scalar pipeline ───────────────────────────────────────────────────────
    dispatch_width: int     # NUM_DP_UOP   (µops/cycle)
    retire_width: int       # NUM_RT_UOP   (µops/cycle)
    rob_depth: int          # ROB_DEPTH

    # ── Memory bus ────────────────────────────────────────────────────────────
    lsu_data_bits: int      # AXI/LSU data width – varies by target
    cache_line_bits: int    # L1D cache refill width (usually lsuDataBits)

    # ── Cache / TCM ───────────────────────────────────────────────────────────
    l1d_size_kb: int
    l1i_size_kb: int
    l1d_latency_cycles: int   # L1D hit latency
    tcm_latency_cycles: int   # TCM access on L1D miss (on-chip SRAM)

    # ── Estimated scalar throughput (no RVV, -O3 compiled) ───────────────────
    scalar_macs_per_cycle: float  # effective MACs/cycle for pure-scalar int8 MAC

    # ── Clock ─────────────────────────────────────────────────────────────────
    clock_ghz: float

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def vlenb(self) -> int:
        return self.vlen_bits // 8

    @property
    def lsu_bw_bytes_per_cycle(self) -> float:
        """Sustained LSU bandwidth (bytes/cycle) for a single access stream."""
        return self.lsu_data_bits / 8

    def peak_macs_per_cycle(self, dtype: str = "int8") -> float:
        """Fully-pipelined (no-dependency) peak MAC throughput [MACs/cycle].

        Uses vwmacc (widening): SEW_input → 2×SEW_input accumulator.

        Args:
            dtype: "int8"  → vwmacc with 8-bit inputs,  16-bit accumulator
                   "int16" → vwmacc with 16-bit inputs, 32-bit accumulator
                   "int32" → vmacc  with 32-bit inputs/output (non-widening)
        """
        sew = {"int8": 8, "int16": 16, "int32": 32}[dtype]
        if dtype == "int32":
            # Non-widening vmacc: each unit processes vlen/sew elements/cycle
            return self.num_mul_units * (self.vlen_bits / sew)
        else:
            # Widening vwmacc: accumulator is 2×sew, so elements/unit = vlen/(2*sew)
            return self.num_mul_units * (self.vlen_bits / (2 * sew))

    def chained_macs_per_cycle(self, dtype: str = "int8", lmul: int = 1) -> float:
        """RAW-limited throughput when each vwmacc depends on the previous.

        In an accumulation loop (e.g. computing a dot product), each
        vwmacc_vv reads and writes the same accumulator register group.
        The next instruction cannot start until the previous completes.

        With the 3-stage MUL pipeline (from doc/microarch/mlu.md) and
        an LMUL-wide instruction, the effective throughput is:

          elements per instruction / (pipeline_latency * lmul) cycles

        Args:
            lmul: vector length multiplier used in the kernel
        """
        mul_pipeline_latency = 3  # from doc/microarch/mlu.md
        elements_per_instr = self.peak_macs_per_cycle(dtype) / self.num_mul_units
        return elements_per_instr / (mul_pipeline_latency * lmul)

    def ridge_point(self, dtype: str = "int8") -> float:
        """Arithmetic intensity [MACs/byte] at the roofline ridge point."""
        return self.peak_macs_per_cycle(dtype) / self.lsu_bw_bytes_per_cycle

    def attainable_macs_cycle(
        self, arithmetic_intensity: float, dtype: str = "int8"
    ) -> float:
        """min(compute_ceiling, memory_ceiling × AI) [MACs/cycle]."""
        mem_bound = arithmetic_intensity * self.lsu_bw_bytes_per_cycle
        return min(self.peak_macs_per_cycle(dtype), mem_bound)

    def roofline_cycles(
        self, total_macs: int, bytes_accessed: int, dtype: str = "int8"
    ) -> float:
        """Lower-bound cycle estimate from the roofline model."""
        ai = total_macs / bytes_accessed
        throughput = self.attainable_macs_cycle(ai, dtype)
        return total_macs / throughput

    def scalar_cycles(self, total_macs: int) -> float:
        """Cycle estimate for fully-scalar (non-vectorised) execution."""
        return total_macs / self.scalar_macs_per_cycle

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  VLEN         : {self.vlen_bits} bits",
            f"  MUL units    : {self.num_mul_units}",
            f"  ALU units    : {self.num_alu_units}",
            f"  LSU data bus : {self.lsu_data_bits} bits = {self.lsu_bw_bytes_per_cycle} B/cycle",
            f"  L1D cache    : {self.l1d_size_kb} KB",
            f"  Clock        : {self.clock_ghz} GHz",
            "",
            f"  Peak int8  vwmacc : {self.peak_macs_per_cycle('int8'):.1f} MACs/cycle",
            f"  Peak int16 vwmacc : {self.peak_macs_per_cycle('int16'):.1f} MACs/cycle",
            f"  Peak int32 vmacc  : {self.peak_macs_per_cycle('int32'):.1f} MACs/cycle",
            f"  Ridge point (int8): {self.ridge_point('int8'):.2f} MACs/byte",
            f"  Scalar MAC        : {self.scalar_macs_per_cycle:.2f} MACs/cycle",
        ]
        return "\n".join(lines)


# ── Canonical configs ──────────────────────────────────────────────────────────

# RvvCoreMiniAxi / RvvCoreMiniHighmemAxi
# BUILD flag: --lsuDataBits=128
# Parameters: VRvvCoreMiniHighmemAxi_parameters.h
CORAL_CORE_MINI = CoralNPUConfig(
    name="RvvCoreMiniHighmemAxi",
    vlen_bits=128,
    num_mul_units=2,
    num_alu_units=2,
    num_lsu_units=2,
    dispatch_width=3,
    retire_width=4,
    rob_depth=8,
    lsu_data_bits=128,
    cache_line_bits=256,  # L1D refill from external AXI is 256-bit
    l1d_size_kb=16,
    l1i_size_kb=8,
    l1d_latency_cycles=2,
    tcm_latency_cycles=5,
    scalar_macs_per_cycle=0.25,  # conservative: ~4 scalar instr/MAC, 1 CPI
    clock_ghz=1.0,
)

# Hypothetical wider-bus variant (lsuDataBits=256, Parameters.scala default)
CORAL_CORE_WIDE = CoralNPUConfig(
    name="RvvCoreMiniWide (lsuDataBits=256)",
    vlen_bits=128,
    num_mul_units=2,
    num_alu_units=2,
    num_lsu_units=2,
    dispatch_width=3,
    retire_width=4,
    rob_depth=8,
    lsu_data_bits=256,
    cache_line_bits=256,
    l1d_size_kb=16,
    l1i_size_kb=8,
    l1d_latency_cycles=2,
    tcm_latency_cycles=5,
    scalar_macs_per_cycle=0.25,
    clock_ghz=1.0,
)

# Technology scaling helpers
def scale_to_node(config: CoralNPUConfig, clock_ghz: float) -> CoralNPUConfig:
    """Return a copy with a different clock (technology node scaling)."""
    return CoralNPUConfig(
        **{**config.__dict__, "clock_ghz": clock_ghz,
           "name": f"{config.name} @ {clock_ghz}GHz"}
    )


ALL_CONFIGS: Dict[str, CoralNPUConfig] = {
    "core_mini": CORAL_CORE_MINI,
    "core_wide": CORAL_CORE_WIDE,
}
