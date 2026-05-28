"""Roofline model: combines hardware config and workload to predict cycles.

The model produces three estimates per layer:

1. Roofline lower bound
   Assumes the right kernel (vectorised or scalar) is used.
   min(compute_ceiling, memory_ceiling × arithmetic_intensity)

2. Memory-bound ceiling
   Assuming 100 % of data accesses are served at the LSU bus rate.

3. Compute-bound ceiling
   Assuming 100 % MAC unit utilisation with no memory stalls.

Additionally, the model flags whether each layer is compute-bound or
memory-bound under the roofline assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .hardware import CoralNPUConfig
from .workload import LayerSpec, WorkloadSpec


@dataclass
class LayerResult:
    layer: LayerSpec

    # Roofline bounds
    compute_ceiling_macs_cycle: float   # peak for this layer's dtype
    memory_ceiling_macs_cycle: float    # bandwidth × AI
    roofline_macs_cycle: float          # min of the two
    roofline_cycles: float              # total_macs / roofline_macs_cycle

    # Individual components
    arithmetic_intensity: float         # MACs/byte
    is_compute_bound: bool

    # Realistic estimate (accounts for LMUL latency, scalar overhead)
    realistic_cycles: float

    # RTL validation (filled in later from sim output)
    rtl_cycles: Optional[float] = None

    @property
    def efficiency(self) -> Optional[float]:
        """roofline_cycles / rtl_cycles — 1.0 = perfect, <1 = impossible."""
        if self.rtl_cycles is None:
            return None
        return self.roofline_cycles / self.rtl_cycles

    @property
    def rtl_throughput(self) -> Optional[float]:
        """Actual MACs/cycle from RTL sim."""
        if self.rtl_cycles is None:
            return None
        return self.layer.total_macs / self.rtl_cycles

    @property
    def roofline_ms(self) -> float:
        return NotImplemented  # filled by RooflineModel with clock

    def summary_row(self) -> str:
        ei = f"{self.efficiency:.1%}" if self.efficiency is not None else "—"
        rtl = f"{self.rtl_cycles/1e6:.2f}M" if self.rtl_cycles else "—"
        return (
            f"  {self.layer.name:<35s}"
            f"  AI={self.arithmetic_intensity:.2f}"
            f"  {'COMPUTE' if self.is_compute_bound else 'MEMORY ':7s}"
            f"  roofline={self.roofline_cycles/1e6:.2f}M"
            f"  realistic={self.realistic_cycles/1e6:.2f}M"
            f"  RTL={rtl}"
            f"  eff={ei}"
        )


@dataclass
class WorkloadResult:
    workload: WorkloadSpec
    hw: CoralNPUConfig
    layers: List[LayerResult] = field(default_factory=list)

    # RTL total cycles for the full workload (filled from sim output)
    rtl_total_cycles: Optional[float] = None

    # Framework overhead cycles (TFLM interpreter, allocate tensors, etc.)
    framework_overhead_cycles: float = 3_000_000   # analytical estimate

    @property
    def total_roofline_cycles(self) -> float:
        return sum(r.roofline_cycles for r in self.layers) + self.framework_overhead_cycles

    @property
    def total_realistic_cycles(self) -> float:
        return sum(r.realistic_cycles for r in self.layers) + self.framework_overhead_cycles

    @property
    def overall_efficiency(self) -> Optional[float]:
        if self.rtl_total_cycles is None:
            return None
        return self.total_roofline_cycles / self.rtl_total_cycles

    def summary(self) -> str:
        lines = [
            f"=== Roofline Analysis: {self.workload.name} on {self.hw.name} ===",
            "",
        ]
        for r in self.layers:
            lines.append(r.summary_row())
        lines += [
            "",
            f"  Framework overhead (est.)  : {self.framework_overhead_cycles/1e6:.2f}M cycles",
            f"  Total roofline lower bound : {self.total_roofline_cycles/1e6:.2f}M cycles",
            f"  Total realistic estimate   : {self.total_realistic_cycles/1e6:.2f}M cycles",
        ]
        if self.rtl_total_cycles is not None:
            eff = self.overall_efficiency
            lines += [
                f"  RTL simulation             : {self.rtl_total_cycles/1e6:.2f}M cycles",
                f"  Overall efficiency         : {eff:.1%}",
                f"  Overhead factor            : {1/eff:.1f}×  (RTL / roofline)",
            ]
        else:
            lines.append("  RTL simulation             : (not yet available)")
        return "\n".join(lines)


class RooflineModel:
    """Computes roofline predictions for a workload on given hardware."""

    def __init__(self, hw: CoralNPUConfig):
        self.hw = hw

    def analyse_layer(self, layer: LayerSpec) -> LayerResult:
        hw = self.hw

        # Determine which compute ceiling applies
        if layer.op_type == "scalar_conv2d":
            # Standard TFLM reference implementation: no RVV, pure scalar
            compute_ceil = hw.scalar_macs_per_cycle
            lmul = 1
        else:
            # Custom RVV kernel
            dtype = layer.compute_dtype  # "int16" for DWConv (int16→int32 vwmacc)
            compute_ceil = hw.peak_macs_per_cycle(dtype)
            lmul = 8  # DWConv kernel uses e32m8 / e16m4

        ai = layer.arithmetic_intensity
        mem_ceil = hw.lsu_bw_bytes_per_cycle * ai
        roofline = min(compute_ceil, mem_ceil)
        roofline_cycles = layer.total_macs / roofline

        is_compute_bound = compute_ceil <= mem_ceil

        # ── Realistic estimate ───────────────────────────────────────────────
        # Accounts for:
        #   - LMUL pipeline serialisation (chained accumulation loop)
        #   - Scalar overhead: load latency, pointer arithmetic
        #   - Quantisation post-processing per output element
        if layer.op_type == "scalar_conv2d":
            # Scalar: ~4–8 instructions/MAC, realistic CPI 0.5
            realistic_macs_per_cycle = hw.scalar_macs_per_cycle * 0.8
            quant_overhead = layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3] * 15
            realistic_cycles = layer.total_macs / realistic_macs_per_cycle + quant_overhead
        else:
            # RVV depthwise: LMUL=8 accumulation chain limits to chained throughput
            chained = hw.chained_macs_per_cycle(layer.compute_dtype, lmul=lmul)
            # The kernel's Reuse6 optimisation amortises the accumulation latency
            # by processing 6 output pixels in parallel → ~3× improvement
            reuse_factor = 3.0
            effective_throughput = chained * reuse_factor
            # Quantise/store overhead: ~20 cycles per output pixel pair
            n_pixels = layer.output_shape[1] * layer.output_shape[2]
            quant_overhead = n_pixels * 20
            realistic_cycles = layer.total_macs / effective_throughput + quant_overhead

        return LayerResult(
            layer=layer,
            compute_ceiling_macs_cycle=compute_ceil,
            memory_ceiling_macs_cycle=mem_ceil,
            roofline_macs_cycle=roofline,
            roofline_cycles=roofline_cycles,
            arithmetic_intensity=ai,
            is_compute_bound=is_compute_bound,
            realistic_cycles=realistic_cycles,
        )

    def analyse(
        self,
        workload: WorkloadSpec,
        rtl_layer_cycles: Optional[List[float]] = None,
        rtl_total_cycles: Optional[float] = None,
        framework_overhead_cycles: float = 3_000_000,
    ) -> WorkloadResult:
        """Run the full roofline analysis.

        Args:
            workload:               the workload to analyse
            rtl_layer_cycles:       per-layer RTL cycle counts (optional)
            rtl_total_cycles:       total RTL cycle count for the full workload
            framework_overhead_cycles: cycles consumed by TFLite Micro framework
        """
        layer_results = []
        for i, layer in enumerate(workload.layers):
            result = self.analyse_layer(layer)
            if rtl_layer_cycles and i < len(rtl_layer_cycles):
                result.rtl_cycles = rtl_layer_cycles[i]
            layer_results.append(result)

        wr = WorkloadResult(
            workload=workload,
            hw=self.hw,
            layers=layer_results,
            rtl_total_cycles=rtl_total_cycles,
            framework_overhead_cycles=framework_overhead_cycles,
        )
        return wr
