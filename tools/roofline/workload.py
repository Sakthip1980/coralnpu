"""Workload specification: layer shapes, MAC counts, and memory traffic.

Supports:
  - Manual layer definitions
  - Automatic parsing from a .tflite flatbuffer

For each layer we compute:
  - Total MACs
  - Bytes accessed (full read of inputs + weights + bias + write of output)
  - Arithmetic intensity (MACs / bytes_accessed)
  - Whether a custom RVV kernel is registered
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class LayerSpec:
    """One convolutional or depthwise-conv layer."""

    name: str
    op_type: str                # "conv2d" | "depthwise_conv2d" | "scalar_conv2d"

    # Input tensor shape  [batch, height, width, channels]
    input_shape: Tuple[int, int, int, int]
    # Filter tensor shape
    #   Conv2D:          [out_ch, kH, kW, in_ch]
    #   DepthwiseConv2D: [1, kH, kW, in_ch]   (depth_multiplier folded in)
    filter_shape: Tuple[int, int, int, int]
    # Output tensor shape [batch, out_H, out_W, out_ch]
    output_shape: Tuple[int, int, int, int]

    stride_h: int = 1
    stride_w: int = 1
    depth_multiplier: int = 1

    # dtype used for computation (determines which vectorised kernel runs)
    compute_dtype: str = "int8"     # "int8" | "int16" | "scalar"

    # Element sizes in bytes
    input_elem_bytes: int = 1       # int8
    weight_elem_bytes: int = 1      # int8
    bias_elem_bytes: int = 4        # int32
    output_elem_bytes: int = 1      # int8

    # ── Derived ──────────────────────────────────────────────────────────────

    @property
    def total_macs(self) -> int:
        """Total multiply-accumulate operations.

        Conv2D:
          MACs = out_H × out_W × out_C × in_C × kH × kW

        DepthwiseConv2D:
          MACs = out_H × out_W × in_C × kH × kW × depth_multiplier
        """
        B, out_H, out_W, _ = self.output_shape
        _, kH, kW, in_C = self.filter_shape
        _, _, _, out_C = self.output_shape

        if self.op_type in ("conv2d", "scalar_conv2d"):
            return out_H * out_W * out_C * in_C * kH * kW
        else:  # depthwise_conv2d
            return out_H * out_W * in_C * kH * kW * self.depth_multiplier

    @property
    def bytes_input(self) -> int:
        _, H, W, C = self.input_shape
        return H * W * C * self.input_elem_bytes

    @property
    def bytes_weights(self) -> int:
        out_C, kH, kW, in_C = self.filter_shape
        return out_C * kH * kW * in_C * self.weight_elem_bytes

    @property
    def bytes_bias(self) -> int:
        _, _, _, out_C = self.output_shape
        return out_C * self.bias_elem_bytes

    @property
    def bytes_output(self) -> int:
        _, H, W, C = self.output_shape
        return H * W * C * self.output_elem_bytes

    @property
    def bytes_accessed(self) -> int:
        """Total bytes read + written (cold-cache, no reuse)."""
        return (
            self.bytes_input
            + self.bytes_weights
            + self.bytes_bias
            + self.bytes_output
        )

    @property
    def bytes_accessed_with_weight_reuse(self) -> int:
        """Bytes accessed assuming weights fully cached after first load.

        For 1×1 conv over a 56×56 map, the 32×16 = 512-byte weight matrix
        is small enough to sit in L1D registers across all spatial positions.
        """
        _, out_H, out_W, _ = self.output_shape
        spatial = out_H * out_W
        # Input and output stream once; weights loaded once
        return (
            self.bytes_input
            + self.bytes_weights          # loaded once
            + self.bytes_bias
            + self.bytes_output
        )

    @property
    def arithmetic_intensity(self) -> float:
        """MACs per byte (cold-cache)."""
        return self.total_macs / self.bytes_accessed

    @property
    def arithmetic_intensity_cached(self) -> float:
        """MACs per byte (weights cached)."""
        return self.total_macs / self.bytes_accessed_with_weight_reuse

    def summary(self) -> str:
        return (
            f"  {self.name} ({self.op_type})\n"
            f"    input    : {list(self.input_shape)}\n"
            f"    filter   : {list(self.filter_shape)}\n"
            f"    output   : {list(self.output_shape)}\n"
            f"    stride   : {self.stride_h}×{self.stride_w}\n"
            f"    MACs     : {self.total_macs:,}\n"
            f"    bytes    : {self.bytes_accessed:,}  "
            f"(input {self.bytes_input:,} + weights {self.bytes_weights:,} "
            f"+ bias {self.bytes_bias} + output {self.bytes_output:,})\n"
            f"    AI       : {self.arithmetic_intensity:.2f} MACs/byte  "
            f"({self.arithmetic_intensity_cached:.2f} with weight reuse)"
        )


@dataclass
class WorkloadSpec:
    """A full inference workload made up of ordered layers."""

    name: str
    layers: List[LayerSpec] = field(default_factory=list)

    @property
    def total_macs(self) -> int:
        return sum(l.total_macs for l in self.layers)

    @property
    def total_bytes(self) -> int:
        return sum(l.bytes_accessed for l in self.layers)

    @property
    def overall_arithmetic_intensity(self) -> float:
        return self.total_macs / self.total_bytes

    def summary(self) -> str:
        lines = [f"Workload: {self.name}", f"  Layers: {len(self.layers)}"]
        for layer in self.layers:
            lines.append(layer.summary())
        lines += [
            f"  Total MACs  : {self.total_macs:,}",
            f"  Total bytes : {self.total_bytes:,}",
            f"  Overall AI  : {self.overall_arithmetic_intensity:.2f} MACs/byte",
        ]
        return "\n".join(lines)


# ── TFLite flatbuffer parser ───────────────────────────────────────────────────

def _resolve(buf: bytes, ref_pos: int) -> int:
    """Resolve a relative flatbuffer offset reference."""
    rel = struct.unpack_from("<i", buf, ref_pos)[0]
    return ref_pos + rel


def _vtable(buf: bytes, offset: int) -> dict:
    """Return {field_index: absolute_byte_offset} for a flatbuffer table."""
    vt_off = offset - struct.unpack_from("<i", buf, offset)[0]
    vt_size = struct.unpack_from("<H", buf, vt_off)[0]
    fields = {}
    for i in range(2, vt_size // 2):
        fo = struct.unpack_from("<H", buf, vt_off + i * 2)[0]
        if fo != 0:
            fields[i - 2] = offset + fo
    return fields


def _vec(buf: bytes, offset: int) -> Tuple[int, int]:
    """Return (count, data_start_offset) for a flatbuffer vector."""
    n = struct.unpack_from("<I", buf, offset)[0]
    return n, offset + 4


def _read_string(buf: bytes, offset: int) -> str:
    n = struct.unpack_from("<I", buf, offset)[0]
    return buf[offset + 4 : offset + 4 + n].decode("utf-8", errors="replace")


def parse_tflite(path: str) -> WorkloadSpec:
    """Parse a TFLite flatbuffer and build a WorkloadSpec.

    Supports Conv2D (builtin_code=3) and DepthwiseConv2D (builtin_code=4).
    Assumes int8 quantised model (typical for CoralNPU workloads).
    """
    buf = Path(path).read_bytes()
    root = struct.unpack_from("<I", buf, 0)[0]
    model_fields = _vtable(buf, root)

    # Read operator_codes to map index → builtin_code
    op_codes_vec = _resolve(buf, model_fields[1])
    num_op_codes, oc_start = _vec(buf, op_codes_vec)
    builtin_codes = []
    for i in range(num_op_codes):
        oc_off = _resolve(buf, oc_start + i * 4)
        oc_fields = _vtable(buf, oc_off)
        code = -1
        if 3 in oc_fields:   # builtin_code (extended, int32)
            code = struct.unpack_from("<I", buf, oc_fields[3])[0]
        elif 0 in oc_fields:  # deprecated_builtin_code (int8)
            code = struct.unpack_from("<B", buf, oc_fields[0])[0]
        builtin_codes.append(code)

    # Read first subgraph
    sg_vec = _resolve(buf, model_fields[2])
    num_sg, sg_start = _vec(buf, sg_vec)
    sg_off = _resolve(buf, sg_start)
    sg_fields = _vtable(buf, sg_off)

    # Read tensors
    t_vec = _resolve(buf, sg_fields[0])
    num_t, t_start = _vec(buf, t_vec)

    def read_tensor(idx):
        t_off = _resolve(buf, t_start + idx * 4)
        tf = _vtable(buf, t_off)
        shape = []
        if 0 in tf:
            sh_vec = _resolve(buf, tf[0])
            nd, sh_start = _vec(buf, sh_vec)
            shape = [struct.unpack_from("<I", buf, sh_start + d * 4)[0] for d in range(nd)]
        name = _read_string(buf, _resolve(buf, tf[3])) if 3 in tf else ""
        return name, shape

    # Read operators
    op_vec = _resolve(buf, sg_fields[3])
    num_ops, op_start = _vec(buf, op_vec)

    CONV2D = 3
    DW_CONV2D = 4

    layers: List[LayerSpec] = []
    for j in range(num_ops):
        op_off = _resolve(buf, op_start + j * 4)
        op_fields = _vtable(buf, op_off)

        # TFLite Operator field layout (flatbuffer vtable field indices):
        #   0: opcode_index (uint)
        #   1: inputs ([int])
        #   2: outputs ([int])
        #   3: builtin_options_type
        #   4: builtin_options (union [ubyte])
        opcode_idx = struct.unpack_from("<I", buf, op_fields[0])[0] if 0 in op_fields else 0
        builtin_code = builtin_codes[opcode_idx] if opcode_idx < len(builtin_codes) else -1

        in_ids, out_ids = [], []
        if 1 in op_fields:
            in_vec = _resolve(buf, op_fields[1])
            n_in, in_start = _vec(buf, in_vec)
            in_ids = [struct.unpack_from("<i", buf, in_start + k * 4)[0] for k in range(n_in)]
            in_ids = [i for i in in_ids if i >= 0]  # -1 = optional/missing tensor

        if 2 in op_fields:
            out_vec = _resolve(buf, op_fields[2])
            n_out, out_start = _vec(buf, out_vec)
            out_ids = [struct.unpack_from("<I", buf, out_start + k * 4)[0] for k in range(n_out)]

        # Read shapes
        input_name, input_shape = read_tensor(in_ids[0])
        filter_name, filter_shape = read_tensor(in_ids[1])
        output_name, output_shape = read_tensor(out_ids[0])

        # Parse convolution options (stride, padding, depth_multiplier)
        # builtin_options is field 4 (union value stored as offset)
        stride_h = stride_w = 1
        depth_mul = 1
        if 4 in op_fields:
            try:
                opt_off = _resolve(buf, op_fields[4])
                opt_f = _vtable(buf, opt_off)
                # Conv2DOptions / DepthwiseConv2DOptions:
                #   0: padding, 1: stride_w, 2: stride_h, 3: depth_multiplier (DW only)
                if 1 in opt_f:
                    stride_w = struct.unpack_from("<i", buf, opt_f[1])[0]
                if 2 in opt_f:
                    stride_h = struct.unpack_from("<i", buf, opt_f[2])[0]
                if 3 in opt_f and builtin_code == DW_CONV2D:
                    depth_mul = struct.unpack_from("<i", buf, opt_f[3])[0]
            except Exception:
                pass

        if builtin_code == CONV2D:
            op_type = "conv2d"
            compute_dtype = "int16"  # TFLM reference Conv2D: scalar → model as int8-scalar
            # Actually TFLM Conv2D reference is scalar (no custom RVV kernel registered)
            op_type = "scalar_conv2d"
        elif builtin_code == DW_CONV2D:
            op_type = "depthwise_conv2d"
            compute_dtype = "int16"  # custom RVV kernel uses vwmacc(int16→int32)
        else:
            continue

        layer = LayerSpec(
            name=output_name or f"op{j}",
            op_type=op_type,
            input_shape=tuple(input_shape),
            filter_shape=tuple(filter_shape),
            output_shape=tuple(output_shape),
            stride_h=stride_h,
            stride_w=stride_w,
            depth_multiplier=depth_mul,
            compute_dtype=compute_dtype,
        )
        layers.append(layer)

    model_name = Path(path).stem
    return WorkloadSpec(name=model_name, layers=layers)


# ── Hardcoded workload for the partial MobileNet test ─────────────────────────

def mobilenet_v1_025_partial() -> WorkloadSpec:
    """Manual spec matching mobilenet_v1_025_partial_layers.tflite.

    Parsed from the flatbuffer:
      Op 0  CONV2D (builtin_code=3)
        input:  [1, 56, 56, 16]   int8   (50,176 bytes)
        filter: [32, 1, 1, 16]    int8   (   512 bytes)
        bias:   [32]              int32  (   128 bytes)
        output: [1, 56, 56, 32]  int8   (100,352 bytes)
        stride: 1×1, SAME padding
        implementation: TFLM reference (SCALAR, no custom RVV kernel)
        MACs: 56 × 56 × 32 × 16 = 1,605,632

      Op 1  DEPTHWISE_CONV2D (builtin_code=4)
        input:  [1, 56, 56, 32]  int8   (100,352 bytes)
        filter: [1, 3, 3, 32]   int8   (   288 bytes)
        bias:   [32]             int32  (   128 bytes)
        output: [1, 56, 56, 32] int8   (100,352 bytes)
        stride: 1×1, SAME padding, depth_multiplier=1
        implementation: custom RVV kernel (sw/opt/litert-micro/depthwise_conv.cc)
        MACs: 56 × 56 × 32 × 9 = 903,168
    """
    pwconv = LayerSpec(
        name="conv_2d (1×1 pointwise)",
        op_type="scalar_conv2d",
        input_shape=(1, 56, 56, 16),
        filter_shape=(32, 1, 1, 16),
        output_shape=(1, 56, 56, 32),
        stride_h=1, stride_w=1,
        compute_dtype="scalar",
    )
    dwconv = LayerSpec(
        name="depthwise_conv_2d (3×3)",
        op_type="depthwise_conv2d",
        input_shape=(1, 56, 56, 32),
        filter_shape=(1, 3, 3, 32),
        output_shape=(1, 56, 56, 32),
        stride_h=1, stride_w=1,
        depth_multiplier=1,
        compute_dtype="int16",  # vwmacc uses int16 inputs
    )
    return WorkloadSpec(
        name="mobilenet_v1_025_partial",
        layers=[pwconv, dwconv],
    )
