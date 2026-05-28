# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cocotb
import numpy as np

from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles


@cocotb.test()
async def core_mini_rvv_mobilenet_v1(dut):

    fixture = await Fixture.Create(dut, highmem=True)
    r = runfiles.Create()
    elf_files = ['run_mobilenet_v1_025_partial_binary.elf']
    for elf_file in elf_files:
        await fixture.load_elf_and_lookup_symbols(
            r.Rlocation('coralnpu_hw/tests/cocotb/tutorial/tfmicro/' + elf_file),
            ['inference_status', 'inference_status_message',
             'inference_cycles_lo', 'inference_cycles_hi',
             'framework_cycles_lo', 'framework_cycles_hi'])
        # NOTE: Running the example in DEBUG mode is too slow could take more than 500Million cycles
        cycle_count = await fixture.run_to_halt(timeout_cycles=130_000_000)
        print(f"Total number of execution cycles: {cycle_count}", flush=True)

        tflite_inference_status = (await fixture.read_word('inference_status')).view(np.int32)
        tflite_inference_message = bytes((await fixture.read('inference_status_message', 31))).decode()
        assert tflite_inference_status == 0, tflite_inference_message

        # Per-phase cycle counts (written by rdcycle64 in run_mobilenet.cc)
        inv_lo = int((await fixture.read_word('inference_cycles_lo')).view(np.uint32))
        inv_hi = int((await fixture.read_word('inference_cycles_hi')).view(np.uint32))
        fw_lo  = int((await fixture.read_word('framework_cycles_lo')).view(np.uint32))
        fw_hi  = int((await fixture.read_word('framework_cycles_hi')).view(np.uint32))
        invoke_cycles    = (inv_hi << 32) | inv_lo
        framework_cycles = (fw_hi  << 32) | fw_lo

        print(f"ROOFLINE_DATA invoke_cycles={invoke_cycles}", flush=True)
        print(f"ROOFLINE_DATA framework_cycles={framework_cycles}", flush=True)
        print(f"ROOFLINE_DATA total_cycles={cycle_count}", flush=True)
        print(f"\nPartial mobilenet Invoke() successful", flush=True)