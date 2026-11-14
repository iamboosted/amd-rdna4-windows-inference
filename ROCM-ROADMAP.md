# From one HIP kernel to useful ROCm contributions

The most direct next engineering project is a **model-free RDNA 4 benchmark that distinguishes application-kernel inefficiency from library selection, runtime overhead and compiler behavior**. That provides a route to an upstream fix in the component that actually owns the problem. The current MiniCPM result already demonstrates useful AMD GPU work; it does not yet demonstrate a defect in a ROCm library.

This roadmap is based on source and documentation reviewed on 2026-11-13. It describes proposed work. No new compiler/runtime installation, driver change, library build or GPU experiment was performed while preparing this publication.

## The source boundary is now verified

The retained [llama.cpp HIP build](https://github.com/ggml-org/llama.cpp/blob/c83f3058b6f876f2f7e6ccbada858b2446860c85/ggml/src/ggml-hip/CMakeLists.txt) compiles its own quantized kernels from the shared ggml CUDA/HIP source and also links hipBLAS/rocBLAS. Linking a library does not mean every matrix operation uses it.

The [matrix dispatch](https://github.com/ggml-org/llama.cpp/blob/c83f3058b6f876f2f7e6ccbada858b2446860c85/ggml/src/ggml-cuda/ggml-cuda.cu) selects custom quantized matrix-vector/matrix kernels separately from the BLAS fallback. The [HIP adapter](https://github.com/ggml-org/llama.cpp/blob/c83f3058b6f876f2f7e6ccbada858b2446860c85/ggml/src/ggml-cuda/vendors/hip.h) maps the relevant BLAS calls to hipBLAS.

Our selected Q8 Markov operation takes the custom-kernel path. A rocBLAS or hipBLASLt tuning setting will not automatically change that operation. Its packed Q8_0/Q8_1 block layouts and per-block scales also are not interchangeable with a generic BLAS int8 array. Converting formats, changing reduction order or allocating expanded weights would be a separate change requiring its own accounting and correctness checks.

| Work to investigate | Component that owns it | Evidence needed before calling it a fix |
|---|---|---|
| Generalize one-wave-per-row and multiple-rows-per-block dispatch | llama.cpp / ggml HIP kernels | Shape sweep, exactness, fallback coverage and end-to-end tests |
| Improve native GEMV/GEMM selection for gfx1200/gfx1201 | ROCm libraries: rocBLAS or hipBLASLt | Same API/data/layout, faster valid solution, reproducible selector miss |
| Reduce a demonstrated bad code-generation choice | AMD LLVM compiler | Small source/ISA reproducer with identical work and measured effect |
| Fix launch, event, stream or copy behavior | HIP runtime / CLR | Model-free reproducer that excludes application synchronization mistakes |
| Improve native Windows measurement/build reproducibility | GPUOpen tools, TheRock or relevant ROCm project | Exact supported stack, reproducible tooling failure and bounded fix |

AMD now consolidates many math libraries in [rocm-libraries](https://github.com/ROCm/rocm-libraries) and HIP/CLR in [rocm-systems](https://github.com/ROCm/rocm-systems). A future contribution should use the current owning repository rather than assuming an older standalone repository is still the development target.

## 1. Generalize the proven result first

Build a standalone reproducer around the existing arithmetic, using synthetic block-quantized data. No model or agent harness is needed. Start with the exact K=256 shape and fused bias, then vary one factor at a time.

| Sweep | Initial candidates | What it tells us |
|---|---|---|
| Reduction length K | 128, 256, 512, 1,024 | Where one wave ceases to cover the useful work |
| Output rows | 256, 4,096, 65,536, 248,144 | Launch overhead versus enough work to occupy the GPU |
| Rows per block | 1, 2, 4, 8 | Whether the four-row choice generalizes |
| Input vectors | One initially; 2, 4 and 8 only as separate variants | Whether speculative verification needs a different dispatch |
| Bias | None, separate buffer, valid output alias | Fusion and alias correctness |
| Device | gfx1200 first; gfx1201 independently afterward | Whether a rule transfers between the two RDNA 4 GPUs |

Include non-multiple row counts, alignment constraints, seeded data, cancellation cases, finite-value checks and guard regions. Start with exact float-bit agreement against the retained implementation. Changing accumulation order must not silently weaken that acceptance gate.

Compare the retained generic path, the measured four-row specialization and the existing upstream narrow-matrix approach. The [dynamic-wave proposal](https://github.com/ggml-org/llama.cpp/pull/20831) already predicts one useful wave for our shape; the unanswered question is where row packing adds a reliable benefit. Restrict a generalized selector to validated shapes and keep a fallback.

Only after the model-free tests pass should the same repository task and additional tasks be repeated. A synthetic kernel win does not establish a model-throughput win. Graph-enabled behavior and model layouts outside the original fixture require separate checks.

## 2. Make the first direct ROCm target a small GEMV selection study

There is a concrete source location to investigate: [rocBLAS GEMV dispatch](https://github.com/ROCm/rocm-libraries/blob/ce7bf3ffc51be09f091b8f87e8a0647e0ada5bf9/projects/rocblas/library/src/blas2/rocblas_gemv_kernels.cpp). The inspected code already recognizes architecture major 12, selects gfx12-specific cases and has skinny-matrix algorithms. A blanket claim that ROCm ignores RDNA 4 would be wrong.

The research question is whether **its existing selection boundaries choose poorly for particular native Windows gfx1200/gfx1201 workloads**. That is a testable question, not a finding yet. The current [ROCm release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html) also describe GEMM-to-GEMV routing improvements and skinny-matrix work, so checking the actual pinned library matters before proposing a duplicate fix.

Use supported native floating-point library types and explicit transpose, leading-dimension, stride, alpha/beta and batch settings. Begin with one-vector and narrow-output cases, then sweep around the existing selector boundaries. Bound memory and run duration. Record host launch cost separately from GPU work, validate against a trusted reference and compare equivalent math under the same precision policy.

For actual hipBLASLt calls, AMD documents [offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.14.1/how-to/how-to-use-hipblaslt-offline-tuning.html). A best-solution index is tied to a device architecture and library release. Use it to find a reproducible gap between the default choice and an available faster solution; do not copy indices between libraries or cards. First verify that the application and installed library reach this backend at all.

A useful contribution would include the smallest failing selection case, the existing kernel versus candidate, correctness, neighboring-shape regressions and a tested selection rule. This would be direct ROCm-library development. It would not establish a MiniCPM speedup unless that application actually uses the improved path.

## 3. Isolate Windows launch, copy and timing behavior

The inference profile contains variable small readbacks and synchronization intervals. A measured copy interval can include queued work or a fence; its duration alone does not prove a slow PCIe transfer or a runtime defect.

Use tiny model-free tests to separate:

- Empty/minimal kernel launch latency from device execution, with batched launches to avoid timing noise dominating each sample.
- Same-stream timing from event ordering across streams; compare validated device clocks, host wall time and supported tool output.
- Pageable versus pinned host buffers, separately measuring copy submission, completion and synchronization.
- Repeated allocation versus buffer reuse, and required data dependencies versus unnecessary waits.

AMD's [HIP performance guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) supports retaining intermediates on-device, batching small transfers and using pinned buffers where appropriate. These are application changes unless a smaller reproduction proves a problem in HIP itself.

The current [Radeon GPU Profiler page](https://gpuopen.com/rgp/) lists HIP, Windows 11 and RX 9000 GPUs. That makes RGP a candidate for native Windows inspection after its exact driver/tool pairing and dependency review are checked. This documentation does not establish compatibility with the retained local driver, and does not justify replacing it.

The inspected [TheRock Windows matrix](https://github.com/ROCm/TheRock/blob/5f229423e407f3bf9bd0d20d253a8040597edf2e/docs/development/windows_support.md) marks ROCprofiler-SDK, Compute Profiler, Systems Profiler and RCCL unsupported for its native Windows build. It lists BLAS libraries and several primitives as supported, while CLR still has special build requirements. Linux profiling instructions and a Linux collective library are therefore not an assumed native Windows solution.

A reproducible event, stream or copy defect that survives these controls could become a targeted report or fix in HIP/CLR. Earlier rejected profiler measurements remain an attribution problem until such a reproducer exists.

## 4. Use compiler analysis only after identifying a bad instruction pattern

Inspect resource usage and generated ISA for an exact kernel: registers, spills, local/shared memory, repeated loads, barriers and integer dot instructions. Compare equivalent implementations with the same compiler before blaming the compiler. The existing work already showed that attractive source-level ideas can yield little end-to-end benefit.

If a minimal source example consistently generates avoidable instructions or spills, it becomes a focused [AMD LLVM](https://github.com/ROCm/llvm-project) investigation. A faster handwritten kernel alone does not prove a compiler bug: the program may simply have expressed the wrong work distribution.

## 5. Preserve the dual-GPU workflow as a separate gate

The previous device/driver failures make a full-model multi-GPU run a poor first diagnostic. Start later with bounded transfers, explicit capability checks, checksums and stream-order tests on the retained driver. Direct peer access must be measured as available, not assumed. If unavailable, host staging is a possible baseline, not proof that a new bridge is required.

Then vary layer placement while keeping model bytes, cache settings and output checks fixed. Record bytes transferred, synchronization, per-device time and memory headroom. A mostly-filled 9060 configuration and a balanced configuration answer different practical questions; a layer split is not the same as tensor parallelism. Passing a short transfer test does not prove endurance or every split stable.

## Which work is most likely to move the current result?

The MiniCPM host accounting needs about **2.006 ms less per round** to reach 100 tok/s at unchanged acceptance. Verification/synchronization consumes **21.692 ms** of the **31.035 ms** optimized round. The measured Q3_K expert group, full-vocabulary projection and handoff synchronization are better immediate inference leads than further polishing the remaining 0.689 ms Markov group.

For an AMD development pathway, the first deliverable should be the small shape-and-overhead benchmark described above. It can produce either a general llama.cpp dispatch improvement or a reproducible ROCm-library/runtime case. Both are useful; only the latter should be labeled a ROCm component fix.

All proposed execution remains isolated, uses the existing power setting, retains original model precision and the protected production setup, and checks dependencies/CVEs before any new acquisition. No speed target, cross-model transfer or multi-GPU stability result is promised in advance.
