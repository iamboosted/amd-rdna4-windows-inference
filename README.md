# MiniCPM and LFM inference on Windows RDNA 4

Two measured Windows HIP case studies: **MiniCPM's 85.56 → 93.54 tok/s kernel optimization on a dual-GPU setup**, and **LFM's 95.48 → 144.38 tok/s DSpark result on one RX 9060 XT**. They use different models and are not a model-speed ranking.

**Separate highlight: [Qwen3.8-27B at approximately 69 tok/s on one RX 9060 XT](qwen-9060xt/README.md)**, with unchanged target weights, exact six-task output reproduction, power-limit evidence and a larger-context follow-up.

## MiniCPM: a narrow Q8 kernel case study

A specialized HIP kernel increased pooled decode throughput from **85.56 to 93.54 tokens/s (+9.33%)** in a controlled MiniCPM experiment. All **3,109 generated token IDs** stayed identical, the generated repository fix passed **25/25 tests**, and a separate numerical audit compared **797,286,672 scores with zero bit mismatches**.

The change runs on an **RX 9060 XT** inside a **dual-GPU RX 9070 XT + RX 9060 XT inference setup**. It accelerates one calculation in the DSpark drafter. This is a llama.cpp HIP optimization, with the existing target weights, vocabulary and cache precision preserved.

Evidence date: 2026-11-13. This is a sanitized research report and measured case study; the specialization remains isolated from the everyday launcher. No upstream submission or general ROCm speedup is claimed.

## What the model actually did

The task was to fix a real Server-Sent Events parser bug in a frozen repository fixture. The parser needed to recognize a completion marker with no space after the field separator, such as `data:[DONE]`, while preserving streaming behavior and handling surrounding whitespace.

Thinking was enabled. Each run generated **3,109 tokens**, including reasoning and the final answer, from a **4,570-token prompt**. The generated edit was inspected and then evaluated against **17 existing upstream tests and eight regression tests**. No manual repair was made to the model's patch.

The unmodified fixture passed the 17 upstream tests but only four of the eight regression tests. Every measured candidate passed all 25. These are meaningful checks of this bug fix, not a broad coding benchmark or a guarantee of general model accuracy.

## The measured change

The drafter scores its full vocabulary using a quantized matrix-vector operation. In the selected operation, each output row reduces 256 input values: **K = 256**, with **248,144 output rows** and one input vector. A bias is added to each result.

The retained RDNA 4 implementation launched **eight wavefronts per row**. A wavefront is a group of GPU threads executing together; here it contains 32 threads. For this exact K dimension and quantization, the first wavefront performs the dot-product arithmetic. The other seven still participate in the general kernel's bookkeeping and shared reduction.

The replacement assigns **one wavefront to each row and four independent rows to each block**:

| Property | Retained implementation | Specialized implementation |
|---|---:|---:|
| Threads per wavefront | 32 | 32 |
| Wavefronts assigned to one row | 8 | 1 |
| Independent output rows per block | 1 | 4 |
| Threads per block | 256 | 128 |
| Blocks for 248,144 rows | 248,144 | 62,036 |
| Vocabulary entries scored | 248,144 | 248,144 |

The replacement preserves signed integer dot products, half-precision scale conversion, the tested floating-point reduction order, and bias addition. It does not reduce the vocabulary, requantize the target, change training, or substitute a smaller target model.

The dispatch is deliberately narrow: the exact quantization, shape, fusion, argument layout and RX 9060 XT architecture must match. Other operations keep their original implementation. The target verification kernels are unchanged.

## Clean A/B results

The controls and optimized runs used the **same host executable**, retained runtime, model files, settings, prompt and seed. Only the specialization mode changed. GPU graph replay was disabled equally in both arms. Numerical-audit and GPU-marker runs are excluded from these headline rates.

| Run, in execution order | Server-reported decode tok/s | Draft ms/round | Verification + synchronization ms/round | Functional checks |
|---|---:|---:|---:|---:|
| Control 1 | 86.406 | 10.064 | 20.634 | 25/25 |
| Optimized 1 | 93.315 | 6.555 | 21.741 | 25/25 |
| Optimized 2 | 93.698 | 6.603 | 21.644 | 25/25 |
| Control 2 | 84.668 | 10.096 | 21.330 | 25/25 |

Pooled throughput is total generated tokens divided by total decode time, not an average of instantaneous peaks:

- **Control: 85.556 tok/s. Optimized: 93.536 tok/s. Gain: 9.33%, or 7.98 tok/s.**
- Mean drafting time fell from **10.080 to 6.579 ms per round**.
- Mean verification plus synchronization time moved from **20.982 to 21.692 ms per round**. The result does not depend on claiming this phase got faster.
- All runs used **1,071 speculative rounds**, proposed **3,213 tokens**, and accepted **2,038 proposals (63.43%)**. Acceptance did not change.

There is a small reporting-convention difference: every recorded server rate in the table equals **(3,109 − 1) / decode seconds**, whereas the pooled figure in the saved report counts all 3,109 output tokens per request. Using the server convention for both pooled arms gives **85.528 → 93.506 tok/s**, with the same **9.33%** gain. Both conventions and their inputs are retained in the data; the roughly 0.03 tok/s difference is not additional performance.

The complete output tokens, reasoning, final answer and generated code matched across the clean runs. Two repetitions per arm and balanced order reduce one ordering concern, but do not establish a statistical confidence interval. Background graphics activity was not controlled.

## Numerical audit and profiling

For every selected operation in a separate audit run, the candidate kernel first wrote into a guarded temporary output. The original kernel then ran and supplied the output used by inference. Every candidate output float was compared bit-for-bit with the original. Running the candidate first preserves the original bias when bias and output share storage.

| Audit check | Result |
|---|---:|
| Selected kernel calls | 3,213 |
| Scores compared | 797,286,672 |
| Bit mismatches | 0 |
| Nonfinite values | 0 |
| Guard corruption | 0 |

The clean optimized runs used the replacement output directly, providing a separate end-to-end check beyond the numerical audit. This establishes observed equivalence for the tested workload and inputs, not a proof for every possible input.

The sampled group of three Markov scoring calls fell from **4.057 to 0.689 ms**, about **5.89× faster for that operation group**. The original measurement comes from the frozen preceding profile; the replacement was measured with the same validated profiling approach. Instrumentation affects sampled intervals, so the whole-request clean A/B supplies the actual inference speed claim. A 5.89× kernel improvement is not a 5.89× model speedup.

The profiler used sparse device-clock markers and independent ordering, stream-containment and host-wall-time checks. Earlier inconsistent timing methods were rejected for kernel attribution. The profile is evidence about where to investigate next, not a substitute for clean throughput measurements.

## Configuration and scope

| Setting | Value |
|---|---|
| OS / backend | Native Windows; llama.cpp HIP |
| Target | MiniCPM-V-4.7-35B-A3B, existing `mix_q3gud_im` GGUF |
| Drafter | Matching DSpark Q8_0, three proposals; no n-gram drafting |
| Vocabulary | Full 248,144 entries |
| Context allocation | 131,072 tokens |
| Actual prompt / output | 4,570 / 3,109 tokens |
| KV precision | Q8_0 keys and values for target and drafter |
| Target split | 20/80 in RX 9070 XT / RX 9060 XT order; 9/31 trunk layers |
| Drafter / projector | Drafter on RX 9060 XT; projector on RX 9070 XT |
| Workload | One text-only repository bug fix, one request at a time |
| RX 9060 XT power setting | User-reported −30% power limit, retained in both arms |
| Runtime | Retained b1343 package, llama.cpp b11558 at `c83f3058b6f876f2f7e6ccbada858b2446860c85` |
| Custom host compiler | Existing ROCm Core SDK 10.1.0; HIP headers 7.16.26385 |
| Runtime replacement | None; the custom host loads the retained runtime libraries |

**128K is allocated capacity, not a filled 128K prompt.** This does not measure long-context scaling or visual reasoning. Decode counts reasoning and answer tokens and excludes prompt processing and loading. The measured target uses both GPUs; this is not a 9060-only inference result.

Across the clean runs, the RX 9060 XT's observed peaks were **45°C core, 55°C hotspot, 134 W, and 14,983 MiB used VRAM**. That leaves **1,321 MiB** from the reported 16,304 MiB total. Both optimized runs left **7,663 MiB** on the RX 9070 XT at their observed peak allocation. These are whole-card sampled allocations and short-run telemetry, not a long-duration stability or energy-efficiency guarantee. Peak wattage alone does not measure energy per token.

No guard tripped. Owned test processes were cleaned up and 921 recorded file pins passed the post-run preservation check. Production launchers, original model files, runtime binaries, the protected agent harness, Vulkan, driver and GPU tuning were preserved. No new package or executable acquisition was needed for this experiment.

## Related work and contribution boundary

The specialization was implemented from local profiling and the retained llama.cpp arithmetic before the following proposals were reviewed in this investigation. Their code was not incorporated into this experiment. They nevertheless establish relevant prior work:

- [Dynamic MMVQ wave counts for narrow matrices](https://github.com/ggml-org/llama.cpp/pull/20831) addresses the same class of excessive wave allocation on RDNA 3/4. Applying its reviewed formula to this Q8_0 K=256 shape gives one useful wave. That is a source-based inference; the proposal was not benchmarked locally. It was open and unmerged when reviewed.
- [RDNA 4 MMVQ tuning for K-quants](https://github.com/ggml-org/llama.cpp/pull/24386) changes Q4_K/Q6_K wave counts and includes external Windows results in its discussion. It covers different formats and is not the MiniCPM Markov implementation. It was open and unmerged when reviewed; external performance claims were not reproduced here.
- [More output rows per block for small K](https://github.com/ggml-org/llama.cpp/pull/20635) is merged prior work on amortizing small-K operations across rows, with NVIDIA measurements. The retained llama.cpp source contains a small-K path but disables that path for RDNA.

The demonstrated contribution is a working, tightly scoped Windows HIP implementation and its numerical and functional evidence. General GPU work-distribution techniques are established practice. This report makes no first-ever claim.

The underlying [llama.cpp MMVQ source](https://github.com/ggml-org/llama.cpp/blob/c83f3058b6f876f2f7e6ccbada858b2446860c85/ggml/src/ggml-cuda/mmvq.cu) supplies the arithmetic being specialized. The change is in the inference application's GPU code. It does not modify ROCm globally, GPU firmware or the physical RDNA 4 architecture.

## What remains before 100 tok/s

Using the all-emitted-token accounting above, **3,109 / 1,071 = 2.9029 tokens per round**. The optimized mean round is **31.035 ms**. Reaching 100 tok/s at unchanged acceptance would require **29.029 ms**, a saving of about **2.006 ms per round**. Under the server's one-fewer-token convention, the required saving is about **2.015 ms** instead. These are fixed-count arithmetic targets, not measured forecasts.

The remaining sampled Markov group is only about 0.689 ms, so removing that entire group would not close the gap by itself under this accounting. Verification and synchronization now account for about **70% of the clean round time**. Sampled leads include approximately **5.07 ms in Q3_K expert kernels** and **1.53 ms in the target Q6_K vocabulary projection** on the RX 9060 XT. These are measured regions to investigate, not promised recoverable time. Copy intervals can include waiting; they must not automatically be attributed to PCIe bandwidth.

See [the ROCm development roadmap](ROCM-ROADMAP.md) for the distinction between generalizing this llama.cpp optimization and demonstrating a contribution to AMD's libraries, runtime or compiler.

## LFM: 144.38 tok/s and a working repository fix on one RX 9060 XT

Adding LiquidAI's official F16 DSpark sidecar increased LFM2.5-24B-A2B decode from **95.483 to 144.378 tok/s (+51.21%)** on the same repository bug. Both unedited fixes passed **25/25 tests**. Thinking was enabled and allowed to finish naturally.

**LFM used the 9060 XT alone.** Although the machine contains two GPUs, the saved runtime evidence places all 41/41 target and 6/6 draft layers on the 9060 XT, with no 9070 model, cache, recurrent-state or compute-buffer allocation. Enumerating two adapters is not dual-GPU inference.

| Measurement | Target only | Target + DSpark |
|---|---:|---:|
| Decode tok/s, server convention | 95.483 | 144.378 |
| Prompt tokens | 4,705 | 4,705 |
| Output tokens including thinking | 9,326 | 10,177 |
| Decode seconds | 97.661 | 70.482 |
| Prefill tok/s | 1,228.924 | 885.663 |
| Prefill seconds | 3.829 | 5.312 |
| Complete request seconds | 101.499 | 75.802 |
| Existing / regression tests | 17/17 + 8/8 | 17/17 + 8/8 |
| Peak used / remaining VRAM, MiB | 14,107 / 2,197 | 15,523 / 781 |
| Peak core / hotspot, °C | 51 / 65 | 51 / 65 |
| Peak reported GPU power, W | 132 | 137 |

The complete request took **25.32% less time** despite emitting **9.13% more tokens**. Prompt processing became slower, which faster decode outweighed. The two generated fixes have different text and lengths; this is **one A/B workload**, not a repeated statistical comparison or proof of identical model behavior. These rates use the server's `(output tokens − 1) / decode seconds` convention, with the original counts retained in [the numeric evidence](data/lfm-results.json).

### LFM task and configuration

The task repaired the same class of SSE parser bug described above: recognize `data:[DONE]` and `data: [DONE]`, preserving usage, timing, reasoning, tools and error classification. It used a frozen fixture at commit `6e9af930261a88bb28ac0bfb157ce8e57e9eea07`. Only the stream-parser module could change. Both candidate diffs were reviewed and tested without hand repairs against 17 upstream tests and eight regressions. The original fixture failed four regressions. The passes demonstrate this fix works, not general quality parity.

| Setting | Value |
|---|---|
| Target | Unchanged full-vocabulary `LFM2.5-24B-A2B-QAD-Q4_0` GGUF |
| Draft | Official `LFM2.5-24B-A2B-DSpark-F16` GGUF |
| Engine | Retained llama.cpp b11558, commit `c83f3058b6f876f2f7e6ccbada858b2446860c85`, HIP |
| Target / draft placement | RX 9060 XT, split mode none |
| Target / draft context capacity | 65,536 / 65,536 tokens |
| Target and draft KV | Q8_0 keys and values; no Q4 KV |
| Target / draft cache | 680 / 340 MiB |
| Proposals | Nine; runtime confirmed DSpark active |
| Sampling | Temperature 0.1, top-p 1, top-k 50, min-p 0, repeat penalty 1.05, seed 2046 |
| Thinking / output allowance | Enabled, native template / 16,384 tokens |
| Batch / microbatch / threads | 256 / 64 / 8 |
| Attention / cache / sequence count | Flash attention on / prompt cache off / one |
| Power limit | User-set −30% on the 9060 XT, unchanged |

**64K is allocated capacity, not actual prompt occupancy.** The requests began at 4,705 tokens and ended below 15K total tokens. This does not measure speed at a filled 64K context. Host embeddings and CPU orchestration remain part of this backend.

No custom kernel, n-gram accelerator, target requantization or runtime upgrade was used for this LFM A/B. The engine is llama.cpp; the repository fixture used to evaluate the generated edit is not the inference engine. LFM's trained sidecar result is separate from MiniCPM's specialized scoring kernel.

### LFM drafting and limits

The drafter ran **1,672 rounds**, proposed **15,048 tokens** and had **8,504 accepted (56.51%)**. Accepted proposals account for approximately **83.56%** of the 10,177 emitted tokens; including the target token gives approximately 6.09 emitted tokens per round.

Runtime wall timers recorded **19.392 s drafting**, or **11.60 ms per round**. Subtracting that from the 70.482 s decode leaves **51.090 s**, or **30.56 ms per round**. This remainder includes verification, sampling, transfers and scheduling. It is **not an isolated HIP verification-kernel measurement** and cannot all be claimed as recoverable time.

The official sidecar was acquired only after the recorded dependency/advisory review, at revision `9222226d8d0fea777a25988e7389e543e99e4e0d` from [LiquidAI's DSpark GGUF repository](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark-GGUF). Model and runtime hashes are retained in the allowlisted evidence. No executable dependencies, Python packages, drivers, HIP or Vulkan upgrades were added for this A/B. The retained software is not represented as CVE-free: it was used locally with authenticated loopback, RPC unused and WebUI disabled.

No guard tripped in either bounded run; owned processes were cleaned up and **894 runtime/input pins** passed re-verification. GPU-only ADLX monitoring used 0.5-second polling and a three-second stale-data stop, with thermal, power and memory limits. Sampled peaks are not calibrated energy readings. These runs do not establish endurance, graceful unload behavior or two-GPU reliability.

## Evidence and reproduction

- [Methodology and reconstruction requirements](METHODOLOGY.md)
- [Allowlisted numeric evidence](data/results.json), [clean-run CSV](data/runs.csv), and [artifact provenance](data/provenance.json)
- [LFM numeric evidence and provenance](data/lfm-results.json)
- [Standalone Qwen 27B single-card report](qwen-9060xt/README.md) and [its numeric evidence](qwen-9060xt/results.json)
- [Privacy boundary and verification](PRIVACY.md)

This package contains a write-up and sanitized measurements. The machine-specific runner, model weights, raw prompts, responses, logs and binaries are not redistributed. It is not a turnkey benchmark release. The included verifier checks the public bundle's arithmetic, file inventory, hashes and common privacy-leak patterns using Python's standard library.
