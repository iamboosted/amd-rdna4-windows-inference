# Qwen3.8-27B at approximately 69 tok/s on one RX 9060 XT

**Native Windows. One 16 GB Radeon. The existing full-vocabulary Qwen3.8-27B UD-Q3_K_XL target.** An isolated llama.cpp HIP build with DFlash2 speculative decoding sustained approximately **69 tokens/s median decode** across six new Python coding tasks. The retained runs reproduced all **4,069 target token IDs** and the same **155/167 functional checks, with 5/6 tasks passing**.

The latest six-task run measured **69.278 tok/s median** with the operator's **−30% power-limit setting**. The preceding run measured **68.992 tok/s**. Request-window average reported GPU power fell from **148.09 to 126.14 W**, and the sampled hotspot peak fell from **70°C to 63°C**. This is an observed short-run power comparison, not proof of a speed increase or a 30% reduction in electricity use.

Evidence through **2026-11-13**. This is a separate single-GPU case study. [MiniCPM and LFM results](../README.md) cover different models, workloads and acceleration paths.

## Why this result matters

Both the 27B target and its drafter performed their GPU inference on the **RX 9060 XT**. No second GPU supplied layers or verification work. The result is one request at a time, not summed throughput from parallel streams or a fastest-second peak.

The target was not replaced by a Swift fine-tune, a smaller-vocabulary model, a new lower-quality quant, or a new set of weights. The existing GGUF is already quantized; the optimization preserved that artifact. A DFlash2 sidecar proposes text, and the full target checks those proposals. The retained changes improve how the inference program schedules GPU matrix operations.

**This is llama.cpp through HIP, not a TensorFold inference-engine result or a global ROCm upgrade.** Some private experiment flag names carry historical project labels. They do not identify the runtime executing the model. Production launchers and the protected agent harness were preserved.

## What the six tasks ask the model to do

| Group | Tasks | Plain-English purpose |
|---|---|---|
| 1 | `available`, `schedule` | Find free time slots; order jobs with dependencies. |
| 2 | `snapshot`, `shortest` | Look up historical values; find a maze path with one permitted wall. |
| 3 | `cheapest`, `cache_reads` | Find a route with an edge limit; implement an expiring LRU cache. |

These are new-code requests with executable Python checks. Thinking is enabled with a **256-token reasoning budget**. The **4,069 tokens are the sum of six requests**, including reasoning, not a single 4K answer. Each request generated 390–1,175 tokens from a short prompt.

The cheapest-route answer fails 12 of its 28 checks. The snapshot answer passes its functional checks but includes extra text. Exact reproduction preserves these errors too. The six-task suite is useful local evidence, not a general coding-quality benchmark.

## The controlled optimization

DFlash2 proposes up to seven tokens. The target evaluates eight positions, including the anchor. Earlier target work established a forced-MMQ path. The clearest subsequent A/B in this report changes the drafter's matrix dispatch in the **same executable and backend build**.

`TENSORFOLD_DRAFT_MMQ=1` extends that path to Q4_K/Q6_K matrices on RDNA 4 with eight columns. The final version excludes matrices with fewer than 1,024 output rows. It requires `GGML_HIP_LAB_FORCE_MMQ_BATCH8=1` and is disabled by default.

This is a type-and-shape rule, not a check that a tensor belongs to a drafter. In this workload the selected Q4_K/Q6_K matrices belong to the sidecar; another model could use those types in its target. The rule must not be enabled indiscriminately for other models.

| Clean condition | Median decode tok/s | Aggregate decode tok/s | Draft ms/round | Verification + synchronization ms/round | Checks |
|---|---:|---:|---:|---:|---:|
| Same-build draft switch OFF | 65.101 | 64.934 | 12.993 | 72.440 | 155/167 |
| Same-build draft switch ON | 68.251 | 68.182 | 9.877 | 71.711 | 155/167 |
| Final small-row exclusion | 68.328 | 67.919 | 9.958 | 71.926 | 155/167 |
| Final version, repeat | 68.826 | 68.433 | 9.758 | 71.625 | 155/167 |

The controlled switch improved median decode by **4.84%** and reduced drafting time by approximately **24.0%**. Accepted proposals also increased from 3,350 to 3,354, reducing verification rounds from 713 to 709. The whole improvement therefore cannot be assigned entirely to matrix execution time. All six complete target token streams remained identical.

The target MMQ change precedes this table. The table does not claim a 4.84% improvement over an unmodified launcher, nor invent a matched baseline for the complete optimization sequence. The original approximately 25–30 tok/s expectations and unrelated multi-stream demonstrations are not controlled baselines here.

## Restoration and the power-limit follow-up

The retained backend was reused after the operator restored the pinned graphics driver. No new compiler, HIP runtime or llama.cpp build was needed for these two runs.

| Measurement | Before power-limit change | User-set −30% |
|---|---:|---:|
| Median decode tok/s | 68.992 | 69.278 |
| Aggregate decode tok/s | 68.755 | 69.023 |
| Exact reference token streams | 6/6 | 6/6 |
| Functional checks | 155/167 | 155/167 |
| Request-window average reported GPU power | 148.09 W | 126.14 W |
| Peak core / hotspot | 50 / 70°C | 48 / 63°C |
| Peak reported GPU power | 156 W | 155 W |
| Peak used VRAM | 15,607 MiB | 15,607 MiB |

Average reported draw decreased **14.82%** while the median rate differed by only **0.41%**. One run per condition does not establish a speed gain or statistical equivalence. The data supports no observed throughput loss on this particular repeat.

Power is integrated over the six request windows with linear interpolation and trapezoidal integration; model loading, warm-up and idle are excluded. Coverage was **99.76%** and **99.26%**, with no extrapolation beyond available samples. The ADLX helper prefers total-board power but can fall back to GPU power without recording which source it selected. These are reported GPU readings, not measured wall power or a universal energy-efficiency claim.

The power-limit setting was supplied by the operator and was not modified by the experiment. The percentage changes the permitted power envelope; it does not force measured average draw to drop by the same percentage.

## Larger context: a separate, honest result

A later run allocated **16,384 context tokens** with **Q8_0 keys and values for both target and drafter**. It used a **10,258-token application prompt**, asking for a warehouse reservation command handler with stock allocation, expiry, retries, cancellation, shipping and ownership rules.

| Measurement | 16K / Q8 follow-up |
|---|---:|
| Prompt / generated tokens | 10,258 / 2,608 |
| Decode rate | 68.896 tok/s |
| Prompt processing | 20.425 s |
| Decode time | 37.840 s |
| Complete request | 58.274 s |
| Peak used / remaining VRAM | 15,565 / 739 MiB |
| Raw output accepted by strict loader | No |
| Separate formatting-only diagnostic | 875/876 checks; 39/40 workflows |

The raw answer started with an unmatched Markdown fence and was rejected. Removing only that opening line in a separate diagnostic copy made the code testable; no code logic was repaired. It then failed one whitespace-only SKU validation check. **The diagnostic score is not an unedited pass.**

The six short tasks under the new configuration scored **132/167, 4/6 tasks**, at **67.479 tok/s median**. An unused forbidden import caused one task's strict rejection; removing it in a diagnostic copy passed that task. Context, batch sizes and draft cache precision changed together, so this run does not isolate why the answers changed or establish quality equivalence with the 8K configuration.

These results show approximately 69 tok/s decode at roughly 10K actual prompt tokens. They do not establish speed with a filled 16K window or at 32K. **32K Q8 was not demonstrated on this card.** No Q4-cache fallback was used in the later context test.

## Profiling, rejected changes and remaining work

Accepted device-clock profiling measured approximately **61.28 ms of target matrix work per verification pass**, including **26.08 ms in IQ3_S gate/up** and **10.92 ms in IQ4_XS down projections**. The IQ3_S label refers to tensors inside the unchanged UD-Q3_K_XL GGUF, not a replacement model quant.

In a separate matched drafter profile, the sampled graph span fell from **11.748 to 8.591 ms**, with matrix-operation spans falling from **10.427 to 7.254 ms**. The full-vocabulary top-k selection cost only approximately **0.17 ms**. Rewriting it was not justified by these measurements.

Native SDK timestamps and event-based stage timings produced inconsistent measurements on the installed Windows stack and were rejected for attribution. The accepted marker method checked ordering, interval bounds and host wall time. Instrumented runs used disabled graphs; clean runs with graph replay provide the throughput claim. A profile interval is not automatically removable overhead.

| Investigation | Observed result | Status |
|---|---|---|
| Signed IQ3 lookup grid | 60.384 tok/s; unchanged reference answers | Regressed; not retained. |
| Small target MMVQ routing | 66.331 tok/s, 159/167 checks; 0/6 identical streams | Changed answers; not an accuracy-preserving win. |
| Alternate draft selection policies | Approximately 67.85–68.12 tok/s; fewer accepted proposals | No retained speed win; some answers changed. |
| 110-byte versus 112-byte IQ3 block alignment | Approximately 2.3–2.5% median paired kernel gain | Microbenchmark only; not integrated into inference. |
| N8/N16 attention split-count intervention | 312/312 required comparisons bit-exact across six forced contexts | Correctness diagnostic, not a timed tree verifier. |

The attention diagnostic also passed 54 auxiliary checks, 42 bit-exact, with zero greedy-choice mismatches or nonfinite values. It uses forced tokens, not newly generated and graded coding answers. It does not add another measured speed improvement.

For the final repeat, `4069 = 3354 accepted proposals + 709 target bonus tokens + 6 initial tokens`. That is **5.7306 emitted tokens per speculative round**, excluding initial tokens, with **83.8635 ms per round**. At unchanged acceptance, 100 tok/s would require approximately **57.306 ms rounds**, saving **26.56 ms**. Even perfect acceptance would give only approximately **95.39 tok/s** at that unchanged round time. These are arithmetic bounds, not performance forecasts.

## Configuration, provenance and limits

| Setting | Retained six-task configuration |
|---|---|
| GPU | RX 9060 XT, gfx1200, 16,304 MiB reported VRAM |
| Target | Qwen3.8-27B UD-Q3_K_XL, full 248,320-token vocabulary |
| Sidecar | DFlash2 Q4_K_M weights, seven proposals |
| Backend | Isolated llama.cpp b12110 plus experimental HIP patches |
| Upstream commit | `cd959bae99ff4bed22bef2604f45241c09bbb8c8` |
| Build SDK | Existing ROCm Core SDK 10.1.0; HIP headers 7.16.26385 |
| Context / target KV / draft KV | 8,192 / Q8_0 K and V / Q4_0 K and V |
| Sampling | Greedy, temperature 0, seed 2046 |
| Reasoning / maximum output | 256-token budget / 1,536 tokens per request |
| Batch / microbatch | 1,024 / 256 |
| Flash attention / graph replay | Enabled / enabled in clean runs |
| Requests / prompt cache | Sequential, one slot / disabled |

The restoration record identifies operator-confirmed **Adrenalin 26.9.2**, with installer evidence for package **32.0.32015.2008**. It did not independently query the live kernel driver through WMI or PnP. The SDK version used to compile custom code is separate from the installed graphics driver.

The full target and draft layer offload was on the 9060 XT; approximately 521 MiB of target embeddings remained CPU-mapped. This is single-GPU inference, not a claim that every model byte resides on the GPU.

Bounded runs used GPU-only ADLX monitoring, thermal/power/time limits, fresh HIP memory admission and owned-process cleanup. No guard failure was recorded for the reported successful runs. This does not establish long-duration stability or dual-GPU reliability. Historical dual-GPU driver failures are not validated away by these single-card results.

The experiment changed GPU execution software and dispatch, not model training, weights, firmware, voltage or clocks. Driver restoration and the operator's power-limit adjustment were separate actions. The everyday launcher, protected harness and Vulkan were not changed by this publication or the retained experimental patch.

[Machine-readable measurements and artifact hashes](results.json) retain the selected clean and negative results, restoration/power comparison and context follow-up. Run `python verify.py --self-test` from the repository root to check arithmetic, file inventory, hashes and privacy patterns. [Publication boundaries](../PRIVACY.md) explain what is omitted. Raw prompts, answers, token IDs, logs, binaries and machine-specific runners are not redistributed. This is a measured case study, not a turnkey backend release or a claim of first-ever performance.
