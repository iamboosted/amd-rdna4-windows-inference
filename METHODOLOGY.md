# Methodology and reconstruction requirements

This document separates what was executed from what another developer would need to reproduce. The publication bundle contains aggregate evidence and hashes. It does not contain the private integration host, complete fixture, model weights or a runnable inference distribution.

## Frozen comparison

The original runtime was llama.cpp b11558, commit `c83f3058b6f876f2f7e6ccbada858b2446860c85`, retained from the b1343 Windows HIP package. A custom host loaded those original libraries and redirected one validated kernel launch inside its own process. It did not rewrite the runtime DLLs or modify another process.

The host was compiled using the already-present ROCm Core SDK 10.1.0, whose HIP header reports 7.16.26385. It contained code for gfx1200 and gfx1201, but the selected replacement was restricted to the RX 9060 XT/gfx1200. Compiler version and runtime identity are separate: compiling with that SDK did not upgrade the runtime DLLs used by inference.

The preceding recovery record identifies an operator-confirmed Adrenalin 26.9.2 installation and a corroborating installer package revision of 32.0.32015.2008. This A/B did not independently query the active kernel-driver revision. A fresh reproduction must record its actual driver identity; the historical installation record is not a fresh device-status assertion.

Artifact hashes in [data/provenance.json](data/provenance.json) distinguish the target, sidecar, projector, retained DLLs, custom host and kernel source. These hashes are identifiers, not redistributable copies or a complete public build recipe.

## Workload and settings

The frozen TensorFold repository fixture is based on upstream commit `6e9af930261a88bb28ac0bfb157ce8e57e9eea07`. TensorFold supplied the repository bug being fixed; llama.cpp HIP supplied inference. The task repairs SSE completion-marker parsing and preserves the rest of the streaming behavior.

Use the same full-vocabulary MiniCPM target and matching Q8_0 DSpark sidecar, three proposals, Q8_0 K/V for both models, 131,072 context capacity, target layer split 20/80 in 9070/9060 order, drafter on 9060 and projector on 9070. The target was fully offloaded. The actual prompt was 4,570 tokens, so the allocated cache was not filled.

The request used temperature 1.0, top-p 0.95, top-k 20, seed 2046, thinking enabled, prompt caching disabled and a 16,384-token output limit. The retained sampling configuration included min-p 0.05. The answer stopped naturally at 3,109 tokens. A fixed seed made these observed runs repeatable; temperature was not zero and the task should not be described as greedy decoding.

Batch size was 256, microbatch size 64, eight CPU inference threads, one parallel request, layer splitting, FlashAttention enabled, graph replay disabled, automatic model fitting disabled and no context shift. Both arms preserved these settings. Device ordinals must be resolved from device identity during a reproduction, not copied blindly from this machine.

## Correctness gates

1. Establish the fixture's failing baseline: 17/17 upstream tests and 4/8 regression tests pass. Keep these tests unchanged when evaluating generated edits.
2. Audit every selected score operation into a guarded scratch buffer before the original operation. Compare all 248,144 float bit patterns after each call. Check nonfinite values and both guard regions. Original results feed inference in this diagnostic mode.
3. Require the exact shape, Q8 weight/input layout, fused-bias operation, selected device and call count. The measured audit covered 3,213 calls and 797,286,672 scores.
4. Run the replacement as the actual source of scores in the optimized arm. Compare generated token IDs, reasoning, final answer, generated code, acceptance counts and functional test outcomes to control.
5. Inspect the generated edit before executing its tests. A passing parser test suite cannot establish general language-model quality.

All these gates passed in the reported experiment. Identical output tokens are stronger evidence of unchanged behavior on this particular request than matching only the pass count. Neither result proves equivalence for all possible requests.

## Timing

Clean order was control, optimized, optimized, control. Each clean arm shares one executable and differs by its mode selector. Diagnostic score comparison and device timestamp markers were off; normal host-phase accounting and the same test logging remained enabled.

For each arm:

```text
pooled_decode_tok_s = sum(output_tokens) / (sum(decode_ms) / 1000)
pooled_server_convention_tok_s = sum(output_tokens - 1) / (sum(decode_ms) / 1000)
gain_percent = 100 * (optimized_pooled / control_pooled - 1)
tokens_per_round = (accepted_proposals + rounds) / rounds
required_ms_saving = optimized_mean_round_ms - 1000 * tokens_per_round / 100
```

The recorded count relationship is 2,038 accepted proposals + 1,071 rounds = 3,109 output tokens. Every recorded server rate instead matches 3,108 divided by decode duration. The publication preserves the saved report's all-emitted-token pooled metric and separately exports the server convention; it does not silently treat the two definitions as identical. They produce the same relative gain because every run has the same output count. The per-round calculation is a count-normalized estimate, not an assertion about the exact timing of the first emitted token. Pooled throughput uses summed durations, not the arithmetic mean of reported rates.

The five host phases are verification/synchronization, feature submission, sampling/commit, drafting and other work. Their sum reconciles with total decode duration divided by the number of rounds. Host phase intervals are not pure kernel compute durations.

At fixed counts, the 100 tok/s target needs approximately 2.006 ms less per round with all output tokens counted, or 2.015 ms with the server convention. The difference is bookkeeping, not a change to measured latency.

The GPU profile used sparse device-clock markers, checked against timestamp ordering, stream containment and host bounds. Nine sampled speculative cycles supplied the optimized operation-group profile. Earlier Windows SDK/event measurements that failed validation were not used for attribution. A failed timing method alone does not identify a driver or ROCm bug.

The original Markov group comes from a preceding frozen profile. The 5.89× group ratio is diagnostic; the separate matched clean A/B establishes the 9.33% whole-decode gain. Neither audit throughput nor the faster instrumented profile is substituted for the clean result.

## Preservation and limitations

The original safety guard remained in place, with its CPU checks, memory admission, thermal/power/VRAM/deadline limits, independent watchdog and owned-process cleanup. Audit scratch added about 1 MiB on the 9060, not another full model allocation. No guard tripped in these short runs. The recorded post-run check verified 921 distinct pins and cleanup of the isolated service.

The −30% power setting was operator-reported and preserved; it was not independently measured as an enforced cap. Telemetry is sampled whole-card data. Background graphics activity was not controlled, so peak power and allocation differences must not be presented as a causal energy-efficiency comparison.

Only one repository task, one target/sidecar combination, one power setting and this GPU split were tested. Longer contexts, other models, filled-cache behavior, concurrent requests, alternate devices, graph replay and endurance remain separate validation work.

Before any new acquisition, refresh the relevant dependency and CVE review and pin the exact source/artifacts. The original experiment used already-present tools and files, disabled RPC/WebUI, and retained known residual exposure rather than claiming a CVE-free installation. The [llama.cpp advisory index](https://github.com/ggml-org/llama.cpp/security/advisories) and [reviewed AMD bulletin](https://www.amd.com/en/resources/product-security/bulletin/amd-sb-6028.html) are source references, not certificates of safety.

## What a public reproduction still needs

A portable, source-integrated benchmark must replace the private launch hook with a reviewed dispatch change, preserve upstream notices, provide a public fixture and comparison implementation, and identify all build/runtime inputs. It should begin with synthetic matrices that need no model downloads, then independently reproduce the real task. The next-step design is in [ROCM-ROADMAP.md](ROCM-ROADMAP.md). None of those future experiments is represented as already completed here.
