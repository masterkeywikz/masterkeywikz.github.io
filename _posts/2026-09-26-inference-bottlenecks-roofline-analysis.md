---
layout: post
title: "LLM Fundamentals: Understanding Inference Bottlenecks with Roofline Analysis"
date: 2026-09-26 08:00:00 -0700
permalink: /essays/inference-bottlenecks-roofline-analysis/
categories: inference machine-learning systems
---

In my [first LLM fundamentals post](/essays/implementing-byte-pair-encoding/), I explored how text becomes tokens. This post picks up further down the pipeline: once those tokens reach the model, what determines how fast it can respond?

A GPU's advertised FLOPs tell us how much arithmetic it can perform. But arithmetic needs data. If the GPU spends most of its time moving weights and cached activations from memory, more compute alone will not make the response proportionally faster.

**Roofline analysis puts compute and data movement in the same picture.** It gives me a way to reason about prefill, decoding, and batching before reaching for a profiler.

## One request, two different workloads

After tokenization and scheduling, a typical autoregressive request has two phases:

- **Prefill:** process the prompt, store its keys and values in the KV cache, and produce the logits used to sample the first output token.
- **Decode:** feed the sampled token back through the model, attend to cached history, append its keys and values, and sample the next token. Repeat until the response ends.

Prompt tokens are already known, so prefill can process many positions together within each layer, while respecting the causal attention mask. Ordinary decoding has a dependency between successive generated tokens. Each active sequence contributes one new input token per decode step.

That difference changes weight reuse. A prefill matrix multiplication can apply the same weights to many prompt positions; a batch-one decode applies them to just one position. Batching independent requests supplies more positions to work on together. The [inference chapter of *How To Scale Your Model*](https://jax-ml.github.io/scaling-book/inference/) explains this distinction in detail.

## Count operations and bytes

For a particular kernel or workload, define:

- `F`: floating-point operations performed, in FLOPs.
- `M`: bytes transferred between high-bandwidth memory (HBM) and the processor. This is traffic, not just allocated memory.
- `C`: compute ceiling for the relevant precision and operation, in FLOP/s.
- `BW`: HBM bandwidth, in bytes/s.

Arithmetic intensity measures how much computation we get from each byte moved:

```text
I = F / M                         [FLOPs/byte]

compute time = F / C              [seconds]
memory time  = M / BW             [seconds]

t ≥ max(F / C, M / BW)
```

The maximum is an optimistic lower bound on runtime: even with perfect overlap, neither resource can finish faster than its own limit. Actual runtime can be higher because of dependencies, inefficient kernels, and overhead.

Rearranging gives the familiar roofline:

```text
achievable FLOP/s ≤ min(C, BW × I)
ridge point = C / BW              [FLOPs/byte]
```

Below the ridge point, bandwidth sets the lower ceiling. Above it, compute does. NVIDIA's [GPU performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) describes these limits and the additional role of latency when work is too small to saturate the device.

<figure class="visual-diagram">
  <img src="/assets/images/inference-roofline.svg" alt="Roofline for an illustrative accelerator with 120 TFLOP/s compute and 2 TB/s bandwidth. The bandwidth ceiling rises with arithmetic intensity until it meets the compute ceiling at 60 FLOPs per byte. Weight-only BF16 decode batches of 1 and 16 fall below that ridge." />
  <figcaption>The roof is an upper bound, not measured performance. Points mark the ideal ceilings for the weight-only example below.</figcaption>
</figure>

## A worked example: an 8B model

Use an illustrative dense model and accelerator:

- Parameters: `P = 8 × 10⁹`.
- BF16 weights: `s = 2` bytes per parameter, or **16 GB** of weights.
- Compute: `C = 120 × 10¹²` FLOP/s, or **120 TFLOP/s**.
- Bandwidth: `BW = 2 × 10¹²` bytes/s, or **2 TB/s**.

These are round teaching numbers, not a benchmark for a particular GPU. All GB and TB here are decimal. Assume weights and cache fit in HBM.

Start with a deliberately limited model: count about `2P` FLOPs per new token for the dense weight multiplications, treating a multiply-add as two operations. Ignore attention over the cached sequence, activation traffic, and overhead for now. Assume each step streams the weights once from HBM and reuses them across its batch.

For batch size `B`:

```text
F ≈ 2PB
M ≈ sP
I ≈ 2PB / sP = 2B / s
```

With BF16 weights, the intensity is approximately **B FLOPs/byte**. This cancellation is useful: under these assumptions, model size changes the amount of work and weight traffic together; batch size changes their ratio.

At batch one:

```text
compute time = 16 × 10⁹ / (120 × 10¹²)
             ≈ 0.133 ms

memory time  = 16 × 10⁹ / (2 × 10¹²)
             = 8 ms
```

The weight transfer takes 60 times the ideal compute time. Our weight-only latency bound is **8 ms per step**, corresponding to a ceiling of **125 tokens/s**. That is an optimistic ceiling, not an expected measured rate.

The ridge point is:

```text
C / BW = 120 × 10¹² / (2 × 10¹²)
       = 60 FLOPs/byte
```

Since `I ≈ B` in this simplified BF16 model, the dense weight multiplications reach the ridge around **B = 60**. This is not a claim that the entire model becomes compute-bound at batch 60.

## Batching improves throughput, not necessarily each user's speed

At batch 16, we do 16 times the dense arithmetic but still ideally read the same 16 GB of weights once. Compute takes about 2.13 ms; the weight-read bound remains 8 ms.

Each step now produces 16 output tokens across 16 sequences. Two different rates matter:

```text
aggregate throughput = B / t_step
per-sequence rate    = 1 / t_step
```

<div class="roofline-table" role="region" aria-label="Weight-only decoding estimates" tabindex="0" markdown="1">

| Batch | Compute time | Weight-read time | Step-time bound | Aggregate ceiling | Per-sequence ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.133 ms | 8 ms | 8 ms | 125 tok/s | 125 tok/s |
| 16 | 2.13 ms | 8 ms | 8 ms | 2,000 tok/s | 125 tok/s |
| 60 | 8 ms | 8 ms | 8 ms | 7,500 tok/s | 125 tok/s |
| 120 | 16 ms | 8 ms | 16 ms | 7,500 tok/s | 62.5 tok/s |

</div>

In this model, going from batch one to batch 16 increases aggregate throughput 16-fold while leaving the per-sequence rate unchanged. Past the ridge, aggregate throughput plateaus and each sequence slows down. Real systems can see latency increase earlier.

This is why a serving result labeled only “tokens per second” is incomplete. I want to know the batch size and whether the number describes the whole server or one response. Time to first token also includes queueing and prefill; inter-token latency describes the pace once generation is underway.

## Why prefill often makes better use of compute

For a linear layer, input shaped `[N, d]` multiplies weights shaped `[d, h]`. Ignoring activation traffic, its arithmetic intensity is roughly:

```text
F ≈ 2Ndh
M ≈ sdh
I ≈ 2N / s
```

During prefill, `N` can include many prompt positions. During decode, it is typically the number of active sequences. This gives prefill more opportunities to reuse weights and perform large matrix multiplications.

“Prefill is compute-bound” is still a tendency, not a universal rule. Short prompts, small matrix dimensions, attention kernels, and activation traffic can change the limit. NVIDIA's [matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) connects matrix dimensions and arithmetic intensity to these performance regimes.

## The missing term: reading the KV cache

The weight-only model becomes too optimistic as context grows. Full attention reads historical keys and values on every decode step. For equal-length sequences, an idealized cache read is:

```text
KV bytes per cached token = 2 × L × H_kv × d_h × s_kv
KV read bytes per step    ≈ B × T × KV bytes per cached token
```

Here `L` is the number of layers, `H_kv` the KV-head count, `d_h` the head dimension, `s_kv` bytes per cached element, and `T` the cached context length. The factor two accounts for both keys and values. Unlike weights, these cached tensors generally differ across requests, so increasing the batch also increases cache traffic.

Extend our example with 32 layers, 8 KV heads, head dimension 128, and BF16 cache entries:

```text
KV bytes per cached token = 2 × 32 × 8 × 128 × 2
                          = 131,072 bytes

At B = 16 and T = 8,000:
KV read per step ≈ 16 × 8,000 × 131,072
                 = 16.777216 GB

weights + KV reads ≈ 32.777216 GB per step
memory-time bound ≈ 16.39 ms
```

The memory bound alone now limits aggregate throughput to about **976 tokens/s** and the per-sequence rate to about **61 tokens/s**, before cache writes and other costs. Our earlier 2,000-token/s ceiling missed more than half the relevant traffic.

<figure class="visual-diagram">
  <img src="/assets/images/inference-kv-traffic.svg" alt="At batch 16, weight reads remain 16 GB per decode step. For contexts of 1,000, 4,000, 8,000, and 16,000 tokens, ideal KV reads rise from about 2.1 to 8.4, 16.8, and 33.6 GB." />
  <figcaption>For this cache configuration, KV reads overtake weight reads at roughly 7,629 cached tokens per sequence. Longer context changes the bottleneck without changing the weights.</figcaption>
</figure>

Attention also adds computation. For this example with 32 query heads, its two main matrix products cost approximately `4 × L × H_q × d_h × B × T`, or 67.1 GFLOPs per step. Together with the 256 GFLOPs of dense work, that gives a compute bound of about 2.69 ms—still below the 16.39 ms memory bound.

These calculations assume efficient reuse of shared KV heads. Real traffic depends on kernels and caching. They also assume enough memory: weights plus the batch's KV cache already occupy about 32.8 GB, before workspace and other allocations. Cache *capacity* determines what fits; cache *bandwidth* helps determine how fast it runs.

## How I would use this before optimizing

First, specify the workload: model, precision, prompt lengths, generated lengths, and concurrent requests. Then keep three questions separate:

1. **Does it fit?** Account for weights, KV cache, and workspace.
2. **What is the optimistic limit?** Estimate operations and traffic, then compare compute and memory times.
3. **Where does measured time go?** Profile prefill and decode separately, including scheduling, kernels, and any communication between devices.

The estimate suggests what to investigate. If weight reads dominate, batching or weight quantization may help. If KV reads dominate, cache precision and architectures with fewer KV heads become relevant. If dense compute dominates, kernel efficiency and the appropriate compute path matter. Each change has its own quality, memory, and latency tradeoffs.

A single roofline for an entire decode step can hide mixed bottlenecks: a compute-limited linear kernel and a bandwidth-limited attention kernel execute in sequence. Summing bounds for those stages gives a more informative estimate than assuming all their costs overlap perfectly.

The useful habit is to ask **how much arithmetic happens for each byte moved**. It explains why the same model behaves differently during prefill and decode, why batching helps, and why long context can consume the gains. The next question is how MHA, GQA, and MQA change that cache traffic.
