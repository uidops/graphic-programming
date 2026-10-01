# Graphic / GPU Computing — Self-Study

A personal study plan for learning how GPUs actually execute code, built around
three resources that complement each other:

| Resource | Role in this study |
| --- | --- |
| [GPU Architecture Fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals) (Modular LLM Inference Handbook) | **Theory** — the mental model of hardware + execution model |
| [GPU Glossary](https://modal.com/gpu-glossary/) (Modal) | **Reference** — precise definitions, looked up on demand, not read cover-to-cover |
| [GPU Puzzles, Mojo 🔥 Edition 1](https://puzzles.modular.com/) (Modular) | **Practice** — hands-on kernels, from raw memory to Tensor Cores |

> "For the things we have to learn before we can do them, we learn by doing them."
> — Aristotle, *Nicomachean Ethics* (quoted on the Puzzles landing page)

Each concept gets three passes: **read the theory → solve the puzzle → look up the
exact terminology**. Nothing counts as learned until it has been written in code.

---

## Why these three fit together

- The **Handbook** explains the *why*: SMs, warps, occupancy, the memory hierarchy,
  Tensor Cores. Without it, advice like "increase occupancy" is a rule to memorize
  rather than a trade-off you understand.
- The **Puzzles** force the *how*: 35 progressive challenges in Mojo/MAX where every
  concept from the Handbook shows up as an indexing, memory, or scheduling problem.
- The **Glossary** is the *lookup layer*: when a puzzle says "bank conflict" or
  "scoreboard stall", there is a short, exact definition waiting — organized in three
  tiers (device hardware, device software, performance) that map onto the Handbook's
  hardware/execution split.

Note: all three resources are CUDA/NVIDIA-centric in vocabulary (SM, warp, PTX,
HBM), while the puzzles are written in Mojo. Concepts transfer; the API names don't.
This repo also contains a `example.mojo` / `01/example.mojo` starting point built on
`max.gpu` (`DeviceContext`, `global_idx`, `LayoutTensor`) — that is the API the
puzzles use.

---

## Core concepts to master

The Handbook's central claim: **kernel optimization = mapping the execution model
onto the hardware while respecting limits on compute, memory, registers, shared
memory, and scheduling capacity.** Everything below is one side or the other of
that sentence.

### Execution model (how work is organized)
- **Thread** — smallest logical unit of work; one element (or a small group) per thread
- **Warp** — 32 threads scheduled together, executed in lockstep (SIMT)
- **Thread block** — cooperative group sharing memory and synchronization
- **Grid** — all blocks of a kernel launch
- **Kernel** — the function launched across the grid

### Hardware (what physically exists)
- **SM (streaming multiprocessor)** — on-chip compute unit: warp schedulers,
  register file, shared memory, L1 cache; accepts blocks, issues ready warps
- **Registers** — private per thread, fastest tier
- **Shared memory / L1** — per-block programmer-managed / per-SM hardware-managed
- **L2 cache** — shared across all SMs
- **HBM (global memory / VRAM)** — largest, slowest, ~400+ cycle latency;
  kernel optimization mostly means minimizing round-trips here
- **Tensor Cores** — tiled matrix operations with precision/shape/layout constraints

### Performance vocabulary (is it fast, and if not, why not)
- **Roofline model**, arithmetic intensity, compute-bound vs. memory-bound
- **Occupancy** vs. **latency hiding** vs. **issue efficiency**
- **Memory coalescing**, **bank conflicts**, **register pressure**
- **Warp divergence**, **branch efficiency**, **scoreboard stalls**

---

## Study plan

### Phase 0 — Orientation (½ day)
- [ ] Read: Handbook → [GPU architecture](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals)
- [ ] Read: Puzzles → Introduction, usage guide, rewards
- [ ] Skim: Glossary table of contents — know *where* things live, don't read them yet
- [ ] Get the environment running: solve nothing, just execute `example.mojo`
- [ ] Append `.md` to any Handbook URL for a clean markdown version; see `llms.txt` for the full index

### Phase 1 — Threads, blocks, raw memory (Puzzles 1–8)
- [ ] Read: Handbook → [GPU threads, warps, blocks, and grids](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/threads-warps-blocks)
- [ ] Glossary: Thread, Warp, Kernel, Thread Block, Thread Block Grid, Memory Hierarchy
- [ ] Puzzles 1–4: Map, Zip, Guards, 2D Map — indexing + bounds guards
- [ ] Puzzles 5–7: Broadcast, Blocks, 2D Blocks
- [ ] Puzzle 8: Shared memory — first taste of inter-thread cooperation
- [ ] **Checkpoint:** write your own elementwise kernel from scratch (no puzzle prompt) and explain why the `if tid < N` guard exists

### Phase 2 — Debugging (Puzzles 9–10)
- [ ] Puzzles 9: debugger workflow, three detective cases
- [ ] Puzzles 10: sanitizers — memory violations, race conditions
- [ ] Glossary: PTX, SASS, compute capability, Nsight Systems, CUDA Profiling Tools Interface
- [ ] **Checkpoint:** deliberately introduce a race in a shared-memory kernel, find it with tooling, fix it

### Phase 3 — Parallel algorithms (Puzzles 11–16)
- [ ] Read: Handbook → [Streaming multiprocessors](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors) (block residency, occupancy)
- [ ] Glossary: SM, Warp Scheduler, Occupancy, Latency Hiding, Active Cycle
- [ ] Puzzles 11–13: Pooling, Dot Product, 1D Convolution (simple → block-boundary)
- [ ] Puzzles 14–15: Prefix Sum (scan), Axis Sum — the two hard primitives
- [ ] Puzzle 16: MatMul — naïve → shared memory → tiled; then read the Handbook's roofline intro
- [ ] Glossary: Roofline Model, Arithmetic Intensity, Compute-bound, Memory-bound
- [ ] **Checkpoint:** state your MatMul's arithmetic intensity and say whether it is roofline-limited by bandwidth or compute

### Phase 4 — Memory hierarchy (deep dive)
- [ ] Read: Handbook → [GPU memory hierarchy](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory)
- [ ] Glossary: Registers, Shared Memory, Global Memory, Bank Conflict, Memory Coalescing, Register Pressure
- [ ] Puzzles 21, 30–32, 35: coalescing comparison, profiling, cache-hit paradox, occupancy, bank conflicts, alignment
- [ ] **Checkpoint:** profile one of your own kernels and identify its actual bottleneck (not the one you assumed)

### Phase 5 — Real operations (Puzzles 17–22)
- [ ] Puzzles 17–19: MAX Graph ops — 1D Conv, Softmax, Attention
- [ ] Puzzles 20–22: PyTorch integration — CustomOpLibrary, `torch.compile`, kernel fusion, custom backward pass
- [ ] Glossary: Performance Bottleneck, Overhead, Little's Law, Pipe Utilization
- [ ] **Checkpoint:** fuse two elementwise ops and measure the speedup yourself

### Phase 6 — Warp & block-level programming (Puzzles 23–29)
- [ ] Puzzles 23: functional patterns — elementwise, tile, vectorize; SIMD vs. GPU threading
- [ ] Puzzles 24–26: warp sum, `shuffle_down`, `broadcast`, `shuffle_xor` butterfly, warp prefix sum
- [ ] Puzzles 27: block-wide `sum` / `prefix_sum` / `broadcast`, histogram binning, normalization
- [ ] Puzzles 28–29: async memory ops, copy/compute overlap, double-buffered pipelines, synchronization primitives
- [ ] Glossary: Warpgroup, Warp Execution State, SIMD, CUDA Tile Programming Model

### Phase 7 — Hardware frontiers (Puzzles 33–34)
- [ ] Read: Handbook → [Tensor Cores](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/tensor-cores)
- [ ] Glossary: Tensor Core, TMA (Tensor Memory Accelerator), Tensor Memory, TPC, GPC
- [ ] Puzzle 33: Tensor Core operations + performance bonus challenge
- [ ] Puzzle 34: GPU cluster programming (SM90+) — multi-block coordination, cluster collectives
- [ ] Glossary: CUDA (Device Architecture), compute capability

### Phase 8 — Consolidation
- [ ] Re-read the Handbook's GPU architecture section — it should now read as a
      summary of things you've already done, not new material
- [ ] Re-do Puzzle 16 from memory without looking at the solution
- [ ] Write up: *my model of the memory hierarchy* — one page, no copying
- [ ] Pick a follow-on: write a small wgpu/WebGPU or CUDA kernel for something the
      puzzles didn't cover (ray tracing, a shader, a reduction) to prove the
      concepts are portable beyond Mojo

---

## Progress log

| Date | Puzzle / topic | Notes |
| --- | --- | --- |
|  |  |  |

Keep a one-line entry per session: what you solved, what broke, what you finally
understood. The puzzles are cumulative — a note about *why* something failed is
worth more than the solution.

---

## Repo layout

```
graphic-programming/
├── README.md          # this study plan
├── example.mojo       # minimal max.gpu kernel (scale_kernel)
└── 01/                # numbered exercises go here
    └── example.mojo
```

Run a puzzle/exercise with the Mojo toolchain once installed (see the
[Puzzles usage guide](https://puzzles.modular.com/)); Part II puzzles need `pixi`
and an NVIDIA GPU with CUDA support for the debugging tools.

## Useful links

- Handbook markdown index: <https://handbook.modular.com/llms.txt> (append `.md` to any page)
- Puzzles repo: linked from <https://puzzles.modular.com/>
- Mojo Manual: linked from <https://puzzles.modular.com/>
- Glossary: <https://modal.com/gpu-glossary/>
