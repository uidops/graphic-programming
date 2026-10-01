# Kernel Execution Model — Concept Notes

The dynamic view of one kernel launch: what happens between the host call and
the retirement of the last thread. Companion to [BASIC.md](BASIC.md), which
describes the static architecture; this file describes the machinery in motion.

Notes compiled from:

- [Nsight Compute Profiling Guide — Hardware Model](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  (SM sub-partitions, warp lifecycle, stall reasons, sector/cache-line model)
- [CUDA Programming Guide — Advanced Kernel Programming](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
  and §1.2 Programming Model (SIMT, warps, independent thread scheduling,
  asynchronous barriers)
- [Modular Handbook — GPU architecture fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals)
  (threads/warps/blocks/grids, SM, block residency, occupancy, memory)
- [ROCm Compute Profiler — Workgroup processor (WGP)](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html)
  (AMD dispatch, waves, VGPR/LDS limits, wait states)
- Two arXiv studies on control flow: [Control Flow Management in Modern GPUs](https://arxiv.org/html/2407.02944v1)
  and [Characterizing Warp Divergence from Pascal to Blackwell](https://arxiv.org/html/2607.23402v1)
- Apple M3 Pro device limits measured directly (see BASIC.md §3)

NVIDIA hardware is used wherever a concrete number is needed, because it is the
most documented; AMD and Apple equivalents are named at each stage. Stage
numbers ①–⑩ below follow the pipeline order.

---

## 0. The governing rule

A global-memory load takes ~400–800 cycles. An instruction issue slot takes one
cycle. The hardware never closes that gap by waiting — it hides it. The whole
execution model exists to keep **more warps resident than there are slots to
fill**, so that whenever one warp stalls, another is ready to issue:

> Latency is not reduced; it is **overlapped**. The warp that is waiting is not
> the warp being issued, and no state moves to make that swap.

Everything in stages ①–⑩ is in service of that sentence.

```
HOST                                GPU
────                                ───
kernel<<<grid, block>>>(...)        ① enqueue command (stream / queue / command buffer)
                                    ② distribute: grid → blocks → compute units
                                    ③ admit: fit test → warps formed, resources carved
                                    ④ warp state: always-resident contexts
                                    ⑤ issue: schedulers pick ready warps each cycle
                                    ⑥ SIMT execute → ⑦ memory path
                                    ⑧ synchronize (block barrier / kernel end)
                                    ⑨ occupancy governs how many fit at once
                                    ⑩ retire: warp → block → grid → event → host
```

---

## ① The launch: what the host actually hands over

```cpp
vec_add<<<16, 256>>>(d_a, d_b, d_c, n);   // returns immediately (asynchronous)
```

The host does not run the kernel. It writes a small command into a queue — a
CUDA stream, a HIP stream, a WebGPU queue, a Metal command buffer — containing
roughly:

```
{ kernel handle, grid dims, block dims, argument values,
  shared-memory size / launch attributes, completion signal }
```

The kernel handle is the compiled binary (PTX has been compiled to a cubin and
loaded earlier); the driver parses the launch configuration and issues a GPU
command the front end can decode. The CPU is then free until it explicitly
waits (`cudaDeviceSynchronize`, `queue.onSubmittedWorkDone()`,
`waitUntilCompleted`, or a Metal completion handler).

**What is specified vs. what is decided**

| chosen by the program | meaning | never chosen by the program |
|---|---|---|
| grid = 16 blocks | total parallelism — how many blocks exist | which compute unit runs which block |
| block = 256 threads | cooperation unit — threads sharing shared memory + barrier | in what order blocks start or finish |
| shared mem / register config | residency limits (stage ③) | which warp issues next, when |

Two structural rules fall out, both stated in the vendor docs:

1. **Blocks are independent.** "Each CTA can be scheduled on any of the
   available SMs, where there is no guarantee in the order of execution. As
   such, CTAs must be entirely independent" — no ordering, no communication,
   no waiting on another block's result
   ([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
   A grid larger than the hardware fits is still legal: the architecture runs
   what fits and queues the rest.
2. **Threads of one block cooperate.** The architecture guarantees all threads
   of a block run concurrently **on the same SM**, which is what makes fast
   shared memory and a barrier meaningful within a block only.

Ordering exists only **within one stream/queue**: commands are issued in order,
and a later command sees all writes of an earlier one after it completes.

---

## ② Distribution: grid → blocks → compute units

A front-end hardware distributor feeds queued blocks into compute units that
have spare capacity:

| vendor | distributor | compute unit |
|---|---|---|
| NVIDIA | GigaThread engine (front end) | Streaming Multiprocessor (SM) |
| AMD | Command Processor → Workgroup Manager (SPI) | Workgroup Processor = 2 CUs (RDNA), or a CU (CDNA) |
| Apple | command processor (not publicly documented) | GPU core / threadgroup scheduler |

Facts that hold everywhere:

- **Not all blocks run at once.** Nsight Compute defines the **wave**: the
  number of blocks that run concurrently on the whole GPU. If a grid has
  10,000 blocks and the wave is 800, the other 9,200 wait in queue.
- **Placement is arbitrary.** 16 blocks may land 4/4/4/4 on four SMs or
  unevenly; the program may not assume anything. AMD's SPI hands workgroups to
  WGPs "after the Workgroup Manager hands off work"
  ([ROCm](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html)).
- **No migration.** "The GPU assigns each thread block to one SM. Under normal
  execution, a block remains on that SM until completion"
  ([Modular Handbook](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors));
  its shared memory and registers are physically allocated there.
- **Kernels can overlap.** An SM "is designed to simultaneously execute
  multiple CTAs. CTAs can be from different grid launches" — two kernels in
  flight can share the machine without observing each other.

---

## ③ Admission and residency: the fit test

When a block reaches the head of the queue, the target unit checks whether the
whole block fits. The limiter list from the Nsight Compute hardware model:
**threads, registers, shared memory, hardware barriers** (plus blocks-per-SM
and architecture-specific limits in the
[Modular list](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors#block-residency)).

- **Warp formation.** The 256 threads are split by index into groups of 32:
  {0–31}, {32–63}, …, {224–255} → 8 warps. A 300-thread block makes 10 warps,
  the last one partial — its idle lanes still consume a slot, which is why a
  thread count that is a multiple of 32 is recommended.
- **Register allocation.** A sub-partition's registers are granted in
  fixed-size chunks (per-warp granularity), so the effective register count is
  rounded up before the budget is checked. On NVIDIA parts the SM holds 65,536
  32-bit registers (4 sub-partitions × 16,384).
- **Shared-memory allocation.** Whatever `__shared__` / `var<workgroup>` /
  `[[threadgroup_memory]]` the block declared is carved out of the unit's
  pool (up to 228 KB SMEM on an H100 SM).
- **All or nothing.** Because all threads of the block must be concurrently
  resident on one unit, a block that does not fit waits for a later slot —
  there is no partial admission and no migration.

**AMD variant:** the SPI allocates VGPRs (vector registers per wave), SGPRs
(scalar/uniform registers), LDS bytes per workgroup and scratch bytes per
work-item before the wave runs; any of them can be the binding limiter
([ROCm WGP docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html)).

**Apple variant (measured on M3 Pro):**

| limit | value |
|---|---|
| `max-compute-invocations-per-workgroup` | 1024 threads/threadgroup |
| `max-compute-workgroup-size-x` | 1024 |
| `max-compute-workgroups-per-dimension` | 65535 |
| `max-compute-workgroup-storage-size` | 32768 (32 KB shared) |

The result of admission is the **resident warp pool** — the pool stage ⑤ draws
from. Its size is what occupancy later measures.

---

## ④ Warp state: why switching costs nothing

Each resident warp owns a complete execution context that never leaves the
chip while the warp is alive:

```
warp slot = { PC(s) — per-lane since Volta, 32×R registers, active/predicate mask,
              scoreboard entries, convergence-barrier state, stall state }
```

Three architectural facts:

1. **A warp is pinned for life.** "A warp is allocated to a sub-partition and
   resides on the sub-partition from launch to completion"
   ([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
   A block's warps are spread across the SM's four sub-partitions and never
   move.
2. **There is no context switch.** "Registers and scheduling state for
   resident warps are already present on the SM" — switching means only that
   the issue stage points at a different warp's registers
   ([Modular](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors#how-warp-scheduling-hides-latency)).
   No save/restore, no cache cold-start, no pipeline flush. This is the
   difference from a CPU thread switch, which is memory traffic.
3. **Registers are the occupancy ceiling.** 65,536 registers / (32 × R per
   thread): a 128-register kernel can keep at most 16 warps resident per SM.
   Fewer resident warps = fewer latency-hiding alternatives.

**The hidden cost of register pressure:** a spill does not go to a small
on-chip buffer — local memory is per-thread and **resides in device memory**,
so spilled accesses behave like global loads (hundreds of cycles) and show up
in the memory-path analysis ([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

Preemption exists — "the execution context (registers, shared memory, etc.) is
saved at preemption and restored later, at instruction-level granularity" —
but it serves the OS/graphics, not in-kernel scheduling.

---

## ⑤ Instruction issue: the cycle-by-cycle heart

An SM is divided into four **sub-partitions** (also called processing blocks /
SMSPs). Each contains its own warp scheduler, register file, and execution
pipelines (integer, floating point, load/store, special function, tensor);
the four share the unified L1/shared-memory pool, texture units, and RT cores
([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

**Eligibility** — a warp is *eligible* when it has a decoded instruction, all
input dependencies resolved, and the needed functional unit is free. Every
cycle, each scheduler independently picks at most one eligible warp:

```
for each of the 4 schedulers:
    eligible warps = { decoded instr, scoreboard clear, unit free }
    pick ONE; issue its next instruction to a pipe; advance its PC
    if none eligible -> that scheduler idles that cycle (a bubble)
```

One SM per cycle: 4 warp-instructions × 32 lanes = up to 128 lane-ops. When a
warp cannot issue, the scheduler silently issues a different warp instead —
zero-cost swap, no state moved. That is latency hiding, mechanically.

**The scoreboard** marks a warp ineligible until its operands exist: a load
issued N cycles ago has not written its destination register yet, so the
consumer stalls; when the data arrives, the warp re-enters the eligible set.

### Stall taxonomy (authoritative, from Nsight Compute)

Sampling of warp scheduler states; `selected` and `not_selected` are *not*
stalls — `not_selected` means the warp was eligible but another was picked
that cycle.

| stall reason | waiting on | typical cause / lever |
|---|---|---|
| `long_scoreboard` | L1TEX result (global/local/texture) | memory latency — fix coalescing, locality, or add ILP |
| `short_scoreboard` | MIO operation, mainly **shared memory** | bank conflicts, MUFU ops, dynamic branches |
| `barrier` | sibling warps at a block barrier | divergence/skew before `__syncthreads()` |
| `wait` | fixed-latency pipe (e.g. FFMA = 4 cycles) | dependency chains — unroll, restructure |
| `not_selected` | — (scheduler state) | other warps issued first; high value = healthy pool |
| `branch_resolving` | branch target / PC update | too many jumps, heavy divergence |
| `math_pipe_throttle` | math pipe busy | compute-bound — more warps or rebalance mix |
| `mio_throttle` | MIO queue full | shared/branch pressure — fewer, wider loads |
| `lg_throttle` | L1 local/global queue full | excessive global/local traffic or register spills |
| `tex_throttle` | L1 texture queue full | texture accepts only 4 threads/cycle vs 32 for global |
| `no_instructions` | instruction fetch / I-cache miss | large kernels, giant strides; also grids smaller than one wave |
| `membar` | memory fence | overuse of `__threadfence*` |
| `drain` | outstanding memory ops after `EXIT` | large end-of-kernel stores |
| `dispatch_stall` | dispatcher holds issue (conflicts/events) | front-end contention |
| `warpgroup_arrive` | Hopper `WARPGROUP.ARRIVES/WAIT` (WGMMA) | skew across the four schedulers |

Two interpretation rules from the same documentation:

- "The fundamental optimization target is increasing **issue-slot utilization**,
  not driving every stall counter to zero."
- Samples are attributed to the **consumer** instruction waiting on a
  dependency, not the producer — read stall data with the source/SASS context.

**AMD issue path:** a WGP pairs two CUs (sharing LDS and instruction cache),
each with dual SIMD32 units; a wave issues vector ALU (VALU), scalar ALU
(SALU), scalar memory (SMEM), vector memory (VMEM) and LDS instructions, with
optional dual-issue (VOPD) in wave32. Wait states are reported as
*instruction fetch*, *barrier*, and *counter* (memory) — the same three
waiting categories as NVIDIA's taxonomy
([ROCm](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html#wait-state-analysis)).

**Apple:** scheduler internals are not published; what is observable is the
simdgroup width (32, measured `threadExecutionWidth`), the resource limits of
stage ③, and command-buffer timing.

---

## ⑥ SIMT execution: one instruction, 32 lanes

The issued instruction reaches a 32-lane datapath. Lane *i* applies it to thread
*i*'s registers. Every active lane writes its own destination; lanes masked off
by divergence are suppressed. The pipeline runs the same instruction stream for
all lanes, so branches resolve **per lane**:

- **Pre-Volta:** one PC per warp plus an implicit reconvergence stack; paths
  execute serialized and lanes rejoin at the immediate post-dominator. A lane
  spinning on a lock could deadlock against a lane in the same warp that holds
  it — the structural flaw of lockstep.
- **Volta and later (Independent Thread Scheduling):** each lane has a private
  PC and call stack; the compiler emits explicit convergence machinery —
  `BSSY` (arm a convergence barrier), `BSYNC` (reconverge), `BREAK` (drop an
  exiting lane) over a small file of barrier registers — and `__syncwarp()`
  for programmer-requested convergence
  ([CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html),
  [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)).
  The practical consequence: lockstep assumptions from older code are invalid;
  intra-warp data exchange needs explicit synchronization.

**Cost of divergence** (measured across Pascal → Blackwell): paths serialize
linearly, `T(k) ≈ s·k` for k paths; per-warp efficiency falls as `32/k`; and
**occupancy does not change the cost** — more warps cannot hide serialization
inside a warp
([arXiv 2607.23402](https://arxiv.org/html/2607.23402v1)).
Small conditionals are recovered by **predication** (mask the branch instead of
taking it), which the compiler does when both sides are cheap.

Divergence before a barrier is the main source of `barrier` stalls: fast warps
arrive and idle while stragglers take the slow path — the barrier itself is
cheap when all warps arrive uniformly
([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

**The divergence-free alternative:** tile programming (CUDA 12+) has the whole
block follow one control flow, with the compiler distributing tile operations
across threads — no concept of warp divergence at all
([CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)).

---

## ⑦ The memory path of one instruction

For `c[i] = a[i] + b[i]` executed by one warp:

```
32 lanes produce 32 addresses
        │
        ▼
coalescing / L1TEX unit (load-store pipelines)
   ├─ contiguous + aligned → 1 request, 4 sectors = 128 B cache line
   └─ scattered           → up to 32 cache lines, serialized as pipeline "wavefronts"
        │ miss
        ▼
L1 / shared-memory pool (per SM)     ~20–30 cycles
        │ miss
        ▼
L2 (chip-wide, shared by all SMs)    ~200 cycles   (H100: ~50 MB, ~12 TB/s)
        │ miss
        ▼
HBM / DRAM                          ~400+ cycles  (H100: 80 GB, 3.35 TB/s)
        on Apple Silicon: unified system RAM
```

The unit of traffic is the **sector**: an aligned 32-byte chunk; a cache line
(L1 and L2 both) is four sectors = 128 bytes. One warp load is one *request*;
the request expands into `1:N` sectors, and the L1TEX pipeline processes at
most one internal *wavefront* per cycle — so a scattered access pattern costs
both extra sectors **and** extra pipeline cycles
([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
32 consecutive floats = exactly 128 bytes = one cache line: the ideal case.

Latency table (sizes/latencies for H100, from the
[Modular memory page](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory)):

| level | latency | scope | managed by |
|---|---|---|---|
| registers | ~1 cycle | per thread | compiler |
| shared memory / L1 | ~20–30 cycles | per block / per SM | programmer / hardware |
| L2 | ~200 cycles | whole chip | hardware |
| HBM | ~400+ cycles | global | programmer / runtime |

The two scoreboards of stage ⑤ are exactly a statement about *where* data
lives: `long_scoreboard` = waiting on the L1TEX path (global), `short_scoreboard`
= waiting on the MIO path (shared memory and special functions).

Two finer points:

- **Constant/immediate reads serialize per distinct address**: if lanes of a
  warp read different constant addresses, each unique address is a separate
  access; if all lanes read the same address, it costs about a register read
  ([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
- **Shared memory breaks the trip**: the block stages its tile once
  (`ld.shared`), every thread reads it at ~20 cycles, and the block barrier
  makes those writes visible across warps. The tiling discipline every
  efficient kernel follows — global → shared → registers → math — exists to
  minimize trips down the far end of this diagram.

---

## ⑧ Synchronization inside the model

The model defines exactly two synchronization points:

1. **Block barrier** (`__syncthreads()` / `workgroupBarrier()` /
   `threadgroup_barrier()`): every thread of the block must arrive before any
   proceeds, and shared-memory writes before the barrier are visible to the
   whole block afterward. It works because the block's threads are guaranteed
   to be on one unit (stage ①). Mechanically it is a counted hardware barrier —
   barrier slots are one of the admission limiters in stage ③, and an
   undersubscribed or unbalanced block can exhaust them.
2. **Kernel end**: an implicit grid-wide barrier. When the command completes,
   every thread of every block has finished; the next command in the same
   stream/queue observes all writes — and this happens whether or not the CPU
   is woken.

**There is no inter-block synchronization during a kernel.** Blocks may be
scheduled in any order and any subset may be resident, so spinning in one block
until another block finishes can deadlock (the other block may be queued). The
supported ways to get grid-wide ordering are:

- launch a dependent kernel on the same stream/queue (kernel end does the work);
- a cooperative launch (CUDA Cooperative Groups grid-wide sync), which requires
  the whole grid to be resident simultaneously.

Cross-thread correctness on shared addresses additionally needs **atomics**
(`atomicAdd`, `atomic_fetch_add`): warp scheduling does not make a
read-modify-write sequence atomic — two warps writing the same address
serialize nondeterministically.

An evolution worth knowing: **asynchronous barriers** split the barrier into an
*arrive* point and a *wait* point (`cuda::barrier`, hardware-accelerated on
compute capability 8.0+), so a thread can keep doing unrelated work while
waiting instead of parking at `__syncthreads()`
([CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)).

---

## ⑨ Occupancy: the governor of the whole machine

**Occupancy** = resident warps / maximum active warps per unit
([Modular](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors#occupancy)):

```
warps_per_SM = min( hardware cap,                                   // e.g. 64 warps
                    65536 / (32 × registers_per_thread),            // register budget
                    shared_pool / shared_per_block × warps_per_block, // shared budget
                    thread_limit / threads_per_block )              // thread budget
occupancy   = warps_per_SM / warps_cap
```

(After rounding for allocation granularity — see stage ③. On AMD the same
computation runs against VGPR capacity, LDS bytes and wave-count limits; the
[ROCm docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html)
list VGPRs, SGPRs, LDS and scratch as the occupancy inputs.)

Occupancy has three distinct meanings in profiling vocabulary
([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)):

- **occupancy** — warps resident vs. the cap;
- **CTA occupancy** — blocks resident on one SM, limited by threads,
  registers, shared memory, barriers;
- **wave** — blocks resident on the whole GPU at once (sets how many can
  execute concurrently across the device).

**It is a means, not a goal.** High occupancy gives the scheduler a larger
pool of alternatives; it cannot help if every warp waits on the same
bandwidth wall, and a kernel that keeps a large tile in shared memory may
deliberately run at lower occupancy and win — FlashAttention is the standard
example ([Modular](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors#is-low-occupancy-always-a-problem)).
Nsight's guidance is the same from the other side: the target is issue-slot
utilization, not the stall counters themselves.

**Wave quantization and tails.** If the grid does not divide evenly into
waves, the last wave runs partially idle; a grid smaller than one wave cannot
fill the machine at all (it surfaces as `no_instructions`). Grids should
expose enough independent blocks to cover all units.

---

## ⑩ Retirement and completion

- **Warp retires** when all its threads reach `return`/`exit`. The hardware
  first *drains*: "stalled after EXIT waiting for all outstanding memory
  operations to complete so that the warp's resources can be freed" — a kernel
  that writes heavily at the end shows `drain` samples
  ([Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
- **Block retires** when all its warps retire: shared memory, warp slots,
  register allocation and barrier slots are released, and the distributor
  admits the next waiting block — a rolling window, the grid behaving like a
  conveyor belt of blocks.
- **Grid completes** when the last block retires → the command completes →
  the queue signals an event → a dependent command in the same stream proceeds
  (without waking the CPU), or the host's blocking wait returns.

Nothing about this order is observable from inside the kernel: which block
started first, which warp issued first, where anything ran — all of it is
hardware state, not program semantics.

---

## Worked example: vector add

```
launch:   grid 16 blocks × 256 threads = 4096 threads = 128 warps total
per block: 256 / 32 = 8 warps;  24 registers/thread → 24 × 32 = 768 regs/warp

per-SM budget (NVIDIA example): 65,536 registers
  by registers: 65,536 / 768 ≈ 85 warps  → capped by hardware cap (64 warps)
  by threads:   e.g. 1024 threads/SM → 4 blocks of 256
  shared memory: 0 bytes → not a constraint
  ⇒ binding limiter: thread/block slots

timeline on a 2-SM machine, 4 blocks per SM:
  t=0    SM0 resident: blocks 0–3, SM1: blocks 4–7   (64 / 128 warps resident)
  ...    schedulers pick eligible warps each cycle; loads coalesce to
         4 sectors (128 B) per warp; stalled warps swapped at zero cost
  t=1    blocks 0–3 retire → blocks 8–11 admitted (rolling window)
  t=end  last block retires → event → completion handler / read back
```

At no point does the program choose an SM, a start time, or an issue order —
that separation of concerns is the execution model.

---

## Observing each stage

| stage | NVIDIA | AMD | Apple |
|---|---|---|---|
| ① launch | `nsys` timeline, launch statistics | rocprofiler dispatch stats | Instruments → Metal System Trace |
| ②③ distribution/admission | Nsight *Launch Statistics*, *Occupancy* sections (limiting resource named explicitly) | WGP panels: wave dispatch, VGPR/SGPR/LDS allocation | device limits (probe in BASIC.md §3) |
| ⑤ issue | *Scheduler Statistics* (active / eligible / issued warps), *Warp State Statistics* (stall table above) | *Wait state analysis* (fetch / barrier / counter) | not exposed |
| ⑥ divergence | source-level SASS view, predication counts | — | — |
| ⑦ memory | *Memory Workload Analysis* (sectors, cache lines, hit rates) | GL0/GL1/GL2/LDS bandwidth panels | — |
| ⑨ occupancy | *Occupancy* section | WGP utilization, wave life | measured limits |

---

## Invariants and common misconceptions

- Block start order, placement and completion order are **undefined** — any
  code depending on them is broken on every vendor.
- A stall counter driven to zero is not a goal; issue-slot utilization is.
  `not_selected` and `selected` are scheduler states, not stalls.
- Stall samples name the **consumer** instruction, not the guilty producer.
- Divergence cost is fixed by the number of paths — occupancy does not reduce
  it; predication does.
- Barrier samples are almost always **skew** (stragglers), not barrier overhead.
- More resident warps only help if they are *different* work: if all wait on
  the same saturated resource, extra occupancy adds nothing.
- Register spills are not "slightly slower registers" — they are DRAM
  transactions (stage ⑦).
- Kernel end is the only grid-wide barrier (unless a cooperative launch was
  used); cross-block spinning can deadlock.
- A thread count below a multiple of 32 wastes lanes in the final warp.

---

## The same model, several names

| concept | CUDA (NVIDIA) | HIP (AMD) | Metal (Apple) | WGSL (WebGPU) |
|---|---|---|---|---|
| kernel | `__global__` fn | `__global__` fn | `kernel` fn | `@compute` fn |
| launch | `<<<grid, block>>>` | `<<<grid, block>>>` | `dispatchThreadgroups` | `dispatchWorkgroups` |
| grid | grid | grid | grid of threadgroups | grid of workgroups |
| block | thread block (CTA) | workgroup | threadgroup | workgroup |
| scheduling unit | warp (32) | wavefront (64 GCN/CDNA; 32 or 64 RDNA) | simdgroup (32, measured) | subgroup (32) |
| shared memory | `__shared__` | LDS (`__shared__`) | threadgroup address space | `var<workgroup>` |
| barrier | `__syncthreads()` | `__syncthreads()` / `s_barrier` | `threadgroup_barrier()` | `workgroupBarrier()` |
| thread index | `threadIdx` / `blockIdx` | same | `[[thread_position_in_grid]]` | `global_invocation_id` |
| compute unit | SM (+4 sub-partitions) | CU (a WGP pairs 2 CUs on RDNA) | not published (limits measured) | implementation-defined |
| distributor | GigaThread engine | Command Processor + Workgroup Manager (SPI) | command processor | implementation-defined |
| async queue | stream | stream | command buffer | queue |

---

## References

- [Nsight Compute Profiling Guide — Hardware Model & Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUDA Programming Guide — Advanced Kernel Programming (warp divergence, Independent Thread Scheduling, asynchronous barriers)](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [Volta Tuning Guide — Independent Thread Scheduling](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- [Modular Handbook — GPU threads, warps, blocks, grids](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/threads-warps-blocks)
- [Modular Handbook — Streaming multiprocessors (block residency, occupancy)](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/streaming-multiprocessors)
- [Modular Handbook — GPU memory hierarchy](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory)
- [ROCm Compute Profiler — Workgroup processor (WGP)](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.0.0/conceptual/rdna/wgp.html)
- [The CUDA Handbook — 2.6 GPU Architecture (GigaThread front end)](https://www.cudahandbook.com/book/ch2/gpu-architecture)
- [arXiv — Control Flow Management in Modern GPUs (BSSY/BSYNC mechanics)](https://arxiv.org/html/2407.02944v1)
- [arXiv — Characterizing Warp Divergence from Pascal to Blackwell](https://arxiv.org/html/2607.23402v1)
