# GPU Architecture — Concept Notes

Notes compiled from three sources:

- [Modular Handbook — GPU architecture fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals)
- [CUDA Programming Guide — 1.1 Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [CUDA Programming Guide — 1.2 Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

§2 (the heterogeneous model) is written from Wikipedia and vendor sources,
listed in References. §3–§8 are written vendor-generally: NVIDIA (CUDA),
AMD (HIP/ROCm), and Apple (Metal), with platform-specific facts sourced from
the CUDA Programming Guide, AMD ROCm/HIP docs and MI300X data sheet, and
Apple's Metal documentation (Metal Feature Set Tables, plus limits measured
directly on an Apple M3 Pro).

---

## 1. Why GPUs

The GPU began as fixed-function hardware to accelerate parallel operations in
real-time 3D rendering, then grew more programmable — by 2003 some stages of
the graphics pipeline ran custom code in parallel. Compute APIs then made GPU
throughput available independently of graphics: CUDA (NVIDIA, 2006), OpenCL
(Khronos, 2008), Metal (Apple, 2014), Vulkan compute, DirectCompute
(Microsoft), and ROCm/HIP (AMD, 2016).

CPUs and GPUs are designed for different goals:

- A **CPU** is built to execute a serial sequence of operations (a thread) as
  fast as possible, with a few tens of such threads in parallel; more
  transistors go to data caching and flow control.
- A **GPU** is built to execute thousands of threads in parallel, trading lower
  single-thread performance for much greater total throughput; more
  transistors go to data processing units.

Within a similar price and power envelope a GPU provides much higher
instruction throughput and memory bandwidth than a CPU. FPGAs are also
energy-efficient but far less flexible programmatically.

---

## 2. The heterogeneous model

**Heterogeneous computing** refers to systems that use more than one kind of
processor or core. Performance and energy efficiency come not from adding more
of the same processor, but from adding dissimilar coprocessors with specialized
capabilities for particular tasks ([Wikipedia](https://en.wikipedia.org/wiki/Heterogeneous_computing)).
A **homogeneous** system, by contrast, is many identical cores running the same
work — which is also what a single CPU or a single GPU is internally.

### Forms of heterogeneity

Heterogeneity appears at several levels, from coarse to fine:

- **Different ISAs on one chip** — compute elements with genuinely different
  instruction sets (binary-incompatible). Research reports a heterogeneous-ISA
  chip multiprocessor outperforming the best same-ISA homogeneous design by
  ~21% with ~23% energy savings.
- **Same ISA, different cores** — a heterogeneous CPU topology: high-performance
  "big"/P-cores paired with power-efficient "small"/E-cores. ARM's big.LITTLE
  (now DynamIQ) is the prototypical case, Apple silicon and Intel's Alder Lake
  follow it; some designs add a third "prime" tier. These are technically
  asymmetric multiprocessors, and the point is power efficiency, especially in
  mobile SoCs.
- **SoC integration** — as fabrication shrinks, formerly discrete components
  (GPUs, cryptography co-processors, programmable network processors, A/V
  codecs, DSPs, hardware accelerators) become parts of a single system-on-chip,
  so the level of heterogeneity in modern systems keeps rising.
- **Accelerator devices** — the GPU/FPGA/ASIC class: a CPU plus one or more
  devices of a completely different architecture attached over an interconnect
  (PCIe, NVLink). This is the form relevant to GPU programming.

### Why split the work

CPUs and GPUs are optimized for opposite things: the CPU excels at a serial
sequence of operations with complex control flow (branching, caching, flow
control); the GPU excels at applying the same operation to thousands of data
items at once (SIMD/SIMT). Industry framings of the same idea:

- AMD calls it **accelerated computing**: separate the data-intensive parts of
  an application onto an acceleration device while leaving control
  functionality on the CPU, so each processor does what its hardware is
  efficient at ([AMD](https://www.amd.com/en/resources/articles/what-is-accelerated-computing-and-why-is-it-important.html)).
- IBM: **accelerated computing** is the use of specially designed hardware and
  software — GPUs, ASICs, FPGAs — to speed up computing tasks
  ([IBM](https://www.ibm.com/think/topics/accelerated-computing)).

**GPGPU** is the historical name for the GPU side of this: using a graphics
processor for computation traditionally done on the CPU. Programmable shaders
(~2001) made it practical; in 2003 two research groups independently showed
GPU-based linear algebra beating CPUs; early work required recasting problems
as graphics primitives over OpenGL/Direct3D until CUDA (2006), OpenCL,
DirectCompute, and later ROCm (AMD, 2016) let programmers write ordinary
compute code. Most TOP500 systems today use GPUs
([Wikipedia: GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units)).

### The cost of the split

The moment compute is spread across devices, data has to move between them,
and this is where the model's main burdens appear:

- **Programmer-managed data movement** — with traditional CUDA/OpenCL, host and
  device memories are disjoint and the programmer plans every transfer.
- **Non-uniformity** — different ISAs break binary compatibility; ABIs may
  differ in endianness, calling convention, memory layout; libraries and OS
  services are not equally available on every element; cache structures and
  coherency differ between device types
  ([Wikipedia: challenges](https://en.wikipedia.org/wiki/Heterogeneous_computing#Challenges)).
- **Multiple toolchains** — vendor-specific models (CUDA for NVIDIA, HIP/ROCm
  for AMD, Metal for Apple) each demand their own toolchain per target
  ([Intel/SYCL docs](https://www.intel.com/content/www/us/en/docs/sycl/introduction/latest/01-homogeneous-vs-heterogeneous.html)).

Standards exist precisely to smooth this. **HSA (Heterogeneous System
Architecture)**, a cross-vendor spec from the HSA Foundation (AMD, ARM,
others), puts CPU and GPU on the same bus with shared memory and tasks: a
unified virtual address space where devices share page tables and can exchange
data by sharing pointers, an ISA-agnostic intermediate language (HSAIL), and
heterogeneous task queues where any core can schedule work for any other with
load balancing — the stated aim is reducing communication latency and removing
the programmer's burden of planning data movement between disjoint memories
([Wikipedia: HSA](https://en.wikipedia.org/wiki/Heterogeneous_system_architecture)).
NVIDIA's **Unified Memory** is the CUDA-side version of the same idea (see
§7).

### How CUDA expresses the model

The CUDA programming model assumes a heterogeneous system of GPUs and CPUs:

| Term | Meaning |
| --- | --- |
| **Host** / host memory | the CPU and the memory directly attached to it |
| **Device** / device memory | a GPU and the memory directly attached to it |
| **Kernel** | a function invoked for execution on the GPU (historical name) |
| **Launch** | starting a kernel — conceptually starting many threads executing the kernel code in parallel |

Applications always start execution on the CPU. Host code uses CUDA APIs to
copy data between host and device memory, start GPU code, and wait for
completion; CPU and GPU can execute simultaneously, and best performance comes
from keeping both busy. On SoC systems host and device may share a package;
larger systems may have multiple CPUs or GPUs. Everything in §3–§8 — the
compute units, cooperative groups, scheduling groups, the memory pyramid — is
the machinery for the device half of this picture.

---

## 3. Hardware model

All three platforms model a GPU the same way at a high level: many similar
compute units, each receiving groups of threads, executing thousands of threads
in parallel, feeding off a shared last-level cache backed by DRAM. Only the
names differ:

| Concept               | NVIDIA (CUDA)                 | AMD (HIP/ROCm)                                                      | Apple (Metal)                                            |
| --------------------- | ----------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| Compute unit          | Streaming Multiprocessor (SM) | Compute Unit (CU); CUs grouped into Workgroup Processors on RDNA    | GPU core (Metal does not expose units to the programmer) |
| Scheduling group      | **Warp** — 32 threads         | **Wavefront** — 64 on GCN/CDNA; 32 (default) or 64 (opt-in) on RDNA | **SIMD-group** — 32 threads on Apple GPUs                |
| Cooperative group     | Thread block                  | Workgroup                                                           | Threadgroup                                              |
| Top-level launch      | Grid of blocks                | NDRange of work-groups                                              | Grid of threadgroups                                     |
| On-chip shared memory | Shared memory                 | LDS (local data share)                                              | Threadgroup memory                                       |
| Matrix hardware       | Tensor Cores                  | Matrix Cores (CDNA MFMA); WMMA (RDNA)                               | SIMD-scoped matrix multiply (Metal 3, Apple7 family)     |
| Main memory           | HBM or GDDR (discrete)        | HBM (Instinct) / GDDR (Radeon)                                      | Unified LPDDR, shared with the CPU                       |
| Programming stack     | CUDA                          | HIP (source-compatible with CUDA C++)                               | Metal / Metal Shading Language                           |

A compute unit contains roughly the same parts everywhere: a **register
file**, a **shared-memory resource** (frequently the same physical SRAM as the
L1 cache, with the split configurable), **schedulers** that pick ready thread
groups each cycle, and the execution units. Above the units sits a larger
cache shared by all units; behind it, DRAM. Sizes and counts vary per
architecture — the actual hardware layout may differ from the programming
model without affecting software correctness, because the programming model is
the contract, not the silicon.

The Modular Handbook frames this as two views that must meet:

- **Hardware** — compute units, matrix units, registers, caches, shared
  memory, and DRAM: what determines how much work and data stays close to the
  execution units.
- **Execution model** — threads organized into scheduling groups, cooperative
  groups, and grids: the abstraction used to write a kernel.

Kernel optimization is mapping the execution model onto the hardware while
respecting limits on compute, memory bandwidth, registers, shared memory, and
scheduling capacity. Without that model, advice such as "increase occupancy"
or "reduce shared-memory bank conflicts" is a rule to memorize rather than a
trade-off that can be reasoned about — and the advice transfers between
vendors, only the numbers change.

### Execution and memory map (NVIDIA H100 SXM example)

```
thread → warp (32 threads) → thread block → SM          [on-chip]
                                                    SM 0 … SM 131
  register file    256 KB, private per thread
  shared memory    per block, programmer-managed
  L1 cache         per SM, hardware-managed
  warp schedulers  issue ready warps each cycle
                          │
                     L2 cache   50 MB, shared by all SMs, hardware-managed
                          │ memory bus
   HBM (global memory / VRAM)   80 GB, 3.35 TB/s, ~400+ cycle latency
   weights · KV cache · activations · intermediate buffers
```

Equivalent figures on the other two:

| | NVIDIA H100 SXM | AMD MI300X | Apple M3 Pro |
| --- | --- | --- | --- |
| Compute units | 132 SMs | 304 CUs (CDNA 3) | 14–18 GPU cores (unified) |
| Scheduling group | warp, 32 | wavefront, 64 (CDNA) | SIMD-group, 32 (measured) |
| Main memory | 80 GB HBM3, 3.35 TB/s | 192 GB HBM3, 5.3 TB/s | LPDDR5 unified, 150 GB/s |
| Host relationship | separate device (PCIe/NVLink) | separate device (PCIe/Infinity Fabric) | same DRAM as CPU (`hasUnifiedMemory: true`) |

The shape of the hierarchy — many parallel units, tiny fast private storage,
programmable shared storage, hardware caches, large slow backing memory — is
the same on all three. DRAM is the largest but slowest tier; kernel
optimization focuses on minimizing round-trips to it. On Apple silicon the
"device memory" tier is literally the system memory pool, so host↔device
copies disappear but *on-chip reuse still matters exactly as much*, because
the bandwidth gap between threadgroup memory and main memory remains.

The four terms that define the execution hierarchy (vendor names in
parentheses):

- **Thread** (work-item) — smallest logical unit of work; usually one element
  or a small group of elements.
- **Scheduling group** — warp (NVIDIA) / wavefront (AMD) / SIMD-group (Apple):
  the threads that execute one instruction together.
- **Cooperative group** — thread block / workgroup / threadgroup: threads that
  cooperate through shared memory and synchronization.
- **Compute unit** — SM / CU / GPU core: the on-chip unit that accepts groups
  and issues instructions from ready scheduling groups.

---

## 4. Thread blocks and grids

A kernel launches with many threads — often millions. They are organized into
cooperative groups (**thread blocks** in CUDA, **workgroups** in HIP/OpenCL,
**threadgroups** in Metal), and those into a **grid** (an **NDRange** in
OpenCL). All groups in a grid have the same size and dimensions. Groups and
grids may be 1-, 2-, or 3-dimensional, which simplifies mapping threads onto
units of work or data.

| | NVIDIA CUDA | AMD HIP | Apple Metal |
| --- | --- | --- | --- |
| Cooperative group | thread block | workgroup | threadgroup |
| Top-level structure | grid | grid | grid (`threadgroupsPerGrid`) |
| Position built-ins | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` | same names (HIP mirrors CUDA) | `[[thread_position_in_threadgroup]]`, `[[threadgroup_position_in_grid]]`, `[[threads_per_threadgroup]]`, `[[threadgroups_per_grid]]` |
| Shared storage | `__shared__` | `__shared__` (HIP mirrors CUDA) | `threadgroup` address space |
| Sync inside group | `__syncthreads()` | `__syncthreads()` | `threadgroup_barrier()` |

A launch uses an **execution configuration** specifying grid and group
dimensions, plus optional parameters (stream/queue, cluster size, tuning
hints). Built-in variables give each thread its position within its group, its
group's position within the grid, and the dimensions themselves — a unique
identity among all threads of the kernel, commonly used to decide which data
or operation that thread is responsible for:

```cpp
// CUDA (and HIP — identical syntax)
int i = threadIdx.x + blockDim.x * blockIdx.x;   // global ID
if (i < n) C[i] = A[i] + B[i];                  // bounds guard
```

```metal
// Metal — same arithmetic via attributes
kernel void vecAdd(device float *A [[buffer(0)]],
                   device float *B [[buffer(1)]],
                   device float *C [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) C[i] = A[i] + B[i];
}
```

The bounds guard matters because the grid rarely divides the input evenly.
CUDA launch syntax is `vecAdd<<<blocks, threads>>>(...)` with the usual
ceiling division `blocks = (n + threads - 1) / threads`; Metal dispatches with
`dispatchThreadgroups` / `dispatchThreads`; HIP uses the same `<<<>>>` launch
syntax as CUDA.

Scheduling rules the model depends on (identical everywhere):

- All threads of a cooperative group execute on a **single compute unit**,
  which is what makes efficient communication and synchronization within the
  group possible through shared memory (LDS on AMD, threadgroup memory on
  Apple).
- A grid may contain millions of groups while the GPU has only tens or
  hundreds of compute units. Groups are assigned in **no guaranteed order**,
  and (with rare exceptions) a group runs to completion on its unit.
- Therefore there must be **no data dependencies between threads of different
  groups** in the same grid: a thread must not depend on or synchronize with a
  thread in another group. Groups may execute in any order, in parallel or in
  series.

This is what allows arbitrarily large grids to run on GPUs of any size, from
one unit to thousands — the same kernel, unmodified. The grid usually contains
more groups than can run at once; as groups finish, new ones fill free
capacity.

Metal-specific note: because CPU and GPU share DRAM on Apple silicon, the
kernel above reads and writes the *same* buffers the CPU allocated — no
device copy is required, only a memory hazard handshake (the encoder's
commit/synchronize). The compute rules above are unaffected.

### Thread block clusters (CUDA, compute capability 9.0+)

An NVIDIA-specific optional grouping level: a **cluster** is a group of
adjacent thread blocks, also laid out in 1–3 dimensions. Cluster membership
does not change grid dimensions or a block's index in the grid. All blocks of
a cluster execute within a **single GPC**, scheduled simultaneously, so
threads in different blocks of the same cluster can communicate and
synchronize through Cooperative Groups and access each other's shared memory —
**distributed shared memory**. Maximum cluster size is hardware dependent.
Neither HIP nor Metal exposes this grouping; the equivalent intent there is
solved with larger cooperative groups or queue-level synchronization.

---

## 5. Scheduling groups and the SIMT model

Within a cooperative group, threads are grouped into the vendor's scheduling
group — the unit that executes one instruction together:

| | NVIDIA | AMD | Apple |
| --- | --- | --- | --- |
| Name | warp | wavefront | SIMD-group |
| Width | 32 | 64 (GCN, CDNA); 32 default / 64 opt-in (RDNA) | 32 (measured `threadExecutionWidth` on M3 Pro) |
| Lane ids | 0–31 | 0–31 or 0–63 | 0–31 |
| Intra-group ops | warp shuffles (`__shfl_sync`) | DPP / `ds_swizzle`, WMMA | `simd_*` ops (shuffle, reduce, matrix; Metal 3+) |

All three execute the kernel code in the **SIMT** paradigm — Single-Instruction,
Multiple-Threads: all threads in the scheduling group run the same kernel
code, but each thread may follow different branches through it. The threads of
a group are assigned lanes in a predictable fashion.

- All threads of the group execute the same instruction simultaneously.
- Threads that do not follow the current branch are **masked off** while the
  ones that do are executed. When threads of one group follow different code
  paths, this is **divergence** (warp divergence on NVIDIA, wavefront
  divergence on AMD; the same masking happens inside a SIMD-group on Apple);
  utilization is maximized when threads within a group follow the same
  control-flow path.
- In the SIMT model all threads of the group progress through the kernel in
  **lock step**. Hardware execution may differ, but exploiting how it is
  actually mapped to hardware is discouraged: the model says the group
  progresses together, and code that violates the model risks undefined
  behavior that can differ between GPU generations.
- Scheduling groups need not be considered to write correct code, but
  understanding them explains **global memory coalescing** and **shared-memory
  bank access patterns**, and some advanced techniques specialize lanes
  within a group to limit divergence.
- Practical rule: give a cooperative group a thread count that is a **multiple
  of the scheduling-group width** — 32 on NVIDIA/Apple and RDNA, 64 on CDNA
  (32 also works on RDNA). Any count is legal, but a partial final group
  leaves lanes unused for its whole lifetime, degrading functional-unit and
  memory utilization.

**SIMT vs SIMD:** SIMD follows a single control-flow path and has a fixed data
width; SIMT lets each thread follow its own path and has no fixed data width.
(Separately, the CPU's own vector units are SIMD — Apple's AMX/NEON, x86 AVX
— but that is a different level: one CPU thread issuing vector instructions.)

```cpp
if (threadIdx.x % 2 == 0) { /* only even lanes execute this */ }
// odd lanes masked off during the body, then the group re-converges
```

Intra-group primitives make the fast paths portable: NVIDIA's
`__shfl_down_sync` / `__reduce_*`, AMD's wavefront operations, and Metal's
`simd_shuffle` / `simd_sum` / `simd_broadcast` are the same idea under three
names — exchange values between lanes without touching shared memory.

---

## 6. Two kernel styles: per-thread SIMT and tiles

The per-thread SIMT style above is universal — CUDA, HIP, and Metal all write
it. CUDA additionally offers a second, higher-level **tile programming**
model (this specific feature is CUDA's; the same idea appears elsewhere as
SYCL spans/DPC++, Triton blocks, and Metal's higher-level frameworks built on
simdgroup ops):

| | SIMT kernels | Tile kernels (CUDA) |
| --- | --- | --- |
| Programmer writes | per-thread code | per-block code over **tiles** |
| Thread count | chosen at launch | chosen by the compiler from the tile operations |
| Control flow | each thread may diverge | block has a single control flow; no warp divergence |
| Portability | thread-level decisions are yours | thread-level decisions are the compiler's, so one source runs across architectures |
| Best for | fine-grained control, custom optimizations | simpler, higher-level kernels |

Both kinds can operate on the same device memory, and both are built on the
same hardware — compute units, cooperative groups, grids — and the same memory
spaces. The choice is **per kernel**.

**Tile programming** basics:

- The programmer describes operations on multidimensional data collections
  called **tiles**; the compiler maps these onto the block's individual
  threads. Only **grid dimensions** are specified at launch.
- Each block runs the tile kernel, queries its grid position, and takes its
  portion of the data. Scalar operations (index math, loop bounds) run on one
  thread; tile operations run collectively across the block.
- **Arrays** are containers in device memory: mutable, with a shape and a data
  type.
- **Tiles** are local to a single block: immutable (every operation produces a
  new tile), possibly without any memory representation (the compiler may keep
  them in registers, shared memory, or other compute-unit resources), every
  dimension a compile-time-known power of two, never passed as kernel
  parameters.
- Data moves between arrays and tiles by **load**/**store** over a conceptual
  **tile space**: the array partitioned into equal, non-overlapping tiles; a
  load at tile-space index `(i, j)` returns that tile, with out-of-bounds
  elements handled explicitly (e.g. zero fill), while a store's out-of-bounds
  writes are discarded. Gather/scatter access arbitrary positions.
- Built-in tile operations: elementwise arithmetic, matrix multiplication,
  reductions along axes (sum, max), shape manipulation (reshape, transpose),
  type conversion; tiles of different shapes broadcast to match.
- Blocks are units of **execution**; tiles are units of **data** — one block
  may create many tiles of different shapes and types.

---

## 7. GPU memory

In heterogeneous systems, efficiently using memory matters as much as using
the functional units. The same tiers exist on all three platforms, with
vendor-specific names:

| Tier | NVIDIA | AMD | Apple (Metal) |
| --- | --- | --- | --- |
| Private per thread | registers | registers (VGPRs) | registers |
| Shared per group | shared memory | LDS (local data share) | threadgroup memory |
| Fast cache | L1 per SM, L2 shared | L1 per CU/WGP, L2 shared | L1/L2, plus SoC system-level cache (SLC) |
| Backing store | HBM/GDDR = global memory | HBM/GDDR = global memory | unified LPDDR = device memory (= system memory) |
| Constants | constant cache | constant/read-only cache | `constant` address space |

**Global vs host memory.** From device code, the DRAM attached to (or shared
with) the GPU is **global/device memory**, accessible by all compute units;
DRAM attached to the CPU is **system/host memory**. On discrete GPUs (typical
NVIDIA and AMD parts) these are physically separate pools connected by
PCIe/NVLink/Infinity Fabric, and explicit APIs allocate and copy between them.
GPUs use virtual memory addressing over a unified virtual address space on
currently supported systems, so every address unambiguously identifies which
memory it belongs to. **Unified Memory** (CUDA), fine-grained system SVM/HSA
(AMD), and Apple silicon's always-on **unified memory** are three answers to
the same problem — let the runtime or hardware place data instead of the
programmer. Optimal performance still means minimizing migration and
accessing data where it physically resides.

Measured on an Apple M3 Pro (`MTLDevice`, this machine):

```
maxThreadsPerThreadgroup:      1024 per dimension
maxThreadgroupMemoryLength:    32768 B (32 KB threadgroup memory)
threadExecutionWidth:          32 (SIMD-group width)
hasUnifiedMemory:              true
recommendedMaxWorkingSetSize:  ~14.3 GB
```

**On-chip memory** (per compute unit):

- **Register file** — thread-local variables, usually compiler-allocated.
- **Shared memory** — accessible by all threads of a cooperative group
  (thread block / workgroup / threadgroup, or cluster), used to exchange data;
  allocated at group level, unlike registers which are per thread.
- The register file, shared-memory space, and L1 cache are shared among the
  threads of a group residing on that unit, all of finite size.

Scheduling constraint: `registers per thread × threads per group ≤ available
registers on the unit`, otherwise the group cannot be scheduled — and if a
kernel's per-thread register requirement exceeds the register file, **the
kernel is not launchable** and the group size must be reduced. (On Metal the
practical effect shows up as `maxTotalThreadsPerThreadgroup` dropping below
1024 for register-hungry kernels.)

**Caches:** L1 lives on each compute unit; L2 is larger and shared by all
units; a constant cache serves values that are constant for a kernel's
lifetime (kernel parameters may live there too). Apple additionally has the
SoC's **system-level cache** sitting between GPU and DRAM, shared with the
CPU and other blocks.

**Locality domains** (CUDA) — a subset of a GPU's memory and compute units;
larger GPUs may have several. Memory allocated within one domain ("localized")
and kernels scheduled on units in the same domain can improve performance;
code that does not localize still executes correctly.

### The memory pyramid

Approximate NVIDIA H100 values (the canonical example):

| Level | Size | Latency | Scope | Managed by |
| --- | --- | --- | --- | --- |
| Registers | 256 KB per SM | ~1 cycle | per thread | compiler |
| Shared memory / L1 | ~256 KB combined pool (up to 228 KB SMEM) | ~20–30 cycles | block / SM | programmer / hardware |
| L2 cache | 50 MB | ~200 cycles | all SMs | hardware |
| HBM (global) | 80 GB, 3.35 TB/s | ~400+ cycles | global | programmer / runtime |

The same shape, other vendors' numbers: AMD MI300X — 192 GB HBM3 at ~5.3 TB/s
behind 304 CUs, with LDS per CU; Apple M3 Pro — unified LPDDR5 at 150 GB/s
behind 14–18 GPU cores, with 32 KB of threadgroup memory per group (measured)
and no separate device pool at all. The exact capacities differ; the *ratios*
between tiers — orders of magnitude in capacity and latency between the top
and bottom of the pyramid — are what stay constant.

Each step down the pyramid buys capacity and moves data farther from compute.
Registers are fastest but private per thread and finite. Shared memory is the
group's programmable workspace, organized in banks (32 on NVIDIA; 32 or 64
four-byte banks on AMD depending on architecture): same-bank accesses from
different lanes serialize, unless threads read the same address, which
broadcasts. L1 and L2 are hardware-managed. DRAM holds weights, KV cache, and
activations and is hundreds of cycles slow, so kernel optimization centers on
minimizing round-trips to it.

Why shared memory and L1 are needed despite registers: registers are private
(lane shuffles only reach within a scheduling group; shared memory spans a
whole cooperative group), limited (a weight tensor or large tile does not
fit), and cannot be used for coordination between threads at all.

Most kernel optimization reduces to moving less data or reusing it at a faster
tier — tiling, fusion, and layout changes keep frequently used data in
registers or shared memory instead of re-reading DRAM. One related rule from
the execution hierarchy: adjacent threads should access adjacent global memory
addresses so the GPU can combine a scheduling group's requests into few memory
transactions.

---

## 8. What makes a kernel fast

The checklist is vendor-neutral — only the group name and numbers change:

- Map threads to data with index arithmetic and keep the `if (i < n)` guard.
- Cooperative-group size a multiple of the scheduling-group width (32 on
  NVIDIA/Apple/RDNA, 64 on CDNA); enough groups in the grid to occupy every
  compute unit; no cross-group dependencies, since group order is unspecified.
- Keep scheduling groups on one control-flow path to avoid divergence and idle
  masked lanes.
- Give lanes of a group adjacent addresses; load reused data into shared
  memory (LDS / threadgroup memory) once instead of re-reading DRAM.
- Watch per-thread register use and per-group shared memory: they decide how
  many groups fit per unit, and excessive register demand makes the kernel
  unlaunchable (or lowers the attainable group size, e.g. Metal's
  `maxTotalThreadsPerThreadgroup`).
- On unified-memory systems (Apple silicon, CUDA UM, HSA), skip the
  host↔device copy thinking — but not the reuse thinking; the on-chip tiers
  still dominate performance.
- Prefer libraries and DSLs where they already cover the operation — cuBLAS/
  cuDNN/CUTLASS (NVIDIA), rocBLAS/rocMIOpen/Composable Kernel (AMD), Metal
  Performance Shaders / BNNS (Apple) — and write custom kernels for what they
  don't.

---

## References

- [Modular Handbook — GPU architecture fundamentals](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals) — section pages (threads/warps/blocks/grids, streaming multiprocessors, memory hierarchy, Tensor Cores); markdown versions via `.md` suffix, index at [`llms.txt`](https://handbook.modular.com/llms.txt)
- [CUDA Programming Guide — Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [CUDA Programming Guide — Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- AMD: [HIP documentation — Hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) (wavefront widths 32/64), [Composable Kernel — LDS and bank conflicts](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html) (LDS banks), [MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) (304 CUs, 192 GB HBM3)
- Apple: [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) (1024 threads/group, 32 KB threadgroup memory), [MTLDevice.maxThreadgroupMemoryLength](https://developer.apple.com/documentation/metal/mtldevice/maxthreadgroupmemorylength), [M3 Pro tech specs](https://support.apple.com/en-us/117736) (150 GB/s unified memory); SIMD-group width 32 and unified-memory limits measured on this machine's M3 Pro via `MTLDevice`/`MTLComputePipelineState`
- Heterogeneous model: [Wikipedia — Heterogeneous computing](https://en.wikipedia.org/wiki/Heterogeneous_computing), [Wikipedia — Heterogeneous System Architecture](https://en.wikipedia.org/wiki/Heterogeneous_system_architecture), [Wikipedia — GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units), [AMD — What is accelerated computing](https://www.amd.com/en/resources/articles/what-is-accelerated-computing-and-why-is-it-important.html), [IBM — What is accelerated computing](https://www.ibm.com/think/topics/accelerated-computing), [Intel — Heterogeneous vs. homogeneous computing environments](https://www.intel.com/content/www/us/en/docs/sycl/introduction/latest/01-homogeneous-vs-heterogeneous.html)
- [Modal GPU Glossary](https://modal.com/gpu-glossary/) — terminology lookup
- [GPU Puzzles](https://puzzles.modular.com/) — practice; study plan in `../README.md`
