from max.gpu import thread_idx, block_idx, block_dim, grid_dim, global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.sys import exit
from std.math import ceildiv

comptime SIZE = 16 * 256
comptime THREADS_PER_BLOCK = 256
comptime BLOCKS = ceildiv(SIZE, THREADS_PER_BLOCK)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)


def watch_kernel(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < SIZE:
        output[tid] = Scalar[dtype](block_idx.x)


def main() raises:
    var ctx = DeviceContext()

    var out_dev = ctx.enqueue_create_buffer[dtype](SIZE)
    out_dev.enqueue_fill(-1)

    ctx.synchronize()

    var out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out_dev)

    ctx.enqueue_function[watch_kernel](
        out_tensor,
        grid_dim=BLOCKS,
        block_dim=THREADS_PER_BLOCK,
    )

    ctx.synchronize()

    var pre = Float32(-1)
    with out_dev.map_to_host() as out_host:
        for i in range(SIZE):
            var curr = out_host[i]
            if curr != pre:
                print("\n======= BLOCK", Int(curr), "=======")
                pre = curr

            print(i, end=', ')

    print()
    exit()
