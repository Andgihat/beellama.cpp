#include "mmv-fv5.cuh"

// Fermion FV5/FV5B fused F32-activation GEMV.
//
// Weight reconstruction is identical to the CPU reference (ggml-quants.c /
// ggml-cpu/quants.c):
//   FV5:  w[j] = (bp[j] - bn[j]) * (br[j] ? s_hi : s_lo)
//   FV5B: w[j] = s * qs[j]
// The dot against the raw f32 activation vector factors into masked f32
// activation sums (sum_lo/sum_hi for FV5), scaled once per row: the per-row
// dual scales are stored as exact f32 copies replicated into every block of
// the row (see block_fv5 in ggml-common.h), so scaling the row totals is
// exact w.r.t. the container's per-row scales. All accumulation is f32.
//
// One CUDA block per output row; each thread walks plane bytes (8 weights per
// byte) with a stride of blockDim.x, then the two partial sums are reduced
// block-wide (warp shuffle + shared memory).

#define MMV_FV5_BLOCK_SIZE 128

static __global__ void mul_mat_vec_fv5_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5 * x = (const block_fv5 *) vx + row*nblocks_per_row;

    float sum_lo = 0.0f;
    float sum_hi = 0.0f;

    const int64_t nbytes = nblocks_per_row * (QK_FV5/8);
    const int64_t stride = MMV_FV5_BLOCK_SIZE;

    // Co-located "v2" layout (block_fv5.q[j] = uchar4 [bp,bn,br,pad]): one coalesced
    // uchar4 load fetches all three planes of an 8-weight group (vs 2-3 separate/divergent
    // loads in the SoA layout) -> DRAM utilization 63%->~100%, measured +1.23x decode.
    // 4x unroll keeps 8 independent loads in flight; byte sparsity + float4 activations.
    for (int64_t base = tid; base < nbytes; base += stride*4) {
        int64_t ib[4]; int jj[4]; bool ok[4];
        uchar4 t[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int64_t idx = base + u*stride;
            ok[u] = idx < nbytes;
            ib[u] = idx >> 5;
            jj[u] = (int)(idx & 31);
            if (ok[u]) {
                t[u] = *(const uchar4 *) x[ib[u]].q[jj[u]]; // [bp,bn,br,pad] in one load
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (!ok[u] || !(t[u].x | t[u].y)) {
                continue;
            }
            const uint8_t bp = t[u].x;
            const uint8_t bn = t[u].y;
            const uint8_t br = t[u].z;
            const float4 * y4 = (const float4 *)(y + ib[u]*QK_FV5 + 8*jj[u]);
            const float4 a  = y4[0];
            const float4 b4 = y4[1];
            const float yy[8] = { a.x, a.y, a.z, a.w, b4.x, b4.y, b4.z, b4.w };
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint8_t bit = 1u << b;
                float v = 0.0f;
                if (bp & bit) {
                    v = yy[b];
                } else if (bn & bit) {
                    v = -yy[b];
                }
                if (br & bit) {
                    sum_hi += v;
                } else {
                    sum_lo += v;
                }
            }
        }
    }

    float2 sums = make_float2(sum_lo, sum_hi);
    sums = warp_reduce_sum(sums);

    __shared__ float2 s_sums[MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_sums[warp_id] = sums;
    }
    __syncthreads();
    if (warp_id == 0) {
        sums = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sums[lane_id] : make_float2(0.0f, 0.0f);
        sums = warp_reduce_sum(sums);
        if (lane_id == 0) {
            dst[row] = x[0].s_lo*sums.x + x[0].s_hi*sums.y;
        }
    }
}

static __global__ void mul_mat_vec_fv5b_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5b * x = (const block_fv5b *) vx + row*nblocks_per_row;

    float sum = 0.0f;

    const int64_t ngroups = nblocks_per_row * (QK_FV5/8); // groups of 8 int8

    for (int64_t idx = tid; idx < ngroups; idx += MMV_FV5_BLOCK_SIZE) {
        const int64_t ib = idx >> 5;
        const int     j  = (int)(idx & 31);

        const int8_t * q  = x[ib].qs + 8*j;
        const float  * yj = y + ib*QK_FV5 + 8*j;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            sum += (float) q[b] * yj[b];
        }
    }

    sum = warp_reduce_sum(sum);

    __shared__ float s_sum[MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_sum[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            dst[row] = x[0].s * sum;
        }
    }
}

// Batched FV5 GEMV for small ne11 (speculative verify): each weight group is loaded
// ONCE (uchar4) and applied to all N activation columns -> weight DRAM traffic == one
// decode, advancing N tokens. lo/hi accumulators kept (bit-identical to single-token
// decode -> maximal draft acceptance). Larger ne11 (prefill) stays on dequant+cuBLAS.
// y: N columns at y + c*ncols_y (ncols_y = K).  dst: dst + c*ncols_dst (ncols_dst = nrows).
template <int N>
static __global__ void mul_mat_fv5_batched_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row, const int64_t ncols_y, const int64_t ncols_dst) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;
    const block_fv5 * x = (const block_fv5 *) vx + row*nblocks_per_row;

    float sum_lo[N], sum_hi[N];
#pragma unroll
    for (int c = 0; c < N; ++c) { sum_lo[c] = 0.0f; sum_hi[c] = 0.0f; }

    const int64_t nbytes = nblocks_per_row * (QK_FV5/8);
    for (int64_t idx = tid; idx < nbytes; idx += MMV_FV5_BLOCK_SIZE) {
        const int64_t ib = idx >> 5;
        const int     j  = (int)(idx & 31);
        const uchar4 t = *(const uchar4 *) x[ib].q[j];   // weight group loaded ONCE
        if (!(t.x | t.y)) {
            continue;
        }
        const uint8_t bp = t.x, bn = t.y, br = t.z;
        const int64_t koff = ib*QK_FV5 + 8*j;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            const float4 * y4 = (const float4 *)(y + c*ncols_y + koff);
            const float4 a = y4[0], b4 = y4[1];
            const float yy[8] = { a.x, a.y, a.z, a.w, b4.x, b4.y, b4.z, b4.w };
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const uint8_t bit = 1u << b;
                float v = 0.0f;
                if (bp & bit) { v = yy[b]; } else if (bn & bit) { v = -yy[b]; }
                if (br & bit) { sum_hi[c] += v; } else { sum_lo[c] += v; }
            }
        }
    }

    const float s_lo = x[0].s_lo, s_hi = x[0].s_hi;
    __shared__ float2 s_red[MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
#pragma unroll
    for (int c = 0; c < N; ++c) {
        float2 s = make_float2(sum_lo[c], sum_hi[c]);
        s = warp_reduce_sum(s);
        if (lane_id == 0) { s_red[warp_id] = s; }
        __syncthreads();
        if (warp_id == 0) {
            s = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_red[lane_id] : make_float2(0.0f, 0.0f);
            s = warp_reduce_sum(s);
            if (lane_id == 0) { dst[c*ncols_dst + row] = s_lo*s.x + s_hi*s.y; }
        }
        __syncthreads();
    }
}

#define FV5_BATCHED_MAX 8

bool ggml_cuda_can_mul_mat_fv5_batched(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_FV5 &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->ne[1] >= 2 && src1->ne[1] <= FV5_BATCHED_MAX;
}

void ggml_cuda_mul_mat_fv5_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat_fv5_batched(src0, src1, dst));
    GGML_ASSERT(src0->ne[0] % QK_FV5 == 0);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const int64_t nrows           = src0->ne[1];
    const int64_t nblocks_per_row = src0->ne[0] / QK_FV5;
    const int64_t ncols_y         = src1->ne[0]; // K
    const int64_t ncols_dst       = dst->ne[0];  // = nrows (M)
    const int64_t ncols           = src1->ne[1];

    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(MMV_FV5_BLOCK_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

    const void  * X = src0->data;
    const float * Y = (const float *) src1->data;
    float       * D = (float *) dst->data;

#define FV5_LAUNCH(N) mul_mat_fv5_batched_f32<N><<<block_nums, block_dims, 0, stream>>>(X, Y, D, nblocks_per_row, ncols_y, ncols_dst)
    switch (ncols) {
        case 2: FV5_LAUNCH(2); break;
        case 3: FV5_LAUNCH(3); break;
        case 4: FV5_LAUNCH(4); break;
        case 5: FV5_LAUNCH(5); break;
        case 6: FV5_LAUNCH(6); break;
        case 7: FV5_LAUNCH(7); break;
        case 8: FV5_LAUNCH(8); break;
        default: GGML_ABORT("fv5 batched: unsupported ncols %d", (int) ncols);
    }
#undef FV5_LAUNCH
}

bool ggml_cuda_can_mul_mat_vec_fv5(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return (src0->type == GGML_TYPE_FV5 || src0->type == GGML_TYPE_FV5B) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

void ggml_cuda_mul_mat_vec_fv5(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat_vec_fv5(src0, src1, dst));
    GGML_ASSERT(src0->ne[0] % QK_FV5 == 0);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const int64_t nrows           = src0->ne[1];
    const int64_t nblocks_per_row = src0->ne[0] / QK_FV5;

    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(MMV_FV5_BLOCK_SIZE, 1, 1);

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_FV5) {
        mul_mat_vec_fv5_f32<<<block_nums, block_dims, 0, stream>>>(
            src0->data, (const float *) src1->data, (float *) dst->data, nblocks_per_row);
    } else {
        mul_mat_vec_fv5b_f32<<<block_nums, block_dims, 0, stream>>>(
            src0->data, (const float *) src1->data, (float *) dst->data, nblocks_per_row);
    }
}
