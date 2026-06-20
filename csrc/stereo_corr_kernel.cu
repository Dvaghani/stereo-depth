// CUDA kernel for stereo correlation volume.
//
// One thread per output element (b, d, h, w). For each (b, d, h, w) we
// compute the dot product over the C feature channels between
//   left[b, :, h, w]  and  right[b, :, h, w - d]
// returning 0 when (w - d) < 0.

// ("The core matching kernel is implemented in CUDA/C++ for speed,
// wrapped in Python for integration."). It is intentionally simple and
// register-friendly so it fits Jetson Nano (sm_53, 4 KB shared mem / SM).
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void stereo_corr_forward_kernel(
    const scalar_t* __restrict__ left,
    const scalar_t* __restrict__ right,
    scalar_t* __restrict__ out,
    int B, int C, int H, int W, int D) {

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bd = blockIdx.z; // batch * D
    const int b = bd / D;
    const int d = bd % D;

    if (w >= W || h >= H || b >= B) return;

    const int rw = w - d;
    scalar_t acc = scalar_t(0);
    if (rw >= 0) {
        const int spatial = H * W;
        const int chan_stride = spatial;
        const int batch_stride = C * spatial;
        const scalar_t* lptr = left  + b * batch_stride + h * W + w;
        const scalar_t* rptr = right + b * batch_stride + h * W + rw;
        #pragma unroll 8
        for (int c = 0; c < C; ++c) {
            acc += lptr[c * chan_stride] * rptr[c * chan_stride];
        }
    }
    out[((b * D + d) * H + h) * W + w] = acc;
}

void stereo_corr_forward_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    torch::Tensor& out,
    int max_disp) {

    const int B = left.size(0);
    const int C = left.size(1);
    const int H = left.size(2);
    const int W = left.size(3);
    const int D = max_disp;

    const dim3 block(16, 16, 1);
    const dim3 grid((W + block.x - 1) / block.x,
                    (H + block.y - 1) / block.y,
                    B * D);

    AT_DISPATCH_FLOATING_TYPES(left.scalar_type(), "stereo_corr_forward_cuda", [&] {
        stereo_corr_forward_kernel<scalar_t><<<grid, block>>>(
            left.data_ptr<scalar_t>(),
            right.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, C, H, W, D);
    });
    C10_CUDA_CHECK(cudaGetLastError());
}
