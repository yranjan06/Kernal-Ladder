// 01_vector_add.cu — one thread computes one element. The mental flip from CPU:
// no loop over data; parallelism is the default.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *out,
                                  int n) {
  int i = blockIdx.x * blockDim.x +
          threadIdx.x; // my global index across the whole grid
  if (i <
      n) { // guard: we launch in round multiples, extra threads must do nothing
    out[i] = a[i] + b[i];
  }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(),
              "inputs must be CUDA");
  TORCH_CHECK(a.numel() == b.numel(), "size mismatch");
  a = a.contiguous();
  b = b.contiguous(); // ensure flat row-major memory before raw pointer math

  auto out = torch::empty_like(a);
  int n = a.numel();

  int threads =
      256; // warp granularity is 32; 256 = 8 warps, good occupancy default
  int blocks = (n + threads - 1) /
               threads; // ceil division so every element gets a thread

  vector_add_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); // catch launch errors immediately, not
                                  // silently later
  return out;
}