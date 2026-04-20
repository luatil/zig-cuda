#include <cuda_runtime.h>

__global__ void addKernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// C-compatible wrapper called from Zig
extern "C" int cuda_add(const float *h_a, const float *h_b, float *h_c, int n) {
  float *d_a, *d_b, *d_c;
  size_t size = (size_t)n * sizeof(float);

  cudaError_t err;

  err = cudaMalloc(&d_a, size);
  if (err != cudaSuccess)
    return (int)err;

  err = cudaMalloc(&d_b, size);
  if (err != cudaSuccess) {
    cudaFree(d_a);
    return (int)err;
  }

  err = cudaMalloc(&d_c, size);
  if (err != cudaSuccess) {
    cudaFree(d_a);
    cudaFree(d_b);
    return (int)err;
  }

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return (int)err;
  }

  err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return (int)err;
}
