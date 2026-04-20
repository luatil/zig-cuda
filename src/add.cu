// Only the kernel lives here. All host-side memory management is done in Zig
// via the CUDA Driver API. The extern "C" prevents C++ name mangling so the
// symbol is visible as "add_kernel" in the PTX.
extern "C" __global__ void add_kernel(const float *a, const float *b, float *c,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

extern "C" __global__ void sub_kernel(const float *a, const float *b, float *c,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] - b[i];
  }
}
