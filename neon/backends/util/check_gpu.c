#include <cuda_runtime.h>
#include <cuda.h>

int main() {
  int deviceCount;
  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  // return 0 if GPUs are present
  return e == CUDA_SUCCESS ? 0 : -1;
}
