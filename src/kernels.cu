// kernels.cu  (must be compiled with nvcc)
#include <cuda_runtime.h>
#include "kernels.h"

/* --------------------------------------------------------------
 * 1.  The actual reduction kernel
 * --------------------------------------------------------------*/
__global__ void sum_kernel(const unsigned char *data,
                           size_t               nbytes,
                           unsigned long long  *out)
{
    extern __shared__ unsigned long long s[];
    const size_t stride = blockDim.x * gridDim.x;
    unsigned long long local = 0;

    /* Sum 8-byte chunks — keep it naturally aligned */
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < (nbytes >> 3);                       /* nbytes / 8 */
         i += stride)
        local +=
            reinterpret_cast<const unsigned long long*>(data)[i];

    /* Block-wide reduction */
    s[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t d = blockDim.x >> 1; d; d >>= 1) {
        if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

/* --------------------------------------------------------------
 * 2.  Thin C wrapper  (exported in kernels.h)
 * --------------------------------------------------------------*/
extern "C"    /* C linkage for the host program written in C */
void launch_sum_kernel(const unsigned char *d_data,
                       size_t               nbytes,
                       unsigned long long  *d_out,
                       cudaStream_t         stream)
{
    const int    TPB   = 256;
    const int    blocks = (nbytes >> 3) / TPB + 1;
    const size_t shmem  = TPB * sizeof(unsigned long long);

    sum_kernel<<<blocks, TPB, shmem, stream>>>(d_data, nbytes, d_out);
}
