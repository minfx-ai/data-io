/* kernels.h :  C-visible wrapper around the CUDA reduction launch  */
#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda_runtime.h> /* cudaStream_t */
#include <stddef.h>       /* size_t   */
#include <stdint.h>       /* uint32_t */

#ifdef __cplusplus
extern "C" {
#endif

void launch_sum_kernel(const unsigned char *d_data, size_t nbytes,
                       unsigned long long *d_out,
                       cudaStream_t stream); /* C ABI */

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif