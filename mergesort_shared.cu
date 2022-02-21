#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define SIZE 4096
#define MAX_SHARED_SIZE 4096
#define BLOCKSIZE 1
#define GRIDSIZE SIZE / 2 / BLOCKSIZE
#define SHARED 4

#define CUDA_CHECK(X)                                               \
  {                                                                 \
    cudaError_t _m_cudaStat = X;                                    \
    if (cudaSuccess != _m_cudaStat) {                               \
      fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
              cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
      exit(1);                                                      \
    }                                                               \
  }

int _is_sorted(DATA *arr, size_t size);
void MergeSortOnDevice(DATA *arr, size_t size);
__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width);
__global__ void gpu_shared_mergesort(DATA *A, DATA *B, size_t size);
__device__ void gpu_bottomUpMerge(DATA *arr1, size_t size1, DATA *arr2,
                                  size_t size2, DATA *tmp);
__device__ void gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n);

int main(int argc, char **argv) {
  DATA *arr;
  size_t size = SIZE;
  assert(GRIDSIZE * BLOCKSIZE == SIZE / 2);
  assert(size == 0 || !(size & (size - 1)));

  arr = (DATA *)malloc(size * sizeof(DATA));
  if (arr == NULL) {
    fprintf(stderr, "Memory could not be allocated");
    exit(EXIT_FAILURE);
  }

  srand(0);
  for (size_t i = 0; i < size; i++) {
    arr[i] = rand();  // TODO: generate with sign and maybe in a range
  }

  MergeSortOnDevice(arr, size);
  assert(_is_sorted(arr, size) == 1);
}

int _is_sorted(DATA *arr, size_t size) {
  for (size_t i = 0; i < size - 1; i++)
    if (arr[i] > arr[i + 1]) return 0;
  return 1;
}

void MergeSortOnDevice(DATA *arr, size_t size) {
  if (size == 0) return;

  DATA *dArr, *tmp;

  size_t byteSize = size * sizeof(DATA);
  CUDA_CHECK(cudaMalloc(&dArr, byteSize));
  CUDA_CHECK(cudaMemcpy(dArr, arr, byteSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&tmp, byteSize));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  //
  // Slice up the list and give pieces of it to each thread, letting the pieces
  // grow bigger and bigger until the whole list is sorted
  //

  DATA *A = dArr, *B = tmp;

  int nBlocks = GRIDSIZE;
  int blockSize = BLOCKSIZE;

  gpu_shared_mergesort<<<GRIDSIZE, BLOCKSIZE, MIN(MAX_SHARED_SIZE, SIZE)>>>(A, B, MIN(MAX_SHARED_SIZE, SIZE));

  // for (size_t width = 2; width <= size; width <<= 1) {
  //   // int slices = size / (nBlocks * blockSize * width);

  //   // Actually call the kernel
  //   gpu_mergesort<<<nBlocks, blockSize>>>(A, B, size, width);
  //   CUDA_CHECK(cudaDeviceSynchronize());
  //   // gpu_mergesort<<<nBlocks, nThreads / blocksPerGrid>>>(
  //   //     A, B, size, width, slices, D_threads, D_blocks);

  //   // Switch the input / output arrays instead of copying them around
  //   A = A == dArr ? tmp : dArr;
  //   B = B == dArr ? tmp : dArr;

  //   if (blockSize > 1) {
  //     blockSize /= 2;
  //   } else {
  //     nBlocks /= 2;
  //   }
  // }

  // MergeSortKernel<<<numBlocks, numThreads>>>(
  //     dArr, size, ceil(size / float(BLOCKSIZE * GRIDSIZE)));

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed = elapsed / 1000.f;  // convert to seconds
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  printf("Kernel elapsed time %fs \n", elapsed);

  CUDA_CHECK(cudaMemcpy(arr, A, byteSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dArr));
}

__device__ void gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n) {
  if (n == 0) return;

  for (size_t curr_size = 1; curr_size <= n - 1; curr_size *= 2) {
    for (size_t left_start = 0; left_start <= n - curr_size - 1;
         left_start += 2 * curr_size) {
      // int left_size = MIN(curr_size, n - left_start);
      int right_size = MIN(curr_size, n - left_start - curr_size);

      // if (left_size < curr_size) break;
      gpu_bottomUpMerge(arr + left_start, curr_size,
                        arr + left_start + curr_size, right_size, tmp);
    }
  }
}

__global__ void gpu_shared_mergesort(DATA *A, DATA *B, size_t size) {
  extern __shared__ DATA localA[];
  extern __shared__ DATA localB[];
  // Copy to shared memory
  size_t local_n = size / blockDim.x;
  size_t localStart = local_n * threadIdx.x;
  size_t globalStart = local_n * (blockIdx.x * blockDim.x + threadIdx.x);

  for (size_t i = 0; i < local_n; i++) {
    localA[localStart + i] = A[globalStart + i];
  }

  if (localStart >= size) return;

  // Mergesort on local_n elements
  gpu_serial_merge_sort(localA + localStart, localB + localStart, local_n);

  __syncthreads();

  // for (size_t width = 2; width <= size; width <<=2) {

  // }
  DATA *localAptr = localA;
  DATA *localBptr = localB;
  // if (BLOCKSIZE > 1) {
  // int shouldWork, halfSize, counter = 1;

  // while (local_n <= size) {
  //   shouldWork = threadIdx.x % counter == 0;
  //   if (shouldWork) {
  //     halfSize = local_n / 2;
  //     gpu_bottomUpMerge(localAptr + localStart, halfSize,
  //                       localAptr + localStart + halfSize, halfSize,
  //                       localBptr + localStart);
  //     local_n <<= 1;
  //   }
  //   counter <<= 1;
  //   __syncthreads();
  //   localAptr = localAptr == localA ? localB : localA;
  //   localBptr = localBptr == localA ? localB : localA;
  // }
  // }

  local_n = size / blockDim.x;
  for (size_t i = 0; i < local_n; i++) {
    A[globalStart + i] = localAptr[localStart + i];
  }
}

// __global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width) {
//   __shared__ DATA localA[SIZE];
//   __shared__ DATA localB[SIZE];
//   size_t localStart = width * threadIdx.x;
//   size_t globalStart = width * (blockIdx.x * blockDim.x + threadIdx.x);

//   for (size_t i = 0; i < width; i++) {
//     localA[localStart + i] = A[globalStart + i];
//   }

//   if (localStart >= size) return;

//   size_t halfSize = width / 2;

//   gpu_bottomUpMerge(localA + localStart, halfSize,
//                     localA + localStart + halfSize, halfSize,
//                     localB + localStart);
//   for (size_t i = 0; i < width; i++) {
//     A[globalStart + i] = localA[localStart + i];
//     B[globalStart + i] = localB[localStart + i];
//   }
// }

__device__ void gpu_bottomUpMerge(DATA *arr1, size_t size1, DATA *arr2,
                                  size_t size2, DATA *tmp) {
  size_t i = 0, j = 0;

  while (i < size1 && j < size2) {
    if (arr1[i] < arr2[j]) {
      tmp[i + j] = arr1[i];
      i++;
    } else {
      tmp[i + j] = arr2[j];
      j++;
    }
  }
  while (i < size1) {
    tmp[i + j] = arr1[i];
    i++;
  }
  while (j < size2) {
    tmp[i + j] = arr2[j];
    j++;
  }
}