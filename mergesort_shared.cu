#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define SIZE 65536
#define MAX_SHARED_SIZE 4096
#define BLOCKSIZE 32
#define TASK_SIZE (MAX_SHARED_SIZE / BLOCKSIZE)
#define GRIDSIZE (SIZE / TASK_SIZE / BLOCKSIZE)

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
__device__ int gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n);

void print_array_(DATA *arr, size_t n) {
  for (size_t i = 0; i < n; i++) {
    printf("%d\n", arr[i]);
  }
}

int main(int argc, char **argv) {
  DATA *arr;
  size_t size = SIZE;
  // assert(GRIDSIZE * BLOCKSIZE == SIZE / 2);
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

  gpu_shared_mergesort<<<GRIDSIZE, BLOCKSIZE>>>(A, B, SIZE);

  size_t starting_width = (MIN(MAX_SHARED_SIZE, SIZE / GRIDSIZE)) * 2;
  for (size_t width = starting_width; width <= size; width <<= 1) {
    // int slices = size / (nBlocks * blockSize * width);

    // Actually call the kernel
    gpu_mergesort<<<nBlocks, blockSize>>>(A, B, size, width);
    // gpu_mergesort<<<nBlocks, nThreads / blocksPerGrid>>>(
    //     A, B, size, width, slices, D_threads, D_blocks);

    // Switch the input / output arrays instead of copying them around
    A = A == dArr ? tmp : dArr;
    B = B == dArr ? tmp : dArr;

    if (blockSize > 1) {
      blockSize /= 2;
    } else {
      nBlocks /= 2;
    }
  }

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

__device__ void print_array(DATA *arr, size_t n) {
  for (size_t i = 0; i < n; i++) {
    printf("%d\n", arr[i]);
  }
}

__device__ int gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n) {
  if (n == 0) return;
  // print_array(arr, n);
  // printf(
  //     "------------------------------------------------------------------\n");
  // print_array(tmp, n);
  // printf(
  //     "------------------------------------------------------------------\n");

  int n_swaps = 0;
  DATA *A = arr, *B = tmp;

  for (size_t curr_size = 1; curr_size <= n - 1; curr_size *= 2) {
    for (size_t left_start = 0; left_start <= n - curr_size - 1;
         left_start += 2 * curr_size) {
      // int left_size = MIN(curr_size, n - left_start);
      int right_size = MIN(curr_size, n - left_start - curr_size);

      // if (left_size < curr_size) break;
      gpu_bottomUpMerge(A + left_start, curr_size, A + left_start + curr_size,
                        // right_size, B);
                        curr_size, B + left_start);
    }
    A = A == arr ? tmp : arr;
    B = B == arr ? tmp : arr;
    n_swaps++;
    // print_array(A, n);
    // printf(
    //     "------------------------------------------------------------------\n");
  }
  return n_swaps;
}

__global__ void gpu_shared_mergesort(DATA *A, DATA *B, size_t size) {
  __shared__ DATA localA[MIN(MAX_SHARED_SIZE, SIZE / GRIDSIZE)];
  __shared__ DATA localB[MIN(MAX_SHARED_SIZE, SIZE / GRIDSIZE)];
  // Copy to shared memory
  size_t blockDataSize =
      size / gridDim.x;  // TODO: does not work if MIN(MAX_SHARED_SIZE, SIZE) ==
                         // MAX_SHARED_SIZE
  size_t local_n = blockDataSize / blockDim.x;
  size_t localStart = local_n * threadIdx.x;
  size_t globalStart = local_n * (blockIdx.x * blockDim.x + threadIdx.x);

  // if (threadIdx.x == 0)
  //   printf("local_n: %d, localStart: %d, globalStart: %d, blockSize: %d,
  //   blockIdx: %d\n",
  //          local_n, localStart, globalStart, local_n * blockDim.x,
  //          blockIdx.x);

  memcpy(localA + localStart, A + globalStart, local_n * sizeof(DATA));

  if (localStart >= size) return;
  __syncthreads();

  // Mergesort on local_n elements
  int n_swaps =
      gpu_serial_merge_sort(localA + localStart, localB + localStart, local_n);
  DATA *localAptr = n_swaps % 2 == 1 ? localB : localA;
  DATA *localBptr = localAptr == localA ? localB : localA;
  __syncthreads();

  // if (blockIdx.x == 1 && threadIdx.x == 0) {
  //   print_array(localAptr, blockDataSize);
  //   printf(
  //       "------------------------------------------------------------------\n");
  // }
  // __syncthreads();

  if (BLOCKSIZE > 1) {
    int shouldWork, halfSize, counter = 2;
    local_n <<= 1;

    while (local_n <= blockDataSize) {
      shouldWork = threadIdx.x % counter == 0;
      if (shouldWork) {
        halfSize = local_n / 2;
        gpu_bottomUpMerge(localAptr + localStart, halfSize,
                          localAptr + localStart + halfSize, halfSize,
                          localBptr + localStart);
      }
      local_n <<= 1;
      counter <<= 1;
      localAptr = localAptr == localA ? localB : localA;
      localBptr = localBptr == localA ? localB : localA;
      __syncthreads();
    }
  }

  // if (threadIdx.x == 0) {
  //   print_array(localAptr, local_n /2);
  //   printf(
  //       "------------------------------------------------------------------\n");
  // }
  //     __syncthreads();

  // if (threadIdx.x == 1) {
  //   print_array(localBptr, local_n /2);
  //   printf(
  //       "------------------------------------------------------------------\n");
  // }

  local_n = blockDataSize / blockDim.x;
  // for (size_t i = 0; i < local_n; i++) {
  //   A[globalStart + i] = localAptr[localStart + i];
  // }
  memcpy(A + globalStart, localAptr + localStart, local_n * sizeof(DATA));

  // if (threadIdx.x == 0) {
  //   print_array(A, local_n * 2);
  //   printf(
  //       "------------------------------------------------------------------\n");
  // }
}

__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width) {
  size_t start = width * (blockIdx.x * blockDim.x + threadIdx.x);

  if (start >= size) return;

  size_t halfSize = width / 2;

  gpu_bottomUpMerge(A + start, halfSize, A + start + halfSize, halfSize,
                    B + start);
}

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
  if (i < size1) memcpy(tmp + i + j, arr1 + i, (size1 - i) * sizeof(int));
  if (j < size2) memcpy(tmp + size1 + j, arr2 + j, (size2 - j) * sizeof(int));
}