/*
 * Course: High Performance Computing 2021/2022
 * 
 * Lecturer: Francesco Moscato    fmoscato@unisa.it
 * 
 * Group:
 * De Stefano Alessandro   0622701470  a.destefano56@studenti.unisa.it
 * Della Rocca Marco   0622701573  m.dellarocca22@studenti.unisa.it
 * 
 * CUDA implementation of mergesort algorithm 
 * Copyright (C) 2022 Alessandro De Stefano (EarendilTiwele) Marco Della Rocca (marco741)
 * 
 * This file is part of CUDA Mergesort implementation.
 * 
 * CUDA Mergesort implementation is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * CUDA Mergesort implementation is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with CUDA Mergesort implementation.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file mergesort.cu
 * @brief Measures the execution time of the global memory-based Mergesort algorithm
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define SIZE 65536
#define BLOCKSIZE 1024
#define TASKSIZE 2

#define CUDA_CHECK(X)                                               \
  {                                                                 \
    cudaError_t _m_cudaStat = X;                                    \
    if (cudaSuccess != _m_cudaStat) {                               \
      fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
              cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
      exit(1);                                                      \
    }                                                               \
  }

void MergeSortOnDevice(DATA *arr, size_t size, int blockSize, int gridSize, int taskSize);
void MergeSortOnHost(DATA *arr, size_t size);
void _merge(DATA *arr1, size_t size1, DATA *arr2, size_t size2, DATA *tmp);
__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width);
__device__ void gpu_bottomUpMerge(DATA *arr1, size_t size1, DATA *arr2,
                                  size_t size2, DATA *tmp);
__global__ void gpu_mergesort_tasksize(DATA *arr, DATA *tmp, size_t size,
                                       size_t tasksize);
__device__ int gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n);


/**
 * @brief Creates the array to be sorted and calls the mergesort on host and device
 * 
 * @param argc number of arguments
 * @param argv arguments. Accepts size as first argument, blocksize as second argument and tasksize as third argument
 * @return int status code 
 */
int main(int argc, char **argv) {
  size_t size = SIZE;
  if (argc > 1)  sscanf(argv[1], "%zu", &size);
  int blockSize = (argc > 2) ? atoi(argv[2]) : BLOCKSIZE;
  int taskSize = (argc > 3) ? atoi(argv[3]) : TASKSIZE;
  int gridSize = size / taskSize / blockSize;

  assert(size == 0 || !(size & (size - 1)));
  assert(gridSize * blockSize == size / taskSize);
  DATA *arr;

  arr = (DATA *)malloc(size * sizeof(DATA));
  if (arr == NULL) {
    fprintf(stderr, "Memory could not be allocated");
    exit(EXIT_FAILURE);
  }

  srand(0);
  for (size_t i = 0; i < size; i++) {
    arr[i] = rand() - RAND_MAX/2;
  }

  DATA *hostArr;
  hostArr = (DATA *)malloc(size * sizeof(DATA));
  if (hostArr == NULL) {
    fprintf(stderr, "Memory could not be allocated");
    exit(EXIT_FAILURE);
  }
  memcpy(hostArr, arr, size * sizeof(DATA));
  MergeSortOnHost(hostArr, size);
  MergeSortOnDevice(arr, size, blockSize, gridSize, taskSize);
  assert(memcmp(hostArr, arr, size * sizeof(DATA)) == 0);
}

/**
 * @brief Merge sorts an array of size 'n' using CPU
 *
 * @param arr the array to be sorted
 * @param n the size of the array
 */
void MergeSortOnHost(DATA *arr, size_t n) {
  if (n == 0) return;
  DATA *tmp;

  tmp = (DATA *)malloc(n * sizeof(DATA));
  if (tmp == NULL) {
    fprintf(stderr, "Memory could not be allocated");
    exit(EXIT_FAILURE);
  }

  for (size_t curr_size = 1; curr_size <= n - 1; curr_size *= 2) {
    for (size_t left_start = 0; left_start <= n - curr_size - 1;
         left_start += 2 * curr_size) {
      size_t right_size = MIN(curr_size, n - left_start - curr_size);

      _merge(arr + left_start, curr_size, arr + left_start + curr_size,
             right_size, tmp);
    }
  }
}

/**
 * @brief Utility to implement the merging part in the merge sort algorithm
 *
 * @param arr1 the first array to be merged
 * @param size1 the size of the first array
 * @param arr2 the second array to be merged
 * @param size2 the size of the second array
 * @param tmp the temporary array to implement the algorithm 
 */
void _merge(DATA *arr1, size_t size1, DATA *arr2, size_t size2, DATA *tmp) {
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
  if (i < size1) memcpy(tmp + i + j, arr1 + i, (size1 - i) * sizeof(DATA));
  if (j < size2) memcpy(tmp + size1 + j, arr2 + j, (size2 - j) * sizeof(DATA));

  memcpy(arr1, tmp, (size1 + size2) * sizeof(DATA));
}

/**
 * @brief Merge sorts an array of size 'size' using GPU
 *
 * @param arr the array to be sorted
 * @param size the size of the array
 * @param blockSize the number of threads per block to use
 * @param gridSize the number of blocks per grid to use
 * @param taskSize the initial workload of a single thread
 */
void MergeSortOnDevice(DATA *arr, size_t size, int blockSize, int gridSize, int taskSize) {
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

  DATA *A = dArr, *B = tmp;
  gpu_mergesort_tasksize<<<gridSize, blockSize>>>(A, B, size, taskSize);

  for (size_t width = 2 * taskSize; width <= size; width <<= 1) {

    // Actually call the kernel
    gpu_mergesort<<<gridSize, blockSize>>>(A, B, size, width);

    // Switch the input / output arrays instead of copying them around
    A = A == dArr ? tmp : dArr;
    B = B == dArr ? tmp : dArr;

    if (blockSize > 1) {
      blockSize /= 2;
    } else {
      gridSize /= 2;
    }
  }

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  elapsed = elapsed / 1000.f;  // convert to seconds
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  printf("%f", elapsed);

  CUDA_CHECK(cudaMemcpy(arr, A, byteSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dArr));
}

/**
 * @brief GPU kernel to implement the serial work for the merge sort algorithm,
 * when the tasksize is greater than two
 *
 * @param arr the input array to work with
 * @param tmp the temporary array to implement the algorithm
 * @param size the size of the array
 * @param tasksize the initial workload of a single thread
 */
__global__ void gpu_mergesort_tasksize(DATA *arr, DATA *tmp, size_t size,
                                       size_t tasksize) {
  int n_swaps;
  size_t start = tasksize * (blockIdx.x * blockDim.x + threadIdx.x);
  if (start >= size) return;

  n_swaps = gpu_serial_merge_sort(arr + start, tmp + start, tasksize);
  if (n_swaps % 2 == 0) return;
  memcpy(arr + start, tmp + start, tasksize * sizeof(DATA));
}

/**
 * @brief GPU kernel to implement the parallel work for the merge sort algorithm
 *
 * @param A the input array to work with
 * @param B the temporary array to implement the algorithm
 * @param size the size of the array
 * @param width the width of the sorted blocks wanted in output
 */
__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width) {
  size_t start = width * (blockIdx.x * blockDim.x + threadIdx.x);

  if (start >= size) return;

  size_t halfSize = width / 2;

  gpu_bottomUpMerge(A + start, halfSize, A + start + halfSize, halfSize,
                    B + start);
}

/**
 * @brief GPU utility to implement the serial merge sort
 *
 * @param arr the array to be sorted
 * @param tmp the support array used in the implementation
 * @param n the size of the array
 * @return int number of pointer swaps in the process
 */
__device__ int gpu_serial_merge_sort(DATA *arr, DATA *tmp, size_t n) {
  if (n == 0) return;

  int n_swaps = 0;
  DATA *A = arr, *B = tmp;

  for (size_t curr_size = 1; curr_size <= n - 1; curr_size *= 2) {
    for (size_t left_start = 0; left_start <= n - curr_size - 1;
         left_start += 2 * curr_size) {
      gpu_bottomUpMerge(A + left_start, curr_size, A + left_start + curr_size,
                        curr_size, B + left_start);
    }
    A = A == arr ? tmp : arr;
    B = B == arr ? tmp : arr;
    n_swaps++;
  }
  return n_swaps;
}

/**
 * @brief GPU utility to implement the merge part in the algorithm
 *
 * @param arr1 the first array to be merged
 * @param size1 the size of the first array
 * @param arr2 the second array to be merged
 * @param size2 the size of the second array
 * @param tmp the temporary array to implement the algorithm
 */
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
