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
 * This file is part of OMP Mergesort implementation.
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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define SIZE 2048
#define MAX_SHARED_SIZE 4096
#define BLOCKSIZE 32

#define CUDA_CHECK(X)                                               \
  {                                                                 \
    cudaError_t _m_cudaStat = X;                                    \
    if (cudaSuccess != _m_cudaStat) {                               \
      fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
              cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
      exit(1);                                                      \
    }                                                               \
  }

void MergeSortOnDevice(DATA *arr, size_t size, int blockSize, int gridSize,
                       int sharedBlockSize);
void MergeSortOnHost(DATA *arr, size_t size);
void _merge(DATA *arr1, size_t size1, DATA *arr2, size_t size2, DATA *tmp);
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
  size_t size = SIZE;
  if (argc > 1) sscanf(argv[1], "%zu", &size);
  int blockSize = (argc > 2) ? atoi(argv[2]) : BLOCKSIZE;
  int sharedBlockSize = (argc > 3) ? atoi(argv[3]) : BLOCKSIZE;
  int gridSize = MAX(1, size / MAX_SHARED_SIZE);
  assert(size == 0 || !(size & (size - 1)));
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
  MergeSortOnDevice(arr, size, blockSize, gridSize, sharedBlockSize);
  assert(memcmp(hostArr, arr, size * sizeof(DATA)) == 0);
}

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

void MergeSortOnDevice(DATA *arr, size_t size, int blockSize, int gridSize,
                       int sharedBlockSize) {
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

  gpu_shared_mergesort<<<gridSize, sharedBlockSize>>>(A, B, size);

  size_t starting_width = (MIN(MAX_SHARED_SIZE, size / gridSize)) * 2;
  for (size_t width = starting_width; width <= size; width <<= 1) {
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

__device__ void print_array(DATA *arr, size_t n) {
  for (size_t i = 0; i < n; i++) {
    printf("%d\n", arr[i]);
  }
}

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

__global__ void gpu_shared_mergesort(DATA *A, DATA *B, size_t size) {
  __shared__ DATA localA[MAX_SHARED_SIZE];
  __shared__ DATA localB[MAX_SHARED_SIZE];

  // Copy to shared memory
  size_t blockDataSize = size / gridDim.x;
  size_t local_n = blockDataSize / blockDim.x;
  size_t localStart = local_n * threadIdx.x;
  size_t globalStart = local_n * (blockIdx.x * blockDim.x + threadIdx.x);

  memcpy(localA + localStart, A + globalStart, local_n * sizeof(DATA));

  if (localStart >= size) return;
  __syncthreads();

  // Mergesort on local_n elements
  int n_swaps;
  if(local_n == blockDataSize) {
    // returns if there is only one thread per block
    n_swaps = gpu_serial_merge_sort(localA + localStart, localB + localStart, local_n);
    DATA *localAptr = n_swaps % 2 == 1 ? localB : localA;
    memcpy(A + globalStart, localAptr + localStart, local_n * sizeof(DATA));
    return;
  }
  n_swaps =
      gpu_serial_merge_sort(localA + localStart, localB + localStart, local_n);
  DATA *localAptr = n_swaps % 2 == 1 ? localB : localA;
  DATA *localBptr = localAptr == localA ? localB : localA;
  __syncthreads();

  int shouldWork, halfSize, counter = 2;
  local_n <<= 1;

  while (local_n < blockDataSize) {
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
  if (threadIdx.x == 0) {
    halfSize = local_n / 2;
    gpu_bottomUpMerge(localAptr + localStart, halfSize,
                      localAptr + localStart + halfSize, halfSize,
                      A + globalStart);
  }
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