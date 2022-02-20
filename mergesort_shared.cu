#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define SIZE 33554432
#define BLOCKSIZE 8
#define GRIDSIZE SIZE / 2 / BLOCKSIZE
#define SHARED 4

#define CUDA_CHECK(X)                                                     \
    {                                                                     \
        cudaError_t _m_cudaStat = X;                                      \
        if (cudaSuccess != _m_cudaStat)                                   \
        {                                                                 \
            fprintf(stderr, "\nCUDA_ERROR: %s in file %s line %d\n",      \
                    cudaGetErrorString(_m_cudaStat), __FILE__, __LINE__); \
            exit(1);                                                      \
        }                                                                 \
    }

int _is_sorted(DATA *arr, size_t size);
void MergeSortOnDevice(DATA *arr, size_t size);
__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width);
__device__ void gpu_bottomUpMerge(DATA *arr1, size_t size1, DATA *arr2, size_t size2, DATA *tmp);

int main(int argc, char **argv)
{
    DATA *arr;
    size_t size = SIZE;
    assert(GRIDSIZE * BLOCKSIZE == SIZE / 2);
    assert(size == 0 || !(size & (size - 1)));

    arr = (DATA *)malloc(size * sizeof(DATA));
    if (arr == NULL)
    {
        fprintf(stderr, "Memory could not be allocated");
        exit(EXIT_FAILURE);
    }

    srand(0);
    for (size_t i = 0; i < size; i++)
    {
        arr[i] = rand(); // TODO: generate with sign and maybe in a range
    }

    MergeSortOnDevice(arr, size);
    assert(_is_sorted(arr, size) == 1);
}

int _is_sorted(DATA *arr, size_t size)
{
    for (size_t i = 0; i < size - 1; i++)
        if (arr[i] > arr[i + 1])
            return 0;
    return 1;
}

void MergeSortOnDevice(DATA *arr, size_t size)
{
    if (size == 0)
        return;

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

    DATA *A = dArr,
         *B = tmp;

    int nBlocks = GRIDSIZE;
    int blockSize = BLOCKSIZE;
    for (size_t width = 2; width <= size; width <<= 1)
    {
        // int slices = size / (nBlocks * blockSize * width);

        // Actually call the kernel
        gpu_mergesort<<<nBlocks, blockSize>>>(A, B, size, width);
        cudaDeviceSynchronize();
        // gpu_mergesort<<<nBlocks, nThreads / blocksPerGrid>>>(
        //     A, B, size, width, slices, D_threads, D_blocks);

        // Switch the input / output arrays instead of copying them around
        A = A == dArr ? tmp : dArr;
        B = B == dArr ? tmp : dArr;

        if (blockSize > 1)
        {
            blockSize /= 2;
        }
        else
        {
            nBlocks /= 2;
        }
    }

    // MergeSortKernel<<<numBlocks, numThreads>>>(
    //     dArr, size, ceil(size / float(BLOCKSIZE * GRIDSIZE)));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    elapsed = elapsed / 1000.f; // convert to seconds
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Kernel elapsed time %fs \n", elapsed);

    CUDA_CHECK(cudaMemcpy(arr, A, byteSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dArr));
}

__global__ void gpu_mergesort(DATA *A, DATA *B, size_t size, size_t width)
{
    __shared__ DATA localA[BLOCKSIZE * 2];
    __shared__ DATA localB[BLOCKSIZE * 2];
    unsigned int currentThread = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + currentThread;

    for (size_t i = 0; i < width; i++)
    {
        localA[currentThread + i] = A[globalIdx + i];
    }

    size_t start = width * currentThread;
    if (start >= size)
        return;

    size_t halfSize = width / 2;

    gpu_bottomUpMerge(localA + start, halfSize, localA + start + halfSize, halfSize, localB + start);
    for (size_t i = 0; i < width; i++)
    {
        A[globalIdx + i] = localA[currentThread + i];
        B[globalIdx + i] = localB[currentThread + i];
    }
}

__device__ void gpu_bottomUpMerge(DATA *arr1, size_t size1, DATA *arr2, size_t size2, DATA *tmp)
{
    size_t i = 0, j = 0;

    while (i < size1 && j < size2)
    {
        if (arr1[i] < arr2[j])
        {
            tmp[i + j] = arr1[i];
            i++;
        }
        else
        {
            tmp[i + j] = arr2[j];
            j++;
        }
    }
    while (i < size1)
    {
        tmp[i + j] = arr1[i];
        i++;
    }
    while (j < size2)
    {
        tmp[i + j] = arr2[j];
        j++;
    }
}