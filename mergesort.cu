#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define MIN(a, b) (a < b ? a : b)
#define SIZE 5120
#define BLOCKSIZE 16
#define GRIDSIZE SIZE / BLOCKSIZE
#define SHARED 102400

int _is_sorted(DATA *arr, size_t size);
void MergeSortOnDevice(DATA *arr, size_t size);
__global__ void MergeSortKernel(DATA *dArr, size_t size);

int main(int argc, char **argv)
{
  DATA *arr;
  size_t size = SIZE;

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

  DATA *dArr;

  cudaMalloc(&dArr, size);
  cudaMemcpy(dArr, arr, size, cudaMemcpyHostToDevice);

  cudaError_t myCudaError;
  myCudaError = cudaGetLastError();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  MergeSortKernel<<<GRIDSIZE, BLOCKSIZE>>>(dArr, size, ceil(size / float(BLOCKSIZE * GRIDSIZE)));

  myCudaError = cudaGetLastError();
  if (myCudaError != cudaSuccess)
  {
    fprintf(stderr, "%s\n", cudaGetErrorString(myCudaError));
    exit(1);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed / 1000.f; // convert to seconds
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Kernel elapsed time %fs \n", elapsed);

  cudaMemcpy(arr, dArr, size, cudaMemcpyDeviceToHost);

  cudaFree(dArr);
}

__device__ solve(int **tempList, int left_start, int right_start, int old_left_start, int my_start, int my_end, int left_end, int right_end, int headLoc)
{
  for (int i = 0; i < walkLen; i++)
  {
    if (tempList[current_list][left_start] < tempList[current_list][right_start])
    {
      tempList[!current_list][headLoc] = tempList[current_list][left_start]; /*Compare if my left value is smaller than the
       left_start++;                                                           right value store it into the !current_list
       headLoc++;                                                               row of array tempList*/
      // Check if l is now empty
      if (left_start == left_end)
      {
        // place the left over elements into the array
        for (int j = right_start; j < right_end; j++)
        {
          tempList[!current_list][headLoc] = tempList[current_list][right_start];
          right_start++;
          headLoc++;
        }
      }
    }
    else
    {
      tempList[!current_list][headLoc] = tempList[current_list][right_start]; /*Compare if my right value is smaller than the
       right_start++;                                                             left value store it into the !current_list
       //Check if r is now empty                                                   row of array tempList*/
      if (right_start == right_end)
      {
        // place the left over elements into the array
        for (int j = left_start; j < left_end; j++)
        {
          tempList[!current_list][headLoc] = tempList[current_list][right_start];
          right_start++;
          headLoc++;
        }
      }
    }
  }
}

__global__ void MergeSortKernel(DATA *dArr, size_t size, int elementsPerThread)
{
  int my_start, my_end; // indices of each thread's start/end

  // Declare counters requierd for recursive mergesort
  int left_start, right_start; // Start index of the two lists being merged
  int old_left_start;
  int left_end, right_end; // End index of the two lists being merged
  int headLoc;             // current location of the write head on the newList
  short current_list = 0;  /* Will be used to determine which of two lists is the
     most up-to-date */

  // allocate enough shared memory for this block's list...

  __shared__ DATA tempList[2][SHARED / sizeof(DATA)];

  // Load memory
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < elementsPerThread; i++)
  {
    if (index + i < length)
    {
      tempList[current_list][elementsPerThread * threadIdx.x + i] = d_list[index + i];
    }
  }

  // Wait until all memory has been loaded
  __syncthreads();

  // Merge the left and right lists.
  for (int walkLen = 1; walkLen < length; walkLen *= 2)
  {
    // Set up start and end indices.
    my_start = elementsPerThread * threadIdx.x;
    my_end = my_start + elementsPerThread;
    left_start = my_start;

    while (left_start < my_end)
    {
      old_left_start = left_start; // left_start will be getting incremented soon.
      // If this happens, we are done.
      if (left_start > my_end)
      {
        left_start = length;
        break;
      }

      left_end = left_start + walkLen;
      if (left_end > my_end)
      {
        left_end = length;
      }

      right_start = left_end;
      if (right_start > my_end)
      {
        right_end = length;
      }

      right_end = right_start + walkLen;
      if (right_end > my_end)
      {
        right_end = length;
      }

      solve(&tempList, left_start, right_start, old_left_start, my_start, int my_end, left_end, right_end, headLoc);
      left_start = old_left_start + 2 * walkLen;
      current_list = !current_list;
    }
  }
  // Wait until all thread completes swapping if not race condition will appear
  // as it might update non sorted value to d_list
  __syncthreads();

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < elementsPerThread; i++)
  {
    if (index + i < length)
    {
      d_list[index + i] = subList[current_list][elementsPerThread * threadIdx.x + i];
    }
  }
  // Wait until all memory has been loaded
  __syncthreads();

  return;
}
