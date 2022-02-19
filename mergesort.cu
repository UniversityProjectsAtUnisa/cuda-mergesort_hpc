#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATA int
#define THREADxBLOCKalongXorY 16
#define MIN(a, b) (a < b ? a : b)
#define SIZE 5120

int _is_sorted(DATA *arr, size_t size);
void MergeSortOnHost(DATA *arr, size_t n);
void _merge_maxsize(DATA *arr1, size_t size1, DATA *arr2, size_t size2,
                    DATA *tmp, size_t maxsize);

int main(int argc, char **argv) {
  DATA *arr;
  size_t size = SIZE;

  arr = (DATA *)malloc(size * sizeof(DATA));
  if (arr == NULL) {
    fprintf(stderr, "Memory could not be allocated");
    exit(EXIT_FAILURE);
  }

  srand(0);
  for (size_t i = 0; i < size; i++) {
    arr[i] = rand();  // TODO: generate with sign and maybe in a range
  }

  MergeSortOnHost(arr, size);
  assert(_is_sorted(arr, size) == 1);
}

int _is_sorted(DATA *arr, size_t size) {
  for (size_t i = 0; i < size - 1; i++)
    if (arr[i] > arr[i + 1]) return 0;
  return 1;
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

      _merge_maxsize(arr + left_start, curr_size, arr + left_start + curr_size,
                     right_size, tmp, n);
    }
  }
}

void _merge_maxsize(DATA *arr1, size_t size1, DATA *arr2, size_t size2,
                    DATA *tmp, size_t maxsize) {
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

  memcpy(arr1, tmp, MIN((size1 + size2), maxsize) * sizeof(DATA));
}
