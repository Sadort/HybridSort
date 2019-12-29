#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>

void PipeDataSort(uint64_t *h_key_array, uint64_t *d_key_array[], uint64_t number_of_elements, uint64_t batch_size, uint64_t pinned_M_size, int nstreams);
