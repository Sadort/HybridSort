#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream> 
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include "bitonic_sort.h"

using namespace std;

int main(void)
{
    uint64_t number_of_elements = 1024L*1024*1024;
    uint64_t *h_key_array = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *d_key_array;

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i] = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }
    printf("size : %lu\n", sizeof(uint64_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start, 0);

    cudaMalloc( (void**)&d_key_array, number_of_elements * sizeof(uint64_t) );
    cudaMemcpy( d_key_array,
                h_key_array, 
                number_of_elements * sizeof(uint64_t),
                cudaMemcpyHostToDevice );

    BitonicSort<uint64_t, cmp>(d_key_array, number_of_elements, 256, 32);

    cudaMemcpy( h_key_array,
                d_key_array,
                number_of_elements * sizeof(uint64_t),
                cudaMemcpyDeviceToHost );

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time: %f s.\n", milliseconds/1000);

    //std::sort(h_key_ref.begin(), h_key_ref.end());
    //bool result = compareAB(h_key_array, h_key_ref);
    //printf("Test: %s\n", result == true ? "SUCCESS" : "FAIL");

    return 0;
}
