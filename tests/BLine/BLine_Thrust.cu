#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream> 
#include <cuda_runtime.h>
#include <algorithm>
#include "type.h"

using namespace std;

int main(void)
{
    uint64_t number_of_elements = 256L*1024*1024;
    ulong2 *h_key_array = (ulong2 *)malloc(number_of_elements*sizeof(ulong2));
    ulong2 *d_key_array;

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i].x = ((uint64_t)rand()) << 32 | (uint64_t)rand();
        h_key_array[i].y = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }
    printf("size : %lu\n", sizeof(ulong2));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start, 0);

    cudaMalloc( (void**)&d_key_array, number_of_elements * sizeof(ulong2) );
    cudaMemcpy( d_key_array,
                h_key_array, 
                number_of_elements * sizeof(ulong2),
                cudaMemcpyHostToDevice );

    thrust::device_ptr<ulong2> th_key_array( d_key_array );
    
    //thrust::sort_by_key( th_key_array, th_key_array+number_of_elements, th_value_array );
    thrust::sort( th_key_array, th_key_array+number_of_elements );

    cudaMemcpy( h_key_array,
                d_key_array,
                number_of_elements * sizeof(ulong2),
                cudaMemcpyDeviceToHost );

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time: %f s.\n", milliseconds/1000);

    printf("Test: %s\n", std::is_sorted(h_key_array, h_key_array+number_of_elements) == true ? "SUCCESS" : "FAIL");

    //std::sort(h_key_ref.begin(), h_key_ref.end());
    //bool result = compareAB(h_key_array, h_key_ref);
    //printf("Test: %s\n", result == true ? "SUCCESS" : "FAIL");

    return 0;
}
