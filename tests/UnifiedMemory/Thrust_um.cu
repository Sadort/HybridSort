#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream> 
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;

int main(void)
{
    uint64_t number_of_elements = 1024L*1024*1024;
    uint64_t *h_key_array;

    cudaMallocManaged(&h_key_array, number_of_elements*sizeof(uint64_t));

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i] = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }
    printf("size : %lu\n", sizeof(uint64_t));

    thrust::device_ptr<uint64_t> th_key_array( h_key_array );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float totalseconds = 0;
    int iterations = 1;
    for(int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start, 0);
        //thrust::sort_by_key( th_key_array, th_key_array+number_of_elements, th_value_array );
        thrust::sort( th_key_array, th_key_array+number_of_elements );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalseconds = totalseconds + milliseconds;
        if(i == iterations - 1) break;
    }
    printf("Elapsed time: %f s.\n", totalseconds/(iterations*1000));

    //std::sort(h_key_ref.begin(), h_key_ref.end());
    //bool result = compareAB(h_key_array, h_key_ref);
    //printf("Test: %s\n", result == true ? "SUCCESS" : "FAIL");

    return 0;
}
