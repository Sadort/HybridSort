#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include "bitonic.hxx"

void BitonicSort(uint64_t *h_key_array, uint64_t *d_key_array, uint64_t number_of_elements, uint64_t batch_size)
{
    int number_of_batches = number_of_elements / batch_size;

    cudaMalloc( (void**)&d_key_array, batch_size * sizeof(uint64_t) );
    cudaMemcpy( d_key_array,
                h_key_array,
                batch_size * sizeof(uint64_t),
                cudaMemcpyHostToDevice );

    for (int i = 0; i < number_of_batches; i++)
    {        
        bitonicSort<uint64_t, cmp>(d_key_array, batch_size, 256, 32);
        
        cudaMemcpy( &h_key_array[i*batch_size],
                d_key_array,
                batch_size * sizeof(uint64_t),
                cudaMemcpyDeviceToHost );
        cudaDeviceSynchronize();
        
	if(i == number_of_batches-1)
	    break;

	cudaMemcpy( d_key_array,
                &h_key_array[(i+1)*batch_size],
                batch_size * sizeof(uint64_t),
                cudaMemcpyHostToDevice );
	cudaDeviceSynchronize();
        
    }
   
    return;
}
