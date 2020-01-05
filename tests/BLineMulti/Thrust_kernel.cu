#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>

void BLineMultiSort(uint64_t *h_key_array, uint64_t *d_key_array, uint64_t number_of_elements, uint64_t batch_size)
{
    int number_of_batches = number_of_elements / batch_size;

    cudaMalloc( (void**)&d_key_array, batch_size * sizeof(uint64_t) );
    cudaMemcpy( d_key_array,
                h_key_array,
                batch_size * sizeof(uint64_t),
                cudaMemcpyHostToDevice );
    thrust::device_ptr<uint64_t> th_key_array( d_key_array );

    for (int i = 0; i < number_of_batches; i++)
    {        
        //thrust::sort_by_key( th_key_array, th_key_array+number_of_elements, th_value_array );
        thrust::sort( th_key_array, th_key_array+batch_size );
        
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
