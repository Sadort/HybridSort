#include <stdio.h>
#include "BLineMulti_Thrust.h"

void BLineMultiSort(uint64_t *h_key_array, uint64_t number_of_elements, uint64_t batch_size)
{
    int number_of_batches = number_of_elements / batch_size;
    uint64_t *d_key_array;

    cudaMalloc( (void**)&d_key_array, batch_size * sizeof(uint64_t) );
    thrust::device_ptr<uint64_t> th_key_array( d_key_array );

    for (int i = 0; i < number_of_batches; i++)
    {
        cudaMemcpy( d_key_array,
                h_key_array+i*batch_size, 
                number_of_elements * sizeof(uint64_t),
                cudaMemcpyHostToDevice );

        //thrust::sort_by_key( th_key_array, th_key_array+number_of_elements, th_value_array );
        thrust::sort( th_key_array, th_key_array+number_of_elements );

        cudaMemcpy( h_key_array+i*batch_size,
                d_key_array,
                number_of_elements * sizeof(uint64_t),
                cudaMemcpyDeviceToHost );
    }
    cudaDeviceSynchronize();
    
    return;
}
