#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "bitonic.hxx"

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size-batch_size

void BitonicSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, int nstreams = 2)
{
    int number_of_batches = number_of_elements / batch_size;

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    
    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }
    
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
		cudaMemcpyAsync(d_key_array[0],
                               &h_key_array[start_index_s0],
                               (batch_size)*sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
               		       streams[0]);
                //thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                bitonicSort<uint64_t, cmp>(d_key_array[0], batch_size, 256, 32, streams[0]);
		cudaStreamSynchronize(streams[0]);
            }
            else if (i > 0 && s == 0) {
                //Overlapping
		cudaMemcpyAsync(&h_key_array[start_index_s2],
                                d_key_array[1],
                                (batch_size)*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[1]);
		cudaMemcpyAsync(d_key_array[0],
                                &h_key_array[start_index_s0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[0]);
		cudaDeviceSynchronize();
                //thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                bitonicSort<uint64_t, cmp>(d_key_array[0], batch_size, 256, 32, streams[0]);
		cudaStreamSynchronize(streams[0]);
            }
            else if (s == 1) {
                //Overlapping
                cudaMemcpyAsync(&h_key_array[start_index_s0],
                                d_key_array[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                cudaMemcpyAsync(d_key_array[1],
                                &h_key_array[start_index_s1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[1]);
		cudaDeviceSynchronize();
                //thrust::sort(thrust::cuda::par.on(streams[1]), th_key_array[1], th_key_array[1]+batch_size);
                bitonicSort<uint64_t, cmp>(d_key_array[1], batch_size, 256, 32, streams[1]);
		cudaStreamSynchronize(streams[1]);
                
                if (i == (number_of_batches / 2) - 1) {
		    cudaMemcpyAsync(&h_key_array[start_index_s1],
                                    d_key_array[1],
                                    batch_size*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
		    cudaStreamSynchronize(streams[1]);
                }
            }
            
            
        }
        
    }
    
    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
