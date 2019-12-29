#include <stdio.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "PipeData_Thrust.h"

#define start_index 2*i*batch_size+s*batch_size

void PipeDataSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, uint64_t pinned_M_size, int nstreams)
{
    int number_of_batches = number_of_elements / batch_size;
    int number_of_buffers = 2 * batch_size / pinned_M_size;
    
    uint64_t *pinned_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    cudaHostAlloc( (void**)&pinned_M[0], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_M[1], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    
    cudaStream_t streams[nstreams];
    for (int s = 0; s < nstreams; s++) {
        cudaStreamCreate(&streams[s]);
    }
    
    thrust::device_ptr<uint64_t> th_key_array[nstreams];
    for (int s = 0; s < nstreams; s++) {
        th_key_array[s] = thrust::device_pointer_cast(d_key_array[s]);
    }
    
    //int start_index = 0;
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < nstreams; s++) {
            //start_index = 2*i*batch_size+s*batch_size;
            //Staged HtoD
            for (int b = 0; b < number_of_buffers; b++) {
                cudaMemcpyAsync(pinned_M[s], 
				&h_key_array[start_index+b*(pinned_M_size/2)], 
				(pinned_M_size/2)*sizeof(uint64_t), 
				cudaMemcpyHostToHost, 
				streams[s]);
                //cudaStreamSynchronize(streams[s]);
                cudaMemcpyAsync(&d_key_array[s][b*(pinned_M_size/2)], 
				pinned_M[s], 
				(pinned_M_size/2)*sizeof(uint64_t), 
				cudaMemcpyHostToDevice, 
				streams[s]);
                //cudaStreamSynchronize(streams[s]);
            }
            
            //Sort on GPU
	    thrust::sort(thrust::cuda::par.on(streams[s]), th_key_array[s], th_key_array[s]+batch_size);
            //cudaStreamSynchronize(streams[s]);
            
            //Staged DtoH
            for (int b = 0; b < number_of_buffers; b++) {
                cudaMemcpyAsync(pinned_M[s], 
				&d_key_array[s][b*(pinned_M_size/2)], 
				(pinned_M_size/2)*sizeof(uint64_t), 
				cudaMemcpyDeviceToHost, 
				streams[s]);
                //cudaStreamSynchronize(streams[s]);
                cudaMemcpyAsync(&h_key_array[start_index+b*(pinned_M_size/2)], 
				pinned_M[s], 
				(pinned_M_size/2)*sizeof(uint64_t), 
				cudaMemcpyHostToHost, 
				streams[s]);
                //cudaStreamSynchronize(streams[s]);
            }
        }
    }
    
    for (int s = 0; s < nstreams; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
