#include <stdio.h>
#include <stdlib.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size-batch_size

void PipeDataSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, uint64_t pinned_M_size, int nstreams = 2)
{
    int number_of_batches = number_of_elements / batch_size;
    int number_of_buffers = 2 * batch_size / pinned_M_size;
    
    uint64_t *pinned_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    cudaHostAlloc( (void**)&pinned_M[0], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_M[1], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    
    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }
    
    thrust::device_ptr<uint64_t> th_key_array[2];
    for (int s = 0; s < 2; s++) {
        th_key_array[s] = thrust::device_pointer_cast(d_key_array[s]);
    }
    
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
                for (int b = 0; b < number_of_buffers; b++) {
                    std::memcpy(pinned_M[0],
                                &h_key_array[start_index_s0+b*(pinned_M_size/2)],
                                (pinned_M_size/2)*sizeof(uint64_t));
                    cudaStreamSynchronize(streams[0]);
                    
                    cudaMemcpyAsync(&d_key_array[0][b*(pinned_M_size/2)],
                                    pinned_M[0],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[0]);
                    cudaStreamSynchronize(streams[0]);
                }
                thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                cudaStreamSynchronize(streams[0]);
            }
            else if (i > 0 && s == 0) {
                //Overlapping
                for (int b = 0; b < number_of_buffers; b++) {
                    std::memcpy(pinned_M[0],
                                &h_key_array[start_index_s0+b*(pinned_M_size/2)],
                                (pinned_M_size/2)*sizeof(uint64_t));
                    
                    cudaMemcpyAsync(pinned_M[1],
                                    &d_key_array[1][b*(pinned_M_size/2)],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
                    cudaStreamSynchronize(streams[1]);
                    
                    std::memcpy(&h_key_array[start_index_s2+b*(pinned_M_size/2)],
                                pinned_M[1],
                                (pinned_M_size/2)*sizeof(uint64_t));
                    
                    cudaMemcpyAsync(&d_key_array[0][b*(pinned_M_size/2)],
                                    pinned_M[0],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[0]);
                    cudaStreamSynchronize(streams[0]);
                }
                thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                cudaStreamSynchronize(streams[0]);
            }
            else if (s == 1) {
                //Overlapping
                for (int b = 0; b < number_of_buffers; b++) {
                    std::memcpy(pinned_M[1],
                                &h_key_array[start_index_s1+b*(pinned_M_size/2)],
                                (pinned_M_size/2)*sizeof(uint64_t));
                    
                    cudaMemcpyAsync(pinned_M[0],
                                    &d_key_array[0][b*(pinned_M_size/2)],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[0]);
                    cudaStreamSynchronize(streams[0]);
                    
                    std::memcpy(&h_key_array[start_index_s0+b*(pinned_M_size/2)],
                                pinned_M[0],
                                (pinned_M_size/2)*sizeof(uint64_t));
                    
                    cudaMemcpyAsync(&d_key_array[1][b*(pinned_M_size/2)],
                                    pinned_M[1],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[1]);
                    cudaStreamSynchronize(streams[1]);
                }
                thrust::sort(thrust::cuda::par.on(streams[1]), th_key_array[1], th_key_array[1]+batch_size);
                cudaStreamSynchronize(streams[1]);
                
                if (i == (number_of_batches / 2) - 1) {
                    for (int b = 0; b < number_of_buffers; b++) {
                        cudaMemcpyAsync(pinned_M[1],
                                        &d_key_array[1][b*(pinned_M_size/2)],
                                        (pinned_M_size/2)*sizeof(uint64_t),
                                        cudaMemcpyDeviceToHost,
                                        streams[1]);
                        cudaStreamSynchronize(streams[1]);
                        
                        std::memcpy(&h_key_array[start_index_s1+b*(pinned_M_size/2)],
                                    pinned_M[1],
                                    (pinned_M_size/2)*sizeof(uint64_t));
                        cudaStreamSynchronize(streams[1]);
                    }
                }
            }
            
            
        }
        
    }
    
    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
