#include <stdio.h>
#include <stdlib.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include "type.h"

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size-batch_size

void ThrustSort(ulong2 *h_key_array, ulong2 *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, int nstreams = 2)
{
    int number_of_batches = number_of_elements / batch_size;
    
    ulong2 *pinned_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(ulong2) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(ulong2) );
    cudaHostAlloc( (void**)&pinned_M[0], batch_size * sizeof(ulong2), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_M[1], batch_size * sizeof(ulong2), cudaHostAllocDefault );
    
    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }
    
    thrust::device_ptr<ulong2> th_key_array[2];
    for (int s = 0; s < 2; s++) {
        th_key_array[s] = thrust::device_pointer_cast(d_key_array[s]);
    }
    
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
                std::memcpy(pinned_M[0],
                            &h_key_array[start_index_s0],
                            batch_size*sizeof(ulong2));
                cudaMemcpyAsync(d_key_array[0],
                                pinned_M[0],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaDeviceSynchronize();
                thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                cudaDeviceSynchronize();
            }
            else if (i > 0 && s == 0) {
                //Overlapping
                std::memcpy(pinned_M[0],
                            &h_key_array[start_index_s0],
                            batch_size*sizeof(ulong2));
                
                cudaMemcpyAsync(pinned_M[1],
                                d_key_array[1],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyDeviceToHost,
                                streams[1]);
                cudaMemcpyAsync(d_key_array[0],
                                pinned_M[0],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaDeviceSynchronize();
                
                std::memcpy(&h_key_array[start_index_s2],
                            pinned_M[1],
                            batch_size*sizeof(ulong2));
                
                thrust::sort(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
                cudaDeviceSynchronize();
            }
            else if (s == 1) {
                //Overlapping
                std::memcpy(pinned_M[1],
                            &h_key_array[start_index_s1],
                            batch_size*sizeof(ulong2));
                
                cudaMemcpyAsync(pinned_M[0],
                                d_key_array[0],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                cudaMemcpyAsync(d_key_array[1],
                                pinned_M[1],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyHostToDevice,
                                streams[1]);
                cudaDeviceSynchronize();
                
                std::memcpy(&h_key_array[start_index_s0],
                            pinned_M[0],
                            batch_size*sizeof(ulong2));
                
                thrust::sort(thrust::cuda::par.on(streams[1]), th_key_array[1], th_key_array[1]+batch_size);
                cudaDeviceSynchronize();
                
                if (i == (number_of_batches / 2) - 1) {
                    cudaMemcpyAsync(pinned_M[1],
                                    d_key_array[1],
                                    batch_size*sizeof(ulong2),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
                    std::memcpy(&h_key_array[start_index_s1],
                            pinned_M[1],
                            batch_size*sizeof(ulong2));
                    cudaDeviceSynchronize();
                }
            }
            
            
        }
        
    }
    
    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
