#include <stdio.h>
#include <cuda_runtime.h>
#include <map>
#include <cassert>
#include <cstring>
#include "bitonic.hxx"

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size+2*batch_size

void ParMemcpy(uint64_t *dest, uint64_t *src, int number_of_elements, int nthreads);

void BitonicSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, uint64_t pinned_M_size, int nthreads)
{
    int number_of_batches = number_of_elements / batch_size;
    int number_of_buffers = 2 * batch_size / pinned_M_size;
    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);
    
    uint64_t *pinned_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    cudaHostAlloc( (void**)&pinned_M[0], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_M[1], (pinned_M_size / 2) * sizeof(uint64_t), cudaHostAllocDefault );
    
    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }
       
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
                for (int b = 0; b < number_of_buffers; b++) {
                    //std::memcpy(pinned_M[0],
                    //            &h_key_array[start_index_s0+b*(pinned_M_size/2)],
                    //            (pinned_M_size/2)*sizeof(uint64_t));
                    
                    ParMemcpy(pinned_M[0], &h_key_array[start_index_s0+b*(pinned_M_size/2)], pinned_M_size/2, mem_threads);
                    
                    cudaStreamSynchronize(streams[0]);
                    
                    cudaMemcpyAsync(&d_key_array[0][b*(pinned_M_size/2)],
                                    pinned_M[0],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[0]);
                    cudaStreamSynchronize(streams[0]);
                }
            }
            if (s == 0) {
                bitonicSort<uint64_t, cmp>(d_key_array[0], batch_size, 256, 32, streams[0]);
                for (int b = 0; b < number_of_buffers; b++) {
                    //std::memcpy(pinned_M[1],
                    //            &h_key_array[start_index_s1+b*(pinned_M_size/2)],
                    //            (pinned_M_size/2)*sizeof(uint64_t));
                    
                    ParMemcpy(pinned_M[1], &h_key_array[start_index_s1+b*(pinned_M_size/2)], pinned_M_size/2, mem_threads);
                    
                    cudaStreamSynchronize(streams[1]);
                    
                    cudaMemcpyAsync(&d_key_array[1][b*(pinned_M_size/2)],
                                    pinned_M[1],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[1]);
                    cudaStreamSynchronize(streams[1]);
                }
            }
            else if (s == 1) {
                bitonicSort<uint64_t, cmp>(d_key_array[1], batch_size, 256, 32, streams[1]);
                for (int b = 0; b < number_of_buffers; b++) {
                    cudaMemcpyAsync(pinned_M[0],
                                    &d_key_array[0][b*(pinned_M_size/2)],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[0]);
                    cudaStreamSynchronize(streams[0]);
                    
                    //std::memcpy(&h_key_array[start_index_s0+b*(pinned_M_size/2)],
                        //        pinned_M[0],
                        //        (pinned_M_size/2)*sizeof(uint64_t));
                    
                    ParMemcpy(&h_key_array[start_index_s0+b*(pinned_M_size/2)], pinned_M[0], pinned_M_size/2, mem_threads);
                    
                    cudaStreamSynchronize(streams[0]);
                }
            }
            cudaDeviceSynchronize();
            if (s == 1 && i != (number_of_batches / 2) - 1) {
                for (int b = 0; b < number_of_buffers; b++) {
                    //std::memcpy(pinned_M[0],
                    //            &h_key_array[start_index_s2+b*(pinned_M_size/2)],
                    //            (pinned_M_size/2)*sizeof(uint64_t));
                    
                    cudaMemcpyAsync(pinned_M[1],
                                    &d_key_array[1][b*(pinned_M_size/2)],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
                    ParMemcpy(pinned_M[0], &h_key_array[start_index_s2+b*(pinned_M_size/2)], pinned_M_size/2, mem_threads);
                    cudaStreamSynchronize(streams[1]);
                    
                    //std::memcpy(&h_key_array[start_index_s1+b*(pinned_M_size/2)],
                        //        pinned_M[1],
                        //        (pinned_M_size/2)*sizeof(uint64_t));
                   
                    cudaMemcpyAsync(&d_key_array[0][b*(pinned_M_size/2)],
                                    pinned_M[0],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    streams[0]);
                    ParMemcpy(&h_key_array[start_index_s1+b*(pinned_M_size/2)], pinned_M[1], pinned_M_size/2, mem_threads);
                    cudaStreamSynchronize(streams[0]);
                }
            }
            else if (s == 1 && i == (number_of_batches / 2) - 1) {
                for (int b = 0; b < number_of_buffers; b++) {
                    cudaMemcpyAsync(pinned_M[1],
                                    &d_key_array[1][b*(pinned_M_size/2)],
                                    (pinned_M_size/2)*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
                    cudaStreamSynchronize(streams[1]);
                    
                    //std::memcpy(&h_key_array[start_index_s1+b*(pinned_M_size/2)],
                        //        pinned_M[1],
                        //        (pinned_M_size/2)*sizeof(uint64_t));
                    
                    ParMemcpy(&h_key_array[start_index_s1+b*(pinned_M_size/2)], pinned_M[1], pinned_M_size/2, mem_threads);
                    
                    cudaStreamSynchronize(streams[1]);
                }
            }
        }
    }
    cudaDeviceSynchronize();  
    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
