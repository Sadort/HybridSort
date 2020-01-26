#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include "bitonic.hxx"

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size-batch_size

#define merge_index_1 2*(i-1)*batch_size
#define merge_index_2 2*(i-1)*batch_size+batch_size

void PairMerge(uint64_t *key_array_1, uint64_t *key_array_2, uint64_t batch_size, int nthreads);
void ParMemcpy(uint64_t *dest, uint64_t *src, int number_of_elements, int nthreads);

void BitonicSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, int nthreads)
{
    int number_of_batches = number_of_elements / batch_size;
    int mem_threads = (int)exp2((int)log2(nthreads));
    
    uint64_t *pinned_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    cudaHostAlloc( (void**)&pinned_M[0], batch_size * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_M[1], batch_size * sizeof(uint64_t), cudaHostAllocDefault );
    
    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }
    
    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
                cudaMemcpyAsync(d_key_array[0],
                                &h_key_array[start_index_s0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaDeviceSynchronize();
            }
            if (s == 0)
            {
                //std::memcpy(pinned_M[1],
                //            &h_key_array[start_index_s1],
                //            batch_size*sizeof(uint64_t));
                
                ParMemcpy(pinned_M[1], &h_key_array[start_index_s1], batch_size, mem_threads);
                
                cudaMemcpyAsync(d_key_array[1],
                                pinned_M[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[1]);
                bitonicSort<uint64_t, cmp>(d_key_array[0], batch_size, 256, 32, streams[0]);
                if (i > 0)
                {
                    PairMerge(&h_key_array[merge_index_1], &h_key_array[merge_index_2], batch_size, nthreads);
                }
                cudaDeviceSynchronize();
            }
            else if (s == 1)
            {
                cudaMemcpyAsync(pinned_M[0],
                                d_key_array[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                bitonicSort<uint64_t, cmp>(d_key_array[1], batch_size, 256, 32, streams[1]);
                cudaDeviceSynchronize();
                
                ParMemcpy(&h_key_array[start_index_s0], pinned_M[0], batch_size, mem_threads);
                
                //std::memcpy(&h_key_array[start_index_s0],
                //            pinned_M[0],
                //            batch_size*sizeof(uint64_t));
            }
            if (s == 1 && i != (number_of_batches / 2) - 1) {
                //std::memcpy(pinned_M[0],
                //            &h_key_array[start_index_s2],
                //            batch_size*sizeof(uint64_t));
                
                ParMemcpy(pinned_M[0], &h_key_array[start_index_s2], batch_size, mem_threads);
                
                cudaMemcpyAsync(pinned_M[1],
                                d_key_array[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[1]);
                cudaMemcpyAsync(d_key_array[0],
                                pinned_M[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaDeviceSynchronize();
                
                ParMemcpy(&h_key_array[start_index_s1], pinned_M[1], batch_size, mem_threads);
                
                //std::memcpy(&h_key_array[start_index_s1],
                //            pinned_M[1],
                //            batch_size*sizeof(uint64_t));
            }
            else if (s == 1 && i == (number_of_batches / 2) - 1) {
                cudaMemcpyAsync(&h_key_array[start_index_s1],
                                d_key_array[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[1]);
                cudaDeviceSynchronize();
            }
        }
    }
    
    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }
   
    return;
}
