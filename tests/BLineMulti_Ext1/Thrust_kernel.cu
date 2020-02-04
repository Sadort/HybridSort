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

void ThrustSort(uint64_t *h_key_array, uint64_t *d_key_array[2], uint64_t *h_value_array, uint64_t *d_value_array[2], uint64_t number_of_elements, uint64_t batch_size)
{
    int number_of_batches = number_of_elements / batch_size;

    uint64_t *pinned_key_M[2];
    uint64_t *pinned_value_M[2];

    cudaMalloc( (void**)&d_key_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_key_array[1], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_value_array[0], batch_size * sizeof(uint64_t) );
    cudaMalloc( (void**)&d_value_array[1], batch_size * sizeof(uint64_t) );
    cudaHostAlloc( (void**)&pinned_key_M[0], batch_size * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_key_M[1], batch_size * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_value_M[0], batch_size * sizeof(uint64_t), cudaHostAllocDefault );
    cudaHostAlloc( (void**)&pinned_value_M[1], batch_size * sizeof(uint64_t), cudaHostAllocDefault );

    cudaStream_t streams[2];
    for (int s = 0; s < 2; s++) {
        cudaStreamCreate(&streams[s]);
    }

    thrust::device_ptr<uint64_t> th_key_array[2];
    thrust::device_ptr<uint64_t> th_value_array[2];
    for (int s = 0; s < 2; s++) {
        th_key_array[s] = thrust::device_pointer_cast(d_key_array[s]);
        th_value_array[s] = thrust::device_pointer_cast(d_value_array[s]);
    }

    for (int i = 0; i < number_of_batches / 2; i++) {
        for (int s = 0; s < 2; s++) {
            if (i == 0 && s == 0) {
                std::memcpy(pinned_key_M[0],
                            &h_key_array[start_index_s0],
                            batch_size*sizeof(uint64_t));
                std::memcpy(pinned_value_M[0],
                            &h_value_array[start_index_s0],
                            batch_size*sizeof(uint64_t));

                cudaMemcpyAsync(d_key_array[0],
                                pinned_key_M[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaMemcpyAsync(d_value_array[0],
                                pinned_value_M[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[1]);

                cudaDeviceSynchronize();
                thrust::sort_by_key(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size, th_value_array[0]);
                cudaDeviceSynchronize();
            }
            else if (i > 0 && s == 0) {
                //Overlapping
                std::memcpy(pinned_key_M[0],
                            &h_key_array[start_index_s0],
                            batch_size*sizeof(uint64_t));

                std::memcpy(pinned_value_M[0],
                            &h_value_array[start_index_s0],
                            batch_size*sizeof(uint64_t));

                cudaMemcpyAsync(pinned_key_M[1],
                                d_key_array[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[1]);
                cudaMemcpyAsync(d_key_array[0],
                                pinned_key_M[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[0]);

                cudaDeviceSynchronize();

                std::memcpy(&h_key_array[start_index_s2],
                            pinned_key_M[1],
                            batch_size*sizeof(uint64_t));
                std::memcpy(&h_value_array[start_index_s2],
                            pinned_value_M[1],
                            batch_size*sizeof(uint64_t));

                thrust::sort_by_key(thrust::cuda::par.on(streams[0]), th_key_array[0], th_key_array[0]+batch_size, th_value_array[0]);
                cudaDeviceSynchronize();
            }
            else if (s == 1) {
                //Overlapping
                std::memcpy(pinned_key_M[1],
                            &h_key_array[start_index_s1],
                            batch_size*sizeof(uint64_t));
                std::memcpy(pinned_value_M[1],
                            &h_value_array[start_index_s1],
                            batch_size*sizeof(uint64_t));

                cudaMemcpyAsync(pinned_key_M[0],
                                d_key_array[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                cudaMemcpyAsync(d_key_array[1],
                                pinned_key_M[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[1]);
                cudaMemcpyAsync(pinned_value_M[0],
                                d_value_array[0],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                cudaMemcpyAsync(d_value_array[1],
                                pinned_value_M[1],
                                batch_size*sizeof(uint64_t),
                                cudaMemcpyHostToDevice,
                                streams[1]);
                cudaDeviceSynchronize();

                std::memcpy(&h_key_array[start_index_s0],
                            pinned_key_M[0],
                            batch_size*sizeof(uint64_t));
                std::memcpy(&h_value_array[start_index_s0],
                            pinned_value_M[0],
                            batch_size*sizeof(uint64_t));

                thrust::sort_by_key(thrust::cuda::par.on(streams[1]), th_key_array[1], th_key_array[1]+batch_size, th_value_array[1]);
                cudaDeviceSynchronize();

                if (i == (number_of_batches / 2) - 1) {
                    cudaMemcpyAsync(pinned_key_M[1],
                                    d_key_array[1],
                                    batch_size*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[1]);
                    cudaMemcpyAsync(pinned_value_M[1],
                                    d_value_array[1],
                                    batch_size*sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams[0]);
                    cudaDeviceSynchronize();
                    std::memcpy(&h_key_array[start_index_s1],
                                pinned_key_M[1],
                                batch_size*sizeof(uint64_t));
                    std::memcpy(&h_value_array[start_index_s1],
                                pinned_value_M[1],
                                batch_size*sizeof(uint64_t));

                }
            }


        }

    }

    for (int s = 0; s < 2; s++) {
        cudaStreamDestroy(streams[s]);
    }

    return;
}
