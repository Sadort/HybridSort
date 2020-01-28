#include <stdio.h>
#include <math.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <map>
#include <cassert>
#include "type.h"

#define start_index_s0 2*i*batch_size
#define start_index_s1 2*i*batch_size+batch_size
#define start_index_s2 2*i*batch_size-batch_size

#define merge_index_1 2*(i-1)*batch_size
#define merge_index_2 2*(i-1)*batch_size+batch_size

void PairMerge(ulong2 *key_array_1, ulong2 *key_array_2, uint64_t batch_size, int nthreads);
void ParMemcpy(ulong2 *dest, ulong2 *src, int number_of_elements, int nthreads);

// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;

    cached_allocator() {}

    ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
      char *result = 0;

      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

      if(free_block != free_blocks.end())
      {
        std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
        {
          std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;

          // allocate memory and convert cuda::pointer to raw pointer
          result = thrust::cuda::malloc<char>(num_bytes).get();
        }
        catch(std::runtime_error &e)
        {
          throw;
        }
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_bytes));

      return result;
    }

    void deallocate(char *ptr, size_t n)
    {
      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

  private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all()
    {
      std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

      // deallocate all outstanding blocks in both lists
      for(free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
      }

      for(allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
    }

};

void ThrustSort(ulong2 *h_key_array, ulong2 *d_key_array[2], uint64_t number_of_elements, uint64_t batch_size, int nthreads)
{
    cached_allocator alloc;
    int number_of_batches = number_of_elements / batch_size;
    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);
    
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
                cudaMemcpyAsync(d_key_array[0],
                                &h_key_array[start_index_s0],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyHostToDevice,
                                streams[0]);
                cudaDeviceSynchronize();
            }
            if (s == 0)
            {
                ParMemcpy(pinned_M[1], &h_key_array[start_index_s1], batch_size, mem_threads);
                
                cudaMemcpyAsync(d_key_array[1],
                                pinned_M[1],
                                batch_size*sizeof(ulong2),
                                cudaMemcpyHostToDevice,
                                streams[1]);
                thrust::sort(thrust::cuda::par(alloc).on(streams[0]), th_key_array[0], th_key_array[0]+batch_size);
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
                                batch_size*sizeof(ulong2),
                                cudaMemcpyDeviceToHost,
                                streams[0]);
                thrust::sort(thrust::cuda::par(alloc).on(streams[1]), th_key_array[1], th_key_array[1]+batch_size);
                cudaDeviceSynchronize();
                
                ParMemcpy(&h_key_array[start_index_s0], pinned_M[0], batch_size, mem_threads);
                
            }
            if (s == 1 && i != (number_of_batches / 2) - 1) {
                
                ParMemcpy(pinned_M[0], &h_key_array[start_index_s2], batch_size, mem_threads);
                
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
                
                ParMemcpy(&h_key_array[start_index_s1], pinned_M[1], batch_size, mem_threads);
                
            }
            else if (s == 1 && i == (number_of_batches / 2) - 1) {
                cudaMemcpyAsync(&h_key_array[start_index_s1],
                                d_key_array[1],
                                batch_size*sizeof(ulong2),
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
