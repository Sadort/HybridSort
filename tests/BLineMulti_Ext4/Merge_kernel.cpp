#include <cstring>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>
#include "type.h"

void PairMerge(uint64_t *key_array, uint64_t *value_array, uint64_t batch_size, int nthreads)
{
    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);

    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *indices = (uint64_t *)malloc(2*batch_size*sizeof(uint64_t));
    uint64_t *sorted_indices = (uint64_t *)malloc(2*batch_size*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = 2*batch_size / mem_threads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            indices[i] = i;
        }
     }

    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id1 = nvtxRangeStart("Pairwise-merge");

    std::vector< std::pair<uint64_t*, uint64_t*> > batches;
    batches.push_back(std::make_pair(&indices[0], &indices[batch_size]));
    batches.push_back(std::make_pair(&indices[batch_size], &indices[2*batch_size]));

    __gnu_parallel::multiway_merge(batches.begin(), batches.end(), sorted_indices, 2*batch_size, SortIndices(key_array));

    nvtxRangeEnd(id1);

    free(indices);
    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *sorted_key = (uint64_t *)malloc(2*batch_size*sizeof(uint64_t));
    uint64_t *sorted_value = (uint64_t *)malloc(2*batch_size*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = 2*batch_size / mem_threads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            sorted_key[i] = key_array[sorted_indices[i]];
            sorted_value[i] = value_array[sorted_indices[i]];
        }
    }
    free(sorted_indices);

    std::memcpy(key_array, sorted_key, (2*batch_size)*sizeof(uint64_t));
    std::memcpy(value_array, sorted_value, (2*batch_size)*sizeof(uint64_t));

    free(sorted_key);
    free(sorted_value);
    return;
}
