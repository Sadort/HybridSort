#include <cstring>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>

void ParMemcpy(uint64_t *dest, uint64_t *src, int number_of_elements, int nthreads)
{
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {
            int tid = omp_get_thread_num();
            int len = number_of_elements / nthreads;
            int start_ind = tid * len;
            std::memcpy(&dest[start_ind], &src[start_ind], len*sizeof(uint64_t));
    }
    return;
}

void PairMerge(uint64_t *key_array_1, uint64_t *key_array_2, uint64_t batch_size, int nthreads)
{
    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);

    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id1 = nvtxRangeStart("Pairwise-merge");
    
    std::vector< std::pair<uint64_t*, uint64_t*> > batches;
    batches.push_back(std::make_pair(&key_array_1[0], &key_array_1[batch_size]));
    batches.push_back(std::make_pair(&key_array_2[0], &key_array_2[batch_size]));
    uint64_t *output_v = (uint64_t *)malloc(2*batch_size*sizeof(uint64_t));
    
    __gnu_parallel::multiway_merge(batches.begin(), batches.end(), output_v, 2*batch_size, std::less<uint64_t>());

    //std::memcpy(key_array_1, output_v, (batch_size)*sizeof(uint64_t));
    //std::memcpy(key_array_2, &output_v[batch_size], (batch_size)*sizeof(uint64_t));
    ParMemcpy(key_array_1, output_v, batch_size, mem_threads);
    ParMemcpy(key_array_2, &output_v[batch_size], batch_size, mem_threads);
    
    nvtxRangeEnd(id1);
   
    return;
}

