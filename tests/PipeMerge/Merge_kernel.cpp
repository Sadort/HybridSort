#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

void PairMerge(uint64_t *key_array_1, uint64_t *key_array_2, uint64_t batch_size, int nthreads)
{
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id1 = nvtxRangeStart("Pairwise-merge");
    
    std::vector<uint64_t> v1(key_array_1, key_array_1+batch_size);
    std::vector<uint64_t> v2(key_array_2, key_array_2+batch_size);
    std::vector<uint64_t> output_v(2*batch_size);
    
    __gnu_parallel::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), output_v.begin(), std::less<uint64_t>());
    
    std::copy(output_v.begin(), output_v.begin()+batch_size, key_array_1);
    std::copy(output_v.begin()+batch_size, output_v.end(), key_array_2);
    
    nvtxRangeEnd(id1);
   
    return;
}
