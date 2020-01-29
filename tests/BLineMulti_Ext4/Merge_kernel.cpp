#include <cstring>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>

extern const unsigned long MASK;
extern __host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b);

void PairMerge(ulong2 *key_array_1, ulong2 *key_array_2, uint64_t batch_size, int nthreads)
{
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id1 = nvtxRangeStart("Pairwise-merge");
    
    std::vector< std::pair<ulong2*, ulong2*> > batches;
    batches.push_back(std::make_pair(&key_array_1[0], &key_array_1[batch_size]));
    batches.push_back(std::make_pair(&key_array_2[0], &key_array_2[batch_size]));
    ulong2 *output_v = (ulong2 *)malloc(2*batch_size*sizeof(ulong2));
    
    __gnu_parallel::multiway_merge(batches.begin(), batches.end(), output_v, 2*batch_size, std::less<ulong2>());

    std::memcpy(key_array_1, output_v, (batch_size)*sizeof(ulong2));
    std::memcpy(key_array_2, &output_v[batch_size], (batch_size)*sizeof(ulong2));
        
    nvtxRangeEnd(id1);
   
    return;
}
