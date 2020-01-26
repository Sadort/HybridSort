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
