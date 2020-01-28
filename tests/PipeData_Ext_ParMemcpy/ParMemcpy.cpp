#include <cstring>
#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>
#include "type.h"

void ParMemcpy(ulong2 *dest, ulong2 *src, int number_of_elements, int nthreads)
{
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {
            int tid = omp_get_thread_num();
            int len = number_of_elements / nthreads;
            int start_ind = tid * len;
            std::memcpy(&dest[start_ind], &src[start_ind], len*sizeof(ulong2));
    }
    return;
}
