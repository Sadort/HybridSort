#include <stdio.h>
#include <parallel/algorithm>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "type.h"

#define nthreads 20

__host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b) {
    if      (a.x < b.x) return true;
    else return false;
}

int main()
{
    uint64_t number_of_elements = 4096L*1024*1024;
    std::vector<ulong2> h_key_array(number_of_elements);

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i].x = ((uint64_t)rand()) << 32 | (uint64_t)rand();
        h_key_array[i].y = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }

    struct timeval CPUstart;
    gettimeofday(&CPUstart, NULL);

    //printf("Real number of threads: %d\n", omp_get_num_threads());
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    __gnu_parallel::sort(h_key_array.begin(), h_key_array.end());

    //std::sort(h_key_array.begin(), h_key_array.end(), __gnu_parallel::parallel_tag());

    struct timeval CPUend;
    gettimeofday(&CPUend, NULL);
    printf("\n%d threads. Elapsed time on CPU: %f s.\n", nthreads, ((CPUend.tv_sec - CPUstart.tv_sec) * 1000000u + CPUend.tv_usec - CPUstart.tv_usec) / 1.e6 );

    return 0;
}
