#include <stdio.h>
#include <parallel/algorithm>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include "type.h"

#define nthreads 20

int main()
{
    uint64_t number_of_elements = 512L*1024*1024;
    uint64_t *h_key_array = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *h_value_array = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i] = ((uint64_t)rand()) << 32 | (uint64_t)rand();
        h_value_array[i] = h_key_array[i];
    }

    struct timeval CPUstart;
    gettimeofday(&CPUstart, NULL);

    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);

    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *indices = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = number_of_elements / nthreads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            indices[i] = i;
        }
    }
    //printf("Real number of threads: %d\n", omp_get_num_threads());
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    __gnu_parallel::sort(indices, indices+number_of_elements, SortIndices(h_key_array));

    //std::sort(h_key_array.begin(), h_key_array.end(), __gnu_parallel::parallel_tag());

    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *sorted_key = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *sorted_value = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = number_of_elements / nthreads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            sorted_key[i] = h_key_array[indices[i]];
            sorted_value[i] = h_value_array[indices[i]];
        }
    }

    struct timeval CPUend;
    gettimeofday(&CPUend, NULL);
    printf("\n%d threads. Elapsed time on CPU: %f s.\n", nthreads, ((CPUend.tv_sec - CPUstart.tv_sec) * 1000000u + CPUend.tv_usec - CPUstart.tv_usec) / 1.e6 );

    return 0;
}
