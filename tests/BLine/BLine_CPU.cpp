#include <stdio.h>
#include <parallel/algorithm>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

#define nthreads 16

int main()
{
    uint64_t number_of_elements = 512L*1024*1024;
    std::vector<uint64_t> h_key_array(number_of_elements);

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i] = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }

    double start = omp_get_wtime();

    //#pragma omp parallel num_threads(nthreads)
    	__gnu_parallel::sort(h_key_array.begin(), h_key_array.end());

    double end = omp_get_wtime();
    printf("%d threads, elapsed time is %f\n", nthreads, end-start);
    return 0;
}
