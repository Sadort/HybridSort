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

    //double start = omp_get_wtime();
    double start = time(NULL);
    
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);

    //printf("Real number of threads: %d\n", omp_get_num_threads());
    __gnu_parallel::sort(h_key_array.begin(), h_key_array.end());
    
    //std::sort(h_key_array.begin(), h_key_array.end(), __gnu_parallel::parallel_tag());

    //double end = omp_get_wtime();
    double end = time(NULL);
    printf("%d threads, elapsed time is %f\n", nthreads, end-start);
   
    return 0;
}
