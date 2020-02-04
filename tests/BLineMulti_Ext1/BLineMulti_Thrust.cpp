#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <nvToolsExt.h>
#include "type.h"

uint64_t number_of_elements = 1024L*1024*1024;
uint64_t batch_size = 128L*1024*1024;
int nthreads = 20;
int nstreams = 2;

void ThrustSort(uint64_t *h_key_array, uint64_t *d_key_array[], uint64_t *h_value_array, uint64_t *d_value_array[], uint64_t number_of_elements, uint64_t batch_size);

int main(void)
{
    int number_of_batches = number_of_elements / batch_size;
    uint64_t *h_key_array = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *h_value_array = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *d_key_array[2];
    uint64_t *d_value_array[2];

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i] = ((uint64_t)rand()) << 32 | (uint64_t)rand();
        h_value_array[i] = h_key_array[i];
    }

    printf("size : %lu\n", sizeof(uint64_t));

    /**************************/
    /* Sorting batches on GPU */
    /**************************/

    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    float GPU_milliseconds = 0;

    cudaEventRecord(GPUstart, 0);

    ThrustSort(h_key_array, d_key_array, h_value_array, d_value_array, number_of_elements, batch_size);

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);
    cudaEventElapsedTime(&GPU_milliseconds, GPUstart, GPUstop);

    /**************************/
    /* Merging batches on GPU */
    /**************************/

    struct timeval CPUstart;
    gettimeofday(&CPUstart, NULL);

    int mem_threads = (int)log2((float)nthreads);
    mem_threads = (int)exp2((float)mem_threads);

    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *indices = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *sorted_indices = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = number_of_elements / mem_threads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            indices[i] = i;
        }
    }

    std::vector< std::pair<uint64_t*, uint64_t*> > batches;
    for (int i = 0; i < number_of_batches; i++)
    {
        batches.push_back(std::make_pair(&indices[i*batch_size], &indices[(i+1)*batch_size]));
    }

    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id0 = nvtxRangeStart("Multiway-merge");

    //printf("Real number of threads: %d\n", omp_get_num_threads());
    __gnu_parallel::multiway_merge(batches.begin(), batches.end(), sorted_indices, number_of_elements, SortIndices(h_key_array));

    nvtxRangeEnd(id0);
    //double end = omp_get_wtime();

    free(indices);
    omp_set_dynamic(false);
    omp_set_num_threads(mem_threads);
    uint64_t *sorted_key = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    uint64_t *sorted_value = (uint64_t *)malloc(number_of_elements*sizeof(uint64_t));
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t len = number_of_elements / mem_threads;
        uint64_t start_ind = tid * len;
        for (uint64_t i = start_ind; i < start_ind + len; i++) {
            sorted_key[i] = h_key_array[sorted_indices[i]];
            sorted_value[i] = h_value_array[sorted_indices[i]];
        }
    }
    free(sorted_indices);

    struct timeval CPUend;
    gettimeofday(&CPUend, NULL);

    printf("Elapsed time on GPU: %f s.\n", (GPU_milliseconds/1000));
    printf("Elapsed time on CPU: %f s.\n", ((CPUend.tv_sec - CPUstart.tv_sec) * 1000000u + CPUend.tv_usec - CPUstart.tv_usec) / 1.e6 );

    printf("Test: %s\n", (std::is_sorted(sorted_key, sorted_key+number_of_elements) && std::is_sorted(sorted_value, sorted_value+number_of_elements)) == true ? "SUCCESS" : "FAIL");
    
//    std::vector<uint64_t> h_key_ref(sorted_array, sorted_array+number_of_elements);
//    printf("Test: %s\n", std::is_sorted(h_key_ref.begin(), h_key_ref.end()) == true ? "SUCCESS" : "FAIL");

//    std::vector<uint64_t> h_key_ref(h_key_array, h_key_array+number_of_elements);
//    std::sort(h_key_ref.begin(), h_key_ref.end());
//    std::vector<uint64_t> sorted_v(sorted_array, sorted_array+number_of_elements);
//    bool result = (sorted_v == h_key_ref);
//    printf("Test: %s\n", result == true ? "SUCCESS" : "FAIL");

    return 0;
}
