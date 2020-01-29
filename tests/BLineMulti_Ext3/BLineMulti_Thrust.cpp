#include <iostream>
#include <algorithm>
#include <parallel/algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <nvToolsExt.h>

uint64_t number_of_elements = 512L*1024*1024;
uint64_t batch_size = 128L*1024*1024;
int nthreads = 8;
int nstreams = 2;

extern const unsigned long MASK;
extern __host__ __device__ bool operator<(const ulong2 &a, const ulong2 &b);

void ThrustSort(ulong2 *h_key_array, ulong2 *d_key_array[], uint64_t number_of_elements, uint64_t batch_size, int nthreads);

int main(void)
{
    int number_of_batches = number_of_elements / batch_size;
    ulong2 *h_key_array = (ulong2 *)malloc(number_of_elements*sizeof(ulong2));
    ulong2 *sorted_array = (ulong2 *)malloc(number_of_elements*sizeof(ulong2));
    ulong2 *d_key_array[2];

    for (uint64_t i = 0; i < number_of_elements; i++) {
        h_key_array[i].x = ((uint64_t)rand()) << 32 | (uint64_t)rand();
        h_key_array[i].y = ((uint64_t)rand()) << 32 | (uint64_t)rand();
    }
    
    printf("size : %lu\n", sizeof(ulong2));

    /**************************/
    /* Sorting batches on GPU */
    /**************************/

    cudaEvent_t GPUstart, GPUstop;
    cudaEventCreate(&GPUstart);
    cudaEventCreate(&GPUstop);
    float GPU_milliseconds = 0;

    cudaEventRecord(GPUstart, 0);

    ThrustSort(h_key_array, d_key_array, number_of_elements, batch_size, nthreads);

    cudaEventRecord(GPUstop, 0);
    cudaEventSynchronize(GPUstop);
    cudaEventElapsedTime(&GPU_milliseconds, GPUstart, GPUstop);

    /**************************/
    /* Merging batches on GPU */
    /**************************/

    struct timeval CPUstart;
    gettimeofday(&CPUstart, NULL);
    std::vector< std::pair<ulong2*, ulong2*> > batches;
    
    for (int i = 0; i < number_of_batches / 2; i++)
    {
        if (i == (number_of_batches / 2) - 1) {
            batches.push_back(std::make_pair(&h_key_array[2*i*batch_size], &h_key_array[2*i*batch_size+batch_size]));
            batches.push_back(std::make_pair(&h_key_array[2*i*batch_size+batch_size], &h_key_array[number_of_elements]));
            break;
        }
        batches.push_back(std::make_pair(&h_key_array[2*i*batch_size], &h_key_array[2*(i+1)*batch_size]));
    }
    
    omp_set_dynamic(false);
    omp_set_num_threads(nthreads);
    nvtxRangeId_t id0 = nvtxRangeStart("Multiway-merge");

    //printf("Real number of threads: %d\n", omp_get_num_threads());
    __gnu_parallel::multiway_merge(batches.begin(), batches.end(), sorted_array, number_of_elements, std::less<ulong2>());

    nvtxRangeEnd(id0);
    //double end = omp_get_wtime();
    struct timeval CPUend;
    gettimeofday(&CPUend, NULL);

    printf("Elapsed time on GPU: %f s.\n", (GPU_milliseconds/1000));
    printf("Elapsed time on CPU: %f s.\n", ((CPUend.tv_sec - CPUstart.tv_sec) * 1000000u + CPUend.tv_usec - CPUstart.tv_usec) / 1.e6 );

    printf("Test: %s\n", std::is_sorted(sorted_array, sorted_array+number_of_elements) == true ? "SUCCESS" : "FAIL");

    return 0;
}
