# A Hybrid CPU/GPU Approach for Optimizing Sorting Throughput

- Implementation of a hybrid CPU/GPU sort as described in [paper by Gowanlock and Karsin](http://jan.ucc.nau.edu/mg2745/publications/Gowanlock_Karsin_ParCo2019.pdf)

## Unified Memory
CUDA programs in tests/UnifiedMemory are for Unified Memory [[1]](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)[[2]](https://devblogs.nvidia.com/beyond-gpu-memory-limits-unified-memory-pascal/) testing only. Programs are compiled by CUDA 10.0 and run in NVidia Tesla P100 (PCIe-Based 12GB Graphics 1189MHz Memory 715MHz).

- Thrust Library (out-of-place sorting) takes about 35 seconds (doesn't include time for initializing unified memory and CPU page table) to sort a large input dataset with 1 Billion 64-bit integers.

![Alt text](image/Thrust_UM_1G.png?raw=true "Image 1")

## Base Line
- BLine 

BLine performs sorting routine on GPU when dataset can fit into GPU on-chip memory.

- BLineMulti 

BLineMulti is the baseline approach when sorting multiple batches on the GPU and performs a single multiway merging on the CPU once all batches are sorted.

![Alt text](image/BLineMulti.png?raw=true "Image 2")

## Pinned Memory
- PipeData 

PipeData uses pinned memory and CUDA streams to pipeline the data transfers to/from the device to overlap transfers and utilize more bidirectional bandwidth over PCIe.

![Alt text](image/PipeData.png?raw=true "Image 3")

- PipeData_Ext

PipeData Extension is more compact than PipeData. 

![Alt text](image/PipeData_Ext.png?raw=true "Image 4")

- PipeMerge

PipeMerge extends PipeData by concurrently sorting on the GPU and merging on the CPU to reduce the overhead of the multiway merge at the end.

![Alt text](image/PipeMerge.png?raw=true "Image 5")
