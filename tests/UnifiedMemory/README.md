CUDA programs in this directory are for Unified Memory [[1]](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)[[2]](https://devblogs.nvidia.com/beyond-gpu-memory-limits-unified-memory-pascal/) testing only. Programs are compiled by CUDA 10.0 and run in NVidia Tesla P100 (PCIe-Based 12GB Graphics 1189MHz Memory 715MHz).

1. Thrust Library (out-of-place sorting) takes about 35 seconds (doesn't include time for initializing unified memory and CPU page table) to sort a large input dataset with 1 Billion 64-bit integers.

![Alt text](image/Thrust_UM_1G.png?raw=true "Image 1")

