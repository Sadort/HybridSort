rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc -Xcompiler -fopenmp -c -I/usr/local/cuda/include -I../../include Merge_kernel.cpp -std=c++11
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include Thrust_kernel.cu -std=c++11
/usr/local/cuda/bin/nvcc -o PipeMerge_Thrust.out -Xcompiler -fopenmp -Xcompiler -O3 PipeMerge_Thrust.cpp Thrust_kernel.o Merge_kernel.o -I../../include -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lnvToolsExt -std=c++11

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o
