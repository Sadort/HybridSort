rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include BLine_Thrust.cu
/usr/local/cuda/bin/nvcc  -o BLine_Thrust.out BLine_Thrust.o -lnvidia-ml -lcudart

#base line bitonic sort
/usr/local/cuda/bin/nvcc  -o BLine_Bitonic.out BLine_Bitonic.cu -I/usr/local/cuda/include -I../../include -I../../ext -lnvidia-ml -lcudart

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o
