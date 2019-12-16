rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include BLine_Thrust.cu
/usr/local/cuda/bin/nvcc  -o BLine_Thrust.out BLine_Thrust.o 

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -fopenmp

rm -rf *.o
