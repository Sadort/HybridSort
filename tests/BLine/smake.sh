rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include BLine_Thrust.cu
/usr/local/cuda/bin/nvcc  -o BLine_Thrust.out BLine_Thrust.o 

#cpu openmp gnu parallel
g++ test_openmp.cpp -o test_openmp.out -fopenmp

rm -rf *.o
