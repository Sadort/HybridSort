rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include -I../../include BLine_Thrust.cu -std=c++11
/usr/local/cuda/bin/nvcc  -o BLine_Thrust.out BLine_Thrust.o -I../../include -lnvidia-ml -lcudart -std=c++11

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -I../../include -O3 -fopenmp

rm -rf *.o
