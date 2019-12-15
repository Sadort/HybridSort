rm -rf *.o

#unified memory thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include BLine_Thrust.cu
/usr/local/cuda/bin/nvcc  -o BLine_Thrust.out BLine_Thrust.o 

rm -rf *.o
