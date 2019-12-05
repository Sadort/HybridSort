rm -rf *.o

#unified memory thrust::sort
/usr/local/cuda/bin/nvcc  -c -I/usr/local/cuda/include Thrust_um.cu
/usr/local/cuda/bin/nvcc  -o Thrust_um.out Thrust_um.o 

rm -rf *.o
