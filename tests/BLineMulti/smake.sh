rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include Thrust_kernel.cu -lnvidia-ml -lcudart -lcuda
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include BLineMulti_Thrust.cpp
/usr/local/cuda/bin/nvcc -o BLineMulti_Thrust.out -Xcompiler -fopenmp -O3 BLineMulti_Thrust.o Thrust_kernel.o -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o
