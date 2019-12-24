rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include Thrust_kernel.cu
/usr/local/cuda/bin/nvcc -o BLineMulti_Thrust.out -Xcompiler -fopenmp -Xcompiler -O3 BLineMulti_Thrust.cpp Thrust_kernel.o -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lnvToolsExt

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o