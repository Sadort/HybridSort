rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc -Xcompiler -fopenmp -c -I/usr/local/cuda/include -I../../include ParMemcpy.cpp -std=c++11
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include -I../../include Thrust_kernel.cu -std=c++11
/usr/local/cuda/bin/nvcc -o PipeData_Thrust.out -Xcompiler -fopenmp -Xcompiler -O3 PipeData_Thrust.cpp Thrust_kernel.o ParMemcpy.o -I../../include -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lnvToolsExt -std=c++11

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o
