rm -rf *.o

#base line thrust::sort
/usr/local/cuda/bin/nvcc -Xcompiler -fopenmp -c -I/usr/local/cuda/include ParMemcpy.cpp -std=c++11
/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include Thrust_kernel.cu -std=c++11
/usr/local/cuda/bin/nvcc -o PipeData_Thrust.out -Xcompiler -fopenmp -Xcompiler -O3 PipeData_Thrust.cpp Thrust_kernel.o ParMemcpy.o -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lnvToolsExt -std=c++11

#base line bitonic sort
/usr/local/cuda/bin/nvcc -c -I../../ext/inplace-gpusort -I/usr/local/cuda/include Bitonic_kernel.cu -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg -lnvidia-ml -lcudart -D_FORCE_INLINES -std=c++11
/usr/local/cuda/bin/nvcc  -o PipeData_Bitonic.out -Xcompiler -fopenmp -Xcompiler -O3 PipeData_Bitonic.cpp Bitonic_kernel.o ParMemcpy.o -I../../ext/inplace-gpusort -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg -lnvidia-ml -lcudart -D_FORCE_INLINES -lnvToolsExt -std=c++11

#cpu openmp gnu parallel
#g++ BLine_CPU.cpp -o BLine_CPU.out -O3 -fopenmp

rm -rf *.o
