#include "inplace-gpusort/bitonic.hxx"

template<typename T,fptr_t f>
void BitonicSort(T* data, int N, int BLOCKS, int THREADS, cudaStream_t stream=0) {

  int baseBlocks=((N/M)/(THREADS/W));
  int roundDist=1;
  int subDist=1;

// baseBlocks = 16384;
// Sort the base case into blocks of 1024 elements each
  squareSort<T,f><<<baseBlocks,32,0,stream>>>(data, N);

  int levels = (int)log2((float)(N/M)+1)+1;
    
  for(int i=1; i<levels; i++) {
      
    swapAllRev<T,f><<<BLOCKS,THREADS,0,stream>>>(data,N,roundDist);
//    swapAllRevRegs<T,f><<<BLOCKS,THREADS>>>(data,N,roundDist);
    subDist = roundDist/2;
    for(int j=i-1; j>0; j--) {
//      swapAllBlock<T,f><<<BLOCKS,THREADS>>>(data,N,subDist);
      swapAll<T,f><<<BLOCKS,THREADS,0,stream>>>(data,N,subDist);
      subDist /=2;
    }

//    squareSort<T,f><<<BLOCKS,32>>>(data, N);
//    cudaDeviceSynchronize();
    roundDist *=2;
  }

}

