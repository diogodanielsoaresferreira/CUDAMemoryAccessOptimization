////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tomás Oliveira e Silva,  November 2017
//
// ACA 2017/2018
//
// modify_sector CUDA kernel (each thread deals with one sector)
//

extern "C" __global__
void modify_sector_cuda_kernel(unsigned int * __restrict__ sector_data,unsigned int * __restrict__ sector_number,unsigned int n_sectors,unsigned int sector_size)
{
  unsigned int x,y,idx,i,a,c,n_words;
  unsigned int *lo,*hi;

  lo = sector_data;
  hi = sector_data + n_sectors * sector_size / 4;
  //
  // compute the thread number
  //
  x = (unsigned int)threadIdx.x + (unsigned int)blockDim.x * (unsigned int)blockIdx.x;
  y = (unsigned int)threadIdx.y + (unsigned int)blockDim.y * (unsigned int)blockIdx.y;
  idx = (unsigned int)blockDim.x * (unsigned int)gridDim.x * y + x;
  if(idx >= n_sectors)
    return; // safety precaution
  //
  // convert the sector size into number of 4-byte words (it is assumed that sizeof(unsigned int) = 4)
  //
  n_words = sector_size / 4u;
  //
  // adjust pointers (N.B. the memory layout may not be optimal)
  //
  sector_data += n_words * idx;
  sector_number += idx;
  //
  // initialize the linear congruencial pseudo-random number generator
  // (section 3.2.1.2 of The Art of Computer Programming presents the theory behind the restrictions on a and c)
  //
  i = sector_number[0];                       // get the sector number
  a = 0xACA00001u ^ ((i & 0x0F0F0F0Fu) << 2); // a must be a multiple of 4 plus 1
  c = 0x00ACA001u ^ ((i & 0xF0F0F0F0u) >> 3); // c must be odd
  x = 0xACA02017u;                            // initial state
  //
  // modify the sector data
  //
  for(i = 0u;i < n_words;i++)
  {
    unsigned int *addr;

    x = a * x + c;       // update the pseudo-random generator state
    addr = &sector_data[i];
    if(addr >= lo && addr < hi)
      *addr ^= x; // modify the sector data
  }
}
