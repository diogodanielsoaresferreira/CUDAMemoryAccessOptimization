////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tomás Oliveira e Silva,  November 2017
//
// ACA 2017/2018
//
// modify_sector OpenCL kernel (each thread deals with one sector)
//

__kernel
void modify_sector_opencl_kernel(__global unsigned int *sector_data,__global unsigned int *sector_number,unsigned int n_sectors,unsigned int sector_size)
{
  unsigned int x,i,a,c,n_words;
  size_t idx;

  //
  // compute the thread number
  //
  idx = get_global_id(1) * get_global_size(0) + get_global_id(0);
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
    x = a * x + c;       // update the pseudo-random generator state
    sector_data[i] ^= x; // modify the sector data
  }
}
