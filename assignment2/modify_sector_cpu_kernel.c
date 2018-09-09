////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tomás Oliveira e Silva,  November 2017
//
// ACA 2017/2018
//
// modify_sector cpu kernel (deals with one sector)
//

void modify_sector_cpu_kernel(unsigned int *sector_data,unsigned int sector_number,unsigned int sector_size)
{
  unsigned int x,i,a,c,n_words;

  //
  // convert the sector size into number of 4-byte words (it is assumed that sizeof(unsigned int) = 4)
  //
  n_words = sector_size / 4u;
  //
  // initialize the linear congruencial pseudo-random number generator
  // (section 3.2.1.2 of The Art of Computer Programming presents the theory behind the restrictions on a and c)
  //
  i = sector_number;                          // get the sector number
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
