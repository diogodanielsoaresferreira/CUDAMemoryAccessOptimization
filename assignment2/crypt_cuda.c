////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tomás Oliveira e Silva, November 2017
//
// ACA 2017/2018
//
// Reference implementation
//
// To attain a grade up to 14: optimize the launch grid
// To attain a grade up to 17: optimize the memory layout and the launch grid
// In both cases, answewr the following question: "Is offloading the computation to a CUDA or OpenCL device worthwhile?"
//


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// program configuration
//

#ifndef SECTOR_SIZE
# define SECTOR_SIZE  512
#endif
#ifndef N_SECTORS
# define N_SECTORS    (1 << 21)  // can go as high as (1 << 21)
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// includes
//

#define _GNU_SOURCE
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "modify_sector_cpu_kernel.c"  // for such a small program there is no need to compile the code in this file separately


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Measure real elapsed time
//

static double get_delta_time(void)
{
  static struct timespec t0,t1;

  t0 = t1;
  if(clock_gettime(CLOCK_MONOTONIC,&t1) != 0)
  {
    perror("clock_gettime");
    exit(1);
  }
  return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA support code
//

//
// Macro that can be used to call a CUDA driver API function and to test its return value
// It can, and should, be used to test the return value of calls such as
//   cuInit(device_number);
//   cuDeviceGet(&cu_device,device_number);
// In these cases, f_name is, respectively, cuInit and cuDeviceGet, and args is, respectively,
//   (device_number) and (&cu_device,device_number)
//

#define cu_call(f_name,args)                                                                \
  do                                                                                        \
  {                                                                                         \
    CUresult e = f_name args;                                                               \
    if(e != CUDA_SUCCESS)                                                                   \
    {                                                                                       \
      fprintf(stderr,"" # f_name "() returned %s (line %d)\n",cu_error_string(e),__LINE__); \
      exit(1);                                                                              \
    }                                                                                       \
  }                                                                                         \
  while(0)

//
// "User-friendly" description of CUDA error codes
//

static char *cu_error_string(int e)
{
  static const char *error_description[1000] =
  { // warning: C99 array initialization feature
    [  0] = "CUDA_SUCCESS",
    [  1] = "CUDA_ERROR_INVALID_VALUE",
    [  2] = "CUDA_ERROR_OUT_OF_MEMORY",
    [  3] = "CUDA_ERROR_NOT_INITIALIZED",
    [  4] = "CUDA_ERROR_DEINITIALIZED",
    [  5] = "CUDA_ERROR_PROFILER_DISABLED",
    [  6] = "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
    [  7] = "CUDA_ERROR_PROFILER_ALREADY_STARTED",
    [  8] = "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
    [100] = "CUDA_ERROR_NO_DEVICE",
    [101] = "CUDA_ERROR_INVALID_DEVICE",
    [200] = "CUDA_ERROR_INVALID_IMAGE",
    [201] = "CUDA_ERROR_INVALID_CONTEXT",
    [202] = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
    [205] = "CUDA_ERROR_MAP_FAILED",
    [206] = "CUDA_ERROR_UNMAP_FAILED",
    [207] = "CUDA_ERROR_ARRAY_IS_MAPPED",
    [208] = "CUDA_ERROR_ALREADY_MAPPED",
    [209] = "CUDA_ERROR_NO_BINARY_FOR_GPU",
    [210] = "CUDA_ERROR_ALREADY_ACQUIRED",
    [211] = "CUDA_ERROR_NOT_MAPPED",
    [212] = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
    [213] = "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
    [214] = "CUDA_ERROR_ECC_UNCORRECTABLE",
    [215] = "CUDA_ERROR_UNSUPPORTED_LIMIT",
    [216] = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
    [217] = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
    [218] = "CUDA_ERROR_INVALID_PTX",
    [219] = "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
    [220] = "CUDA_ERROR_NVLINK_UNCORRECTABLE",
    [300] = "CUDA_ERROR_INVALID_SOURCE",
    [301] = "CUDA_ERROR_FILE_NOT_FOUND",
    [302] = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
    [303] = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
    [304] = "CUDA_ERROR_OPERATING_SYSTEM",
    [400] = "CUDA_ERROR_INVALID_HANDLE",
    [500] = "CUDA_ERROR_NOT_FOUND",
    [600] = "CUDA_ERROR_NOT_READY",
    [700] = "CUDA_ERROR_ILLEGAL_ADDRESS",
    [701] = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    [702] = "CUDA_ERROR_LAUNCH_TIMEOUT",
    [703] = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
    [704] = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
    [705] = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
    [708] = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
    [709] = "CUDA_ERROR_CONTEXT_IS_DESTROYED",
    [710] = "CUDA_ERROR_ASSERT",
    [711] = "CUDA_ERROR_TOO_MANY_PEERS",
    [712] = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
    [713] = "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
    [714] = "CUDA_ERROR_HARDWARE_STACK_ERROR",
    [715] = "CUDA_ERROR_ILLEGAL_INSTRUCTION",
    [716] = "CUDA_ERROR_MISALIGNED_ADDRESS",
    [717] = "CUDA_ERROR_INVALID_ADDRESS_SPACE",
    [718] = "CUDA_ERROR_INVALID_PC",
    [719] = "CUDA_ERROR_LAUNCH_FAILED",
    [800] = "CUDA_ERROR_NOT_PERMITTED",
    [801] = "CUDA_ERROR_NOT_SUPPORTED",
    [999] = "CUDA_ERROR_UNKNOWN"
  };
  static char error_string[256];

  sprintf(error_string,"%d[%s]",e,(e >= 0 && e < 1000 && error_description[e] != NULL) ? error_description[e] : "UNKNOWN");
  return &error_string[0];
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main program
//

int main(void)
{
  if(sizeof(unsigned int) != (size_t)4)
    return 1; // fail with prejudice if an integer does not have 4 bytes

  //
  // open the first CUDA device
  //
  int device_number;
  CUdevice cu_device;

  device_number = 0;
  cu_call( cuInit , (device_number) );
  cu_call( cuDeviceGet , (&cu_device,device_number) );

  //
  // get information about the first CUDA device
  //
  char device_name[256];

  cu_call( cuDeviceGetName , (device_name,(int)sizeof(device_name) - 1,cu_device) );
  printf("CUDA code running on a %s (device %d)\n\n",device_name,device_number);

  //
  // create a context
  //
  CUcontext cu_context;

  cu_call( cuCtxCreate , (&cu_context,CU_CTX_SCHED_YIELD,cu_device) ); // CU_CTX_SCHED_SPIN may be slightly faster
  cu_call( cuCtxSetCacheConfig , (CU_FUNC_CACHE_PREFER_L1) );
 
  //
  // load the precompiled module (a module may have more than one kernel)
  //
  CUmodule cu_module;

  cu_call( cuModuleLoad , (&cu_module,"./modify_sector_cuda_kernel.cubin") );

  //
  // get the kernel function pointer
  //
  CUfunction cu_kernel;

  cu_call( cuModuleGetFunction, (&cu_kernel,cu_module,"modify_sector_cuda_kernel") );
 
  //
  // create memory areas in host and device memory where the disk sectors data and sector numbers will be placed
  //
  size_t sector_data_size;
  size_t sector_number_size;
  unsigned int *host_sector_data,*host_sector_number,*modified_host_sector_data;
  CUdeviceptr device_sector_data,device_sector_number;

  sector_data_size = (size_t)N_SECTORS * (size_t)SECTOR_SIZE;
  sector_number_size = (size_t)N_SECTORS * sizeof(unsigned int);
  if(sector_data_size + sector_number_size > (size_t)1.3e9)
  {
    fprintf(stderr,"The GTX 480 cannot handle more than 1.5GiB of memory!\n");
    exit(1);
  }
  cu_call( cuMemAllocHost , ((void **)&host_sector_data,sector_data_size) );
  //cu_call( cuMemHostAlloc , ((void **)&host_sector_data,sector_data_size,CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED) );
  cu_call( cuMemAlloc , (&device_sector_data,sector_data_size) );
  cu_call( cuMemAllocHost , ((void **)&host_sector_number,sector_number_size) );
  //cu_call( cuMemHostAlloc , ((void **)&host_sector_number,sector_number_size,CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED) );
  cu_call( cuMemAlloc , (&device_sector_number,sector_number_size) );
  //cu_call( cuMemHostAlloc , ((void **)&modified_host_sector_data,sector_data_size,CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED) );
  cu_call( cuMemAllocHost , ((void **)&modified_host_sector_data,sector_data_size) );

  //
  // initialize the host data with random numbers
  //
  (void)get_delta_time();
  srand(0xACA2017);
  for(int i = 0;i < (int)(sector_data_size / (int)sizeof(unsigned int));i++)
    host_sector_data[i] = 108584447u * (unsigned int)i; // "pseudo-random" data (faster than using the rand() function)
  for(int i = 0;i < (int)(sector_number_size / (int)sizeof(unsigned int));i++)
    host_sector_number[i] = (rand() & 0xFFFF) | ((rand() & 0xFFFF) << 16);
  printf("The initialization of host data took %.3e seconds\n",get_delta_time());

  //
  // copy the host data to device memory
  //
  cu_call( cuStreamSynchronize , (0) );
  (void)get_delta_time();
  cu_call( cuMemcpyHtoD , (device_sector_data,host_sector_data,sector_data_size) );
  cu_call( cuMemcpyHtoD , (device_sector_number,host_sector_number,sector_number_size) );
  cu_call( cuStreamSynchronize , (0) );
  printf("The transfer of %ld bytes from the host to the device took %.3e seconds\n",(long)sector_data_size + (long)sector_number_size,get_delta_time());
 
  //
  // run the kernel (set its arguments first)
  //
  // we are launching N_SECTORS threads here; each thread deals with one sector
  //
  void *cu_params[4];
  unsigned int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;
  int n_sectors;
  int sector_size;

  n_sectors = N_SECTORS;
  sector_size = SECTOR_SIZE;
  cu_params[0] = &device_sector_data;
  cu_params[1] = &device_sector_number;
  cu_params[2] = &n_sectors;
  cu_params[3] = &sector_size;
  gridDimX = 256;                     // optimize!
  gridDimY = (N_SECTORS + 255) / 256; // optimize!
  gridDimZ = 1;                       // not used in the CUDA kernel: do not change!
  blockDimX = 1;                      // optimize!
  blockDimY = 1;                      // optimize!
  blockDimZ = 1;                      // not used in the CUDA kernel: do not change!
  (void)get_delta_time();
  cu_call( cuLaunchKernel , (cu_kernel,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,0,(CUstream)0,&cu_params[0],NULL) );
  cu_call( cuStreamSynchronize , (0) );
  printf("The CUDA kernel took %.3e seconds to run\n",get_delta_time());

  //
  // copy the buffer form device memory to CPU memory
  //
  cu_call( cuStreamSynchronize , (0) );
  (void)get_delta_time();
  cu_call( cuMemcpyDtoH , (modified_host_sector_data,device_sector_data,(size_t)sector_data_size) );
  printf("The transfer of %ld bytes from the device to the host took %.3e seconds\n",(long)sector_data_size,get_delta_time());
 
  //
  // compute the modified sector data on the CPU
  //
  (void)get_delta_time();
  for(int i = 0;i < N_SECTORS;i++)
    modify_sector_cpu_kernel(&host_sector_data[i * (SECTOR_SIZE / (int)sizeof(unsigned int))],host_sector_number[i],SECTOR_SIZE);
  printf("The cpu kernel took %.3e seconds to run (single core)\n",get_delta_time());

  //
  // compare
  //
  for(int i = 0;i < sector_data_size / (int)sizeof(unsigned int);i++)
    if(host_sector_data[i] != modified_host_sector_data[i])
    {
      int sector_words = sector_size / (int)sizeof(unsigned int);
      printf("mismatch in sector %d, word %d\n",i / sector_words,i % sector_words);
      exit(1);
    }
  printf("All is well!\n");

  //
  // clean up
  //
  cu_call( cuMemFree , (device_sector_data) );
  cu_call( cuMemFree , (device_sector_number) );
  cu_call( cuMemFreeHost , (host_sector_data) );
  cu_call( cuMemFreeHost , (host_sector_number) );
  cu_call( cuMemFreeHost , (modified_host_sector_data) );
  cu_call( cuModuleUnload , (cu_module) );
  cu_call( cuCtxDestroy , (cu_context) );

  //
  // all done!
  //
  return 0;
}
