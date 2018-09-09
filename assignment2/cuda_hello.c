//
// Tomás Oliveira e Silva,  Noovember 2017
//
// Hello world program in CUDA (driver API)
//
// What it does:
//   the CUDA kernel initializes an array with the string "Hello, world!"
//
// Assuming a CUDA installation in /usr/local/cuda, compile with
//   cc -Wall -O2 -I/usr/local/cuda/include cuda_hello.c -o cuda_hello -L/usr/local/cuda/lib64 -lcuda
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


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


//
// Main program
//

int main(void)
{
  int i;

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
  // load a precompiled module (a module may have more than one kernel)
  //
  CUmodule cu_module;

  cu_call( cuModuleLoad , (&cu_module,"./cuda_hello.cubin") );

  //
  // create a memory area in device memory where the "Hello, world!" string will be placed
  //
  char host_buffer[128];
  CUdeviceptr device_buffer;
  int buffer_size;

  buffer_size = (int)sizeof(host_buffer);
  cu_call( cuMemAlloc , (&device_buffer,(size_t)buffer_size) );

  //
  // get the kernel function pointer
  //
  CUfunction cu_kernel;

  cu_call( cuModuleGetFunction, (&cu_kernel,cu_module,"hello_kernel") );

  //
  // run the kernel (set its arguments first)
  //
  // we are launching sizeof(host_buffer) threads here; each thread initializes only one byte of the device_buffer array
  //
  void *cu_params[2];

  cu_params[0] = &device_buffer;
  cu_params[1] = &buffer_size;
  cu_call( cuLaunchKernel , (cu_kernel,1,1,1,buffer_size,1,1,0,(CUstream)0,&cu_params[0],NULL) );
  cu_call( cuStreamSynchronize , (0) );

  //
  // copy the buffer form device memory to CPU memory (copy only after the kernel has finished and block host execution until the copy is completed)
  //
  cu_call( cuMemcpyDtoH , (&host_buffer,device_buffer,(size_t)buffer_size) );
 
  //
  // display host_buffer
  //
  for(i = 0;i < buffer_size;i++)
    printf("%3d %02X %c\n",i,(int)host_buffer[i] & 0xFF,((int)host_buffer[i] >= 32 && (int)host_buffer[i] < 127) ? host_buffer[i] : '_');
 
  //
  // clean up (optional)
  //
  cu_call( cuMemFree , (device_buffer) );
  cu_call( cuModuleUnload , (cu_module) );
  cu_call( cuCtxDestroy , (cu_context) );

  //
  // all done!
  //
  return 0;
}
