//
// Tomás Oliveira e Silva,  December 2016
//
// Hello world program in OpenCL
//
// What it does:
//   the OpenCL kernel initializes an array with the string "Hello, world!"
//
// Assuming a CUDA installation in /usr/local/cuda, compile with
//   cc -Wall -O2 open_cl_hello.c -o open_cl_hello -L/usr/local/cuda/lib64 -lOpenCL
// (AMD or Intel implementations of OpenCL will require a different -L option and possibly a -I option as well)
//

#include <stdio.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // for the clCreateCommandQueue function
#include <CL/cl.h>


//
// Macro that can be used to call an OpenCL function and to test its return value
// It can, and should, be used to test the return value of calls such as
//   e = clGetPlatformIDs(1,&platform_id[0],&num_platforms);
// In this case, f_name is clGetPlatformIDs and args is (1,&platform_id[0],&num_platforms)
//

#define cl_call(f_name,args)                                                                     \
  do                                                                                             \
  {                                                                                              \
    cl_int e = f_name args;                                                                      \
    if(e != CL_SUCCESS)                                                                          \
    { /* the call failed, terminate the program */                                               \
      fprintf(stderr,"" # f_name "() returned %s (line %d)\n",cl_error_string((int)e),__LINE__); \
      exit(1);                                                                                   \
    }                                                                                            \
  }                                                                                              \
  while(0)


//
// Another macro that can be used to call an OpenCL function and to test its return value
// It can, and should, be used the test the error code value of calls such as
//   context = clCreateContext(NULL,1,&device_id[0],NULL,NULL,&e);
// In this case, f_name is context = clCreateContext and args is (NULL,1,&device_id[0],NULL,NULL,&e)
//

#define cl_call_alt(f_name,args)                                                                 \
  do                                                                                             \
  {                                                                                              \
    cl_int e;                                                                                    \
    f_name args;                                                                                 \
    if(e != CL_SUCCESS)                                                                          \
    { /* the call failed, terminate the program */                                               \
      fprintf(stderr,"" # f_name "() returned %s (line %d)\n",cl_error_string((int)e),__LINE__); \
      exit(1);                                                                                   \
    }                                                                                            \
  }                                                                                              \
  while(0)


//
// "User-friendly" description of OpenCL error codes
//

static char *cl_error_string(int e)
{
  static const char *error_description[100] =
  { // warning: C99 array initialization feature
    [ 0] = "CL_SUCCESS",
    [ 1] = "CL_DEVICE_NOT_FOUND",
    [ 2] = "CL_DEVICE_NOT_AVAILABLE",
    [ 3] = "CL_COMPILER_NOT_AVAILABLE",
    [ 4] = "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    [ 5] = "CL_OUT_OF_RESOURCES",
    [ 6] = "CL_OUT_OF_HOST_MEMORY",
    [ 7] = "CL_PROFILING_INFO_NOT_AVAILABLE",
    [ 8] = "CL_MEM_COPY_OVERLAP",
    [ 9] = "CL_IMAGE_FORMAT_MISMATCH",
    [10] = "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    [11] = "CL_BUILD_PROGRAM_FAILURE",
    [12] = "CL_MAP_FAILURE",
    [13] = "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    [14] = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    [15] = "CL_COMPILE_PROGRAM_FAILURE",
    [16] = "CL_LINKER_NOT_AVAILABLE",
    [17] = "CL_LINK_PROGRAM_FAILURE",
    [18] = "CL_DEVICE_PARTITION_FAILED",
    [19] = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
    [30] = "CL_INVALID_VALUE",
    [31] = "CL_INVALID_DEVICE_TYPE",
    [32] = "CL_INVALID_PLATFORM",
    [33] = "CL_INVALID_DEVICE",
    [34] = "CL_INVALID_CONTEXT",
    [35] = "CL_INVALID_QUEUE_PROPERTIES",
    [36] = "CL_INVALID_COMMAND_QUEUE",
    [37] = "CL_INVALID_HOST_PTR",
    [38] = "CL_INVALID_MEM_OBJECT",
    [39] = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    [40] = "CL_INVALID_IMAGE_SIZE",
    [41] = "CL_INVALID_SAMPLER",
    [42] = "CL_INVALID_BINARY",
    [43] = "CL_INVALID_BUILD_OPTIONS",
    [44] = "CL_INVALID_PROGRAM",
    [45] = "CL_INVALID_PROGRAM_EXECUTABLE",
    [46] = "CL_INVALID_KERNEL_NAME",
    [47] = "CL_INVALID_KERNEL_DEFINITION",
    [48] = "CL_INVALID_KERNEL",
    [49] = "CL_INVALID_ARG_INDEX",
    [50] = "CL_INVALID_ARG_VALUE",
    [51] = "CL_INVALID_ARG_SIZE",
    [52] = "CL_INVALID_KERNEL_ARGS",
    [53] = "CL_INVALID_WORK_DIMENSION",
    [54] = "CL_INVALID_WORK_GROUP_SIZE",
    [55] = "CL_INVALID_WORK_ITEM_SIZE",
    [56] = "CL_INVALID_GLOBAL_OFFSET",
    [57] = "CL_INVALID_EVENT_WAIT_LIST",
    [58] = "CL_INVALID_EVENT",
    [59] = "CL_INVALID_OPERATION",
    [60] = "CL_INVALID_GL_OBJECT",
    [61] = "CL_INVALID_BUFFER_SIZE",
    [62] = "CL_INVALID_MIP_LEVEL",
    [63] = "CL_INVALID_GLOBAL_WORK_SIZE",
    [64] = "CL_INVALID_PROPERTY",
    [65] = "CL_INVALID_IMAGE_DESCRIPTOR",
    [66] = "CL_INVALID_COMPILER_OPTIONS",
    [67] = "CL_INVALID_LINKER_OPTIONS",
    [68] = "CL_INVALID_DEVICE_PARTITION_COUNT",
    [69] = "CL_INVALID_PIPE_SIZE",
    [70] = "CL_INVALID_DEVICE_QUEUE"
  };
  static char error_string[256];

  sprintf(error_string,"%d[%s]",e,(-e >= 0 && -e < 70 && error_description[-e] != NULL) ? error_description[-e] : "UNKNOWN");
  return &error_string[0];
}


//
// Main program
//

int main(void)
{
  int i;

  //
  // read the OpenCL kernel source code (this could be a string in our source code, but it is easier during code development to read it from a file)
  //
  char open_cl_source_code[8192];
  size_t open_cl_source_code_size;
  FILE *fp;

  fp = fopen("open_cl_hello.cl","r");
  if(fp == NULL)
  {
    perror("fopen()");
    exit(1);
  }
  open_cl_source_code_size = fread((void *)&open_cl_source_code[0],sizeof(char),sizeof(open_cl_source_code),fp);
  if(open_cl_source_code_size < (size_t)1 || open_cl_source_code_size >= sizeof(open_cl_source_code))
  {
    fprintf(stderr,"fread(): the OpenCL kernel code is either too small or too large\n");
    exit(1);
  }
  fclose(fp);

  //
  // get the first OpenCL platform ID
  //
  cl_uint num_platforms;
  cl_platform_id platform_id[1];

  cl_call( clGetPlatformIDs , (1,&platform_id[0],&num_platforms) );
  if(num_platforms < 1)
  {
    fprintf(stderr,"No OpenCL platform\n");
    exit(1);
  }
  if(num_platforms > 1)
    fprintf(stderr,"Warning: more than one OpenCL platform found (using the first one)\n");

  //
  // get information about the OpenCL platform (not truly needed, but this information may be useful)
  //
  char info_data[256];

  printf("OpenCL platform information\n");
  cl_call( clGetPlatformInfo , (platform_id[0],CL_PLATFORM_PROFILE,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  profile ................... %s\n",info_data);
  cl_call( clGetPlatformInfo , (platform_id[0],CL_PLATFORM_VERSION,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  version ................... %s\n",info_data);
  cl_call( clGetPlatformInfo , (platform_id[0],CL_PLATFORM_NAME,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  name ...................... %s\n",info_data);
  cl_call( clGetPlatformInfo , (platform_id[0],CL_PLATFORM_VENDOR,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  vendor .................... %s\n",info_data);
//cl_call( clGetPlatformInfo , (platform_id[0],CL_PLATFORM_EXTENSIONS,sizeof(info_data),(void *)&info_data[0],NULL) );
//printf("  extensions ................ %s\n",info_data);

  //
  // get information about the first OpenCL device (use CL_DEVICE_TYPE_CPU or CL_DEVICE_TYPE_GPU to force a specific device type)
  //
  cl_uint num_devices;
  cl_device_id device_id[1];

  cl_call( clGetDeviceIDs , (platform_id[0],CL_DEVICE_TYPE_DEFAULT,1,&device_id[0],&num_devices) );
  if(num_devices < 1)
  {
    fprintf(stderr,"No OpenCL device\n");
    exit(1);
  }
  if(num_devices > 1)
    fprintf(stderr,"Warning: more than one OpenCL device found (using the first one)\n");

  //
  // get information about the OpenCL device we have chosen (not truly needed, but this information is useful)
  //
  cl_device_type device_type;
  cl_uint n_compute_units,n_dimensions;
  size_t work_group_limits[4],local_index_limits[4]; // 4 is more than enough for current hardware
  size_t max_local_size;
  cl_ulong device_mem_size,device_global_cache_size,device_local_mem_size;

  printf("OpenCL device information\n");
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_NAME,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  name ...................... %s\n",info_data);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_VENDOR,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  vendor .................... %s\n",info_data);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DRIVER_VERSION,sizeof(info_data),(void *)&info_data[0],NULL) );
  printf("  driver version ............ %s\n",info_data);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_TYPE,sizeof(device_type),(void *)&device_type,NULL) );
  printf("  type ...................... 0x%08X\n",(int)device_type);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(n_compute_units),(void *)&n_compute_units,NULL) );
  printf("  number of compute units ... %u\n",(unsigned int)n_compute_units);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(n_dimensions),(void *)&n_dimensions,NULL) );
  printf("  number of indices ......... %u%s\n",(unsigned int)n_dimensions,((int)n_dimensions > 3) ? " (more than three!)" : "");
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(work_group_limits),(void *)&work_group_limits[0],NULL) );
  printf("  work group limits .........");
  for(i = 0;i < (int)n_dimensions;i++)
    printf(" %d",(int)work_group_limits[i]);
  printf(" # N.B. these are not the total limits!\n");
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(local_index_limits),(void *)&local_index_limits[0],NULL) );
  printf("  local index limits ........");
  for(i = 0;i < (int)n_dimensions;i++)
    printf(" %d",(int)local_index_limits[i]);
  printf("\n");
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_local_size),(void *)&max_local_size,NULL) );
  printf("  max local threads ......... %u\n",(unsigned int)max_local_size);
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(device_mem_size),(void *)&device_mem_size,NULL) );
  printf("  device memory ............. %.3fGiB\n",(double)device_mem_size / (double)(1 << 30));
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(device_global_cache_size),(void *)&device_global_cache_size,NULL) );
  printf("  cache memory .............. %.3fKiB\n",(double)device_global_cache_size / (double)(1 << 10));
  cl_call( clGetDeviceInfo , (device_id[0],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(device_local_mem_size),(void *)&device_local_mem_size,NULL) );
  printf("  local memory .............. %.3fKiB\n",(double)device_local_mem_size / (double)(1 << 10));

  //
  // create an OpenCL context
  //
  cl_context context;

  cl_call_alt( context = clCreateContext , (NULL,1,&device_id[0],NULL,NULL,&e) );

  //
  // create an OpenCL command queue
  //
  cl_command_queue command_queue;

//cl_call_alt( command_queue = clCreateCommandQueueWithProperties, (context,device_id[0],NULL,&e) );
  cl_call_alt( command_queue = clCreateCommandQueue, (context,device_id[0],0,&e) );

  //
  // create a memory area in device memory where the "Hello, world!" string will be placed
  //
  char host_buffer[128];
  cl_mem device_buffer;
  int buffer_size;

  buffer_size = (int)sizeof(host_buffer);
  cl_call_alt( device_buffer = clCreateBuffer, (context,CL_MEM_READ_WRITE,(size_t)buffer_size,NULL,&e) );

  //
  // transfer the OpenCL code to the OpenCL context
  //
  char *program_lines[1];
  size_t program_line_lengths[1];
  cl_program program;

  program_lines[0] = &open_cl_source_code[0];
  program_line_lengths[0] = open_cl_source_code_size;
  cl_call_alt( program = clCreateProgramWithSource, (context,1,(const char **)&program_lines[0],&program_line_lengths[0],&e) );

  //
  // compile the OpenCL code and get the hello() kernel handle
  //
  cl_kernel kernel;
  size_t simd_width,max_group_size,compiled_group_size[3];
  char build_log[1024];

  cl_call( clBuildProgram , (program,1,&device_id[0],NULL,NULL,NULL) );
  cl_call_alt( kernel = clCreateKernel , (program,"hello_kernel",&e) );
  printf("kernel information\n");
  cl_call( clGetKernelWorkGroupInfo , (kernel,device_id[0],CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(simd_width),(void *)&simd_width,NULL) );
  printf("  simd width ................ %d\n",(int)simd_width);
  cl_call( clGetKernelWorkGroupInfo , (kernel,device_id[0],CL_KERNEL_WORK_GROUP_SIZE,sizeof(max_group_size),(void *)&max_group_size,NULL) );
  printf("  max group size ............ %d\n",(int)max_group_size);
  cl_call( clGetKernelWorkGroupInfo , (kernel,device_id[0],CL_KERNEL_COMPILE_WORK_GROUP_SIZE,sizeof(compiled_group_size),(void *)&compiled_group_size[0],NULL) );
  printf("  compiled group size .......");
  for(i = 0;i < 3;i++)
    printf(" %d",(int)compiled_group_size[i]);
  printf("\n");
  cl_call( clGetProgramBuildInfo , (program,device_id[0],CL_PROGRAM_BUILD_LOG,sizeof(build_log),(void *)&build_log[0],NULL) );
  printf("  build log ................. %s\n",build_log);

  //
  // run the kernel (set its arguments iand launch grid first)
  //
  // we are launching sizeof(host_buffer) threads here; each thread initializes only one byte of the device_buffer array
  // as we will be launching a single kernel, there is no need to specify the events this kernel has to wait for
  //
  size_t total_work_size[1],local_work_size[1]; // number of threads
  cl_event hello_kernel_done[1];

  cl_call( clSetKernelArg , (kernel,0,sizeof(cl_mem),(void *)&device_buffer) );
  cl_call( clSetKernelArg , (kernel,1,sizeof(int),&buffer_size) );
  total_work_size[0] = (size_t)buffer_size; // the total number of threads (one dimension)
  local_work_size[0] = (size_t)buffer_size; // the number of threads in each work group (in this small example, all of them)
  cl_call( clEnqueueNDRangeKernel , (command_queue,kernel,1,NULL,&total_work_size[0],&local_work_size[0],0,NULL,&hello_kernel_done[0]) );

  //
  // copy the buffer form device memory to CPU memory (copy only after the kernel has finished and block host execution until the copy is completed)
  //
  cl_call( clEnqueueReadBuffer , (command_queue,device_buffer,CL_TRUE,0,(size_t)buffer_size,(void *)host_buffer,1,&hello_kernel_done[0],NULL) );
 
  //
  // display host_buffer
  //
  for(i = 0;i < buffer_size;i++)
    printf("%3d %02X %c\n",i,(int)host_buffer[i] & 0xFF,((int)host_buffer[i] >= 32 && (int)host_buffer[i] < 127) ? host_buffer[i] : '_');
 
  //
  // clean up (optional)
  //
  cl_call( clFlush , (command_queue) );
  cl_call( clFinish , (command_queue) );
  cl_call( clReleaseKernel , (kernel) );
  cl_call( clReleaseProgram , (program) );
  cl_call( clReleaseMemObject , (device_buffer) );
  cl_call( clReleaseCommandQueue , (command_queue) );
  cl_call( clReleaseContext , (context) );

  //
  // all done!
  //
  return 0;
}
