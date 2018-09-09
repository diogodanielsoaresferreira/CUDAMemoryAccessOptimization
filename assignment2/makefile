#
# TOS, November 2017
#
# ACA 2017/2018
#
# the CUDA stuff (optional) is assumed to be installed in /usr/local/cuda
# the CUDA device is assumed to have architecture sm_20 (for an old GTX 480)
#

CUDA_ARCH = sm_20


clean:
	rm -f open_cl_hello cuda_hello cuda_hello.cubin crypt_cuda modify_sector_cuda_kernel.cubin


open_cl_hello:		open_cl_hello.c
	cc -Wall -O2 -std=c99 open_cl_hello.c -o open_cl_hello -L/usr/local/cuda/lib64 -lOpenCL


cuda_hello:		cuda_hello.c cuda_hello.cubin
	cc -Wall -O2 -std=c99 -I/usr/local/cuda/include cuda_hello.c -o cuda_hello -L/usr/local/cuda/lib64 -lcuda

cuda_hello.cubin:	cuda_hello.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I/usr/local/cuda/include --cubin cuda_hello.cu -o cuda_hello.cubin

crypt_cuda:		crypt_cuda.c modify_sector_cpu_kernel.c modify_sector_cuda_kernel.cubin
	cc -Wall -O2 -std=c99 -I/usr/local/cuda/include crypt_cuda.c -o crypt_cuda -L/usr/local/cuda/lib64 -lcuda

cryptCuda:		cryptCuda.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I/usr/local/cuda/include cryptCuda.cu -o cryptCuda

modify_sector_cuda_kernel.cubin:	modify_sector_cuda_kernel.cu
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I/usr/local/cuda/include --cubin modify_sector_cuda_kernel.cu -o modify_sector_cuda_kernel.cubin
