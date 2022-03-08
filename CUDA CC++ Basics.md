# CUDA C/C++ Basics

[Souce link](https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)

## Cooperating Threads

Unlike parallel blocks, threads have mechanisms to efficiently: 

- Communicate
- Synchronize

### Sharing Data Between Threads

**shared memory**: shared data for threads within a block

- exetremely fast on-chip memory, by opposition to the global memory
- like a user-managed cache
- declared using `__shared__`, allocated per block
- data is not visible to threads in other blocks

**\__syncthreads()**: synchronizes all threads **within a block**

- prevents data hazards

## Managing The Device

### Coordinating Host & Device

Kernal launches are asynchronous

- contol returns to the CPU immediately

CPU needs to synchronize before consuming the results

- cudaMemcpy(): 
  - blocks the CPU until the copy is complete
  - **copy begins when all preceding CUDA calls have complete**
- cudaMemcpyAsync(): asynchronous, does not block the CPU
- cudaDeviceSynchronize(): blocks the CPU until all preceding CUDA calls have completed

### Reporting Errors

```c++
// Get the error code for the last error
cudaError_t cudaGetLastError(void);

// Get a string to describe the error
char *cudaGetErrorString(cudaError_t);
printf("%s\n", cudaGetErrorString(cudaGetLastError()));
```

### Device Management

**Application can query and select GPUs**

- cudaGetDeviceCount(int *count)
- cudaSetDevice(int device)
- cudaGetDevice(int *device)
- cudaGetDeviceProperties(cudaDeviceProp *prop, int device)

```c++
// Source: https://forums.developer.nvidia.com/t/beginner-cudagetdevicecount/16403
int device = 0; 
int gpuDeviceCount = 0; 
struct cudaDeviceProp properties; 

cudaError_t cudaResultCode = cudaGetDeviceCount(&gpuDeviceCount); 

if (cudaResultCode == cudaSuccess) 
{ 
	cudaGetDeviceProperties(&properties, device); 
	printf("%d GPU CUDA devices(s)(%d)\n", gpuDeviceCount, properties.major); 
	printf("\t Product Name: %s\n"		, properties.name);
	printf("\t TotalGlobalMem: %d MB\n"	, properties.totalGlobalMem/(1024^2));
	printf("\t GPU Count: %d\n"		, properties.multiProcessorCount);
	printf("\t Kernels found: %d\n"		, properties.concurrentKernels);
}
```

**Multiple host threads can share a device**

**A single host thread can manage multiple devices**

- cudaSetDevice(i): to select curent device
- cudaMemcpy(...): for peer-to-peer copies

