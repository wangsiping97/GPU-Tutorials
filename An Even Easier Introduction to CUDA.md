# An Even Easier Introduction to CUDA

[Source link](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

## CUDA Terminologies

**kernel**: a funtion that the GPU can run

**\__global__**: a specifier telling the CUDA C++ complier that this is a funtion that runs on the GPU and can be called from CPU code

**device code**: code that runs on the GPU

**host code**: code that runs on the CPU

## Memory Allocation in CUDA

### Unified Memory[^1]

Unified Mmeory creates a pool of managed memory that is shared between the CPU and GPU, briding the CPU-GPU divide. Managed memory is accessible to both the CPU and GPU using **a single pointer**. The key is that the system automatically **migrates** data allocated in Unified Memory between host and device so that it **looks like** CPU memory to code running on the CPU, and like GPU memory to code running on the GPU. 

```c++
char *data;
cudaMallocManaged(&data, N); // making the data pointer accessible from both the host and the device.
```

**Note**: this is a programming model to simplify CUDA codes. However, a carefully tuned CUDA program that uses streams and `cudaMemcpyAsync()` to efficiently overlap execution with data transfers **may very well perform better than only using Unified Memory**. 

Q: Are we using unified memory or traditional memory allocation techniques? 

## Execution Configuration

**execution configuration**: tells the CUDA runtime how many parallel threads to use for the launch on the GPU. 

**Streaming Multiprocessors**: each runs multiple concurrent thread blocks

**threadIdx.x / y / z**: the index of the current thread within its block

**blockIdx.x / y / z**: the index of the current thread block in the grid

**blockDim.x / y / z**: the number of threads in the block

**gridDim.x / y / z**: the number of blocks in the grid

### Grid-Stride Loop

```c++
__global__
void add (int n, float *x, float *y) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; // the thread index in the grid
	int stride = blockDim.x * gridDim.x; // number of threads in the grid
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}
```

[^1]: https://developer.nvidia.com/blog/unified-memory-in-cuda-6/