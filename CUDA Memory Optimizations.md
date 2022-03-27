# CUDA Memory Optimizations

[CUDA TOOLKIT DOCUMENTATION Chapter 9](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)

Maximize bandwidth => More fast memory, less slow-access memory

## 1 Data Transfer Between Host and Device

**Goals**: 

- Minimize data transfer between the host and the device -- even if have to run kernels on the GPU that do not demonstrate any speedup compared with running them on the CPU. 
- Intermediate data structures: created and operated on, and destroyed by the device.
- Batch small transters into one larger transfer -- even if have to pack non-contiguous regions of memory into a contiguous buffer and then unpack after the transfer. 
- Use page-locked (or pinned) memory. 

### Pinned Memory

> With paged memory, the specific memory, which is allowed to be paged in or paged out, is called *pageable memory*. Conversely, the specific memory, which is not allowed to be paged in or paged out, is called *page-locked memory* or *pinned memory*.
>
> Page-locked memory will not communicate with hard drive. Therefore, the efficiency of reading and writing in page-locked memory is more guaranteed.[^1]

[`cudaHostAlloc()` ](http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g15a3871f15f8c38f5b7190946845758c.html): Allocates host memory that is page-locked and accesible to the device. The driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as `cudaMemcpy()`. 

[`cudaHostRegister()`](http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g36b9fe28f547f28d23742e8c7cd18141.html): Page-locks a specified range of memory. 

**Note:** Allocating excessive pinned memory may degrade system performance, since it reduces the memory for paging. **Test the application and the systems it runs on for optimal performance parameters.** 

### Asynchronous and Overlapping Transfers with Computation

`cudaMemcpyAsync()`: A non-blocking variant of `cudaMemcpy()`. Requires pinned host memory.

Asynchronous tranfers enable overlap of data transfers by:

- Overlapping host computation with async data transfers and with device computations. 

- Overlapping kernel execution with async data transfer. On devices that are capable of concurrent copy and compute (see `asyncEngineCount`), the data transfer and kernel must use different, non-default streams (stream with non-zero IDs). 

  ```c++
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaMemcpy(a_d, a_h, ..., ..., stream1);
  kernel<<<grid, block, 0, stream2>>>(otherData_d);
  ```

  

### Zero Copy

This feature enables GPU threads to directly access host memory. It requires mapped pinned memory. 

**Note:** Mapped pinned host memory allows you to overlap CPU-GPU memory transfers with computation while avoiding the use of CUDA streams. But since any repeated access to such memory areas causes repeated CPU-GPU transfers, consider creating a second area in device memory to manually cache the previously read host memory data.

### Unified Virtual Addressing

With UVA, the host memory and the device memories of all installed supported devices share a single virtual address space. 

## 2 Device Memory Spaces

### Coalesced Access to Global Memory

### L2 Cache

### Shared Memory

### Local Memory

### Texture Memory

## 3 Allocation

## 4 NUMA Best Practices



[^1]: [Page-Locked Host Memory for Data Transfer](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/#:~:text=With%20paged%20memory%2C%20the%20specific,locked%20memory%20or%20pinned%20memory.)