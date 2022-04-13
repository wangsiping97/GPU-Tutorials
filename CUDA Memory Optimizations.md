# CUDA Memory Optimizations

[CUDA TOOLKIT DOCUMENTATION Chapter 9](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)

[How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) 

Maximize bandwidth => More fast memory, less slow-access memory

## 1 Data Transfer Between Host and Device

**Choices**: 

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

### Overlap Data Transfers with Computation on the Host

(See Nvidia's Technical Blog: [How to Overlay Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/))

#### (1) CUDA Streams

**Definition:** A sequence of operations that execute on the device in the order in which they are issued by the host code. All device operations (kernels and data transfers) run in a stream. 

- Default stream: Used when no stream is specified

  - Synchronizing stream => synchronize with operation in other streams => an operation begins after all previously issued operations *in any stream on the device* have completed; and completes before any other operation *in any stream on the device* will begin. (Exception: [CUDA 7](https://developer.nvidia.com/blog/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/))
  - Overlapping strategy: based on the **asynchronous behavior of kernel launches**

- Non-default stream: Explicitly declared, created, and destroyed by the host

  - Non-blocking stream => sometimes need to synchronize with the host code
    - `cudaDeviceSynchronize()`: blocks the host code until *all* previously issued operations on the device have completed
    - `cudaStreamSynchronize(stream)`: blocks the host thread until all previoulsy issued operations *in the specified stream* have completed
    - `cudaEventSynchronize(event)`: blocks the host thread until all previously issued operations *in the specified event* have completed

  - Overlapping strategy: based on **asynchronous data transfers**

#### (2) Overlapping Kernel Execution and Data Transfers

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

##### **Notice:** Different GPU architectures have different numbers of copy and kernel engines, which may differ in performance when using asynchronous transfers. 

### Zero Copy

This feature enables GPU threads to directly access host memory. It requires mapped pinned memory. 

**Note:** Mapped pinned host memory allows you to overlap CPU-GPU memory transfers with computation while avoiding the use of CUDA streams. But since any repeated access to such memory areas causes repeated CPU-GPU transfers, consider creating a second area in device memory to manually cache the previously read host memory data.

### Unified Virtual Addressing

With UVA, the host memory and the device memories of all installed supported devices share a single virtual address space. 

## 2 Device Memory Spaces

[Salient Features of Device Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces__salient-features-device-memory)

**Choices**: 

- Ensure global memory accesses are coalesced whenever possible.
- Non-unit-stride global memory accesses should be avoided whenever possible.
- Use shared memory to avoid redundant transfers from global memory.
- Use asynchronous copies from global to shared memory with an element of size 8 or 16 bytes.
- Use texture reads for streaming fetches with a constant latency.
- Use constant memory when threads in the same warp accesses only a few distinct locations. 

### Coalesced Access to Global Memory

**Ensure global memory accesses are coalesced whenever possible.** 

> Coalesced memory access of memory coalescing refers to combining multiple memory accesses into a single transaction. However, the following conditions may result in uncoalesced load (serialized memory accesses):[^2]
>
> - Memory (access) is not sequential
> - Memory access is sparse
> - Misaligned memory access
>
> Memory is accessed at 32 byte granularity.[^3]

The global access requirements for coalescing depend on the compute capability of the device. 

Coalescing concepts are illustrated in the following simple examples. Assume: 

- Compute capability 6.0 or higher
- Accesses are for 4-byte words, unless otherwise noted.

#### (1) A Simple Access Pattern

Sequential and aligned access: The k-th thread accesses the k-th word in a 32-byte aligned array. Not all threads need to participate. 

#### (2) A Sequential but Misaligned Access Pattern

Will require the original transactions to load the first `X` words, and another transaction to load the rest `32-X` words, where `X` is the offset of the misalignment. More transactions are required. 

When `X` is a multiple of 8, the global memory access bandwidth can be the same as the aligned accesses. 

**Cache line reuse increases throughput**: Adjacent warps reuse the cache lines their neighbors fetched. So the impact of misalignment is not as large as we might have expected. 

#### (3) Strided Accesses

**Ensure that as much as possible of the data in each cache line fetched is actually used is an important part of performance optimization of memory accesses.** Strided access results in low load/store efficiency since elements in the transaction are not fully used and represent wasted bandwidth. 

In this case, **non-unit-stride global memory accesses should be avoided whenever possible**. 

=> utilize shared memory

### L2 Cache

On-chip => higher bandwidth and lower latency accesses to global memory

A portion of the L2 cache can be set aside for persistent (repeatedly) accesses to a data region in global memory:  

```c++
cudaGetDeviceProperties(&prop, device_id);                
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for persisting accesses */ 
```

**Cache Access Window** -- `accessPolicyWindow` includes: 

- `base_ptr`: Global memory data pointer
- `num_bytes`: Number of bytes for persisting accesses. Must be less than the max window size. 
- `hitProp`: Type of access property on cache hit (persisting access)
- `missProp`: Type of access property on cache miss (streaming access)
- `hitRatio`: Percentage of lines assigned `hitProp`, rest are assigned `missProp` [^4]

Depending on the value of the `num_bytes` parameter and the size of L2 cache, one may need to tune the value of `hitRatio` to avlod thrashing of L2 cache lines. 

### Shared Memory

On-chip => higher bandwidth and lower latency than local and global memory

#### (1) Memory Banks

**Definition:** Shared memory is divided into equally sized memory modules (banks) for concurrent accesses. 

**Memory Bank Conflict:** Accessing addresses that are mapped to the same memory bank can only be done serially. 

Exception: **(Broadcast)** Multiple threads in a warp address the same shared memory location. In this case, multiple broadcasts from different banks are coalesced into a single multicast from the requested shared memory locations to the threads. 

**Addresses <-> Banks Mapping Strategy**: 

- On devices of compute capability 5.x or newer, each bank has a bandwidth of 32 bits every clock cycle, and successive 32-bit words are assigned to successive banks. 
- On devices of compute capability 3.x, each bank has a bandwidth of 64 bits every clock cycle. Either successive 32-bit words (in 32-bit mode) or successive 64-bit words (64-bit mode) are assigned to successive banks. 

#### (1) Matrix Multiplication (C=AB)

Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses by a warp in shared memory. 

**Use shared memory to avoid redundant transfers from global memory.**

#### (2) Matrix Multiplication (C=AAT)

**Analysis and eliminating bank conflicts:** Pad the shared memory array

```c++
__shared__ float transposedTile[TILE_DIM][TILE_DIM+1];
```

This padding eliminates the conflicts entirely, because now the stride between threads is w+1 banks (i.e., 33 for current devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a unit stride.

#### (4) Async Copy from Global Memory to Shared Memory

Overlapping copying data from global to shared memory with computation. 

The synchronous version for the kernel loads an element from global memory to an intermediate register and then stores the intermediate register value to shared memory. 

In the asynchronous version of the kernel, instructions to load from global memory and store directly into shared memory are issued as soon as `__pipeline_memcpy_async()` function is called. Using asynchronous copies does not use any intermediate register.

Best performance is achieved when using asynchronous copies with an element of size 8 or 16 bytes.

**Q: What is the role of intermediate registers?** 

### Local Memory

Off-chip => as expensive as access to global memory => **no faster access**

Usage: holding automatic variables 

- large structures or arrays that would consume too much register space
- arrays that the compiler determines may be indexed dynamically)

Inspection of the PTX assembly code (obtained by compiling with `-ptx` or `-keep` command-line options to `nvcc`) reveals whether a variable has been placed in local memory during the first compilation phases. 

### Texture Memory (Texture Cache)

Read-only => costs device memory read only on a cache miss

Optimized for 2D spatial locality => reading texture addresses that are closer will achieve best performance

Designed for **streaming fetches** with a constant latency

**Caveat: Within a kernel call, the texture cache is not kept coherent with respect to global memory write.** A thread can safely read a memory location via texture if the location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread within the same kernel call.

### Constant Memory (Constant Cache)

Size: 64KB

Costs device memory read only on a cache miss

Best when threads in the same warp accesses only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access.

## 3 Allocation

**Device memory should be reused and/or sub-allocated by the application whenever possible** to minimize the impact of allocations on overall performance. 

## Other References

[^1]: [Page-Locked Host Memory for Data Transfer](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/#:~:text=With%20paged%20memory%2C%20the%20specific,locked%20memory%20or%20pinned%20memory.)
[^2]: [Introduction to GPGPU and CUDA Programming: Memory Coalescing](https://cvw.cac.cornell.edu/gpu/coalesced#:~:text=Coalesced%20memory%20access%20or%20memory,threads%20in%20a%20single%20transaction.)
[^3]: [GPU Optimization Fundamentals](https://www.olcf.ornl.gov/wp-content/uploads/2013/02/GPU_Opt_Fund-CW1.pdf)
[^4]: [CUaccessPolicyWindow_v1 Struct Reference](https://docs.nvidia.com/cuda/cuda-driver-api/structCUaccessPolicyWindow__v1.html#structCUaccessPolicyWindow__v1_1d6ed5cd7bb416976b45e75bafce547e9)