# CUDA IPC

CUDA IPC (Inter-Process Communication) is a feature in NVIDIA's CUDA (Compute Unified Device Architecture) programming model, allowing data sharing and synchronization between different processes running on the same GPU. It's helpful when you have multiple applications that need to share GPU resources, without copying data between host and device memory.

## Terminologies

### Cuda Context

A CUDA context is like a container that holds the GPU resources for a specific process. It includes device memory, loaded modules, and other resources needed for executing your GPU kernels (the code that runs on the GPU). Each process has its own context, which isolates its resources from other processes.

Imagine the context as a "workspace" for each GPU application, keeping its data and settings separate from other applications.

The CUDA context is implemented as an opaque data structure in the CUDA runtime, which is managed by the CUDA driver. When a process initializes the CUDA runtime (usually by calling `cudaSetDevice()` or similar functions), a context is created for that process. The context is associated with the chosen GPU device and provides a separate environment for each process that runs on the GPU.

The context ensures resource isolation between different processes, preventing them from interfering with each other's memory, kernels, and other resources. The context also maintains the memory allocation state and handles memory operations such as allocation, deallocation, and data transfers between the host and the device memory.

### Memory Handle

A MemHandle, or memory handle, is a lightweight reference to a piece of GPU memory that can be shared between processes. It allows different processes to access the same device memory without copying data between host and device.

Think of the MemHandle as a "ticket" that grants access to a specific memory location on the GPU. You can share this ticket with other processes, allowing them to use the same memory.

Example: 

Process A has a piece of data in its GPU memory that it wants to share with Process B. Process A creates a MemHandle for that memory and shares it with Process B. Now, both processes can access the same data on the GPU without copying it back and forth.

### CudaEvent

A CUDA event is a synchronization primitive used to track the progress of various operations in the GPU. It can be used to measure the time taken by a specific operation, or to coordinate the execution of multiple tasks.

Imagine events as "milestones" in your GPU code. When a certain task reaches a milestone, it records an event. Other tasks can then wait for these events to occur, ensuring proper execution order.

### CUDA IPC Event Handle

An EventHandle is similar to a MemHandle, but for events instead of memory. It's a reference to a CUDA event that can be shared between processes to synchronize their execution.

Consider the EventHandle as a "ticket" for a specific event, like the one we described for MemHandles. By sharing this ticket, multiple processes can coordinate their execution based on the same event.

Example: 

Process A and Process B both need to perform calculations on the shared memory (accessed using a MemHandle). Process A needs to finish its calculations before Process B can start. To ensure this, Process A records a CUDA event when it completes its task, and shares the EventHandle with Process B. Process B then waits for the event to be recorded before starting its calculations.

## APIs

### cudaIpcGetMemHandle

`cudaIpcGetMemHandle()` is a function in the CUDA API that allows you to create a memory handle (IPC handle) for a specific GPU memory allocation. This handle can be shared with other processes, enabling them to access the same device memory without copying data.

#### Function signature

```c++
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
```

#### Parameters

1. `handle`: A pointer to a `cudaIpcMemHandle_t` structure that will store the memory handle upon successful completion of the function. This handle can be shared with other processes to access the device memory.
2. `devPtr`: A pointer to the device memory for which you want to create the memory handle. This memory must have been allocated using `cudaMalloc()` or a similar function **within the same process**.

#### Return type

`cudaError_t`: An enumerated type that indicates the success or failure of the function. A value of `cudaSuccess` (0) means the function succeeded, while any other value indicates an error.

#### Possible failure cases

1. `cudaErrorInvalidDevicePointer`: If `devPtr` is not a valid device memory pointer, this error is returned.

   **Example:** If you accidentally pass a host pointer or an uninitialized pointer to the function, you may get this error.

   ```c++
   int *h_data = (int *) malloc(sizeof(int) * 10);
   cudaIpcMemHandle_t handle;
   // Incorrectly passing a host pointer instead of a device pointer
   cudaError_t err = cudaIpcGetMemHandle(&handle, h_data);
   ```

2. `cudaErrorMemoryAllocation`: If there's a failure in allocating the memory handle, this error is returned. It typically occurs when the system is under high memory pressure or if there is a bug in the driver.

3. `cudaErrorIpcMemoryHandleNotValid`: Occurs when the memory handle for the device pointer cannot be created or is not valid. This error may occur if the device memory pointer is not aligned properly or if there is an issue with the CUDA driver.

   ```c++
   int *d_A;
   cudaMalloc((void **)&d_A, 10 * sizeof(int) + 1); // Allocating misaligned memory
   cudaIpcMemHandle_t handle;
   cudaError_t err = cudaIpcGetMemHandle(&handle, d_A); // d_A is misaligned, will result in cudaErrorIpcMemoryHandleNotValid
   ```

### cudaIpcOpenMemHandle

`cudaIpcOpenMemHandle` is a function used in CUDA IPC (Inter-Process Communication) to open a remote memory handle for access by the calling process. This allows different processes to share GPU memory without copying data between host and device.

### cudaEventCreate

### cudaIpcGetEventHandle

### cudaIpcOpenEventHandle

