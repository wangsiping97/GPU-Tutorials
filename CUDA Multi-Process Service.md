# CUDA Multi-Process Service

[Improving GPU Utilization with MPS](https://on-demand.gputechconf.com/gtc/2015/presentation/S5584-Priyanka-Sah.pdf)

Used to optimize the performance in multi CUDA processes scenario. 

## Background

### CUDA Context [^1]

- Global memory allocated by the CPU
- Stack/Heap space (local memory) from the kernel
- CUDA streams & events objects
- Code module (*.cubin, *.ptx)

Each process has its own CUDA context

Each context has its own memory space, and cannot access other CUDA contexts' spaces 

### Hyper-Q [^2]-- Hyper Queue (Hardware Property)

Hyper-Q enables multiple threads or processes to launch work on a single GPU simultaneously. 

- Increases GPU utilization and reduces CPU idle times
- Eliminates false dependencies across tasks 

Before Hyper-Q: Fermi's single pipeline. There is only one hardware work queue so there can be false dependencies across the tasks. 

Kepler GK110 introduces the Grid Management Unit, which creates multiple hardware work queues to reduce or eliminate false dependencies. (Also the feedback path from SMXs to the GMU provides dynamic parallelism.)

Kelper allows 32-way concurrency. 

## Multi-Process Service (MPS)

A feature that allows multiple CUDA processes (contexts) to share a single GPU context. Each process receive some subset of the available connections to that GPU. 

MPS allows overlapping of kernel and memcopy operations *from different processes* on the GPU to achieve maximum utilization.

MPS Server: Hyper-Q/MPI

- All MPS Client Processes started after starting MPS Server will communicate through MPS Server only

- Many-to-one context mapping
- Allows multiple CUDA processes to share a/multiple GPU context(s)

## Usage

See the slides

## Summary

Best for GPU acceleration for legacy applications

Enables overlapping of memory copies and compute between different MPI ranks

Ideal for applications with

- MPI-everywhere
- Non-negligible CPU work
- Partially migrated to GPU

## Other References

[^1]: [如何使用MPS提升GPU计算收益](https://www.nvidia.cn/content/dam/en-zz/zh_cn/assets/webinars/31oct2019c/20191031_MPS_davidwu.pdf)
[^2]: [Hyper-Q Example](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/simpleHyperQ/doc/HyperQ.pdf)