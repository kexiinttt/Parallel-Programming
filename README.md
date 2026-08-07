# Introduction

Personal notes and experiments from the University of Illinois Urbana-Champaign [ECE 408: Parallel Programming](https://ece.illinois.edu/academics/courses/ece408).

The goal of this repository is to build a practical understanding of how parallel programs are designed, implemented, and measured—especially on modern CPUs and GPUs.

This repository covers following topics:
- **CUDA and C/C++** — kernels, memory management, thread organization, and GPU execution
- **Parallel programming** — decomposition, synchronization, communication, and scalability
- **Computer architecture** — the hardware features that shape CPU and GPU performance
- **Performance analysis** — profiling, benchmarking, bottleneck analysis, and optimization

# Environment Setup

I use a Mac, which does not have an NVIDIA GPU or support running CUDA locally. I therefore run the CUDA programs in [Google Colab](https://colab.research.google.com/) using a hosted NVIDIA GPU.

1. Create a new Colab notebook and select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Clone this repository in a code cell:
   ```bash
   !git clone https://github.com/kexiinttt/Parallel-Programming.git
   %cd Parallel-Programming
   ```
3. Confirm that the GPU and CUDA compiler are available:
   ```bash
   !nvidia-smi
   !nvcc --version
   ```
4. Compile a CUDA source file with `nvcc`, then run the resulting executable.
   ```bash
   %cd projects/xyz
   !nvcc solution.cu -o solution
   !./solution
   ```
   