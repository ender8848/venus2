# gemmc

gemm C API for sound floating-point number calculation

# Prerequisite

|required|version|
|--------|-------|
|Linux kernel|>=5.15 (tested on Ubuntu 22.04)|
|GPU hardware|nvidia GPU|
|CUDA toolkit|compatible with GPU architecture|
|CUDA driver|compatible with CUDA toolkit|

The compatibility matters and below is a brief summary of environment setup. Example below uses device GTX GeForce 1060. 

## 1 check GPU opcode class and compute compatibility. 

It can be seen from the this [link](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/) that GTX 1060 belongs to Pascal architecture. 

Then we search from the web to know that **opcode class** of Pascal architecture is **SIMT**. 

After that, we can see from this [link](https://developer.nvidia.com/cuda-gpus#compute) that the **compute compatibility** of GeForce GTX 1060 is **6.1**. 

## 2 check compatible cudatoolkit version

use the above opcode class and compute compatibility info, we can find in [cutlass documentation](https://github.com/NVIDIA/cutlass/blob/master/media/docs/functionality.md#device-level-gemm) that the minimum compatible cudatoolkit version is **9.2+**.

## 3 check CUDA driver version 

Check CUDA driver version in system using the following command

```
nvidia-smi
```

If a driver is not installed, you can jump to step 4 as cudatoolkit installation includes driver installation. If the version is not compatible with cudatoolkit version, simply uninstall the current driver. 

## 4 install cudatoolkit

[Cuda toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) contains cudatoolkit installation file. 

If CUDA driver version matches with cudatoolkit version, then you can exclude the cuda driver installation when installing cudatoolkit. Otherwise simply follow the guide and install everything.

Use the following commands to see if cudatoolkit and driver are installed successfully.

```
nvidia-smi
```

```
nvcc -V
```

## 5 adapt opcode class and compute compatibility

In the function ```gemmGPUCUsingGPUPtr``` in header file ```gemmPU.cuh```, change the ```using``` statement so that ```cutlass::arch::``` tags indicate correct opcode class and compute compatibility (the ones found in step 1).

For example, given that the opcode class of current GPU is **SIMT** and compute compatibility is **6.1**, the code snippet should be defined as follows:

```
using Gemm = cutlass::gemm::device::Gemm<
        T,                                    // ElementA, namely type of A
        cutlass::layout::RowMajor,            // LayoutA, column major means colum of A is contiguous in memory
        T,                                    // ElementB
        cutlass::layout::RowMajor,            // LayoutB
        T,                                    // ElementOutput
        cutlass::layout::RowMajor,            // LayoutOutput
        T,                                    // ElementAccumulator
        cutlass::arch::OpClassSimt,           // tag indicating GPU opcode class, architecture-dependent
        cutlass::arch::Sm61                   // tag indicating GPU compute compatability, architecture-dependent
>;
```

## 6 run the test

In project root folder, run the following commands

```
cmake .
make
./CutlassTest
```

If the code compiles and executes successfully (with some print result), then the environment is all set up.

You can use ```git clean -fXd``` to clean test and build artifacts.