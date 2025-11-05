#pragma once
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <timer.hpp>

#define gpuErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		std::cerr << cudaGetErrorString(code) << " " << file << " " << line << '\n';
		if (abort)
			std::exit(code);
	}
}


cudaDeviceProp query_device(int devNo = 0)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Failed to get device count: " << cudaGetErrorString(err) << '\n';
        return cudaDeviceProp{};
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found.\n";
        return cudaDeviceProp{};
    }

    std::cout << "Found " << deviceCount << " CUDA device(s):\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\n=== Device " << i << " : " << prop.name << " ===\n";
        std::cout << "Compute capability:          " << prop.major << "." << prop.minor << '\n';
        std::cout << "Multiprocessors:              " << prop.multiProcessorCount << '\n';
        std::cout << "CUDA cores per multiprocessor (approx): "
            << (prop.major >= 2 ? 128 : 8) << '\n'; // rough estimate
        std::cout << "Total global memory:          " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "Shared memory per block:      " << (prop.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << "Shared memory per MP:      " << (prop.sharedMemPerMultiprocessor/ 1024) << " KB\n";
        std::cout << "Registers per block:          " << prop.regsPerBlock << '\n';
        std::cout << "Warp size:                    " << prop.warpSize << '\n';
        std::cout << "Max threads per block:        " << prop.maxThreadsPerBlock << '\n';
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << '\n';
        std::cout << "Max grid dimensions:          ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")\n";
        std::cout << "Max block dimensions:         ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")\n";
        std::cout << "Clock rate:                   " << (prop.clockRate / 1000.0f) << " MHz\n";
        std::cout << "Memory Clock Rate:            " << (prop.memoryClockRate / 1000.0f) << " MHz\n";
        std::cout << "Memory Bus Width:             " << prop.memoryBusWidth << " bits\n";
        std::cout << "L2 Cache Size:                " << (prop.l2CacheSize / 1024) << " KB\n";
    }

    std::cout << "Using device " << devNo << "\n\n";

    cudaDeviceProp currentProp;
    cudaGetDeviceProperties(&currentProp, devNo);
    cudaSetDevice(devNo);

    return currentProp;
}


template <typename T>
bool compareTwoArrays(T* a, T* b, int size) {
	for (int i{ 0 }; i < size; ++i) {
		if (a[i] != b[i]) {
			std::cout << "The Two Array are NOT the same\n";
			return false;
		}
	}
	std::cout << "The Two Array are the same\n";
	return true;
}