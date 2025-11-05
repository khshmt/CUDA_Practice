#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <array>
#include <algorithm>
#include <iostream>
#include <chrono>

#include <common.hpp>

__global__ void addTwoArray(int* arr1, int* arr2, int* sum, int size) {
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int block_offset = blockDim.x * blockDim.y * blockIdx.x;
	int row_offset = blockDim.y * blockDim.x * gridDim.x * blockIdx.y;
	int gid = row_offset + block_offset + tid;
	
	if (gid < size) {
		sum[gid] = arr1[gid] + arr2[gid];
	}
}

void addTwoArrayCPU(int* arr1, int* arr2, int* sum, int size) {
	for (int i = 0; i < size; ++i) {
		sum[i] = arr1[i] + arr2[i];
	}
}

int main() {
	auto size{10'000};
	auto block_size{ 512 };
	auto NUM_OF_BYTES = size * sizeof(int);
	Timer timer;

	int* h_a, *h_b, *gpu_results, *h_c;

	h_a = (int*)malloc(NUM_OF_BYTES);
	h_b = (int*)malloc(NUM_OF_BYTES);
	gpu_results = (int*)malloc(NUM_OF_BYTES);
	h_c  = (int*)malloc(NUM_OF_BYTES);

	time_t t;
	srand((unsigned)time(&t));

	for (int i{ 0 }; i < size; ++i) {
		h_a[i] = (int)(rand() & 0xFF);
	}

	for (int i{ 0 }; i < size; ++i) {
		h_b[i] = (int)(rand() & 0xFF);
	}

	int* d_a, *d_b, *d_c;
	gpuErrCheck(cudaMalloc((int**)&d_a, NUM_OF_BYTES));
	gpuErrCheck(cudaMalloc((int**)&d_b, NUM_OF_BYTES));
	gpuErrCheck(cudaMalloc((int**)&d_c, NUM_OF_BYTES));

	dim3 block(block_size, 1, 1);
	dim3 grid((size / block.x) + 1, 1, 1);
	
	timer.startWatch();
	cudaMemcpy(d_a, h_a, NUM_OF_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NUM_OF_BYTES, cudaMemcpyHostToDevice);
	timer.endWatch().printElapsedTime("CPU to GPU: ");
	
	timer.startWatch();
	addTwoArray << <grid, block >> > (d_a, d_b, d_c, size);
	timer.endWatch().printElapsedTime("Kernel launch: ");

	// waits untill all kernels return
	cudaDeviceSynchronize();

	// transfer data back to Host
	timer.startWatch();
	gpuErrCheck(cudaMemcpy(gpu_results, d_c, NUM_OF_BYTES, cudaMemcpyDeviceToHost));
	timer.endWatch().printElapsedTime("GPU to CPU: ");
	
	timer.startWatch();
	addTwoArrayCPU(h_a, h_b, h_c, size);
	timer.endWatch().printElapsedTime("CPU: ");

	compareTwoArrays<int>(gpu_results, h_c, size);

	// Free GPU Data
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	// Free CPU Data
	free(h_a);
	free(h_b);
	free(gpu_results);

	cudaDeviceReset();
	
	return 0;
}