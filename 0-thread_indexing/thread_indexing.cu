
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <array>
#include <algorithm>
#include <thread>

#include <common.hpp>

// threadIdx is the thread index in the  block
// blockIdx is the block index in the grid

// blockDim is the number of threads in each dimension of a block (all the block have the same size)
// gridDim is the number of block in each dimension of agrid

__global__ void print_details() {
	printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, \
			blockDim.x: %d, blockDim.y: %d, blockDim.z: %d,\
			girdDim.x: %d, girdDim.y: %d, girdDim.z: %d\n", 
		threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

__global__ void unqiue_gid_calc_2d_2d(int* input) {
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int block_offset = blockDim.x * blockDim.y * blockIdx.x;
	int row_offset = blockDim.y * blockDim.x * gridDim.x * blockIdx.y;
	int gid = row_offset + block_offset + tid  ; // global thread index in the grid
	printf("blockIdx: %d, threadIdx: %d, gid: %d, value: %d\n", blockIdx.x, tid, gid, input[gid]);
}

int main() {
	query_device();

	int* d_data{nullptr};
	std::array<int, 16> h_data{ 23, 9, 4, 53, 65, 12, 1, 33, 45, 32, 76, 32, 11, 3, 55, 99 };
	std::for_each(std::begin(h_data), std::end(h_data), [](const auto& elem) {printf("%d ", elem); });
	std::cout << std::endl;

	cudaMalloc((void**)&d_data, sizeof(h_data));
	cudaMemcpy(d_data, h_data.data(), sizeof(h_data), cudaMemcpyHostToDevice);

	dim3 block(2, 2, 1);
	dim3 grid(2, 2, 1);
	unqiue_gid_calc_2d_2d << <grid, block >> > (d_data);

	cudaDeviceSynchronize();

	cudaFree(d_data);
	cudaDeviceReset();
	return 0;
}
