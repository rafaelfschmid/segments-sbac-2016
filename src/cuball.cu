/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <bitset>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void print(int* host_data, int n) {
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	int mem_size_seg = sizeof(int) * (num_of_segments + 1);
	int *h_seg = (int *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(int) * num_of_elements;
	int *h_vec = (int *) malloc(mem_size_vec);
	int *h_value = (int *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	std::chrono::high_resolution_clock::time_point start1 =
			std::chrono::high_resolution_clock::now();
	int *h_norm = (int *) malloc(mem_size_seg);
	int previousMax = 0;
	for (i = 0; i < num_of_segments; i++) {
		int currentMin = h_vec[h_seg[i]];
		int currentMax = h_vec[h_seg[i]];

		for (int j = h_seg[i] + 1; j < h_seg[i + 1]; j++) {
			if (h_vec[j] < currentMin)
				currentMin = h_vec[j];
			else if (h_vec[j] > currentMax)
				currentMax = h_vec[j];
		}

		int normalize = previousMax - currentMin;
		h_norm[i] = ++normalize;
		for (int j = h_seg[i]; j < h_seg[i + 1]; j++) {
			h_vec[j] += normalize;
		}
		previousMax = currentMax + normalize;
	}
	std::chrono::high_resolution_clock::time_point stop1 =
			std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<
			std::chrono::duration<double>>(stop1 - start1);
//	print(h_vec);
//	print(h_norm);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *d_value, *d_value_out, *d_vec, *d_vec_out;
	void *d_temp = NULL;
	size_t temp_bytes = 0;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out,
			d_value, d_value_out, num_of_elements);
	cudaMalloc((void **) &d_temp, temp_bytes);
	cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out,
			d_value, d_value_out, num_of_elements);
	cudaEventRecord(stop);

	//cudaTest(cudaPeekAtLastError());
	cudaMemcpy(h_value, d_value_out, mem_size_vec, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

	start1 = std::chrono::high_resolution_clock::now();
	for (i = 0; i < num_of_segments; i++) {
		for (int j = h_seg[i]; j < h_seg[i + 1]; j++) {
			h_vec[j] -= h_norm[i];
		}
	}
	stop1 = std::chrono::high_resolution_clock::now();
	time_span += std::chrono::duration_cast<std::chrono::duration<double>>(
			stop1 - start1);

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds + time_span.count() * 1000 << "\n";
		//std::cout << milliseconds << "\n";
	} else
		print(h_vec, num_of_elements);

	free(h_vec);
	cudaFree(d_vec);
	cudaFree(d_vec_out);
	cudaFree(d_value);
	cudaFree(d_value_out);
	cudaFree(d_temp);

	return 0;
}
