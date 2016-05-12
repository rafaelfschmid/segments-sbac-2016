/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <chrono>
#include <iostream>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

void print(thrust::host_vector<int> h_vec) {
	std::cout << "\n";
	for (int i = 0; i < h_vec.size(); i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	thrust::host_vector<int> h_seg(num_of_segments + 1);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	thrust::host_vector<int> h_vec(num_of_elements);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	std::chrono::high_resolution_clock::time_point start1 =
				std::chrono::high_resolution_clock::now();
	thrust::host_vector<int> h_norm(num_of_segments);
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
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
//	print(h_vec);
//	print(h_norm);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	thrust::device_vector<int> d_vec = h_vec;

	cudaEventRecord(start);
	thrust::sort(d_vec.begin(), d_vec.end());
	cudaEventRecord(stop);

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	start1 = std::chrono::high_resolution_clock::now();
	for (i = 0; i < num_of_segments; i++) {
		for (int j = h_seg[i]; j < h_seg[i + 1]; j++) {
			h_vec[j] -= h_norm[i];
		}
	}
	stop1 = std::chrono::high_resolution_clock::now();
	time_span += std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds << "\n";
	} else
		print(h_vec);

	return 0;
}
