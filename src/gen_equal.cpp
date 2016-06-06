#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

void segments_gen(int num_segments, int size_segment) {
	for (int i = 0; i < num_segments+1; i++)
		printf("%d ", i * size_segment);
}

#ifdef RAND
void vectors_gen(int num_elements, int bits_size_elements) {

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << rand() % bits_size_elements;
		std::cout << " ";
	}
}
#elif SORTASC
void vectors_gen(int num_elements, int bits_size_elements) {
	std::vector<int> vec;

	for (int i = 0; i < num_elements; i++)
	{
		vec.push_back(rand() % bits_size_elements);
	}

	std::sort(vec.begin(), vec.end());

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << vec[i];
		std::cout << " ";
	}
}
#elif SORTDESC
void vectors_gen(int num_elements, int bits_size_elements) {
	std::vector<int> vec;

	for (int i = 0; i < num_elements; i++)
	{
		vec.push_back(rand() % bits_size_elements);
	}

	std::sort(vec.begin(), vec.end(), std::greater<int>());

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << vec[i];
		std::cout << " ";
	}
}
#endif

int main(int argc, char** argv) {

	if (argc < 3) {
		printf(
				"Parameters needed: <number of segments> <size of each segment> <exp size number>\n\n");
		return 0;
	}

	int number_of_segments = atoi(argv[1]);
	int size_of_segments = atoi(argv[2]);
	int number_of_elements = number_of_segments*size_of_segments;

	srand(time(NULL));
	printf("%d\n", number_of_segments);
	segments_gen(number_of_segments, size_of_segments);
	printf("\n");

	printf("%d\n", number_of_elements);
	vectors_gen(number_of_elements, pow(2, EXP_BITS_SIZE));
	printf("\n");

	return 0;
}

