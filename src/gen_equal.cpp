#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

#ifdef RAND
#define generate_number(i, n, bits_size_elements) (rand() % bits_size_elements);
#elif SORTASC
#define generate_number(i, n, bits_size_elements) (i);
#elif SORTDESC
#define generate_number(i, n, bits_size_elements) (n-i-1);
#endif

void segments_gen(int num_segments, int size_segment) {
	for (int i = 0; i < num_segments+1; i++)
		printf("%d ", i * size_segment);
}

void vectors_gen(int num_elements, int bits_size_elements) {
	for (int i = 0; i < num_elements; i++)
	{
		std::cout << generate_number(i, num_elements, bits_size_elements);
		std::cout << " ";
	}
}

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

