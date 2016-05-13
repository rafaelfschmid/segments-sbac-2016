export BLOCK=32
export TIME=0
export BITS=10

all: generate cpu global segmented segmentedradix cub cuball
#share nthrust thrust

generate: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 parser.cpp -o parser.exe -std=c++11

cpu:	
	g++ -O3 cpu.cpp -o cpu.exe -std=c++11 -DELAPSED_TIME=$(TIME)

#share: sharev1.cu
#	nvcc sharev1.cu -o sharev1.exe -std=c++11 --expt-relaxed-constexpr -DEXP_BITS_SIZE=$(BITS) -DBLOCK_SIZE=$(BLOCK) -DELAPSED_TIME=$(TIME)

global: globalv1.cu
	nvcc -O3 globalv1.cu -o globalv1.exe -std=c++11 --expt-relaxed-constexpr -DEXP_BITS_SIZE=$(BITS) -DBLOCK_SIZE=$(BLOCK) -DELAPSED_TIME=$(TIME)

segmented: segmented_sort.cu
	nvcc -O3 segmented_sort.cu -o segmented.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

segmentedradix: segmented_radix_sort.cu
	nvcc -O3 segmented_radix_sort.cu -o segmentedradix.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

#nthrust: nthrust.cu
#	nvcc nthrust.cu -o nthrust.exe -std=c++11 -DELAPSED_TIME=$(TIME)

#thrust: thrust.cu
#	nvcc thrust.cu -o thrust.exe -std=c++11 -DELAPSED_TIME=$(TIME)

#by_key: by_key.cu
#	nvcc -O3 by_key.cu -o by_key.exe -std=c++11 -DELAPSED_TIME=$(TIME)

cub: cub.cu
	nvcc -O3 cub.cu -o cub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

cuball: cuball.cu
	nvcc -O3 cuball.cu -o cuball.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)
