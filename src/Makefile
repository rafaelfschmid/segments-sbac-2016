export BLOCK=32
export TIME=1
export BITS=10

all: generate cpu mergeseg radixseg fixcub fixcuball 
#bitonicsort mergesort oddevensort quicksort

generate: gen_equal.cpp gen_diff.cpp
	g++ -O3 gen_equal.cpp -o equal.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 gen_diff.cpp -o diff.exe -DEXP_BITS_SIZE=$(BITS)
	g++ -O3 parser.cpp -o parser.exe -std=c++11

cpu:	
	g++ -O3 cpu.cpp -o cpu.exe -std=c++11 -DELAPSED_TIME=$(TIME)

mergeseg: mergeseg.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 mergeseg.cu -o mergeseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

radixseg: radixseg.cu
	nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 radixseg.cu -o radixseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

fixcub: fixcub.cu
	nvcc -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixcub.cu -o fixcub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)

fixcuball: fixcuball.cu
	nvcc -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 fixcuball.cu -o fixcuball.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME)



