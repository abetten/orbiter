CC=gcc
CPP=g++
OPTFLAGS=-O3 -march=native -mtune=native -funroll-loops

CUDA_LIB=-lcuda -lcudart -lcudadevrt
CUDA_FLAGS= -gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_60,code=sm_60 \
			-gencode arch=compute_61,code=sm_61 \
			-Xptxas -O3 \
			-Xcompiler -O3 \
			-Xcompiler -march=native \
			-Xcompiler -mtune=native \
			-Xcompiler -funroll-loops \
			-Xcompiler -fgcse-lm \
			-Xcompiler -ftree-vectorize \
			-Xcompiler -mavx \
			-Xcompiler -mfpmath=both
			

all: main.o gpuErrchk.o	
	nvcc $(CUDA_FLAGS) main.o gpuErrchk.o -o main $(CUDA_LIB)
	

main.o: main.cpp Matrix.h linalg.h gpu_table.h
	nvcc -x cu -c $(CUDA_FLAGS) -rdc=true main.cpp -o main.o
	

gpuErrchk.o: gpuErrchk.cpp gpuErrchk.h
	nvcc -x cu -c $(CUDA_FLAGS) -rdc=true gpuErrchk.cpp -o gpuErrchk.o


clean:
	rm -rf ./*.o