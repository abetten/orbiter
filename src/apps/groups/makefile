SRC=../..
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I../../lib/foundations

INCLUDES=\
			-I$(SRC)/lib -I../../lib/foundations \
			-I$(SRC)/lib -I../../lib/foundations/CUDA/
			
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64





ALL_TARGETS=\
	action_on_set_partitions.out \
	exceptional_isomorphism_O4_main.out \
	linear_group.out \
	orthogonal_group.out \
	wreath_product_main.out

all: $(ALL_TARGETS)

install:
	cp $(ALL_TARGETS) ../../../bin




#-------------------------------------------------------------
# vars required by the CUDA compiler
#-------------------------------------------------------------
CSTD=-std=c++14
CUDA_LIB=-lcudart -lcudadevrt -L/scratch/cuda10/lib64
CUDA_FLAGS= -gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_52,code=sm_52 \
			-gencode arch=compute_53,code=sm_53 \
			-gencode arch=compute_60,code=sm_60 \
			-gencode arch=compute_61,code=sm_61 \
			-gencode arch=compute_62,code=sm_62 \
			-gencode arch=compute_70,code=sm_70 \
			-gencode arch=compute_72,code=sm_72 \
			-gencode arch=compute_75,code=sm_75 \
			-Xptxas -O3 \
			-Xcompiler -O3 \
			-Xcompiler -march=native \
			-Xcompiler -mtune=native \
			-Xcompiler -funroll-loops \
			-Xcompiler -fgcse-lm \
			-Xcompiler $(CSTD)

TRUE=1
FALSE=0


#-------------------------------------------------------------


all:
	junk=$$(which nvidia-smii);\
	if [ $$? -eq 0 ]; then \
		$(MAKE) COMPILE_ON_GPU=$(TRUE) cuda_all; \
	else \
		$(MAKE) COMPILE_ON_GPU=$(FALSE) not_cuda_all; \
	fi; \





not_cuda_all: $(ALL_TARGETS)


action_on_set_partitions.out:  action_on_set_partitions.o;
	$(DISCRETA_CPP)  action_on_set_partitions.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



exceptional_isomorphism_O4_main.out:  exceptional_isomorphism_O4_main.o;
	$(DISCRETA_CPP)  exceptional_isomorphism_O4_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


linear_group.out:  linear_group.o;
	$(DISCRETA_CPP)  linear_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


orthogonal_group.out:  orthogonal_group.o;
	$(DISCRETA_CPP)  orthogonal_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


wreath_product_main.out:  wreath_product_main.o;
ifeq ($(TRUE), $(COMPILE_ON_GPU))
	@echo "wreath_product_main.out"
	$(DISCRETA_CPP) wreath_product_main.o wreath_product_gpuCode.o $(CPPFLAGS) $(LFLAGS) $(LIB) $(CUDA_LIB) -o $@
else
	$(DISCRETA_CPP)  wreath_product_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
endif

## CUDA

cuda_all: $(ALL_TARGETS)
	@echo "Done with cuda_all"


wreath_product_main.o: wreath_product_main.cpp
ifeq ($(TRUE), $(COMPILE_ON_GPU))
	@echo "----------------------------------------"
	@echo -n "compiling with CUDA wreath_product_main.o: "
	@echo $<
	@echo "----------------------------------------"
	nvcc -x cu -c $(CUDA_FLAGS) $(INCLUDES) -rdc=true $< -o $@ $(CUDA_LIB)
	nvcc $(CUDA_FLAGS) -I. -dlink $(CUDA_LIB) wreath_product_main.o -o wreath_product_gpuCode.o
	@echo "----------------------------------------"
else
	@echo "----------------------------------------"
	@echo -n "compiling without CUDA: "
	@echo $@
	@echo $<
	@echo "----------------------------------------"
	$(DISCRETA_CPP) -c $< $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
endif


clean:
	- rm *.o *.out *~

