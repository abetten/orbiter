SRC=../../../../ORBITER/src
DISCRETA_CPP = nvcc
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm






ALL_TARGETS=wreath_product.out




#-------------------------------------------------------------
# vars required by the CUDA compiler
#-------------------------------------------------------------
CUDA_LIB=-lcudart -lcudadevrt
CUDA_FLAGS= -gencode arch=compute_50,code=sm_50 \
			-gencode arch=compute_60,code=sm_60 \
			-gencode arch=compute_61,code=sm_61 \
			-Xptxas -O3 \
			-Xcompiler -O3 \
			-Xcompiler -march=native \
			-Xcompiler -mtune=native \
			-Xcompiler -funroll-loops \
			-Xcompiler -fgcse-lm \
			-Xcompiler $(CSTD)

TRUE=1
FALSE=0

# Check for CUDA compiler
COMPILE_ON_GPU=
ifeq ( , $(shell which nvcc))
	COMPILE_ON_GPU = $(FALSE)
else
	COMPILE_ON_GPU = $(TRUE)
endif
#-------------------------------------------------------------




all: $(ALL_TARGETS)
ifeq ($(TRUE), $(COMPILE_ON_GPU))
	nvcc $(CUDA_FLAGS) $(CPPFLAGS) wreath_product.o -o wreath_product.out $(LIB) $(CUDA_LIB)
else
	$(CC) $(CSTD) $(CFLAGS) $(OPTFLAGS) $(ALL_TARGETS) $(INCLUDES) $(LIBS_DIR) $(LIBS) -o main
endif

	
wreath_product.o: wreath_product.cpp
ifeq ($(TRUE), $(COMPILE_ON_GPU))
	nvcc -x cu -c $(CUDA_FLAGS) $(CPPFLAGS) -rdc=true $< -o $@
else
	@echo "----------------------------------------"
	@echo -n "compiling without CUDA: "
	@echo $<
	@echo "----------------------------------------"
	$(CC) $(CSTD) $(CFLAGS) $(CPPFLAGS) $(OPTFLAGS) $(INCLUDES) $(LIBS_DIR) $(LIBS) -c $<
endif













not_cuda_all:  \
	action_on_set_partitions.out \
	direct_product.out \
	linear_group.out \
	orthogonal_group.out \
	wreath_product.out


action_on_set_partitions.out:  action_on_set_partitions.o;
	$(DISCRETA_CPP)  action_on_set_partitions.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


direct_product.out:  direct_product.o;
	$(DISCRETA_CPP)  direct_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


linear_group.out:  linear_group.o;
	$(DISCRETA_CPP)  linear_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


orthogonal_group.out:  orthogonal_group.o;
	$(DISCRETA_CPP)  orthogonal_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


wreath_product.out:  wreath_product.o;
	$(DISCRETA_CPP)  wreath_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.out *~

