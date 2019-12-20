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





TARGETS=\
	spread_classify.out \
	spread_create.out

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin



spread_classify.out: spread_classify.o  ;
	$(DISCRETA_CPP) spread_classify.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

spread_create.out: spread_create.o ;
	$(DISCRETA_CPP) spread_create.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

clean:
	- rm *.o *.a *.out *~
