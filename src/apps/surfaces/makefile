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
	surfaces_arc_lifting_main.out 

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin



surfaces_arc_lifting_main.out:  surfaces_arc_lifting_main.o ;
	$(DISCRETA_CPP)  surfaces_arc_lifting_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~
