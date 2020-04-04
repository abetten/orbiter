SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++14
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



TARGETS=\
		orbiter.out 

all: $(TARGETS)


orbiter.out:  orbiter.o ;
	$(DISCRETA_CPP)  orbiter.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~

