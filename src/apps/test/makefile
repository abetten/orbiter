SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



TARGETS=\
		test_longinteger.out 

all: $(TARGETS)


test_longinteger.out:  test_longinteger.o ;
	$(DISCRETA_CPP)  test_longinteger.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~

