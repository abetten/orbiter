SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm

all: test_longinteger.out 



test_longinteger.out:  test_longinteger.o ;
	$(DISCRETA_CPP)  test_longinteger.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




clean:
	- rm *.o *.a *.out *~

