SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



TARGETS=\
	regular_ls.out transpose.out

all: $(TARGETS)


install:
	cp $(TARGETS) ../../../bin


regular_ls.out:  regular_ls.o ;
	$(DISCRETA_CPP)  regular_ls.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

transpose.out:  transpose.o ;
	$(DISCRETA_CPP)  transpose.o  \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~

