SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++17
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64
#LFLAGS = -lm -lpthread -Wl,-rpath -Wl,/projects/abetten@colostate.edu/gcc-install-destdir/usr/local/gcc-8.1/lib64



TARGETS=\
		orbiter.out 

all: $(TARGETS)


orbiter.out:  orbiter.o ;
	$(DISCRETA_CPP)  orbiter.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~

