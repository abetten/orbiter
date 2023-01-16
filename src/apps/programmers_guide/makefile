SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++17
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64
#LFLAGS = -lm -lpthread -Wl,-rpath -Wl,/projects/abetten@colostate.edu/gcc-install-destdir/usr/local/gcc-8.1/lib64



TARGETS=\
		ug_3_2_F_2.out \
		ug_3_2_vandermonde.out \
		ug_5_3_quaternion.out

all: $(TARGETS)


ug_3_2_F_2.out:  ug_3_2_F_2.o ;
	$(DISCRETA_CPP)  ug_3_2_F_2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ug_3_2_vandermonde.out:  ug_3_2_vandermonde.o ;
	$(DISCRETA_CPP)  ug_3_2_vandermonde.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ug_5_3_quaternion.out:  ug_5_3_quaternion.o ;
	$(DISCRETA_CPP)  ug_5_3_quaternion.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~

