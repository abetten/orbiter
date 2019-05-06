SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm


all:  desarguesian_spread.out \
	linear_set_main.out \
	long_orbit.out \
	subspace_orbits_main.out 



desarguesian_spread.out:  desarguesian_spread.o;
	$(DISCRETA_CPP)  desarguesian_spread.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

linear_set_main.out:  linear_set_main.o linear_set.o linear_set2.o ;
	$(DISCRETA_CPP) linear_set_main.o linear_set.o linear_set2.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

long_orbit.out:  long_orbit.o;
	$(DISCRETA_CPP)  long_orbit.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subspace_orbits_main.out:  subspace_orbits_main.o;
	$(DISCRETA_CPP)  subspace_orbits_main.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~


