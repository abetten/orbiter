DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	orbit_of_equations.o \
	orbit_of_sets.o \
	orbit_of_subspaces.o \
	orbit_rep.o \
	representatives.o \
	subspace_orbits.o


all : $(MODULES)

clean:
	- rm *.o *.a *~


#%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



