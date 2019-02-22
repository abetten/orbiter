DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++11
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	kramer_mesner.o \
	orbit_of_equations.o \
	orbit_of_sets.o \
	orbit_of_subspaces.o \
	orbit_rep.o \
	subspace_orbits.o


all : $(MODULES)

clean:
	- rm *.o *.a *~

