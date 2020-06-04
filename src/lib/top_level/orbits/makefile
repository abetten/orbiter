DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	kramer_mesner.o \
	orbit_of_equations.o \
	orbit_of_sets.o \
	orbit_of_subspaces.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

