DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
		boolean_function_domain.o \
		brick_domain.o \
		combinatorial_object_activity_description.o \
		combinatorial_object_activity.o \
		combinatorial_object_create.o \
		combinatorial_object_description.o \
		combinatorics_domain.o \
		geo_parameter.o \
		pentomino_puzzle.o \
		tdo_data.o \
		tdo_refinement_description.o \
		tdo_refinement.o \
		tdo_scheme.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

