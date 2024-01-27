DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -std=c++17
# -frename-registers
#-march=native 
LIB = 
LFLAGS = -lm

MODULES = \
		apn_functions.o \
		boolean_function_domain.o \
		brick_domain.o \
		combinatorics_domain.o \
		decomposition_scheme.o \
		decomposition.o \
		domino_assignment.o \
		domino_change.o \
		geo_parameter.o \
		pentomino_puzzle.o \
		polynomial_function_domain.o \
		row_and_col_partition.o \
		tdo_data.o \
		tdo_refinement_description.o \
		tdo_refinement.o \
		tdo_scheme_compute.o \
		tdo_scheme_synthetic.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

