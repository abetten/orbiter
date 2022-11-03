DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -std=c++14
# -frename-registers
#-march=native 
LIB = 
LFLAGS = -lm

MODULES = \
		boolean_function_domain.o \
		brick_domain.o \
		classification_of_objects_description.o \
		classification_of_objects_report_options.o \
		classification_of_objects.o \
		combinatorics_domain.o \
		domino_assignment.o \
		domino_change.o \
		encoded_combinatorial_object.o \
		geo_parameter.o \
		pentomino_puzzle.o \
		polynomial_function_domain.o \
		tdo_data.o \
		tdo_refinement_description.o \
		tdo_refinement.o \
		tdo_scheme_compute.o \
		tdo_scheme_synthetic.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

