DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
		brick_domain.o \
		combinatorial_object_create.o \
		combinatorial_object_description.o \
		combinatorics_domain.o \
		geo_parameter.o \
		tdo_data.o \
		tdo_refinement_description.o \
		tdo_refinement.o \
		tdo_scheme.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

