DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	finite_ring.o \
	homogeneous_polynomial_domain.o \
	longinteger_domain.o \
	longinteger_object.o \
	partial_derivative.o \
	polynomial_double_domain.o \
	polynomial_double.o \
	polynomial_ring_activity_description.o \
	polynomial_ring_description.o \
	ring_theory_global.o \
	table_of_irreducible_polynomials.o \
	unipoly_domain.o \
	unipoly_domain2.o 


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



