DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I.
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	a_domain.o \
	finite_field.o \
	finite_field_linear_algebra.o \
	finite_field_representations.o \
	finite_field_tables.o \
	finite_ring.o \
	generators_symplectic_group.o \
	gl_classes.o \
	group_generators.o \
	heisenberg.o \
	homogeneous_polynomial_domain.o \
	longinteger_domain.o \
	longinteger_object.o \
	norm_tables.o \
	null_polarity_generator.o \
	number_theory.o \
	rank_checker.o \
	subfield_structure.o \
	unipoly_domain.o


all : $(MODULES)

clean:
	- rm *.o *.a *~

