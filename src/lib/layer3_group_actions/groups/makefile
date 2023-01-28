DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	direct_product.o \
	exceptional_isomorphism_O4.o \
	linear_group.o \
	linear_group_description.o \
	matrix_group.o \
	orbits_on_something.o \
	permutation_group_create.o \
	permutation_group_description.o \
	permutation_representation_domain.o \
	permutation_representation.o \
	schreier.o \
	schreier_io.o \
	schreier_sims.o \
	sims_group_theory.o \
	sims_io.o \
	sims_main.o \
	sims.o \
	sims2.o \
	sims3.o \
	strong_generators.o \
	strong_generators_groups.o \
	subgroup.o \
	sylow_structure.o \
	wreath_product.o \
	linalg.o \
	shallow_schreier_ai.o


all : $(MODULES)

%.o: %.cpp %.c
	$(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

