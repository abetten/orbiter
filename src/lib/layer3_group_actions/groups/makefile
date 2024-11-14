DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	any_group_linear.o \
	any_group.o \
	conjugacy_class_of_elements.o \
	conjugacy_class_of_subgroups.o \
	exceptional_isomorphism_O4.o \
	group_theory_global.o \
	orbits_on_something.o \
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
	subgroup_lattice_layer.o \
	subgroup_lattice.o \
	subgroup.o \
	sylow_structure.o \
	linalg.o \
	shallow_schreier_ai.o


all : $(MODULES)

%.o: %.cpp %.c
	$(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

