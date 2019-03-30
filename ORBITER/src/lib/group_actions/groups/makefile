DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	direct_product.o \
	exceptional_isomorphism_O4.o \
	linear_group.o \
	linear_group_description.o \
	matrix_group.o \
	perm_group.o \
	schreier.o \
	schreier_io.o \
	schreier_sims.o \
	sims.o \
	sims2.o \
	sims3.o \
	strong_generators.o \
	strong_generators_groups.o \
	subgroup.o \
	sylow_structure.o \
	wreath_product.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~

