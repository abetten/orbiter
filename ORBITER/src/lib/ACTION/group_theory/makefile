DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	linear_group.o \
	linear_group_description.o \
	matrix_group.o \
	perm_group.o \
	schreier.o \
	schreier_sims.o \
	sims.o \
	sims2.o \
	sims3.o \
	sims_global.o \
	strong_generators.o \
	strong_generators_groups.o \
	subgroup.o \
	wreath_product.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



