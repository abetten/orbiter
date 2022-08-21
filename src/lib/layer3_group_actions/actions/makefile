DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action.o \
	action_cb.o \
	action_global.o \
	action_group_theory.o \
	action_indexing_cosets.o \
	action_induce.o \
	action_init.o \
	action_io.o \
	action_pointer_table.o \
	action_projective.o \
	backtrack.o \
	interface_direct_product.o \
	interface_induced_action.o \
	interface_matrix_group.o \
	interface_perm_group.o \
	interface_permutation_representation.o \
	interface_wreath_product.o \
	nauty_interface_with_group.o \
	stabilizer_chain_base_data.o 
	

all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

