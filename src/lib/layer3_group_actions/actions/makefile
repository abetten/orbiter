DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action.o \
	action_cb.o \
	action_global.o \
	action_indexing_cosets.o \
	action_induce.o \
	action_init.o \
	action_io.o \
	action_pointer_table.o \
	backtrack.o \
	group_element.o \
	induced_action.o \
	interface_direct_product.o \
	interface_induced_action.o \
	interface_matrix_group.o \
	interface_perm_group.o \
	interface_permutation_representation.o \
	interface_polarity_extension.o \
	interface_wreath_product.o \
	known_groups.o \
	stabilizer_chain_base_data.o 
	

all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

