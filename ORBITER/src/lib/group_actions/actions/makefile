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
	action_pointer_table.o \
	backtrack.o \
	interface.o \
	interface_direct_product.o \
	interface_matrix_group.o \
	interface_perm_group.o \
	interface_wreath_product.o
	

all : $(MODULES)

clean:
	- rm *.o *.a *~

