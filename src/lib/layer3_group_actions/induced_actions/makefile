DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action_by_conjugation.o \
	action_by_representation.o \
	action_by_restriction.o \
	action_by_right_multiplication.o \
	action_by_subfield_structure.o \
	action_on_andre.o \
	action_on_bricks.o \
	action_on_cosets.o \
	action_on_determinant.o \
	action_on_factor_space.o \
	action_on_flags.o \
	action_on_galois_group.o \
	action_on_grassmannian.o \
	action_on_homogeneous_polynomials.o \
	action_on_interior_direct_product.o \
	action_on_k_subsets.o \
	action_on_module.o \
	action_on_orbits.o \
	action_on_orthogonal.o \
	action_on_set_partitions.o \
	action_on_sets.o \
	action_on_sign.o \
	action_on_spread_set.o \
	action_on_subgroups.o \
	action_on_wedge_product.o \
	product_action.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
