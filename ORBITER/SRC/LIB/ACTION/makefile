DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action.o \
	action_by_conjugation.o \
	action_by_representation.o \
	action_by_restriction.o \
	action_by_right_multiplication.o \
	action_by_subfield_structure.o \
	action_cb.o \
	action_global.o \
	action_indexing_cosets.o \
	action_induce.o \
	action_init.o \
	action_on_andre.o \
	action_on_bricks.o \
	action_on_determinant.o \
	action_on_cosets.o \
	action_on_factor_space.o \
	action_on_grassmannian.o \
	action_on_homogeneous_polynomials.o \
	action_on_k_subsets.o \
	action_on_orbits.o \
	action_on_flags.o \
	action_on_orthogonal.o \
	action_on_sets.o \
	action_on_subgroups.o \
	action_on_sign.o \
	action_on_spread_set.o \
	action_on_wedge_product.o \
	backtrack.o \
	group.o \
	interface.o \
	interface_matrix_group.o \
	interface_perm_group.o \
	linear_group.o \
	linear_group_description.o \
	matrix_group.o \
	orbit_transversal.o \
	page_storage.o \
	perm_group.o \
	product_action.o \
	projective_space_with_action.o \
	schreier.o \
	schreier_sims.o \
	schreier_vector.o \
	set_and_stabilizer.o \
	sims.o \
	sims2.o \
	sims3.o \
	sims_global.o \
	strong_generators.o \
	strong_generators_groups.o \
	subgroup.o \
	union_find.o \
	union_find_on_k_subsets.o \
	vector_ge.o 


all : $(MODULES) 


clean:
	- rm *.o *.a *~



%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



