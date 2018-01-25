DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm
CPP_PREFIX=b_
UNIX_TO_DOS = ~/bin/unix2dos -n 

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

LIB = libaction.a

libaction.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~



%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


cpp:
	$(UNIX_TO_DOS) action.C $(CPP_PREFIX)action.cpp
	$(UNIX_TO_DOS) action_global.C $(CPP_PREFIX)action_global.cpp
	$(UNIX_TO_DOS) action_indexing_cosets.C $(CPP_PREFIX)action_indexing_cosets.cpp
	$(UNIX_TO_DOS) action_by_conjugation.C $(CPP_PREFIX)action_by_conjugation.cpp
	$(UNIX_TO_DOS) action_by_representation.C $(CPP_PREFIX)action_by_representation.cpp
	$(UNIX_TO_DOS) action_by_restriction.C $(CPP_PREFIX)action_by_restriction.cpp
	$(UNIX_TO_DOS) action_by_right_multiplication.C $(CPP_PREFIX)action_by_right_multiplication.cpp
	$(UNIX_TO_DOS) action_by_subfield_structure.C $(CPP_PREFIX)action_by_subfield_structure.cpp
	$(UNIX_TO_DOS) action_on_andre.C $(CPP_PREFIX)action_on_andre.cpp
	$(UNIX_TO_DOS) action_on_bricks.C $(CPP_PREFIX)action_on_bricks.cpp
	$(UNIX_TO_DOS) action_cb.C $(CPP_PREFIX)action_cb.cpp
	$(UNIX_TO_DOS) action_induce.C $(CPP_PREFIX)action_induce.cpp
	$(UNIX_TO_DOS) action_init.C $(CPP_PREFIX)action_init.cpp
	$(UNIX_TO_DOS) action_on_determinant.C $(CPP_PREFIX)action_on_determinant.cpp
	$(UNIX_TO_DOS) action_on_cosets.C $(CPP_PREFIX)action_on_cosets.cpp
	$(UNIX_TO_DOS) action_on_factor_space.C $(CPP_PREFIX)action_on_factor_space.cpp
	$(UNIX_TO_DOS) action_on_grassmannian.C $(CPP_PREFIX)action_on_grassmannian.cpp
	$(UNIX_TO_DOS) action_on_homogeneous_polynomials.C $(CPP_PREFIX)action_on_homogeneous_polynomials.cpp
	$(UNIX_TO_DOS) action_on_orthogonal.C $(CPP_PREFIX)action_on_orthogonal.cpp
	$(UNIX_TO_DOS) action_on_sets.C $(CPP_PREFIX)action_on_sets.cpp
	$(UNIX_TO_DOS) action_on_subgroups.C $(CPP_PREFIX)action_on_subgroups.cpp
	$(UNIX_TO_DOS) action_on_orbits.C $(CPP_PREFIX)action_on_orbits.cpp
	$(UNIX_TO_DOS) action_on_flags.C $(CPP_PREFIX)action_on_flags.cpp
	$(UNIX_TO_DOS) action_on_k_subsets.C $(CPP_PREFIX)action_on_k_subsets.cpp
	$(UNIX_TO_DOS) action_on_wedge_product.C $(CPP_PREFIX)action_on_wedge_product.cpp
	$(UNIX_TO_DOS) backtrack.C $(CPP_PREFIX)backtrack.cpp
	$(UNIX_TO_DOS) group.C $(CPP_PREFIX)group.cpp
	$(UNIX_TO_DOS) interface.C $(CPP_PREFIX)interface.cpp
	$(UNIX_TO_DOS) interface_matrix_group.C $(CPP_PREFIX)interface_matrix_group.cpp
	$(UNIX_TO_DOS) interface_perm_group.C $(CPP_PREFIX)interface_perm_group.cpp
	$(UNIX_TO_DOS) linear_group.C $(CPP_PREFIX)linear_group.cpp
	$(UNIX_TO_DOS) linear_group_description.C $(CPP_PREFIX)linear_group_description.cpp
	$(UNIX_TO_DOS) matrix_group.C $(CPP_PREFIX)matrix_group.cpp
	$(UNIX_TO_DOS) orbit_transversal.C $(CPP_PREFIX)orbit_transversal.cpp
	$(UNIX_TO_DOS) page_storage.C $(CPP_PREFIX)page_storage.cpp
	$(UNIX_TO_DOS) perm_group.C $(CPP_PREFIX)perm_group.cpp
	$(UNIX_TO_DOS) product_action.C $(CPP_PREFIX)product_action.cpp
	$(UNIX_TO_DOS) projective_space_with_action.C $(CPP_PREFIX)projective_space_with_action.cpp
	$(UNIX_TO_DOS) schreier.C $(CPP_PREFIX)schreier.cpp
	$(UNIX_TO_DOS) schreier_sims.C $(CPP_PREFIX)schreier_sims.cpp
	$(UNIX_TO_DOS) schreier_vector.C $(CPP_PREFIX)schreier_vector.cpp
	$(UNIX_TO_DOS) set_and_stabilizer.C $(CPP_PREFIX)set_and_stabilizer.cpp
	$(UNIX_TO_DOS) sims.C $(CPP_PREFIX)sims.cpp
	$(UNIX_TO_DOS) sims2.C $(CPP_PREFIX)sims2.cpp
	$(UNIX_TO_DOS) sims3.C $(CPP_PREFIX)sims3.cpp
	$(UNIX_TO_DOS) sims_global.C $(CPP_PREFIX)sims_global.cpp
	$(UNIX_TO_DOS) strong_generators.C $(CPP_PREFIX)strong_generators.cpp
	$(UNIX_TO_DOS) strong_generators_groups.C $(CPP_PREFIX)strong_generators_groups.cpp
	$(UNIX_TO_DOS) subgroup.C $(CPP_PREFIX)subgroup.cpp
	$(UNIX_TO_DOS) union_find.C $(CPP_PREFIX)union_find.cpp
	$(UNIX_TO_DOS) union_find_on_k_subsets.C $(CPP_PREFIX)union_find_on_k_subsets.cpp
	$(UNIX_TO_DOS) vector_ge.C $(CPP_PREFIX)vector_ge.cpp
	$(UNIX_TO_DOS) action.h ../WINDOWS/action.h
	mv $(CPP_PREFIX)* ../WINDOWS

	

