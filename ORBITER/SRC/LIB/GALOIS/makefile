DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm
CPP_PREFIX=a_
UNIX_TO_DOS = ../../../../ORBITER2/SRC2/TOOLS/unix2dos.out

MODULES = \
	INT_matrix.o \
	INT_vector.o \
	a_domain.o \
	andre_construction.o \
	andre_construction_line_element.o \
	andre_construction_point_element.o \
	brick_domain.o \
	buekenhout_metz.o \
	classify.o \
	classify_bitvectors.o \
	clique_finder.o \
	colored_graph.o \
	combinatorics.o \
	data.o \
	data_file.o \
	decomposition.o \
	desarguesian_spread.o \
	diophant.o \
	dlx.o \
	draw.o \
	eckardt_point.o \
	fancy_set.o \
	file_output.o \
	finite_ring.o \
	finite_field.o \
	finite_field_tables.o \
	finite_field_linear_algebra.o \
	finite_field_representations.o \
	flag.o \
	galois_global.o \
	generators_symplectic_group.o \
	geometric_object.o \
	geometric_operations.o \
	gl_classes.o \
	graph_layer.o \
	graph_node.o \
	grassmann.o \
	grassmann_embedded.o \
	group_generators.o \
	heisenberg.o \
	hermitian.o \
	hjelmslev.o \
	homogeneous_polynomial_domain.o \
	incidence_structure.o \
	klein_correspondence.o \
	knarr.o \
	layered_graph.o \
	layered_graph_draw_options.o \
	longinteger_object.o \
	longinteger_domain.o \
	magma_interface.o \
	mckay.o \
	memory.o \
	memory_object.o \
	mindist.o \
	mp_graphics.o \
	nauty_interface.o \
	nauty.o \
	nautil.o \
	naugraph.o \
	norm_tables.o \
	null_polarity_generator.o \
	number_theory.o \
	object_in_projective_space.o \
	orthogonal.o \
	orthogonal_points.o \
	partitionstack.o \
	plot.o \
	projective.o \
	projective_space.o \
	rainbow_cliques.o \
	rank_checker.o \
	set_of_sets.o \
	sorting.o \
	spreadsheet.o \
	subfield_structure.o \
	super_fast_hash.o \
	surface.o \
	surface_object.o \
	tensor.o \
	tree.o \
	tree_node.o \
	unipoly.o \
	unusual.o \
	util.o \
	vector_hashing.o \
	w3q.o


LIB = libgalois.a

libgalois.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



cpp:
	$(UNIX_TO_DOS) INT_matrix.C $(CPP_PREFIX)INT_matrix.cpp
	$(UNIX_TO_DOS) INT_vector.C $(CPP_PREFIX)INT_vector.cpp
	$(UNIX_TO_DOS) a_domain.C $(CPP_PREFIX)a_domain.cpp
	$(UNIX_TO_DOS) andre_construction.C $(CPP_PREFIX)andre_construction.cpp
	$(UNIX_TO_DOS) andre_construction_line_element.C $(CPP_PREFIX)andre_construction_line_element.cpp
	$(UNIX_TO_DOS) andre_construction_point_element.C $(CPP_PREFIX)andre_construction_point_element.cpp
	$(UNIX_TO_DOS) brick_domain.C $(CPP_PREFIX)brick_domain.cpp
	$(UNIX_TO_DOS) buekenhout_metz.C $(CPP_PREFIX)buekenhout_metz.cpp
	$(UNIX_TO_DOS) classify.C $(CPP_PREFIX)classify.cpp
	$(UNIX_TO_DOS) classify_bitvectors.C $(CPP_PREFIX)classify_bitvectors.cpp
	$(UNIX_TO_DOS) clique_finder.C $(CPP_PREFIX)clique_finder.cpp
	$(UNIX_TO_DOS) colored_graph.C $(CPP_PREFIX)colored_graph.cpp
	$(UNIX_TO_DOS) combinatorics.C $(CPP_PREFIX)combinatorics.cpp
	$(UNIX_TO_DOS) data.C $(CPP_PREFIX)data.cpp
	$(UNIX_TO_DOS) data_file.C $(CPP_PREFIX)data_file.cpp
	$(UNIX_TO_DOS) decomposition.C $(CPP_PREFIX)decomposition.cpp
	$(UNIX_TO_DOS) desarguesian_spread.C $(CPP_PREFIX)desarguesian_spread.cpp
	$(UNIX_TO_DOS) diophant.C $(CPP_PREFIX)diophant.cpp
	$(UNIX_TO_DOS) dlx.C $(CPP_PREFIX)dlx.cpp
	$(UNIX_TO_DOS) draw.C $(CPP_PREFIX)draw.cpp
	$(UNIX_TO_DOS) eckardt_point.C $(CPP_PREFIX)eckardt_point.cpp
	$(UNIX_TO_DOS) fancy_set.C $(CPP_PREFIX)fancy_set.cpp
	$(UNIX_TO_DOS) file_output.C $(CPP_PREFIX)file_output.cpp
	$(UNIX_TO_DOS) finite_field.C $(CPP_PREFIX)finite_field.cpp
	$(UNIX_TO_DOS) finite_field_linear_algebra.C $(CPP_PREFIX)finite_field_linear_algebra.cpp
	$(UNIX_TO_DOS) finite_field_representations.C $(CPP_PREFIX)finite_field_representations.cpp
	$(UNIX_TO_DOS) finite_field_tables.C $(CPP_PREFIX)finite_field_tables.cpp
	$(UNIX_TO_DOS) finite_ring.C $(CPP_PREFIX)finite_ring.cpp
	$(UNIX_TO_DOS) flag.C $(CPP_PREFIX)flag.cpp
	$(UNIX_TO_DOS) galois_global.C $(CPP_PREFIX)galois_global.cpp
	$(UNIX_TO_DOS) generators_symplectic_group.C $(CPP_PREFIX)generators_symplectic_group.cpp
	$(UNIX_TO_DOS) geometric_object.C $(CPP_PREFIX)geometric_object.cpp
	$(UNIX_TO_DOS) geometric_operations.C $(CPP_PREFIX)geometric_operations.cpp
	$(UNIX_TO_DOS) gl_classes.C $(CPP_PREFIX)gl_classes.cpp
	$(UNIX_TO_DOS) graph_layer.C $(CPP_PREFIX)graph_layer.cpp
	$(UNIX_TO_DOS) graph_node.C $(CPP_PREFIX)graph_node.cpp
	$(UNIX_TO_DOS) grassmann.C $(CPP_PREFIX)grassmann.cpp
	$(UNIX_TO_DOS) grassmann_embedded.C $(CPP_PREFIX)grassmann_embedded.cpp
	$(UNIX_TO_DOS) heisenberg.C $(CPP_PREFIX)heisenberg.cpp
	$(UNIX_TO_DOS) hermitian.C $(CPP_PREFIX)hermitian.cpp
	$(UNIX_TO_DOS) hjelmslev.C $(CPP_PREFIX)hjelmslev.cpp
	$(UNIX_TO_DOS) homogeneous_polynomial_domain.C $(CPP_PREFIX)homogeneous_polynomial_domain.cpp
	$(UNIX_TO_DOS) incidence_structure.C $(CPP_PREFIX)incidence_structure.cpp
	$(UNIX_TO_DOS) klein_correspondence.C $(CPP_PREFIX)klein_correspondence.cpp
	$(UNIX_TO_DOS) knarr.C $(CPP_PREFIX)knarr.cpp
	$(UNIX_TO_DOS) layered_graph.C $(CPP_PREFIX)layered_graph.cpp
	$(UNIX_TO_DOS) layered_graph_draw_options.C $(CPP_PREFIX)layered_graph_draw_options.cpp
	$(UNIX_TO_DOS) longinteger_domain.C $(CPP_PREFIX)longinteger_domain.cpp
	$(UNIX_TO_DOS) longinteger_object.C $(CPP_PREFIX)longinteger_object.cpp
	$(UNIX_TO_DOS) magma_interface.C $(CPP_PREFIX)magma_interface.cpp
	$(UNIX_TO_DOS) mckay.C $(CPP_PREFIX)mckay.cpp
	$(UNIX_TO_DOS) memory.C $(CPP_PREFIX)memory.cpp
	$(UNIX_TO_DOS) memory_object.C $(CPP_PREFIX)memory_object.cpp
	$(UNIX_TO_DOS) mindist.C $(CPP_PREFIX)mindist.cpp
	$(UNIX_TO_DOS) mp_graphics.C $(CPP_PREFIX)mp_graphics.cpp
	$(UNIX_TO_DOS) nauty_interface.C $(CPP_PREFIX)nauty_interface.cpp
	$(UNIX_TO_DOS) norm_tables.C $(CPP_PREFIX)norm_tables.cpp
	$(UNIX_TO_DOS) null_polarity_generator.C $(CPP_PREFIX)null_polarity_generator.cpp
	$(UNIX_TO_DOS) number_theory.C $(CPP_PREFIX)number_theory.cpp
	$(UNIX_TO_DOS) object_in_projective_space.C $(CPP_PREFIX)object_in_projective_space.cpp
	$(UNIX_TO_DOS) orthogonal.C $(CPP_PREFIX)orthogonal.cpp
	$(UNIX_TO_DOS) orthogonal_points.C $(CPP_PREFIX)orthogonal_points.cpp
	$(UNIX_TO_DOS) partitionstack.C $(CPP_PREFIX)partitionstack.cpp
	$(UNIX_TO_DOS) plot.C $(CPP_PREFIX)plot.cpp
	$(UNIX_TO_DOS) projective.C $(CPP_PREFIX)projective.cpp
	$(UNIX_TO_DOS) projective_space.C $(CPP_PREFIX)projective_space.cpp
	$(UNIX_TO_DOS) rainbow_cliques.C $(CPP_PREFIX)rainbow_cliques.cpp
	$(UNIX_TO_DOS) rank_checker.C $(CPP_PREFIX)rank_checker.cpp
	$(UNIX_TO_DOS) set_of_sets.C $(CPP_PREFIX)set_of_sets.cpp
	$(UNIX_TO_DOS) sorting.C $(CPP_PREFIX)sorting.cpp
	$(UNIX_TO_DOS) spreadsheet.C $(CPP_PREFIX)spreadsheet.cpp
	$(UNIX_TO_DOS) subfield_structure.C $(CPP_PREFIX)subfield_structure.cpp
	$(UNIX_TO_DOS) super_fast_hash.C $(CPP_PREFIX)super_fast_hash.cpp
	$(UNIX_TO_DOS) surface.C $(CPP_PREFIX)surface.cpp
	$(UNIX_TO_DOS) surface_object.C $(CPP_PREFIX)surface_object.cpp
	$(UNIX_TO_DOS) tensor.C $(CPP_PREFIX)tensor.cpp
	$(UNIX_TO_DOS) tree.C $(CPP_PREFIX)tree.cpp
	$(UNIX_TO_DOS) tree_node.C $(CPP_PREFIX)tree_node.cpp
	$(UNIX_TO_DOS) unipoly.C $(CPP_PREFIX)unipoly.cpp
	$(UNIX_TO_DOS) unusual.C $(CPP_PREFIX)unusual.cpp
	$(UNIX_TO_DOS) util.C $(CPP_PREFIX)util.cpp
	$(UNIX_TO_DOS) vector_hashing.C $(CPP_PREFIX)vector_hashing.cpp
	$(UNIX_TO_DOS) w3q.C $(CPP_PREFIX)w3q.cpp
	$(UNIX_TO_DOS) galois.h ../WINDOWS/galois.h
	mv *.cpp ../WINDOWS

	#$(UNIX_TO_DOS) nauty.h ../WINDOWS/nauty.h
	#$(UNIX_TO_DOS) nauty.c ../WINDOWS/$(CPP_PREFIX)nauty.c
	#$(UNIX_TO_DOS) nautil.c ../WINDOWS/$(CPP_PREFIX)nautil.c
	#$(UNIX_TO_DOS) naugraph.c ../WINDOWS/$(CPP_PREFIX)naugraph.c


