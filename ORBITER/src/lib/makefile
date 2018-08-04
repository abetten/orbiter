DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall
LIB = 
LFLAGS = -lm


all: galois action snakesandladders discreta toplevel liborbiter.a


galois : galois2 ; 
action : action2 ; 
snakesandladders : snakesandladders2 ; 
discreta : discreta2 ; 
toplevel : toplevel2 ; 

galois2 : ; cd GALOIS; $(MAKE)
action2 : ; cd ACTION; $(MAKE)
snakesandladders2 : ; cd SNAKES_AND_LADDERS; $(MAKE)
discreta2 : ; cd DISCRETA; $(MAKE)
toplevel2 : ; cd TOP_LEVEL; $(MAKE)


clean:
	cd GALOIS; $(MAKE) clean
	cd ACTION; $(MAKE) clean
	cd SNAKES_AND_LADDERS; $(MAKE) clean
	cd DISCRETA; $(MAKE) clean
	cd TOP_LEVEL; $(MAKE) clean
	- rm *.o *.a 




MODULES= \
		GALOIS/algebra_and_number_theory/a_domain.o \
		GALOIS/algebra_and_number_theory/finite_field.o \
		GALOIS/algebra_and_number_theory/finite_field_linear_algebra.o \
		GALOIS/algebra_and_number_theory/finite_field_representations.o \
		GALOIS/algebra_and_number_theory/finite_field_tables.o \
		GALOIS/algebra_and_number_theory/finite_ring.o \
		GALOIS/algebra_and_number_theory/generators_symplectic_group.o \
		GALOIS/algebra_and_number_theory/gl_classes.o \
		GALOIS/algebra_and_number_theory/group_generators.o \
		GALOIS/algebra_and_number_theory/heisenberg.o \
		GALOIS/algebra_and_number_theory/homogeneous_polynomial_domain.o \
		GALOIS/algebra_and_number_theory/longinteger_domain.o \
		GALOIS/algebra_and_number_theory/longinteger_object.o \
		GALOIS/algebra_and_number_theory/norm_tables.o \
		GALOIS/algebra_and_number_theory/null_polarity_generator.o \
		GALOIS/algebra_and_number_theory/number_theory.o \
		GALOIS/algebra_and_number_theory/rank_checker.o \
		GALOIS/algebra_and_number_theory/subfield_structure.o \
		GALOIS/algebra_and_number_theory/unipoly_domain.o \
		GALOIS/coding_theory/mindist.o \
		GALOIS/coding_theory/tensor.o \
		GALOIS/combinatorics/brick_domain.o \
		GALOIS/combinatorics/classify_bitvectors.o \
		GALOIS/combinatorics/combinatorics.o \
		GALOIS/data_structures/INT_matrix.o \
		GALOIS/data_structures/INT_vector.o \
		GALOIS/data_structures/data_file.o \
		GALOIS/data_structures/fancy_set.o \
		GALOIS/data_structures/partitionstack.o \
		GALOIS/data_structures/set_of_sets.o \
		GALOIS/data_structures/sorting.o \
		GALOIS/data_structures/spreadsheet.o \
		GALOIS/data_structures/super_fast_hash.o \
		GALOIS/data_structures/vector_hashing.o \
		GALOIS/geometry/andre_construction.o \
		GALOIS/geometry/andre_construction_line_element.o \
		GALOIS/geometry/andre_construction_point_element.o \
		GALOIS/geometry/buekenhout_metz.o \
		GALOIS/geometry/data.o \
		GALOIS/geometry/decomposition.o \
		GALOIS/geometry/desarguesian_spread.o \
		GALOIS/geometry/eckardt_point.o \
		GALOIS/geometry/flag.o \
		GALOIS/geometry/geo_parameter.o \
		GALOIS/geometry/geometric_object.o \
		GALOIS/geometry/geometric_operations.o \
		GALOIS/geometry/grassmann.o \
		GALOIS/geometry/grassmann_embedded.o \
		GALOIS/geometry/hermitian.o \
		GALOIS/geometry/hjelmslev.o \
		GALOIS/geometry/inc_gen_global.o \
		GALOIS/geometry/incidence_structure.o \
		GALOIS/geometry/klein_correspondence.o \
		GALOIS/geometry/knarr.o \
		GALOIS/geometry/object_in_projective_space.o \
		GALOIS/geometry/orthogonal.o \
		GALOIS/geometry/orthogonal_points.o \
		GALOIS/geometry/packing.o \
		GALOIS/geometry/point_line.o \
		GALOIS/geometry/projective.o \
		GALOIS/geometry/projective_space.o \
		GALOIS/geometry/surface.o \
		GALOIS/geometry/surface_object.o \
		GALOIS/geometry/tdo_data.o \
		GALOIS/geometry/tdo_scheme.o \
		GALOIS/geometry/unusual.o \
		GALOIS/geometry/w3q.o \
		GALOIS/globals/galois_global.o \
		GALOIS/globals/magma_interface.o \
		GALOIS/globals/numerics.o \
		GALOIS/graph_theory/clique_finder.o \
		GALOIS/graph_theory/colored_graph.o \
		GALOIS/graph_theory/graph_layer.o \
		GALOIS/graph_theory/graph_node.o \
		GALOIS/graph_theory/layered_graph.o \
		GALOIS/graph_theory/layered_graph_draw_options.o \
		GALOIS/graph_theory/rainbow_cliques.o \
		GALOIS/graph_theory_nauty/naugraph.o \
		GALOIS/graph_theory_nauty/naurng.o \
		GALOIS/graph_theory_nauty/nautil.o \
		GALOIS/graph_theory_nauty/nauty.o \
		GALOIS/graph_theory_nauty/nauty_interface.o \
		GALOIS/graph_theory_nauty/schreier.o \
		GALOIS/graphics/draw.o \
		GALOIS/graphics/mp_graphics.o \
		GALOIS/graphics/plot.o \
		GALOIS/graphics/scene.o \
		GALOIS/graphics/tree.o \
		GALOIS/graphics/tree_node.o \
		GALOIS/io_and_os/dynamic_memory.o \
		GALOIS/io_and_os/file_output.o \
		GALOIS/io_and_os/memory.o \
		GALOIS/io_and_os/memory_object.o \
		GALOIS/io_and_os/orbiter_data_file.o \
		GALOIS/io_and_os/util.o \
		GALOIS/solvers/diophant.o \
		GALOIS/solvers/dlx.o \
		GALOIS/solvers/mckay.o \
		GALOIS/statistics/classify.o \
		ACTION/data_structures/group.o \
		ACTION/data_structures/orbit_transversal.o \
		ACTION/data_structures/page_storage.o \
		ACTION/data_structures/projective_space_with_action.o \
		ACTION/data_structures/schreier_vector.o \
		ACTION/data_structures/set_and_stabilizer.o \
		ACTION/data_structures/union_find.o \
		ACTION/data_structures/union_find_on_k_subsets.o \
		ACTION/data_structures/vector_ge.o \
		ACTION/group_actions/action.o \
		ACTION/group_actions/action_cb.o \
		ACTION/group_actions/action_global.o \
		ACTION/group_actions/action_indexing_cosets.o \
		ACTION/group_actions/action_induce.o \
		ACTION/group_actions/action_init.o \
		ACTION/group_actions/backtrack.o \
		ACTION/group_actions/interface.o \
		ACTION/group_actions/interface_matrix_group.o \
		ACTION/group_actions/interface_perm_group.o \
		ACTION/group_actions/interface_wreath_product.o \
		ACTION/group_theory/linear_group.o \
		ACTION/group_theory/linear_group_description.o \
		ACTION/group_theory/matrix_group.o \
		ACTION/group_theory/perm_group.o \
		ACTION/group_theory/schreier.o \
		ACTION/group_theory/schreier_sims.o \
		ACTION/group_theory/sims.o \
		ACTION/group_theory/sims2.o \
		ACTION/group_theory/sims3.o \
		ACTION/group_theory/sims_global.o \
		ACTION/group_theory/strong_generators.o \
		ACTION/group_theory/strong_generators_groups.o \
		ACTION/group_theory/subgroup.o \
		ACTION/group_theory/wreath_product.o \
		ACTION/induced_actions/action_by_conjugation.o \
		ACTION/induced_actions/action_by_representation.o \
		ACTION/induced_actions/action_by_restriction.o \
		ACTION/induced_actions/action_by_right_multiplication.o \
		ACTION/induced_actions/action_by_subfield_structure.o \
		ACTION/induced_actions/action_on_andre.o \
		ACTION/induced_actions/action_on_bricks.o \
		ACTION/induced_actions/action_on_cosets.o \
		ACTION/induced_actions/action_on_determinant.o \
		ACTION/induced_actions/action_on_factor_space.o \
		ACTION/induced_actions/action_on_flags.o \
		ACTION/induced_actions/action_on_grassmannian.o \
		ACTION/induced_actions/action_on_homogeneous_polynomials.o \
		ACTION/induced_actions/action_on_k_subsets.o \
		ACTION/induced_actions/action_on_orbits.o \
		ACTION/induced_actions/action_on_orthogonal.o \
		ACTION/induced_actions/action_on_sets.o \
		ACTION/induced_actions/action_on_sign.o \
		ACTION/induced_actions/action_on_spread_set.o \
		ACTION/induced_actions/action_on_subgroups.o \
		ACTION/induced_actions/action_on_wedge_product.o \
		ACTION/induced_actions/product_action.o \
		SNAKES_AND_LADDERS/classification.o \
		SNAKES_AND_LADDERS/compute_stabilizer.o \
		SNAKES_AND_LADDERS/extension.o \
		SNAKES_AND_LADDERS/flag_orbit_node.o \
		SNAKES_AND_LADDERS/flag_orbits.o \
		SNAKES_AND_LADDERS/generator.o \
		SNAKES_AND_LADDERS/generator_classify.o \
		SNAKES_AND_LADDERS/generator_combinatorics.o \
		SNAKES_AND_LADDERS/generator_draw.o \
		SNAKES_AND_LADDERS/generator_init.o \
		SNAKES_AND_LADDERS/generator_io.o \
		SNAKES_AND_LADDERS/generator_trace.o \
		SNAKES_AND_LADDERS/oracle.o \
		SNAKES_AND_LADDERS/oracle_downstep.o \
		SNAKES_AND_LADDERS/oracle_downstep_subspace_action.o \
		SNAKES_AND_LADDERS/oracle_io.o \
		SNAKES_AND_LADDERS/oracle_upstep.o \
		SNAKES_AND_LADDERS/oracle_upstep_subspace_action.o \
		SNAKES_AND_LADDERS/orbit_node.o \
		SNAKES_AND_LADDERS/recognize.o \
		SNAKES_AND_LADDERS/set_stabilizer_compute.o \
		SNAKES_AND_LADDERS/snakes_and_ladders_global.o \
		SNAKES_AND_LADDERS/upstep_work.o \
		SNAKES_AND_LADDERS/upstep_work_subspace_action.o \
		SNAKES_AND_LADDERS/upstep_work_trace.o \
		DISCRETA/base.o \
		DISCRETA/bitmatrix.o \
		DISCRETA/bt_key.o \
		DISCRETA/btree.o \
		DISCRETA/database.o \
		DISCRETA/design.o \
		DISCRETA/design_parameter.o \
		DISCRETA/design_parameter_source.o \
		DISCRETA/discreta_global.o \
		DISCRETA/domain.o \
		DISCRETA/geometry.o \
		DISCRETA/global.o \
		DISCRETA/group_selection.o \
		DISCRETA/hollerith.o \
		DISCRETA/integer.o \
		DISCRETA/longinteger.o \
		DISCRETA/matrix.o \
		DISCRETA/memory.o \
		DISCRETA/number_partition.o \
		DISCRETA/page_table.o \
		DISCRETA/perm_group_gens.o \
		DISCRETA/permutation.o \
		DISCRETA/solid.o \
		DISCRETA/unipoly.o \
		DISCRETA/vector.o \
		TOP_LEVEL/algebra_and_number_theory/analyze_group.o \
		TOP_LEVEL/algebra_and_number_theory/elliptic_curve.o \
		TOP_LEVEL/algebra_and_number_theory/extra.o \
		TOP_LEVEL/algebra_and_number_theory/factor_group.o \
		TOP_LEVEL/algebra_and_number_theory/top_level_global.o \
		TOP_LEVEL/algebra_and_number_theory/young.o \
		TOP_LEVEL/geometry/BLT_set_create.o \
		TOP_LEVEL/geometry/BLT_set_create_description.o \
		TOP_LEVEL/geometry/arc_generator.o \
		TOP_LEVEL/geometry/arc_lifting.o \
		TOP_LEVEL/geometry/choose_points_or_lines.o \
		TOP_LEVEL/geometry/classify_double_sixes.o \
		TOP_LEVEL/geometry/classify_trihedral_pairs.o \
		TOP_LEVEL/geometry/decomposition.o \
		TOP_LEVEL/geometry/incidence_structure.o \
		TOP_LEVEL/geometry/k_arc_generator.o \
		TOP_LEVEL/geometry/object_in_projective_space_with_action.o \
		TOP_LEVEL/geometry/polar.o \
		TOP_LEVEL/geometry/projective_space.o \
		TOP_LEVEL/geometry/recoordinatize.o \
		TOP_LEVEL/geometry/search_blocking_set.o \
		TOP_LEVEL/geometry/singer_cycle.o \
		TOP_LEVEL/geometry/six_arcs_not_on_a_conic.o \
		TOP_LEVEL/geometry/spread.o \
		TOP_LEVEL/geometry/spread2.o \
		TOP_LEVEL/geometry/spread_create.o \
		TOP_LEVEL/geometry/spread_create_description.o \
		TOP_LEVEL/geometry/spread_lifting.o \
		TOP_LEVEL/geometry/surface_classify_wedge.o \
		TOP_LEVEL/geometry/surface_create.o \
		TOP_LEVEL/geometry/surface_create_description.o \
		TOP_LEVEL/geometry/surface_object_with_action.o \
		TOP_LEVEL/geometry/surface_with_action.o \
		TOP_LEVEL/geometry/translation_plane_via_andre_model.o \
		TOP_LEVEL/isomorph/isomorph.o \
		TOP_LEVEL/isomorph/isomorph_arguments.o \
		TOP_LEVEL/isomorph/isomorph_database.o \
		TOP_LEVEL/isomorph/isomorph_files.o \
		TOP_LEVEL/isomorph/isomorph_global.o \
		TOP_LEVEL/isomorph/isomorph_testing.o \
		TOP_LEVEL/isomorph/isomorph_trace.o \
		TOP_LEVEL/orbits/orbit_of_equations.o \
		TOP_LEVEL/orbits/orbit_of_sets.o \
		TOP_LEVEL/orbits/orbit_of_subspaces.o \
		TOP_LEVEL/orbits/orbit_rep.o \
		TOP_LEVEL/orbits/representatives.o \
		TOP_LEVEL/orbits/subspace_orbits.o \
		TOP_LEVEL/solver/exact_cover.o \
		TOP_LEVEL/solver/exact_cover_arguments.o


liborbiter.a:  $(MODULES) ;
	ar ruvs liborbiter.a $(MODULES)

