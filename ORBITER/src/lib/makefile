DISCRETA_CPP = g++
CPPFLAGS = -Wall
LIB = 
LFLAGS = -lm


all: foundations groups_and_group_actions poset_classification discreta toplevel liborbiter.a


foundations : foundations2 
groups_and_group_actions : groups_and_group_actions2 
poset_classification : poset_classification2 
discreta : discreta2 
toplevel : toplevel2 

foundations2 : ; cd foundations; $(MAKE)
groups_and_group_actions2 : ; cd groups_and_group_actions; $(MAKE)
poset_classification2 : ; cd poset_classification; $(MAKE)
discreta2 : ; cd DISCRETA; $(MAKE)
toplevel2 : ; cd top_level; $(MAKE)


clean:
	cd foundations; $(MAKE) clean
	cd groups_and_group_actions; $(MAKE) clean
	cd poset_classification; $(MAKE) clean
	cd DISCRETA; $(MAKE) clean
	cd top_level; $(MAKE) clean
	- rm *.o *.a 




MODULES= \
		foundations/algebra_and_number_theory/a_domain.o \
		foundations/algebra_and_number_theory/finite_field.o \
		foundations/algebra_and_number_theory/finite_field_linear_algebra.o \
		foundations/algebra_and_number_theory/finite_field_projective.o \
		foundations/algebra_and_number_theory/finite_field_representations.o \
		foundations/algebra_and_number_theory/finite_field_tables.o \
		foundations/algebra_and_number_theory/finite_ring.o \
		foundations/algebra_and_number_theory/generators_symplectic_group.o \
		foundations/algebra_and_number_theory/gl_classes.o \
		foundations/algebra_and_number_theory/group_generators.o \
		foundations/algebra_and_number_theory/heisenberg.o \
		foundations/algebra_and_number_theory/homogeneous_polynomial_domain.o \
		foundations/algebra_and_number_theory/longinteger_domain.o \
		foundations/algebra_and_number_theory/longinteger_object.o \
		foundations/algebra_and_number_theory/norm_tables.o \
		foundations/algebra_and_number_theory/null_polarity_generator.o \
		foundations/algebra_and_number_theory/number_theory.o \
		foundations/algebra_and_number_theory/rank_checker.o \
		foundations/algebra_and_number_theory/subfield_structure.o \
		foundations/algebra_and_number_theory/unipoly_domain.o \
		foundations/coding_theory/mindist.o \
		foundations/coding_theory/tensor.o \
		foundations/combinatorics/brick_domain.o \
		foundations/combinatorics/classify_bitvectors.o \
		foundations/combinatorics/combinatorics.o \
		foundations/data_structures/int_matrix.o \
		foundations/data_structures/int_vector.o \
		foundations/data_structures/data_file.o \
		foundations/data_structures/fancy_set.o \
		foundations/data_structures/partitionstack.o \
		foundations/data_structures/set_of_sets.o \
		foundations/data_structures/sorting.o \
		foundations/data_structures/spreadsheet.o \
		foundations/data_structures/super_fast_hash.o \
		foundations/data_structures/vector_hashing.o \
		foundations/geometry/andre_construction.o \
		foundations/geometry/andre_construction_line_element.o \
		foundations/geometry/andre_construction_point_element.o \
		foundations/geometry/buekenhout_metz.o \
		foundations/geometry/data.o \
		foundations/geometry/decomposition.o \
		foundations/geometry/desarguesian_spread.o \
		foundations/geometry/eckardt_point_info.o \
		foundations/geometry/eckardt_point.o \
		foundations/geometry/flag.o \
		foundations/geometry/geo_parameter.o \
		foundations/geometry/geometric_object.o \
		foundations/geometry/geometric_operations.o \
		foundations/geometry/grassmann.o \
		foundations/geometry/grassmann_embedded.o \
		foundations/geometry/hermitian.o \
		foundations/geometry/hjelmslev.o \
		foundations/geometry/inc_gen_global.o \
		foundations/geometry/incidence_structure.o \
		foundations/geometry/klein_correspondence.o \
		foundations/geometry/knarr.o \
		foundations/geometry/object_in_projective_space.o \
		foundations/geometry/orthogonal.o \
		foundations/geometry/orthogonal_points.o \
		foundations/geometry/packing.o \
		foundations/geometry/point_line.o \
		foundations/geometry/projective_space.o \
		foundations/geometry/surface.o \
		foundations/geometry/surface_object.o \
		foundations/geometry/tdo_data.o \
		foundations/geometry/tdo_scheme.o \
		foundations/geometry/unusual.o \
		foundations/geometry/w3q.o \
		foundations/globals/galois_global.o \
		foundations/globals/magma_interface.o \
		foundations/globals/numerics.o \
		foundations/graph_theory/clique_finder.o \
		foundations/graph_theory/clique_finder_control.o \
		foundations/graph_theory/colored_graph.o \
		foundations/graph_theory/graph_layer.o \
		foundations/graph_theory/graph_node.o \
		foundations/graph_theory/layered_graph.o \
		foundations/graph_theory/layered_graph_draw_options.o \
		foundations/graph_theory/rainbow_cliques.o \
		foundations/graph_theory_nauty/naugraph.o \
		foundations/graph_theory_nauty/naurng.o \
		foundations/graph_theory_nauty/nautil.o \
		foundations/graph_theory_nauty/nauty.o \
		foundations/graph_theory_nauty/nauty_interface.o \
		foundations/graph_theory_nauty/schreier.o \
		foundations/graphics/draw.o \
		foundations/graphics/mp_graphics.o \
		foundations/graphics/plot.o \
		foundations/graphics/scene.o \
		foundations/graphics/tree.o \
		foundations/graphics/tree_node.o \
		foundations/io_and_os/file_output.o \
		foundations/io_and_os/memory.o \
		foundations/io_and_os/memory_object.o \
		foundations/io_and_os/orbiter_data_file.o \
		foundations/io_and_os/util.o \
		foundations/solvers/diophant.o \
		foundations/solvers/dlx.o \
		foundations/solvers/mckay.o \
		foundations/statistics/classify.o \
		groups_and_group_actions/data_structures/group.o \
		groups_and_group_actions/data_structures/orbit_transversal.o \
		groups_and_group_actions/data_structures/page_storage.o \
		groups_and_group_actions/data_structures/poset.o \
		groups_and_group_actions/data_structures/poset_description.o \
		groups_and_group_actions/data_structures/projective_space_with_action.o \
		groups_and_group_actions/data_structures/schreier_vector_handler.o \
		groups_and_group_actions/data_structures/schreier_vector.o \
		groups_and_group_actions/data_structures/set_and_stabilizer.o \
		groups_and_group_actions/data_structures/union_find.o \
		groups_and_group_actions/data_structures/union_find_on_k_subsets.o \
		groups_and_group_actions/data_structures/vector_ge.o \
		groups_and_group_actions/group_actions/action.o \
		groups_and_group_actions/group_actions/action_cb.o \
		groups_and_group_actions/group_actions/action_global.o \
		groups_and_group_actions/group_actions/action_indexing_cosets.o \
		groups_and_group_actions/group_actions/action_induce.o \
		groups_and_group_actions/group_actions/action_init.o \
		groups_and_group_actions/group_actions/backtrack.o \
		groups_and_group_actions/group_actions/interface.o \
		groups_and_group_actions/group_actions/interface_direct_product.o \
		groups_and_group_actions/group_actions/interface_matrix_group.o \
		groups_and_group_actions/group_actions/interface_perm_group.o \
		groups_and_group_actions/group_actions/interface_wreath_product.o \
		groups_and_group_actions/group_theory/direct_product.o \
		groups_and_group_actions/group_theory/linear_group.o \
		groups_and_group_actions/group_theory/linear_group_description.o \
		groups_and_group_actions/group_theory/matrix_group.o \
		groups_and_group_actions/group_theory/perm_group.o \
		groups_and_group_actions/group_theory/schreier.o \
		groups_and_group_actions/group_theory/schreier_io.o \
		groups_and_group_actions/group_theory/schreier_sims.o \
		groups_and_group_actions/group_theory/sims.o \
		groups_and_group_actions/group_theory/sims2.o \
		groups_and_group_actions/group_theory/sims3.o \
		groups_and_group_actions/group_theory/sims_global.o \
		groups_and_group_actions/group_theory/strong_generators.o \
		groups_and_group_actions/group_theory/strong_generators_groups.o \
		groups_and_group_actions/group_theory/subgroup.o \
		groups_and_group_actions/group_theory/wreath_product.o \
		groups_and_group_actions/induced_actions/action_by_conjugation.o \
		groups_and_group_actions/induced_actions/action_by_representation.o \
		groups_and_group_actions/induced_actions/action_by_restriction.o \
		groups_and_group_actions/induced_actions/action_by_right_multiplication.o \
		groups_and_group_actions/induced_actions/action_by_subfield_structure.o \
		groups_and_group_actions/induced_actions/action_on_andre.o \
		groups_and_group_actions/induced_actions/action_on_bricks.o \
		groups_and_group_actions/induced_actions/action_on_cosets.o \
		groups_and_group_actions/induced_actions/action_on_determinant.o \
		groups_and_group_actions/induced_actions/action_on_factor_space.o \
		groups_and_group_actions/induced_actions/action_on_flags.o \
		groups_and_group_actions/induced_actions/action_on_grassmannian.o \
		groups_and_group_actions/induced_actions/action_on_homogeneous_polynomials.o \
		groups_and_group_actions/induced_actions/action_on_k_subsets.o \
		groups_and_group_actions/induced_actions/action_on_orbits.o \
		groups_and_group_actions/induced_actions/action_on_orthogonal.o \
		groups_and_group_actions/induced_actions/action_on_sets.o \
		groups_and_group_actions/induced_actions/action_on_sign.o \
		groups_and_group_actions/induced_actions/action_on_spread_set.o \
		groups_and_group_actions/induced_actions/action_on_subgroups.o \
		groups_and_group_actions/induced_actions/action_on_wedge_product.o \
		groups_and_group_actions/induced_actions/product_action.o \
		poset_classification/classify/classification.o \
		poset_classification/classify/flag_orbit_node.o \
		poset_classification/classify/flag_orbits.o \
		poset_classification/classify/orbit_node.o \
		poset_classification/other/snakes_and_ladders_global.o \
		poset_classification/snakes_and_ladders/extension.o \
		poset_classification/snakes_and_ladders/poset_classification.o \
		poset_classification/snakes_and_ladders/poset_classification_classify.o \
		poset_classification/snakes_and_ladders/poset_classification_combinatorics.o \
		poset_classification/snakes_and_ladders/poset_classification_draw.o \
		poset_classification/snakes_and_ladders/poset_classification_init.o \
		poset_classification/snakes_and_ladders/poset_classification_io.o \
		poset_classification/snakes_and_ladders/poset_classification_recognize.o \
		poset_classification/snakes_and_ladders/poset_classification_trace.o \
		poset_classification/snakes_and_ladders/poset_orbit_node.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_downstep.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_downstep_subspace_action.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_group_theory.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_io.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_upstep.o \
		poset_classification/snakes_and_ladders/poset_orbit_node_upstep_subspace_action.o \
		poset_classification/snakes_and_ladders/upstep_work.o \
		poset_classification/snakes_and_ladders/upstep_work_subspace_action.o \
		poset_classification/snakes_and_ladders/upstep_work_trace.o \
		poset_classification/set_stabilizer/compute_stabilizer.o \
		poset_classification/set_stabilizer/set_stabilizer_compute.o \
		DISCRETA/base.o \
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
		top_level/algebra_and_number_theory/analyze_group.o \
		top_level/algebra_and_number_theory/elliptic_curve.o \
		top_level/algebra_and_number_theory/extra.o \
		top_level/algebra_and_number_theory/factor_group.o \
		top_level/algebra_and_number_theory/top_level_global.o \
		top_level/algebra_and_number_theory/young.o \
		top_level/geometry/BLT_set_create.o \
		top_level/geometry/BLT_set_create_description.o \
		top_level/geometry/arc_generator.o \
		top_level/geometry/arc_lifting.o \
		top_level/geometry/choose_points_or_lines.o \
		top_level/geometry/classify_double_sixes.o \
		top_level/geometry/classify_trihedral_pairs.o \
		top_level/geometry/decomposition.o \
		top_level/geometry/incidence_structure.o \
		top_level/geometry/k_arc_generator.o \
		top_level/geometry/object_in_projective_space_with_action.o \
		top_level/geometry/polar.o \
		top_level/geometry/projective_space.o \
		top_level/geometry/recoordinatize.o \
		top_level/geometry/search_blocking_set.o \
		top_level/geometry/singer_cycle.o \
		top_level/geometry/six_arcs_not_on_a_conic.o \
		top_level/geometry/spread.o \
		top_level/geometry/spread2.o \
		top_level/geometry/spread_create.o \
		top_level/geometry/spread_create_description.o \
		top_level/geometry/spread_lifting.o \
		top_level/geometry/surface_classify_wedge.o \
		top_level/geometry/surface_create.o \
		top_level/geometry/surface_create_description.o \
		top_level/geometry/surface_object_with_action.o \
		top_level/geometry/surface_with_action.o \
		top_level/geometry/translation_plane_via_andre_model.o \
		top_level/isomorph/isomorph.o \
		top_level/isomorph/isomorph_arguments.o \
		top_level/isomorph/isomorph_database.o \
		top_level/isomorph/isomorph_files.o \
		top_level/isomorph/isomorph_global.o \
		top_level/isomorph/isomorph_testing.o \
		top_level/isomorph/isomorph_trace.o \
		top_level/isomorph/representatives.o \
		top_level/orbits/orbit_of_equations.o \
		top_level/orbits/orbit_of_sets.o \
		top_level/orbits/orbit_of_subspaces.o \
		top_level/orbits/orbit_rep.o \
		top_level/orbits/subspace_orbits.o \
		top_level/solver/exact_cover.o \
		top_level/solver/exact_cover_arguments.o


liborbiter.a:  $(MODULES) ;
	ar ruvs liborbiter.a $(MODULES)

