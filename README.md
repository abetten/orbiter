Orbiter
=======

A C++ class library devoted to the classification of combinatorial objects.


Please find the user's guide here:


https://www.math.colostate.edu/~betten/orbiter/users_guide.pdf


Please find the programmer's guide here:


https://www.math.colostate.edu/~betten/orbiter/orbiter_programmers_guide.pdf



Requirements:
terminal application (for instance SuperShell in Windows, Terminal on Macintosh)


a) if you decide to compile Orbiter and install it locally:
C++ development and standard library
make


b) if you want to run Orbiter through docker:
docker


Optional:
latex, povray


Please see the project 

    abetten/orbiter-boilerplate

for how to use Orbiter as a library in C++ code.


Anton Betten
Sept 26, 2021



Tree structure:
===============

    .
    ├── Dockerfile
    ├── Doxyfile
    ├── LICENSE
    ├── README.md
    ├── bin
    │   └── README
    ├── build_number
    ├── count.sh
    ├── doc
    │   ├── Doxyfile
    │   ├── README
    │   ├── logo_small.png
    │   └── orbiter_text.png
    ├── examples
    │   └── users_guide
    │       └── makefile
    ├── git_update.sh
    ├── lsr.sh
    ├── makefile
    ├── rename_extension.sh
    └── src
        ├── apps
        │   ├── ginac
        │   │   ├── 13_report.tex
        │   │   ├── clebsch.cpp
        │   │   ├── cra.cpp
        │   │   ├── differentiation.cpp
        │   │   ├── ginac_linear_algebra.cpp
        │   │   ├── hilbert_cohn_vossen.cpp
        │   │   ├── linear_system.cpp
        │   │   ├── makefile
        │   │   ├── mersenne.cpp
        │   │   ├── solve_linear.cpp
        │   │   ├── thirteen_eckardt_points.cpp
        │   │   └── tutorial.cpp
        │   ├── makefile
        │   ├── makefile~
        │   ├── orbiter
        │   │   ├── makefile
        │   │   ├── orbiter.cpp
        │   │   └── orbiter.out.dSYM
        │   │       └── Contents
        │   │           ├── Info.plist
        │   │           └── Resources
        │   │               └── DWARF
        │   │                   └── orbiter.out
        │   └── sandbox
        │       ├── makefile
        │       └── sandbox.cpp
        ├── lib
        │   ├── DISCRETA
        │   │   ├── base.cpp
        │   │   ├── bt_key.cpp
        │   │   ├── btree.cpp
        │   │   ├── database.cpp
        │   │   ├── design.cpp
        │   │   ├── design_parameter.cpp
        │   │   ├── design_parameter_source.cpp
        │   │   ├── discreta.h
        │   │   ├── discreta_global.cpp
        │   │   ├── discreta_matrix.cpp
        │   │   ├── domain.cpp
        │   │   ├── global.cpp
        │   │   ├── hollerith.cpp
        │   │   ├── integer.cpp
        │   │   ├── longinteger.cpp
        │   │   ├── makefile
        │   │   ├── memory.cpp
        │   │   ├── number_partition.cpp
        │   │   ├── page_table.cpp
        │   │   ├── permutation.cpp
        │   │   ├── unipoly.cpp
        │   │   └── vector.cpp
        │   ├── classification
        │   │   ├── classification.h
        │   │   ├── classify
        │   │   │   ├── classification_step.cpp
        │   │   │   ├── classify.h
        │   │   │   ├── flag_orbit_node.cpp
        │   │   │   ├── flag_orbits.cpp
        │   │   │   ├── makefile
        │   │   │   └── orbit_node.cpp
        │   │   ├── makefile
        │   │   ├── makefile~
        │   │   ├── poset_classification
        │   │   │   ├── classification_base_case.cpp
        │   │   │   ├── extension.cpp
        │   │   │   ├── makefile
        │   │   │   ├── orbit_based_testing.cpp
        │   │   │   ├── poset_classification.cpp
        │   │   │   ├── poset_classification.h
        │   │   │   ├── poset_classification_classify.cpp
        │   │   │   ├── poset_classification_combinatorics.cpp
        │   │   │   ├── poset_classification_control.cpp
        │   │   │   ├── poset_classification_draw.cpp
        │   │   │   ├── poset_classification_export_source_code.cpp
        │   │   │   ├── poset_classification_init.cpp
        │   │   │   ├── poset_classification_io.cpp
        │   │   │   ├── poset_classification_recognize.cpp
        │   │   │   ├── poset_classification_report.cpp
        │   │   │   ├── poset_classification_report_options.cpp
        │   │   │   ├── poset_classification_trace.cpp
        │   │   │   ├── poset_description.cpp
        │   │   │   ├── poset_of_orbits.cpp
        │   │   │   ├── poset_orbit_node.cpp
        │   │   │   ├── poset_orbit_node_downstep.cpp
        │   │   │   ├── poset_orbit_node_downstep_subspace_action.cpp
        │   │   │   ├── poset_orbit_node_group_theory.cpp
        │   │   │   ├── poset_orbit_node_io.cpp
        │   │   │   ├── poset_orbit_node_upstep.cpp
        │   │   │   ├── poset_orbit_node_upstep_subspace_action.cpp
        │   │   │   ├── poset_with_group_action.cpp
        │   │   │   ├── upstep_work.cpp
        │   │   │   ├── upstep_work_subspace_action.cpp
        │   │   │   └── upstep_work_trace.cpp
        │   │   └── set_stabilizer
        │   │       ├── compute_stabilizer.cpp
        │   │       ├── makefile
        │   │       ├── set_stabilizer.h
        │   │       ├── stabilizer_orbits_and_types.cpp
        │   │       ├── substructure_classifier.cpp
        │   │       └── substructure_stats_and_selection.cpp
        │   ├── foundations
        │   │   ├── BitSet
        │   │   │   ├── ansi_color.h
        │   │   │   ├── bitset.cpp
        │   │   │   ├── bitset.h
        │   │   │   ├── bitset_not_operator_test.h
        │   │   │   ├── bitset_test_values.h
        │   │   │   ├── count.sh
        │   │   │   ├── main.cpp
        │   │   │   ├── main.out
        │   │   │   └── makefile
        │   │   ├── CUDA
        │   │   │   ├── Linalg
        │   │   │   │   ├── Makefile
        │   │   │   │   ├── Matrix.h
        │   │   │   │   ├── drvapi_error_string.h
        │   │   │   │   ├── gpuErrchk.cpp
        │   │   │   │   ├── gpuErrchk.h
        │   │   │   │   ├── gpu_table.h
        │   │   │   │   ├── linalg.h
        │   │   │   │   └── main.cpp
        │   │   │   └── gpuErrchk.h
        │   │   ├── algebra_and_number_theory
        │   │   │   ├── a_domain.cpp
        │   │   │   ├── algebra_and_number_theory.h
        │   │   │   ├── algebra_global.cpp
        │   │   │   ├── cyclotomic_sets.cpp
        │   │   │   ├── finite_ring.cpp
        │   │   │   ├── generators_symplectic_group.cpp
        │   │   │   ├── gl_class_rep.cpp
        │   │   │   ├── gl_classes.cpp
        │   │   │   ├── group_generators_domain.cpp
        │   │   │   ├── heisenberg.cpp
        │   │   │   ├── homogeneous_polynomial_domain.cpp
        │   │   │   ├── longinteger_domain.cpp
        │   │   │   ├── longinteger_object.cpp
        │   │   │   ├── makefile
        │   │   │   ├── matrix_block_data.cpp
        │   │   │   ├── null_polarity_generator.cpp
        │   │   │   ├── number_theoretic_transform.cpp
        │   │   │   ├── number_theory_domain.cpp
        │   │   │   ├── partial_derivative.cpp
        │   │   │   ├── rank_checker.cpp
        │   │   │   ├── table_of_irreducible_polynomials.cpp
        │   │   │   ├── unipoly_domain.cpp
        │   │   │   ├── unipoly_domain2.cpp
        │   │   │   └── vector_space.cpp
        │   │   ├── coding_theory
        │   │   │   ├── coding_theory.h
        │   │   │   ├── coding_theory_domain.cpp
        │   │   │   ├── cyclic_codes.cpp
        │   │   │   ├── makefile
        │   │   │   ├── mindist.cpp
        │   │   │   └── tensor_codes.cpp
        │   │   ├── combinatorics
        │   │   │   ├── boolean_function_domain.cpp
        │   │   │   ├── brick_domain.cpp
        │   │   │   ├── combinatorial_object_activity.cpp
        │   │   │   ├── combinatorial_object_activity_description.cpp
        │   │   │   ├── combinatorial_object_create.cpp
        │   │   │   ├── combinatorial_object_description.cpp
        │   │   │   ├── combinatorics.h
        │   │   │   ├── combinatorics_domain.cpp
        │   │   │   ├── geo_parameter.cpp
        │   │   │   ├── makefile
        │   │   │   ├── pentomino_puzzle.cpp
        │   │   │   ├── tdo_data.cpp
        │   │   │   ├── tdo_refinement.cpp
        │   │   │   ├── tdo_refinement_description.cpp
        │   │   │   └── tdo_scheme_synthetic.cpp
        │   │   ├── cryptography
        │   │   │   ├── cryptography.h
        │   │   │   ├── cryptography_domain.cpp
        │   │   │   └── makefile
        │   │   ├── data_structures
        │   │   │   ├── bitmatrix.cpp
        │   │   │   ├── bitvector.cpp
        │   │   │   ├── classify_bitvectors.cpp
        │   │   │   ├── classify_using_canonical_forms.cpp
        │   │   │   ├── data_file.cpp
        │   │   │   ├── data_input_stream.cpp
        │   │   │   ├── data_structures.h
        │   │   │   ├── data_structures_global.cpp
        │   │   │   ├── encoded_combinatorial_object.cpp
        │   │   │   ├── fancy_set.cpp
        │   │   │   ├── int_matrix.cpp
        │   │   │   ├── int_vec.cpp
        │   │   │   ├── int_vector.cpp
        │   │   │   ├── lint_vec.cpp
        │   │   │   ├── makefile
        │   │   │   ├── nauty_output.cpp
        │   │   │   ├── page_storage.cpp
        │   │   │   ├── partitionstack.cpp
        │   │   │   ├── set_builder.cpp
        │   │   │   ├── set_builder_description.cpp
        │   │   │   ├── set_of_sets.cpp
        │   │   │   ├── set_of_sets_lint.cpp
        │   │   │   ├── sorting.cpp
        │   │   │   ├── spreadsheet.cpp
        │   │   │   ├── string_tools.cpp
        │   │   │   ├── super_fast_hash.cpp
        │   │   │   └── vector_hashing.cpp
        │   │   ├── expression_parser
        │   │   │   ├── expression_parser.cpp
        │   │   │   ├── expression_parser.h
        │   │   │   ├── expression_parser_domain.cpp
        │   │   │   ├── formula.cpp
        │   │   │   ├── lexer.cpp
        │   │   │   ├── makefile
        │   │   │   ├── syntax_tree.cpp
        │   │   │   ├── syntax_tree_node.cpp
        │   │   │   └── syntax_tree_node_terminal.cpp
        │   │   ├── finite_fields
        │   │   │   ├── finite_field.cpp
        │   │   │   ├── finite_field_RREF.cpp
        │   │   │   ├── finite_field_activity.cpp
        │   │   │   ├── finite_field_activity_description.cpp
        │   │   │   ├── finite_field_applications.cpp
        │   │   │   ├── finite_field_description.cpp
        │   │   │   ├── finite_field_implementation_by_tables.cpp
        │   │   │   ├── finite_field_implementation_wo_tables.cpp
        │   │   │   ├── finite_field_io.cpp
        │   │   │   ├── finite_field_linear_algebra.cpp
        │   │   │   ├── finite_field_linear_algebra2.cpp
        │   │   │   ├── finite_field_orthogonal.cpp
        │   │   │   ├── finite_field_projective.cpp
        │   │   │   ├── finite_field_representations.cpp
        │   │   │   ├── finite_field_tables.cpp
        │   │   │   ├── finite_fields.h
        │   │   │   ├── makefile
        │   │   │   ├── norm_tables.cpp
        │   │   │   └── subfield_structure.cpp
        │   │   ├── foundations.h
        │   │   ├── galois.h~
        │   │   ├── geometry
        │   │   │   ├── andre_construction.cpp
        │   │   │   ├── andre_construction_line_element.cpp
        │   │   │   ├── andre_construction_point_element.cpp
        │   │   │   ├── buekenhout_metz.cpp
        │   │   │   ├── cubic_curve.cpp
        │   │   │   ├── decomposition.cpp
        │   │   │   ├── desarguesian_spread.cpp
        │   │   │   ├── elliptic_curve.cpp
        │   │   │   ├── flag.cpp
        │   │   │   ├── geometry.h
        │   │   │   ├── geometry_global.cpp
        │   │   │   ├── grassmann.cpp
        │   │   │   ├── grassmann_embedded.cpp
        │   │   │   ├── hermitian.cpp
        │   │   │   ├── hjelmslev.cpp
        │   │   │   ├── incidence_structure.cpp
        │   │   │   ├── klein_correspondence.cpp
        │   │   │   ├── knarr.cpp
        │   │   │   ├── makefile
        │   │   │   ├── object_in_projective_space.cpp
        │   │   │   ├── point_line.cpp
        │   │   │   ├── points_and_lines.cpp
        │   │   │   ├── projective_space.cpp
        │   │   │   ├── projective_space2.cpp
        │   │   │   ├── quartic_curve_domain.cpp
        │   │   │   ├── quartic_curve_object.cpp
        │   │   │   ├── quartic_curve_object_properties.cpp
        │   │   │   ├── spread_tables.cpp
        │   │   │   └── w3q.cpp
        │   │   ├── geometry_builder
        │   │   │   ├── cperm.cpp
        │   │   │   ├── gen_geo.cpp
        │   │   │   ├── gen_geo_conf.cpp
        │   │   │   ├── geo_frame.cpp
        │   │   │   ├── geo_iso.cpp
        │   │   │   ├── geometry_builder.cpp
        │   │   │   ├── geometry_builder.h
        │   │   │   ├── geometry_builder_description.cpp
        │   │   │   ├── globals.cpp
        │   │   │   ├── grid.cpp
        │   │   │   ├── inc_encoding.cpp
        │   │   │   ├── incidence.cpp
        │   │   │   ├── iso_grid.cpp
        │   │   │   ├── iso_info.cpp
        │   │   │   ├── iso_type.cpp
        │   │   │   ├── makefile
        │   │   │   ├── os.cpp
        │   │   │   ├── tactical_decomposition.cpp
        │   │   │   ├── tdo_gradient.cpp
        │   │   │   └── tdo_scheme.cpp
        │   │   ├── globals
        │   │   │   ├── function_command.cpp
        │   │   │   ├── function_polish.cpp
        │   │   │   ├── function_polish_description.cpp
        │   │   │   ├── globals.h
        │   │   │   ├── magma_interface.cpp
        │   │   │   ├── makefile
        │   │   │   ├── numerics.cpp
        │   │   │   ├── orbiter_session.cpp
        │   │   │   ├── orbiter_symbol_table.cpp
        │   │   │   ├── orbiter_symbol_table_entry.cpp
        │   │   │   ├── polynomial_double.cpp
        │   │   │   └── polynomial_double_domain.cpp
        │   │   ├── graph_theory
        │   │   │   ├── Clique
        │   │   │   │   ├── Graph.h
        │   │   │   │   ├── KClique.cpp
        │   │   │   │   ├── KClique.h
        │   │   │   │   ├── RainbowClique.h
        │   │   │   │   ├── main.cpp
        │   │   │   │   ├── main_k_clique.cpp
        │   │   │   │   ├── makefile
        │   │   │   │   └── runtime_stats.bin
        │   │   │   ├── clique_finder.cpp
        │   │   │   ├── clique_finder_control.cpp
        │   │   │   ├── colored_graph.cpp
        │   │   │   ├── graph_layer.cpp
        │   │   │   ├── graph_node.cpp
        │   │   │   ├── graph_theory.h
        │   │   │   ├── graph_theory_domain.cpp
        │   │   │   ├── layered_graph.cpp
        │   │   │   ├── layered_graph_draw_options.cpp
        │   │   │   ├── makefile
        │   │   │   └── rainbow_cliques.cpp
        │   │   ├── graph_theory_nauty
        │   │   │   ├── graph_theory_nauty.h
        │   │   │   ├── makefile
        │   │   │   ├── naugraph.c
        │   │   │   ├── naurng.c
        │   │   │   ├── naurng.h
        │   │   │   ├── nautil.c
        │   │   │   ├── nauty.c
        │   │   │   ├── nauty.h
        │   │   │   ├── nauty_interface.cpp
        │   │   │   ├── schreier.c
        │   │   │   ├── schreier.h
        │   │   │   └── sorttemplates.c
        │   │   ├── graphics
        │   │   │   ├── EasyBMP.cpp
        │   │   │   ├── EasyBMP.h
        │   │   │   ├── EasyBMP_BMP.h
        │   │   │   ├── EasyBMP_DataStructures.h
        │   │   │   ├── EasyBMP_VariousBMPutilities.h
        │   │   │   ├── animate.cpp
        │   │   │   ├── draw_bitmap_control.cpp
        │   │   │   ├── draw_mod_n_description.cpp
        │   │   │   ├── draw_projective_curve_description.cpp
        │   │   │   ├── drawable_set_of_objects.cpp
        │   │   │   ├── graphical_output.cpp
        │   │   │   ├── graphics.h
        │   │   │   ├── makefile
        │   │   │   ├── mp_graphics.cpp
        │   │   │   ├── parametric_curve.cpp
        │   │   │   ├── parametric_curve_point.cpp
        │   │   │   ├── plot_tools.cpp
        │   │   │   ├── povray_interface.cpp
        │   │   │   ├── scene.cpp
        │   │   │   ├── scene2.cpp
        │   │   │   ├── tree.cpp
        │   │   │   ├── tree_node.cpp
        │   │   │   └── video_draw_options.cpp
        │   │   ├── io_and_os
        │   │   │   ├── create_file_description.cpp
        │   │   │   ├── file_io.cpp
        │   │   │   ├── file_output.cpp
        │   │   │   ├── io_and_os.h
        │   │   │   ├── latex_interface.cpp
        │   │   │   ├── makefile
        │   │   │   ├── mem_object_registry.cpp
        │   │   │   ├── mem_object_registry_entry.cpp
        │   │   │   ├── memory_object.cpp
        │   │   │   ├── orbiter_data_file.cpp
        │   │   │   ├── os_interface.cpp
        │   │   │   ├── override_double.cpp
        │   │   │   ├── prepare_frames.cpp
        │   │   │   └── util.cpp
        │   │   ├── knowledge_base
        │   │   │   ├── DATA
        │   │   │   │   ├── data_BLT.cpp
        │   │   │   │   ├── data_BLT.cpp~
        │   │   │   │   ├── data_DH.cpp
        │   │   │   │   ├── data_hyperovals.cpp
        │   │   │   │   ├── data_packings_PG_3_3.cpp
        │   │   │   │   ├── data_spreads.cpp
        │   │   │   │   ├── data_tensor.cpp
        │   │   │   │   ├── planes_16.cpp
        │   │   │   │   ├── quartic_curves_q13.cpp
        │   │   │   │   ├── quartic_curves_q19.cpp
        │   │   │   │   ├── quartic_curves_q23.cpp
        │   │   │   │   ├── quartic_curves_q25.cpp
        │   │   │   │   ├── quartic_curves_q27.cpp
        │   │   │   │   ├── quartic_curves_q29.cpp
        │   │   │   │   ├── quartic_curves_q31.cpp
        │   │   │   │   ├── quartic_curves_q9.cpp
        │   │   │   │   ├── surface_101.cpp
        │   │   │   │   ├── surface_103.cpp
        │   │   │   │   ├── surface_107.cpp
        │   │   │   │   ├── surface_109.cpp
        │   │   │   │   ├── surface_11.cpp
        │   │   │   │   ├── surface_113.cpp
        │   │   │   │   ├── surface_121.cpp
        │   │   │   │   ├── surface_127.cpp
        │   │   │   │   ├── surface_128.cpp
        │   │   │   │   ├── surface_13.cpp
        │   │   │   │   ├── surface_16.cpp
        │   │   │   │   ├── surface_17.cpp
        │   │   │   │   ├── surface_19.cpp
        │   │   │   │   ├── surface_23.cpp
        │   │   │   │   ├── surface_25.cpp
        │   │   │   │   ├── surface_27.cpp
        │   │   │   │   ├── surface_29.cpp
        │   │   │   │   ├── surface_31.cpp
        │   │   │   │   ├── surface_32.cpp
        │   │   │   │   ├── surface_37.cpp
        │   │   │   │   ├── surface_4.cpp
        │   │   │   │   ├── surface_41.cpp
        │   │   │   │   ├── surface_43.cpp
        │   │   │   │   ├── surface_47.cpp
        │   │   │   │   ├── surface_49.cpp
        │   │   │   │   ├── surface_53.cpp
        │   │   │   │   ├── surface_59.cpp
        │   │   │   │   ├── surface_61.cpp
        │   │   │   │   ├── surface_64.cpp
        │   │   │   │   ├── surface_67.cpp
        │   │   │   │   ├── surface_7.cpp
        │   │   │   │   ├── surface_71.cpp
        │   │   │   │   ├── surface_73.cpp
        │   │   │   │   ├── surface_79.cpp
        │   │   │   │   ├── surface_8.cpp
        │   │   │   │   ├── surface_81.cpp
        │   │   │   │   ├── surface_83.cpp
        │   │   │   │   ├── surface_89.cpp
        │   │   │   │   ├── surface_9.cpp
        │   │   │   │   └── surface_97.cpp
        │   │   │   ├── knowledge_base.cpp
        │   │   │   ├── knowledge_base.h
        │   │   │   └── makefile
        │   │   ├── makefile
        │   │   ├── makefile~
        │   │   ├── orthogonal
        │   │   │   ├── blt_set_domain.cpp
        │   │   │   ├── blt_set_invariants.cpp
        │   │   │   ├── makefile
        │   │   │   ├── orthogonal.cpp
        │   │   │   ├── orthogonal.h
        │   │   │   ├── orthogonal_blt.cpp
        │   │   │   ├── orthogonal_group.cpp
        │   │   │   ├── orthogonal_hyperbolic.cpp
        │   │   │   ├── orthogonal_io.cpp
        │   │   │   ├── orthogonal_parabolic.cpp
        │   │   │   ├── orthogonal_rank_unrank.cpp
        │   │   │   └── unusual_model.cpp
        │   │   ├── solvers
        │   │   │   ├── diophant.cpp
        │   │   │   ├── diophant_activity.cpp
        │   │   │   ├── diophant_activity_description.cpp
        │   │   │   ├── diophant_create.cpp
        │   │   │   ├── diophant_description.cpp
        │   │   │   ├── dlx.cpp
        │   │   │   ├── makefile
        │   │   │   ├── mckay.cpp
        │   │   │   └── solvers.h
        │   │   ├── statistics
        │   │   │   ├── makefile
        │   │   │   ├── statistics.h
        │   │   │   ├── tally.cpp
        │   │   │   └── tally_vector_data.cpp
        │   │   └── surfaces
        │   │       ├── arc_lifting_with_two_lines.cpp
        │   │       ├── clebsch_map.cpp
        │   │       ├── del_pezzo_surface_of_degree_two_domain.cpp
        │   │       ├── del_pezzo_surface_of_degree_two_object.cpp
        │   │       ├── eckardt_point.cpp
        │   │       ├── eckardt_point_info.cpp
        │   │       ├── makefile
        │   │       ├── schlaefli.cpp
        │   │       ├── schlaefli_labels.cpp
        │   │       ├── seventytwo_cases.cpp
        │   │       ├── surface_domain.cpp
        │   │       ├── surface_domain2.cpp
        │   │       ├── surface_domain_families.cpp
        │   │       ├── surface_domain_io.cpp
        │   │       ├── surface_domain_lines.cpp
        │   │       ├── surface_object.cpp
        │   │       ├── surface_object_properties.cpp
        │   │       ├── surfaces.h
        │   │       └── web_of_cubic_curves.cpp
        │   ├── group_actions
        │   │   ├── action.h~
        │   │   ├── actions
        │   │   │   ├── action.cpp
        │   │   │   ├── action_cb.cpp
        │   │   │   ├── action_global.cpp
        │   │   │   ├── action_group_theory.cpp
        │   │   │   ├── action_indexing_cosets.cpp
        │   │   │   ├── action_induce.cpp
        │   │   │   ├── action_init.cpp
        │   │   │   ├── action_io.cpp
        │   │   │   ├── action_pointer_table.cpp
        │   │   │   ├── action_projective.cpp
        │   │   │   ├── actions.h
        │   │   │   ├── backtrack.cpp
        │   │   │   ├── interface_direct_product.cpp
        │   │   │   ├── interface_induced_action.cpp
        │   │   │   ├── interface_matrix_group.cpp
        │   │   │   ├── interface_perm_group.cpp
        │   │   │   ├── interface_permutation_representation.cpp
        │   │   │   ├── interface_wreath_product.cpp
        │   │   │   ├── makefile
        │   │   │   ├── nauty_interface_with_group.cpp
        │   │   │   └── stabilizer_chain_base_data.cpp
        │   │   ├── data_structures
        │   │   │   ├── data_structures.h
        │   │   │   ├── group_container.cpp
        │   │   │   ├── incidence_structure_with_group.cpp
        │   │   │   ├── makefile
        │   │   │   ├── orbit_rep.cpp
        │   │   │   ├── orbit_transversal.cpp
        │   │   │   ├── orbit_type_repository.cpp
        │   │   │   ├── schreier_vector.cpp
        │   │   │   ├── schreier_vector_handler.cpp
        │   │   │   ├── set_and_stabilizer.cpp
        │   │   │   ├── union_find.cpp
        │   │   │   ├── union_find_on_k_subsets.cpp
        │   │   │   └── vector_ge.cpp
        │   │   ├── group_actions.h
        │   │   ├── groups
        │   │   │   ├── chrono.h
        │   │   │   ├── direct_product.cpp
        │   │   │   ├── exceptional_isomorphism_O4.cpp
        │   │   │   ├── groups.h
        │   │   │   ├── linalg.cpp
        │   │   │   ├── linalg.h
        │   │   │   ├── linear_group.cpp
        │   │   │   ├── linear_group_description.cpp
        │   │   │   ├── makefile
        │   │   │   ├── matrix_group.cpp
        │   │   │   ├── orbits_on_something.cpp
        │   │   │   ├── permutation_group_create.cpp
        │   │   │   ├── permutation_group_description.cpp
        │   │   │   ├── permutation_representation.cpp
        │   │   │   ├── permutation_representation_domain.cpp
        │   │   │   ├── schreier.cpp
        │   │   │   ├── schreier_io.cpp
        │   │   │   ├── schreier_sims.cpp
        │   │   │   ├── shallow_schreier_ai.cpp
        │   │   │   ├── shallow_schreier_ai.h
        │   │   │   ├── sims.cpp
        │   │   │   ├── sims2.cpp
        │   │   │   ├── sims3.cpp
        │   │   │   ├── sims_group_theory.cpp
        │   │   │   ├── sims_io.cpp
        │   │   │   ├── sims_main.cpp
        │   │   │   ├── strong_generators.cpp
        │   │   │   ├── strong_generators_groups.cpp
        │   │   │   ├── subgroup.cpp
        │   │   │   ├── sylow_structure.cpp
        │   │   │   └── wreath_product.cpp
        │   │   ├── induced_actions
        │   │   │   ├── action_by_conjugation.cpp
        │   │   │   ├── action_by_representation.cpp
        │   │   │   ├── action_by_restriction.cpp
        │   │   │   ├── action_by_right_multiplication.cpp
        │   │   │   ├── action_by_subfield_structure.cpp
        │   │   │   ├── action_on_andre.cpp
        │   │   │   ├── action_on_bricks.cpp
        │   │   │   ├── action_on_cosets.cpp
        │   │   │   ├── action_on_determinant.cpp
        │   │   │   ├── action_on_factor_space.cpp
        │   │   │   ├── action_on_flags.cpp
        │   │   │   ├── action_on_galois_group.cpp
        │   │   │   ├── action_on_grassmannian.cpp
        │   │   │   ├── action_on_homogeneous_polynomials.cpp
        │   │   │   ├── action_on_interior_direct_product.cpp
        │   │   │   ├── action_on_k_subsets.cpp
        │   │   │   ├── action_on_orbits.cpp
        │   │   │   ├── action_on_orthogonal.cpp
        │   │   │   ├── action_on_set_partitions.cpp
        │   │   │   ├── action_on_sets.cpp
        │   │   │   ├── action_on_sign.cpp
        │   │   │   ├── action_on_spread_set.cpp
        │   │   │   ├── action_on_subgroups.cpp
        │   │   │   ├── action_on_wedge_product.cpp
        │   │   │   ├── induced_actions.h
        │   │   │   ├── makefile
        │   │   │   └── product_action.cpp
        │   │   ├── makefile
        │   │   └── makefile~
        │   ├── makefile
        │   ├── makefile~
        │   ├── orbiter.h
        │   ├── orbiter.h~
        │   └── top_level
        │       ├── algebra_and_number_theory
        │       │   ├── algebra_global_with_action.cpp
        │       │   ├── any_group.cpp
        │       │   ├── any_group_linear.cpp
        │       │   ├── character_table_burnside.cpp
        │       │   ├── group_theoretic_activity.cpp
        │       │   ├── group_theoretic_activity_description.cpp
        │       │   ├── makefile
        │       │   ├── orbits_on_polynomials.cpp
        │       │   ├── tl_algebra_and_number_theory.h
        │       │   └── young.cpp
        │       ├── combinatorics
        │       │   ├── boolean_function_classify.cpp
        │       │   ├── combinatorics_global.cpp
        │       │   ├── delandtsheer_doyen.cpp
        │       │   ├── delandtsheer_doyen_description.cpp
        │       │   ├── design_activity.cpp
        │       │   ├── design_activity_description.cpp
        │       │   ├── design_create.cpp
        │       │   ├── design_create_description.cpp
        │       │   ├── design_tables.cpp
        │       │   ├── difference_set_in_heisenberg_group.cpp
        │       │   ├── hadamard_classify.cpp
        │       │   ├── hall_system_classify.cpp
        │       │   ├── large_set_activity.cpp
        │       │   ├── large_set_activity_description.cpp
        │       │   ├── large_set_classify.cpp
        │       │   ├── large_set_was.cpp
        │       │   ├── large_set_was_activity.cpp
        │       │   ├── large_set_was_activity_description.cpp
        │       │   ├── large_set_was_description.cpp
        │       │   ├── makefile
        │       │   ├── regular_linear_space_description.cpp
        │       │   ├── regular_ls_classify.cpp
        │       │   ├── tactical_decomposition.cpp
        │       │   └── tl_combinatorics.h
        │       ├── geometry
        │       │   ├── arc_generator.cpp
        │       │   ├── arc_generator_description.cpp
        │       │   ├── arc_lifting_simeon.cpp
        │       │   ├── choose_points_or_lines.cpp
        │       │   ├── classify_cubic_curves.cpp
        │       │   ├── cubic_curve_with_action.cpp
        │       │   ├── hermitian_spreads_classify.cpp
        │       │   ├── linear_set_classify.cpp
        │       │   ├── makefile
        │       │   ├── ovoid_classify.cpp
        │       │   ├── ovoid_classify_description.cpp
        │       │   ├── polar.cpp
        │       │   ├── search_blocking_set.cpp
        │       │   ├── singer_cycle.cpp
        │       │   ├── tensor_classify.cpp
        │       │   ├── tl_geometry.h
        │       │   └── top_level_geometry_global.cpp
        │       ├── graph_theory
        │       │   ├── cayley_graph_search.cpp
        │       │   ├── create_graph.cpp
        │       │   ├── create_graph_description.cpp
        │       │   ├── graph_classification_activity.cpp
        │       │   ├── graph_classification_activity_description.cpp
        │       │   ├── graph_classify.cpp
        │       │   ├── graph_classify_description.cpp
        │       │   ├── graph_theoretic_activity.cpp
        │       │   ├── graph_theoretic_activity_description.cpp
        │       │   ├── graph_theory.h
        │       │   └── makefile
        │       ├── interfaces
        │       │   ├── activity_description.cpp
        │       │   ├── interface_algebra.cpp
        │       │   ├── interface_coding_theory.cpp
        │       │   ├── interface_combinatorics.cpp
        │       │   ├── interface_cryptography.cpp
        │       │   ├── interface_povray.cpp
        │       │   ├── interface_projective.cpp
        │       │   ├── interface_symbol_table.cpp
        │       │   ├── interface_toolkit.cpp
        │       │   ├── interfaces.h
        │       │   ├── makefile
        │       │   ├── orbiter_command.cpp
        │       │   ├── orbiter_top_level_session.cpp
        │       │   └── symbol_definition.cpp
        │       ├── isomorph
        │       │   ├── isomorph.cpp
        │       │   ├── isomorph.h
        │       │   ├── isomorph_arguments.cpp
        │       │   ├── isomorph_database.cpp
        │       │   ├── isomorph_files.cpp
        │       │   ├── isomorph_global.cpp
        │       │   ├── isomorph_testing.cpp
        │       │   ├── isomorph_trace.cpp
        │       │   ├── makefile
        │       │   └── representatives.cpp
        │       ├── makefile
        │       ├── makefile~
        │       ├── orbits
        │       │   ├── makefile
        │       │   ├── orbit_of_equations.cpp
        │       │   ├── orbit_of_sets.cpp
        │       │   ├── orbit_of_subspaces.cpp
        │       │   └── orbits.h
        │       ├── orthogonal
        │       │   ├── BLT_set_create.cpp
        │       │   ├── BLT_set_create_description.cpp
        │       │   ├── blt_set_classify.cpp
        │       │   ├── blt_set_with_action.cpp
        │       │   ├── makefile
        │       │   ├── orthogonal_space_activity.cpp
        │       │   ├── orthogonal_space_activity_description.cpp
        │       │   ├── orthogonal_space_with_action.cpp
        │       │   ├── orthogonal_space_with_action_description.cpp
        │       │   └── tl_orthogonal.h
        │       ├── packings
        │       │   ├── invariants_packing.cpp
        │       │   ├── makefile
        │       │   ├── packing_classify.cpp
        │       │   ├── packing_classify2.cpp
        │       │   ├── packing_invariants.cpp
        │       │   ├── packing_long_orbits.cpp
        │       │   ├── packing_long_orbits_description.cpp
        │       │   ├── packing_was.cpp
        │       │   ├── packing_was_activity.cpp
        │       │   ├── packing_was_activity_description.cpp
        │       │   ├── packing_was_description.cpp
        │       │   ├── packing_was_fixpoints.cpp
        │       │   ├── packing_was_fixpoints_activity.cpp
        │       │   ├── packing_was_fixpoints_activity_description.cpp
        │       │   ├── packings.h
        │       │   └── regular_packing.cpp
        │       ├── projective_space
        │       │   ├── canonical_form_classifier.cpp
        │       │   ├── canonical_form_classifier_description.cpp
        │       │   ├── canonical_form_nauty.cpp
        │       │   ├── canonical_form_substructure.cpp
        │       │   ├── makefile
        │       │   ├── object_in_projective_space_with_action.cpp
        │       │   ├── projective_space.h
        │       │   ├── projective_space_activity.cpp
        │       │   ├── projective_space_activity_description.cpp
        │       │   ├── projective_space_object_classifier.cpp
        │       │   ├── projective_space_object_classifier_description.cpp
        │       │   ├── projective_space_with_action.cpp
        │       │   └── projective_space_with_action_description.cpp
        │       ├── semifields
        │       │   ├── makefile
        │       │   ├── semifield_classify.cpp
        │       │   ├── semifield_classify_description.cpp
        │       │   ├── semifield_classify_with_substructure.cpp
        │       │   ├── semifield_downstep_node.cpp
        │       │   ├── semifield_flag_orbit_node.cpp
        │       │   ├── semifield_level_two.cpp
        │       │   ├── semifield_lifting.cpp
        │       │   ├── semifield_substructure.cpp
        │       │   ├── semifield_trace.cpp
        │       │   ├── semifields.h
        │       │   └── trace_record.cpp
        │       ├── solver
        │       │   ├── exact_cover.cpp
        │       │   ├── exact_cover_arguments.cpp
        │       │   ├── makefile
        │       │   └── solver.h
        │       ├── spreads
        │       │   ├── makefile
        │       │   ├── recoordinatize.cpp
        │       │   ├── spread_classify.cpp
        │       │   ├── spread_classify2.cpp
        │       │   ├── spread_create.cpp
        │       │   ├── spread_create_description.cpp
        │       │   ├── spread_lifting.cpp
        │       │   ├── spread_table_activity.cpp
        │       │   ├── spread_table_activity_description.cpp
        │       │   ├── spread_table_with_selection.cpp
        │       │   ├── spreads.h
        │       │   └── translation_plane_via_andre_model.cpp
        │       ├── surfaces
        │       │   ├── makefile
        │       │   ├── quartic_curves
        │       │   │   ├── makefile
        │       │   │   ├── quartic_curve_activity.cpp
        │       │   │   ├── quartic_curve_activity_description.cpp
        │       │   │   ├── quartic_curve_create.cpp
        │       │   │   ├── quartic_curve_create_description.cpp
        │       │   │   ├── quartic_curve_domain_with_action.cpp
        │       │   │   ├── quartic_curve_from_surface.cpp
        │       │   │   ├── quartic_curve_object_with_action.cpp
        │       │   │   └── quartic_curves.h
        │       │   ├── surfaces_and_arcs
        │       │   │   ├── arc_lifting.cpp
        │       │   │   ├── arc_orbits_on_pairs.cpp
        │       │   │   ├── arc_partition.cpp
        │       │   │   ├── classify_trihedral_pairs.cpp
        │       │   │   ├── makefile
        │       │   │   ├── six_arcs_not_on_a_conic.cpp
        │       │   │   ├── surface_classify_using_arc.cpp
        │       │   │   ├── surface_create_by_arc_lifting.cpp
        │       │   │   ├── surfaces_and_arcs.h
        │       │   │   ├── surfaces_arc_lifting.cpp
        │       │   │   ├── surfaces_arc_lifting_definition_node.cpp
        │       │   │   ├── surfaces_arc_lifting_trace.cpp
        │       │   │   ├── surfaces_arc_lifting_upstep.cpp
        │       │   │   └── trihedral_pair_with_action.cpp
        │       │   ├── surfaces_and_double_sixes
        │       │   │   ├── classification_of_cubic_surfaces_with_double_sixes_activity.cpp
        │       │   │   ├── classification_of_cubic_surfaces_with_double_sixes_activity_description.cpp
        │       │   │   ├── classify_double_sixes.cpp
        │       │   │   ├── makefile
        │       │   │   ├── surface_classify_wedge.cpp
        │       │   │   └── surfaces_and_double_sixes.h
        │       │   └── surfaces_general
        │       │       ├── cubic_surface_activity.cpp
        │       │       ├── cubic_surface_activity_description.cpp
        │       │       ├── makefile
        │       │       ├── surface_clebsch_map.cpp
        │       │       ├── surface_create.cpp
        │       │       ├── surface_create_description.cpp
        │       │       ├── surface_domain_high_level.cpp
        │       │       ├── surface_object_with_action.cpp
        │       │       ├── surface_study.cpp
        │       │       ├── surface_with_action.cpp
        │       │       └── surfaces_general.h
        │       ├── top_level.h
        │       └── top_level.h~
        ├── makefile
        └── makefile~
    
    68 directories, 765 files
