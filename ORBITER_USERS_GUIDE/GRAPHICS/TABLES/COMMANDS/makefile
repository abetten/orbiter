###############################################################################
# ORBITER_USERS_GUIDE/GRAPHICS/TABLES/COMMANDS
###############################################################################

ORBITER_PATH=~/orbiter

#ORBITER_PATH=/s/hipft-logger/b/nobackup/abetten/orbiter
#ORBITER_PATH=/scratch/betten/COMPILE/orbiter
#ORBITER_PATH=/data/accounts/fac/betten/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 

ORBITER_EXE_PATH=$(ORBITER_PATH)/src/apps/orbiter/

ORBITER=$(ORBITER_EXE_PATH)orbiter.out

SANDBOX=$(ORBITER_PATH)/src/apps/sandbox/sandbox.out

###############################################################################
# start
###############################################################################

make_tables:
	cp /Users/betten/orbiter/doc/tables/commands/*.csv .
	$(ORBITER) -v 6 \
		-prepare_tables_for_users_guide \
		code_modification.csv \
		coding_theoretic_activity_1.csv \
		coding_theoretic_activity_2.csv \
		coding_theoretic_activity_3.csv \
		canonical_form_classifier.csv \
		classification_of_cubic_surfaces_with_double_sixes_activity.csv \
		classification_of_objects.csv \
		clique_finder_control.csv \
		combinatorial_object_activity.csv \
		combinatorics_1.csv \
		combinatorics_2.csv \
		combo_activity.csv \
		crc_code_description.csv \
		crc_options.csv \
		crc_process_description.csv \
		create_graph_1.csv \
		create_graph_2.csv \
		cubic_surface_activity.csv \
		create_code.csv \
		cryptography_1.csv \
		cryptography_2.csv \
		data_input_stream.csv \
		data_input_stream2.csv \
		delandtsheer_doyen.csv \
		design_activity.csv \
		design_create.csv \
		diophant_activity.csv \
		diophant_create.csv \
		draw_bitmap_control.csv \
		draw_incidence_structure_description.csv \
		draw_mod_n.csv \
		drawable_set_of_objects.csv \
		finite_field.csv \
		finite_field_activities.csv \
		finite_field_activities_2.csv \
		finite_field_activities_linear_algebra.csv \
		finite_field_activities_polynomials.csv \
		finite_field_activities_summary_1.csv \
		finite_field_activities_summary_2.csv \
		geometric_object_1.csv \
		geometric_object_2.csv \
		geometry_builder.csv \
		global_basic_number_theory.csv \
		global_coding_theory.csv \
		global_projective.csv \
		graph_classification_activity.csv \
		graph_classify.csv \
		graph_modification.csv \
		graph_theoretic_activities_1.csv \
		graph_theoretic_activities_2.csv \
		group_modification.csv \
		group_theoretic_activity_1.csv \
		group_theoretic_activity_2.csv \
		group_theoretic_activity_3.csv \
		group_theoretic_activity_4.csv \
		group_theoretic_activity_5.csv \
		group_theoretic_activity_gap.csv \
		group_theoretic_activity_magma.csv \
		interface_algebra.csv \
		interface_povray.csv \
		interface_toolkit_1.csv \
		interface_toolkit_2.csv \
		interface_toolkit_3.csv \
		isomorph.csv \
		large_set_was.csv \
		large_set_was_activity.csv \
		layered_graph_draw_options_1.csv \
		layered_graph_draw_options_2.csv \
		linear_group_1.csv \
		linear_group_2.csv \
		nauty_control.csv \
		number_theoretic_commands.csv \
		objects_report_options.csv \
		orbiter_session.csv \
		orbits_activity.csv \
		orbits_create1.csv \
		orbits_create2.csv \
		orthogonal_space_activity.csv \
		orthogonal_space_with_action.csv \
		packing_long_orbits_description.csv \
		packing_was.csv \
		packing_was_activity.csv \
		packing_was_fixpoints_activity.csv \
		permutation_group.csv \
		polynomial_ring.csv \
		polynomial_ring_activity.csv \
		poset_classification_activity_1.csv \
		poset_classification_activity_2.csv \
		poset_classification_control.csv \
		poset_classification_report_options.csv \
		povray_job_description.csv \
		prepare_frames.csv \
		projective_space_activity_1.csv \
		projective_space_activity_2.csv \
		projective_space_activity_3.csv \
		projective_space_activity_4.csv \
		projective_space_with_action.csv \
		scene_1.csv \
		scene_2.csv \
		set_builder.csv \
		surface_create_1.csv \
		surface_create_2.csv \
		surface_create_3.csv \
		special_actions.csv \
		spread_activity.csv \
		spread_classify.csv \
		spread_classify_activity.csv \
		spread_create.csv \
		spread_table.csv \
		spread_table_activity.csv \
		subgroups.csv \
		symbolic_object_1.csv \
		symbolic_object_2.csv \
		symbolic_object_activity.csv \
		variety.csv \
		variety_activity.csv \
		vector_builder.csv \
		vector_activity.csv \
		vector_ge_activity.csv \
		vector_ge_builder.csv \
		video_draw_options_1.csv \
		video_draw_options_2.csv \
		BLT_set_create.csv \
		arc_generator.csv \
		blt_set_activity.csv \
		blt_set_classify.csv \
		blt_set_classify_activity.csv \
		element_processing.csv \
		mapping.csv \
		quartic_curve_activity.csv \
		quartic_curve_create.csv \
		representation_theory.csv \
		tdo_refinement.csv \
		translation_plane_activity.csv \
		tree_draw_options.csv \
		text_builder.csv \
		classify_cubic_surfaces.csv \
		packing_classify.csv \
		plesken_ring_activity.csv \
		-end




#make_table_design_create:
#	cp /Users/betten/orbiter/doc/tables/commands/*.csv .
#	$(ORBITER) -v 6 \
#		-prepare_tables_for_users_guide \
#		design_create.csv \
#	-end
