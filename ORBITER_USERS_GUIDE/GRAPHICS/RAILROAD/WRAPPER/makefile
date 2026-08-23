###############################################################################
# TEX.24/ORBITER_USERS_GUIDE/GRAPHICS/ORBITER_RAILROAD/diagram
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
# additional configurations for when you want to 
# compile automatically generated code
###############################################################################


WRAPPER_FILE_MASK="rr_%s.tex"

WRAPPER_FILE_CONTENT_MASK="\\[\n\\begin{array}{c}\n\\includegraphics{GRAPHICS/RAILROAD/diagram/%s}\n\\end{array}\n\\]"


create_wrapper:
	$(ORBITER) -create_files_direct $(WRAPPER_FILE_MASK) \
		$(WRAPPER_FILE_CONTENT_MASK) \
		Activity \
		BLT_set \
		BLT_set_classifier \
		BLT_set_classify_activity \
		Command \
		CommandSequence \
		Definition \
		Assignment \
		DrawIncOpts \
		DrawOpts \
		FileMask \
		FilePath \
		Fork \
		GlobalCmd \
		Grammar \
		Integer \
		Literal \
		MagmaPath \
		MemoryDebug \
		OrbiterPath \
		OverridePoly \
		Program \
		SeedRandomGen \
		Setup \
		Verbosity \
		action_on_forms \
		action_on_forms_activity \
		arc_generator_control \
		blt_set_activity \
		classification_of_cubic_surfaces_with_double_sixes_activity \
		code \
		coding_theoretic_activity \
		collection \
		combinatorial_object \
		combinatorial_object_activity \
		crc_code \
		cubic_surface \
		cubic_surface_activity \
		design \
		design_activity \
		design_table \
		diophant \
		diophant_activity \
		finite_field \
		finite_field_activity \
		geometric_object \
		geometry_builder \
		graph \
		graph_classification \
		graph_classification_activity \
		graph_theoretic_activity \
		group_modification \
		group_theoretic_activity \
		large_set_was \
		large_set_was_activity \
		linear_group \
		mapping \
		orbits \
		orbits_activity \
		orthogonal_space \
		orthogonal_space_activity \
		packing_classify \
		packing_fixed_points_activity \
		packing_long_orbits \
		packing_was \
		packing_was_choose_fixed_points \
		packing_with_symmetry_assumption_activity \
		permutation_group \
		polynomial_ring \
		polynomial_ring_activity \
		poset_classification_activity \
		poset_classification_control \
		poset_classification_report_options \
		projective_space \
		projective_space_activity \
		quartic_curve \
		quartic_curve_activity \
		set \
		spread \
		spread_activity \
		spread_classifier \
		spread_classify_activity \
		spread_table \
		spread_table_activity \
		symbolic_object \
		symbolic_object_activity \
		translation_plane \
		translation_plane_activity \
		variety \
		variety_activity \
		vector \
		vector_ge \
		DataStructure \
		AlgebraObject \
		GeometryObject \
		CombinatoricsObject \
		CodingTheoryObject \
		OtherObject \
		ObjectTypeAndArgs \
		AlgebraActivity \
		GeometryActivity \
		CombinatoricsActivity \
		CodingTheoryActivity \
		OtherActivity \
		ObjectActivity \
		-end



WRAPPER_FILE_CONTENT_MASK_LARGE="\\[\n\\begin{array}{c}\n\\includegraphics[height=20cm]{GRAPHICS/RAILROAD/diagram/%s}\n\\end{array}\n\\]"


#create_wrapper_large:
#	$(ORBITER) -create_files_direct $(WRAPPER_FILE_MASK) \
#		$(WRAPPER_FILE_CONTENT_MASK_LARGE) \
#		ObjectActivity \
#		-end


