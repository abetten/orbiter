DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm
CPP_PREFIX=f_
UNIX_TO_DOS = ~/bin/unix2dos -n 

MODULES = \
	analyze_group.o \
	arc_generator.o \
	arc_lifting.o \
	choose_points_or_lines.o \
	classify_double_sixes.o \
	classify_trihedral_pairs.o \
	decomposition.o \
	elliptic_curve.o \
	exact_cover.o \
	exact_cover_arguments.o \
	extra.o \
	factor_group.o \
	incidence_structure.o \
	isomorph.o \
	isomorph_arguments.o \
	isomorph_database.o \
	isomorph_files.o \
	isomorph_global.o \
	isomorph_testing.o \
	isomorph_trace.o \
	object_in_projective_space_with_action.o \
	orbit_of_sets.o \
	orbit_of_subspaces.o \
	orbit_rep.o \
	polar.o \
	projective_space.o \
	recoordinatize.o \
	representatives.o \
	search_blocking_set.o \
	singer_cycle.o \
	subspace_orbits.o \
	surface_classify_wedge.o \
	surface_create.o \
	surface_create_description.o \
	surface_object_with_action.o \
	surface_with_action.o \
	top_level_global.o \
	translation_plane.o \
	translation_plane2.o \
	translation_plane_via_andre_model.o \
	young.o
		

LIB = libtoplevel.a

libtoplevel.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



cpp:
	$(UNIX_TO_DOS) analyze_group.C $(CPP_PREFIX)analyze_group.cpp
	$(UNIX_TO_DOS) arc_generator.C $(CPP_PREFIX)arc_generator.cpp
	$(UNIX_TO_DOS) arc_lifting.C $(CPP_PREFIX)arc_lifting.cpp
	$(UNIX_TO_DOS) buekenhout_metz.C $(CPP_PREFIX)buekenhout_metz.cpp
	$(UNIX_TO_DOS) choose_points_or_lines.C $(CPP_PREFIX)choose_points_or_lines.cpp
	$(UNIX_TO_DOS) classify_double_sixes.C $(CPP_PREFIX)classify_double_sixes.cpp
	$(UNIX_TO_DOS) classify_trihedral_pairs.C $(CPP_PREFIX)classify_trihedral_pairs.cpp
	$(UNIX_TO_DOS) decomposition.C $(CPP_PREFIX)decomposition.cpp
	$(UNIX_TO_DOS) elliptic_curve.C $(CPP_PREFIX)elliptic_curve.cpp
	$(UNIX_TO_DOS) exact_cover.C $(CPP_PREFIX)exact_cover.cpp
	$(UNIX_TO_DOS) exact_cover_arguments.C $(CPP_PREFIX)exact_cover_arguments.cpp
	$(UNIX_TO_DOS) extra.C $(CPP_PREFIX)extra.cpp
	$(UNIX_TO_DOS) factor_group.C $(CPP_PREFIX)factor_group.cpp
	$(UNIX_TO_DOS) incidence_structure.C $(CPP_PREFIX)incidence_structure.cpp
	$(UNIX_TO_DOS) isomorph.C $(CPP_PREFIX)isomorph.cpp
	$(UNIX_TO_DOS) isomorph_arguments.C $(CPP_PREFIX)isomorph_arguments.cpp
	$(UNIX_TO_DOS) isomorph_database.C $(CPP_PREFIX)isomorph_database.cpp
	$(UNIX_TO_DOS) isomorph_files.C $(CPP_PREFIX)isomorph_files.cpp
	$(UNIX_TO_DOS) isomorph_global.C $(CPP_PREFIX)isomorph_global.cpp
	$(UNIX_TO_DOS) isomorph_testing.C $(CPP_PREFIX)isomorph_testing.cpp
	$(UNIX_TO_DOS) isomorph_trace.C $(CPP_PREFIX)isomorph_trace.cpp
	$(UNIX_TO_DOS) object_in_projective_space_with_action.C $(CPP_PREFIX)object_in_projective_space_with_action.cpp
	$(UNIX_TO_DOS) orbit_of_sets.C $(CPP_PREFIX)orbit_of_sets.cpp
	$(UNIX_TO_DOS) orbit_of_subspaces.C $(CPP_PREFIX)orbit_of_subspaces.cpp
	$(UNIX_TO_DOS) orbit_rep.C $(CPP_PREFIX)orbit_rep.cpp
	$(UNIX_TO_DOS) polar.C $(CPP_PREFIX)polar.cpp
	$(UNIX_TO_DOS) projective_space.C $(CPP_PREFIX)projective_space.cpp
	$(UNIX_TO_DOS) recoordinatize.C $(CPP_PREFIX)recoordinatize.cpp
	$(UNIX_TO_DOS) representatives.C $(CPP_PREFIX)representatives.cpp
	$(UNIX_TO_DOS) search_blocking_set.C $(CPP_PREFIX)search_blocking_set.cpp
	$(UNIX_TO_DOS) singer_cycle.C $(CPP_PREFIX)singer_cycle.cpp
	$(UNIX_TO_DOS) subspace_orbits.C $(CPP_PREFIX)subspace_orbits.cpp
	$(UNIX_TO_DOS) surface_classify_wedge.C $(CPP_PREFIX)surface_classify_wedge.cpp
	$(UNIX_TO_DOS) surface_create.C $(CPP_PREFIX)surface_create.cpp
	$(UNIX_TO_DOS) surface_create_description.C $(CPP_PREFIX)surface_create_description.cpp
	$(UNIX_TO_DOS) surface_object_with_action.C $(CPP_PREFIX)surface_object_with_action.cpp
	$(UNIX_TO_DOS) surface_with_action.C $(CPP_PREFIX)surface_with_action.cpp
	$(UNIX_TO_DOS) top_level_global.C $(CPP_PREFIX)top_level_global.cpp
	$(UNIX_TO_DOS) translation_plane.C $(CPP_PREFIX)translation_plane.cpp
	$(UNIX_TO_DOS) translation_plane2.C $(CPP_PREFIX)translation_plane2.cpp
	$(UNIX_TO_DOS) translation_plane_via_andre_model.C $(CPP_PREFIX)translation_plane_via_andre_model.cpp
	$(UNIX_TO_DOS) young.C $(CPP_PREFIX)young.cpp
	$(UNIX_TO_DOS) top_level.h ../WINDOWS/top_level.h
	mv $(CPP_PREFIX)* ../WINDOWS


