DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

MODULES = \
	BLT_set_create.o \
	BLT_set_create_description.o \
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
	k_arc_generator.o \
	object_in_projective_space_with_action.o \
	orbit_of_equations.o \
	orbit_of_sets.o \
	orbit_of_subspaces.o \
	orbit_rep.o \
	polar.o \
	projective_space.o \
	recoordinatize.o \
	representatives.o \
	search_blocking_set.o \
	singer_cycle.o \
	six_arcs_not_on_a_conic.o \
	spread.o \
	spread2.o \
	spread_create.o \
	spread_create_description.o \
	spread_lifting.o \
	subspace_orbits.o \
	surface_classify_wedge.o \
	surface_create.o \
	surface_create_description.o \
	surface_object_with_action.o \
	surface_with_action.o \
	top_level_global.o \
	translation_plane_via_andre_model.o \
	young.o
		
all: $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



