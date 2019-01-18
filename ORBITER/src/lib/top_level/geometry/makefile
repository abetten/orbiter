DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	BLT_set_create_description.o \
	BLT_set_create.o \
	blt_set.o \
	arc_generator.o \
	arc_lifting_simeon.o \
	arc_lifting.o \
	arc_orbits_on_pairs.o \
	arc_partition.o \
	choose_points_or_lines.o \
	classify_double_sixes.o \
	classify_trihedral_pairs.o \
	decomposition.o \
	incidence_structure.o \
	k_arc_generator.o \
	polar.o \
	projective_space.o \
	recoordinatize.o \
	search_blocking_set.o \
	singer_cycle.o \
	six_arcs_not_on_a_conic.o \
	spread.o \
	spread2.o \
	spread_create.o \
	spread_create_description.o \
	spread_lifting.o \
	surface_classify_wedge.o \
	surface_create.o \
	surface_create_description.o \
	surface_object_with_action.o \
	surface_with_action.o \
	surfaces_arc_lifting.o \
	translation_plane_via_andre_model.o


all : $(MODULES)

clean:
	- rm *.o *.a *~

