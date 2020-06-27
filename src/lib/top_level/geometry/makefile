DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_generator_description.o \
	arc_generator.o \
	arc_lifting_simeon.o \
	arc_lifting.o \
	arc_orbits_on_pairs.o \
	arc_partition.o \
	blt_set_classify.o \
	BLT_set_create_description.o \
	BLT_set_create.o \
	blt_set_with_action.o \
	choose_points_or_lines.o \
	classify_cubic_curves.o \
	classify_double_sixes.o \
	classify_trihedral_pairs.o \
	cubic_curve_with_action.o \
	hermitian_spreads_classify.o \
	hill_cap.o \
	invariants_packing.o \
	linear_set_classify.o \
	ovoid_classify.o \
	packing_classify.o \
	packing_classify2.o \
	packing_invariants.o \
	packing_long_orbits.o \
	packing_was_description.o \
	packing_was.o \
	polar.o \
	recoordinatize.o \
	search_blocking_set.o \
	singer_cycle.o \
	six_arcs_not_on_a_conic.o \
	spread_classify.o \
	spread_classify2.o \
	spread_create_description.o \
	spread_create.o \
	spread_lifting.o \
	surface_classify_wedge.o \
	surface_create_description.o \
	surface_create.o \
	surface_object_with_action.o \
	surface_study.o \
	surface_with_action.o \
	surfaces_arc_lifting.o \
	tensor_classify.o \
	translation_plane_via_andre_model.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

