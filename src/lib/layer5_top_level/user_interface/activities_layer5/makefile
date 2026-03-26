DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	action_on_forms_activity_description.o \
	action_on_forms_activity.o \
	blt_set_activity_description.o \
	blt_set_activity.o \
	blt_set_classify_activity_description.o \
	blt_set_classify_activity.o \
	classification_of_cubic_surfaces_with_double_sixes_activity_description.o \
	classification_of_cubic_surfaces_with_double_sixes_activity.o \
	coding_theoretic_activity_description.o \
	coding_theoretic_activity.o \
	combinatorial_object_activity_description.o \
	combinatorial_object_activity.o \
	combo_activity_description.o \
	combo_activity.o \
	cubic_surface_activity_description.o \
	cubic_surface_activity.o \
	design_activity_description.o \
	design_activity.o \
	graph_classification_activity_description.o \
	graph_classification_activity.o \
	graph_theoretic_activity_description.o \
	graph_theoretic_activity.o \
	group_theoretic_activity_description.o \
	group_theoretic_activity.o \
	large_set_activity_description.o \
	large_set_activity.o \
	large_set_was_activity_description.o \
	large_set_was_activity.o \
	orbits_activity_description.o \
	orbits_activity.o \
	orthogonal_space_activity_description.o \
	orthogonal_space_activity.o \
	packing_classify_activity_description.o \
	packing_classify_activity.o \
	packing_was_activity_description.o \
	packing_was_activity.o \
	packing_was_fixpoints_activity_description.o \
	packing_was_fixpoints_activity.o \
	plesken_ring_activity_description.o \
	plesken_ring_activity.o \
	poset_classification_activity_description.o \
	poset_classification_activity.o \
	projective_space_activity_description.o \
	projective_space_activity.o \
	quartic_curve_activity_description.o \
	quartic_curve_activity.o \
	spread_activity_description.o \
	spread_activity.o \
	spread_classify_activity_description.o \
	spread_classify_activity.o \
	spread_table_activity_description.o \
	spread_table_activity.o \
	translation_plane_activity_description.o \
	translation_plane_activity.o \
	variety_activity_description.o \
	variety_activity.o \
	vector_ge_activity_description.o \
	vector_ge_activity.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~




