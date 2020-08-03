DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_lifting.o \
	arc_orbits_on_pairs.o \
	arc_partition.o \
	classify_double_sixes.o \
	classify_trihedral_pairs.o \
	six_arcs_not_on_a_conic.o \
	surface_classify_using_arc.o \
	surface_classify_wedge.o \
	surface_clebsch_map.o \
	surface_create_by_arc_lifting.o \
	surface_create_description.o \
	surface_create.o \
	surface_object_tangent_cone.o \
	surface_object_with_action.o \
	surface_study.o \
	surface_with_action.o \
	surfaces_arc_lifting_definition_node.o \
	surfaces_arc_lifting_trace.o \
	surfaces_arc_lifting_upstep.o \
	surfaces_arc_lifting.o \
	trihedral_pair_with_action.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

