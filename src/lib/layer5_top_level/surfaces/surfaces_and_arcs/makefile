DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_lifting.o \
	arc_orbits_on_pairs.o \
	arc_partition.o \
	classify_trihedral_pairs.o \
	six_arcs_not_on_a_conic.o \
	surface_classify_using_arc.o \
	surface_create_by_arc_lifting.o \
	surfaces_arc_lifting_definition_node.o \
	surfaces_arc_lifting_trace.o \
	surfaces_arc_lifting_upstep.o \
	surfaces_arc_lifting.o \
	trihedral_pair_with_action.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

