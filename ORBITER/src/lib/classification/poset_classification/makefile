DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	extension.o \
	orbit_based_testing.o \
	poset.o \
	poset_description.o \
	poset_classification.o \
	poset_classification_classify.o \
	poset_classification_combinatorics.o \
	poset_classification_draw.o \
	poset_classification_init.o \
	poset_classification_io.o \
	poset_classification_recognize.o \
	poset_classification_report.o \
	poset_classification_trace.o \
	poset_orbit_node.o \
	poset_orbit_node_downstep.o \
	poset_orbit_node_downstep_subspace_action.o \
	poset_orbit_node_group_theory.o \
	poset_orbit_node_io.o \
	poset_orbit_node_upstep.o \
	poset_orbit_node_upstep_subspace_action.o \
	upstep_work.o \
	upstep_work_subspace_action.o \
	upstep_work_trace.o





all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

