DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	extension.o \
	generator.o \
	generator_classify.o \
	generator_combinatorics.o \
	generator_draw.o \
	generator_init.o \
	generator_io.o \
	generator_recognize.o \
	generator_trace.o \
	poset_orbit_node.o \
	poset_orbit_node_downstep.o \
	poset_orbit_node_downstep_subspace_action.o \
	poset_orbit_node_io.o \
	poset_orbit_node_upstep.o \
	poset_orbit_node_upstep_subspace_action.o \
	upstep_work.o \
	upstep_work_subspace_action.o \
	upstep_work_trace.o





all : $(MODULES)

clean:
	- rm *.o *.a *~


#%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



