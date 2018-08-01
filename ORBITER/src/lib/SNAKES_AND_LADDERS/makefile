DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm



MODULES = \
	classification.o \
	compute_stabilizer.o \
	extension.o \
	flag_orbit_node.o \
	flag_orbits.o \
	generator.o \
	generator_init.o \
	generator_classify.o \
	generator_combinatorics.o \
	generator_draw.o \
	generator_io.o \
	generator_trace.o \
	oracle.o \
	oracle_downstep.o \
	oracle_downstep_subspace_action.o \
	oracle_io.o \
	oracle_upstep.o \
	oracle_upstep_subspace_action.o \
	orbit_node.o \
	recognize.o \
	set_stabilizer_compute.o \
	snakes_and_ladders_global.o \
	upstep_work.o \
	upstep_work_subspace_action.o \
	upstep_work_trace.o 


all: $(MODULES)

		

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



