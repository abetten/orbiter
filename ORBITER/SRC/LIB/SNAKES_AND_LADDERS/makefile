DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm
CPP_PREFIX=c_
UNIX_TO_DOS = ~/bin/unix2dos -n 



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


		

LIB = libsnakesandladders.a

libsnakesandladders.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



cpp:
	$(UNIX_TO_DOS) classification.C $(CPP_PREFIX)classification.cpp
	$(UNIX_TO_DOS) compute_stabilizer.C $(CPP_PREFIX)compute_stabilizer.cpp
	$(UNIX_TO_DOS) extension.C $(CPP_PREFIX)extension.cpp
	$(UNIX_TO_DOS) flag_orbit_node.C $(CPP_PREFIX)flag_orbit_node.cpp
	$(UNIX_TO_DOS) flag_orbits.C $(CPP_PREFIX)flag_orbits.cpp
	$(UNIX_TO_DOS) generator.C $(CPP_PREFIX)generator.cpp
	$(UNIX_TO_DOS) generator_init.C $(CPP_PREFIX)generator_init.cpp
	$(UNIX_TO_DOS) generator_classify.C $(CPP_PREFIX)generator_classify.cpp
	$(UNIX_TO_DOS) generator_combinatorics.C $(CPP_PREFIX)generator_combinatorics.cpp
	$(UNIX_TO_DOS) generator_draw.C $(CPP_PREFIX)generator_draw.cpp
	$(UNIX_TO_DOS) generator_io.C $(CPP_PREFIX)generator_io.cpp
	$(UNIX_TO_DOS) generator_trace.C $(CPP_PREFIX)generator_trace.cpp
	$(UNIX_TO_DOS) oracle.C $(CPP_PREFIX)oracle.cpp
	$(UNIX_TO_DOS) oracle_downstep.C $(CPP_PREFIX)oracle_downstep.cpp
	$(UNIX_TO_DOS) oracle_downstep_subspace_action.C $(CPP_PREFIX)oracle_downstep_subspace_action.cpp
	$(UNIX_TO_DOS) oracle_io.C $(CPP_PREFIX)oracle_io.cpp
	$(UNIX_TO_DOS) oracle_upstep.C $(CPP_PREFIX)oracle_upstep.cpp
	$(UNIX_TO_DOS) oracle_upstep_subspace_action.C $(CPP_PREFIX)oracle_upstep_subspace_action.cpp
	$(UNIX_TO_DOS) orbit_node.C $(CPP_PREFIX)orbit_node.cpp
	$(UNIX_TO_DOS) recognize.C $(CPP_PREFIX)recognize.cpp
	$(UNIX_TO_DOS) set_stabilizer_compute.C $(CPP_PREFIX)set_stabilizer_compute.cpp
	$(UNIX_TO_DOS) snakes_and_ladders_global.C $(CPP_PREFIX)snakes_and_ladders_global.cpp
	$(UNIX_TO_DOS) upstep_work.C $(CPP_PREFIX)upstep_work.cpp
	$(UNIX_TO_DOS) upstep_work_subspace_action.C $(CPP_PREFIX)upstep_work_orbits_on_subspaces.cpp
	$(UNIX_TO_DOS) upstep_work_trace.C $(CPP_PREFIX)upstep_work_trace.cpp
	$(UNIX_TO_DOS) snakesandladders.h ../WINDOWS/snakesandladders.h
	mv $(CPP_PREFIX)* ../WINDOWS
	
	
