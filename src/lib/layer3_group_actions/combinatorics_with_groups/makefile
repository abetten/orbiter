DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	combinatorics_with_action.o \
	fixed_objects_in_PG.o \
	flag_orbits_incidence_structure.o \
	group_action_on_combinatorial_object.o \
	orbit_type_repository.o \
	translation_plane_via_andre_model.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
