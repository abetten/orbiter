DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	invariants_packing.o \
	packing_classify.o \
	packing_classify2.o \
	packing_invariants.o \
	packing_long_orbits_description.o \
	packing_long_orbits.o \
	packing_was_activity_description.o \
	packing_was_activity.o \
	packing_was_description.o \
	packing_was_fixpoints_activity_description.o \
	packing_was_fixpoints_activity.o \
	packing_was_fixpoints.o \
	packing_was.o \
	recoordinatize.o \
	regular_packing.o \
	spread_classify.o \
	spread_classify2.o \
	spread_create_description.o \
	spread_create.o \
	spread_lifting.o \
	spread_table_activity_description.o \
	spread_table_activity.o \
	spread_table_with_selection.o \
	translation_plane_via_andre_model.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

