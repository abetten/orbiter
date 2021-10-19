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
	packings_global.o \
	regular_packing.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

