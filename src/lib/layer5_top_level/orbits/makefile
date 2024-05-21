DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	orbit_cascade.o \
	orbits_activity_description.o \
	orbits_activity.o \
	orbits_create_description.o \
	orbits_create.o \
	orbits_global.o \
	orbits_on_pairs.o \
	orbits_on_polynomials.o \
	orbits_on_subspaces.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

