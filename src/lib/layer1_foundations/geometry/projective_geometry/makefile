DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	grassmann.o \
	grassmann_embedded.o \
	klein_correspondence.o \
	polarity.o \
	projective_space.o \
	projective_space2.o \
	projective_space_basic.o \
	projective_space_implementation.o \
	projective_space_of_dimension_three.o \
	projective_space_plane.o \
	projective_space_reporting.o \
	projective_space_subspaces.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~




