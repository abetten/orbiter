DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_basic.o \
	arc_in_projective_space.o \
	flag.o \
	geometric_object_create.o \
	geometric_object_description.o \
	geometry_global.o \
	hermitian.o \
	hjelmslev.o \
	incidence_structure.o \
	intersection_type.o \
	point_line.o \
	points_and_lines.o \
	three_skew_subspaces.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~



