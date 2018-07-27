DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	andre_construction.o \
	andre_construction_line_element.o \
	andre_construction_point_element.o \
	buekenhout_metz.o \
	data.o \
	decomposition.o \
	desarguesian_spread.o \
	eckardt_point.o \
	flag.o \
	geometric_object.o \
	geometric_operations.o \
	grassmann.o \
	grassmann_embedded.o \
	hermitian.o \
	hjelmslev.o \
	incidence_structure.o \
	klein_correspondence.o \
	knarr.o \
	object_in_projective_space.o \
	orthogonal.o \
	orthogonal_points.o \
	projective.o \
	projective_space.o \
	surface.o \
	surface_object.o \
	unusual.o \
	w3q.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



