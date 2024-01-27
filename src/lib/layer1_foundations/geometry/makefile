DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -std=c++17 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	andre_construction.o \
	andre_construction_line_element.o \
	andre_construction_point_element.o \
	arc_basic.o \
	arc_in_projective_space.o \
	buekenhout_metz.o \
	desarguesian_spread.o \
	flag.o \
	geometric_object_create.o \
	geometric_object_description.o \
	geometry_global.o \
	grassmann.o \
	grassmann_embedded.o \
	hermitian.o \
	hjelmslev.o \
	incidence_structure.o \
	intersection_type.o \
	klein_correspondence.o \
	knarr.o \
	point_line.o \
	points_and_lines.o \
	polarity.o \
	projective_space_basic.o \
	projective_space_implementation.o \
	projective_space_of_dimension_three.o \
	projective_space_plane.o \
	projective_space_reporting.o \
	projective_space_subspaces.o \
	projective_space.o \
	projective_space2.o \
	spread_domain.o \
	spread_tables.o \
	three_skew_subspaces.o \
	w3q.o 

 

all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

