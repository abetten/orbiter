DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	andre_construction.o \
	andre_construction_line_element.o \
	andre_construction_point_element.o \
	buekenhout_metz.o \
	cubic_curve.o \
	decomposition.o \
	desarguesian_spread.o \
	elliptic_curve.o \
	flag.o \
	geometry_global.o \
	grassmann.o \
	grassmann_embedded.o \
	hermitian.o \
	hjelmslev.o \
	incidence_structure.o \
	klein_correspondence.o \
	knarr.o \
	object_in_projective_space.o \
	point_line.o \
	points_and_lines.o \
	polarity.o \
	projective_space_implementation.o \
	projective_space.o \
	projective_space2.o \
	quartic_curve_domain.o \
	quartic_curve_object_properties.o \
	quartic_curve_object.o \
	spread_tables.o \
	w3q.o 



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

