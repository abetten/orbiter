DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	andre_construction.o \
	andre_construction_line_element.o \
	andre_construction_point_element.o \
	arc_lifting_with_two_lines.o \
	blt_set_domain.o \
	blt_set_invariants.o \
	buekenhout_metz.o \
	clebsch_map.o \
	cubic_curve.o \
	decomposition.o \
	desarguesian_spread.o \
	eckardt_point_info.o \
	eckardt_point.o \
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
	knowledge_base.o \
	object_in_projective_space.o \
	orthogonal.o \
	point_line.o \
	projective_space.o \
	spread_tables.o \
	surface_domain_lines.o \
	surface_domain_io.o \
	surface_domain.o \
	surface_object.o \
	unusual.o \
	w3q.o 



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

