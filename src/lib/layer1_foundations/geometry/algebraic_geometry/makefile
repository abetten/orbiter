DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
#-march=native
LIB = 
LFLAGS = -lm

MODULES = \
	algebraic_geometry_global.o \
	arc_lifting_with_two_lines.o \
	clebsch_map.o \
	cubic_curve.o \
	del_pezzo_surface_of_degree_two_domain.o \
	del_pezzo_surface_of_degree_two_object.o \
	eckardt_point_info.o \
	eckardt_point.o \
	kovalevski_points.o \
	quartic_curve_domain.o \
	quartic_curve_object_properties.o \
	quartic_curve_object.o \
	schlaefli_double_six.o \
	schlaefli_labels.o \
	schlaefli_trihedral_pairs.o \
	schlaefli_tritangent_planes.o \
	schlaefli.o \
	seventytwo_cases.o \
	smooth_surface_object_properties.o \
	surface_domain_families.o \
	surface_domain_io.o \
	surface_domain_lines.o \
	surface_domain.o \
	surface_domain2.o \
	surface_object_properties.o \
	surface_object.o \
	surface_polynomial_domains.o \
	variety_description.o \
	variety_object.o \
	web_of_cubic_curves.o 




all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

