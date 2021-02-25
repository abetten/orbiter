DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	arc_lifting_with_two_lines.o \
	clebsch_map.o \
	del_pezzo_surface_of_degree_two_domain.o \
	del_pezzo_surface_of_degree_two_object.o \
	eckardt_point_info.o \
	eckardt_point.o \
	schlaefli.o \
	seventytwo_cases.o \
	surface_domain_families.o \
	surface_domain_io.o \
	surface_domain_lines.o \
	surface_domain.o \
	surface_domain2.o \
	surface_object_properties.o \
	surface_object.o \
	web_of_cubic_curves.o 



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

