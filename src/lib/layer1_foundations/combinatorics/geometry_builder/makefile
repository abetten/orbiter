DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
		cperm.o \
		decomposition_with_fuse.o \
		gen_geo_conf.o \
		gen_geo.o \
		geometric_backtrack_search.o \
		geometry_builder_description.o \
		geometry_builder.o \
		girth_test.o \
		inc_encoding.o \
		incidence.o \
		test_semicanonical.o \
		iso_type.o



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

