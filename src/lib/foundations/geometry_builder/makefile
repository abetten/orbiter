DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
		cperm.o \
		gen_geo_conf.o \
		gen_geo.o \
		geo_frame.o \
		geo_iso.o \
		geo_main.o \
		geometry_builder_description.o \
		geometry_builder.o \
		globals.o \
		grid.o \
		inc_encoding.o \
		incidence.o \
		iso_grid.o \
		iso_info.o \
		iso_type.o \
		os.o \
		tactical_decomposition.o \
		tdo_gradient.o \
		tdo_scheme.o



all : $(MODULES) 

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

