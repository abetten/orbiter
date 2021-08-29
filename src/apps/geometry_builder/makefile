# GSLFF makefile
# Anton Betten 1998
#
# 

INCLUDES = -I.
CC=gcc
CPP=g++
#CC=gcc
#CC=cxx
CFLAGS=-O2 -DSYSTEMUNIX
LFLAGS=


MODULES = \
		cperm.o \
		gen_geo_conf.o \
		gen_geo.o \
		geo_iso.o \
		geo_main.o \
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

all :  $(LIB) geometry_builder.out ;

LIB = libgeo.a



geometry_builder.out: geometry_builder_main.cpp $(LIB) ;
	$(CPP) $(CFLAGS) $(INCLUDES) $(LFLAGS) geometry_builder_main.cpp libgeo.a -o $@

cperm.o: cperm.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) cperm.cpp
gen_geo_conf.o: gen_geo_conf.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) gen_geo_conf.cpp
gen_geo.o: gen_geo.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) gen_geo.cpp
geo_iso.o: geo_iso.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) geo_iso.cpp
geo_main.o: geo_main.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) geo_main.cpp
geometry_builder.o: geometry_builder.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) geometry_builder.cpp
globals.o: globals.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) globals.cpp
grid.o: grid.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) grid.cpp
inc_encoding.o: inc_encoding.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) inc_encoding.cpp
incidence.o: incidence.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) incidence.cpp
iso_grid.o: iso_grid.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) iso_grid.cpp
iso_info.o: iso_info.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) iso_info.cpp
iso_type.o: iso_type.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) iso_type.cpp
os.o: os.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) os.cpp
tactical_decomposition.o: tactical_decomposition.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) tactical_decomposition.cpp
tdo_gradient.o: tdo_gradient.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) tdo_gradient.cpp
tdo_scheme.o: tdo_scheme.cpp ; $(CPP) -c $(CFLAGS) $(INCLUDES) tdo_scheme.cpp


erzeuger erzeuger.c:
	$(CC) erzeuger.c -o erzeuger

libgeo.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

	
clean:
	- rm -f *.o *.a  a.out erzeuger *.out *~


test_2:
	./geometry_builder.out -v 5 -no 2 cc



test_7_3:
	./geometry_builder.out -v 5 -no 22 cc

test_8_3:
	./geometry_builder.out -v 5 -no 23 cc

test_24:
	./geometry_builder.out -v 5 -no 24 cc

test_25:
	./geometry_builder.out -v 5 -no 25 cc

test_26:
	./geometry_builder.out -v 5 -no 26 cc

test_27:
	./geometry_builder.out -v 5 -no 27 cc




