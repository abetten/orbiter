DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm
CPP_PREFIX=d_
UNIX_TO_DOS = ~/bin/unix2dos -n 

MODULES = \
	dynamic_memory.o \
	geo_parameter.o \
	geometric_tests.o \
	inc_gen_global.o \
	incidence_global.o \
	packing.o \
	point_line.o \
	refine_rows.o \
	refine_columns.o \
	refine_3design.o \
	tdo_scheme.o \
	tdo_data.o 





LIB = libincidence.a

libincidence.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<






cpp:
	$(UNIX_TO_DOS) dynamic_memory.C $(CPP_PREFIX)dynamic_memory.cpp
	$(UNIX_TO_DOS) geo_parameter.C $(CPP_PREFIX)geo_parameter.cpp
	$(UNIX_TO_DOS) geometric_tests.C $(CPP_PREFIX)geometric_tests.cpp
	$(UNIX_TO_DOS) inc_gen_global.C $(CPP_PREFIX)inc_gen_global.cpp
	$(UNIX_TO_DOS) incidence_global.C $(CPP_PREFIX)incidence_global.cpp
	$(UNIX_TO_DOS) nauty_interface.C $(CPP_PREFIX)nauty_interface.cpp
	$(UNIX_TO_DOS) packing.C $(CPP_PREFIX)packing.cpp
	$(UNIX_TO_DOS) point_line.C $(CPP_PREFIX)plane.cpp
	$(UNIX_TO_DOS) refine_3design.C $(CPP_PREFIX)refine_3design.cpp
	$(UNIX_TO_DOS) refine_columns.C $(CPP_PREFIX)refine_columns.cpp
	$(UNIX_TO_DOS) refine_rows.C $(CPP_PREFIX)refine_rows.cpp
	$(UNIX_TO_DOS) tdo_data.C $(CPP_PREFIX)tdo_data.cpp
	$(UNIX_TO_DOS) tdo_scheme.C $(CPP_PREFIX)tdo_scheme.cpp
	$(UNIX_TO_DOS) incidence.h ../WINDOWS/incidence.h
	mv $(CPP_PREFIX)* ../WINDOWS
	

