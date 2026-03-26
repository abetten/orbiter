DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	boolean_function_classify.o \
	combinatorial_object_stream.o \
	combinatorics_global.o \
	dd_lifting.o \
	dd_search_singletons.o \
	delandtsheer_doyen_description.o \
	delandtsheer_doyen.o \
	design_create_description.o \
	design_create.o \
	design_tables.o \
	difference_set_in_heisenberg_group.o \
	hadamard_classify.o \
	hall_system_classify.o \
	large_set_classify.o \
	large_set_was_description.o \
	large_set_was.o \
	plesken_ring.o \
	variety_with_TDO_and_TDA.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

