DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	group.o \
	orbit_transversal.o \
	page_storage.o \
	projective_space_with_action.o \
	schreier_vector.o \
	set_and_stabilizer.o \
	union_find.o \
	union_find_on_k_subsets.o \
	vector_ge.o


all : $(MODULES)

clean:
	- rm *.o *.a *~
