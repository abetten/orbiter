DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	group_container.o \
	hash_table_subgroups.o \
	orbit_rep.o \
	orbit_transversal.o \
	schreier_vector_handler.o \
	schreier_vector.o \
	set_and_stabilizer.o \
	union_find.o \
	union_find_on_k_subsets.o \
	vector_ge_description.o \
	vector_ge.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
