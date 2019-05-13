DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	classify_bitvectors.o \
	data_file.o \
	fancy_set.o \
	int_matrix.o \
	int_vector.o \
	page_storage.o \
	partitionstack.o \
	set_of_sets_lint.o \
	set_of_sets.o \
	sorting.o \
	spreadsheet.o \
	super_fast_hash.o \
	vector_hashing.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
