DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	INT_matrix.o \
	INT_vector.o \
	data_file.o \
	fancy_set.o \
	partitionstack.o \
	set_of_sets.o \
	sorting.o \
	spreadsheet.o \
	super_fast_hash.o \
	vector_hashing.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~


#%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



