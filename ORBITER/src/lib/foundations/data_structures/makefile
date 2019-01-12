DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	int_matrix.o \
	int_vector.o \
	data_file.o \
	fancy_set.o \
	page_storage.o \
	partitionstack.o \
	set_of_sets.o \
	sorting.o \
	spreadsheet.o \
	super_fast_hash.o \
	vector_hashing.o 


all : $(MODULES)

clean:
	- rm *.o *.a *~
