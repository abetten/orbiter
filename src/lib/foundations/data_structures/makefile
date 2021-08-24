DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	bitmatrix.o \
	bitvector.o \
	classify_bitvectors.o \
	data_file.o \
	data_input_stream.o \
	data_structures_global.o \
	encoded_combinatorial_object.o \
	fancy_set.o \
	int_matrix.o \
	int_vec.o \
	int_vector.o \
	lint_vec.o \
	nauty_output.o \
	page_storage.o \
	partitionstack.o \
	set_builder_description.o \
	set_builder.o \
	set_of_sets_lint.o \
	set_of_sets.o \
	sorting.o \
	spreadsheet.o \
	string_tools.o \
	super_fast_hash.o \
	vector_hashing.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~
