DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm

MODULES = \
	base.o \
	bt_key.o \
	btree.o \
	database.o \
	design.o \
	design_parameter.o \
	design_parameter_source.o \
	discreta_global.o \
	discreta_matrix.o \
	domain.o \
	geometry.o \
	global.o \
	hollerith.o \
	integer.o \
	longinteger.o \
	memory.o \
	number_partition.o \
	page_table.o \
	permutation.o \
	solid.o \
	unipoly.o \
	vector.o
	 
		
all: $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


