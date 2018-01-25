DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
LIB = 
LFLAGS = -lm
CPP_PREFIX=e_
UNIX_TO_DOS = ~/bin/unix2dos -n 

MODULES = \
	base.o \
	bitmatrix.o \
	bt_key.o \
	btree.o \
	database.o \
	design.o \
	design_parameter.o \
	design_parameter_source.o \
	discreta_global.o \
	domain.o \
	geometry.o \
	global.o \
	group_selection.o \
	hollerith.o \
	integer.o \
	longinteger.o \
	matrix.o \
	memory.o \
	number_partition.o \
	page_table.o \
	permutation.o \
	perm_group_gens.o \
	solid.o \
	unipoly.o \
	vector.o
	 
		

LIB = libdiscreta.a

libdiscreta.a: $(MODULES) ;
	ar ruvs $(LIB) $(MODULES)

clean:
	- rm *.o *.a *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

cpp:
	$(UNIX_TO_DOS) base.C $(CPP_PREFIX)base.cpp
	$(UNIX_TO_DOS) bitmatrix.C $(CPP_PREFIX)bitmatrix.cpp
	$(UNIX_TO_DOS) bt_key.C $(CPP_PREFIX)bt_key.cpp
	$(UNIX_TO_DOS) btree.C $(CPP_PREFIX)btree.cpp
	$(UNIX_TO_DOS) database.C $(CPP_PREFIX)database.cpp
	$(UNIX_TO_DOS) design.C $(CPP_PREFIX)design.cpp
	$(UNIX_TO_DOS) design_parameter.C $(CPP_PREFIX)design_parameter.cpp
	$(UNIX_TO_DOS) design_parameter_source.C $(CPP_PREFIX)design_parameter_source.cpp
	$(UNIX_TO_DOS) domain.C $(CPP_PREFIX)domain.cpp
	$(UNIX_TO_DOS) geometry.C $(CPP_PREFIX)geometry.cpp
	$(UNIX_TO_DOS) global.C $(CPP_PREFIX)global.cpp
	$(UNIX_TO_DOS) group_selection.C $(CPP_PREFIX)group_selection.cpp
	$(UNIX_TO_DOS) hollerith.C $(CPP_PREFIX)hollerith.cpp
	$(UNIX_TO_DOS) integer.C $(CPP_PREFIX)integer.cpp
	$(UNIX_TO_DOS) longinteger.C $(CPP_PREFIX)longinteger.cpp
	$(UNIX_TO_DOS) matrix.C $(CPP_PREFIX)matrix.cpp
	$(UNIX_TO_DOS) memory.C $(CPP_PREFIX)memory.cpp
	$(UNIX_TO_DOS) number_partition.C $(CPP_PREFIX)number_partition.cpp
	$(UNIX_TO_DOS) orbit.C $(CPP_PREFIX)orbit.cpp
	$(UNIX_TO_DOS) page_table.C $(CPP_PREFIX)page_table.cpp
	$(UNIX_TO_DOS) perm_group_gens.C $(CPP_PREFIX)perm_group_gens.cpp
	$(UNIX_TO_DOS) permutation.C $(CPP_PREFIX)permutation.cpp
	$(UNIX_TO_DOS) snakesandladders.C $(CPP_PREFIX)snakesandladders.cpp
	$(UNIX_TO_DOS) solid.C $(CPP_PREFIX)solid.cpp
	$(UNIX_TO_DOS) unipoly.C $(CPP_PREFIX)unipoly.cpp
	$(UNIX_TO_DOS) vector.C $(CPP_PREFIX)vector.cpp
	$(UNIX_TO_DOS) discreta.h ../WINDOWS/discreta.h
	mv $(CPP_PREFIX)* ../WINDOWS
	

