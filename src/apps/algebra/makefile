SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




TARGETS=\
	a5_in_PSL.out \
	BN_pair.out \
	borel.out \
	burnside.out \
	cheat_sheet_GF.out \
	code_cosets.out \
	create_element_of_order.out \
	create_element.out \
	eigenstuff.out \
	factor_cyclotomic.out \
	field_plot.out \
	find_element.out \
	flag.out \
	get_poly.out \
	gl_classes.out \
	polynomial_orbits.out \
	reflection.out

all: $(TARGETS)


install:
	cp $(TARGETS) ../../../bin


%.out: %.o;
	$(DISCRETA_CPP) $< $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@





clean:
	- rm *.o *.a *.out *~


