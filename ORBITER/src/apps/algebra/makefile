SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



all: \
	a5_in_PSL.out \
	BN_pair.out \
	borel.out \
	cheat_sheet_GF.out \
	code_cosets.out \
	counting_flags.out \
	create_element_of_order.out \
	create_element.out \
	eigenstuff.out \
	factor_cyclotomic.out \
	field_plot.out \
	find_element.out \
	finite_field.out \
	flag.out \
	get_poly.out \
	gl_classes.out \
	long_orbit.out \
	pascal_matrix.out \
	polynomial_orbits.out \
	projective_group.out  \
	reflection.out \
	subprimitive.out \
	three_squares.out


a5_in_PSL.out:  a5_in_PSL.o;
	$(DISCRETA_CPP)  a5_in_PSL.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

BN_pair.out:  BN_pair.o;
	$(DISCRETA_CPP)  BN_pair.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

borel.out:  borel.o ;
	$(DISCRETA_CPP)  borel.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cheat_sheet_GF.out:  cheat_sheet_GF.o;
	$(DISCRETA_CPP)  cheat_sheet_GF.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

code_cosets.out:  code_cosets.o ;
	$(DISCRETA_CPP)  code_cosets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

counting_flags.out:  counting_flags.o ;
	$(DISCRETA_CPP)  counting_flags.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_element_of_order.out:  create_element_of_order.o;
	$(DISCRETA_CPP)  create_element_of_order.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_element.out:  create_element.o;
	$(DISCRETA_CPP)  create_element.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

eigenstuff.out:  eigenstuff.o;
	$(DISCRETA_CPP)  eigenstuff.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

factor_cyclotomic.out:  factor_cyclotomic.o;
	$(DISCRETA_CPP)  factor_cyclotomic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

field_plot.out:  field_plot.o;
	$(DISCRETA_CPP)  field_plot.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

find_element.out:  find_element.o;
	$(DISCRETA_CPP)  find_element.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

finite_field.out:  finite_field.o ;
	$(DISCRETA_CPP)  finite_field.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

flag.out:  flag.o;
	$(DISCRETA_CPP)  flag.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

get_poly.out:  get_poly.o ;
	$(DISCRETA_CPP)  get_poly.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

gl_classes.out:  gl_classes.o ;
	$(DISCRETA_CPP)  gl_classes.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

long_orbit.out:  long_orbit.o ;
	$(DISCRETA_CPP)  long_orbit.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

pascal_matrix.out:  pascal_matrix.o ;
	$(DISCRETA_CPP)  pascal_matrix.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

polynomial_orbits.out:  polynomial_orbits.o ;
	$(DISCRETA_CPP)  polynomial_orbits.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

projective_group.out:  projective_group.o ;
	$(DISCRETA_CPP)  projective_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

reflection.out:  reflection.o;
	$(DISCRETA_CPP)  reflection.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subprimitive.out:  subprimitive.o;
	$(DISCRETA_CPP)  subprimitive.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

three_squares.out:  three_squares.o ;
	$(DISCRETA_CPP)  three_squares.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~


