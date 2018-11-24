SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



all: \
	BN_pair.out \
	cheat_sheet_GF.out \
	conjugate.out \
	conjugate_O4.out \
	create_element.out \
	create_element_of_order.out \
	eigenstuff.out \
	factor_cyclotomic.out \
	find_element.out \
	get_poly.out \
	gl_classes.out \
	move_point_orthogonal.out \
	reflection.out \
	subprimitive.out \
	test_group.out \
	matrix.out \
	projective_group.out  \
	pascal_matrix.out \
	flag.out \
	borel.out \
	counting_flags.out \
	three_squares.out \
	code_cosets.out 

BN_pair.out:  BN_pair.o;
	$(DISCRETA_CPP)  BN_pair.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cheat_sheet_GF.out:  cheat_sheet_GF.o;
	$(DISCRETA_CPP)  cheat_sheet_GF.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

conjugate.out:  conjugate.o;
	$(DISCRETA_CPP)  conjugate.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

conjugate_O4.out:  conjugate_O4.o;
	$(DISCRETA_CPP)  conjugate_O4.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_element.out:  create_element.o;
	$(DISCRETA_CPP)  create_element.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_element_of_order.out:  create_element_of_order.o;
	$(DISCRETA_CPP)  create_element_of_order.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

eigenstuff.out:  eigenstuff.o;
	$(DISCRETA_CPP)  eigenstuff.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

factor_cyclotomic.out:  factor_cyclotomic.o;
	$(DISCRETA_CPP)  factor_cyclotomic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

find_element.out:  find_element.o;
	$(DISCRETA_CPP)  find_element.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

get_poly.out:  get_poly.o ;
	$(DISCRETA_CPP)  get_poly.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

gl_classes.out:  gl_classes.o ;
	$(DISCRETA_CPP)  gl_classes.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

move_point_orthogonal.out:  move_point_orthogonal.o;
	$(DISCRETA_CPP)  move_point_orthogonal.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

reflection.out:  reflection.o;
	$(DISCRETA_CPP)  reflection.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subprimitive.out:  subprimitive.o;
	$(DISCRETA_CPP)  subprimitive.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_group.out:  test_group.o;
	$(DISCRETA_CPP)  test_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

matrix.out:  matrix.o ;
	$(DISCRETA_CPP)  matrix.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

projective_group.out:  projective_group.o ;
	$(DISCRETA_CPP)  projective_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

pascal_matrix.out:  pascal_matrix.o ;
	$(DISCRETA_CPP)  pascal_matrix.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

flag.out:  flag.o;
	$(DISCRETA_CPP)  flag.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

borel.out:  borel.o ;
	$(DISCRETA_CPP)  borel.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

counting_flags.out:  counting_flags.o ;
	$(DISCRETA_CPP)  counting_flags.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

code_cosets.out:  code_cosets.o ;
	$(DISCRETA_CPP)  code_cosets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

three_squares.out:  three_squares.o ;
	$(DISCRETA_CPP)  three_squares.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~


