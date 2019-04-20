SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm



all: arcs_main.out \
	arcs_orderly.out \
	test_hyperoval.out \
	create_group.out \
	k_arc_generator_main.out \
	k_arc_lifting.out \
	test_arc.out


arcs_main.out:  arcs_main.o ;
	$(DISCRETA_CPP)  arcs_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_hyperoval.out:  test_hyperoval.o;
	$(DISCRETA_CPP)  test_hyperoval.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_group.out:  create_group.o;
	$(DISCRETA_CPP)  create_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_arc.out:  test_arc.o;
	$(DISCRETA_CPP)  test_arc.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

k_arc_generator_main.out:  k_arc_generator_main.o;
	$(DISCRETA_CPP)  k_arc_generator_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

k_arc_lifting.out:  k_arc_lifting.o;
	$(DISCRETA_CPP)  k_arc_lifting.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

arcs_orderly.out:  arcs_orderly.o;
	$(DISCRETA_CPP)  arcs_orderly.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.out *~

