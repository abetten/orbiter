SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = \
	$(SRC)/LIB/TOP_LEVEL/libtoplevel.a \
	$(SRC)/LIB/DISCRETA/libdiscreta.a \
	$(SRC)/LIB/SNAKES_AND_LADDERS/libsnakesandladders.a  \
	$(SRC)/LIB/ACTION/libaction.a \
	$(SRC)/LIB/INCIDENCE/libincidence.a \
	$(SRC)/LIB/GALOIS/libgalois.a 
LFLAGS = -lm


all: arcs_main.out \
	test_hyperoval.out \
	create_group.out \
	test_arc.out \
	k_arc_lifting.out \
	k_arc_generator_main.out \
	arcs_orderly.out 


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




%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.out *~

