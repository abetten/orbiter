SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64




TARGETS=\
	 arcs_main.out \
	arcs_orderly.out \
	awss.out \
	k_arc_generator_main.out \
	k_arc_lifting.out \
	simeon.out \
	test_hyperoval.out 

all: $(TARGETS)


install:
	cp $(TARGETS) ../../../../bin

arcs_main.out:  arcs_main.o ;
	$(DISCRETA_CPP)  arcs_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

arcs_orderly.out:  arcs_orderly.o;
	$(DISCRETA_CPP)  arcs_orderly.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

awss.out:  awss.o;
	$(DISCRETA_CPP)  awss.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

k_arc_generator_main.out:  k_arc_generator_main.o;
	$(DISCRETA_CPP)  k_arc_generator_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

k_arc_lifting.out:  k_arc_lifting.o;
	$(DISCRETA_CPP)  k_arc_lifting.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

simeon.out:  simeon.o;
	$(DISCRETA_CPP)  simeon.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

test_hyperoval.out:  test_hyperoval.o;
	$(DISCRETA_CPP)  test_hyperoval.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@





clean:
	- rm *.o *.out *~

