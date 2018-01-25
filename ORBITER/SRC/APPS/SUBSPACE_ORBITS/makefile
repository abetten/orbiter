SRC=../..
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


all:  subspace_orbits_main.out \
	linear_set_main.out \
	desarguesian_spread.out 

desarguesian_spread.out:  desarguesian_spread.o;
	$(DISCRETA_CPP)  desarguesian_spread.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

subspace_orbits_main.out:  subspace_orbits_main.o;
	$(DISCRETA_CPP)  subspace_orbits_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

linear_set_main.out:  linear_set_main.o linear_set.o linear_set2.o ;
	$(DISCRETA_CPP) linear_set_main.o linear_set.o linear_set2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *.out *~


