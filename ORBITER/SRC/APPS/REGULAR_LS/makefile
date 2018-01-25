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



all: regular_ls.out transpose.out




regular_ls.out:  regular_ls.o regular_ls_generator.o;
	$(DISCRETA_CPP)  regular_ls.o regular_ls_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

transpose.out:  transpose.o regular_ls_generator.o;
	$(DISCRETA_CPP)  transpose.o regular_ls_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


