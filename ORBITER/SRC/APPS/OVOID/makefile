SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = \
	$(SRC)/LIB/TOP_LEVEL/libtoplevel.a \
	$(SRC)/LIB/DISCRETA/libdiscreta.a \
	$(SRC)/LIB/SNAKES_AND_LADDERS/libsnakesandladders.a  \
	$(SRC)/LIB/ACTION/libaction.a \
	$(SRC)/LIB/GALOIS/libgalois.a 
LFLAGS = -lm



all: ovoid.out make_design.out

ovoid.out:  ovoid.o;
	$(DISCRETA_CPP)  ovoid.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_design.out:  make_design.o;
	$(DISCRETA_CPP)  make_design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.a *.out *~




