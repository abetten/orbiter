SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm





all: ovoid.out make_design.out

ovoid.out:  ovoid.o ovoid_generator.o ;
	$(DISCRETA_CPP)  ovoid.o ovoid_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_design.out:  make_design.o;
	$(DISCRETA_CPP)  make_design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.a *.out *~




