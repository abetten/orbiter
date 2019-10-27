SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm


all: ovoid.out make_design.out

ovoid.out:  ovoid.o ;
	$(DISCRETA_CPP)  ovoid.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_design.out:  make_design.o;
	$(DISCRETA_CPP)  make_design.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

clean:
	- rm *.o *.a *.out *~

