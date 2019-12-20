SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




TARGETS=\
		cc2widor.out \
		maxfit.out \
		packing.out \
		parameters.out \
		tdo_start.out \
		tdo_refine.out \
		tdo_print.out \
		widor.out

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin


cc2widor.out:  cc2widor.o 
	$(DISCRETA_CPP)  cc2widor.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

maxfit.out:  maxfit.o ;
	$(DISCRETA_CPP)  maxfit.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

packing.out:  packing.o ;
	$(DISCRETA_CPP)  packing.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

parameters.out:  parameters.o ;
	$(DISCRETA_CPP)  parameters.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_start.out:  tdo_start.o ;
	$(DISCRETA_CPP)  tdo_start.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_refine.out:  tdo_refine.o;
	$(DISCRETA_CPP)  tdo_refine.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_refine_all.out:  tdo_refine_all.o;
	$(DISCRETA_CPP)  tdo_refine_all.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_print.out:  tdo_print.o ;
	$(DISCRETA_CPP)  tdo_print.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

widor.out:  widor.o ;
	$(DISCRETA_CPP)  widor.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.out *~
