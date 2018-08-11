SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm





	

all:  maxfit.out packing.out tdo_can.out tdo_start.out tdo_refine.out tdo_print.out 

#tdo_refine_all.out 

maxfit.out:  maxfit.o ;
	$(DISCRETA_CPP)  maxfit.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

packing.out:  packing.o ;
	$(DISCRETA_CPP)  packing.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_can.out:  tdo_can.o ;
	$(DISCRETA_CPP)  tdo_can.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_start.out:  tdo_start.o ;
	$(DISCRETA_CPP)  tdo_start.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_refine.out:  tdo_refine.o;
	$(DISCRETA_CPP)  tdo_refine.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_refine_all.out:  tdo_refine_all.o;
	$(DISCRETA_CPP)  tdo_refine_all.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tdo_print.out:  tdo_print.o ;
	$(DISCRETA_CPP)  tdo_print.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.out *~




