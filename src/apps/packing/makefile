SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm
# specific for tx1:
#LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64







TARGETS=\
	packing_main.out \
	packing_was_main.out 



all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin



packing_main.out:  packing_main.o;
	$(DISCRETA_CPP)  packing_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

packing_was_main.out:  packing_was_main.o;
	$(DISCRETA_CPP)  packing_was_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@





clean:
	- rm *.o *.a *.out *~

