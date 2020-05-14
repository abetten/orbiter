SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64




TARGETS=\
	arcs_orderly.out \
	awss.out \
	k_arc_lifting.out \
	simeon.out 

all: $(TARGETS)


install:
	cp $(TARGETS) ../../../bin

%.out: %.o;
	$(DISCRETA_CPP) $< $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

# arcs_orderly.out:  arcs_orderly.o;
# 	$(DISCRETA_CPP)  arcs_orderly.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
#
# awss.out:  awss.o;
# 	$(DISCRETA_CPP)  awss.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
#
# k_arc_lifting.out:  k_arc_lifting.o;
# 	$(DISCRETA_CPP)  k_arc_lifting.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
#
# simeon.out:  simeon.o;
# 	$(DISCRETA_CPP)  simeon.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@
#





clean:
	- rm *.o *.out *~

