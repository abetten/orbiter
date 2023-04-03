DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	naugraph.o \
	naurng.o \
	nautil.o \
	nauty.o \
	schreier.o 


all : $(MODULES)

#%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

naugraph.o : naugraph.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
naurng.o : naurng.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
nautil.o : nautil.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
nauty.o : nauty.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
schreier.o : schreier.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.a *~

