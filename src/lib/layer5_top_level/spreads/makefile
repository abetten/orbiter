DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	recoordinatize.o \
	spread_classify_description.o \
	spread_classify.o \
	spread_classify2.o \
	spread_create_description.o \
	spread_create.o \
	spread_lifting.o \
	spread_table_with_selection.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

