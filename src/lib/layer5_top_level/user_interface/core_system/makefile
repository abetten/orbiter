DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	orbiter_top_level_session.o \
	symbol_definition.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~





