DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	activity_description.o \
	interface_algebra.o \
	interface_coding_theory.o \
	interface_combinatorics.o \
	interface_cryptography.o \
	interface_povray.o \
	interface_projective.o \
	interface_symbol_table.o \
	interface_toolkit.o \
	orbiter_command.o \
	orbiter_top_level_session.o \
	symbol_definition.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

%.out: %.o;
	$(DISCRETA_CPP) $< $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *~

