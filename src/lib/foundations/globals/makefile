DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	function_command.o \
	function_polish_description.o \
	function_polish.o \
	magma_interface.o \
	numerics.o \
	orbiter_session.o \
	orbiter_symbol_table_entry.o \
	orbiter_symbol_table.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

