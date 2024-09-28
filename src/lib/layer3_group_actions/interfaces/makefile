DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	conjugacy_classes_and_normalizers.o \
	conjugacy_classes_of_subgroups.o \
	l3_interface_gap.o \
	magma_interface.o \
	nauty_interface_for_graphs.o \
	nauty_interface_with_group.o
	

all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

