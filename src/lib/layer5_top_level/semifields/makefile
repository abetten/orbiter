DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	semifield_classify_description.o \
	semifield_classify_with_substructure.o \
	semifield_classify.o \
	semifield_downstep_node.o \
	semifield_flag_orbit_node.o \
	semifield_level_two.o \
	semifield_lifting.o \
	semifield_substructure.o \
	semifield_trace.o \
	trace_record.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

