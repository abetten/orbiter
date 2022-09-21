DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	finite_field_activity_description.o \
	finite_field_activity.o \
	finite_field_description.o \
	finite_field_implementation_by_tables.o \
	finite_field_implementation_wo_tables.o \
	finite_field_io.o \
	finite_field_projective.o \
	finite_field.o \
	norm_tables.o \
	nth_roots.o \
	subfield_structure.o 


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



