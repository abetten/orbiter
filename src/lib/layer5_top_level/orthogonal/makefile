DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	blt_set_activity_description.o \
	blt_set_activity.o \
	blt_set_classify_activity_description.o \
	blt_set_classify_activity.o \
	blt_set_classify_description.o \
	blt_set_classify.o \
	BLT_set_create_description.o \
	BLT_set_create.o \
	blt_set_with_action.o \
	flock.o \
	orthogonal_space_activity_description.o \
	orthogonal_space_activity.o \
	orthogonal_space_with_action_description.o \
	orthogonal_space_with_action.o \
	table_of_blt_sets.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

