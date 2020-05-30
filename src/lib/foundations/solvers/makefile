DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	diophant_activity_description.o \
	diophant_activity.o \
	diophant_create.o \
	diophant_description.o \
	diophant.o \
	dlx.o \
	mckay.o


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~





