DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I..
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	dynamic_memory.o \
	file_output.o \
	memory.o \
	memory_object.o \
	orbiter_data_file.o \
	util.o


all : $(MODULES)

clean:
	- rm *.o *.a *~


#%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



