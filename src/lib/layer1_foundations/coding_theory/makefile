DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
		coding_theory_domain_crc.o \
		coding_theory_domain.o \
		crc_options_description.o \
		create_BCH_code.o \
		cyclic_codes.o \
		error_repository.o \
		tensor_codes.o \
		mindist.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

