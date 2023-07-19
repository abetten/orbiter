DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -std=c++17
# -frename-registers
#-march=native 
LIB = 
LFLAGS = -lm

MODULES = \
		code_diagram.o \
		coding_theory_domain_crc.o \
		coding_theory_domain.o \
		crc_code_description.o \
		crc_codes_search.o \
		crc_codes.o \
		crc_object.o \
		crc_options_description.o \
		create_BCH_code.o \
		create_RS_code.o \
		cyclic_codes.o \
		error_pattern_generator.o \
		error_repository.o \
		mindist.o \
		ttp_codes.o 


all : $(MODULES)

%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

