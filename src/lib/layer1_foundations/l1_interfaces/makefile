DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I. -std=c++17
# -frename-registers
#-march=native
LIB = 
LFLAGS = -lm

MODULES = \
	easy_BMP_interface.o \
	EasyBMP.o \
	eigen_interface.o \
	expression_parser_sajeeb.o \
	gnuplot_interface.o \
	interface_gap_low.o \
	interface_magma_low.o \
	latex_interface.o \
	nauty_interface.o \
	nauty_output.o \
	povray_interface.o \
	pugixml_interface.o \
	pugixml.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



