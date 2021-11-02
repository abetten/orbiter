DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I. -std=c++14
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	a_domain.o \
	algebra_global.o \
	generators_symplectic_group.o \
	gl_class_rep.o \
	gl_classes.o \
	group_generators_domain.o \
	heisenberg.o \
	matrix_block_data.o \
	null_polarity_generator.o \
	rank_checker.o \
	vector_space.o


all : $(MODULES)



%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	- rm *.o *.a *~



