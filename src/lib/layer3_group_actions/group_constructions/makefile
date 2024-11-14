DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	direct_product.o \
	group_constructions_global.o \
	group_modification_description.o \
	linear_group.o \
	linear_group_description.o \
	modified_group_create.o \
	permutation_group_create.o \
	permutation_group_description.o \
	permutation_representation_domain.o \
	permutation_representation.o \
	polarity_extension.o \
	wreath_product.o


all : $(MODULES)

%.o: %.cpp %.c
	$(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *~

