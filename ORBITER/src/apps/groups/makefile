SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm


all:  \
	direct_product.out \
	linear_group.out \
	orthogonal_group.out \
	wreath_product.out


direct_product.out:  direct_product.o;
	$(DISCRETA_CPP)  direct_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


linear_group.out:  linear_group.o;
	$(DISCRETA_CPP)  linear_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


orthogonal_group.out:  orthogonal_group.o;
	$(DISCRETA_CPP)  orthogonal_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


wreath_product.out:  wreath_product.o;
	$(DISCRETA_CPP)  wreath_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.out *~

