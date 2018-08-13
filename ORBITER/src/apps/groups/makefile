SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm





all:  \
	direct_product.out \
	orthogonal_group.out \
	wreath_product.out


direct_product.out:  direct_product.o;
	$(DISCRETA_CPP)  direct_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

orthogonal_group.out:  orthogonal_group.o;
	$(DISCRETA_CPP)  orthogonal_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


wreath_product.out:  wreath_product.o;
	$(DISCRETA_CPP)  wreath_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.out *~




