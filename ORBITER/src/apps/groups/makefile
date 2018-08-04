SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = $(SRC)/LIB/liborbiter.a
LFLAGS = -lm




all:  orthogonal_group.out \
		wreath_product.out


orthogonal_group.out:  orthogonal_group.o;
	$(DISCRETA_CPP)  orthogonal_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


wreath_product.out:  wreath_product.o;
	$(DISCRETA_CPP)  wreath_product.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.out *~




