SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I$(SRC)/lib/DISCRETA
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




all: regular_ls.out transpose.out




regular_ls.out:  regular_ls.o regular_ls_generator.o;
	$(DISCRETA_CPP)  regular_ls.o regular_ls_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

transpose.out:  transpose.o regular_ls_generator.o;
	$(DISCRETA_CPP)  transpose.o regular_ls_generator.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


