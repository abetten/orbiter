SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = $(SRC)/LIB/liborbiter.a
LFLAGS = -lm






all: spread_classify.out \
	spread_create.out



spread_classify.out: spread_classify.o  ;
	$(DISCRETA_CPP) spread_classify.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

spread_create.out: spread_create.o ;
	$(DISCRETA_CPP) spread_create.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.a *.out *~


