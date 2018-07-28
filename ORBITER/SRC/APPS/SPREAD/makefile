SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = \
	$(SRC)/LIB/TOP_LEVEL/libtoplevel.a \
	$(SRC)/LIB/DISCRETA/libdiscreta.a \
	$(SRC)/LIB/SNAKES_AND_LADDERS/libsnakesandladders.a  \
	$(SRC)/LIB/ACTION/libaction.a \
	$(SRC)/LIB/GALOIS/libgalois.a 
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


