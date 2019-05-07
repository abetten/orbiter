SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm
# specific for tx1:
#LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64


all:  deep_search.out \
	semifield_classify_main.out \
	semifield_main.out 


deep_search.out:  deep_search.o  ;
	$(DISCRETA_CPP)  deep_search.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

semifield_classify_main.out:  semifield_classify_main.o  ;
	$(DISCRETA_CPP)  semifield_classify_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

semifield_main.out:  semifield_main.o  ;
	$(DISCRETA_CPP)  semifield_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~
