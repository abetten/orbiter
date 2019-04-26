SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm
# specific for tx1:
#LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64


all:  semifield_classify_main.out \
	semifield_main.out 


semifield_classify_main.out:  semifield_classify_main.o  ;
	$(DISCRETA_CPP)  semifield_classify_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

semifield_main.out:  semifield_main.o  ;
	$(DISCRETA_CPP)  semifield_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~
