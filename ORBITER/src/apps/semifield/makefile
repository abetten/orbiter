SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm
# specific for tx1:
#LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64




TARGETS=\
	deep_search.out \
	isomorph_testing.out \
	read_solutions.out \
	read_types.out \
	semifield_classify_main.out \
	semifield_main.out 

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin



deep_search.out:  deep_search.o  ;
	$(DISCRETA_CPP)  deep_search.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

isomorph_testing.out:  isomorph_testing.o  ;
	$(DISCRETA_CPP)  isomorph_testing.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

read_solutions.out:  read_solutions.o  ;
	$(DISCRETA_CPP)  read_solutions.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

read_types.out:  read_types.o  ;
	$(DISCRETA_CPP)  read_types.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

semifield_classify_main.out:  semifield_classify_main.o  ;
	$(DISCRETA_CPP)  semifield_classify_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

semifield_main.out:  semifield_main.o  ;
	$(DISCRETA_CPP)  semifield_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~
