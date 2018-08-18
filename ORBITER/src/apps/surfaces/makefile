SRC=../../../../ORBITER/src
DISCRETA_CPP = /usr/bin/g++
DISCRETA_CC = /usr/bin/gcc
CPPFLAGS = -Wall -I$(SRC)/lib
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm


all:  arc_lifting_main.out \
	create_surface_main.out \
	surface_classify.out


arc_lifting_main.out:  arc_lifting_main.o  ;
	$(DISCRETA_CPP)  arc_lifting_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_surface_main.out:  create_surface_main.o   ;
	$(DISCRETA_CPP)  create_surface_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

surface_classify.out:  surface_classify.o ;
	$(DISCRETA_CPP)  surface_classify.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~
