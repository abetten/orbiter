SRC=../..
DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = \
	$(SRC)/LIB/TOP_LEVEL/libtoplevel.a \
	$(SRC)/LIB/DISCRETA/libdiscreta.a \
	$(SRC)/LIB/SNAKES_AND_LADDERS/libsnakesandladders.a  \
	$(SRC)/LIB/ACTION/libaction.a \
	$(SRC)/LIB/INCIDENCE/libincidence.a \
	$(SRC)/LIB/GALOIS/libgalois.a 
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



%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

clean:
	- rm *.o *.a *.out *~


