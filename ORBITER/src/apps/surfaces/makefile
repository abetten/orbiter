SRC=../..
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I../../lib/foundations

INCLUDES=\
			-I$(SRC)/lib -I../../lib/foundations \
			-I$(SRC)/lib -I../../lib/foundations/CUDA/
			
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64


all:  arc_lifting_main.out \
	create_surface_main.out \
	intersection.out \
	surface_classify.out \
	surfaces_arc_lifting_main.out 


arc_lifting_main.out:  arc_lifting_main.o  ;
	$(DISCRETA_CPP)  arc_lifting_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_surface_main.out:  create_surface_main.o   ;
	$(DISCRETA_CPP)  create_surface_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

intersection.out:  intersection.o   ;
	$(DISCRETA_CPP)  intersection.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

surface_classify.out:  surface_classify.o ;
	$(DISCRETA_CPP)  surface_classify.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

surfaces_arc_lifting_main.out:  surfaces_arc_lifting_main.o ;
	$(DISCRETA_CPP)  surfaces_arc_lifting_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~
