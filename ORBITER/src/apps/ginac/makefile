SRC=../..
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -I../../lib/foundations -std=c++11

INCLUDES=\
			-I$(SRC)/lib -I../../lib/foundations \
			-I$(SRC)/lib -I../../lib/foundations/CUDA/
			
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64 -lginac -lcln





ALL_TARGETS=\
	cra.out \
	differentiation.out \
	hilbert_cohn_vossen.out \
	mersenne.out \
	solve_linear.out 


all: $(ALL_TARGETS)



cra.out:  cra.o  ;
	$(DISCRETA_CPP)  cra.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


differentiation.out:  differentiation.o  ;
	$(DISCRETA_CPP)  differentiation.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


hilbert_cohn_vossen.out:  hilbert_cohn_vossen.o  ;
	$(DISCRETA_CPP)  hilbert_cohn_vossen.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


mersenne.out:  mersenne.o  ;
	$(DISCRETA_CPP)  mersenne.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

solve_linear.out:  solve_linear.o  ;
	$(DISCRETA_CPP)  solve_linear.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



install:
	cp $(ALL_TARGETS) ../../../../bin






clean:
	- rm *.o *.out *~

