SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




TARGETS=\
	dio.out \
	dlx.out \
	pentomino_5x5.out \
	puzzle.out \
	solve_diophant.out 


all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin



#%.o: %.C %.cpp; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
#
#%.out: %.o; 
#	@echo "------------------------------------------------------------------"
#	@echo -n "$<"
#	@echo -n ": "
#	$(DISCRETA_CPP) $(CPPFLAGS)  $(LIB) $(LFLAGS)  $<
#	@echo "------------------------------------------------------------------"


dio.out:  dio.o 
	$(DISCRETA_CPP)  dio.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

dlx.out:  dlx.o
	$(DISCRETA_CPP)  dlx.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

pentomino_5x5.out:  pentomino_5x5.o;
	$(DISCRETA_CPP)  pentomino_5x5.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

puzzle.out:  puzzle.o;
	$(DISCRETA_CPP)  puzzle.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

solve_diophant.out:  solve_diophant.o
	$(DISCRETA_CPP)  solve_diophant.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




clean:
	@echo "------------------------------------------------------------------"
	@echo -n "cleaning"
	@echo -n ": "
	- rm *.o *.a *.out *~

