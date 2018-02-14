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



all: \
	blt_main.out \
	plot_stats_on_graphs.out 




blt_main.out:  blt_main.o blt_set.o blt_set2.o ;
	$(DISCRETA_CPP)  blt_main.o blt_set.o blt_set2.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plot_stats_on_graphs.out:  plot_stats_on_graphs.o ;
	$(DISCRETA_CPP)  plot_stats_on_graphs.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~

%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<




