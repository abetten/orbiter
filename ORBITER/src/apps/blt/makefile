SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm





all: \
	blt_main.out \
	plot_stats_on_graphs.out \
	create_BLT_set_main.out




blt_main.out:  blt_main.o ;
	$(DISCRETA_CPP)  blt_main.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plot_stats_on_graphs.out:  plot_stats_on_graphs.o ;
	$(DISCRETA_CPP)  plot_stats_on_graphs.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_BLT_set_main.out:  create_BLT_set_main.o ;
	$(DISCRETA_CPP)  create_BLT_set_main.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~

