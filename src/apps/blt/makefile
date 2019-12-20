SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64






TARGETS=\
	blt_main.out \
	create_BLT_set_main.out \
	plot_stats_on_graphs.out \
	run_blt.out

all: $(TARGETS)



install:
	cp $(TARGETS) ../../../bin


blt_main.out:  blt_main.o ;
	$(DISCRETA_CPP)  blt_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_BLT_set_main.out:  create_BLT_set_main.o ;
	$(DISCRETA_CPP)  create_BLT_set_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plot_stats_on_graphs.out:  plot_stats_on_graphs.o ;
	$(DISCRETA_CPP)  plot_stats_on_graphs.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

run_blt.out:  run_blt.o ;
	$(DISCRETA_CPP)  run_blt.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


clean:
	- rm *.o *.a *.out *~

