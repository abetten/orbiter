SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++17
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64
#LFLAGS = -lm -lpthread -Wl,-rpath -Wl,/projects/abetten@colostate.edu/gcc-install-destdir/usr/local/gcc-8.1/lib64



TARGETS=\
		sandbox.out \
		rabbits_and_wolves.out

all: $(TARGETS)


sandbox.out:  sandbox.o ;
	$(DISCRETA_CPP)  sandbox.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rabbits_and_wolves.out:  rabbits_and_wolves.o ;
	$(DISCRETA_CPP)  rabbits_and_wolves.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@


install:
	cp $(TARGETS) ../../../bin


rabbit:
	gnuplot plotting.gnuplot
	gs -sDEVICE=pdfwrite -r120 -orabbit.pdf \
		out.eps \

clean:
	- rm *.o *.a *.out *~
	- rm data.txt
	- rm out.eps
	- rm out.pdf
	- rm log
	- rm test_hex.csv


