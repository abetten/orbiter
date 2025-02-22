DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops -I.. -I../.. -std=c++17
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	basic_algebra1 \
	expression_parser1 \
	field_theory1 \
	linear_algebra1 \
	number_theory1 \
	ring_theory1





basic_algebra1 :  ; cd basic_algebra; $(MAKE)
expression_parser1 :  ; cd expression_parser; $(MAKE)
field_theory1 :  ; cd field_theory; $(MAKE)
linear_algebra1 :  ; cd linear_algebra; $(MAKE)
number_theory1 :  ; cd number_theory; $(MAKE)
ring_theory1 :  ; cd ring_theory; $(MAKE)






%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd basic_algebra; $(MAKE) clean
	cd expression_parser; $(MAKE) clean
	cd field_theory; $(MAKE) clean
	cd linear_algebra; $(MAKE) clean
	cd number_theory; $(MAKE) clean
	cd ring_theory; $(MAKE) clean
	- rm *.o *.a 



