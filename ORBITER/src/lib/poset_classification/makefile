DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: classify \
	other \
	snakes_and_ladders \
	set_stabilizer




classify : classify2 ; 
other : other2 ; 
snakes_and_ladders : snakes_and_ladders2 ; 
set_stabilizer : set_stabilizer2 ; 

classify2 :  ; cd classify; $(MAKE)
other2 :  ; cd other; $(MAKE)
snakes_and_ladders2 :  ; cd snakes_and_ladders; $(MAKE)
set_stabilizer2 :  ; cd set_stabilizer; $(MAKE)


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd classify; $(MAKE) clean
	cd other; $(MAKE) clean
	cd snakes_and_ladders; $(MAKE) clean
	cd set_stabilizer; $(MAKE) clean
	- rm *.o *.a 




