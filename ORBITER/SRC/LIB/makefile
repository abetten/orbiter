DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall
LIB = 
LFLAGS = -lm


all: galois action snakesandladders discreta toplevel


galois : galois2 ; 
action : action2 ; 
snakesandladders : snakesandladders2 ; 
discreta : discreta2 ; 
toplevel : toplevel2 ; 

galois2 : ; cd GALOIS; $(MAKE)
action2 : ; cd ACTION; $(MAKE)
snakesandladders2 : ; cd SNAKES_AND_LADDERS; $(MAKE)
discreta2 : ; cd DISCRETA; $(MAKE)
toplevel2 : ; cd TOP_LEVEL; $(MAKE)


clean:
	cd GALOIS; $(MAKE) clean
	cd ACTION; $(MAKE) clean
	cd SNAKES_AND_LADDERS; $(MAKE) clean
	cd DISCRETA; $(MAKE) clean
	cd TOP_LEVEL; $(MAKE) clean
	- rm *.o *.a 







