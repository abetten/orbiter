DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall
LIB = 
LFLAGS = -lm


all: galois action incidence snakesandladders discreta toplevel

cpp: galois_cpp action_cpp incidence_cpp snakesandladders_cpp discreta_cpp toplevel_cpp

galois : galois2 ; cd GALOIS; $(MAKE)
action : action2 ; cd ACTION; $(MAKE)
incidence : incidence2 ; cd INCIDENCE; $(MAKE)
snakesandladders : snakesandladders2 ; cd SNAKES_AND_LADDERS; $(MAKE)
discreta : discreta2 ; cd DISCRETA; $(MAKE)
toplevel : toplevel2 ; cd TOP_LEVEL; $(MAKE)

galois2 : ; cd GALOIS; $(MAKE)
action2 : ; cd ACTION; $(MAKE)
incidence2 : ; cd INCIDENCE; $(MAKE)
snakesandladders2 : ; cd SNAKES_AND_LADDERS; $(MAKE)
discreta2 : ; cd DISCRETA; $(MAKE)
toplevel2 : ; cd TOP_LEVEL; $(MAKE)

galois_cpp : galois_cpp2 ; cd GALOIS; $(MAKE) cpp
action_cpp : action_cpp2 ; cd ACTION; $(MAKE) cpp
incidence_cpp : incidence_cpp2 ; cd INCIDENCE; $(MAKE) cpp
snakesandladders_cpp : snakesandladders_cpp2 ; cd SNAKES_AND_LADDERS; $(MAKE) cpp
discreta_cpp : discreta_cpp2 ; cd DISCRETA; $(MAKE) cpp
toplevel_cpp : toplevel_cpp2 ; cd TOP_LEVEL; $(MAKE) cpp


galois_cpp2 : ; cd GALOIS; $(MAKE) cpp
action_cpp2 : ; cd ACTION; $(MAKE) cpp
incidence_cpp2 : ; cd INCIDENCE; $(MAKE) cpp
snakesandladders_cpp2 : ; cd SNAKES_AND_LADDERS; $(MAKE) cpp
discreta_cpp2 : ; cd DISCRETA; $(MAKE) cpp
toplevel_cpp2 : ; cd TOP_LEVEL; $(MAKE) cpp


clean:
	cd GALOIS; $(MAKE) clean
	cd ACTION; $(MAKE) clean
	cd INCIDENCE; $(MAKE) clean
	cd SNAKES_AND_LADDERS; $(MAKE) clean
	cd DISCRETA; $(MAKE) clean
	cd TOP_LEVEL; $(MAKE) clean
	- rm *.o *.a 







