DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: classify \
	poset_classification 
	
	




classify : classify2 
poset_classification : poset_classification2 

classify2 :  ; cd classify; $(MAKE)
poset_classification2 :  ; cd poset_classification; $(MAKE)


clean:
	cd classify; $(MAKE) clean
	cd poset_classification; $(MAKE) clean
	- rm *.o *.a 




