DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: classify \
	isomorph \
	orbits \
	poset_classification \
	set_stabilizer \
	solver
	
	

classify : classify2 
isomorph : isomorph2 
orbits : orbits2 
poset_classification : poset_classification2 
set_stabilizer : set_stabilizer2 
solver : solver2 

classify2 :  ; cd classify; $(MAKE)
isomorph2 :  ; cd isomorph; $(MAKE)
orbits2 :  ; cd orbits; $(MAKE)
poset_classification2 :  ; cd poset_classification; $(MAKE)
set_stabilizer2 :  ; cd set_stabilizer; $(MAKE)
solver2 :  ; cd solver; $(MAKE)


clean:
	cd classify; $(MAKE) clean
	cd isomorph; $(MAKE) clean
	cd orbits; $(MAKE) clean
	cd poset_classification; $(MAKE) clean
	cd set_stabilizer; $(MAKE) clean
	cd solver; $(MAKE) clean
	- rm *.o *.a 




