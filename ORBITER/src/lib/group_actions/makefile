DISCRETA_CPP = g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: \
	actions \
	data_structures \
	groups \
	induced_actions




actions : actions2 
data_structures : data_structures2 
groups : groups2 
induced_actions : induced_actions2 

actions2 :  ; cd actions; $(MAKE)
data_structures2 :  ; cd data_structures; $(MAKE)
groups2 :  ; cd groups; $(MAKE)
induced_actions2 :  ; cd induced_actions; $(MAKE)


clean:
	cd actions; $(MAKE) clean
	cd data_structures; $(MAKE) clean
	cd groups; $(MAKE) clean
	cd induced_actions; $(MAKE) clean
	- rm *.o *.a 




