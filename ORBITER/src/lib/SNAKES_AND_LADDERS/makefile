DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I.. -I../GALOIS -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm

all: classify \
	other \
	poset_classification \
	set_stabilizer




classify : classify2 ; 
other : other2 ; 
poset_classification : poset_classification2 ; 
set_stabilizer : set_stabilizer2 ; 

classify2 :  ; cd classify; $(MAKE)
other2 :  ; cd other; $(MAKE)
poset_classification2 :  ; cd poset_classification; $(MAKE)
set_stabilizer2 :  ; cd set_stabilizer; $(MAKE)


%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<



clean:
	cd classify; $(MAKE) clean
	cd other; $(MAKE) clean
	cd poset_classification; $(MAKE) clean
	cd set_stabilizer; $(MAKE) clean
	- rm *.o *.a 




