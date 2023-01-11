SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
CC_FLAGS = -Wall -I$(SRC)/lib/layer1_foundations/Jeffrey_Leon_partition_backtrack -DINT_SIZE=32 -DCPU_TIME=cpuTime -DTICK=1000 -DLONG_EXTERNAL_NAMES 
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64
#LFLAGS = -lm -lpthread -Wl,-rpath -Wl,/projects/abetten@colostate.edu/gcc-install-destdir/usr/local/gcc-8.1/lib64



TARGETS=\
		cent \
		cjrndper \
		commut \
		compgrp \
		desauto \
		fndelt \
		generate \
		inter \
		orbdes \
		orblist \
		randobj \
		setstab \
		wtdist




all: $(TARGETS)


cent: 
	$(DISCRETA_CC) cent.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

cjrndper: 
	$(DISCRETA_CC) cjrndper.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

commut: 
	$(DISCRETA_CC) commut.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

compgrp: 
	$(DISCRETA_CC) compgrp.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

desauto: 
	$(DISCRETA_CC) desauto.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

fndelt: 
	$(DISCRETA_CC) fndelt.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

generate: 
	$(DISCRETA_CC) generate.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

inter: 
	$(DISCRETA_CC) inter.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

orbdes: 
	$(DISCRETA_CC) orbdes.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

orblist: 
	$(DISCRETA_CC) orblist.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

randobj: 
	$(DISCRETA_CC) randobj.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

setstab: 
	$(DISCRETA_CC) setstab.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@

wtdist: 
	$(DISCRETA_CC) wtdist.c  $(CC_FLAGS) $(LIB) $(LFLAGS) $(SRC)/lib/libjeffreyleon.a -o $@



install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~ cent \
		cjrndper \
		commut \
		compgrp \
		desauto \
		fndelt \
		generate \
		inter \
		orbdes \
		orblist \
		randobj \
		setstab \
		wtdist

