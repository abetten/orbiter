#makefile for Jeffrey Leon's code
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CC_FLAGS = -Wall -O3 -funroll-loops -I.. -DSUN_UNIX_GCC -DINT_SIZE=32 -DCPU_TIME=cpuTime -DTICK=1000 -DLONG_EXTERNAL_NAMES 
# -frename-registers
LIB = 
LFLAGS = -lm

MODULES = \
	addsgen.o \
	bitmanp.o \
	ccent.o \
	ccommut.o \
	cdesauto.o \
	chbase.o \
	cinter.o \
	cmatauto.o \
	code.o \
	compcrep.o \
	compsg.o \
	copy.o \
	cparstab.o \
	cputime.o \
	csetstab.o \
	cstborb.o \
	cstrbas.o \
	cuprstab.o \
	errmesg.o \
	essentia.o \
	factor.o \
	field.o \
	inform.o \
	matrix.o \
	new.o \
	oldcopy.o \
	optsvec.o \
	orbit.o \
	orbrefn.o \
	partn.o \
	permgrp.o \
	permut.o \
	primes.o \
	ptstbref.o \
	randgrp.o \
	randschr.o \
	readdes.o \
	readgrp.o \
	readpar.o \
	readper.o \
	readpts.o \
	relator.o \
	rprique.o \
	stcs.o \
	storage.o \
	token.o \
	util.o

#	cent.o \
#	cjrndper.o \
#	commut.o \
#	compgrp.o \
#	desauto.o \
#	fndelt.o \
#	generate.o \
#	inter.o \
#	orbdes.o \
#	orblist.o \
#	randobj.o \
#	setstab.o \
#	wtdist.o



all : $(MODULES)

#%.o: %.cpp %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<

addsgen.o : addsgen.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
bitmanp.o : bitmanp.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
ccent.o : ccent.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
ccommut.o : ccommut.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cdesauto.o : cdesauto.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#cent.o : cent.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
chbase.o : chbase.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cinter.o : cinter.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#cjrndper.o : cjrndper.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cmatauto.o : cmatauto.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
code.o : code.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#commut.o : commut.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
compcrep.o : compcrep.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#compgrp.o : compgrp.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
compsg.o : compsg.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
copy.o : copy.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cparstab.o : cparstab.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cputime.o : cputime.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
csetstab.o : csetstab.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cstborb.o : cstborb.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cstrbas.o : cstrbas.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
cuprstab.o : cuprstab.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#desauto.o : desauto.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
errmesg.o : errmesg.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
essentia.o : essentia.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
factor.o : factor.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
field.o : field.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#fndelt.o : fndelt.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#generate.o : generate.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
inform.o : inform.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#inter.o : inter.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
matrix.o : matrix.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
new.o : new.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
oldcopy.o : oldcopy.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
optsvec.o : optsvec.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#orbdes.o : orbdes.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
orbit.o : orbit.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#orblist.o : orblist.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
orbrefn.o : orbrefn.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
partn.o : partn.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
permgrp.o : permgrp.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
permut.o : permut.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
primes.o : primes.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
ptstbref.o : ptstbref.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
randgrp.o : randgrp.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#randobj.o : randobj.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
randschr.o : randschr.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
readdes.o : readdes.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
readgrp.o : readgrp.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
readpar.o : readpar.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
readper.o : readper.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
readpts.o : readpts.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
relator.o : relator.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
rprique.o : rprique.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#setstab.o : setstab.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
stcs.o : stcs.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
storage.o : storage.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
token.o : token.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
util.o : util.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<
#wtdist.o : wtdist.c; $(DISCRETA_CC) $(CC_FLAGS) -c  -o $@ $<




clean:
	- rm *.o *.a *~



