SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++14
LIB = $(SRC)/lib/liborbiter.a
#LFLAGS = -lm 
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64






TARGETS=\
		canonical_form_of_design.out \
		costas.out \
		design_apply.out \
		design_create_main.out \
		design.out \
		ferdinand.out \
		hadamard.out \
		hall_system_main.out \
		poset_of_subsets.out \
		rank_anything.out \
		rank_subsets_lex.out \
		tao.out \
		unrank.out 

all: $(TARGETS)

install:
	cp $(TARGETS) ../../../bin


canonical_form_of_design.out:  canonical_form_of_design.o;
	$(DISCRETA_CPP)  canonical_form_of_design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

costas.out:  costas.o;
	$(DISCRETA_CPP)  costas.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design_apply.out:  design_apply.o;
	$(DISCRETA_CPP)  design_apply.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design_create_main.out:  design_create_main.o;
	$(DISCRETA_CPP)  design_create_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

design.out:  design.o;
	$(DISCRETA_CPP)  design.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ferdinand.out:  ferdinand.o;
	$(DISCRETA_CPP)  ferdinand.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hadamard.out:  hadamard.o;
	$(DISCRETA_CPP)  hadamard.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hall_system_main.out:  hall_system_main.o;
	$(DISCRETA_CPP)  hall_system_main.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

poset_of_subsets.out:  poset_of_subsets.o;
	$(DISCRETA_CPP)  poset_of_subsets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_anything.out:  rank_anything.o;
	$(DISCRETA_CPP)  rank_anything.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

rank_subsets_lex.out:  rank_subsets_lex.o;
	$(DISCRETA_CPP)  rank_subsets_lex.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

tao.out:  tao.o;
	$(DISCRETA_CPP)  tao.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

unrank.out:  unrank.o;
	$(DISCRETA_CPP)  unrank.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~



	
