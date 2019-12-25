SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




TARGETS=\
	codes.out 

all: $(TARGETS)

codes.out:  codes.o  ;
	$(DISCRETA_CPP)  codes.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

install:
	cp $(TARGETS) ../../../bin



clean:
	- rm *.o *.a *.out *~



#------------------------------------------------------------------------------
# Testing code
#------------------------------------------------------------------------------

n=5
k=2
d=8
q=2
v=0

test:
	./codes.out -v $(v) -n $(n) -k $(k) -q $(q) -d $(d)
	echo $$?
	

draw_test:
	./codes.out -v $(v) -n $(n) -k $(k) -q $(q) -d $(d) -w -lex \
		-draw 12 \
		-draw_poset \
		-export_schreier_trees \
		-tools_path ../graph_theory \
		-report
	latex  codes_linear_n$(n)_k$(k)_q$(q)_d$(d).tex
	dvips  codes_linear_n$(n)_k$(k)_q$(q)_d$(d).dvi -o
	evince codes_linear_n$(n)_k$(k)_q$(q)_d$(d).ps
	
clean_test:
	rm -rf *.log *.mp *.commands *.layered_graph *.1 *.tex *.txt *.csv 
	rm -rf *.aux *.ps *.dvi ./CODES* *.mpx

#------------------------------------------------------------------------------
