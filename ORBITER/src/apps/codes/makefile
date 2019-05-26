SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm

n=24
k=12
d=8
q=2
v=0

all: codes.out 

codes.out:  codes.o code_generator.o ;
	$(DISCRETA_CPP)  codes.o code_generator.o \
		$(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	- rm *.o *.a *.out *~



#------------------------------------------------------------------------------
# Testing code
#------------------------------------------------------------------------------

test:
	./codes.out -v $(v) -n $(n) -k $(k) -q $(q) -d $(d) -w -lex
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
	open codes_linear_n$(n)_k$(k)_q$(q)_d$(d).ps
	
clean_test:
	rm -rf *.log *.mp *.commands *.layered_graph *.1 *.tex *.txt *.csv 
	rm -rf *.aux *.ps *.dvi ./CODES*

#------------------------------------------------------------------------------
