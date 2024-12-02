DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -O3 -march=native -funroll-loops 
# -frename-registers
LIB = 
LFLAGS = -lm


# expression_parser_sajeeb must come before expression_parser
# because the file expression_parser_sajeeb/parse_tab.h 
# must be generated
# before it is used in expression_parser


all: \
  expression_parser_sajeeb \
	algebra \
	combinatorics \
	expression_parser_sajeeb \
	geometry \
	other




algebra : algebra2 
combinatorics : combinatorics2  
expression_parser_sajeeb : expression_parser_sajeeb2
geometry : geometry2 
other : other2 

algebra2 : ; cd algebra; $(MAKE)
combinatorics2 : ; cd combinatorics; $(MAKE)
expression_parser_sajeeb2 : ; $(MAKE) -C expression_parser_sajeeb all
geometry2 : ; cd geometry; $(MAKE)
other2 : ; cd other; $(MAKE)

clean:
	cd algebra; $(MAKE) clean
	cd combinatorics; $(MAKE) clean
	cd expression_parser_sajeeb; $(MAKE) clean
	cd geometry; $(MAKE) clean
	cd other; $(MAKE) clean
	- rm *.o *.a 




