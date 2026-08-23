

help:
	java -jar ~/bin/rr.war

all:
	java -jar ~/bin/rr.war grammar.ebnf >grammar.html


all_png:
	java -jar ~/bin/rr.war -png -noembedded grammar.ebnf  >grammar.zip

orbiter:
	java -jar ~/bin/rr.war orbiter_grammar.ebnf >orbiter_grammar.html


orbiter_png:
	java -jar ~/bin/rr.war -png -noembedded orbiter_grammar.ebnf >orbiter_grammar.zip
	unzip orbiter_grammar.zip

