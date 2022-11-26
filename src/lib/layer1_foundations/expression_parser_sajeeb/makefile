include compilers.mk
include compiler-flags.mk
include os-detect.mk
include auto_dependency_generation_variables.mk

TARGET_OBJECT_FILES = main.o \
						lexer.yy.o \
						parser.tab.o \
						IRTree/node.o \
						./Visitors/*.o \
						./Visitors/LatexVisitors/*.o \
						./Visitors/ToStringVisitors/*.o \
						./Visitors/ExpansionVisitors/*.o \
						./Visitors/CopyVisitors/*.o \
						./Visitors/PrintVisitors/*.o \
						./Visitors/ReductionVisitors/*.o \
						./Visitors/EvaluateVisitors/*.o


TEX_FILES = $(wildcard visitor_result/*.tex)
EXECUTABLE_BINARY = main


main: visitors ir_tree | parser.tab.o lexer.yy.o $(SRCS:.cpp=.o)
	$(CC) $(CSTD) $(CPPFLAGS) $(TARGET_OBJECT_FILES) -o $@ $(LDFLAGS) -L ../../ -lorbiter $(LEXLIB) -lpthread

all: visitors ir_tree | parser.tab.o lexer.yy.o $(SRCS:.cpp=.o);

lexer.yy.o: lexer.yy.h 
	$(CC) $(CSTD) $(CPPFLAGS) $(@:%.o=%.cpp) -c -o $@

lexer.yy.h: scanner.lex
	$(LEX) --header-file=lexer.yy.h -o $(@:%.h=%.cpp) $<

parser.tab.hpp: parser.yacc IRTree/node.h IRTree/node.cpp
	$(YACC) $(YACCFLAGS) $< -o $(@:%.hpp=%.cpp)

parser.tab.o: parser.tab.hpp
	$(CC) $(CSTD) $(CPPFLAGS) $(@:%.o=%.cpp) -c -o $@

ir_tree: FORCE
	ln -sf $(shell pwd)/*.mk IRTree/
	$(MAKE) -C IRTree all

visitors: FORCE
	ln -sf $(shell pwd)/*.mk Visitors/
	$(MAKE) -C Visitors all

clean: clean_base_target
	@-rm -rfv $(EXECUTABLE_BINARY)
	-$(MAKE) -C Visitors clean
	-$(MAKE) -C IRTree clean

include clean_base_target.mk

test_run: FORCE
	mkdir -p visitor_result
	./main
	for f in $(TEX_FILES); do \
		pdflatex -output-directory visitor_result $$f; \
	done
	-rm visitor_result/*.log visitor_result/*.aux
	@echo "DONE"

docs: FORCE
	-mkdir -p docs
	doxygen Doxyfile
			
FORCE: ;


include auto_dependency_generation_target.mk