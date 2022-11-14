include compilers.mk
include compiler-flags.mk
include os-detect.mk
include auto_dependency_generation_variables.mk

TARGET_OBJECT_FILES = main.o \
						lexer.yy.o \
						parser.tab.o \
						IRtree/node.o \
						./Visitors/*.o \
						./Visitors/LatexVisitors/*.o \
						./Visitors/ToStringVisitors/*.o \
						./Visitors/ExpansionVisitors/*.o \
						./Visitors/CopyVisitors/*.o \
						./Visitors/PrintVisitors/*.o \
						./Visitors/ReductionVisitors/*.o

TEX_FILES = $(wildcard visitor_result/*.tex)
EXECUTABLE_BINARY = main


main: $(SRCS:.cpp=.o) lexer.yy.o parser.tab.o visitors ir_tree
	$(CC) $(CSTD) $(CPPFLAGS) $(TARGET_OBJECT_FILES) -o $@ $(LDFLAGS) -L ../../ -lorbiter $(LEXLIB)

all: $(SRCS:.cpp=.o) lexer.yy.o parser.tab.o visitors ir_tree;

lexer.yy.o: lexer.yy.h 
	$(CC) $(CSTD) $(CPPFLAGS) $(@:%.o=%.cpp) -c -o $@

lexer.yy.h: scanner.lex
	$(LEX) --header-file=lexer.yy.h -o $(@:%.h=%.cpp) $<

parser.tab.hpp: parser.yacc IRtree/node.h IRtree/node.cpp
	$(YACC) $(YACCFLAGS) $< -o $(@:%.hpp=%.cpp)

parser.tab.o: parser.tab.hpp
	$(CC) $(CSTD) $(CPPFLAGS) $(@:%.o=%.cpp) -c -o $@

ir_tree: FORCE
	ln -sf $(shell pwd)/*.mk IRtree/
	$(MAKE) -C IRtree all

visitors: FORCE
	ln -sf $(shell pwd)/*.mk Visitors/
	$(MAKE) -C Visitors all

clean: clean_base_target
	@-rm -rfv $(EXECUTABLE_BINARY)
	-$(MAKE) -C Visitors clean
	-$(MAKE) -C IRtree clean

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