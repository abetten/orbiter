SRC=../../../src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm
# specific for tx1:
#LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64







TARGETS=\
	analyze_projective_code.out  \
	andre.out  \
	classify_cubic_curves.out \
	desarguesian_spread.out \
	determine_conic.out \
	determine_cubic.out \
	determine_quadric.out \
	hermitian_points.out \
	hermitian_spreads_main.out \
	orthogonal_points.out \
	ovoid_classify_main.out \
	ovoid_make_design.out \
	points.out \
	polar.out 

all: $(TARGETS)


install:
	cp $(TARGETS) ../../../bin




analyze_projective_code.out:  analyze_projective_code.o;
	$(DISCRETA_CPP)  analyze_projective_code.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

andre.out:  andre.o;
	$(DISCRETA_CPP)  andre.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

classify_cubic_curves.out:  classify_cubic_curves.o;
	$(DISCRETA_CPP)  classify_cubic_curves.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

desarguesian_spread.out:  desarguesian_spread.o ;
	$(DISCRETA_CPP)  desarguesian_spread.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

determine_conic.out:  determine_conic.o;
	$(DISCRETA_CPP)  determine_conic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

determine_cubic.out:  determine_cubic.o;
	$(DISCRETA_CPP)  determine_cubic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

determine_quadric.out:  determine_quadric.o;
	$(DISCRETA_CPP)  determine_quadric.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hermitian_points.out:  hermitian_points.o ;
	$(DISCRETA_CPP)  hermitian_points.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hermitian_spreads_main.out:  hermitian_spreads_main.o ;
	$(DISCRETA_CPP)  hermitian_spreads_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

orthogonal_points.out:  orthogonal_points.o ;
	$(DISCRETA_CPP)  orthogonal_points.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ovoid_classify_main.out:  ovoid_classify_main.o ;
	$(DISCRETA_CPP)  ovoid_classify_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ovoid_make_design.out:  ovoid_make_design.o ;
	$(DISCRETA_CPP)  ovoid_make_design.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

points.out:  points.o;
	$(DISCRETA_CPP)  points.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

polar.out:  polar.o;
	$(DISCRETA_CPP)  polar.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@






clean:
	- rm *.o *.a *.out *~

