SRC=../../../../ORBITER/SRC
DISCRETA_CPP = /usr/bin/g++
CPPFLAGS = -Wall -I$(SRC)/LIB -I$(SRC)/LIB/DISCRETA
LIB = \
	$(SRC)/LIB/TOP_LEVEL/libtoplevel.a \
	$(SRC)/LIB/DISCRETA/libdiscreta.a \
	$(SRC)/LIB/SNAKES_AND_LADDERS/libsnakesandladders.a  \
	$(SRC)/LIB/ACTION/libaction.a \
	$(SRC)/LIB/GALOIS/libgalois.a 
LFLAGS = -lm



all: test_PG.out \
	polar.out \
	points.out \
	analyze_projective_code.out  \
	analyze_q_designs.out  \
	hill.out \
	determine_conic.out \
	determine_quadric.out \
	ovoid.out \
	cheat_sheet_PG.out \
	make_BLT_set.out \
	make_group.out \
	make_grassmannian.out \
	baer_subplane.out \
	orthogonal_points.out \
	make_something.out \
	process.out\
	process_PG.out\
	projective_space_main.out\
	plane_search.out \
	example_fano_plane.out \
	simeon.out \
	export_data.out




#	factor_space.out \
#	analyze.out  \



test_PG.out:  test_PG.o;
	$(DISCRETA_CPP)  test_PG.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

factor_space.out:  factor_space.o;
	$(DISCRETA_CPP)  factor_space.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

polar.out:  polar.o;
	$(DISCRETA_CPP)  polar.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

points.out:  points.o;
	$(DISCRETA_CPP)  points.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

analyze.out:  analyze.o;
	$(DISCRETA_CPP)  analyze.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

analyze_projective_code.out:  analyze_projective_code.o;
	$(DISCRETA_CPP)  analyze_projective_code.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

analyze_q_designs.out:  analyze_q_designs.o;
	$(DISCRETA_CPP)  analyze_q_designs.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

hill.out:  hill.o;
	$(DISCRETA_CPP)  hill.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

determine_conic.out:  determine_conic.o;
	$(DISCRETA_CPP)  determine_conic.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

determine_quadric.out:  determine_quadric.o;
	$(DISCRETA_CPP)  determine_quadric.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

ovoid.out:  ovoid.o;
	$(DISCRETA_CPP)  ovoid.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

cheat_sheet_PG.out:  cheat_sheet_PG.o;
	$(DISCRETA_CPP)  cheat_sheet_PG.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_BLT_set.out:  make_BLT_set.o;
	$(DISCRETA_CPP)  make_BLT_set.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_grassmannian.out:  make_grassmannian.o;
	$(DISCRETA_CPP)  make_grassmannian.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_group.out:  make_group.o;
	$(DISCRETA_CPP)  make_group.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

baer_subplane.out:   baer_subplane.o  ;
	$(DISCRETA_CPP)  baer_subplane.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

orthogonal_points.out:  orthogonal_points.o ;
	$(DISCRETA_CPP)  orthogonal_points.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_something.out:  make_something.o ;
	$(DISCRETA_CPP)  make_something.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

process.out:  process.o ;
	$(DISCRETA_CPP)  process.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

process_PG.out:  process_PG.o ;
	$(DISCRETA_CPP)  process_PG.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

projective_space_main.out:  projective_space_main.o ;
	$(DISCRETA_CPP)  projective_space_main.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plane_search.out:  plane_search.o ;
	$(DISCRETA_CPP)  plane_search.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

example_fano_plane.out:  example_fano_plane.o ;
	$(DISCRETA_CPP)  example_fano_plane.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

simeon.out:  simeon.o ;
	$(DISCRETA_CPP)  simeon.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

export_data.out:  export_data.o ;
	$(DISCRETA_CPP)  export_data.o  $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@




%.o: %.C %.c; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<


clean:
	- rm *.o *.a *.out *~

