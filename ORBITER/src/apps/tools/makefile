SRC=../../../../ORBITER/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib 
LIB = $(SRC)/lib/liborbiter.a
LFLAGS = -lm




TARGETS=\
	collect.out \
	concatenate_files.out \
	create_file.out \
	distribution.out \
	join_sets.out \
	latex_table.out \
	loop.out \
	make_poster.out \
	memory_usage.out \
	missing_files.out \
	plot_decomposition_matrix.out  \
	plot_xy.out \
	prepare_frames.out \
	read_orbiter_file.out \
	read_solutions.out \
	read_vector_and_extract_set.out \
	run_blt.out  \
	run_lifting.out  \
	scheduler.out \
	split_spreadsheet.out \
	split.out 


all: $(TARGETS)

install:
	cp $(TARGETS) ../../../../bin



#%.o: %.C %.cpp; $(DISCRETA_CPP) $(CPPFLAGS) -c  -o $@ $<
#
#%.out: %.o; 
#	@echo "------------------------------------------------------------------"
#	@echo -n "$<"
#	@echo -n ": "
#	$(DISCRETA_CPP) $(CPPFLAGS)  $(LIB) $(LFLAGS)  $<
#	@echo "------------------------------------------------------------------"


collect.out:  collect.o 
	$(DISCRETA_CPP)  collect.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

concatenate_files.out:  concatenate_files.o 
	$(DISCRETA_CPP)  concatenate_files.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

create_file.out:  create_file.o 
	$(DISCRETA_CPP)  create_file.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

distribution.out:  distribution.o
	$(DISCRETA_CPP)  distribution.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

join_sets.out:  join_sets.o 
	$(DISCRETA_CPP)  join_sets.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

latex_table.out:  latex_table.o 
	$(DISCRETA_CPP)  latex_table.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

loop.out:  loop.o 
	$(DISCRETA_CPP)  loop.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

make_poster.out:  make_poster.o 
	$(DISCRETA_CPP)  make_poster.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

memory_usage.out:  memory_usage.o 
	$(DISCRETA_CPP)  memory_usage.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

missing_files.out:  missing_files.o 
	$(DISCRETA_CPP)  missing_files.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plot_decomposition_matrix.out:  plot_decomposition_matrix.o
	$(DISCRETA_CPP)  plot_decomposition_matrix.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

plot_xy.out:  plot_xy.o 
	$(DISCRETA_CPP)  plot_xy.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

prepare_frames.out:  prepare_frames.o 
	$(DISCRETA_CPP)  prepare_frames.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

read_orbiter_file.out:  read_orbiter_file.o 
	$(DISCRETA_CPP)  read_orbiter_file.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

read_solutions.out:  read_solutions.o 
	$(DISCRETA_CPP)  read_solutions.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

read_vector_and_extract_set.out:  read_vector_and_extract_set.o 
	$(DISCRETA_CPP)  read_vector_and_extract_set.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

run_blt.out:  run_blt.o
	$(DISCRETA_CPP)  run_blt.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

run_lifting.out:  run_lifting.o
	$(DISCRETA_CPP)  run_lifting.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

scheduler.out:  scheduler.o
	$(DISCRETA_CPP)  scheduler.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

split_spreadsheet.out:  split_spreadsheet.o 
	$(DISCRETA_CPP)  split_spreadsheet.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@

split.out:  split.o
	$(DISCRETA_CPP)  split.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



clean:
	@echo "------------------------------------------------------------------"
	@echo -n "cleaning"
	@echo -n ": "
	- rm *.o *.a *.out *~

