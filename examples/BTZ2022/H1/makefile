#MY_PATH=../orbiter
MY_PATH=~/DEV.22/orbiter
#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/


#############################################################################################################
# End of configuration part
#############################################################################################################


H1_LABEL = "H1"
N1_LABEL = "N1"

H1_GENS=-PGGL 4 4 \
	-subgroup_by_generators "2Cx2_0" 4 2 \
	"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0, \
	1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0"
N1_GENS=-PGGL 4 4 \
	-subgroup_by_generators "normalizer_2Cx2_0" "768" 8 \
	"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1, \
	1,0,0,0,0,1,0,0,0,0,1,0,2,0,0,1,0, \
	1,0,0,0,0,1,0,0,1,0,1,0,3,1,0,1,1, \
	1,0,0,0,0,1,0,0,3,0,1,0,2,3,0,1,1, \
	1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0, \
	1,0,0,0,2,1,1,0,1,0,1,0,1,1,2,1,1, \
	1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0, \
	1,0,0,0,3,1,0,0,1,0,1,0,0,1,3,1,1"

GROUP_and_NORMALIZER = \
	-H $(H1_LABEL) $(H1_GENS) -end \
	-N $(N1_LABEL) $(N1_GENS) -end

SPREAD_TABLES=2 "0,1,2" "../SPREAD_TABLES_4/"

#######################################
# PG(3,4)
#######################################


clean:
	- rm H_*
	- rm N_*
	- rm Packings*


spread_tables:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P 2 "0,1,2" "SPREAD_TABLES_4/"
	mv "SPREAD_TABLES_4/" ".."


# 5096448 spreads
# 1020 self dual spreads
# User time: 56:38 on Mac



H_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-with PW -do \
		-packing_with_symmetry_assumption_activity \
			-report \
		-end \


#5:15




#i : type[i] : number of orbits
#0 : 1 : 192
#1 : 4 : 38912



#Type 0 has 192 spreads:\\
#Type 1 has 155648 spreads:\\
#Written file Packings_q4.tex of size 2343999
#User time: 4:26
#User time: 5:36


#######################################
# 1 fixpoint
#######################################

H_f1:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
				1 -W \
				-problem_label N1_fixp_cliques \
			-end \
		-with PWF -do \
			-packing_fixed_points_activity \
				-report \
			-end \
		-print_symbols
	pdflatex Packings_was_fixp_q4.tex
	open Packings_was_fixp_q4.pdf

# spread orbits:
#i : type[i] : number of orbits
#0 : 1 : 192
#1 : 2 : 7776
#2 : 4 : 1270176


#reduced spread orbits:
#i : type[i] : number of orbits
#0 : 1 : 192
#1 : 4 : 38912


#The 5 representatives at level 1 are:
#0 / 5 : ( 0 )_{1 * 4 * 8 * 1 * 1 * 1 = 32} in action PGGL_4_4

#1 / 5 : ( 2 )_{1 * 4 * 4 * 1 * 1 * 1 = 16} in action PGGL_4_4

#2 / 5 : ( 4 )_{1 * 2 * 8 * 1 * 1 * 1 = 16} in action PGGL_4_4

#3 / 5 : ( 10 )_{1 * 2 * 8 * 1 * 1 * 2 = 32} in action PGGL_4_4

#4 / 5 : ( 96 )_{1 * 4 * 4 * 1 * 1 * 1 = 16} in action PGGL_4_4

#N_2Cx2_fixp_cliques.csv
#Row,C0
#0,0
#1,2
#2,4
#3,10
#4,96
#END


#H1_line_orbits_orbits_report.tex
#H_spread_orbits_orbit_types_report.tex
#H1_spread_types_reduced_orbit_types_report.tex
#H1_fixp_graph.bin
#N1_fixp_cliques_lvl_1
#N1_fixp_cliques.csv
#GF_4.cpp
#Packings_was_fixp_q4.tex
#Packings_was_fixp_q4.aux
#Packings_was_fixp_q4.pdf
#Packings_was_fixp_q4.log
#log_H_fixpoint


#User time: 1:43




H_f1_long_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			1 -W \
				-problem_label N1_fixp_cliques \
			-end \
		-define L -packing_long_orbits PWF \
			-orbit_length 4 -clique_size 5 \
			-list_of_cases_from_file \
				N1_fixp_cliques.csv \
			-create_graphs \
		-end \
		-print_symbols


#User time: 7:35


H_long_orbits_G0_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load_from_file H1_fpc0_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques  -target_size 5 -end

H_long_orbits_G1_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load_from_file H1_fpc1_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques  -target_size 5 -end

H_long_orbits_G2_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load_from_file H1_fpc2_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques  -target_size 5 -end

H_long_orbits_G3_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load_from_file H1_fpc3_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques  -target_size 5 -end

H_long_orbits_G4_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load_from_file H1_fpc4_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity -find_cliques  -target_size 5 -end


# no solutions at all

H_long_orbits_read_solutions:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			1 -W \
				-problem_label N1_fixp_cliques \
			-end \
		-define L -packing_long_orbits PWF \
			-orbit_length 4 -clique_size 5 \
			-list_of_cases_from_file \
				N1_fixp_cliques.csv \
			-read_solutions \
		-end \
		-print_symbols

H_iso:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-draw_options -end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with P -do \
		-projective_space_activity \
		-canonical_form_PG \
		-input -file_of_packings_through_spread_table \
			N1_fixp_cliques_packings.csv \
			SPREAD_TABLES_4/spread_16_spreads.csv -end \
		-classification_prefix packings_PG_3_5_H1 \
		-report \
		-end
	pdflatex packings_PG_3_5_H1_classification.tex
	open packings_PG_3_5_H1_classification.pdf


#######################################
# 5 fixpoints
#######################################

H_f5:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			5 -W \
				-problem_label $(N1_LABEL)_fixp_cliques \
			-end \
		-with PWF -do \
			-packing_fixed_points_activity \
				-report \
			-end \
		-print_symbols
	pdflatex Packings_was_fixp_q4.tex
	open Packings_was_fixp_q4.pdf


# 8 orbits on cliques of size 5


# 8 solutions in N1_fixp_cliques_lvl_5


H_f5_long_orbits:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			5 -W \
				-problem_label $(N1_LABEL)_fixp_cliques \
			-end \
		-define L -packing_long_orbits PWF \
			-orbit_length 4 -clique_size 4 \
			-list_of_cases_from_file \
				$(N1_LABEL)_fixp_cliques.csv \
			-create_graphs \
		-end \
		-print_symbols

#User time: 2:30
#User time: 2:44


#-rw-r--r--  1 betten  staff    4079349 Oct 15 10:00 H1_fpc0_lo.graph
#-rw-r--r--  1 betten  staff    4083390 Oct 15 10:00 H1_fpc1_lo.graph
#-rw-r--r--  1 betten  staff    4259034 Oct 15 10:00 H1_fpc2_lo.graph
#-rw-r--r--  1 betten  staff    3778104 Oct 15 10:00 H1_fpc3_lo.graph
#-rw-r--r--  1 betten  staff    3871986 Oct 15 10:00 H1_fpc4_lo.graph
#-rw-r--r--  1 betten  staff    3778104 Oct 15 10:00 H1_fpc5_lo.graph
#-rw-r--r--  1 betten  staff    3778104 Oct 15 10:00 H1_fpc6_lo.graph
#-rw-r--r--  1 betten  staff     152984 Oct 15 10:01 log_H_f5_long_orbits
#-rw-r--r--  1 betten  staff         64 Oct 15 10:01 N1_fixp_cliques_count.csv
#-rw-r--r--  1 betten  staff         82 Oct 15 10:01 H1_packings.csv
#-rw-r--r--  1 betten  staff    4083390 Oct 15 10:01 H1_fpc7_lo.graph
#-rw-r--r--@ 1 betten  staff      17293 Oct 15 10:13 makefile




H_f5_long_orbits_G0_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc0_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end

#User time: 0.62 of a second, dt=62 tps = 100-output_file 
#nb_sol = 608

H_f5_long_orbits_G1_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc1_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 448

H_f5_long_orbits_G2_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc2_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 1072


H_f5_long_orbits_G3_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc3_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 2816

H_f5_long_orbits_G4_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc4_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 1136

H_f5_long_orbits_G5_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc5_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 2816

H_f5_long_orbits_G6_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc6_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques  -target_size 4 \
		-end


#nb_sol = 2816

H_f5_long_orbits_G7_clique:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define G -graph -load H1_fpc7_lo.graph -end \
		-with G -do \
		-graph_theoretic_activity \
		-find_cliques -target_size 4 \
		-end

#nb_sol = 448

H_f5_long_orbits_read_solutions:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			5 -W \
				-problem_label N1_fixp_cliques \
			-end \
		-define L -packing_long_orbits PWF \
			-orbit_length 4 -clique_size 4 \
			-list_of_cases_from_file \
				N1_fixp_cliques.csv \
			-read_solutions \
		-end \
		-print_symbols

#User time: 2:5
#User time: 2:18



H_f5_iso:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-define C -combinatorial_objects \
			-file_of_packings_through_spread_table \
				H1_packings.csv \
			../SPREAD_TABLES_4/spread_16_spreads.csv 4 \
		-end \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-with C -do \
		-combinatorial_object_activity \
			-canonical_form_PG P \
				-classification_prefix \
					packings_H1_f5 \
				-label H1_f5 \
				-save_ago \
				-max_TDO_depth 4 \
			-end \
			-report \
				-prefix H1_f5 \
				-dont_show_incidence_matrices \
			-end \
		-end
	pdflatex H1_f5_classification.tex
	open H1_f5_classification.pdf


# takes long time!

#######################################
# 9 fixpoints
#######################################

H_f9:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-define F -finite_field -q 4 -end \
		-define P -projective_space 3 F -end \
		-define T -spread_table P $(SPREAD_TABLES) \
		-define PW -packing_with_symmetry_assumption T \
				$(GROUP_and_NORMALIZER) \
			-end \
		-define PWF -packing_choose_fixed_points PW \
			9 -W \
				-problem_label N1_fixp_cliques \
			-end \
		-with PWF -do \
			-packing_fixed_points_activity \
				-report \
			-end \
		-print_symbols
	pdflatex Packings_was_fixp_q4.tex
	open Packings_was_fixp_q4.pdf

#depth : number of orbits
#0 : 1
#1 : 5
#2 : 46
#3 : 207
#4 : 123
#5 : 8
#6 : 0
#7 : 0
#8 : 0
#9 : 0
#poset_classification::compute_orbits_on_subsets N=390

