MY_PATH=~/DEV.20/GITHUB/orbiter
#MY_PATH=../../..
#MY_PATH=/scratch/betten/COMPILE/orbiter

GINAC_PATH=$(MY_PATH)/src/apps/ginac
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter

update:
	cd $(ORBITER_PATH); make clean;
	cd $(MY_PATH); make cleana; git pull; make 


update_all:
	cd $(MY_PATH); make clean; git pull; make 


test:
	$(ORBITER_PATH)/orbiter.out
	
		
#############################################################################################################
# Section 3: Finite Fields


F_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 2
	pdflatex GF_2.tex
	open GF_2.pdf


F_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 4 
	pdflatex GF_4.tex
	open GF_4.pdf



F_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 8  
	pdflatex GF_8.tex
	open GF_8.pdf

F_16:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 16  
	pdflatex GF_16.tex
	open GF_16.pdf


#############################################################################################################
# Section 4: Finite Projective Spaces



PG_3_2:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 3 2
	pdflatex PG_3_2.tex 
	open PG_3_2.pdf


#############################################################################################################
# Section 5: Algebraic Sets


EC1:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -q 11 -n 2 \
		-projective_variety "EC" 3 "1,0,3,2,10,6,1,7"	-monomial_type_PART -end



Hirschfeld4:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3 -q 4 \
		-projective_variety "Hirschfeld_surface_q4" 3 "1,6,1,8,1,11,1,13" \
		-monomial_type_PART -end \
		-save ""


Endrass_7:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3 -q 7 \
		-projective_variety "Endrass_F7" 8 "6,0,4,4,2,7,5,9,6,20,6,23,1,25,3,30,1,32,3,34,4,56,6,59,1,61,6,66,2,68,6,70,3,77,2,79,6,83,6,120,2,123,5,125,3,130,1,132,3,134,3,141,2,143,6,147,3,156" \
		-monomial_type_LEX -end \
		-save ""

# we created a set of 33 points, called Endrass_F7.txt



#############################################################################################################
# Section 6: Linear Groups


PGL_4_2:
	$(ORBITER_PATH)/orbiter.out -linear_group -PGL 4 2 -end

PGGL_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 3 4 -end \
		-group_theoretic_activities -report -sylow -classes -end


#-sylow  
#		-export_schreier_trees \
#		-tools_path $(GRAPH_THEORY_PATH) \
#		-report \
#		-report_schreier_trees

J1:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 7 11 -Janko1 -end \
		-group_theoretic_activities -report -end

PGL_3_11_singer:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 3 11 -singer_and_frobenius 19 -end \
		-group_theoretic_activities -report -end


PG_2_4_order_21:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 3 4  -end \
		-group_theoretic_activities -search_element_of_order 21

PG_2_4_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 3 4  -end \
		-group_theoretic_activities -find_singer_cycle




PSL_3_2_classes:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PSL 3 2 -end \
		-group_theoretic_activities -classes


quaternion:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -SL 2 3 \
		-subgroup_by_generators "quaternion" "8" 3 \
		"1,1,1,2" \
		"2,1,1,1" \
		"0,2,1,0" \
		-end \
		-group_theoretic_activities \
		-print_elements_tex \
		-group_table \
		-report \
		-end


U_3_2:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -override_polynomial "17" -PGL 3 9 \
		-subgroup_by_generators "U_3_3" "6048" 2 \
		"1,6,4, 5,0,6, 8,5,1" \
		"6,2,1, 7,8,4, 0,6,6" \
		-end \
		-group_theoretic_activities \
		-report -end



PGL_2_3:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 2 3 -end  \
		-group_theoretic_activities -report -group_table -end
	pdflatex PGL_2_3_group_table_order_24.tex
	pdflatex PGL_2_3_report.tex
	open PGL_2_3_group_table_order_24.pdf
	open PGL_2_3_report.pdf

	
			
cube_group:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -GL 3 3 \
		-subgroup_by_generators "cube" "48" 3 \
		"0,1,0,2,0,0,0,0,1" \
		"0,0,1,0,1,0,2,0,0" \
		"2,0,0,0,1,0,0,0,1" \
		-end \
		-group_theoretic_activities \
		-print_elements_tex \
		-report
	pdflatex GL_3_3_Subgroup_cube_48_report.tex
	open GL_3_3_Subgroup_cube_48_report.pdf
	pdflatex GL_3_3_Subgroup_cube_48_elements.tex
	open GL_3_3_Subgroup_cube_48_elements.pdf


PG_3_2_again:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 2 3
	pdflatex PG_2_3.tex 
	open PG_2_3.pdf


tetra_group:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -GL 3 3 \
		-subgroup_by_generators "tetra" "12" 2 \
		"0,1,0,0,0,1,1,0,0" \
		"0,0,1,2,0,0,0,2,0" \
		-end \
		-group_theoretic_activities \
		-print_elements_tex \
		-report
	pdflatex GL_3_3_Subgroup_tetra_12_report.tex
	open GL_3_3_Subgroup_tetra_12_report.pdf
	pdflatex GL_3_3_Subgroup_tetra_12_elements.tex
	open GL_3_3_Subgroup_tetra_12_elements.pdf



T4:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_tensors -end \
		-group_theoretic_activities \
		-orbits_on_points
	pdflatex GL_2_2_wreath_Sym4_res65535_orbits.tex
	open GL_2_2_wreath_Sym4_res65535_orbits.pdf

T4r1:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_rank_one_tensors -end \
		-group_theoretic_activities \
		-orbits_on_points
	pdflatex GL_2_2_wreath_Sym4_res81_orbits.tex
	open GL_2_2_wreath_Sym4_res81_orbits.pdf

T4r1_orbits:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_rank_one_tensors -end \
		-group_theoretic_activities \
		-orbits_on_subsets 4 \
		-report




PGL_3_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 3 4 -end







classes_PGGL_4_4:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-classes
	pdflatex PGGL_4_4_classes_out.tex
	open PGGL_4_4_classes_out.pdf


# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2 48960 40320 Baer involution

#2 5355 368640  one block of 10,11

#2 64260 30720 two blocks of 10,11 (problem group)


CLASS_2A=-centralizer_of_element "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" -label "2A" 
# Baer involution

CLASS_2B=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0" -label "2B"

CLASS_2C=-centralizer_of_element "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" -label "2C"  
# problem group


# 3 classes of elements of order 3
# 4 classes of elements of order 4

PGGL_4_4_cent_2A:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-centralizer_of_element "2A" "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1"

#Strong generators for a group of order 40320:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,\\
#1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,\\
#1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0,\\
#0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1,\\


PGGL_4_4_cent_2B:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-centralizer_of_element "2B" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 0"

#Strong generators for a group of order 368640:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,3,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,1,3,1,\\
#1,0,0,0,0,1,0,0,1,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,0,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,0,0,0,1,0,\\
#1,0,0,0,0,1,0,0,3,0,1,3,2,0,0,3,1,\\
#1,0,0,0,0,1,0,0,1,0,0,3,3,0,2,3,0,\\
#1,0,0,0,1,1,0,0,2,0,1,0,2,0,0,1,0,\\
#1,0,0,0,1,1,0,2,2,0,1,3,2,0,0,2,0,\\
#1,0,0,0,2,1,1,2,1,0,2,1,3,0,2,2,0,\\
#1,0,0,0,2,1,0,2,0,0,1,0,2,0,0,3,1,\\

PGGL_4_4_cent_2C:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-centralizer_of_element "2C" "1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0"


#Strong generators for a group of order 30720:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0,\\
#1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1,\\



PGGL_4_4_norm_4A:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4A" "1,0,0,0,1,1,0,0, 0,1,1,0, 0,0,1,1, 0"


# normalizer has order 256:
#Strong generators for a group of order 256:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,3,0,0,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,\\
#1,0,0,0,0,1,0,0,3,1,1,0,1,2,0,1,0,\\
#1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,\\
#1,0,0,0,2,1,0,0,1,3,1,0,2,2,2,1,0,\\



PGGL_4_4_norm_4B:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4B" "1,0,0,0,0,1,0,0, 0,1,1,0, 0,0,1,1, 0"

#Strong generators for a group of order 3072:\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,2,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,2,1,1,0,\\
#1,0,0,0,0,2,0,0,0,2,2,0,0,3,0,2,0,\\
#1,0,0,0,0,3,0,0,0,1,3,0,0,2,1,3,1,\\
#1,2,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,\\




PGGL_4_4_norm_4C:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4C" "3,0,0,0, 1,3,0,0, 0,0,3,0, 0,0,1,3, 1"



#Strong generators for a group of order 384:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,\\
#1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1,\\
#1,0,0,0,2,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,3,1,1,0,0,0,1,0,1,0,2,1,0,\\
#0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0,1,\\



PGGL_4_4_norm_4D:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-normalizer_of_cyclic_subgroup "4D" "3,0,0,0, 0,3,0,0, 0,0,3,0, 0,0,1,3, 1"


#Strong generators for a group of order 768:
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,0,2,1,1,\\
#1,0,0,0,0,1,0,0,0,0,1,0,1,0,2,1,0,\\
#1,0,0,0,0,1,0,0,0,0,1,0,0,1,2,1,0,\\
#1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,\\
#1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,\\
#0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,\\




no:
	pdflatex PGGL_4_4_classes_out.tex
	open PGGL_4_4_classes_out.pdf



cent_2A:
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 \
		-subgroup_by_generators "centralizer_2A" "40320" 10 \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
			"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
			"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
			"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
			"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
			"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
			"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1," \
		-end




#############################################################################################################
# Section 7: Orbits on Subsets




orb:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -GL 2 2 -singer 1 -end  \
			-group_theoretic_activities \
			-orbits_on_subsets 4 \
			-draw_poset \
			-report
	pdflatex unnamed_poset_detailed_lvl_4.tex
	open unnamed_poset_detailed_lvl_4.pdf


orb16:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 2 16 -end  \
			-group_theoretic_activities \
			-orbits_on_subsets 10 \


#############################################################################################################
# Section 8: Orbits on Subspaces


subspaces_orthogonal:
		$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 4 2 -orthogonal 1 -end \
			-group_theoretic_activities -orbits_on_subspaces 4 \
			-draw_poset


sub:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 4 2 -end -group_theoretic_activities -orbits_on_subspaces 4 -draw_poset

sub_4_2_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 4 2 -singer 1 -end -group_theoretic_activities -orbits_on_subspaces 4 -draw_poset

sub_4_2_orthogonal:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 4 2 -orthogonal 1 -end -group_theoretic_activities -orbits_on_subspaces 4 -draw_poset


sub_8_2_singer:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 8 2 -singer 1 -end -group_theoretic_activities -orbits_on_subspaces 8 

# May 7, 2020: 16 sec on Mac


O_6_3:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 6 3 -orthogonal 1 -end \
		-group_theoretic_activities -orbits_on_subspaces 3

# June 3, 2020 on Mac: 0 sec



O_6_11:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGL 6 11 -orthogonal 1 -end \
		-group_theoretic_activities -orbits_on_subspaces 3

# June 3, 2020 on Mac: 12 sec



#############################################################################################################
# Section 9: Graph Theory




petersen:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Johnson 5 2 0 -end -save graph_J520.bin

petersen_csv:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Johnson 5 2 0 -end \
		-graph_theoretic_activity -export_csv graph_J520.bin -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Johnson_5_2_0.csv 20



Paley13:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -Paley 13 -end -save graph_P13.bin


graph_classify_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -graph_classify -n 4

tournament_classify_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -graph_classify -n 4 -v 2 -tournament -draw_graphs_at_level 6
	

small_graph:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -edges_as_pairs 5 "0,1,0,2,0,3,0,4,1,3,1,4,2,4" -end

# creates graph_v5_e7.colored_graph

small_graph_cliques:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file graph_v5_e7.colored_graph -end \
		-graph_theoretic_activity -find_cliques  -target_size 3 -end -end

small_graph_cliques_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file graph_v5_e7.colored_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 3 -end -end


#############################################################################################################
# Section 10: Projective Geometry




EC:
	$(ORBITER_PATH)/orbiter.out -v 5 -create_combinatorial_object -q 11 \
		-elliptic_curve 1 3 -end \
		-save "./"
	$(ORBITER_PATH)/orbiter.out -v 2 -process_combinatorial_objects -q 11 -n 2  \
		-draw_points_in_plane EC_11_1_3 \
		-fname_base_out EC_11_1_3 -embedded \
		-input -file_of_points elliptic_curve_b1_c3_q11.txt -end \
		-end
	$(ORBITER_PATH)/orbiter.out -v 2 -canonical_form_PG  2 11 \
		-input -file_of_points elliptic_curve_b1_c3_q11.txt -end \
		-classification_prefix elliptic_curve_b1_c3_q11 \
		-report \
		-end 
	pdflatex elliptic_curve_b1_c3_q11_classification.tex
	open elliptic_curve_b1_c3_q11_classification.pdf


no:
	pdflatex EC_11_1_3_draw.tex
	open EC_11_1_3_draw.pdf
	pdflatex PG_2_11_EC_classification.tex
	open PG_2_11_EC_classification.pdf
	



Hirschfeld4c:
	$(ORBITER_PATH)/orbiter.out -v 2 -canonical_form_PG  3 4 \
		-input -file_of_points Hirschfeld_surface_q4.txt -end \
		-classification_prefix Hirschfeld_surface_q4 \
		-report \
		-end 
	pdflatex Hirschfeld_surface_q4_classification.tex
	open Hirschfeld_surface_q4_classification.pdf


Endrass_7c:
	$(ORBITER_PATH)/orbiter.out -v 2 -canonical_form_PG  3 7 \
		-input -file_of_points Endrass_F7.txt -end \
		-classification_prefix Endrass_F7 \
		-report \
		-end 
	pdflatex Endrass_F7_classification.tex
	open Endrass_F7_classification.pdf



Hirschfeld16:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_combinatorial_object -n 3  -q 16 \
		-projective_variety "Hirschfeld_surface_q16" 3 "1,6,1,8,1,11,1,13" \
		-end \
		-save ""

Hirschfeld16c:
	$(ORBITER_PATH)/orbiter.out -v 2 -canonical_form_PG  3 16 \
		-input -file_of_points \
			Hirschfeld_surface_q16.txt -end \
		-prefix Hirschfeld_surface_q16 \
		-report
	pdflatex Hirschfeld_surface_q16_classification.tex
	open Hirschfeld_surface_q16_classification.pdf

# nauty_interface_matrix_int n >= MAXN
# nauty_interface_matrix_int n = 74532
# nauty_interface_matrix_int MAXN = 22000






#############################################################################################################
# Section 11: Arcs

hyperoval_4:
	$(ORBITER_PATH)/orbiter.out -v 8 \
		-linear_group -PGGL 3 4 -end \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q4_6_2 -W -end \
			-classify_arcs 6 2 \
		-end

no:
			-exact_cover \
				-input_prefix ./ARCS/ \
				-output_prefix ./SYSTEMS/ \
				-solution_prefix ./SOLUTIONS/ \
				-starter_size 6 \
				-base_fname arcs_q4_d2 \
				-lex \
			-end \

hyperoval_8:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGGL 3 8 -end \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q8_2 -W -end \
			-classify_arcs 10 2 \
		-end




hyperoval_16:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGGL 3 16 -end \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q16_2 -W -end \
			-classify_arcs 18 2 \
		-end




# June 3 2020: 1 sec on Mac
 

hyperoval_32:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGGL 3 32 -end \
		-group_theoretic_activities \
			-poset_classification_control -problem_label arcs_q32_2 -W -end \
			-classify_arcs 34 2 \
		-end






#############################################################################################################
# Section 12: Cubic Surfaces


surface_7:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 4 7 -wedge -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
			-end \
		-surface_classify -end
	

surface_4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 4 -wedge -end \
		-group_theoretic_activities \
		-surface_classify -end


surface_4_report:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 4 -wedge -end \
		-group_theoretic_activities -surface_classify -report


S13:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 4 13 -wedge -end \
		-group_theoretic_activities \
		-create_surface -family_S 3 -q 13 -end


surface_4_0:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities \
		-create_surface -q 4 -catalogue 0 -end


surface_8_0:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities \
		-create_surface -q 8 -catalogue 0 -end \
		-transform_inverse "1,4,4,0,6,0,0,0,6,2,0,1,7,0,4,0,0"


surface_recognize_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities -surface_recognize -q 8 -by_coefficients "1,6,1,8,1,11,1,13,1,19" -end -end


surface_isomorph_8:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGGL 4 8 -wedge -end \
		-group_theoretic_activities -surface_isomorphism_testing \
			-q 8 -by_coefficients "5,5,5,8,5,9,5,10,5,11,5,12,4,14,4,15,1,18,1,19" -end \
			-q 8 -by_coefficients "1,6,1,8,1,11,1,13,1,19" -end 


arc_lifting_4:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q4 -end \
		-trihedra2_control -problem_label tri2_q4 -end \
		-control_six_arcs -problem_label sixarcs_q4 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs \
		-end
	pdflatex arc_lifting_q4.tex
	open arc_lifting_q4.pdf



arc_lifting_13:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 4 13 -end \
		-group_theoretic_activities \
		-trihedra1_control -problem_label tri1_q13 -end \
		-trihedra2_control -problem_label tri2_q13 -end \
		-Control_six_arcs -problem_label sixarcs_q13 -end \
		-classify_surfaces_through_arcs_and_trihedral_pairs 13
	pdflatex arc_lifting_q13.tex
	open arc_lifting_q13.pdf








test_432:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGGL 4 4 \
		-subgroup_by_generators "trihedra" "432" 8 \
		"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0" \
		"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,0,0,0,0,1,0,0,0,0,3,0,0,0,1,3,0" \
		"1,0,0,0,0,1,0,0,0,0,3,2,0,0,0,3,0" \
		"1,0,0,0,0,1,0,0,0,0,0,2,0,0,1,0,0" \
		"1,0,0,0,2,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,3,0,0,2,0,0,0,0,0,2,0,0,0,0,2,0" \
		"0,0,1,0,0,0,2,1,3,0,0,0,0,3,0,0,0" \
		-end \

test_432_t:
	$(ORBITER_PATH)/orbiter.out -v 5 -linear_group -PGGL 4 4 \
		-subgroup_by_generators "trihedra" "432" 8 \
		"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,0" \
		"1,0,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,0,0,0,0,1,0,0,0,0,3,1,0,0,0,3,0" \
		"1,0,0,0,0,1,0,0,0,0,3,0,0,0,2,3,0" \
		"1,0,0,0,0,1,0,0,0,0,0,1,0,0,2,0,0" \
		"1,2,0,0,0,3,0,0,0,0,3,0,0,0,0,2,1" \
		"1,2,0,0,3,0,0,0,0,0,2,0,0,0,0,2,0" \
		"0,0,3,0,0,0,0,3,1,2,0,0,0,1,0,0,0" \
		-end \



#############################################################################################################
# Section 13: Cryptography and Number Theory

inverse_mod:
	$(ORBITER_PATH)/orbiter.out -v 2 -inverse_mod 18059241 58014043

jacobi:
	$(ORBITER_PATH)/orbiter.out -v 5 -jacobi 2221 7817

EC_add:
	$(ORBITER_PATH)/orbiter.out -v 2 -EC_add 11 1 3 "1,4" "1,4"

EC_cyclic_subgroup:
	$(ORBITER_PATH)/orbiter.out -v 2 -EC_cyclic_subgroup 11 1 3 "1,4"

EC_points:
	$(ORBITER_PATH)/orbiter.out -v 2 -EC_points 199 5 7

EC_Koblitz_encoding:
	$(ORBITER_PATH)/orbiter.out -v 6 -seed 17 -EC_Koblitz_encoding 199 5 7 67 "147,164" "DEADBEEF"

EC_bsgs:
	$(ORBITER_PATH)/orbiter.out -v 2 -EC_bsgs 199 5 7 "147,164" 212 \
		"172,158,45,195,50,22,10,103,55,33,50,22,145,105,31,74,73,155,67,60,25,6"

EC_bsgs_decode:
	$(ORBITER_PATH)/orbiter.out -v 2 -EC_bsgs_decode 199 5 7 "129,176" 212 \
		"127,188,51,141,85,29,106,90,41,105,179,71,171,2,16,197,183,72,27,129,37,10" \
		"50,179,169,13,153,169,115,116,188,110,176"


#############################################################################################################
# Section 14: Coding Theory


BCH15:
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 5 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 7 
	$(ORBITER_PATH)/orbiter.out -BCH 15 2 9 


BCH11:
	$(ORBITER_PATH)/orbiter.out -BCH 11 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 11 2 5 


BCH13:
	$(ORBITER_PATH)/orbiter.out -BCH 13 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 13 2 5 


BCH7:
	$(ORBITER_PATH)/orbiter.out -BCH 7 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 7 2 5 


BCH21:
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 3 
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 5 
	$(ORBITER_PATH)/orbiter.out -BCH 21 2 7 
	$(ORBITER_PATH)/orbiter.out -BCH_dual 21 2 7 


Hamming_4_2:
	$(ORBITER_PATH)/orbiter.out -Hamming_graph 4 2
	$(ORBITER_PATH)/orbiter.out -draw_matrix Hamming_n4_q2.csv 20
	open Hamming_n4_q2_draw.bmp

codes_8_4_4:
	$(ORBITER_PATH)/orbiter.out -v 2 -codes_classify -n 8 -k 4 -q 2 -d 4 -w -lex \
		-draw_poset


codes_14_4_9_3:
	$(ORBITER_PATH)/orbiter.out -v 6 -codes_classify -n 14 -k 4 -q 3 -d 9 -w -lex \
		-draw_poset

codes_16_5_9_3:
	$(ORBITER_PATH)/orbiter.out -v 6 -codes_classify -n 16 -k 5 -q 3 -d 9 -w -W -lex \
		-draw_poset

# 5/31/2020: 28 min 22 sec on Mac

#0 : 1 orbits
#1 : 1 orbits
#2 : 1 orbits
#3 : 1 orbits
#4 : 1 orbits
#5 : 1 orbits
#6 : 1 orbits
#7 : 1 orbits
#8 : 1 orbits
#9 : 2 orbits
#10 : 3 orbits
#11 : 4 orbits
#12 : 5 orbits
#13 : 5 orbits
#14 : 4 orbits
#15 : 3 orbits
#16 : 1 orbits
#total: 36


codes_d4:
	$(ORBITER_PATH)/orbiter.out -v 3 -linear_group -PGL 4 2 -end \
		-group_theoretic_activities -poset_classification_control -W \
		-problem_label codes_r4_d4 -draw_poset -embedded -end -linear_codes 4 100 -end


#############################################################################################################
# Section 15: Diophantine Systems


linsp7:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch linsp7 1 6 \
		-coefficient_matrix "21,15,10,6,3,1" -RHS "21,21,1" -x_min_global 0 -x_max_global 21 
	
linsp7_solve:
	$(ORBITER_PATH)/orbiter.out -v 10 -diophant_activity -input_file linsp7.diophant -solve_mckay 

# 32 solutions in 45 backtrack steps

linsp7_again:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch linsp7 1 6 \
		-coefficient_matrix_csv linsp7_coeff_matrix.csv \
		-RHS_csv linsp7_RHS.csv \
		-x_bounds_csv linsp7_x_bounds.csv
	


linsp6:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch linsp6 1 5 \
		-coefficient_matrix "15,10,6,3,1" -RHS "15,15,1" -x_min_global 0 -x_max_global 15
	
linsp6_solve:
	$(ORBITER_PATH)/orbiter.out -v 3 -diophant_activity -input_file linsp6.diophant -solve_mckay 

# Found 15 solutions with 22 backtrack steps


part10:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch part10 1 10 \
		-coefficient_matrix "10,9,8,7,6,5,4,3,2,1" -RHS "10,10,1" -x_min_global 0 -x_max_global 10

part10_again:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch part10 1 10 \
		-coefficient_matrix_csv part10_coeff_matrix.csv \
		-RHS_csv part10_RHS.csv \
		-x_bounds_csv part10_x_bounds.csv
	

part10_solve:
	$(ORBITER_PATH)/orbiter.out -v 10 -diophant_activity -input_file part10.diophant -solve_mckay 

# Found 42 solutions with 67 backtrack steps


octic_monomials:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch octic_monomials 1 4 \
		-coefficient_matrix "1,1,1,1" -RHS "8,8,1" -x_min_global 0 -x_max_global 8


octic_monomials_solve:
	$(ORBITER_PATH)/orbiter.out -v 3 -diophant_activity -input_file octic_monomials.diophant -solve_mckay 
	sort -r octic_monomials.sol >octic_monomials_sorted.txt

#Found 165 solutions with 210 backtrack steps
# 165=binomial(11,3)



linsp30_pt_types:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch linsp30_pt_types 1 3 \
		-coefficient_matrix "6,4,3" -RHS "29,29,1" -x_bounds "0,1,0,27,0,24"
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file linsp30_pt_types.diophant -solve_mckay 

linsp30_pt_distribution:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -from_scratch linsp30_pt_distribution 6 4 \
		-coefficient_matrix "1,1,1,1,1,1,0,0,5,2,5,2,1,5,3,7,10,1,10,1,0,10,3,21" \
			-RHS "30,30,1,7,7,1,135,135,1,96,96,1,0,351,2,0,276,2" \
			-x_min_global 0 -x_max_global 30
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		linsp30_pt_distribution.diophant -solve_mckay 
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file \
		linsp30_pt_distribution.diophant -draw_as_bitmap 20 


#############################################################################################################
# Section 16: Design Theory


design_PG_2_3:
	$(ORBITER_PATH)/orbiter.out -v 8  \
		-create_design -q 3 -family PG_2_q  -end \

design_PG_2_4:
	$(ORBITER_PATH)/orbiter.out -v 8  \
		-create_design -q 4 -family PG_2_q  -end \



#############################################################################################################
# Section 17: Linear Spaces and Tactical Decompositions



max_arc_16_4_start:
	$(ORBITER_PATH)/orbiter.out -v 4 -maximal_arc_parameters 16 4


max_arc_16_4_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4.stack

max_arc_16_4_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4.tdo -dual_is_linear_space -end
	
max_arc_16_4r_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4r.tdo 

max_arc_16_4b_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4b.stack

max_arc_16_4b_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4b.tdo -dual_is_linear_space -end
	
max_arc_16_4br_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4br.tdo 


max_arc_16_4bd_convert_stack_tdo:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo max_arc_q16_r4bd.stack

max_arc_16_4bd_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bd.tdo -dual_is_linear_space -end
	
max_arc_16_4bdr_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4bdr.tdo 

max_arc_16_4bdr_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bdr.tdo -dual_is_linear_space -end
	
max_arc_16_4bdrr_print:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_print max_arc_q16_r4bdrr.tdo 

max_arc_16_4bdrr_refine:
	$(ORBITER_PATH)/orbiter.out -v 4 -tdo_refinement -input_file max_arc_q16_r4bdrr.tdo -dual_is_linear_space -end
	


max_arc_flag_stabilizer:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -PGGL 3 16 -subgroup_by_generators "flag_stab" 3686400 16 \
	"-1,0,0,0,1,0,0,0,1,0" \
	"1,0,0,0,-1,0,0,0,1,0" \
	"1,0,0,0,1,0,0,0,-1,0" \
	"1,0,0,1,1,0,0,0,1,0" \
	"1,0,0,2,1,0,0,0,1,0" \
	"1,0,0,4,1,0,0,0,1,0" \
	"1,0,0,8,1,0,0,0,1,0" \
	"1,0,0,0,1,0,1,0,1,0" \
	"1,0,0,0,1,0,2,0,1,0" \
	"1,0,0,0,1,0,4,0,1,0" \
	"1,0,0,0,1,0,8,0,1,0" \
	"1,0,0,0,1,0,0,1,1,0" \
	"1,0,0,0,1,0,0,2,1,0" \
	"1,0,0,0,1,0,0,4,1,0" \
	"1,0,0,0,1,0,0,8,1,0" \
	"1,0,0,0,1,0,0,0,1,1" \
	-end -group_theoretic_activities -report -end
	pdflatex PGGL_3_16_Subgroup_flag_stab_3686400_report.tex
	open PGGL_3_16_Subgroup_flag_stab_3686400_report.pdf
	
PG_2_16:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 2 16
	pdflatex PG_2_16.tex 
	open PG_2_16.pdf


max_arc_flag_stabilizer_on_pencil:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -PGGL 3 16 -subgroup_by_generators "flag_stab" 3686400 16 \
	"-1,0,0,0,1,0,0,0,1,0" \
	"1,0,0,0,-1,0,0,0,1,0" \
	"1,0,0,0,1,0,0,0,-1,0" \
	"1,0,0,1,1,0,0,0,1,0" \
	"1,0,0,2,1,0,0,0,1,0" \
	"1,0,0,4,1,0,0,0,1,0" \
	"1,0,0,8,1,0,0,0,1,0" \
	"1,0,0,0,1,0,1,0,1,0" \
	"1,0,0,0,1,0,2,0,1,0" \
	"1,0,0,0,1,0,4,0,1,0" \
	"1,0,0,0,1,0,8,0,1,0" \
	"1,0,0,0,1,0,0,1,1,0" \
	"1,0,0,0,1,0,0,2,1,0" \
	"1,0,0,0,1,0,0,4,1,0" \
	"1,0,0,0,1,0,0,8,1,0" \
	"1,0,0,0,1,0,0,0,1,1" \
	-on_k_subspaces 2 \
	-restricted_action "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16" \
	-end -group_theoretic_activities -report \
		-orbits_on_subsets 3 -end 
	
no:
	pdflatex PGGL_3_16_Subgroup_flag_stab_3686400_report.tex
	open PGGL_3_16_Subgroup_flag_stab_3686400_report.pdf


# lines through 0:
#0 & 0 & 1 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 & 17 & 18\\
#1 & 0 & 34 & 3 & 35 & 36 & 37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48\\
#2 & 0 & 209 & 221 & 210 & 222 & 211 & 223 & 212 & 224 & 213 & 217 & 214 & 218 & 215 & 219 & 216 & 220\\
#3 & 0 & 145 & 153 & 154 & 146 & 156 & 148 & 147 & 155 & 160 & 152 & 151 & 159 & 149 & 157 & 158 & 150\\
#4 & 0 & 113 & 119 & 125 & 123 & 114 & 120 & 126 & 124 & 115 & 117 & 127 & 121 & 116 & 118 & 128 & 122\\
#5 & 0 & 257 & 272 & 264 & 265 & 271 & 258 & 266 & 263 & 262 & 267 & 259 & 270 & 268 & 261 & 269 & 260\\
#6 & 0 & 81 & 85 & 89 & 93 & 90 & 94 & 82 & 86 & 92 & 96 & 84 & 88 & 83 & 87 & 91 & 95\\
#7 & 0 & 241 & 255 & 246 & 252 & 251 & 245 & 256 & 242 & 254 & 244 & 249 & 247 & 248 & 250 & 243 & 253\\
#8 & 0 & 65 & 68 & 71 & 70 & 77 & 80 & 75 & 74 & 66 & 67 & 72 & 69 & 78 & 79 & 76 & 73\\
#9 & 0 & 225 & 238 & 228 & 239 & 231 & 236 & 230 & 233 & 237 & 226 & 240 & 227 & 235 & 232 & 234 & 229\\
#10 & 0 & 193 & 204 & 208 & 197 & 200 & 205 & 201 & 196 & 207 & 198 & 194 & 203 & 202 & 195 & 199 & 206\\
#11 & 0 & 177 & 187 & 190 & 184 & 180 & 186 & 191 & 181 & 183 & 189 & 188 & 178 & 182 & 192 & 185 & 179\\
#12 & 0 & 49 & 51 & 53 & 55 & 57 & 59 & 61 & 63 & 58 & 60 & 62 & 64 & 50 & 52 & 54 & 56\\
#13 & 0 & 161 & 170 & 172 & 163 & 176 & 167 & 165 & 174 & 168 & 175 & 173 & 166 & 169 & 162 & 164 & 171\\
#14 & 0 & 129 & 136 & 143 & 138 & 134 & 131 & 140 & 141 & 139 & 142 & 133 & 132 & 144 & 137 & 130 & 135\\
#15 & 0 & 97 & 102 & 107 & 112 & 110 & 105 & 104 & 99 & 100 & 103 & 106 & 109 & 111 & 108 & 101 & 98\\
#16 & 0 & 2 & 19 & 20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 & 33\\


max_arc_flag_stabilizer_pencil1:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -PGGL 3 16 -subgroup_by_generators "pencil1_group" 7680 5 \
	"1,0,0,0,9,0,0,0,9,0," \
	"1,0,0,0,1,0,0,0,8,2," \
	"1,0,0,0,1,0,3,0,1,0," \
	"1,0,0,0,7,0,2,0,7,0," \
	"1,0,0,4,1,0,13,0,1,0," \
	-restricted_action "113,119,125,123,114,120,126,124,115,117,127,121,116,118,128,122,257,272,264,265,271,258,266,263,262,267,259,270,268,261,269,260,81,85,89,93,90,94,82,86,92,96,84,88,83,87,91,95,241,255,246,252,251,245,256,242,254,244,249,247,248,250,243,253,65,68,71,70,77,80,75,74,66,67,72,69,78,79,76,73,225,238,228,239,231,236,230,233,237,226,240,227,235,232,234,229,193,204,208,197,200,205,201,196,207,198,194,203,202,195,199,206,177,187,190,184,180,186,191,181,183,189,188,178,182,192,185,179,49,51,53,55,57,59,61,63,58,60,62,64,50,52,54,56,161,170,172,163,176,167,165,174,168,175,173,166,169,162,164,171,129,136,143,138,134,131,140,141,139,142,133,132,144,137,130,135,97,102,107,112,110,105,104,99,100,103,106,109,111,108,101,98,2,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33" \
	-end -group_theoretic_activities -report \
		-orbits_on_subsets 5 -end 


max_arc_flag_stabilizer_of_two_pencils:
	$(ORBITER_PATH)/orbiter.out -v 4 -linear_group -PGGL 3 16 -subgroup_by_generators "flags_stab" 460800 13 \
	"-1,0,0,0,1,0,0,0,1,0" \
	"1,0,0,0,-1,0,0,0,1,0" \
	"1,0,0,0,1,0,0,0,-1,0" \
	"1,0,0,0,1,0,1,0,1,0" \
	"1,0,0,0,1,0,2,0,1,0" \
	"1,0,0,0,1,0,4,0,1,0" \
	"1,0,0,0,1,0,8,0,1,0" \
	"1,0,0,0,1,0,0,1,1,0" \
	"1,0,0,0,1,0,0,2,1,0" \
	"1,0,0,0,1,0,0,4,1,0" \
	"1,0,0,0,1,0,0,8,1,0" \
	"1,0,0,0,1,0,0,0,1,1" \
	"0,1,0,1,0,0,0,0,1,0" \
	-on_k_subspaces 2 \
	-restricted_action "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,272" \
	-end -group_theoretic_activities -report \
		-orbits_on_subsets 6 -end 

max_arc_system_1:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant -q 16 -maximal_arc 52 4 \
		"1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,34,51,68,85,102,119,136,153,170,187,204,221,238,255,272,0" \
		"0,1,2,16,17,18,32"

max_arc_system_1_solve:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -solve_mckay 


max_arc_system_1_draw:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -draw 
	mpost max_arc_16_52_4_drawing.mp
	latex view.tex
	dvips view.dvi -o
	open view.ps

max_arc_system_1_red:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -perform_column_reductions 

max_arc_system_1_red_draw:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -draw 
	mpost max_arc_16_52_4_red_drawing.mp
	latex view2.tex
	dvips view2.dvi -o
	open view2.ps
	
max_arc_system_1_red_draw_bitmap:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -draw_as_bitmap 5


max_arc_system_1_red_solve:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4_red.diophant -solve_mckay 


max_arc_system_1_equations:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -test_single_equation 20 

max_arc_system_1_split273:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_single_equation_and_solve 273 0 

max_arc_system_1_split0:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_single_equation_and_solve 0 1
	
max_arc_system_1_split0_1:
	$(ORBITER_PATH)/orbiter.out -v 4 -diophant_activity -input_file max_arc_16_52_4.diophant -project_to_two_equations_and_solve 0 1 0 100000





##########################




PG_2_16:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 2 16
	pdflatex PG_2_16.tex 
	open PG_2_16.pdf







max_arc_16_4:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGGL 3 16 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
			-end \
			-classify_arcs 52 \
			-classify_arcs_d 4  \
			-exact_cover \
				-input_prefix ./ARCS/ \
				-output_prefix ./SYSTEMS/ \
				-solution_prefix ./SOLUTIONS/ \
				-starter_size 52 \
				-base_fname maxarcs_q16_d4 \
				-lex \
			-end \
		-end

max_arc_lift:
	$(ORBITER_PATH)/orbiter.out -v 2 \
		-process_combinatorial_objects -q 16 -n 2 \
		-arc_with_given_set_as_s_lines_after_dualizing \
			52 4 4 0 \
		-fname_base_out maxarcs_q16_d4_lvl_9_lifted \
		-input \
			-file_of_points maxarcs_q16_d4_lvl_9 \
		-end \
		-end 



#############################################################################################################
# Section 18: Finite Geometry




spreads4:
	- mkdir SPREADS_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 10 \
		-linear_group -PGL 4 2 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_4 \
			-end \
			-spread_classify 2 \
		-end

spreads16_4:
	- mkdir SPREADS_16_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_16_4 \
			-end \
			-spread_classify 2 \
		-end

#17 : 3 orbits
#total: 1129
#(81600, 1200, 576) average is 27792 + 0 / 3


spreads64_4:
	- mkdir SPREADS_64_4
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 6 4 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_4 \
			-end \
			-spread_classify 3 \
		-end


# 853 orbits at level 5, each with about 60k live points

spreads64_8:
	- mkdir SPREADS_64_8
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 3 \
		-linear_group -PGGL 4 8 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_64_8 \
			-end \
			-spread_classify 2 \
		-end


# 16766 orbits at level 6, each with about 2k live points


spreads16:
	- mkdir SPREADS_16
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -PGL 8 2 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label spreads_16 \
			-end \
			-spread_classify 4 \
		-end

# 8 spreads of order 16
# May 14, 2020: 52 sec on Mac


line_spreads_PG5_2:
	- rm live_points.txt
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGL 6 2 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label line_spreads_PG_5_2 \
				-depth 21 \
			-end \
			-spread_classify 2 \
		-end


# 16167 nodes at level 7
# 357991 nodes at levl 8 with about 150 live points each; 52371304 extension nodes






#############################################################################################################
# Section 19: The Povray Interface

cube:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask cube_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.5 -default_angle 75 -clipping_radius 2.7 \
		-end \
		-scene_objects \
			-obj_file cube_centered.obj \
			-edge "0, 1" \
			-edge "0, 2" \
			-edge "0, 4" \
			-edge "1, 3" \
			-edge "1, 5" \
			-edge "2, 3" \
			-edge "2, 6" \
			-edge "3, 7" \
			-edge "4, 5" \
			-edge "4, 6" \
			-edge "5, 7" \
			-edge "6, 7" \
			-group_of_things_as_interval 0 8 \
			-spheres 0 0.3 "texture{ Polished_Chrome pigment{quick_color White} }" \
			-group_of_things_as_interval 0 6 \
			-prisms 1 0.05 "texture{ pigment{ color Yellow transmit 0.7 } finish {diffuse 0.9 phong 0.6} }" \
			-group_of_things_as_interval 0 12 \
			-cylinders 2 0.15 "texture{ pigment{ color Red  } finish {diffuse 0.9 phong 0.6} }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv cube_0_*.pov POV
	mv makefile_animation POV
	
monkey:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask monkey_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.8 -default_angle 75 -clipping_radius 0.8 \
		-camera 0 "0,0,1" "1,1,0.5" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-cubic_lex "1,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ Gold } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 1 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 0.2}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv monkey_0_*.pov POV
	mv makefile_animation POV

HCV:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask HCV_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 75 -clipping_radius 2.4 \
		-camera 0 "1,1,1" "-3,1,3" "0.12,0.12,0.12"  \
		-end \
		-scene_objects \
			-Hilbert_Cohn_Vossen_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval_with_exceptions 12 15 "14,19,23" \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-label 0 "a1" \
			-label 2 "a2" \
			-label 4 "a3" \
			-label 6 "a4" \
			-label 8 "a5" \
			-label 10 "a6" \
			-label 12 "b1" \
			-label 14 "b2" \
			-label 16 "b3" \
			-label 18 "b4" \
			-label 20 "b5" \
			-label 22 "b6" \
			-label 24 "c12" \
			-label 26 "c13" \
			-label 30 "c15" \
			-label 32 "c16" \
			-label 34 "c23" \
			-label 36 "c24" \
			-label 40 "c26" \
			-label 42 "c34" \
			-label 44 "c35" \
			-label 48 "c45" \
			-label 50 "c46" \
			-label 52 "c56" \
			-group_of_things_as_interval 0 6 \
			-texts 4 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 6 6 \
			-texts 5 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
			-group_of_things_as_interval 12 12 \
			-texts 6 0.2 0.15 "texture{ pigment{Black} } no_shadow" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv HCV_0_*.pov POV
	mv makefile_animation POV

#"-3,2.333,4"  * 1.5 = "-4.5,3.5,6"
#M := Matrix([[-4.5, 3.5, 6], [1, 1, 1]])
#NullSpace(M)
#=0.186080731891197,-0.781539073943026,0.595458342051830
#		-rotate_about_custom_axis "0.186080731891197,-0.781539073943026,0.595458342051830" \
#-W 1024 -H 768 
#-W 2560 -H 1920 
#-W 4096 -H 3072 

Clebsch:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask Clebsch_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.9 -default_angle 80 -clipping_radius 2.4 \
		-camera 0 "1,1,1"  "-4.5,3.5,6" "0,0,0"  \
		-end \
		-scene_objects \
			-Clebsch_surface \
			-group_of_things "0" \
			-cubics 0 "texture{ pigment{ White*0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-group_of_things_as_interval 0 6 \
			-group_of_things_as_interval 6 6 \
			-group_of_things_as_interval 12 15  \
			-lines 1 0.02 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.02 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.02 "texture{ pigment{ color Yellow } finish { diffuse 0.9 phong 1}}" \
			-group_of_things_as_interval 0 12 \
			-spheres 4 0.08 "texture{ pigment{Cyan*1.3} finish {ambient 0.4 diffuse 0.6 roughness 0.001 reflection 0 specular .8}  }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv Clebsch_0_*.pov POV
	mv makefile_animation POV




lissajous:
	$(ORBITER_PATH)/orbiter.out -v 2 -smooth_curve "lissajous" 0.07 2000 15 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
		-code_end \

#function_lissajous_N2000_points.csv

lissajous_plot:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 1 -output_mask lissajous_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.40 \
		-default_angle 45 -clipping_radius 5 -omit_bottom_plane  \
		-camera 0 "0,-1,0" "0,0,12" "0,0,0"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_N2000_points.csv \
			-group_of_things_as_interval 0 6524\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_0_*.pov POV
	mv makefile_animation POV

lissajous_3d:
	$(ORBITER_PATH)/orbiter.out -v 2 -smooth_curve "lissajous_3d" 0.07 2000 50 0 18.85 \
		-const a 3 b 2 c 1.57 r 7 -const_end \
		-var t -var_end \
		-code \
			push t push a mult push c add sin push r mult return \
			push t push b mult sin push r mult  return \
			push t return \
		-code_end \

lissajous_3d_plot:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask lissajous_3d_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.40 \
		-default_angle 45 -clipping_radius 5 -omit_bottom_plane  \
		-camera 0 "0,0,1" "7,7,5" "0,0,1"  \
		-rotate_about_z_axis \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-lines 0 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 1 0.09 "texture{ pigment{ color Yellow } }" \
			-lines 2 0.09 "texture{ pigment{ color Yellow } }" \
			-group_of_things_as_interval 3 39 \
			-lines 3 0.02 "texture{ pigment{ color Black } }" \
			-point_list_from_csv_file function_lissajous_3d_N2000_points.csv \
			-group_of_things_as_interval 0 6538\
			-spheres 4 0.1 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv lissajous_3d_0_*.pov POV
	mv makefile_animation POV

#			-planes 5 "texture{ pigment{ color Blue*0.5 transmit 0.5 } }" \






endrass8:
	$(ORBITER_PATH)/orbiter.out -v 2 -povray \
		-round 0 -nb_frames_default 30 -output_mask endrass_octic_%d_%03d.pov \
		-video_options -W 1024 -H 768  -global_picture_scale 0.75 \
		-default_angle 75 -clipping_radius 3.7 -no_bottom_plane \
		-camera 0 "1,1,1" "6,6,3" "0,0,0"  \
		-rotate_about_111 \
		-end \
		-scene_objects \
			-line_through_two_points_recentered_from_csv_file coordinate_grid.csv \
			-group_of_things "0" \
			-group_of_things "1" \
			-group_of_things "2" \
			-group_of_things_as_interval 3 39 \
			-lines 0 0.15 "texture{ pigment{ color Red } finish { diffuse 0.9 phong 1}}" \
			-lines 1 0.15 "texture{ pigment{ color Green } finish { diffuse 0.9 phong 1}}" \
			-lines 2 0.15 "texture{ pigment{ color Blue } finish { diffuse 0.9 phong 1}}" \
			-lines 3 0.05 "texture{ pigment{ color Black } finish { diffuse 0.9 phong 1}}" \
			-octic_lex_165 "-93.2548,0,0,0,-309.019,0,0,527.529,0,395.647,0,0,0,0,0,0,0,0,0,0,-687.529,0,0,1582.59,0,1186.94,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-309.019,0,0,1582.59,0,1186.94,0,0,0,0,-2110.12,0,-3165.17,0,-1186.94,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-93.2548,0,0,527.529,0,395.647,0,0,0,0,-1055.06,0,-1582.59,0,-593.47,0,0,0,0,0,0,874.039,0,1560.63,0,1677.92,0,343.362,0,0,0,0,0,0,0,0,-256,0,-468.077,0,-789.019,0,-525.726,0,0.941125" \
			-plane_by_dual_coordinates "0,0,1,0" \
			-group_of_things "0" \
			-group_of_things "0" \
			-octics 4 "texture{ pigment{ White*0.5 transmit 0.5 } finish {ambient 0.4 diffuse 0.5 roughness 0.001 reflection 0.1 specular .8} }" \
			-planes 5 "texture{ pigment{ color Blue transmit 0.5 } finish { diffuse 0.9 phong 1}}" \
		-scene_objects_end \
		-povray_end
	- rm -rf POV
	mkdir POV
	mv endrass_octic_0_*.pov POV
	mv makefile_animation POV


#############



no:
			-exact_cover \
				-input_prefix ./SPREADS_16/ \
				-output_prefix ./SPREADS_16/ \
				-solution_prefix ./SPREADS_16/ \
				-starter_size 52 \
				-base_fname spreads_16 \
				-lex \
			-end \

no:
	pdflatex GL_2_2_wreath_Sym4_res81_orbits.tex
	open GL_2_2_wreath_Sym4_res81_orbits.pdf

no:
	pdflatex GL_2_2_wreath_Sym4_res65535_orbits.tex
	open GL_2_2_wreath_Sym4_res65535_orbits.pdf

T4c:
	$(ORBITER_PATH)/orbiter.out -v 4 \
		-linear_group -GL_d_q_wr_Sym_n 2 2 4 -on_tensors -end \
		-group_theoretic_activities \
		-tensor_classify 4 


no:
		-orbits_on_subsets 4 \
		-draw_poset \
		-report

# group of order 31104 = 6^4 * 24



PG_2_4:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 2 4
	pdflatex PG_2_4.tex 
	open PG_2_4.pdf


PROBLEM_27_53 = -d1 1 -q1 27 -d2 1 -q2 53 -K 11 -DDx 2 -DDy 1 -search_control -W -end 
PROBLEM_27_53_GROUP1=-subgroup "1,1,1,0, 1,3,1,0, 1,9,1,0, 1,0,1,1, -2,0,-4,0" "18603" -group_label "group1"

# mask 1:
#  XX.
#  X.X

PROBLEM_27_53_MASK1=		-masktest 3 x ge 1 \
		-masktest 4 x+y ge 3 \
		-depth 4 \
		-mask_label "mask1"

# 18603 = 27 * 53 * 13

DD_M1_G1:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen  $(PROBLEM_27_53) $(PROBLEM_27_53_GROUP1) $(PROBLEM_27_53_MASK1) \
			-end 

DD_M1_G1_S:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen $(PROBLEM_27_53) $(PROBLEM_27_53_GROUP1) $(PROBLEM_27_53_MASK1) \
				-singletons \
			-end 

PROBLEM_3_7 = -d1 1 -q1 3 -d2 1 -q2 7 -K 5 -DDx 3 -DDy 1 -search_control -W -end 
PROBLEM_3_7_GROUP1=-subgroup "1,1,1,0, 1,0,1,1 " "21" -group_label "group_cyclic"
PROBLEM_3_7_MASK1= -mask_label "mask1" -depth 5

DD_PG_2_4_M1_G1:
	$(ORBITER_PATH)/orbiter.out -v 4  \
		-Delandtsheer_Doyen  $(PROBLEM_3_7) $(PROBLEM_3_7_GROUP1) $(PROBLEM_3_7_MASK1) \
			-end 

PG_2_27:
	$(ORBITER_PATH)/orbiter.out -override_polynomial 46 -cheat_sheet_PG 2 27 
	pdflatex PG_2_27.tex 
	open PG_2_27.pdf



canon_design:
	$(ORBITER_PATH)/orbiter.out -v 4 -override_polynomial 46 -canonical_form_PG  2 27 \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-prefix design \
		-save design_classified \
		-report


design_line_type:
	$(ORBITER_PATH)/orbiter.out -v 4 -override_polynomial 46 \
		-process_combinatorial_objects -n 2 -q 27 -poly 46 \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-line_type \
		-fname_base_out design \
		-end
		
design_conic_type:
	$(ORBITER_PATH)/orbiter.out -v 4 -override_polynomial 46 \
		-process_combinatorial_objects -n 2 -q 27 -poly 46 \
		-input \
		-set_of_points "2,56,30,112,253,90,440,508" \
		-end \
		-conic_type \
		-fname_base_out design \
		-end


convert_stack_tdo_32_der:
	$(ORBITER_PATH)/orbiter.out -v 4 -convert_stack_to_tdo 32_der.widor
	
	
#<HTDO id=32_der type=pt ptanz=3 btanz=2 fuse=simple>
#      561 496
#   4   33   0
#  30   33   0
#1023   17  16
#
#1 1 2
#</HTDO>


packing:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
		-end


# elements of order 2:
# conjugacy class reps:
# elt order, class size, centralizer order 

#2A: 2 48960 40320 Baer involution

#2B: 2 5355 368640  one block of 10,11

#2C: 2 64260 30720 two blocks of 10,11 (problem group)



# Baer involution:

packing_2A:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 6 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2A" 2 1 \
					"1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2A" "40320" 10 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0," \
					"1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0," \
					"1,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0," \
					"1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,0," \
					"0,1,0,0,0,1,0,1,1,1,0,1,0,1,1,1,1," \
				-end \
				-cliques_on_fixpoint_graph 7 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2A_2_fixp_graph_cliques -end \
				-process_long_orbits 7 0 1000 2 7 \
			-end \
		-end

# creates PGGL_4_4_Subgroup_2A_2_fpc0_graph and many more


packing_2Ac6_draw:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc6_graph -end \
		-graph_theoretic_activity -export_csv PGGL_4_4_Subgroup_2A_2_fpc6_graph.csv -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGGL_4_4_Subgroup_2A_2_fpc6_graph.csv 2


packing_2A_clique0:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -target_size 7 -end -end

# 578 solutions in 0:26 on Mac


packing_2A_clique0_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 7 -end -end




packing_2A_draw:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2A_2_fixp_graph.bin -end \
		-graph_theoretic_activity -export_csv PGGL_4_4_Subgroup_2A_2_fixp_graph.csv -end
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix PGGL_4_4_Subgroup_2A_2_fixp_graph.bin.csv 2

# 440 orbits on cliques of size 7 on the fixpoint graph

#orbit_transversal::read_from_file fname = G PGGL_4_4_Subgroup_2A_2_fixp_graph_cliques_lvl_7 
#		reduced_spread_orbits_under_H->Orbits_classified->Set_size[fixpoints_idx],
#		reduced_spread_orbits_under_H->Orbits_classified->Sets[fixpoints_idx],


#orbits_on_something::classify_orbits_by_length The distribution of orbit lengths is: 1^1680, 2^90720
#1:17
# writes PGGL_4_4_Subgroup_2A_2_fixp_graph.bin
#-rw-r--r--   1 betten  staff       5886 Jun 11 00:31 PGGL_4_4_Subgroup_2A_2_line_orbits_orbits.bin
#-rw-r--r--   1 betten  staff   81549922 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_spread_orbits_orbits.bin
#-rw-r--r--   1 betten  staff    1440222 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_good_orbits
#-rw-r--r--   1 betten  staff    2936674 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_reduced_spread_orbits_orbits.bin
#-rw-r--r--   1 betten  staff     196479 Jun 11 01:52 PGGL_4_4_Subgroup_2A_2_fixp_graph.bin
#-rw-r--r--@  1 betten  staff   35192935 Jun 11 01:52 log




packing_2C:
	- mkdir SPREAD_TABLES_4
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-linear_group -PGGL 4 4 -end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label packing_4 \
			-end \
			-packing_classify 2 "0,1,2" "SPREAD_TABLES_4/" \
			-packing_with_assumed_symmetry \
				-H -PGGL 4 4 -subgroup_by_generators "2C" 2 1 \
					"1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 0" \
				-end \
				-N -PGGL 4 4 -subgroup_by_generators "centralizer_2C" "30720" 9 \
					"1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1," \
					"1,0,0,0,0,1,0,0,0,0,2,0,0,0,0,2,1," \
					"1,0,0,0,0,1,0,0,0,0,1,0,2,0,3,1,0," \
					"1,0,0,0,0,1,0,0,0,0,1,0,1,0,3,1,0," \
					"1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,1," \
					"1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0," \
					"1,0,0,0,2,1,0,0,0,0,1,0,1,0,0,1,0," \
					"1,0,0,0,1,1,2,0,0,0,1,0,0,0,0,1,0," \
					"1,0,3,0,1,1,1,3,0,0,2,0,0,0,0,2,1," \
				-end \
				-cliques_on_fixpoint_graph 5 \
				-cliques_on_fixpoint_graph_control -W -problem_label PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques -end \
				-process_long_orbits 5 0 1 2 8 \
			-end \
		-end

packing_2C_clique0:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2C_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -target_size 8 -end -end

# ripoff:
# nb colors = 1
# nb vertices = 29901
#nb_sol = 7442
#8:6:41


# computed on ripoff:
#depth : number of orbits
#0 : 1
#1 : 8
#2 : 426
#3 : 70315
#4 : 1557542
#5 : 776928
#6 : 0
#7 : 0
# creates the file PGGL_4_4_Subgroup_2C_2_fixp_graph_cliques.csv
# creates PGGL_4_4_Subgroup_2C_2_fpc0_graph



packing_2C_clique0_Sajeeb:
	$(ORBITER_PATH)/orbiter.out -v 2 -create_graph -load_from_file PGGL_4_4_Subgroup_2C_2_fpc0_graph -end \
		-graph_theoretic_activity -find_cliques -Sajeeb -target_size 8 -end -end


F_17:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 17 
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q17_addition_table.csv 40
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix GF_q17_multiplication_table.csv 40
	pdflatex GF_17.tex
	open GF_17.pdf


F_289:
	$(ORBITER_PATH)/orbiter.out -v 3 -cheat_sheet_GF 289 
	pdflatex GF_289.tex
	open GF_289.pdf


ntt:
	$(ORBITER_PATH)/orbiter.out -v 2 -NTT 4 17 ntt4.cpp


FGDTP:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_F_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Fv_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_AAv_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Tv_k4.csv 20
	
no:
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_F_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Gr_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Dr_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Tr_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_Pr_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k3.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_G_k2.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_D_k2.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_T_k2.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_P_k2.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_the_P_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix ntt_L_k4.csv 20
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N16.csv 40
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N8.csv 40
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N4.csv 40
	$(ORBITER_PATH)/orbiter.out -v 2 -draw_matrix Fourier_q17_N2.csv 40
	
SRC=../../DEV.20/GITHUB/orbiter/src
DISCRETA_CPP = g++
DISCRETA_CC = gcc
CPPFLAGS = -Wall -I$(SRC)/lib -std=c++14
LIB = $(SRC)/lib/liborbiter.a -lpthread
LFLAGS = -lm -Wl,-rpath -Wl,/usr/local/gcc-8.2.0/lib64




ntt4.out:  ntt4.o ;
	$(DISCRETA_CPP)  ntt4.o $(CPPFLAGS) $(LIB) $(LFLAGS) -o $@



conf_10_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 10 -n 10 -r 3 -k 3 -target_size 10  \
		-control -problem_label 10_3 -W -end -end
	
conf_11_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 11 -n 11 -r 3 -k 3 -target_size 11  \
		-control -problem_label 11_3 -w -end -end
	
conf_12_3:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 12 -n 12 -r 3 -k 3 -target_size 12  \
		-control -problem_label 12_3 -w -end -end
	
STS13:
	$(ORBITER_PATH)/orbiter.out -v 2 -regular_linear_space_classify -m 13 -n 26 -r 6 -k 3 -target_size 26  \
		-control -problem_label STS13 -w -depth 7 -end -end
	
	
PG_4_3:
	$(ORBITER_PATH)/orbiter.out -cheat_sheet_PG 4 3
	pdflatex PG_4_3.tex 
	open PG_4_3.pdf


AGL_4_3:
	$(ORBITER_PATH)/orbiter.out -v 5 \
		-linear_group -PGL 5 3 -subgroup_by_generators "AGL_4_3" "1965150720" 14 \
			"1,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1," \
			"1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0," \
			"1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1," \
			"1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1," \
			-end \
		-group_theoretic_activities \
			-poset_classification_control \
				-W \
				-problem_label arcs \
			-end \
			-report \
		-end
