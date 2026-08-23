
#MY_PATH=../orbiter
MY_PATH=~/DEV.22/orbiter

#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/

EXTRACT=./extract.sh $(ORBITER_PATH) ~/bin/ the_makefile

all:
	~/bin/a2tex.out -numbers -text_width 80 <the_makefile >the_makefile.tex
	~/bin/a2tex.out -text_width 10000  <the_makefile >the_makefile_nonumbers.tex
	~/bin/a2tex.out -text_width 10000  <C13.makefile >C13_makefile.tex
	~/bin/a2tex.out -text_width 10000  <PGL_4_2.makefile >PGL_4_2_makefile.tex
	~/bin/a2tex.out -text_width 10000  <Paley_13_group.makefile >Paley_13_group_makefile.tex
	~/bin/a2tex.out -text_width 10000  <15_lines.makefile >15_lines_makefile.tex
	~/bin/a2tex.out -text_width 10000  <Cycle_13_gens.csv >Cycle_13_gens_csv.tex


extract_all: preamble \
	extract_2_1 extract_2_3 extract_2_4 extract_2_5 extract_2_6 extract_2_7 \
	extract_3_1 extract_3_2 extract_3_3 extract_3_4 extract_3_5 extract_3_6 \
	extract_4_1 extract_4_2 extract_4_3 extract_4_4 extract_4_5 extract_4_6 extract_4_7 extract_4_8 extract_4_9 extract_4_10 \
	extract_5_1 extract_5_2 extract_5_3 extract_5_4 extract_5_5 extract_5_6 extract_5_7 \
	extract_6_1 extract_6_2 extract_6_3 extract_6_4 extract_6_5 extract_6_6 \
	extract_7_1 extract_7_2 extract_7_3 extract_7_4 extract_7_5 extract_7_6 \
	extract_8_1 extract_8_2 \
	extract_9_1 extract_9_2 extract_9_3 \
	extract_10_1 extract_10_2 extract_10_3 extract_10_4 extract_10_5 extract_10_6 extract_10_7 extract_10_8 extract_10_9 \
	extract_11_1 extract_11_2 extract_11_3 extract_11_4 extract_11_5 extract_11_6 extract_11_7 extract_11_8 \
	extract_12_1 extract_12_2 extract_12_3 extract_12_4 \
	extract_13_1 extract_13_2 extract_13_3 extract_13_4 \
	extract_14_1 extract_14_2 \
	extract_15_1 extract_15_2 extract_15_3 extract_15_4 extract_15_5 extract_15_6 extract_15_7 extract_15_8 \
	extract_16_1 extract_16_2 extract_16_3 extract_16_4 \
	extract_17_1 extract_17_2


preamble:
	$(EXTRACT) F_abcd_eqn


extract_2_1:
	$(EXTRACT) MY_PATH
	$(EXTRACT) ORBITER_PATH


extract_2_3:
	$(EXTRACT) F_16

extract_2_4:
	$(EXTRACT) object_F_2
	$(EXTRACT) object_PG_3_2


extract_2_5:
	$(EXTRACT) Cremona_map
	$(EXTRACT) create_BLT_5_1
	$(EXTRACT) create_surface_4_0


extract_2_6:
	$(EXTRACT) set_of_primes
	$(EXTRACT) set_interval
	$(EXTRACT) set_builder_examples


extract_2_7:
	$(EXTRACT) vector_example1
	$(EXTRACT) vector_example2
	$(EXTRACT) vector_example_sparse
	$(EXTRACT) vector_example_repeat
	$(EXTRACT) vector_example_all_one_11
	$(EXTRACT) matrix_example1
	$(EXTRACT) matrix_example_co_1
	$(EXTRACT) CONWAY_GEN1
	$(EXTRACT) HAMMING_CODE

extract_3_1:
	$(EXTRACT) order_of_2_mod_n
	$(EXTRACT) Eulerfunction_150
	$(EXTRACT) power_function_2_mod_11
	$(EXTRACT) PR29
	$(EXTRACT) PR_915839
	$(EXTRACT) PR_915839_check
	$(EXTRACT) DL_915839
	$(EXTRACT) IM
	$(EXTRACT) IM_gcd
	$(EXTRACT) PM3a
	$(EXTRACT) sqrt_mod
	$(EXTRACT) draw_mod_13
	$(EXTRACT) draw_mod_127


extract_3_2:
	$(EXTRACT) F_2
	$(EXTRACT) F_11_product_of_all_nonzero_elements
	$(EXTRACT) F_7_vandermonde
	$(EXTRACT) F_101_wo



extract_3_3:
	$(EXTRACT) F_4
	$(EXTRACT) F_16


extract_3_4:
	$(EXTRACT) RREF
	$(EXTRACT) RREF_V7
	$(EXTRACT) nullspace
	$(EXTRACT) eigenstuff
	$(EXTRACT) classes_GL_3_2


extract_3_5:
	$(EXTRACT) normal_basis_2_3
	$(EXTRACT) F_64_over_F8_field_reduction
	$(EXTRACT) F_8_Nth_roots_21
	$(EXTRACT) F_8_vandermonde
	$(EXTRACT) F_1024_vandermonde
	$(EXTRACT) NTT_k4_q17.cpp
	$(EXTRACT) F_17_NTT_compile
	$(EXTRACT) SRC
	$(EXTRACT) V7_VANDERMONDE_EXTENDED


extract_3_6:
	$(EXTRACT) Polynomial_ring

extract_4_1:
	$(EXTRACT) PG_3_2_easy
	$(EXTRACT) PG_2_4
	$(EXTRACT) PG_3_2

extract_4_2:
	$(EXTRACT) PG_2_4_rank_point
	$(EXTRACT) PG_2_8_incidence_matrix


extract_4_3:
	$(EXTRACT) PG_2_16
	$(EXTRACT) PG_2_4_with_decomposition
	$(EXTRACT) PG_2_4_incma_cyclic
	$(EXTRACT) PG_2_4_incma_singer_sub_3


extract_4_4:
	$(EXTRACT) GR_3_2_2
	$(EXTRACT) rank_lines


extract_4_5:
	$(EXTRACT) EC_11.txt
	$(EXTRACT) Hirschfeld_surface_q4.txt
	$(EXTRACT) Endrass_F7.txt
	$(EXTRACT) octic_prepare
	$(EXTRACT) ENDRASS_SPARSE
	$(EXTRACT) HIRSCHFELD_SURFACE_EQUATION
	$(EXTRACT) EC_11_EQUATION

extract_4_6:
	$(EXTRACT) GR_4_2_2



extract_4_7:
	$(EXTRACT) Op_4_2
	$(EXTRACT) O_5_2_incidence_matrix.csv
	$(EXTRACT) Op_6_2
	$(EXTRACT) Op_6_64_line_rank
	$(EXTRACT) Op_6_64_report


extract_4_8:
	$(EXTRACT) H_2_4
	$(EXTRACT) H_3_4

extract_4_9:
	$(EXTRACT) fix_structure_2A
	$(EXTRACT) trans
	$(EXTRACT) del_Pezzo_F13ab_report


extract_4_10:
	$(EXTRACT) elliptic_quadric_ovoid_q8
	$(EXTRACT) ovoid_ST_q8
	$(EXTRACT) Edge_curve_17
	$(EXTRACT) Edge_curve_17_line_type
	$(EXTRACT) EDGE_CURVE_Q17_EQUATION
	$(EXTRACT) EDGE_CURVE_Q17_AS_POINTS
	$(EXTRACT) FILE_Q17


extract_5_1:
	$(EXTRACT) GEN_C13
	$(EXTRACT) C13
	$(EXTRACT) C13_generated
	$(EXTRACT) Symmetric_3
	$(EXTRACT) PGL_4_2_export
	$(EXTRACT) PGL_4_2_generated
	$(EXTRACT) C13_as_subgroup


extract_5_2:
	$(EXTRACT) PGL_4_2
	$(EXTRACT) L_5_3
	$(EXTRACT) PSP_4_4
	$(EXTRACT) PGO_5_2
	$(EXTRACT) PSP_6_2
	$(EXTRACT) PGO_7_2


extract_5_3:
	$(EXTRACT) J1
	$(EXTRACT) PGL_3_11_singer
	$(EXTRACT) PGL_3_11_singer_and_frobenius
	$(EXTRACT) quaternion
	$(EXTRACT) cube_group
	$(EXTRACT) tetra_group
	$(EXTRACT) Hesse_group
	$(EXTRACT) Weyl_E8
	$(EXTRACT) GENERATORS_HESSE_GROUP
	$(EXTRACT) GENERATORS_WEYL_GROUP_E8



extract_5_4:
	$(EXTRACT) U_3_3
	$(EXTRACT) Co3
	$(EXTRACT) Ree_27
	$(EXTRACT) CONWAY_GEN1
	$(EXTRACT) CONWAY_GEN2
	$(EXTRACT) Ree_gen1
	$(EXTRACT) Ree_gen2



extract_5_5:
	$(EXTRACT) T3_on_tensors
	$(EXTRACT) T3r1
	$(EXTRACT) PGGL_2_8_on_conic
	$(EXTRACT) surface_q13_stab_on_tritangents_orbits
	$(EXTRACT) SURFACE_q13_STAB



extract_5_6:
	$(EXTRACT) PGL_3_2_elements
	$(EXTRACT) Sym_3_elements
	$(EXTRACT) Cycle_12_power
	$(EXTRACT) PGL_3_4_singer
	$(EXTRACT) GL_2_8_multiply
	$(EXTRACT) GL_2_7_multiply
	$(EXTRACT) GL_2_7_inv
	$(EXTRACT) GL_2_7_power

extract_5_7:
	$(EXTRACT) PGGL_2_4_classes
	$(EXTRACT) PGGL_2_4_cent_2A
	$(EXTRACT) PGL_4_5_norm_31
	$(EXTRACT) Normalizer_of_Z22_in_PGL_2_9


extract_6_1:
	$(EXTRACT) T3r1_orbits
	$(EXTRACT) T3r1_orbits_draw
	$(EXTRACT) 2C_orbit_under_PGGL_4_4_elements_coded.csv
	$(EXTRACT) PGGL_4_4_subgroups_of_type_2C_2C
	$(EXTRACT) orbits_on_conics_q13
	$(EXTRACT) orbits_cubic_curves_q2
	$(EXTRACT) orbits_cubic_curves_q2_with_draw_tree
	$(EXTRACT) poly_orbits_d3_n3_q2.csv
	$(EXTRACT) poly_orbits_d3_n3_q2_get_ranks
	$(EXTRACT) T4_orbits
	$(EXTRACT) T4r1_orbits
	$(EXTRACT) T4r1_orbits_draw
	$(EXTRACT) T4r1_orbits_4
	$(EXTRACT) PGGL_2_8_on_conic_orbits
	$(EXTRACT) PGGL_7_8_orbits


extract_6_2:
	$(EXTRACT) poset_of_4subsets
	$(EXTRACT) poset_of_4subsets_draw
	$(EXTRACT) poset_of_5subsets
	$(EXTRACT) poset_of_5subsets_draw
	$(EXTRACT) Symmetric_4_on_pairs_poset
	$(EXTRACT) V_3_2_trivial
	$(EXTRACT) V_4_2_trivial


extract_6_3:
	$(EXTRACT) PG_2_2_subsets
	$(EXTRACT) PG_3_2_subsets
	$(EXTRACT) PGL_3_2_singer
	$(EXTRACT) PGL_3_2_on_lines
	$(EXTRACT) PGL_2_5_on_subsets
	$(EXTRACT) PGL_2_7_on_subsets
	$(EXTRACT) PGGL_2_8_on_subsets
	$(EXTRACT) PGGL_2_9_on_subsets
	$(EXTRACT) PGL_2_11_on_subsets
	$(EXTRACT) PGGL_2_16_on_subsets
	$(EXTRACT) PGGL_2_32_on_subsets
	$(EXTRACT) PG_3_4_subsets
	$(EXTRACT) PGGL_2_9_orbits
	$(EXTRACT) PGO_5_2_on_subsets


extract_6_4:
	$(EXTRACT) subspaces_Op_4_2
	$(EXTRACT) PGL_4_2_on_subspaces
	$(EXTRACT) PGL_4_2_singer_on_subspaces
	$(EXTRACT) PGL_8_2_singer_on_subspaces
	$(EXTRACT) Op_6_2_orbits_on_subspaces
	$(EXTRACT) Op_6_3_orbits_on_subspaces
	$(EXTRACT) Op_6_11_orbits_on_subspaces
	$(EXTRACT) Op_8_2_orbits_on_subspaces
	$(EXTRACT) PGO_7_2_on_subspaces


extract_6_5:
	$(EXTRACT) subspaces_Op_4_2
	$(EXTRACT) PGL_3_27
	$(EXTRACT) AGGL_2_27
	$(EXTRACT) hyperoval_4_classify
	$(EXTRACT) hyperoval_8_classify
	$(EXTRACT) frame_stabilizer_PGGL
	$(EXTRACT) frame_stabilizer_PGL
	$(EXTRACT) hyperoval_16_classify
	$(EXTRACT) hyperoval_16_1_conic_type
	$(EXTRACT) hyperoval_16_1_nonconical_type
	$(EXTRACT) hyperoval_16_2_nonconical_type
	$(EXTRACT) hyperoval_16_stab_0_disjoint_sets_graph
	$(EXTRACT) nc_arcs_16
	$(EXTRACT) nc_arcs_32_E13
	$(EXTRACT) F64_work
	$(EXTRACT) F64_frob
	$(EXTRACT) surface_64_0
	$(EXTRACT) nc_arcs_128
	$(EXTRACT) nc_arcs_256_E13
	$(EXTRACT) Example_F64
	$(EXTRACT) six_arcs_4_nbE13
	$(EXTRACT) six_arcs_8_nbE13
	$(EXTRACT) six_arcs_16_nbE13
	$(EXTRACT) six_arcs_32_nbE13
	$(EXTRACT) six_arcs_64_nbE13
	$(EXTRACT) six_arcs_128_nbE13
	$(EXTRACT) six_arcs_256_nbE13
	$(EXTRACT) five_arcs_q13


extract_6_6:
	$(EXTRACT) cubic_curves_PG_2_4
	$(EXTRACT) cubic_curves_PG_2_4_draw
	$(EXTRACT) cubic_curves_PG_2_8
	$(EXTRACT) cubic_curves_PG_2_8_draw



extract_7_1:
	$(EXTRACT) surface_4_0
	$(EXTRACT) surface_7_0
	$(EXTRACT) Family_general_F7
	$(EXTRACT) eckardt_13_4_12
	$(EXTRACT) surface_8_0_catalogue
	$(EXTRACT) surface_8_0_clean
	$(EXTRACT) surface_8_0b
	$(EXTRACT) Eckardt_13
	$(EXTRACT) surface_13_0
	$(EXTRACT) surface_16_0
	$(EXTRACT) G13_8
	$(EXTRACT) F13_8
	$(EXTRACT) F13_16
	$(EXTRACT) F13_32
	$(EXTRACT) F13_64a
	$(EXTRACT) F13_64b
	$(EXTRACT) Colorado1
	$(EXTRACT) Colorado2
	$(EXTRACT) Colorado3
	$(EXTRACT) F13_128a
	$(EXTRACT) F13_128b
	$(EXTRACT) F13_128c
	$(EXTRACT) move_two_lines
	$(EXTRACT) F_alpha_beta_gamma_delta
	$(EXTRACT) F_abcd_Eckardt_q31
	$(EXTRACT) surface_F_abcd
	$(EXTRACT) F_abcd_sweep_4_27_q7
	$(EXTRACT) F_alpha_beta_gamma_delta_q7_override_group
	$(EXTRACT) F_alpha_beta_gamma_delta_sweep_4_q3
	$(EXTRACT) F_alpha_beta_gamma_delta_sweep_4_15_lines_q7
	$(EXTRACT) F_alpha_beta_gamma_delta_q7_recognize
	$(EXTRACT) surf49_recognize
	$(EXTRACT) McKean_15lines_q7
	$(EXTRACT) F_4_4_3_3_q7
	$(EXTRACT) F_alpha_beta_gamma_delta_points.txt



extract_7_2:
	$(EXTRACT) surface_4_0
	$(EXTRACT) quartic_curve_9_0_report
	$(EXTRACT) quartic_curve_13_0_report
	$(EXTRACT) quartic_curve_13_1_report
	$(EXTRACT) surface_4_0_quartic_curves
	$(EXTRACT) quartic_curves_q7
	$(EXTRACT) quartic_curves_q7_classify
	$(EXTRACT) quartic_curves_q13
	$(EXTRACT) quartic_curves_q13_classify
	$(EXTRACT) quartic_curves_q17
	$(EXTRACT) quartic_curves_q17_classify
	$(EXTRACT) quartic_curves_q19
	$(EXTRACT) quartic_curves_q19_classify
	$(EXTRACT) quartic_curves_q19_set_stabilizer
	$(EXTRACT) surface_13_0_quartics
	$(EXTRACT) surface_13_1_quartics
	$(EXTRACT) quartic_curve_13_2_group
	$(EXTRACT) surface_25_0
	$(EXTRACT) quartic_curve_25_report
	$(EXTRACT) quartic_curve_13_table
	$(EXTRACT) quartic_curve_19_table
	$(EXTRACT) quartic_curve_19_table_latex
	$(EXTRACT) quartic_curve_25_table
	$(EXTRACT) quartic_curve_27_table
	$(EXTRACT) quartic_curve_29_table
	$(EXTRACT) quartic_curve_31_table
	$(EXTRACT) surface_25_12
	$(EXTRACT) surface_25_12_t1
	$(EXTRACT) surface_25_12_t2
	$(EXTRACT) surface_25_12_t3
	$(EXTRACT) surface_25_12_t4
	$(EXTRACT) surface_25_12_t5
	$(EXTRACT) PG_2_25
	$(EXTRACT) PG_2_25_lines
	$(EXTRACT) surface_25_12_t6
	$(EXTRACT) PG_2_25_stab_of_triangle
	$(EXTRACT) surface_25_12_t7
	$(EXTRACT) surface_25_12_t8
	$(EXTRACT) surface_25_12_t9
	$(EXTRACT) surface_25_12_t8_quartic_curves
	$(EXTRACT) quartic_curve_13_0_surface
	$(EXTRACT) quartic_curve_13_0_surface_create



extract_7_3:
	$(EXTRACT) surface_classify_q4
	$(EXTRACT) surface_classify_q4_arc_lifting_two_lines
	$(EXTRACT) surface_classify_q7
	$(EXTRACT) surface_classify_q9
	$(EXTRACT) surface_classify_q13


extract_7_4:
	$(EXTRACT) surface_recognize_q7_abcd_2_3_3_4
	$(EXTRACT) surface_isomorph_16
	$(EXTRACT) surface_recognize_8
	$(EXTRACT) surface_recognize_F13_q4
	$(EXTRACT) surface_sweep_Cayley_13
	$(EXTRACT) F_sweep_15_q7



extract_7_5:
	$(EXTRACT) D6_q2
	$(EXTRACT) D3_q4
	$(EXTRACT) D4_q8
	$(EXTRACT) D6_q4
	$(EXTRACT) D8_q4
	$(EXTRACT) D1_q8
	$(EXTRACT) D1_q4_with_select_double_six
	$(EXTRACT) D1_q4_with_select_double_six_b
	$(EXTRACT) D1_q4_trans
	$(EXTRACT) D1_q4_with_select_double_six_c
	$(EXTRACT) orbits_cubic_surfaces_q3
	$(EXTRACT) orbits_cubic_curves_q2_again
	$(EXTRACT) orbits_cubic_curves_q3
	$(EXTRACT) poly_orbits_d3_n3_q2_F2.csv
	$(EXTRACT) Dickson_q2_analyze
	$(EXTRACT) poly_orbits_d3_n3_q2_F4.csv
	$(EXTRACT) Dickson_q4_analyze
	$(EXTRACT) poly_orbits_d3_n3_q2_F8.csv
	$(EXTRACT) Dickson_q8_analyze
	$(EXTRACT) poly_orbits_d3_n3_q2_F16.csv
	$(EXTRACT) Dickson_q16_analyze


extract_7_6:
	$(EXTRACT) MAKE_TABLE_OF_CUBIC_SURFACES
	$(EXTRACT) cubic_surfaces_tables_17
	$(EXTRACT) cubic_surfaces_table_latex_17
	$(EXTRACT) cubic_surfaces_tables_up_to_17
	$(EXTRACT) cubic_surfaces_tables_19_37
	$(EXTRACT) cubic_surfaces_tables_41_and_up
	$(EXTRACT) cubic_surfaces_tables_latex
	$(EXTRACT) cubic_surfaces_tables_latex_big
	$(EXTRACT) surface_table
	$(EXTRACT) surface_atlas
	$(EXTRACT) surface_reports
	$(EXTRACT) quartic_curve_tables_23
	$(EXTRACT) quartic_curve_tables
	$(EXTRACT) quartic_curve_tables_latex


extract_8_1:
	$(EXTRACT) sift_polynomials_deg3_q2
	$(EXTRACT) sift_polynomials_deg4_q2
	$(EXTRACT) poly_division
	$(EXTRACT) poly_division2
	$(EXTRACT) poly_gcd
	$(EXTRACT) poly_mult_mod1
	$(EXTRACT) poly_mult_mod2
	$(EXTRACT) poly_mult_mod_F4
	$(EXTRACT) mult_polynomials_2_5_7
	$(EXTRACT) polynomial_division_ranked_2_27_13
	$(EXTRACT) mult_polynomials_2_8_15
	$(EXTRACT) polynomial_division_ranked_2_120_25
	$(EXTRACT) mult_polynomials_2_7_7
	$(EXTRACT) mult_polynomials_2_4_6
	$(EXTRACT) polynomial_division_ranked_2_24_13
	$(EXTRACT) mult_polynomials_1024_999_997
	$(EXTRACT) polynomial_division_ranked_2_349147_1033
	$(EXTRACT) mult_polynomials_1024_999_997_check
	$(EXTRACT) mult_polynomials_17_12
	$(EXTRACT) polynomial_division_ranked_2_204_37
	$(EXTRACT) CRC32_SPARSE
	$(EXTRACT) test_crc32
	$(EXTRACT) Berlekamp_matrix_crc32
	$(EXTRACT) TWO_TO_THE_32_MINUS_2
	$(EXTRACT) power_mod_inverse
	$(EXTRACT) INVERSE_SPARSE
	$(EXTRACT) mult_mod_to_get_one
	$(EXTRACT) Berlekamp_matrix_2_3
	$(EXTRACT) Berlekamp_matrix_2_4
	$(EXTRACT) Berlekamp_matrix_4_3a
	$(EXTRACT) Berlekamp_matrix_4_3b
	$(EXTRACT) find_roots_a
	$(EXTRACT) find_roots_b
	$(EXTRACT) find_roots_c
	$(EXTRACT) find_roots_d
	$(EXTRACT) find_roots_e
	$(EXTRACT) roots_over_F2
	$(EXTRACT) roots_over_F8
	$(EXTRACT) irred_3_2
	$(EXTRACT) irred_4_2
	$(EXTRACT) irred_5_2
	$(EXTRACT) irred_6_2
	$(EXTRACT) irred_7_2
	$(EXTRACT) irred_8_2
	$(EXTRACT) irred_9_2
	$(EXTRACT) irred_10_2
	$(EXTRACT) irred_2_4
	$(EXTRACT) irred_3_4
	$(EXTRACT) search_primitive_poly_2
	$(EXTRACT) search_primitive_poly_3
	$(EXTRACT) search_primitive_poly_4
	$(EXTRACT) search_primitive_poly_5
	$(EXTRACT) search_primitive_poly_7
	$(EXTRACT) search_primitive_poly_8
	$(EXTRACT) search_primitive_poly_9
	$(EXTRACT) search_primitive_poly_11
	$(EXTRACT) search_primitive_poly_13
	$(EXTRACT) search_primitive_poly_degree_16
	$(EXTRACT) search_primitive_poly_32


extract_8_2:
	$(EXTRACT) arcs_5_2_q11
	$(EXTRACT) arcs_5_2_q11_ideal
	$(EXTRACT) PTS_OF_SURFACE_ORBIT211_Q3_L9_E4
	$(EXTRACT) surface_9lines_4E_ideal
	$(EXTRACT) SURFACE_F_9
	$(EXTRACT) F_9_q7
	$(EXTRACT) random_k_subsets
	$(EXTRACT) line_type_in_PG_2_11
	$(EXTRACT) random_arc_5_2_q11_ideal
	

extract_9_1:
	$(EXTRACT) inverse_mod_a
	$(EXTRACT) jacobi_35_41
	$(EXTRACT) jacobi_33_41
	$(EXTRACT) jacobi_a
	$(EXTRACT) jacobi_5_19
	$(EXTRACT) sqrt_mod_7817

extract_9_2:
	$(EXTRACT) representation_on_polynomials_of_degree_3
	$(EXTRACT) representation_tetrahedral_group_on_polynomials_of_degree_3


extract_9_3:
	$(EXTRACT) EC_add
	$(EXTRACT) EC_cyclic_subgroup
	$(EXTRACT) EC_points_13
	$(EXTRACT) EC_points_199
	$(EXTRACT) EC_Koblitz_encoding
	$(EXTRACT) EC_bsgs
	$(EXTRACT) EC_bsgs_decode
	$(EXTRACT) NTRU_N
	$(EXTRACT) NTRU_P
	$(EXTRACT) NTRU_Q
	$(EXTRACT) NTRU_D
	$(EXTRACT) NTRU_XN1
	$(EXTRACT) ALICE_PRIVATE_F
	$(EXTRACT) ALICE_PRIVATE_G
	$(EXTRACT) NTRU_Alice1
	$(EXTRACT) ALICE_PRIVATE_FQ
	$(EXTRACT) NTRU_Alice2
	$(EXTRACT) ALICE_PRIVATE_FP
	$(EXTRACT) NTRU_Alice_public_key
	$(EXTRACT) ALICE_PUBLIC_KEY
	$(EXTRACT) BOB_MESSAGE
	$(EXTRACT) BOB_ONE_TIME_KEY
	$(EXTRACT) NTRU_encrypt
	$(EXTRACT) BOB_ENCRYPT
	$(EXTRACT) NTRU_decrypt1
	$(EXTRACT) ALICE_C1
	$(EXTRACT) NTRU_decrypt2
	$(EXTRACT) ALICE_C2
	$(EXTRACT) NTRU_decrypt3
	$(EXTRACT) ALICE_C3
	$(EXTRACT) NTRU_decrypt4
	$(EXTRACT) ALICE_C4
	$(EXTRACT) NTRU_decrypt5
	$(EXTRACT) inv_59_mod
	$(EXTRACT) RSA_e
	$(EXTRACT) RSA_d
	$(EXTRACT) im1
	$(EXTRACT) RSA_e1
	$(EXTRACT) RSA_d1
	$(EXTRACT) im1061
	$(EXTRACT) RSA_e2
	$(EXTRACT) RSA_d2
	$(EXTRACT) im3
	$(EXTRACT) RSA_e3
	$(EXTRACT) RSA_d3
	$(EXTRACT) im4
	$(EXTRACT) RSA_e4
	$(EXTRACT) RSA_d4
	$(EXTRACT) im5
	$(EXTRACT) RSA_e5
	$(EXTRACT) RSA_d5
	$(EXTRACT) RSA_d6
	$(EXTRACT) smooth
	$(EXTRACT) im7
	$(EXTRACT) RSA_e7
	$(EXTRACT) im8
	$(EXTRACT) RSA_e8
	$(EXTRACT) sqrt_big
	$(EXTRACT) sqrt_mod_33_41
	$(EXTRACT) quadratic_sieve
	$(EXTRACT) pseudoprime3
	$(EXTRACT) pseudoprime10
	$(EXTRACT) PR10_test1
	$(EXTRACT) pseudoprime11
	$(EXTRACT) pseudoprime20
	$(EXTRACT) PR10
	$(EXTRACT) pseudoprime50
	$(EXTRACT) pseudoprime51
	$(EXTRACT) pseudoprime30
	$(EXTRACT) pseudoprime31
	$(EXTRACT) pseudoprime33
	$(EXTRACT) pseudoprime34
	$(EXTRACT) pseudoprime35
	$(EXTRACT) pseudoprime36
	$(EXTRACT) MATH360_hw2
	$(EXTRACT) F_256_Rijndahl
	$(EXTRACT) all_square_roots_mod_n_1549411
	$(EXTRACT) power_mod_211
	$(EXTRACT) power_mod_2_31
	$(EXTRACT) power_mod_3_31

extract_10_1:
	$(EXTRACT) Hamming_space_4_2_distance_matrix
	$(EXTRACT) code_5_2_3_diagram
	$(EXTRACT) Hamming_5_2_graph
	$(EXTRACT) Hamming_5_2_with_5_2_3_code
	$(EXTRACT) code_6
	$(EXTRACT) CODE_5_2_3_CODEWORDS


extract_10_2:
	$(EXTRACT) Hamming_generator
	$(EXTRACT) Hamming_code_words
	$(EXTRACT) Hamming_weight_enumerator
	$(EXTRACT) Hamming_code_diagram
	$(EXTRACT) code_Hamming_systematic
	$(EXTRACT) Hamming_RREF
	$(EXTRACT) Hamming_nullspace
	$(EXTRACT) Hamming_long
	$(EXTRACT) Hamming_code_macwilliams
	$(EXTRACT) Hamming_singer
	$(EXTRACT) Hamming_cyclic_generator
	$(EXTRACT) Hamming_cyclic_long
	$(EXTRACT) Hamming_cyclic
	$(EXTRACT) Hamming_cyclic_clean_ns
	$(EXTRACT) Hamming_cyclic_clean
	$(EXTRACT) Hamming_cyclic_clean_long
	$(EXTRACT) SIMPLEX_CODE_GENERATOR
	$(EXTRACT) HAMMING_CODE_GENERATOR
	$(EXTRACT) SIMPLEX_CODE_GENMA_CYCLIC


extract_10_3:
	$(EXTRACT) Golay23_code_words
	$(EXTRACT) Golay23_code_diagram
	$(EXTRACT) Golay23_code_diagram_draw
	$(EXTRACT) GOLAY_23_COLUMN_RANKS_PROJECTIVELY

extract_10_4:
	$(EXTRACT) CRC4
	$(EXTRACT) CRC7
	$(EXTRACT) CRC8_ATM
	$(EXTRACT) CRC16_CCITT
	$(EXTRACT) CRC32_ETHERNET
	$(EXTRACT) CRC32_CASTAGNOLI
	$(EXTRACT) CRC64_ECMA182
	$(EXTRACT) CRC64_ROCKSOFT
	$(EXTRACT) encode_text_5bits
	$(EXTRACT) encode_text_5bits_check
	$(EXTRACT) encode_text_5bits_1error
	$(EXTRACT) CRC_3_128_10
	$(EXTRACT) crc32_test
	$(EXTRACT) crc32_test_hexdata
	$(EXTRACT) crc32_Berlekamp_matrix
	$(EXTRACT) F256_roots_771
	$(EXTRACT) F256_BCH_code_d2
	$(EXTRACT) POLY_Q256_DEG2_DENSE
	$(EXTRACT) F256_BCH_write_code_for_division_d2
	$(EXTRACT) F256_BCH_code_d16
	$(EXTRACT) POLY_Q256_DEG30_SPARSE
	$(EXTRACT) POLY_Q256_DEG30_DENSE
	$(EXTRACT) F256_BCH_write_code_for_division_d16




extract_10_5:
	$(EXTRACT) RM_3_1_code_words
	$(EXTRACT) RM_3_1_Hamming_space_diagram
	$(EXTRACT) RM_3_1_split
	$(EXTRACT) RM_3_1_holes_draw
	$(EXTRACT) RM_3_1_hole0
	$(EXTRACT) RM_3_1_hole1
	$(EXTRACT) RM_3_1_hole2
	$(EXTRACT) RM_4_1
	$(EXTRACT) RM_4_1_diagram
	$(EXTRACT) RM_4_1_diagram_draw
	$(EXTRACT) RM_4_1_split
	$(EXTRACT) RM_4_1_diagram_draw_holes
	$(EXTRACT) RM_4_1_diagram_metric_balls
	$(EXTRACT) RM_4_1_hole0
	$(EXTRACT) Reed_Muller_6
	$(EXTRACT) RM_6
	$(EXTRACT) RM6words
	$(EXTRACT) RM6_convert
	$(EXTRACT) RM_6_poster
	$(EXTRACT) REED_MULLER_3_1_BASIS_IN_BINARY

extract_10_6:
	$(EXTRACT) draw_cyclotomic_mod_21_q8
	$(EXTRACT) F_8_BCH_code_d3
	$(EXTRACT) F_8_BCH_code_d4
	$(EXTRACT) F_8_BCH_code_d5
	$(EXTRACT) F_8_BCH_code_d6
	$(EXTRACT) F_8_BCH_code_d7
	$(EXTRACT) F_8_BCH_code_d8
	$(EXTRACT) F2_BCH_code_n21
	$(EXTRACT) F7_RS_code_n6
	$(EXTRACT) F_64_again
	$(EXTRACT) BCH_255_5_evaluate_elementary_symmetric_functions_1
	$(EXTRACT) BCH_255_5_evaluate_elementary_symmetric_functions_2
	$(EXTRACT) BCH15
	$(EXTRACT) BCH11
	$(EXTRACT) BCH13
	$(EXTRACT) BCH7
	$(EXTRACT) BCH21
	$(EXTRACT) BCH93
	$(EXTRACT) BCH255
	$(EXTRACT) BCH273
	$(EXTRACT) draw_mod_31
	$(EXTRACT) PR127
	$(EXTRACT) draw_mod_127_power
	$(EXTRACT) draw_mod_251
	$(EXTRACT) draw_mod_255_cyclotomic_1
	$(EXTRACT) draw_mod_255_cyclotomic_3
	$(EXTRACT) draw_mod_255_cyclotomic_1_and_3
	$(EXTRACT) draw_mod_63_4_cyclotomic_3_6
	$(EXTRACT) BCH_F_64
	$(EXTRACT) BCH_elementary_symmetric_functions_3
	$(EXTRACT) BCH_63_4_evaluate_elementary_symmetric_functions_1
	$(EXTRACT) BCH_63_4_evaluate_elementary_symmetric_functions_2
	$(EXTRACT) BCH_21_poly_mult_mod_F4
	$(EXTRACT) BCH_21_poly_division_a
	$(EXTRACT) BCH_21_poly_division_b
	$(EXTRACT) BCH_21_poly_division_ab
	$(EXTRACT) BCH_21_generator_matrix
	$(EXTRACT) BCH_21_15_weight_enumerator
	$(EXTRACT) BCH_21_15_dual
	$(EXTRACT) BCH_21_6_weight_enumerator
	$(EXTRACT) BCH_21_6_4_macwilliams
	$(EXTRACT) BCH_21_15_4_field_reduction
	$(EXTRACT) BCH_21_poly_division_c
	$(EXTRACT) F16_roots_5
	$(EXTRACT) F64_roots_21
	$(EXTRACT) F256_roots_771
	$(EXTRACT) F256_BCH_code_d16
	$(EXTRACT) F256_BCH_code_d16_division
	$(EXTRACT) F256_BCH_write_code_for_division_d16
	$(EXTRACT) F256_BCH_code_d16_error
	$(EXTRACT) POLY_Q256_DEG30_DENSE

extract_10_7:
	$(EXTRACT) F_7_BCH_code_n6
	$(EXTRACT) RREF_RS_6_4_7_weight_enumerator
	$(EXTRACT) Code_RS_11
	$(EXTRACT) Code_RS_11_weight_enumerator
	$(EXTRACT) RREF_RS_8_weight_enumerator
	$(EXTRACT) RS_8_field_reduction
	$(EXTRACT) RREF_RS_8_reduced_weight_enumerator
	$(EXTRACT) CODE_21_15_4_store
	$(EXTRACT) CODE_21_15_4_weight_enumerator
	$(EXTRACT) Reed_solomon_F8_work
	$(EXTRACT) CODE_21_15_4
	$(EXTRACT) CODE_RS_8
	$(EXTRACT) RS_8_reduced
	$(EXTRACT) CODE_RS_6_4_7


extract_10_8:
	$(EXTRACT) bounds_for_d_given_n6_k4_q7
	$(EXTRACT) bounds_for_d_given_n15_k6_q2
	$(EXTRACT) coding_theory_bounds_q2
	$(EXTRACT) coding_theory_bounds_q8
	$(EXTRACT) GV_n15_k6_d5
	$(EXTRACT) bounds_for_d_given_n12_k4_q13
	$(EXTRACT) GV_n15_k6_d5_weight_enumerator
	$(EXTRACT) code_n15_k6_d6_a_we
	$(EXTRACT) code_n15_k6_d6_RREF
	$(EXTRACT) code_n15_k6_d6_check_RREF
	$(EXTRACT) CODE_GV_N15_K6



extract_10_9:
	$(EXTRACT) codes_8_4_4
	$(EXTRACT) codes_8_4_4_draw
	$(EXTRACT) codes_14_4_9_3
	$(EXTRACT) codes_15_6_6_2
	$(EXTRACT) codes_16_5_9_3
	$(EXTRACT) codes_d4
	$(EXTRACT) codes_24_12_8
	$(EXTRACT) codes_24_12_8_draw
	$(EXTRACT) glynn_arc
	$(EXTRACT) five_points_in_general
	$(EXTRACT) codes_q13_12_4



extract_11_1:
	$(EXTRACT) Sym_4_conj_classes
	$(EXTRACT) Sym_10_conj_classes
	$(EXTRACT) Sym_15_conj_classes
	$(EXTRACT) Char_Sym_4
	$(EXTRACT) Char_Sym_5
	$(EXTRACT) Char_Sym_6
	$(EXTRACT) all_subsets_10_3
	$(EXTRACT) rank_k_subsets_test
	$(EXTRACT) Walsh_matrix_4
	$(EXTRACT) Dedekind_10_10
	$(EXTRACT) Dedekind_30_2
	$(EXTRACT) Dedekind_100_2
	$(EXTRACT) elementary_symmetric_functions_4
	$(EXTRACT) elementary_symmetric_functions_8
	$(EXTRACT) GENERATORS_H5
	$(EXTRACT) GENERATORS_C13
	$(EXTRACT) LARGE_SET_S0
	$(EXTRACT) LARGE_SET_S1
	$(EXTRACT) LARGE_SET_S2
	$(EXTRACT) LARGE_SET_S3
	$(EXTRACT) LARGE_SET_S4
	$(EXTRACT) LARGE_SET_S5
	$(EXTRACT) LARGE_SET_S6
	$(EXTRACT) LARGE_SET_S7
	$(EXTRACT) LARGE_SET_S8
	$(EXTRACT) LARGE_SET_S9
	$(EXTRACT) LARGE_SET_S10
	$(EXTRACT) file_S
	$(EXTRACT) Large_set_H5
	$(EXTRACT) Large_set_C13
	$(EXTRACT) GENERATORS_Perm13_Subgroup_C13_13
	$(EXTRACT) Perm13_Subgroup_C13_13
	$(EXTRACT) Large_set_mult_C13xS
	$(EXTRACT) Large_set_mult_C13xSxH5
	$(EXTRACT) Large_set_mult_C13xSxH5_apply
	$(EXTRACT) domino_portrait
	$(EXTRACT) domino_portrait_input



extract_11_2:
	$(EXTRACT) part10
	$(EXTRACT) octic_monomials
	$(EXTRACT) solve_test_system
	$(EXTRACT) McKay_test
	$(EXTRACT) DLX_test
	$(EXTRACT) TEST_SYSTEM



extract_11_3:
	$(EXTRACT) linsp6
	$(EXTRACT) linsp7
	$(EXTRACT) linsp30_pt_types
	$(EXTRACT) linsp30_pt_distribution


extract_11_4:
	$(EXTRACT) geo_pasch
	$(EXTRACT) geo_petersen
	$(EXTRACT) geo_7_3
	$(EXTRACT) geo_7_3_no_square_test
	$(EXTRACT) geo_7_3_no_square_test_draw
	$(EXTRACT) geo_7_3_orderly
	$(EXTRACT) geo_7_3_orderly_draw
	$(EXTRACT) geo_7_3_orderly_mem_debug
	$(EXTRACT) geo_8_3
	$(EXTRACT) geo_9_3
	$(EXTRACT) geo_10_3
	$(EXTRACT) geo_10_3_inc_draw
	$(EXTRACT) geo_10_3_orderly
	$(EXTRACT) geo_10_3_orderly_mem_debug
	$(EXTRACT) geo_10_3_tree
	$(EXTRACT) geo_10_3_tree_path
	$(EXTRACT) Desargues_path_lex_least_draw
	$(EXTRACT) DESARGUES_PATH_CANONICAL_ANCESTOR
	$(EXTRACT) Desargues_path_can_anc_draw
	$(EXTRACT) geo_11_3
	$(EXTRACT) geo_12_3
	$(EXTRACT) geo_12_3_orderly
	$(EXTRACT) geo_13_3
	$(EXTRACT) geo_13_3_orderly
	$(EXTRACT) geo_14_3
	$(EXTRACT) geo_14_3_orderly
	$(EXTRACT) 15_3.inc
	$(EXTRACT) geo_15_3_g4
	$(EXTRACT) geo_17_3_g4_orderly
	$(EXTRACT) geo_18_3_g4
	$(EXTRACT) geo_19_3_g4
	$(EXTRACT) geo_20_3_g4
	$(EXTRACT) geo_21_3_g4
	$(EXTRACT) geo_15_4
	$(EXTRACT) geo_16_4_g4
	$(EXTRACT) 40_4_g4.inc
	$(EXTRACT) geo_63_3_g6
	$(EXTRACT) geo_LSQ6
	$(EXTRACT) geo_16



extract_11_5:
	$(EXTRACT) design_PG_2_3
	$(EXTRACT) design_PG_2_4
	$(EXTRACT) GENERATORS_H5
	$(EXTRACT) GENERATORS_N5
	$(EXTRACT) design_PG_2_3_table_create
	$(EXTRACT) design_PG_2_3_group_5
	$(EXTRACT) design_PG_2_3_group_5_sol_0
	$(EXTRACT) KM_cyclic_7
	$(EXTRACT) KM_PGGL_2_32
	$(EXTRACT) KM_PSL_3_5
	$(EXTRACT) PLANE_1
	$(EXTRACT) PLANE_2
	$(EXTRACT) PLANE_3
	$(EXTRACT) PLANE_4
	$(EXTRACT) design_large_set_rank_k_subsets_a
	$(EXTRACT) design_large_set_rank_k_subsets_b
	$(EXTRACT) design_large_set_rank_k_subsets_c
	$(EXTRACT) design_large_set_rank_k_subsets_d



extract_11_6:
	$(EXTRACT) AG_2_3.inc
	$(EXTRACT) LS_AG_2_3_design_table_create
	$(EXTRACT) AG_2_3_on_designs
	$(EXTRACT) AG_2_3_stab_orb_0
	$(EXTRACT) AG_2_3_stab_orb_0_Perm840_res192
	$(EXTRACT) LS_AG_2_3_disjoint_sets_graph_and_cliques
	$(EXTRACT) LS_AG_2_3_disjoint_sets_split
	$(EXTRACT) LS_AG_2_3_export_solutions
	$(EXTRACT) AG_2_3_BLOCKS


extract_11_7:
	$(EXTRACT) DD_PP4
	$(EXTRACT) DD_PP4_system
	$(EXTRACT) DD_CC
	$(EXTRACT) DD_CC_system
	$(EXTRACT) DD_M1_G1
	$(EXTRACT) DD_M1_G1_S
	$(EXTRACT) DD_PG_2_4_M1_G1
	$(EXTRACT) PG_2_27_special
	$(EXTRACT) PP4
	$(EXTRACT) PP4_GROUP1
	$(EXTRACT) PP4_MASK1


extract_11_8:
	$(EXTRACT) max_arc_16_4_start
	$(EXTRACT) max_arc_16_4_convert_stack_tdo
	$(EXTRACT) max_arc_16_4_refine
	$(EXTRACT) max_arc_16_4r_print


extract_12_1:
	$(EXTRACT) desarguesian_spread_in_PG_3_2
	$(EXTRACT) desarguesian_spread_in_PG_5_2
	$(EXTRACT) desarguesian_spread_in_PG_3_4
	$(EXTRACT) desarguesian_spread_in_PG_3_5
	$(EXTRACT) spreads4
	$(EXTRACT) spreads16_4


extract_12_2:
	$(EXTRACT) TP_9_0
	$(EXTRACT) TP_9_1
	$(EXTRACT) TP_16_4
	$(EXTRACT) TP_16_2


extract_12_3:
	$(EXTRACT) spread_table_PG_3_4
	$(EXTRACT) spread_table_PG_3_5_regular
	$(EXTRACT) PG_3_5_desarguesian_spread
	$(EXTRACT) PG_3_5_element_of_order_31
	$(EXTRACT) PG_3_5_element_of_order_31_normalizer
	$(EXTRACT) PG_3_5_element_of_order_31_GL_normalizer
	$(EXTRACT) PG_3_5_element_of_order_31_ME_normalizer
	$(EXTRACT) PG_3_5_assume_31_graph
	$(EXTRACT) PG_3_5_assume_31_fpc0_lo_cliques
	$(EXTRACT) PG_3_5_assume_31_read_again
	$(EXTRACT) PG_3_5_packing_0_dualize
	$(EXTRACT) PG_3_5_assume_3
	$(EXTRACT) PG_3_5_3B_create_graph_on_long_orbits
	$(EXTRACT) PG_3_5_assume_3B_fpc0_lo_cliques
	$(EXTRACT) PG_3_5_assume_3B_long_read
	$(EXTRACT) PG_3_5_packing0_print
	$(EXTRACT) ORBIT_OF_LENGTH_1
	$(EXTRACT) PG_3_5_H3_orbit_of_length_1
	$(EXTRACT) H31_ORBIT_OF_SPREAD_0
	$(EXTRACT) PG_3_5_packings_compare
	$(EXTRACT) PG_3_5_assume_31_classify
	$(EXTRACT) PGL_4_5_SUBGROUP_31_ME
	$(EXTRACT) PGL_4_5_SUBGROUP_31_ME_NORMALIZER

extract_12_4:
	$(EXTRACT) BLT_5_1
	$(EXTRACT) BLT_5_Linear
	$(EXTRACT) BLT_9_K1
	$(EXTRACT) BLT_11_0
	$(EXTRACT) BLT_11_Fisher
	$(EXTRACT) BLT_11_Mondello
	$(EXTRACT) BLT_13_FTWKB
	$(EXTRACT) BLT_13_K2
	$(EXTRACT) BLT_13_starter_5
	$(EXTRACT) BLT_13_deep_14
	$(EXTRACT) BLT_13_graphs
	$(EXTRACT) BLT_13_cliques


extract_13_1:
	$(EXTRACT) Cycle_13
	$(EXTRACT) triangle_graph
	$(EXTRACT) Chain_232
	$(EXTRACT) Paley_13_graph
	$(EXTRACT) trihedral_pair_graph
	$(EXTRACT) small_graph
	$(EXTRACT) petersen
	$(EXTRACT) Johnson_6_2_0
	$(EXTRACT) Hamming_graph_3
	$(EXTRACT) Hamming_graph_7
	$(EXTRACT) HJ_graph
	$(EXTRACT) HJ315_orbital_graph_3
	$(EXTRACT) HJ_d2_graph
	$(EXTRACT) Cayley_Z11_1mod3
	$(EXTRACT) Cayley_Sym4_coxeter
	$(EXTRACT) Cayley_Sym4_star
	$(EXTRACT) TRIANGLE_GRAPH



extract_13_2:
	$(EXTRACT) triangle_graph_properties
	$(EXTRACT) Cycle_13_draw
	$(EXTRACT) Cycle_9_eigenvalues
	$(EXTRACT) Paley_13_draw
	$(EXTRACT) Paley_13_eigenvalues
	$(EXTRACT) Cayley_Z11_1mod3_draw
	$(EXTRACT) Cayley_Sym4_coxeter_eigenvalues_and_draw
	$(EXTRACT) Cayley_Sym5_coxeter_draw
	$(EXTRACT) Cayley_Sym4_star_eigenvalues_and_draw
	$(EXTRACT) small_graph_draw
	$(EXTRACT) petersen_draw
	$(EXTRACT) Johnson_6_2_0_draw
	$(EXTRACT) Hamming_graph_3_draw
	$(EXTRACT) Hamming_graph_7_draw
	$(EXTRACT) Chain_232_properties
	$(EXTRACT) Chain_232_eigen
	$(EXTRACT) HJ_properties
	$(EXTRACT) HJ_d2_properties
	$(EXTRACT) PGO_5_2_collinearity_graph
	$(EXTRACT) trihedral_pair_graph_draw



extract_13_3:
	$(EXTRACT) graph_classify_4
	$(EXTRACT) tournament_classify_4
	$(EXTRACT) graph_classify_8_r3
	$(EXTRACT) Symmetric_4_inversion_graph_recognize
	$(EXTRACT) Symmetric_5_inversion_graph_recognize


extract_13_4:
	$(EXTRACT) small_graph_cliques
	$(EXTRACT) small_graph_cliques_Sajeeb
	$(EXTRACT) Paley_13_aut
	$(EXTRACT) Paley_13
	$(EXTRACT) Paley_13_cliques_classify
	$(EXTRACT) Paley_13_cliques
	$(EXTRACT) PGO_5_2_cliques
	$(EXTRACT) HJ_d2_c5
	$(EXTRACT) HJ64_cliques5
	$(EXTRACT) HJ64_cliques5_classify


extract_14_1:
	$(EXTRACT) Hirschfeld_q4_from_set
	$(EXTRACT) hyperoval_16_create
	$(EXTRACT) EC_read
	$(EXTRACT) PG_3_5_assume_31_read
	$(EXTRACT) LS_AG_2_3_read
	$(EXTRACT) geo_7_3_read
	$(EXTRACT) Desargues_path_lex_least_read
	$(EXTRACT) HIRSCHFELD_SURFACE_Q4_SET_OF_POINTS
	$(EXTRACT) HYPEROVAL_16_144
	$(EXTRACT) HYPEROVAL_16_16320


extract_14_2:
	$(EXTRACT) geo_pasch_read
	$(EXTRACT) geo_pasch_given


extract_15_1:

extract_15_2:
	$(EXTRACT) EC_canon
	$(EXTRACT) Hirschfeld_q4_c
	$(EXTRACT) Hirschfeld_q4_set_c
	$(EXTRACT) Dickson_sets_stabilizer
	$(EXTRACT) Endrass_7c
	$(EXTRACT) hyperoval_16_canonical_form
	$(EXTRACT) cubic_curves_PG_2_8_canon
	$(EXTRACT) F_alpha_beta_gamma_delta_classify_q7_nauty
	$(EXTRACT) ovoid_q8_canon
	$(EXTRACT) ovoid_ST_q8_canon
	$(EXTRACT) SUZUKI_8_GENERATORS
	$(EXTRACT) Suzuki_8
	$(EXTRACT) HIRSCHFELD_SURFACE_Q4_SET_OF_POINTS


extract_15_3:
	$(EXTRACT) geo_7_3_c
	$(EXTRACT) geo_10_3_c
	$(EXTRACT) geo_10_3_c_lex_least
	$(EXTRACT) geo_14_3_c
	$(EXTRACT) geo_15_3_c
	$(EXTRACT) FILE_24_3_TFC_INC
	$(EXTRACT) TFC_24_3_c
	$(EXTRACT) geo_40_4_g4_c
	$(EXTRACT) geo_17_3_g4_c
	$(EXTRACT) AG_2_3_c
	$(EXTRACT) geo_LSQ6_c
	$(EXTRACT) geo_16_c


extract_15_4:
	$(EXTRACT) LS_AG_2_3_solutions_classify
	$(EXTRACT) design_27c


extract_15_5:
	$(EXTRACT) code_3_2_aut
	$(EXTRACT) code_6_3_aut
	$(EXTRACT) RM_3_1_group
	$(EXTRACT) RM_3_1_group_and_diagram
	$(EXTRACT) RM_4_1_group
	$(EXTRACT) RS_6_4_7_group
	$(EXTRACT) GV_n15_k6_d5_group
	$(EXTRACT) code_n15_k6_d6_a_group
	$(EXTRACT) code_n15_k6_d6_b_group
	$(EXTRACT) CODE_RM_3_1_GENMA
	$(EXTRACT) REED_MULLER_3_1_CODEWORDS


extract_15_6:
	$(EXTRACT) Hamming_graph_7_with_Hamming_code
	$(EXTRACT) HAMMING_CODE_CODEWORDS


extract_15_7:
	$(EXTRACT) Cycle_13_aut
	$(EXTRACT) inversion_graph
	$(EXTRACT) Chain_232_aut
	$(EXTRACT) JK_graph_pp16_1
	$(EXTRACT) JK_graph_pp16_2
	$(EXTRACT) JK_graph_pp16_9
	$(EXTRACT) JK_graph_grid_3_3
	$(EXTRACT) JK_graph_sts_13
	$(EXTRACT) HJ_aut
	$(EXTRACT) HJ_group_and_orbits
	$(EXTRACT) HJ_orbital_graph_3
	$(EXTRACT) PGO_5_2_graph_group



extract_15_8:
	$(EXTRACT) F_17_edge
	$(EXTRACT) Edge_curve_17_nauty
	$(EXTRACT) Edge_curve_17_group


extract_16_1:
	$(EXTRACT) F_7_tables
	$(EXTRACT) PG_2_4_cyclic_incma
	$(EXTRACT) PGL_4_2_Wedge_4_0_graphical_output
	$(EXTRACT) schreier_tree_graphical_output
	$(EXTRACT) Queens_graph

extract_16_2:
	$(EXTRACT) cube
	$(EXTRACT) math261_test
	$(EXTRACT) plane1
	$(EXTRACT) plane2
	$(EXTRACT) analytic_geo_1
	$(EXTRACT) analytic_geo_1_video
	$(EXTRACT) monkey
	$(EXTRACT) Eckardt
	$(EXTRACT) Eckardt_deform
	$(EXTRACT) Eckardt_deform_2
	$(EXTRACT) Clebsch
	$(EXTRACT) endrass8
	$(EXTRACT) MONKEY_SADDLE_CUBIC
	$(EXTRACT) ENDRASS_OCTIC_LEX_165

extract_16_3:
	$(EXTRACT) dode
	$(EXTRACT) dode_video
	$(EXTRACT) monkey_video
	$(EXTRACT) Eckardt_deform_video
	$(EXTRACT) Eckardt_surface
	$(EXTRACT) Kummer_surface
	$(EXTRACT) Kummer_video
	$(EXTRACT) Beauville_surface
	$(EXTRACT) CLEBSCH_A
	$(EXTRACT) CLEBSCH_B
	$(EXTRACT) CLEBSCH_C
	$(EXTRACT) CLEBSCH_D
	$(EXTRACT) TWO_PI
	$(EXTRACT) T00
	$(EXTRACT) T01
	$(EXTRACT) T02
	$(EXTRACT) T03
	$(EXTRACT) T10
	$(EXTRACT) T11
	$(EXTRACT) T12
	$(EXTRACT) T13
	$(EXTRACT) T20
	$(EXTRACT) T21
	$(EXTRACT) T22
	$(EXTRACT) T23
	$(EXTRACT) T30
	$(EXTRACT) T31
	$(EXTRACT) T32
	$(EXTRACT) T33
	$(EXTRACT) CLEBSCH_CUBICS
	$(EXTRACT) Clebsch_up_create_points
	$(EXTRACT) Clebsch_surface
	$(EXTRACT) Clebsch_surface_defining_equation
	$(EXTRACT) Clebsch_surface_defining_equation_and_curves
	$(EXTRACT) F7_povray
	$(EXTRACT) F7_video
	$(EXTRACT) McKean_povray
	$(EXTRACT) McKean_video


extract_16_4:
	$(EXTRACT) lissajous
	$(EXTRACT) lissajous_plot
	$(EXTRACT) lissajous_3d
	$(EXTRACT) lissajous_3d_plot


extract_17_1:
	$(EXTRACT) misc_select
	$(EXTRACT) misc_join
	$(EXTRACT) table_mod_12


extract_17_2:



