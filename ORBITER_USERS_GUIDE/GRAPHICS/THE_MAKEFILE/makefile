
MY_PATH=~/orbiter

#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/

EXTRACT=./extract.sh $(ORBITER_PATH) ~/bin/  ":" the_makefile EXTRACTS

EXTRACT_DEF=./extract.sh $(ORBITER_PATH) ~/bin/ "=" the_makefile EXTRACTS

all:
	cp ~/DATA/DATA.26/ORBITER_GUIDE/makefile ./the_makefile
	~/bin/a2tex.out -numbers -text_width 80 <the_makefile >the_makefile.tex
	~/bin/a2tex.out -text_width 10000  <the_makefile >the_makefile_nonumbers.tex
	~/bin/a2tex.out -text_width 10000  <C13.makefile >EXTRACTS/C13_makefile.tex
	~/bin/a2tex.out -text_width 10000  <PGL_4_2.makefile >EXTRACTS/PGL_4_2_makefile.tex
	~/bin/a2tex.out -text_width 10000  <Paley_13_group.makefile >EXTRACTS/Paley_13_group_makefile.tex
	~/bin/a2tex.out -text_width 10000  <15_lines.makefile >EXTRACTS/15_lines_makefile.tex
	~/bin/a2tex.out -text_width 10000  <Cycle_13_gens.csv >EXTRACTS/Cycle_13_gens_csv.tex
	~/bin/a2tex.out -text_width 10000  <PGGL_4_4_Subgroup_Hirschfeld_Stab_51840_subgroup_lattice.magma >EXTRACTS/PGGL_4_4_Subgroup_Hirschfeld_Stab_51840_subgroup_lattice.magma.tex
	~/bin/a2tex.out -text_width 80  <PGGL_4_4_Subgroup_Hirschfeld_Stab_51840_generators.magma >EXTRACTS/PGGL_4_4_Subgroup_Hirschfeld_Stab_51840_generators.magma.tex
	~/bin/a2tex.out -text_width 80  <../TEXT_OUTPUT/test_orbiter_session.txt >EXTRACTS/test_orbiter_session_log.tex


#~/bin/a2tex.out -numbers -text_width 60  </Users/betten/DATA.24/SYM_6/create_large_set_from_group_of_degree_143.magma >/Users/betten/TEX.24/FLAG_TRANSITIVE/create_large_set_from_group_of_degree_143.magma.tex


extract_all: preamble \
	extract_2_1 extract_2_2 extract_2_3 extract_2_4 extract_2_5 extract_2_6 extract_2_7 \
	extract_3_1 extract_3_2 extract_3_3 extract_3_4 \
	extract_4_1 extract_4_2 extract_4_3 extract_4_4 extract_4_5 \
	extract_5_1 extract_5_2 extract_5_3 extract_5_4 extract_5_5 extract_5_6 extract_5_7 extract_5_8 extract_5_9 \
	extract_6_1 extract_6_2 \
	extract_7_1 extract_7_2 extract_7_3 extract_7_4 extract_7_5 extract_7_6 extract_7_7 \
	extract_8_1 extract_8_2 extract_8_3 extract_8_4 extract_8_5 extract_8_6 extract_8_7 \
	extract_9_1 extract_9_2 extract_9_3 extract_9_4 extract_9_5 extract_9_6 extract_9_7 \
	extract_10_1 extract_10_2 extract_10_3 \
	extract_11_1 extract_11_2 extract_11_3 extract_11_4 extract_11_5 extract_11_6 extract_11_7 extract_11_8 extract_11_9 extract_11_10 \
	extract_12_1 extract_12_2 extract_12_3 extract_12_4 extract_12_5 extract_12_6 extract_12_7 extract_12_8 \
	extract_13_1 extract_13_2 extract_13_3  \
	extract_14_1 extract_14_2 extract_14_3 extract_14_4 extract_14_5 \
	extract_15_1 extract_15_2 extract_15_3 extract_15_4 \
	extract_16_1 extract_16_2 extract_16_3 \
	extract_17_1 extract_17_2 extract_17_3 extract_17_4 extract_17_5 extract_17_6 extract_17_7 extract_17_8 \
	extract_18_1 \
	extract_19_1 extract_19_2 extract_19_3 extract_19_4 extract_19_5 \
	extract_20_1 extract_20_2 \
	extract_21_1 extract_21_2
	cp EXTRACTS/* ../THE_BOOK_MAKEFILE/EXTRACTS



preamble:
	$(EXTRACT_DEF) SRC
	$(EXTRACT_DEF) HIRSCHFELD_STAB_GENERATORS
	$(EXTRACT_DEF) CLASS_2A_REP
	$(EXTRACT_DEF) FILE_NAME_CLASS_2A
	$(EXTRACT_DEF) TEST_FORMULA
	$(EXTRACT_DEF) BCH_POLYNOMIAL_ALFA_F256
	$(EXTRACT_DEF) BCH_POLYNOMIAL_BRAVO_F256
	$(EXTRACT_DEF) VANDERMONDE_4X4_FORMULA
	$(EXTRACT_DEF) ECKARDT_SURFACE_EQN
	$(EXTRACT_DEF) F_abcd_eqn
	$(EXTRACT_DEF) V7_VANDERMONDE_EXTENDED
	$(EXTRACT_DEF) CODE_RS_6_4_7
	$(EXTRACT_DEF) H1_LABEL
	$(EXTRACT_DEF) H1_GENS
	$(EXTRACT_DEF) DOILY
	$(EXTRACT_DEF) EDGE_CURVE_Q17_EQUATION
	$(EXTRACT_DEF) EDGE_CURVE_Q17_AS_POINTS
	$(EXTRACT_DEF) GENERATORS_M12
	$(EXTRACT_DEF) GEN_C13
	$(EXTRACT_DEF) GENERATORS_HESSE_GROUP
	$(EXTRACT_DEF) GENERATORS_WEYL_GROUP_E8
	$(EXTRACT_DEF) PGOp_6_2_GENS
	$(EXTRACT_DEF) SURFACE_Q13_STAB_GENS
	$(EXTRACT_DEF) HERMITIAN_SURFACE_Q9_EQUATION
	$(EXTRACT_DEF) HERMITIAN_SURFACE_Q9_POINTS
	$(EXTRACT_DEF) HERMITIAN_SURFACE_Q9_NB_POINTS
	$(EXTRACT_DEF) PGL_3_13_CLASS_2A_REP
	$(EXTRACT_DEF) PGL_3_13_CLASS_2B_REP
	$(EXTRACT_DEF) GENERATORS_H5
	$(EXTRACT_DEF) GENERATORS_N5
	$(EXTRACT_DEF) PGOM_6_2_CLASS_2A_REP
	$(EXTRACT_DEF) FILE_NAME_PGOM_6_2_CLASS_2A
	$(EXTRACT_DEF) PGOM_6_2_CLASS_2A_REP_CLEAN
	$(EXTRACT_DEF) PGOM_6_2_CLASS_2A_CENTRALIZER_GENS
	$(EXTRACT_DEF) CONWAY_GEN1
	$(EXTRACT_DEF) CONWAY_GEN2
	$(EXTRACT_DEF) Ree_gen1
	$(EXTRACT_DEF) Ree_gen2
	$(EXTRACT_DEF) HIGMAN_SIMS_GENS
	$(EXTRACT_DEF) HIGMAN_SIMS_NB_GENS
	$(EXTRACT_DEF) HIGMAN_SIMS_GO
	$(EXTRACT_DEF) CRC32_SPARSE
	$(EXTRACT_DEF) TWO_TO_THE_32_MINUS_2
	$(EXTRACT_DEF) INVERSE_SPARSE
	$(EXTRACT_DEF) EDGE_CURVE_EQUATION_ALGEBRAIC
	$(EXTRACT_DEF) QC_Q27_FO159_COEFFS
	$(EXTRACT_DEF) QC_Q27_FO388_COEFFS
	$(EXTRACT_DEF) QC_Q27_FO388_TRANS
	$(EXTRACT_DEF) CURVE_Q13_FLAG_1_EQN
	$(EXTRACT_DEF) CURVE_Q13_FLAG_1_POINTS
	$(EXTRACT_DEF) CURVE_Q13_FLAG_1_BITANGENTS
	$(EXTRACT_DEF) CURVE_Q13_FLAG_1_TRANS_1
	$(EXTRACT_DEF) CURVE_Q13_FLAG_1_TRANS_2
	$(EXTRACT_DEF) QUARTIC_CURVE_Q13_FLAG_0_EQN_COEFFICIENTS
	$(EXTRACT_DEF) QUARTIC_CURVE_Q13_FLAG_0_TRANS
	$(EXTRACT_DEF) ORTHOGONAL_OP_6_2_POINTS_OFF
	$(EXTRACT_DEF) SIEGEL_TRANSFORM_UV_2A
	$(EXTRACT_DEF) SIEGEL_TRANSFORM_UV_2C
	$(EXTRACT_DEF) Op_6_eqn
	$(EXTRACT_DEF) CURVE_Q13_F0_CLEAN2_EQUATION
	$(EXTRACT_DEF) HIRSCHFELD_EQUATION_XYZW
	$(EXTRACT_DEF) BARTH_EQUATION_GENERAL
	$(EXTRACT_DEF) QC_Q9_COEFFS
	$(EXTRACT_DEF) QC_Q9_ORBITER_CATALOGUE_0_AF
	$(EXTRACT_DEF) QC_Q9_HERMITIAN_COEFFS
	$(EXTRACT_DEF) QUARTIC_CURVE_Q9_CLEAN
	$(EXTRACT_DEF) QC_Q9_CLEAN_EQUATION
	$(EXTRACT_DEF) QC_Q9_CLEAN_LINES
	$(EXTRACT_DEF) QC_Q9_TRANSFORM_CLEAN_TO_HERMITIAN
	$(EXTRACT_DEF) QUARTIC_Q9_EQUATION
	$(EXTRACT_DEF) QUARTIC_Q9_NF_COEFFICIENTS
	$(EXTRACT_DEF) QC_Q9_UNITAL_COEFFICIENTS
	$(EXTRACT_DEF) CREMONA_MAP_Y0
	$(EXTRACT_DEF) CREMONA_MAP_Y1
	$(EXTRACT_DEF) CREMONA_MAP_Y2
	$(EXTRACT_DEF) CREMONA_MAP_Y3
	$(EXTRACT_DEF) SURFACE_F9_1_EQN_RESTRICTED
	$(EXTRACT_DEF) HIRSCHFELD_SURFACE_EQN_AF
	$(EXTRACT_DEF) Dickson1
	$(EXTRACT_DEF) Dickson2
	$(EXTRACT_DEF) Dickson3
	$(EXTRACT_DEF) Dickson4
	$(EXTRACT_DEF) Dickson5
	$(EXTRACT_DEF) Dickson6
	$(EXTRACT_DEF) Dickson7
	$(EXTRACT_DEF) Dickson8
	$(EXTRACT_DEF) Dickson9
	$(EXTRACT_DEF) Dickson10
	$(EXTRACT_DEF) Dickson11
	$(EXTRACT_DEF) Dickson12
	$(EXTRACT_DEF) Dickson13
	$(EXTRACT_DEF) Dickson14
	$(EXTRACT_DEF) Dickson15
	$(EXTRACT_DEF) Dickson16
	$(EXTRACT_DEF) Dickson17
	$(EXTRACT_DEF) Dickson18
	$(EXTRACT_DEF) Dickson19
	$(EXTRACT_DEF) Dickson20
	$(EXTRACT_DEF) Dickson21
	$(EXTRACT_DEF) Dickson22
	$(EXTRACT_DEF) Dickson23
	$(EXTRACT_DEF) Dickson24
	$(EXTRACT_DEF) Dickson25
	$(EXTRACT_DEF) Dickson26
	$(EXTRACT_DEF) Dickson27
	$(EXTRACT_DEF) Dickson28
	$(EXTRACT_DEF) Dickson29
	$(EXTRACT_DEF) Dickson30
	$(EXTRACT_DEF) Dickson31
	$(EXTRACT_DEF) Dickson32
	$(EXTRACT_DEF) Dickson33
	$(EXTRACT_DEF) Dickson34
	$(EXTRACT_DEF) Dickson35
	$(EXTRACT_DEF) Dickson36
	$(EXTRACT_DEF) Dickson37
	$(EXTRACT_DEF) Dickson38
	$(EXTRACT_DEF) Dickson39
	$(EXTRACT_DEF) Dickson40
	$(EXTRACT_DEF) Dickson41
	$(EXTRACT_DEF) Dickson42
	$(EXTRACT_DEF) Dickson43
	$(EXTRACT_DEF) Dickson44
	$(EXTRACT_DEF) Dickson45
	$(EXTRACT_DEF) Dickson46
	$(EXTRACT_DEF) Dickson47
	$(EXTRACT_DEF) Dickson1_eqn_af
	$(EXTRACT_DEF) Dickson2_eqn_af
	$(EXTRACT_DEF) Dickson3_eqn_af
	$(EXTRACT_DEF) Dickson4_eqn_af
	$(EXTRACT_DEF) Dickson5_eqn_af
	$(EXTRACT_DEF) Dickson6_eqn_af
	$(EXTRACT_DEF) Dickson7_eqn_af
	$(EXTRACT_DEF) Dickson8_eqn_af
	$(EXTRACT_DEF) Dickson9_eqn_af
	$(EXTRACT_DEF) Dickson10_eqn_af
	$(EXTRACT_DEF) Dickson11_eqn_af
	$(EXTRACT_DEF) Dickson12_eqn_af
	$(EXTRACT_DEF) Dickson13_eqn_af
	$(EXTRACT_DEF) Dickson14_eqn_af
	$(EXTRACT_DEF) Dickson15_eqn_af
	$(EXTRACT_DEF) Dickson16_eqn_af
	$(EXTRACT_DEF) Dickson17_eqn_af
	$(EXTRACT_DEF) Dickson18_eqn_af
	$(EXTRACT_DEF) Dickson19_eqn_af
	$(EXTRACT_DEF) Dickson20_eqn_af
	$(EXTRACT_DEF) Dickson21_eqn_af
	$(EXTRACT_DEF) Dickson22_eqn_af
	$(EXTRACT_DEF) Dickson23_eqn_af
	$(EXTRACT_DEF) Dickson24_eqn_af
	$(EXTRACT_DEF) Dickson25_eqn_af
	$(EXTRACT_DEF) Dickson26_eqn_af
	$(EXTRACT_DEF) Dickson27_eqn_af
	$(EXTRACT_DEF) Dickson28_eqn_af
	$(EXTRACT_DEF) Dickson29_eqn_af
	$(EXTRACT_DEF) Dickson30_eqn_af
	$(EXTRACT_DEF) Dickson31_eqn_af
	$(EXTRACT_DEF) Dickson32_eqn_af
	$(EXTRACT_DEF) Dickson33_eqn_af
	$(EXTRACT_DEF) Dickson34_eqn_af
	$(EXTRACT_DEF) Dickson35_eqn_af
	$(EXTRACT_DEF) Dickson36_eqn_af
	$(EXTRACT_DEF) Dickson37_eqn_af
	$(EXTRACT_DEF) Dickson38_eqn_af
	$(EXTRACT_DEF) Dickson39_eqn_af
	$(EXTRACT_DEF) Dickson40_eqn_af
	$(EXTRACT_DEF) Dickson41_eqn_af
	$(EXTRACT_DEF) Dickson42_eqn_af
	$(EXTRACT_DEF) Dickson43_eqn_af
	$(EXTRACT_DEF) Dickson44_eqn_af
	$(EXTRACT_DEF) Dickson45_eqn_af
	$(EXTRACT_DEF) Dickson46_eqn_af
	$(EXTRACT_DEF) Dickson47_eqn_af
	$(EXTRACT_DEF) HIRSCHFELD_SURFACE_POINTS_OFF
	$(EXTRACT_DEF) HIRSCHFELD_SURFACE_HESSE_PLANES
	$(EXTRACT_DEF) EQUATION_SURFACE_Q29_ISO4_ORB7_F1
	$(EXTRACT_DEF) EQUATION_SURFACE_Q29_ISO4_ORB7_F2
	$(EXTRACT_DEF) EQUATION_SURFACE_Q29_ISO4_ORB7_F3
	$(EXTRACT_DEF) QC_Q9_CLEAN_EQUATION
	$(EXTRACT_DEF) QC_Q9_CLEAN_LINES
	$(EXTRACT_DEF) NB_CUBIC_SURFACES_Q7
	$(EXTRACT_DEF) NB_CUBIC_SURFACES_Q13
	$(EXTRACT_DEF) NB_CUBIC_SURFACES_Q19
	$(EXTRACT_DEF) NB_QUARTIC_CURVES_Q19
	$(EXTRACT_DEF) NTRU_N
	$(EXTRACT_DEF) NTRU_P
	$(EXTRACT_DEF) NTRU_Q
	$(EXTRACT_DEF) NTRU_D
	$(EXTRACT_DEF) NTRU_XN1
	$(EXTRACT_DEF) ALICE_PRIVATE_F
	$(EXTRACT_DEF) ALICE_PRIVATE_G
	$(EXTRACT_DEF) ALICE_PRIVATE_FQ
	$(EXTRACT_DEF) ALICE_PRIVATE_FP
	$(EXTRACT_DEF) ALICE_PUBLIC_KEY
	$(EXTRACT_DEF) BOB_MESSAGE
	$(EXTRACT_DEF) BOB_ONE_TIME_KEY
	$(EXTRACT_DEF) BOB_ENCRYPT
	$(EXTRACT_DEF) ALICE_C1
	$(EXTRACT_DEF) ALICE_C2
	$(EXTRACT_DEF) ALICE_C3
	$(EXTRACT_DEF) ALICE_C4
	$(EXTRACT_DEF) CODE_5_2_3_CODEWORDS
	$(EXTRACT_DEF) SIMPLEX_CODE_GENERATOR
	$(EXTRACT_DEF) CODE_RM_3_1_GENMA
	$(EXTRACT_DEF) CODE_RM_4_1_GENMA
	$(EXTRACT_DEF) HAMMING_CODE_ROWS_IN_BINARY_RANKS
	$(EXTRACT_DEF) HAMMING_CODE_CODEWORDS
	$(EXTRACT_DEF) HAMMING_CODE_GENERATOR
	$(EXTRACT_DEF) GOLAY23_CODE_GENERATOR
	$(EXTRACT_DEF) GOLAY_23_COLUMN_RANKS_PROJECTIVELY
	$(EXTRACT_DEF) GOLAY_24_CODE_PROJECTIVE
	$(EXTRACT_DEF) GOLAY_24_CODE_AUT_GENS
	$(EXTRACT_DEF) M24_ELEMENT_2A
	$(EXTRACT_DEF) M24_ELEMENT_2B
	$(EXTRACT_DEF) M24_2A_PERM
	$(EXTRACT_DEF) M24_2B_PERM
	$(EXTRACT_DEF) GOLAY24_2A_FIX_SUBCODE
	$(EXTRACT_DEF) GOLAY24_2B_FIX_SUBCODE
	$(EXTRACT_DEF) CRC4
	$(EXTRACT_DEF) CRC7
	$(EXTRACT_DEF) CRC8_ATM
	$(EXTRACT_DEF) CRC16_CCITT
	$(EXTRACT_DEF) CRC32_ETHERNET
	$(EXTRACT_DEF) CRC32_ETHERNET_POLY
	$(EXTRACT_DEF) CRC32_CASTAGNOLI
	$(EXTRACT_DEF) CRC64_ECMA182
	$(EXTRACT_DEF) CRC64_ROCKSOFT
	$(EXTRACT_DEF) CRC_POLY_Q256_DEG2_DENSE
	$(EXTRACT_DEF) CRC_POLY_BRAVO_DENSE
	$(EXTRACT_DEF) CRC_POLY_CHARLIE_DENSE
	$(EXTRACT_DEF) POLY_Q256_DEG30_SPARSE
	$(EXTRACT_DEF) POLY_Q256_DEG30_DENSE
	$(EXTRACT_DEF) CRC_FILE
	$(EXTRACT_DEF) CRC_FILE_EXTENSION
	$(EXTRACT_DEF) TEN_MILLION
	$(EXTRACT_DEF) CRC_BCH_Q16_N51_D3_PROJECTIVE_SET
	$(EXTRACT_DEF) CRC_POLY_DELTA_DENSE
	$(EXTRACT_DEF) ERROR_POLY_BRAVO_COEFF
	$(EXTRACT_DEF) ERROR_POLY_BRAVO_PLACE
	$(EXTRACT_DEF) CRC_ERROR_POLY_BRAVO
	$(EXTRACT_DEF) RM_6_GENERATOR_1
	$(EXTRACT_DEF) RM_6_GENERATOR_2
	$(EXTRACT_DEF) RM_6_GENERATOR_3
	$(EXTRACT_DEF) RM_6_GENERATOR_4
	$(EXTRACT_DEF) RM_6_GENERATOR_5
	$(EXTRACT_DEF) RM_6_GENERATOR_6
	$(EXTRACT_DEF) RM_6_GENERATOR_7
	$(EXTRACT_DEF) CODE_BCH_F8_N21_D5_GENMA_OVERRIDE_POLYNOMIAL11
	$(EXTRACT_DEF) CODE_BCH_F8_N21_K14_D5_GENMA
	$(EXTRACT_DEF) CODE_BCH_F8_N63_K9_D43_GENMA
	$(EXTRACT_DEF) BCH_21_15_GENERATOR_MATRIX
	$(EXTRACT_DEF) BCH_21_6_GENERATOR_MATRIX
	$(EXTRACT_DEF) CODE_RS_F8_N7_K5_D3_GENMA
	$(EXTRACT_DEF) RS_8_reduced
	$(EXTRACT_DEF) CODE_21_15_4
	$(EXTRACT_DEF) CODE_15_6_6_A
	$(EXTRACT_DEF) CODE_15_6_6_B
	$(EXTRACT_DEF) CODE_15_6_6_C
	$(EXTRACT_DEF) CODE_15_6_6_D
	$(EXTRACT_DEF) CODE_15_6_6_E
	$(EXTRACT_DEF) CODE_GV_N15_K6
	$(EXTRACT_DEF) CODE_GV_N15_K6_CHECK
	$(EXTRACT_DEF) HIRSCHFELD_SURFACE_Q4_SET_OF_POINTS
	$(EXTRACT_DEF) HYPEROVAL_16_144
	$(EXTRACT_DEF) HYPEROVAL_16_16320
	$(EXTRACT_DEF) NEUMAIER_GRAPH_16_1_ADJ
	$(EXTRACT_DEF) NEUMAIER_GRAPH_16_2_ADJ
	$(EXTRACT_DEF) GEO_BLOCKS_600
	$(EXTRACT_DEF) LSQ_5A_TABLE
	$(EXTRACT_DEF) LSQ_5B_TABLE
	$(EXTRACT_DEF) TRIANGLE_GRAPH
	$(EXTRACT_DEF) GEN_C5
	$(EXTRACT_DEF) AG_2_3_BLOCKS
	$(EXTRACT_DEF) LARGE_SET_AG_2_3_NEIGHBOR_SET
	$(EXTRACT_DEF) PP4
	$(EXTRACT_DEF) PP4_GROUP1
	$(EXTRACT_DEF) PP4_MASK1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_GROUP1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_COLBOURN_COLBOURN_7_13_MASK1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_27_53
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_27_53_GROUP1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_27_53_MASK1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_3_7
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_3_7_GROUP1
	$(EXTRACT_DEF) DELANDTSHEER_DOYEN_PROBLEM_3_7_MASK1
	$(EXTRACT_DEF) PRAEGER_DESIGN_45_12_3_GENS
	$(EXTRACT_DEF) SPREAD_SET_27_RAO_RAO
	$(EXTRACT_DEF) SPREAD_S27_RAO_RAO
	$(EXTRACT_DEF) SPREADS_27_ISO_0
	$(EXTRACT_DEF) SPREADS_27_ISO_1
	$(EXTRACT_DEF) SPREADS_27_ISO_2
	$(EXTRACT_DEF) SPREADS_27_ISO_3
	$(EXTRACT_DEF) SPREADS_27_ISO_4
	$(EXTRACT_DEF) SPREADS_27_ISO_5
	$(EXTRACT_DEF) SPREADS_27_ISO_6
	$(EXTRACT_DEF) PENTTILA_WILLIAMS_PRINCE_REG_PACKING_0
	$(EXTRACT_DEF) PENTTILA_WILLIAMS_PRINCE_REG_PACKING_1
	$(EXTRACT_DEF) PGL_4_3_CLASS_5A_REP
	$(EXTRACT_DEF) PGL_4_3_CLASS_5A_N_GENS
	$(EXTRACT_DEF) PGL_4_5_SUBGROUP_3B_ME
	$(EXTRACT_DEF) PGL_4_5_SUBGROUP_3B_ME_NORMALIZER
	$(EXTRACT_DEF) PGL_4_5_SUBGROUP_31_ME
	$(EXTRACT_DEF) PGL_4_5_SUBGROUP_31_ME_NORMALIZER
	$(EXTRACT_DEF) BLT_ORDER_Q
	$(EXTRACT_DEF) BLT_NUMBER_ISO
	$(EXTRACT_DEF) TEST_SYSTEM
	$(EXTRACT_DEF) PRAEGER_DESIGN_LINE_TYPE
	$(EXTRACT_DEF) KLEIN_CUBIC_THREEFOLD_EQN
	$(EXTRACT_DEF) BARTH_SEXTIC_MOD_11_SINGULAR_POINTS
	$(EXTRACT_DEF) NB_CUBIC_SURFACES_Q13
	$(EXTRACT_DEF) HIRSCHFELD_SURFACE_EQUATION_ALGEBRAIC_FORM
	$(EXTRACT_DEF) SUZUKI_8_GENERATORS
	$(EXTRACT_DEF) FILE_24_3_TFC_INC
	$(EXTRACT_DEF) CODE_N3_K2_Q2_GENMA
	$(EXTRACT_DEF) CODE_N6_K3_Q2_GENMA
	$(EXTRACT_DEF) SYM_6_CLASS_2A_REP
	$(EXTRACT_DEF) SYM_6_CLASS_2B_REP
	$(EXTRACT_DEF) SYM_6_CLASS_2C_REP
	$(EXTRACT_DEF) FILE_NAME_SYM_6_CLASS_2A
	$(EXTRACT_DEF) FILE_NAME_SYM_6_CLASS_2B
	$(EXTRACT_DEF) FILE_NAME_SYM_6_CLASS_2C
	$(EXTRACT_DEF) AUT_SYM_6_GROUP_ORDER
	$(EXTRACT_DEF) AUT_SYM_6_DEGREE
	$(EXTRACT_DEF) AUT_SYM_6_BASE
	$(EXTRACT_DEF) AUT_SYM_6_NB_GENS
	$(EXTRACT_DEF) AUT_SYM_6_GENS
	$(EXTRACT_DEF) POLISHED_CHROME_WHITE
	$(EXTRACT_DEF) YELLOW_TRANSPARENT
	$(EXTRACT_DEF) COLOR_RED
	$(EXTRACT_DEF) COLOR_RED_SHINY
	$(EXTRACT_DEF) COLOR_GREEN_SHINY
	$(EXTRACT_DEF) COLOR_BLUE_SHINY
	$(EXTRACT_DEF) COLOR_YELLOW_SHINY
	$(EXTRACT_DEF) COLOR_BLACK_SHINY
	$(EXTRACT_DEF) COLOR_RED_SEE_THROUGH
	$(EXTRACT_DEF) COLOR_GREEN_SEE_THROUGH
	$(EXTRACT_DEF) COLOR_BLUE_SEE_THROUGH
	$(EXTRACT_DEF) COLOR_YELLOW_SEE_THROUGH
	$(EXTRACT_DEF) COLOR_YELLOW_THICK
	$(EXTRACT_DEF) COLOR_BLACK_NO_SHADOW
	$(EXTRACT_DEF) SURFACE_COLOR
	$(EXTRACT_DEF) SURFACE_COLOR_SEETHROUGH
	$(EXTRACT_DEF) COLOR_GOLD
	$(EXTRACT_DEF) COLOR_TURQUOISE
	$(EXTRACT_DEF) MONKEY_SADDLE_CUBIC
	$(EXTRACT_DEF) ECKARDT_CUBIC_DEFORM1_LEX
	$(EXTRACT_DEF) ECKARDT_CUBIC_DEFORM2_LEX
	$(EXTRACT_DEF) KUMMER_QUARTIC_LEX_35
	$(EXTRACT_DEF) BEAUVILLE_QUINTIC_LEX_56
	$(EXTRACT_DEF) ENDRASS_OCTIC_LEX_165
	$(EXTRACT_DEF) CLEBSCH_A
	$(EXTRACT_DEF) CLEBSCH_B
	$(EXTRACT_DEF) CLEBSCH_C
	$(EXTRACT_DEF) CLEBSCH_D
	$(EXTRACT_DEF) TWO_PI
	$(EXTRACT_DEF) T00
	$(EXTRACT_DEF) T01
	$(EXTRACT_DEF) T02
	$(EXTRACT_DEF) T03
	$(EXTRACT_DEF) T10
	$(EXTRACT_DEF) T11
	$(EXTRACT_DEF) T12
	$(EXTRACT_DEF) T13
	$(EXTRACT_DEF) T20
	$(EXTRACT_DEF) T21
	$(EXTRACT_DEF) T22
	$(EXTRACT_DEF) T23
	$(EXTRACT_DEF) T30
	$(EXTRACT_DEF) T31
	$(EXTRACT_DEF) T32
	$(EXTRACT_DEF) T33
	$(EXTRACT_DEF) CLEBSCH_CUBICS
	$(EXTRACT_DEF) GNUPLOT_TEST_INPUT
	$(EXTRACT_DEF) BLT_ORDER_Q
	$(EXTRACT_DEF) BLT_NUMBER_ISO
	






extract_2_1:
	$(EXTRACT) test_orbiter_session


extract_2_2:

extract_2_3:
	$(EXTRACT) object_F_2
	$(EXTRACT) object_PG_3_2
	$(EXTRACT) vector_ex


extract_2_4:
	$(EXTRACT) set_of_primes
	$(EXTRACT) set_interval


extract_2_5:
	$(EXTRACT) vector_example1
	$(EXTRACT) vector_example2
	$(EXTRACT) vector_example_repeat
	$(EXTRACT) vector_example_all_one_11
	$(EXTRACT) vector_loop_8
	$(EXTRACT) vector_loop_odd_16
	$(EXTRACT) vector_concatenate
	$(EXTRACT) matrix_example1
	$(EXTRACT) vector_example_sparse
	$(EXTRACT) matrix_example2
	$(EXTRACT) matrix_example_co_1
	$(EXTRACT) vector_index_set
	$(EXTRACT) permutation_matrix


extract_2_6:
	$(EXTRACT) Hirschfeld_Class_2A
	$(EXTRACT) surface_4_0_export_action

extract_2_7:
	$(EXTRACT) symbolic_poly
	$(EXTRACT) symbolic_CRC32
	$(EXTRACT) symbolic_CRC_alfa
	$(EXTRACT) symbolic_CRC_alfa_draw_trees
	$(EXTRACT) symbolic_CRC_bravo
	$(EXTRACT) symbolic_CRC_bravo_draw_trees
	$(EXTRACT) symbolic_test_4_expand
	$(EXTRACT) symbolic_test_4_expand_draw
	$(EXTRACT) symbolic_test_5_expand
	$(EXTRACT) test_expand_1
	$(EXTRACT) test_expand_2
	$(EXTRACT) test_expand_3
	$(EXTRACT) test_expand_4
	$(EXTRACT) symbolic_objects_building_on_earlier_objects
	$(EXTRACT) symbolic_objects_self_referential_1
	$(EXTRACT) symbolic_objects_self_referential_2
	$(EXTRACT) symbolic_test_6
	$(EXTRACT) symbolic_matrix1
	$(EXTRACT) symbolic_matrix_2x2
	$(EXTRACT) symbolic_matrix_3x3
	$(EXTRACT) symbolic_determinant_2x2
	$(EXTRACT) symbolic_determinant_3x3
	$(EXTRACT) symbolic_determinant_vandermonde
	$(EXTRACT) symbolic_characteristic_polynomial_2x2
	$(EXTRACT) symbolic_characteristic_polynomial_2x2_draw
	$(EXTRACT) symbolic_characteristic_polynomial_3x3
	$(EXTRACT) symbolic_object1
	$(EXTRACT) symbolic_object1_expand
	$(EXTRACT) symbolic_object1_draw
	$(EXTRACT) symbolic_object1_evaluate
	$(EXTRACT) symbolic_Eckardt_surface
	$(EXTRACT) symbolic_Eckardt_surface_q13_a3_b1
	$(EXTRACT) symbolic_Fabcd
	$(EXTRACT) symbolic_Fabcd_expand
	$(EXTRACT) symbolic_Fabcd_q31_E18
	$(EXTRACT) symbolic_Fabcd_q31_E18_with_implicit_substitution
	$(EXTRACT) symbolic_Fabcd_q31_E18_with_implicit_substitution_do_not_simplify
	$(EXTRACT) symbolic_matrix_minor
	$(EXTRACT) symbolic_matrix_nullspace_1
	$(EXTRACT) symbolic_matrix_nullspace_2
	$(EXTRACT) symbolic_Clebsch
	$(EXTRACT) Clebsch_map_up_Fabcd
	$(EXTRACT) Hirschfeld_evaluate_symbolic



extract_3_1:
	$(EXTRACT) F_2
	$(EXTRACT) F_3
	$(EXTRACT) F_5
	$(EXTRACT) F_5_add_table
	$(EXTRACT) F_7
	$(EXTRACT) F_7_tables
	$(EXTRACT) F_127
	$(EXTRACT) F_11_product_of_all_nonzero_elements
	$(EXTRACT) F_7_vandermonde
	$(EXTRACT) F_101_wo
	$(EXTRACT) F_1021_wo


extract_3_2:
	$(EXTRACT) F_4
	$(EXTRACT) F_4_tables
	$(EXTRACT) F_8
	$(EXTRACT) F_8_tables
	$(EXTRACT) F_8b
	$(EXTRACT) F_16
	$(EXTRACT) F_16_calculate
	$(EXTRACT) F_16_tables
	$(EXTRACT) F_64
	$(EXTRACT) nth_roots_63
	$(EXTRACT) F8_F2_field_reduction
	$(EXTRACT) F_4096
	$(EXTRACT) F_4489
	$(EXTRACT) F_2_power_24


extract_3_3:
	$(EXTRACT) RREF
	$(EXTRACT) RREF_V7
	$(EXTRACT) nullspace
	$(EXTRACT) RREF_RS_6_4_7
	$(EXTRACT) eigenstuff
	$(EXTRACT) classes_GL_3_2
	$(EXTRACT) classes_GL_4_2


extract_3_4:
	$(EXTRACT) normal_basis_2_3
	$(EXTRACT) normal_basis_2_4
	$(EXTRACT) normal_basis_2_4_poly19
	$(EXTRACT) normal_basis_2_6
	$(EXTRACT) normal_basis_2_8
	$(EXTRACT) F8_over_F2_field_reduction
	$(EXTRACT) F_64_over_F8_field_reduction
	$(EXTRACT) F_64_over_F4_field_reduction
	$(EXTRACT) F_64_over_F2_field_reduction
	$(EXTRACT) F_8_Nth_roots_21
	$(EXTRACT) F_8_vandermonde
	$(EXTRACT) F_512_vandermonde
	$(EXTRACT) NTT_k4_q17.cpp
	$(EXTRACT) F_17_NTT_compile


#extract_3_5:
#	$(EXTRACT) Polynomial_ring



extract_4_1:
	$(EXTRACT) PG_3_2_easy
	$(EXTRACT) PG_1_16
	$(EXTRACT) PG_2_4
	$(EXTRACT) PG_2_13
	$(EXTRACT) PG_2_64
	$(EXTRACT) PG_3_2
	$(EXTRACT) PG_3_4
	$(EXTRACT) PG_3_5
	$(EXTRACT) PG_3_7
	$(EXTRACT) PG_3_8
	$(EXTRACT) PG_3_16
	$(EXTRACT) PG_3_25
	$(EXTRACT) PG_4_2
	$(EXTRACT) PG_4_3
	$(EXTRACT) PG_8_2

extract_4_2:
	$(EXTRACT) PG_2_4_rank_point
	$(EXTRACT) PG_2_4_unrank_point
	$(EXTRACT) PG_2_4_print_points
	$(EXTRACT) PG_2_2_rank_lines
	$(EXTRACT) PG_2_2_unrank_lines
	$(EXTRACT) PG_3_3_rank_lines
	$(EXTRACT) PG_3_5_unrank_lines


extract_4_3:
	$(EXTRACT) PG_2_16
	$(EXTRACT) PG_2_2_incidence_matrix
	$(EXTRACT) PG_2_4_incidence_matrix
	$(EXTRACT) PG_2_8_incidence_matrix
	$(EXTRACT) PG_2_4_with_decomposition
	$(EXTRACT) PG_2_4_incma_cyclic
	$(EXTRACT) PG_2_4_incma_singer_sub_3
	$(EXTRACT) PG_2_4_incma_singer_sub_7


extract_4_4:
	$(EXTRACT) fixed_objects_2A
	$(EXTRACT) fixed_objects_2B
	$(EXTRACT) fixed_objects_2C
	$(EXTRACT) fix_structure_G1
	$(EXTRACT) trans
	$(EXTRACT) PG_2_3_plane_pencil


extract_4_5:
	$(EXTRACT) elliptic_quadric_ovoid_q8
	$(EXTRACT) ovoid_ST_q8
	$(EXTRACT) Baer_PG_2_4
	$(EXTRACT) Baer_PG_3_4
	$(EXTRACT) BLT_database_5_0
	$(EXTRACT) BLT_database_7_0
	$(EXTRACT) BLT_database_7_0_print
	$(EXTRACT) BLT_database_67_4







extract_5_1:
	$(EXTRACT) Cyclic_6
	$(EXTRACT) Cyclic_6_from_generator
	$(EXTRACT) Cyclic_6_group_table
	$(EXTRACT) Symmetric_3
	$(EXTRACT) Symmetric_3_group_table
	$(EXTRACT) Symmetric_3_elements
	$(EXTRACT) Symmetric_3_long
	$(EXTRACT) Elementary_abelian_8
	$(EXTRACT) Symmetric_4
	$(EXTRACT) Symmetric_4_group_table
	$(EXTRACT) Symmetric_4_long
	$(EXTRACT) Group_8_1
	$(EXTRACT) Group_8_2
	$(EXTRACT) Group_8_3
	$(EXTRACT) Group_8_4
	$(EXTRACT) Group_8_5
	$(EXTRACT) Group_M12


extract_5_2:
	$(EXTRACT) GL_2_3_elements_dense
	$(EXTRACT) PGL_2_3_elements_dense
	$(EXTRACT) PGL_2_17
	$(EXTRACT) PSL_2_17
	$(EXTRACT) PGL_3_2
	$(EXTRACT) PGL_4_2
	$(EXTRACT) PGL_4_2_with_field
	$(EXTRACT) PGL_4_2_report
	$(EXTRACT) PGL_4_2_multiply_all_elements
	$(EXTRACT) PGL_4_2_export
	$(EXTRACT) PGL_4_2_export_GAP
	$(EXTRACT) PGL_4_2_generated
	$(EXTRACT) PGL_5_2
	$(EXTRACT) PGL_5_2_multiply_all_elements
	$(EXTRACT) L_5_3
	$(EXTRACT) L_4_5
	$(EXTRACT) PGL_4_5
	$(EXTRACT) PGGL_2_9
	$(EXTRACT) PGGL_3_4
	$(EXTRACT) PGGL_3_8
	$(EXTRACT) AGGL_1_4
	$(EXTRACT) AGGL_1_8
	$(EXTRACT) AGL_1_27
	$(EXTRACT) AGGL_2_27
	$(EXTRACT) AGGL_1_81
	$(EXTRACT) SP_4_2
	$(EXTRACT) PSP_4_4
	$(EXTRACT) PGOp_4_2
	$(EXTRACT) PGO_5_2
	$(EXTRACT) PGGO_5_4
	$(EXTRACT) PGOp_6_2
	$(EXTRACT) PGOm_6_2
	$(EXTRACT) PSP_6_2
	$(EXTRACT) PGO_7_2
	$(EXTRACT) NullPolarity_6_2
	$(EXTRACT) M24_matrix_group


extract_5_3:
	$(EXTRACT) C13
	$(EXTRACT) C13_generated
	$(EXTRACT) C13_as_subgroup
	$(EXTRACT) J1
	$(EXTRACT) PGL_3_11_singer
	$(EXTRACT) PGL_3_11_singer_and_frobenius
	$(EXTRACT) PG_2_4_order_21
	$(EXTRACT) Sym3_in_PGL_2_3_group_table
	$(EXTRACT) Sym3_in_PGL_2_3_report
	$(EXTRACT) quaternion
	$(EXTRACT) cube_group
	$(EXTRACT) tetra_group
	$(EXTRACT) Hesse_group
	$(EXTRACT) Weyl_E8
	$(EXTRACT) test_subgroup
	$(EXTRACT) coset_reps
	$(EXTRACT) coset_reps_read
	$(EXTRACT) SP_6_2_point_stab_subgroup
	$(EXTRACT) PGOp_6_2_report
	$(EXTRACT) PGOm_6_2_report
	$(EXTRACT) PGOm_6_2_export_magma
	$(EXTRACT) PGOp_6_2_point_stab_subgroup
	$(EXTRACT) PGOp_6_2_linear
	$(EXTRACT) PGOp_6_2_linear_stab_6



extract_5_4:
	$(EXTRACT) Cyclic_6_aut
	$(EXTRACT) Elementary_abelian_8_aut
	$(EXTRACT) Symmetric_4_on_pairs
	$(EXTRACT) Alternating_5
	$(EXTRACT) Alternating_5_derived
	$(EXTRACT) direct_product_AGL
	$(EXTRACT) PGLstar_3_2
	$(EXTRACT) PGLstar_3_2_restrict
	$(EXTRACT) PGGLstar_3_4
	$(EXTRACT) PGLstar_4_2
	$(EXTRACT) Group_M11
	$(EXTRACT) surface_q13_Eckardt
	$(EXTRACT) surface_q13_Eckardt_on_tritangent_planes
	$(EXTRACT) Hirschfeld_stab_projectivity_group
	$(EXTRACT) Hirschfeld_stab_field_reduction
	$(EXTRACT) on_planes
	$(EXTRACT) on_planes_restricted
	$(EXTRACT) T3_on_tensors
	$(EXTRACT) T3r1
	$(EXTRACT) T4_on_tensors
	$(EXTRACT) T4r1
	$(EXTRACT) PGGL_2_8_on_conic
	$(EXTRACT) PGL_4_2_wedge
	$(EXTRACT) PGL_5_3_wedge
	$(EXTRACT) PGL_4_2_wd
	$(EXTRACT) PGL_5_3_wd
	$(EXTRACT) PGL_4_2_wd_reverse
	$(EXTRACT) M24_perm_group
	$(EXTRACT) M22_bar
	$(EXTRACT) curve_q13_f0_clean2_variety_make_group
	$(EXTRACT) conic_group_q5
	$(EXTRACT) group_of_Hirschfeld_surface_q4
	$(EXTRACT) Higman_Sims_on_176_points
	$(EXTRACT) group_U_4_3



extract_5_5:
	$(EXTRACT) PGL_3_2_order_invariant
	$(EXTRACT) PGL_3_2_elements
	$(EXTRACT) Sym_3_make_element_tree
	$(EXTRACT) Sym_3_elements
	$(EXTRACT) Cycle_13_power
	$(EXTRACT) Cycle_12_power
	$(EXTRACT) PGL_3_4_singer
	$(EXTRACT) GL_2_8_multiply
	$(EXTRACT) GL_2_7_multiply
	$(EXTRACT) GL_2_7_inv
	$(EXTRACT) GL_2_7_power
	$(EXTRACT) PGL_3_2_classes
	$(EXTRACT) PGL_4_2_classes_based_on_normal_form
	$(EXTRACT) PGL_10_2_classes_based_on_normal_form
	$(EXTRACT) PGGL_4_4_small_generating_set
	$(EXTRACT) normal_forms_PGL_4_4
	$(EXTRACT) PGL_4_4_2A_rank
	$(EXTRACT) PGL_4_4_2A_unrank
	$(EXTRACT) PGL_4_5_3B_rank
	$(EXTRACT) PGL_4_5_3B_unrank
	$(EXTRACT) normal_forms_PGL_4_5
	$(EXTRACT) on_planes
	$(EXTRACT) PGL_4_2_random_element
	$(EXTRACT) PGL_2_3_subgroup_lattice
	$(EXTRACT) PGL_2_3_subgroup_lattice_load_and_draw_by_groups
	$(EXTRACT) PGL_2_3_subgroup_lattice_load_and_draw_by_orbits
	$(EXTRACT) PGL_3_13_2B_RNF

extract_5_6:
	$(EXTRACT) Symmetric_4_diagram_of_conjugacy_classes
	$(EXTRACT) PSP_4_4_export_magma
	$(EXTRACT) PGGL_2_4_export_magma
	$(EXTRACT) Hirschfeld_stab_export_magma
	$(EXTRACT) Hirschfeld_stab_subgroup_lattice_magma
	$(EXTRACT) PGGL_2_4_classes
	$(EXTRACT) PGLstar_3_2_classes
	$(EXTRACT) PGL_4_2_classes
	$(EXTRACT) PGL_3_11_classes
	$(EXTRACT) PGL_7_2_classes
	$(EXTRACT) PGL_8_2_classes
	$(EXTRACT) Hirschfeld_stab_classes
	$(EXTRACT) PGGL_2_4_cent_2A
	$(EXTRACT) Normalizer_of_H5
	$(EXTRACT) PGGL_3_4_classes
	$(EXTRACT) classes_PGGL_4_4
	$(EXTRACT) classes_PGL_4_5
	$(EXTRACT) PGL_4_5_3B_class_again
	$(EXTRACT) Normalizer_of_Z22_in_PGL_2_9
	$(EXTRACT) PGOm_6_2_classes
	$(EXTRACT) PGOm_6_2_Class_2A
	$(EXTRACT) PGOm_6_2_Class_2A_print
	$(EXTRACT) PGOm_6_2_Class_2A_centralizer
	$(EXTRACT) PGOm_6_2_Class_2A_centralizer_recover


extract_5_7:
	$(EXTRACT) Hirschfeld_stab_export_gap
	$(EXTRACT) Cyclic_6_canonical_image
	$(EXTRACT) Edge_qurve_q17_canonical_image


extract_5_8:
	$(EXTRACT) U_3_3
	$(EXTRACT) PGL_2_3_group_table
	$(EXTRACT) Co3
	$(EXTRACT) Ree_27
	$(EXTRACT) Higman_Sims_group
	$(EXTRACT) Higman_Sims_group_classes
	$(EXTRACT) Higman_Sims_group_subgroup_lattice
	$(EXTRACT) Higman_Sims_subgroup_176a
	$(EXTRACT) Higman_Sims_subgroup_176b
	$(EXTRACT) Butler12_PSP_6_2_conjugate
	$(EXTRACT) Hirschfeld_surface_get_subfield_subgroup



extract_5_9:
	$(EXTRACT) PGGL_2_4_classes_reps
	$(EXTRACT) PGL_4_4_classes_reps
	$(EXTRACT) PGGL_4_4_classes_reps
	$(EXTRACT) Sym_6_generators
	$(EXTRACT) Sym_6_elements
	$(EXTRACT) GL_2_3_RNF_elements
	$(EXTRACT) PGL_4_2_orthogonal_reflections_report_vector_ge







extract_6_1:
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
	$(EXTRACT) mult_polynomials_17_12
	$(EXTRACT) polynomial_division_ranked_2_204_37
	$(EXTRACT) Berlekamp_matrix_crc32
	$(EXTRACT) power_mod_inverse
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
	$(EXTRACT) F2_get_primitive_poly_of_deg_10
	$(EXTRACT) F2_get_primitive_poly_range_2_10


extract_6_2:
	$(EXTRACT) polynomial_ring_degree3_vars4_q13
	$(EXTRACT) parse_homogeneous_polynomial
	$(EXTRACT) parse_homogeneous_polynomial_with_parameters
	$(EXTRACT) affine_map_F5_0
	$(EXTRACT) affine_map_F5_1
	$(EXTRACT) affine_map_F5_2
	$(EXTRACT) affine_map_F5_3
	$(EXTRACT) affine_map_F5_4
	$(EXTRACT) permutation_polynomial_F5
	$(EXTRACT) conic_variety
	$(EXTRACT) ideal_of_conic
	$(EXTRACT) quartic_curve_q27_fo338_apply_transformation


extract_7_1:
	$(EXTRACT) GR_3_2_2
	$(EXTRACT) planes_in_pencil



extract_7_2:
	$(EXTRACT) GR_4_2_2


extract_7_3:
	$(EXTRACT) Op_4_2
	$(EXTRACT) O_5_2_incidence_matrix.csv
	$(EXTRACT) Op_6_2
	$(EXTRACT) Op_6_2_incidence_matrix.csv
	$(EXTRACT) Op_6_2_with_group
	$(EXTRACT) Op_6_8
	$(EXTRACT) Op_8_2
	$(EXTRACT) Op_6_64
	$(EXTRACT) Op_6_64_line_rank_problem
	$(EXTRACT) Op_6_64_line_rank
	$(EXTRACT) Op_6_64_report
	$(EXTRACT) elliptic_quadric_subspace
	$(EXTRACT) BLT_database_7_1
	$(EXTRACT) BLT_database_7_1_print
	$(EXTRACT) Doily_W_2
	$(EXTRACT) Op_6_2_orthogonal_reflections
	$(EXTRACT) Op_6_2_create_Siegel_transformation_2B
	$(EXTRACT) Op_6_2_create_Siegel_transformation_2C
	$(EXTRACT) Op_6_2_report_Siegel_transformation
	$(EXTRACT) Op_6_2_Siegel_transformation_expand_class_2A
	$(EXTRACT) Op_6_2_Siegel_transformation_expand_class_2C
	$(EXTRACT) Op_6_2_make_all_Siegel_transformations
	$(EXTRACT) Op_6_2_all_Siegel_transformations_tally


extract_7_4:
	$(EXTRACT) H_2_4
	$(EXTRACT) H_2_9
	$(EXTRACT) H_3_4





extract_7_5:
	$(EXTRACT) conic_variety_q13
	$(EXTRACT) Op_6_2_variety
	$(EXTRACT) curve_q13_f0
	$(EXTRACT) Hirschfeld_q2
	$(EXTRACT) quartic_curve_q13_f1_apply_transformation
	$(EXTRACT) quartic_curve_q13_f0_apply_transformation
	$(EXTRACT) cubic_eqn_q3_0





extract_7_6:
	$(EXTRACT) conic_variety_q13_report
	$(EXTRACT) Op_6_2_group
	$(EXTRACT) curve_q13_f0_clean2_variety_compute_group
	$(EXTRACT) Hirschfeld_q2_group
	$(EXTRACT) Hirschfeld_q4_group
	$(EXTRACT) Barth_sextic_mod_11
	$(EXTRACT) variety_Dickson36_9lines_q4_group
	$(EXTRACT) quartic_curve_test_isomorphism
	$(EXTRACT) quartic_curve_q27_fo159_fo338_test_isomorphism
	$(EXTRACT) quartic_curve_q9_hermitian_fo0_test_isomorphism
	$(EXTRACT) quartic_curve_q9_hermitian_and_OCN0_test_isomorphism
	$(EXTRACT) quartic_curve_q9_Hermitian_transform




extract_7_7:
	$(EXTRACT) Cremona_map
	$(EXTRACT) Veronese_1_3_q7







extract_8_1:
	$(EXTRACT) orbits_PGL_4_2_on_points_draw_tree
	$(EXTRACT) orbits_PGL_4_2_on_points_export_trees
	$(EXTRACT) DD_PP4_orbit
	$(EXTRACT) T3r1_orbits
	$(EXTRACT) T3r1_orbits_draw
	$(EXTRACT) 2C_orbit_under_PGGL_4_4_elements_coded.csv
	$(EXTRACT) orbits_on_conics_q13
	$(EXTRACT) orbits_cubic_curves_q2
	$(EXTRACT) orbits_cubic_curves_q2_with_draw_tree
	$(EXTRACT) poly_orbits_d3_n2_q2_compute
	$(EXTRACT) poly_orbits_d3_n3_q2.csv
	$(EXTRACT) poly_orbits_d3_n3_q2_get_ranks
	$(EXTRACT) T4_orbits
	$(EXTRACT) T4r1_orbits
	$(EXTRACT) T4r1_orbits_draw
	$(EXTRACT) T4r1_orbits_4
	$(EXTRACT) PGGL_7_8_orbits


extract_8_2:
	$(EXTRACT) poset_of_4subsets
	$(EXTRACT) poset_of_4subsets_draw
	$(EXTRACT) poset_of_5subsets
	$(EXTRACT) poset_of_5subsets_draw
	$(EXTRACT) Symmetric_4_on_pairs_poset
	$(EXTRACT) V_3_2_trivial
	$(EXTRACT) V_4_2_trivial


extract_8_3:
	$(EXTRACT) poset_of_subsets_under_C6
	$(EXTRACT) orbits_on_subsets_in_action_on_pairs
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


extract_8_4:
	$(EXTRACT) PGL_5_3_wedge_subspace_orbits
	$(EXTRACT) subspaces_Op_4_2
	$(EXTRACT) PGL_4_2_on_subspaces
	$(EXTRACT) PGL_4_2_singer_on_subspaces
	$(EXTRACT) PGL_8_2_singer_on_subspaces
	$(EXTRACT) Op_6_2_orbits_on_subspaces
	$(EXTRACT) Op_6_3_orbits_on_subspaces
	$(EXTRACT) Op_6_11_orbits_on_subspaces
	$(EXTRACT) Op_8_2_orbits_on_subspaces
	$(EXTRACT) PGO_7_2_on_subspaces


extract_8_5:
	$(EXTRACT) C6_on_partition
	$(EXTRACT) PGL_2_17_on_partition


extract_8_6:
	$(EXTRACT) PGL_3_27
	$(EXTRACT) hyperoval_4_classify
	$(EXTRACT) hyperoval_8_classify
	$(EXTRACT) hyperoval_16_classify
	$(EXTRACT) hyperoval_16_1_conic_type
	$(EXTRACT) hyperoval_16_1_nonconical_type
	$(EXTRACT) hyperoval_16_2_nonconical_type
	$(EXTRACT) hyperoval_16_stab_0_disjoint_sets_graph
	$(EXTRACT) nc_arcs_16
	$(EXTRACT) five_arcs_q13


extract_8_7:
	$(EXTRACT) cubic_curves_PG_2_4
	$(EXTRACT) poly_orbits_q4
	$(EXTRACT) cubic_curves_PG_2_9
	$(EXTRACT) cubic_curves_PG_2_9_draw
	$(EXTRACT) curve_orbit_mini
	$(EXTRACT) curve_orbit


extract_9_1:
	$(EXTRACT) surface_4_0_lines_vs_lines
	$(EXTRACT) surface_4_0_lines_vs_tritangent_planes
	$(EXTRACT) surface_4_0_dot_products_columns
	$(EXTRACT) surface_4_0_export_double_sixes
	$(EXTRACT) surface_4_0_multiply_half_double_six_vs_lines
	$(EXTRACT) surface_4_0_multiply_half_double_six_vs_incma
	$(EXTRACT) surface_4_0_dot_products_rows
	$(EXTRACT) surface_4_0_dot_products_rowspan
	$(EXTRACT) surface_4_0_single_six_times_lines_vs_lines_rowspan

extract_9_2:
	$(EXTRACT) surface_4_0
	$(EXTRACT) Hirschfeld_surface_q4
	$(EXTRACT) Family_general_F7
	$(EXTRACT) surface_eckardt_13_4_12
	$(EXTRACT) Eckardt_13
	$(EXTRACT) surface_11_random
	$(EXTRACT) surface_F_abcd_Eckardt_q31_by_equation
	$(EXTRACT) Eckardt_13_by_arc_lifting
	$(EXTRACT) surface_F_abcd_Eckardt_q31_by_symbolic_object
	$(EXTRACT) Hirschfeld_surface_by_arc_lifting_with_two_lines
	$(EXTRACT) Dickson6_q2
	$(EXTRACT) Dickson1_group
	$(EXTRACT) Dickson2_group

extract_9_3:
	$(EXTRACT) surface_4_0_report
	$(EXTRACT) surface_4_0_clean_report
	$(EXTRACT) surface_4_0_export
	$(EXTRACT) Hirschfeld_surface_get_incidence_matrix_40_40
	$(EXTRACT) Hirschfeld_surface_incma_40_40_c
	$(EXTRACT) surface_4_0_plane_type_of_lines_on_Klein_quadric
	$(EXTRACT) surface_7_0
	$(EXTRACT) surface_7_0_plane_type_of_lines_on_Klein_quadric
	$(EXTRACT) Family_general_F7_report
	$(EXTRACT) surface_8_0
	$(EXTRACT) surface_8_0_plane_type_of_lines_on_Klein_quadric
	$(EXTRACT) surface_8_0_catalogue
	$(EXTRACT) surface_8_0_clean
	$(EXTRACT) surface_8_0b
	$(EXTRACT) surface_9_0
	$(EXTRACT) surface_9_0_plane_type_of_lines_on_Klein_quadric
	$(EXTRACT) surface_9_1
	$(EXTRACT) surface_9_1_plane_type_of_lines_on_Klein_quadric
	$(EXTRACT) surface_eckardt_13_4_12_report
	$(EXTRACT) Eckardt_13_report
	$(EXTRACT) Eckardt_13_by_arc_lifting_report
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
	$(EXTRACT) surface_F_abcd_Eckardt_q31_by_symbolic_object_report
	$(EXTRACT) surface_11_random_report
	$(EXTRACT) surface_11_1
	$(EXTRACT) surface_8_0_recognize_abcd





extract_9_4:
	$(EXTRACT) surface_classify_q4
	$(EXTRACT) surface_classify_q4_arc_lifting_two_lines
	$(EXTRACT) surface_classify_q7
	$(EXTRACT) Family_general_F31_25_5_5_25_recognize
	$(EXTRACT) surface_F_abcd_identify_q9
	$(EXTRACT) Family_general_F11_iso_test
	$(EXTRACT) surface_classify_q7_sweep_Cayley



extract_9_5:
	$(EXTRACT) quartic_curve_9_0
	$(EXTRACT) quartic_curve_9_0_report
	$(EXTRACT) quartic_curve_9_0_normal_form_report
	$(EXTRACT) quartic_curve_edge_q9
	$(EXTRACT) surface_7_0_make_quartic_curves
	$(EXTRACT) surface_9_1_make_quartic_curves
	$(EXTRACT) quartic_curve_q9_1_clean2
	$(EXTRACT) quartic_curves_q13
	$(EXTRACT) quartic_curves_q19
	$(EXTRACT) quartic_curves_q29_po4_export
	$(EXTRACT) quartic_curves_q29_po4_flag_orbits_report
	$(EXTRACT) symbolic_equation_of_curve_q29_po4_so7
	$(EXTRACT) quartic_curve_29_121




extract_9_6:
	$(EXTRACT) quartic_curves_q7
	$(EXTRACT) quartic_curve_13_0_report
	$(EXTRACT) PG_2_13_rank_lines
	$(EXTRACT) PG_2_13_orbits_on_lines
	$(EXTRACT) quartic_curve_13_1_report
	$(EXTRACT) quartic_curves_19_report
	$(EXTRACT) quartic_curve_q9_1
	$(EXTRACT) quartic_curve_q13_f1_from_variety



extract_9_7:
	$(EXTRACT) surface_4_0_export_gap






extract_10_1:
	$(EXTRACT) PR101
	$(EXTRACT) PR101_smallest
	$(EXTRACT) PR29
	$(EXTRACT) PR31
	$(EXTRACT) PR37
	$(EXTRACT) PR_2_100
	$(EXTRACT) PR_2_1000
	$(EXTRACT) PE_number_2_1000
	$(EXTRACT) number_of_primitive_roots_10000
	$(EXTRACT) IM_3_19
	$(EXTRACT) IM_723
	$(EXTRACT) IM
	$(EXTRACT) IM_gcd
	$(EXTRACT) PM3a
	$(EXTRACT) PR_915839
	$(EXTRACT) PR_915839_check
	$(EXTRACT) DL_915839
	$(EXTRACT) sqrt_mod
	$(EXTRACT) sqrt_5_mod_11
	$(EXTRACT) sqrt_5_mod_19
	$(EXTRACT) sqrt_mod_20_31
	$(EXTRACT) order_of_2_mod_n
	$(EXTRACT) Eulerfunction_150
	$(EXTRACT) Eulerfunction_900
	$(EXTRACT) Eulerfunction_10000
	$(EXTRACT) power_function_2_mod_11
	$(EXTRACT) draw_mod_13
	$(EXTRACT) inverse_mod_a
	$(EXTRACT) smith_normal_form_1
	$(EXTRACT) smith_normal_form_2
	$(EXTRACT) jacobi_35_41
	$(EXTRACT) jacobi_33_41
	$(EXTRACT) jacobi_a
	$(EXTRACT) jacobi_5_19
	$(EXTRACT) sqrt_mod_7817
	$(EXTRACT) Chinese_remainders_A
	$(EXTRACT) Chinese_remainders_B
	$(EXTRACT) Chinese_remainders_C2
	$(EXTRACT) Chinese_remainders_C3



extract_10_2:
	$(EXTRACT) representation_on_polynomials_of_degree_3
	$(EXTRACT) representation_tetrahedral_group_on_polynomials_of_degree_3


extract_10_3:
	$(EXTRACT) EC_add
	$(EXTRACT) EC_cyclic_subgroup
	$(EXTRACT) EC_points_13
	$(EXTRACT) EC_points_199
	$(EXTRACT) EC_Koblitz_encoding
	$(EXTRACT) EC_bsgs
	$(EXTRACT) EC_bsgs_decode
	$(EXTRACT) NTRU_Alice1
	$(EXTRACT) NTRU_Alice2
	$(EXTRACT) NTRU_Alice_public_key
	$(EXTRACT) NTRU_encrypt
	$(EXTRACT) NTRU_decrypt1
	$(EXTRACT) NTRU_decrypt2
	$(EXTRACT) NTRU_decrypt3
	$(EXTRACT) NTRU_decrypt4
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



extract_11_1:
	$(EXTRACT) Allen_Gates_noise_1_percent
	$(EXTRACT) nonlinear_code_long_n7_k4
	$(EXTRACT) nonlinear_code_n7_d4
	$(EXTRACT) nonlinear_code_n7_d4_classify
	$(EXTRACT) Hamming_space_4_2_distance_matrix
	$(EXTRACT) Hamming_space_4_2_distance_matrix_draw
	$(EXTRACT) Hamming_code_macwilliams
	$(EXTRACT) code_5_2_3_diagram
	$(EXTRACT) Hamming_5_2_graph
	$(EXTRACT) Hamming_5_2_with_5_2_3_code
	$(EXTRACT) Sylvester_Hadamard_code_8


extract_11_2:
	$(EXTRACT) RM_3_1
	$(EXTRACT) code_6
	$(EXTRACT) Hamming_generator
	$(EXTRACT) simplex_code
	$(EXTRACT) Hamming_generator
	$(EXTRACT) Hamming_code
	$(EXTRACT) RM_3_1_and_codewords
	$(EXTRACT) RM_3_1_from_generator_matrix
	$(EXTRACT) RM_4_1_and_codewords
	$(EXTRACT) RM_5_1_and_codewords
	$(EXTRACT) Hamming_code_by_rows
	$(EXTRACT) Hamming_code_long
	$(EXTRACT) Hamming_code_diagram
	$(EXTRACT) Hamming_code_words
	$(EXTRACT) Hamming_weight_enumerator
	$(EXTRACT) Hamming_minimum_distance
	$(EXTRACT) Golay23_minimum_distance
	$(EXTRACT) code_Hamming_systematic
	$(EXTRACT) Hamming_RREF
	$(EXTRACT) Hamming_nullspace


extract_11_3:
	$(EXTRACT) Golay23_code_words
	$(EXTRACT) Golay23_code_diagram
	$(EXTRACT) Golay23_code_diagram_draw
	$(EXTRACT) Golay24_make_code
	$(EXTRACT) Golay24_make_aut_gens
	$(EXTRACT) Golay24_make_aut_gens_classes
	$(EXTRACT) M24_elt_2A
	$(EXTRACT) M24_elt_2B
	$(EXTRACT) M24_2A_restricted
	$(EXTRACT) M24_2B_restricted
	$(EXTRACT) Golay24_2A_fixed_subcode
	$(EXTRACT) Golay24_2B_fixed_subcode
	$(EXTRACT) Golay24_2A_fix_code
	$(EXTRACT) Golay24_2B_fix_code
	$(EXTRACT) Golay24_2A_fix_weight12
	$(EXTRACT) Golay24_2A_fix_weight12_c
	$(EXTRACT) Golay24_2A_fix_weight12_graph_24

extract_11_4:
	$(EXTRACT) CRC_3_128_10
	$(EXTRACT) crc32_test
	$(EXTRACT) crc32_test_hexdata
	$(EXTRACT) crc32_Berlekamp_matrix
	$(EXTRACT) CRC_F256_roots_771
	$(EXTRACT) CRC_F256_BCH_code_d2
	$(EXTRACT) CRC_F256_BCH_write_code_for_division_d2
	$(EXTRACT) CRC_F256_BCH_code_d3
	$(EXTRACT) CRC_F256_BCH_write_code_for_division_Bravo
	$(EXTRACT) CRC_F256_BCH_code_d7
	$(EXTRACT) CRC_F256_BCH_write_code_for_division_Charlie
	$(EXTRACT) F256_BCH_code_d16
	$(EXTRACT) F256_BCH_write_code_for_division_d16
	$(EXTRACT) F256_BCH_code_d16_division
	$(EXTRACT) F256_BCH_code_d16_error
	$(EXTRACT) F_256
	$(EXTRACT) CRC_F_16
	$(EXTRACT) nth_roots_256
	$(EXTRACT) CRC_F16_roots_51
	$(EXTRACT) CRC_F16_BCH_code_d3
	$(EXTRACT) CRC_F16_BCH_code_d3_mindist_dual
	$(EXTRACT) CRC_F16_BCH_code_d3_dual_weight_enumerator
	$(EXTRACT) CRC_F16_BCH_code_d3_projective_set
	$(EXTRACT) CRC_F16_BCH_code_d3_macwilliams
	$(EXTRACT) CRC_F16_BCH_write_code_for_division_Delta
	$(EXTRACT) crc_encode_16
	$(EXTRACT) crc_encode_32




extract_11_5:
	$(EXTRACT) RM_3_export
	$(EXTRACT) RM_3_Hamming_space_diagram
	$(EXTRACT) RM_3_draw
	$(EXTRACT) RM_3_split
	$(EXTRACT) RM_3_distance_draw
	$(EXTRACT) RM_3_algebraic_normal_form
	$(EXTRACT) RM_6
	$(EXTRACT) RM_6_genma_draw
	$(EXTRACT) RM_6_codewords_draw

extract_11_6:
	$(EXTRACT) draw_cyclotomic_mod_51_q16
	$(EXTRACT) draw_cyclotomic_mod_21_q8
	$(EXTRACT) F_8_BCH_code_d3
	$(EXTRACT) F_8_BCH_code_d4
	$(EXTRACT) F_8_BCH_code_d5
	$(EXTRACT) F_8_BCH_code_d5_minimum_distance
	$(EXTRACT) F_8_BCH_code_d7
	$(EXTRACT) F8_BCH_code_n63_d43
	$(EXTRACT) F8_BCH_code_n63_d43_minimum_distance
	$(EXTRACT) F2_BCH_code_n21
	$(EXTRACT) F7_RS_code_n6
	$(EXTRACT) F_64_again
	$(EXTRACT) BCH_15_5
	$(EXTRACT) draw_mod_31
	$(EXTRACT) PR127
	$(EXTRACT) draw_mod_127_power
	$(EXTRACT) draw_mod_251
	$(EXTRACT) draw_mod_255_cyclotomic_1
	$(EXTRACT) draw_mod_255_cyclotomic_3
	$(EXTRACT) draw_mod_255_cyclotomic_1_and_3
	$(EXTRACT) draw_mod_63_4_cyclotomic_3_6
	$(EXTRACT) BCH_F_64
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
	$(EXTRACT) BCH_F256_roots_771
	$(EXTRACT) BCH_F256_BCH_code_d16

extract_11_7:
	$(EXTRACT) RS_6_3
	$(EXTRACT) RS_6_3_weight_enumerator
	$(EXTRACT) RREF_RS_6_4_7_weight_enumerator
	$(EXTRACT) Code_RS_11
	$(EXTRACT) Code_RS_11_weight_enumerator
	$(EXTRACT) RS_7_3
	$(EXTRACT) RS_7_3_weight_enumerator
	$(EXTRACT) RS_7_3_field_reduction
	$(EXTRACT) RS_7_3_reduced_weight_enumerator


extract_11_8:
	$(EXTRACT) TTP_A_4


extract_11_9:
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



extract_11_10:
	$(EXTRACT) codes_8_4_4
	$(EXTRACT) codes_8_4_4_draw
	$(EXTRACT) codes_14_4_9_3
	$(EXTRACT) codes_15_6_6_2
	$(EXTRACT) codes_d4
	$(EXTRACT) codes_24_12_8
	$(EXTRACT) codes_24_12_8_draw
	$(EXTRACT) glynn_arc
	$(EXTRACT) five_points_in_general
	$(EXTRACT) codes_q13_12_4



extract_12_1:
	$(EXTRACT) random_permutation_12
	$(EXTRACT) Sym_4_conj_classes
	$(EXTRACT) Sym_10_conj_classes
	$(EXTRACT) Sym_15_conj_classes
	$(EXTRACT) Char_Sym_4
	$(EXTRACT) Char_Sym_5
	$(EXTRACT) Char_Sym_6
	$(EXTRACT) all_subsets_6_3
	$(EXTRACT) random_k_subsets
	$(EXTRACT) rank_k_subsets_test
	$(EXTRACT) Walsh_matrix_4
	$(EXTRACT) q_binomial_8_3_2
	$(EXTRACT) Dedekind_10_10
	$(EXTRACT) Dedekind_30_2
	$(EXTRACT) Dedekind_100_2
	$(EXTRACT) elementary_symmetric_functions_4
	$(EXTRACT) elementary_symmetric_functions_8
	$(EXTRACT) domino_portrait
	$(EXTRACT) domino_portrait_input




extract_12_2:
	$(EXTRACT) Hirschfeld_q4_from_set
	$(EXTRACT) hyperoval_16_create
	$(EXTRACT) EC_read
	$(EXTRACT) PG_3_5_assume_31_read
	$(EXTRACT) LS_AG_2_3_read
	$(EXTRACT) geo_7_3_read
	$(EXTRACT) geo_20_3_g4_save_canonical_forms
	$(EXTRACT) geo_20_3_g4_read
	$(EXTRACT) geo_pasch_read
	$(EXTRACT) geo_pasch_given
	$(EXTRACT) neumaier_graphs_16
	$(EXTRACT) neumaier_graphs_16_from_file
	$(EXTRACT) Neumaier_graph_25


extract_12_3:
	$(EXTRACT) Edge_curve_17
	$(EXTRACT) Edge_curve_17_line_type


extract_12_4:
	$(EXTRACT) geo_10_3_Desargues_configuration_c
	$(EXTRACT) Praeger_45_12_3_design_group



extract_12_5:
	$(EXTRACT) linsp6
	$(EXTRACT) linsp7
	$(EXTRACT) linsp30_pt_types
	$(EXTRACT) linsp30_pt_distribution


extract_12_6:
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
	$(EXTRACT) geo_16
	$(EXTRACT) geo_40_4_g4
	$(EXTRACT) geo_63_3_g6
	$(EXTRACT) geo_LSQ6.inc
	$(EXTRACT) geo_MOL6.inc



extract_12_7:
	$(EXTRACT) max_arc_16_4_start
	$(EXTRACT) max_arc_16_4_convert_stack_tdo
	$(EXTRACT) max_arc_16_4_refine
	$(EXTRACT) max_arc_16_4r_print

extract_12_8:
	$(EXTRACT) subspaces_GOp_6_2
	$(EXTRACT) subspaces_GOp_6_2_draw_Asup
	$(EXTRACT) subspaces_GOp_6_2_plesken_ring
	$(EXTRACT) subspaces_GOp_6_2_plesken_ring_evaluate_join
	$(EXTRACT) subspaces_GOp_6_2_make_pair_relations



extract_13_1:
	$(EXTRACT) make_graph_from_adjacency_matrix
	$(EXTRACT) make_triangle_graph
	$(EXTRACT) Cycle_graph_5
	$(EXTRACT) Cycle_graph_13
	$(EXTRACT) Chain_232
	$(EXTRACT) Paley_13_graph
	$(EXTRACT) Paley_13_graph_complement
	$(EXTRACT) Winnie_Li_graph_q16_index2
	$(EXTRACT) Sarnak_graph_29_5
	$(EXTRACT) Neumaier_graph_16
	$(EXTRACT) Neumaier_graph_16_reordered
	$(EXTRACT) tritangent_planes_graph
	$(EXTRACT) trihedral_pair_graph
	$(EXTRACT) small_graph
	$(EXTRACT) petersen
	$(EXTRACT) Johnson_6_2_0
	$(EXTRACT) Hamming_graph_3
	$(EXTRACT) Hamming_graph_7
	$(EXTRACT) Op_6_2_coll_graph
	$(EXTRACT) HJ_graph
	$(EXTRACT) Cayley_Z11_1mod3
	$(EXTRACT) Cayley_Sym4_coxeter
	$(EXTRACT) Cayley_Sym4_star
	$(EXTRACT) Queens_graph
	$(EXTRACT) Op_4_2_affine_polar_graph
	$(EXTRACT) Paulus_1
	$(EXTRACT) Paulus_15
	$(EXTRACT) Higman_Sims_graph



extract_13_2:
	$(EXTRACT) triangle_graph_properties
	$(EXTRACT) Hamming_graph_7_drg
	$(EXTRACT) Johnson_8_3_2_drg
	$(EXTRACT) Cycle_13_draw
	$(EXTRACT) Cycle_9_eigenvalues
	$(EXTRACT) Paley_13_draw
	$(EXTRACT) Paley_13_eigenvalues
	$(EXTRACT) tritangent_planes_graph_eigenvalues
	$(EXTRACT) Cayley_Z11_1mod3_eigenvalues_and_draw
	$(EXTRACT) Cayley_Sym4_coxeter_draw
	$(EXTRACT) Cayley_Sym5_coxeter_draw
	$(EXTRACT) Cayley_Sym4_star_eigenvalues_and_draw
	$(EXTRACT) graph_v5_e7.colored_graph
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
	$(EXTRACT) Neumaier_graph_16_reordered_export_csv
	$(EXTRACT) Op_4_2_affine_polar_graph_info
	$(EXTRACT) Neumaier_16_test_SRG_property
	$(EXTRACT) Paulus_1_aut
	$(EXTRACT) Paulus_15_aut
	$(EXTRACT) PG_3_2_packing_make_group_and_test
	$(EXTRACT) Petersen_graph_distance_poset
	$(EXTRACT) Petersen_graph_distance_poset_draw
	$(EXTRACT) Petersen_test_if_distance_regular
	$(EXTRACT) Cayley_Sym5_distance_poset





extract_13_3:
	$(EXTRACT) graph_classify_5
	$(EXTRACT) tournament_classify_4
	$(EXTRACT) graph_classify_8_r3
	$(EXTRACT) Symmetric_4_inversion_graph_recognize
	$(EXTRACT) Symmetric_5_inversion_graph_recognize




extract_14_1:
	$(EXTRACT) design_PG_2_2
	$(EXTRACT) design_PG_2_3
	$(EXTRACT) design_PG_2_4
	$(EXTRACT) design_elliptic_semiplane_Baker1978
	$(EXTRACT) wreath_product_designs_n4_k2_inc.txt
	$(EXTRACT) wreath_product_designs_n8_k6_inc.txt
	$(EXTRACT) design_20
	$(EXTRACT) design_600
	$(EXTRACT) LSQ_5A
	$(EXTRACT) LSQ_5B
	$(EXTRACT) LSQ_5_c


extract_14_2:
	$(EXTRACT) KM_cyclic_7
	$(EXTRACT) KM_PGGL_2_32
	$(EXTRACT) KM_PGGL_2_32_solve
	$(EXTRACT) KM_PSL_3_5
	$(EXTRACT) C5_KM_2_3
	$(EXTRACT) C5_KM_1_3
	$(EXTRACT) C5_concatenate
	$(EXTRACT) C5_solve
	$(EXTRACT) C5_design_classify
	$(EXTRACT) Witt_design_on_22_points



extract_14_3:
	$(EXTRACT) AG_2_3.inc
	$(EXTRACT) LS_AG_2_3_design_table_create
	$(EXTRACT) AG_2_3_on_designs
	$(EXTRACT) AG_2_3_stab_orb_0
	$(EXTRACT) AG_2_3_stab_orb_0_Perm840_res192
	$(EXTRACT) LS_AG_2_3_disjoint_sets_graph_and_cliques
	$(EXTRACT) LS_AG_2_3_disjoint_sets_split
	$(EXTRACT) LS_AG_2_3_export_large_sets
	$(EXTRACT) design_PG_2_3_table_create
	$(EXTRACT) design_PG_2_3_group_5
	$(EXTRACT) design_PG_2_3_group_5_sol_0


extract_14_4:
	$(EXTRACT) DD_PP4
	$(EXTRACT) DD_PP4_span_design
	$(EXTRACT) DD_PP4_design_orbit
	$(EXTRACT) PG_2_27_special


extract_14_5:
	$(EXTRACT) Praeger_design_45_12_3
	$(EXTRACT) Praeger_design_change_values
	$(EXTRACT) Praeger_45_12_3_design
	$(EXTRACT) Praeger_45_12_3_design_c



extract_15_1:
	$(EXTRACT) create_spread_9a
	$(EXTRACT) create_spread_9b
	$(EXTRACT) create_spread_25_7
	$(EXTRACT) create_spread_Rao_Rao_27
	$(EXTRACT) desarguesian_spread_in_PG_3_2
	$(EXTRACT) desarguesian_spread_in_PG_5_2
	$(EXTRACT) desarguesian_spread_in_PG_3_4
	$(EXTRACT) desarguesian_spread_in_PG_3_5
	$(EXTRACT) PG_3_2_spread_table
	$(EXTRACT) classify_spreads_4
	$(EXTRACT) classify_spreads_16_4
	$(EXTRACT) classify_spreads_25_starter_lift_case_0
	$(EXTRACT) spreads_25_starter_0_cliques
	$(EXTRACT) classify_spreads_25_starter_lift_all_cases
	$(EXTRACT) spreads_25_starter_cliques
	$(EXTRACT) classify_spreads_25_isomorph
	$(EXTRACT) classify_spreads_27_3_starter
	$(EXTRACT) classify_spreads_27_starter_lift_all_cases
	$(EXTRACT) spreads_27_starter_cliques
	$(EXTRACT) classify_spreads_27_isomorph_and_recognize
	$(EXTRACT) create_spread_27_0
	$(EXTRACT) create_spread_27_1
	$(EXTRACT) create_spread_27_2
	$(EXTRACT) create_spread_27_3
	$(EXTRACT) create_spread_27_4
	$(EXTRACT) create_spread_27_5
	$(EXTRACT) create_spread_27_6
	$(EXTRACT) PG_3_5_packing_PWP_0_dualize


extract_15_2:
	$(EXTRACT) create_translation_plane_9b
	$(EXTRACT) create_translation_plane_16_4_0
	$(EXTRACT) create_translation_plane_16_2_0
	$(EXTRACT) RREF_plane_16_2_0_rank_of_incma
	$(EXTRACT) create_translation_plane_25_14_rank
	$(EXTRACT) create_translation_plane_27_Rao_Rao
	$(EXTRACT) RREF_Rao_Rao_plane_incma_rank
	$(EXTRACT) create_translation_plane_27_p_rank_of_incidence_matrix
	$(EXTRACT) create_translation_plane_27_5_Rao_Rao
	$(EXTRACT) translation_plane_27_4_orbits
	$(EXTRACT) translation_plane_27_5_orbits
	$(EXTRACT) translation_plane_27_6_orbits


extract_15_3:
	$(EXTRACT) PG_3_3_spread_table
	$(EXTRACT) spread_table_PG_3_5_regular
	$(EXTRACT) PGL_4_3_classes
	$(EXTRACT) PG_3_3_assume_5_graph
	$(EXTRACT) PG_3_3_assume_5_fpc0_lo_cliques
	$(EXTRACT) PG_3_3_assume_5_fpc1_lo_cliques
	$(EXTRACT) PG_3_3_assume_5_fpc2_lo_cliques
	$(EXTRACT) PG_3_3_assume_5_read_solution
	$(EXTRACT) PG_3_3_assume_5_packings_classify


extract_15_4:
	$(EXTRACT) BLT_5_1
	$(EXTRACT) BLT_5_1_export_gap
	$(EXTRACT) BLT_5_Linear
	$(EXTRACT) BLT_9_0
	$(EXTRACT) BLT_9_Kantor1
	$(EXTRACT) BLT_11_0
	$(EXTRACT) BLT_11_Fisher
	$(EXTRACT) BLT_11_Mondello
	$(EXTRACT) BLT_11_FTWKB
	$(EXTRACT) BLT_13_Kantor2
	$(EXTRACT) BLT_67_4
	$(EXTRACT) BLT_67_4_export_gap
	$(EXTRACT) BLT_9_deep_search
	$(EXTRACT) BLT_11_deep_search
	$(EXTRACT) BLT_13_deep_search
	$(EXTRACT) BLT_13_classify_starter
	$(EXTRACT) BLT_13_cliques
	$(EXTRACT) BLT_13_isomorph_read_DB
	$(EXTRACT) BLT_13_isomorph_read_solutions
	$(EXTRACT) BLT_13_isomorph_stabilizer_orbits
	$(EXTRACT) BLT_13_isomorph_testing
	$(EXTRACT) blt_set_export_gap_all


extract_16_1:
	$(EXTRACT) small_graph_cliques
	$(EXTRACT) BLT_q13_graph_5_0_cliques_bw
	$(EXTRACT) BLT_q13_graph_5_0_cliques_rainbow
	$(EXTRACT) small_graph_cliques_Sajeeb
	$(EXTRACT) Paley_13_aut
	$(EXTRACT) Paley_13
	$(EXTRACT) Paley_13_cliques_classify
	$(EXTRACT) Paley_13_cliques_all
	$(EXTRACT) Neumaier_16_cliques_all
	$(EXTRACT) tritangent_planes_graph_aut
	$(EXTRACT) tritangent_planes_graph_cliques_classify
	$(EXTRACT) tritangent_planes_graph_cliques_all
	$(EXTRACT) PGO_5_2_cliques
	$(EXTRACT) Doily_disjoint_sets_graph_cliques_3
	$(EXTRACT) Doily_disjoint_sets_graph_cliques_5
	$(EXTRACT) PG_3_2_test
	$(EXTRACT) HJ_d2_c5
	$(EXTRACT) HJ64_cliques5
	$(EXTRACT) HJ64_cliques5_classify
	$(EXTRACT) Op_4_2_affine_polar_graph_cliques_4


extract_16_2:
	$(EXTRACT) BLT_11_classify_starter
	$(EXTRACT) BLT_11_clique_0
	$(EXTRACT) BLT_11_clique_0_Sajeeb
	$(EXTRACT) BLT_11_0_sort
	$(EXTRACT) BLT_11_0_sorted_draw


extract_16_3:
	$(EXTRACT) part10
	$(EXTRACT) octic_monomials
	$(EXTRACT) solve_test_system
	$(EXTRACT) McKay_test
	$(EXTRACT) DLX_test
	$(EXTRACT) solve_praeger_design_line_type_system





extract_17_1:

extract_17_2:
	$(EXTRACT) quadrangles_PG_2_2
	$(EXTRACT) Barth_mod_11_singular_pts_c
	$(EXTRACT) quartic_curves_q13_classify_nauty
	$(EXTRACT) quartic_curves_q13_filter_isomorphism_reps
	$(EXTRACT) quartic_curves_q13_tally_ago
	$(EXTRACT) Klein_cubic_threefold_q3
	$(EXTRACT) EC_canon
	$(EXTRACT) Hirschfeld_q4_c
	$(EXTRACT) Hirschfeld_stab_group
	$(EXTRACT) Hirschfeld_stab_subgroup
	$(EXTRACT) Hirschfeld_stab_orbits
	$(EXTRACT) Hirschfeld_q4_set_c
	$(EXTRACT) Dickson_sets_stabilizer
	$(EXTRACT) Endrass_7c
	$(EXTRACT) hyperoval_16_canonical_form
	$(EXTRACT) cubic_curves_PG_2_8_canon
	$(EXTRACT) ovoid_q8_canon
	$(EXTRACT) ovoid_ST_q8_canon
	$(EXTRACT) Suzuki_8
	$(EXTRACT) quartic_curve_25_0_0_canonical
	$(EXTRACT) CRC_BCH_Q16_N51_D3_canonical
	$(EXTRACT) Veronese_1_3_q7_c


extract_17_3:
	$(EXTRACT) geo_7_3_c
	$(EXTRACT) geo_10_3_c
	$(EXTRACT) geo_10_3_c_lex_least
	$(EXTRACT) geo_14_3_c
	$(EXTRACT) geo_15_3_c_filter72
	$(EXTRACT) geo_15_3_filtered_c
	$(EXTRACT) TFC_24_3_c
	$(EXTRACT) geo_40_4_g4_c
	$(EXTRACT) geo_17_3_g4_c
	$(EXTRACT) AG_2_3_c
	$(EXTRACT) geo_LSQ6_c


extract_17_4:
	$(EXTRACT) design_27c
	$(EXTRACT) design_PG_2_3_canonical
	$(EXTRACT) wreath_product_designs_n8_k6_c
	$(EXTRACT) wreath_product_designs_n4_k2_c
	$(EXTRACT) LS_AG_2_3_solutions_classify


extract_17_5:
	$(EXTRACT) code_3_2_aut
	$(EXTRACT) code_6_3_aut
	$(EXTRACT) RM_3_1_group
	$(EXTRACT) RM_3_1_group_and_diagram
	$(EXTRACT) RM_4_1_group
	$(EXTRACT) RS_6_4_7_group
	$(EXTRACT) GV_n15_k6_d5_group
	$(EXTRACT) code_n15_k6_d6_a_group
	$(EXTRACT) code_n15_k6_d6_b_group


extract_17_6:
	$(EXTRACT) Hamming_graph_7_with_Hamming_code


extract_17_7:
	$(EXTRACT) Cycle_13_aut
	$(EXTRACT) Neumaier_16_aut
	$(EXTRACT) Queens_graph_aut
	$(EXTRACT) Queens_graph_aut_orbits_draw
	$(EXTRACT) inversion_graph
	$(EXTRACT) Chain_232_aut
	$(EXTRACT) JK_graph_pp16_1
	$(EXTRACT) JK_graph_pp16_2
	$(EXTRACT) JK_graph_pp16_9
	$(EXTRACT) JK_graph_grid_3_3
	$(EXTRACT) JK_graph_sts_13
	$(EXTRACT) HJ_group
	$(EXTRACT) HJ_group_and_orbits
	$(EXTRACT) HJ_orbital_graph_3
	$(EXTRACT) Group_Perm315_Orbital_3.colored_graph
	$(EXTRACT) HJ_d2_graph
	$(EXTRACT) PGO_5_2_graph_group
	$(EXTRACT) Paulus_c
	$(EXTRACT) neumaier_graphs_16_classify
	$(EXTRACT) Neumaier_graph_25_c



extract_17_8:
	$(EXTRACT) F_17_edge
	$(EXTRACT) Edge_curve_17_nauty
	$(EXTRACT) Edge_curve_17_group


extract_18_1:
	$(EXTRACT) Sym_6
	$(EXTRACT) Sym_6_classes
	$(EXTRACT) Sym_6_Class_2A
	$(EXTRACT) Sym_6_Class_2B
	$(EXTRACT) Sym_6_Class_2C
	$(EXTRACT) Sym_6_concatenate_classes
	$(EXTRACT) Sym_6_class_2ABC_order_of_pairs
	$(EXTRACT) Sym_6_order_of_products_change_values
	$(EXTRACT) Sym_6_class_2ABC_make_graph_G2
	$(EXTRACT) Sym_6_class_2ABC_make_graph_G3
	$(EXTRACT) Sym_6_class_2ABC_subgraph_A5
	$(EXTRACT) Sym_6_aut_group
	$(EXTRACT) Sym_6_Schreier_tree_draw
	$(EXTRACT) Aut_Sym_6_group
	$(EXTRACT) Aut_Sym_6_subgroup_lattice
	$(EXTRACT) Aut_Sym_6_subgroup_lattice_load_and_draw


extract_19_1:
	$(EXTRACT) PGL_6_2_Wedge_4_2_detached_report.tex
	$(EXTRACT) schreier_tree_graphical_output
	$(EXTRACT) Queens_graph_again

extract_19_2:
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

extract_19_3:
	$(EXTRACT) dode
	$(EXTRACT) dode_video
	$(EXTRACT) monkey_video
	$(EXTRACT) Eckardt_deform_video
	$(EXTRACT) Eckardt_surface
	$(EXTRACT) Kummer_surface
	$(EXTRACT) Kummer_video
	$(EXTRACT) Beauville_surface
	$(EXTRACT) Clebsch_up_create_points
	$(EXTRACT) Clebsch_surface
	$(EXTRACT) Clebsch_surface_defining_equation
	$(EXTRACT) Clebsch_surface_defining_equation_and_curves
	$(EXTRACT) F7_povray
	$(EXTRACT) F7_video
	$(EXTRACT) McKean_povray
	$(EXTRACT) McKean_video


extract_19_4:
	$(EXTRACT) lissajous
	$(EXTRACT) lissajous_plot
	$(EXTRACT) lissajous_3d
	$(EXTRACT) lissajous_3d_plot


extract_19_5:
	$(EXTRACT) gnuplot_test_data
	$(EXTRACT) gnuplot_Eulerfunction_50


extract_20_1:
	$(EXTRACT) make_table_of_surfaces
	$(EXTRACT) make_table_of_quartic_curves
	$(EXTRACT) cubic_surfaces_table_q17
	$(EXTRACT) quartic_curves_table_q17


extract_20_2:
	$(EXTRACT) blt_set_table_q7
	$(EXTRACT) blt_set_table_q9
	$(EXTRACT) blt_set_table_q13
	$(EXTRACT) blt_set_table
	$(EXTRACT) blt_set_export_gap



extract_21_1:
	$(EXTRACT) misc_select
	$(EXTRACT) misc_join
	$(EXTRACT) table_mod_12

extract_21_2:
	$(EXTRACT) F_15bit
	$(EXTRACT) F_31bit
	$(EXTRACT) F_32bit
	$(EXTRACT) F_63bit
	$(EXTRACT) F_64bit


unclassified:
	$(EXTRACT) stabilizer_of_mmx




