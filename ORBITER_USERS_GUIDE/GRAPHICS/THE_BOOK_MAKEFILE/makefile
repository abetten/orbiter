MY_PATH=~/orbiter

#MY_PATH=/scratch/betten/COMPILE/orbiter


# uncomment exactly one of the following two lines.
# uncomment the first if you want to run orbiter through docker.
# uncomment the second if you have an installed copy of orbiter and you want to run it directly 
#ORBITER_PATH=docker run -it --volume ${PWD}:/mnt -w /mnt abetten/orbiter 
ORBITER_PATH=$(MY_PATH)/src/apps/orbiter/

EXTRACT=./extract.sh $(ORBITER_PATH) ~/bin/  ":" the_book_makefile EXTRACTS

EXTRACT_DEF=./extract.sh $(ORBITER_PATH) ~/bin/ "=" the_book_makefile EXTRACTS

all:
	cp ~/DATA/DATA.24/BOOK/makefile ./the_book_makefile
	~/bin/a2tex.out -numbers -text_width 60 <the_book_makefile >the_book_makefile.tex
	cp ../THE_MAKEFILE/EXTRACTS/* EXTRACTS

no:
	$(EXTRACT) Cayley_Sym4_coxeter
	$(EXTRACT_DEF) BARTH_EQUATION_GENERAL
	$(EXTRACT) sqrt_5_mod_11
	$(EXTRACT) Barth_mod_11_phi
	$(EXTRACT) PGL_4_4
	$(EXTRACT) PGGL_4_4
	$(EXTRACT) PGGLstar_4_4
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
	$(EXTRACT) Aut_Sym_6_group
	$(EXTRACT) Aut_Sym_6_subgroup_lattice
	$(EXTRACT) Aut_Sym_6_subgroup_lattice_load_and_draw
