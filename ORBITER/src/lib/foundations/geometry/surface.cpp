// surface.C
// 
// Anton Betten
// Jul 25, 2016
//
// 
//
//

#include "foundations.h"

surface::surface()
{
	null();
}

surface::~surface()
{
	freeself();
}

void surface::freeself()
{
	INT f_v = FALSE;

	if (f_v) {
		cout << "surface::freeself" << endl;
		}
	if (v) {
		FREE_INT(v);
		}
	if (v2) {
		FREE_INT(v2);
		}
	if (w2) {
		FREE_INT(w2);
		}
	if (P) {
		FREE_OBJECT(P);
		}
	if (P2) {
		FREE_OBJECT(P2);
		}
	if (Gr) {
		FREE_OBJECT(Gr);
		}
	if (Gr3) {
		FREE_OBJECT(Gr3);
		}
	if (O) {
		FREE_OBJECT(O);
		}
	if (Klein) {
		FREE_OBJECT(Klein);
		}
	if (Sets) {
		FREE_INT(Sets);
		}
	if (M) {
		FREE_INT(M);
		}
	if (Sets2) {
		FREE_INT(Sets2);
		}
	if (Pts) {
		FREE_INT(Pts);
		}
	if (pt_list) {
		FREE_INT(pt_list);
		}
	if (System) {
		FREE_INT(System);
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	if (f_v) {
		cout << "before FREE_pchar(Line_label);" << endl;
		}
	if (Line_label) {
		INT i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label[i]);
			}
		FREE_pchar(Line_label);
		}
	if (Line_label_tex) {
		INT i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label_tex[i]);
			}
		FREE_pchar(Line_label_tex);
		}
	if (Eckard_point_label) {
		INT i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label[i]);
			}
		FREE_pchar(Eckard_point_label);
		}
	if (Eckard_point_label_tex) {
		INT i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label_tex[i]);
			}
		FREE_pchar(Eckard_point_label_tex);
		}
	if (f_v) {
		cout << "before FREE_INT(Trihedral_pairs);" << endl;
		}
	if (Trihedral_pairs) {
		FREE_INT(Trihedral_pairs);
		}
	if (Trihedral_pair_labels) {
		INT i;
		
		for (i = 0; i < nb_trihedral_pairs; i++) {
			FREE_char(Trihedral_pair_labels[i]);
			}
		FREE_pchar(Trihedral_pair_labels);
		}
	if (Trihedral_pairs_row_sets) {
		FREE_INT(Trihedral_pairs_row_sets);
		}
	if (Trihedral_pairs_col_sets) {
		FREE_INT(Trihedral_pairs_col_sets);
		}
	if (f_v) {
		cout << "before FREE_OBJECT Classify_trihedral_pairs_row_values;" << endl;
		}
	if (Classify_trihedral_pairs_row_values) {
		FREE_OBJECT(Classify_trihedral_pairs_row_values);
		}
	if (Classify_trihedral_pairs_col_values) {
		FREE_OBJECT(Classify_trihedral_pairs_col_values);
		}
	if (Eckardt_points) {
		FREE_OBJECTS(Eckardt_points);
		}
	if (Trihedral_to_Eckardt) {
		FREE_INT(Trihedral_to_Eckardt);
		}
	if (collinear_Eckardt_triples_rank) {
		FREE_INT(collinear_Eckardt_triples_rank);
		}
	if (Classify_collinear_Eckardt_triples) {
		FREE_OBJECT(Classify_collinear_Eckardt_triples);
		}
	if (Poly1) {
		FREE_OBJECT(Poly1);
		}
	if (Poly2) {
		FREE_OBJECT(Poly2);
		}
	if (Poly3) {
		FREE_OBJECT(Poly3);
		}
	if (Poly1_x123) {
		FREE_OBJECT(Poly1_x123);
		}
	if (Poly2_x123) {
		FREE_OBJECT(Poly2_x123);
		}
	if (Poly3_x123) {
		FREE_OBJECT(Poly3_x123);
		}
	if (Poly4_x123) {
		FREE_OBJECT(Poly4_x123);
		}
	if (Poly1_4) {
		FREE_OBJECT(Poly1_4);
		}
	if (Poly2_4) {
		FREE_OBJECT(Poly2_4);
		}
	if (Poly3_4) {
		FREE_OBJECT(Poly3_4);
		}
	if (Double_six) {
		FREE_INT(Double_six);
		}
	if (Double_six_label_tex) {
		INT i;
		
		for (i = 0; i < 36; i++) {
			FREE_char(Double_six_label_tex[i]);
			}
		FREE_pchar(Double_six_label_tex);
		}
	if (Half_double_sixes) {
		FREE_INT(Half_double_sixes);
		}

	if (Half_double_six_label_tex) {
		INT i;
		
		for (i = 0; i < 72; i++) {
			FREE_char(Half_double_six_label_tex[i]);
			}
		FREE_pchar(Half_double_six_label_tex);
		}

	if (Half_double_six_to_double_six) {
		FREE_INT(Half_double_six_to_double_six);
		}
	if (Half_double_six_to_double_six_row) {
		FREE_INT(Half_double_six_to_double_six_row);
		}
	if (f_has_large_polynomial_domains) {
		if (Poly2_27) {
			FREE_OBJECT(Poly2_27);
			}
		if (Poly4_27) {
			FREE_OBJECT(Poly4_27);
			}
		if (Poly6_27) {
			FREE_OBJECT(Poly6_27);
			}
		if (Poly3_24) {
			FREE_OBJECT(Poly3_24);
			}
		}
	if (Clebsch_Pij) {
		FREE_INT(Clebsch_Pij);
		}
	if (Clebsch_P) {
		FREE_PINT(Clebsch_P);
		}
	if (Clebsch_P3) {
		FREE_PINT(Clebsch_P3);
		}
	if (Clebsch_coeffs) {
		FREE_INT(Clebsch_coeffs);
		}
	if (CC) {
		FREE_PINT(CC);
		}
	null();
	if (f_v) {
		cout << "surface::freeself done" << endl;
		}
}

void surface::null()
{
	v = NULL;
	v2 = NULL;
	w2 = NULL;
	P = NULL;
	P2 = NULL;
	Gr = NULL;
	Gr3 = NULL;
	O = NULL;
	Klein = NULL;
	Sets = NULL;
	M = NULL;
	Sets2 = NULL;

	Pts = NULL;
	pt_list = NULL;
	System = NULL;
	base_cols = NULL;

	Line_label = NULL;
	Line_label_tex = NULL;
	Trihedral_pairs = NULL;
	Trihedral_pair_labels = NULL;
	nb_trihedral_pairs = 0;
	Trihedral_pairs_row_sets = NULL;
	Trihedral_pairs_col_sets = NULL;
	Classify_trihedral_pairs_row_values = NULL;
	Classify_trihedral_pairs_col_values = NULL;
	nb_Eckardt_points = 0;
	Eckardt_points = NULL;
	Eckard_point_label = NULL;
	Eckard_point_label_tex = NULL;
	Trihedral_to_Eckardt = NULL;
	collinear_Eckardt_triples_rank = NULL;
	Classify_collinear_Eckardt_triples = NULL;
	Poly1 = NULL;
	Poly2 = NULL;
	Poly3 = NULL;
	Poly1_x123 = NULL;
	Poly2_x123 = NULL;
	Poly3_x123 = NULL;
	Poly4_x123 = NULL;
	Poly1_4 = NULL;
	Poly2_4 = NULL;
	Poly3_4 = NULL;
	Double_six = NULL;
	Double_six_label_tex = NULL;
	Half_double_sixes = NULL;
	Half_double_six_label_tex = NULL;
	Half_double_six_to_double_six = NULL;
	Half_double_six_to_double_six_row = NULL;

	f_has_large_polynomial_domains = FALSE;
	Poly2_27 = NULL;
	Poly4_27 = NULL;
	Poly6_27 = NULL;
	Poly3_24 = NULL;

	Clebsch_Pij = NULL;
	Clebsch_P = NULL;
	Clebsch_P3 = NULL;
	Clebsch_coeffs = NULL;
	CC = NULL;
}

void surface::init(finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface::init" << endl;
		}
	
	n = 4;
	n2 = 2 * n;
	surface::F = F;
	q = F->q;
	nb_pts_on_surface = q * q + 7 * q + 1;
	if (f_v) {
		cout << "surface::init nb_pts_on_surface = " << nb_pts_on_surface << endl;
		}

	v = NEW_INT(n);
	v2 = NEW_INT(6);
	w2 = NEW_INT(6);
	
	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "surface::init before P->init" << endl;
		}
	P->init(3, F, 
		TRUE /*f_init_incidence_structure */, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "surface::init after P->init" << endl;
		}

	P2 = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "surface::init before P2->init" << endl;
		}
	P2->init(2, F, 
		TRUE /*f_init_incidence_structure */, 
		0 /*verbose_level*/);
	if (f_v) {
		cout << "surface::init after P2->init" << endl;
		}

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, 2, F, 0 /* verbose_level */);
	nb_lines_PG_3 = Gr->nCkq.as_INT();
	if (f_v) {
		cout << "surface::init nb_lines_PG_3 = " << nb_lines_PG_3 << endl;
		}

	Gr3 = NEW_OBJECT(grassmann);
	Gr3->init(4, 3, F, 0 /* verbose_level*/);


	if (f_v) {
		cout << "surface::init initializing orthogonal" << endl;
		}
	O = NEW_OBJECT(orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface::init initializing orthogonal done" << endl;
		}

	Klein = NEW_OBJECT(klein_correspondence);

	if (f_v) {
		cout << "surface::init initializing Klein correspondence" << endl;
		}
	Klein->init(F, O, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface::init initializing Klein correspondence done" << endl;
		}



	if (f_v) {
		cout << "surface::init before init_polynomial_domains" << endl;
		}
	init_polynomial_domains(verbose_level);
	if (f_v) {
		cout << "surface::init after init_polynomial_domains" << endl;
		}

	//init_large_polynomial_domains(verbose_level);

	if (f_v) {
		cout << "surface::init before init_system" << endl;
		}
	init_system(verbose_level);
	if (f_v) {
		cout << "surface::init after init_system" << endl;
		}


	if (f_v) {
		cout << "surface::init before init_line_data" << endl;
		}
	init_line_data(verbose_level);
	if (f_v) {
		cout << "surface::init after init_line_data" << endl;
		}

	if (f_v) {
		cout << "surface::init before make_trihedral_pairs" << endl;
		}
	make_trihedral_pairs(Trihedral_pairs, 
		Trihedral_pair_labels, nb_trihedral_pairs, 
		verbose_level);
	if (f_v) {
		cout << "surface::init after make_trihedral_pairs" << endl;
		}
	
	if (f_v) {
		cout << "surface::init before process_trihedral_pairs" << endl;
		}
	process_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "surface::init after process_trihedral_pairs" << endl;
		}

	if (f_v) {
		cout << "surface::init before make_Eckardt_points" << endl;
		}
	make_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface::init after make_Eckardt_points" << endl;
		}

	if (f_v) {
		cout << "surface::init before init_Trihedral_to_Eckardt" << endl;
		}
	init_Trihedral_to_Eckardt(verbose_level);
	if (f_v) {
		cout << "surface::init after init_Trihedral_to_Eckardt" << endl;
		}

	if (f_v) {
		cout << "surface::init before init_collinear_Eckardt_triples" << endl;
		}
	init_collinear_Eckardt_triples(verbose_level);
	if (f_v) {
		cout << "surface::init after init_collinear_Eckardt_triples" << endl;
		}

	if (f_v) {
		cout << "surface::init before init_double_sixes" << endl;
		}
	init_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface::init after init_double_sixes" << endl;
		}

	if (f_v) {
		cout << "surface::init before create_half_double_sixes" << endl;
		}
	create_half_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface::init after create_half_double_sixes" << endl;
		}

	//clebsch_cubics(verbose_level);

	if (f_v) {
		cout << "surface::init done" << endl;
		}
}


void surface::init_polynomial_domains(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface::init_polynomial_domains" << endl;
		}
	Poly1 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(homogeneous_polynomial_domain);

	Poly1->init(F, 3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly2->init(F, 3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly3->init(F, 3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);

	Poly1_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly1_x123->init(F, 3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly2_x123->init(F, 3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly3_x123->init(F, 3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly4_x123->init(F, 3 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);


	label_variables_3(Poly1, 0 /* verbose_level */);
	label_variables_3(Poly2, 0 /* verbose_level */);
	label_variables_3(Poly3, 0 /* verbose_level */);

	label_variables_x123(Poly1_x123, 0 /* verbose_level */);
	label_variables_x123(Poly2_x123, 0 /* verbose_level */);
	label_variables_x123(Poly3_x123, 0 /* verbose_level */);
	label_variables_x123(Poly4_x123, 0 /* verbose_level */);

	Poly1_4 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2_4 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_4 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly1_4->init(F, 4 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly2_4->init(F, 4 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly3_4->init(F, 4 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);

	label_variables_4(Poly1_4, 0 /* verbose_level */);
	label_variables_4(Poly2_4, 0 /* verbose_level */);
	label_variables_4(Poly3_4, 0 /* verbose_level */);

	nb_monomials = Poly3_4->nb_monomials;

	if (f_v) {
		cout << "nb_monomials = " << nb_monomials << endl;
		}
	if (f_v) {
		cout << "surface::init_polynomial_domains done" << endl;
		}
}

void surface::init_large_polynomial_domains(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface::init_large_polynomial_domains" << endl;
		}
	f_has_large_polynomial_domains = TRUE;
	Poly2_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly6_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_24 = NEW_OBJECT(homogeneous_polynomial_domain);

	Poly2_27->init(F, 27 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly4_27->init(F, 27 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly6_27->init(F, 27 /* nb_vars */, 6 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);
	Poly3_24->init(F, 24 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */, verbose_level);

	nb_monomials2 = Poly2_27->nb_monomials;
	nb_monomials4 = Poly4_27->nb_monomials;
	nb_monomials6 = Poly6_27->nb_monomials;
	nb_monomials3 = Poly3_24->nb_monomials;

	label_variables_27(Poly2_27, 0 /* verbose_level */);
	label_variables_27(Poly4_27, 0 /* verbose_level */);
	label_variables_27(Poly6_27, 0 /* verbose_level */);
	label_variables_24(Poly3_24, 0 /* verbose_level */);

	if (f_v) {
		cout << "nb_monomials2 = " << nb_monomials2 << endl;
		cout << "nb_monomials4 = " << nb_monomials4 << endl;
		cout << "nb_monomials6 = " << nb_monomials6 << endl;
		cout << "nb_monomials3 = " << nb_monomials3 << endl;
		}

	if (f_v) {
		cout << "surface::init_large_polynomial_domains before clebsch_cubics" << endl;
		}
	clebsch_cubics(verbose_level - 1);
	if (f_v) {
		cout << "surface::init_large_polynomial_domains after clebsch_cubics" << endl;
		}

	if (f_v) {
		cout << "surface::init_large_polynomial_domains done" << endl;
		}
}

void surface::label_variables_3(homogeneous_polynomial_domain *HPD, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface::label_variables_3" << endl;
		}
	if (HPD->n != 3) {
		cout << "surface::label_variables_3 HPD->n != 3" << endl;
		exit(1);
		}
	if (HPD->symbols) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols[i]);
			}
		FREE_pchar(HPD->symbols);
		}
	if (HPD->symbols_latex) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols_latex[i]);
			}
		FREE_pchar(HPD->symbols_latex);
		}
	HPD->symbols = NEW_pchar(3);
	HPD->symbols_latex = NEW_pchar(3);
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_%ld", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_{%ld}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface::label_variables_3 done" << endl;
		}
	
}

void surface::label_variables_x123(homogeneous_polynomial_domain *HPD, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface::label_variables_x123" << endl;
		}
	if (HPD->n != 3) {
		cout << "surface::label_variables_x123 HPD->n != 3" << endl;
		exit(1);
		}
	if (HPD->symbols) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols[i]);
			}
		FREE_pchar(HPD->symbols);
		}
	if (HPD->symbols_latex) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols_latex[i]);
			}
		FREE_pchar(HPD->symbols_latex);
		}
	HPD->symbols = NEW_pchar(3);
	HPD->symbols_latex = NEW_pchar(3);
	for (i = 0; i < 3; i++) {
		sprintf(label, "x_%ld", i + 1);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "x_{%ld}", i + 1);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface::label_variables_x123 done" << endl;
		}
	
}

void surface::label_variables_4(homogeneous_polynomial_domain *HPD, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface::label_variables_4" << endl;
		}
	if (HPD->n != 4) {
		cout << "surface::label_variables_4 HPD->n != 4" << endl;
		exit(1);
		}
	if (HPD->symbols) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols[i]);
			}
		FREE_pchar(HPD->symbols);
		}
	if (HPD->symbols_latex) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols_latex[i]);
			}
		FREE_pchar(HPD->symbols_latex);
		}
	HPD->symbols = NEW_pchar(4);
	HPD->symbols_latex = NEW_pchar(4);
	for (i = 0; i < 4; i++) {
		sprintf(label, "x_%ld", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 4; i++) {
		sprintf(label, "x_{%ld}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface::label_variables_4 done" << endl;
		}
	
}

void surface::label_variables_27(homogeneous_polynomial_domain *HPD, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface::label_variables_27" << endl;
		}
	if (HPD->n != 27) {
		cout << "surface::label_variables_27 HPD->n != 27" << endl;
		exit(1);
		}
	if (HPD->symbols) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols[i]);
			}
		FREE_pchar(HPD->symbols);
		}
	if (HPD->symbols_latex) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols_latex[i]);
			}
		FREE_pchar(HPD->symbols_latex);
		}
	HPD->symbols = NEW_pchar(27);
	HPD->symbols_latex = NEW_pchar(27);
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_%ld", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_%ld%ld", i, j);
			l = strlen(label);
			HPD->symbols[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_%ld%ld", i, j);
			l = strlen(label);
			HPD->symbols[3 + 12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + 12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_{%ld}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_{%ld%ld}", i, j);
			l = strlen(label);
			HPD->symbols_latex[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_{%ld%ld}", i, j);
			l = strlen(label);
			HPD->symbols_latex[3 + 12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[3 + 12 + i * 4 + j], label);
			}
		}
	if (f_v) {
		cout << "surface::label_variables_27 done" << endl;
		}
	
}

void surface::label_variables_24(homogeneous_polynomial_domain *HPD, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface::label_variables_24" << endl;
		}
	if (HPD->n != 24) {
		cout << "surface::label_variables_24 HPD->n != 24" << endl;
		exit(1);
		}
	if (HPD->symbols) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols[i]);
			}
		FREE_pchar(HPD->symbols);
		}
	if (HPD->symbols_latex) {
		for (i = 0; i < HPD->n; i++) {
			FREE_char(HPD->symbols_latex[i]);
			}
		FREE_pchar(HPD->symbols_latex);
		}
	HPD->symbols = NEW_pchar(24);
	HPD->symbols_latex = NEW_pchar(24);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_%ld%ld", i, j);
			l = strlen(label);
			HPD->symbols[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_%ld%ld", i, j);
			l = strlen(label);
			HPD->symbols[12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_{%ld%ld}", i, j);
			l = strlen(label);
			HPD->symbols_latex[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_{%ld%ld}", i, j);
			l = strlen(label);
			HPD->symbols_latex[12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[12 + i * 4 + j], label);
			}
		}
	if (f_v) {
		cout << "surface::label_variables_24 done" << endl;
		}
	
}

void surface::init_system(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface::init_system" << endl;
		}

	max_pts = 27 * (q + 1);
	Pts = NEW_INT(max_pts * n);
	pt_list = NEW_INT(max_pts);
	System = NEW_INT(max_pts * nb_monomials);
	base_cols = NEW_INT(nb_monomials);
	
	if (f_v) {
		cout << "surface::init_system done" << endl;
		}

}

void surface::init_line_data(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	INT i, j, h, h2;
	
	if (f_v) {
		cout << "surface::init_line_data" << endl;
		}

	Sets = NEW_INT(30 * 2);
	M = NEW_INT(6 * 6);
	INT_vec_zero(M, 6 * 6);

	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
				}
			M[i * 6 + j] = h;
			Sets[h * 2 + 0] = i;
			Sets[h * 2 + 1] = 6 + j;
			h++;
			}
		}
	

	if (h != 30) {
		cout << "h != 30" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "surface::init_line_data Sets:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Sets, 30, 2, FALSE /* f_tex */);
		//INT_matrix_print(Sets, 30, 2);
		}


	Sets2 = NEW_INT(15 * 2);
	h2 = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			Sets2[h2 * 2 + 0] = M[i * 6 + j];
			Sets2[h2 * 2 + 1] = M[j * 6 + i];
			h2++;
			}
		}
	if (h2 != 15) {
		cout << "h2 != 15" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "Sets2:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Sets2, 15, 2, FALSE /* f_tex */);
		//INT_matrix_print(Sets2, 15, 2);
		}

	Line_label = NEW_pchar(27);
	Line_label_tex = NEW_pchar(27);
	char str[1000];
	INT a, b, c, l;

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			sprintf(str, "a_%ld", i + 1);
			}
		else if (i < 12) {
			sprintf(str, "b_%ld", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			sprintf(str, "c_{%ld%ld}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str 
				<< " for line " << i << endl;
			}
		l = strlen(str);
		Line_label[i] = NEW_char(l + 1);
		strcpy(Line_label[i], str);
		}

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			sprintf(str, "a_{%ld}", i + 1);
			}
		else if (i < 12) {
			sprintf(str, "b_{%ld}", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			sprintf(str, "c_{%ld%ld}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str 
				<< " for line " << i << endl;
			}
		l = strlen(str);
		Line_label_tex[i] = NEW_char(l + 1);
		strcpy(Line_label_tex[i], str);
		}

	if (f_v) {
		cout << "surface::init_line_data done" << endl;
		}
}


INT surface::index_of_monomial(INT *v)
{
	return Poly3_4->index_of_monomial(v);
}

void surface::print_equation(ostream &ost, INT *coeffs)
{
	Poly3_4->print_equation(ost, coeffs);
}

void surface::print_equation_tex(ostream &ost, INT *coeffs)
{
	Poly3_4->print_equation(ost, coeffs);
}

void surface::unrank_point(INT *v, INT rk)
{
	P->unrank_point(v, rk);
}

INT surface::rank_point(INT *v)
{
	INT rk;

	rk = P->rank_point(v);
	return rk;
}

void surface::unrank_line(INT *v, INT rk)
{
	Gr->unrank_INT_here(v, rk, 0 /* verbose_level */);
}

void surface::unrank_lines(INT *v, INT *Rk, INT nb)
{
	INT i;
	
	for (i = 0; i < nb; i++) {
		Gr->unrank_INT_here(v + i * 8, Rk[i], 0 /* verbose_level */);
		}
}

INT surface::rank_line(INT *v)
{
	INT rk;

	rk = Gr->rank_INT_here(v, 0 /* verbose_level */);
	return rk;
}

void surface::unrank_plane(INT *v, INT rk)
{
	Gr3->unrank_INT_here(v, rk, 0 /* verbose_level */);
}

INT surface::rank_plane(INT *v)
{
	INT rk;

	rk = Gr3->rank_INT_here(v, 0 /* verbose_level */);
	return rk;
}

void surface::build_cubic_surface_from_lines(INT len, INT *S, 
	INT *coeff, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT r;

	if (f_v) {
		cout << "surface::build_cubic_surface_from_lines" << endl;
		}
	r = compute_system_in_RREF(len, S, verbose_level);
	if (r != nb_monomials - 1) {
		cout << "surface::build_cubic_surface_from_lines r != nb_monomials - 1" << endl;
		cout << "r=" << r << endl;
		exit(1);
		}

	INT kernel_m, kernel_n;

	F->matrix_get_kernel(System, r, nb_monomials, base_cols, r, 
		kernel_m, kernel_n, coeff);

	//cout << "kernel_m=" << kernel_m << endl;
	//cout << "kernel_n=" << kernel_n << endl;
	if (f_v) {
		cout << "surface::build_cubic_surface_from_lines done" << endl;
		}
}

INT surface::compute_system_in_RREF(INT len, INT *S, INT verbose_level)
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	INT i, j, nb_pts, a, r;

	if (f_v) {
		cout << "surface::compute_system_in_RREF" << endl;
		}
	if (len > 27) {
		cout << "surface::compute_system_in_RREF len > 27" << endl;
		exit(1);
		}

	nb_pts = 0;
	for (i = 0; i < len; i++) {
		a = S[i];

		if (P->Lines) {
			for (j = 0; j < P->k; j++) {
				pt_list[nb_pts++] = P->Lines[a * P->k + j];
				}
			}
		else {
			P->create_points_on_line(a, 
				pt_list + nb_pts, 
				0 /* verbose_level */);
			nb_pts += P->k;
			}
		}

	if (nb_pts > max_pts) {
		cout << "surface::compute_system_in_RREF nb_pts > max_pts" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "surface::compute_system_in_RREF list of covered points by lines:" << endl;
		INT_matrix_print(pt_list, len, P->k);
		}
	for (i = 0; i < nb_pts; i++) {
		unrank_point(Pts + i * n, pt_list[i]);
		}
	if (f_v && FALSE) {
		cout << "surface::compute_system_in_RREF list of covered points in coordinates:" << endl;
		INT_matrix_print(Pts, nb_pts, n);
		}

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < nb_monomials; j++) {
			System[i * nb_monomials + j] = 
				F->evaluate_monomial(
					Poly3_4->Monomials + j * n, 
					Pts + i * n, n);
			}
		}
	if (f_v && FALSE) {
		cout << "surface::compute_system_in_RREF The system:" << endl;
		INT_matrix_print(System, nb_pts, nb_monomials);
		}
	r = F->Gauss_simple(System, nb_pts, nb_monomials, 
		base_cols, 0 /* verbose_level */);
	if (FALSE) {
		cout << "surface::compute_system_in_RREF The system in RREF:" << endl;
		INT_matrix_print(System, nb_pts, nb_monomials);
		}
	if (f_v) {
		cout << "surface::compute_system_in_RREF The system has rank " << r << endl;
		}
	return r;
}

INT surface::test(INT len, INT *S, INT verbose_level)
{
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	INT r, ret;

	if (f_v) {
		cout << "surface::test" << endl;
		}

	r = compute_system_in_RREF(len, S, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface::test The system has rank " << r << endl;
		}
	if (r < nb_monomials) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	if (f_v) {
		cout << "surface::test done ret = " << ret << endl;
		}
	return ret;
}

void surface::enumerate_points(INT *coeff, 
	INT *Pts, INT &nb_pts, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface::enumerate_points" << endl;
		}

	Poly3_4->enumerate_points(coeff, Pts, nb_pts, verbose_level);
	if (f_v) {
		cout << "surface::enumerate_points done" << endl;
		}
}

void surface::substitute_semilinear(INT *coeff_in, INT *coeff_out, 
	INT f_semilinear, INT frob, INT *Mtx_inv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface::substitute_semilinear" << endl;
		}
	Poly3_4->substitute_semilinear(coeff_in, coeff_out, 
		f_semilinear, frob, Mtx_inv, verbose_level);
	if (f_v) {
		cout << "surface::substitute_semilinear done" << endl;
		}
}

void surface::compute_intersection_points(INT *Adj, 
	INT *Lines, INT nb_lines, 
	INT *&Intersection_pt,  
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT j1, j2, a1, a2, pt;
	
	if (f_v) {
		cout << "surface::compute_intersection_points" << endl;
		}
	Intersection_pt = NEW_INT(nb_lines * nb_lines);
	INT_vec_mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->line_intersection(a1, a2);
				Intersection_pt[j1 * nb_lines + j2] = pt;
				Intersection_pt[j2 * nb_lines + j1] = pt;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_intersection_points done" << endl;
		}
}

void surface::compute_intersection_points_and_indices(INT *Adj, 
	INT *Points, INT nb_points, 
	INT *Lines, INT nb_lines, 
	INT *&Intersection_pt, INT *&Intersection_pt_idx, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT j1, j2, a1, a2, pt, idx;
	
	if (f_v) {
		cout << "surface::compute_intersection_points_and_indices" << endl;
		}
	Intersection_pt = NEW_INT(nb_lines * nb_lines);
	Intersection_pt_idx = NEW_INT(nb_lines * nb_lines);
	INT_vec_mone(Intersection_pt, nb_lines * nb_lines);
	for (j1 = 0; j1 < nb_lines; j1++) {
		a1 = Lines[j1];
		for (j2 = j1 + 1; j2 < nb_lines; j2++) {
			a2 = Lines[j2];
			if (Adj[j1 * nb_lines + j2]) {
				pt = P->line_intersection(a1, a2);
				if (!INT_vec_search(Points, nb_points, 
					pt, idx)) {
					cout << "surface::compute_intersection_points_and_indices cannot find point in Points" << endl;
					cout << "Points:";
					INT_vec_print_fully(cout, 
						Points, nb_points);
					cout << endl;
					cout << "j1=" << j1 << endl;
					cout << "j2=" << j2 << endl;
					cout << "a1=" << a1 << endl;
					cout << "a2=" << a2 << endl;
					cout << "pt=" << pt << endl;
					exit(1);
					}
				Intersection_pt[j1 * nb_lines + j2] = pt;
				Intersection_pt[j2 * nb_lines + j1] = pt;
				Intersection_pt_idx[j1 * nb_lines + j2] = idx;
				Intersection_pt_idx[j2 * nb_lines + j1] = idx;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_intersection_points_and_indices done" << endl;
		}
}

void surface::lines_meet3_and_skew3(INT *lines_meet3, INT *lines_skew3, 
	INT *&lines, INT &nb_lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[6];
	INT i, j;
	INT *perp;
	INT perp_sz;
	
	if (f_v) {
		cout << "surface::lines_meet3_and_skew3" << endl;
		cout << "The three lines we will meet are ";
		INT_vec_print(cout, lines_meet3, 3);
		cout << endl;
		cout << "The three lines we will be skew to are ";
		INT_vec_print(cout, lines_skew3, 3);
		cout << endl;
		}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[lines_meet3[i]];
		}
	for (i = 0; i < 3; i++) {
		o_rank[3 + i] = Klein->Line_to_point_on_quadric[lines_skew3[i]];
		}

	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	lines = NEW_INT(perp_sz);
	nb_lines = 0;
	for (i = 0; i < perp_sz; i++) {
		for (j = 0; j < 3; j++) {
			if (O->evaluate_bilinear_form_by_rank(perp[i], 
				o_rank[3 + j]) == 0) {
				break;
				}
			}
		if (j == 3) {
			lines[nb_lines++] = perp[i];
			}
		}
	
	FREE_INT(perp);
	
	for (i = 0; i < nb_lines; i++) {
		lines[i] = Klein->Point_on_quadric_to_line[lines[i]];
		}

	if (f_v) {
		cout << "surface::lines_meet3_and_skew3 done" << endl;
		}
}

void surface::perp_of_three_lines(INT *three_lines, 
	INT *&perp, INT &perp_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[3];
	INT i;
	
	if (f_v) {
		cout << "surface::perp_of_three_lines" << endl;
		cout << "The three lines are ";
		INT_vec_print(cout, three_lines, 3);
		cout << endl;
		}
	for (i = 0; i < 3; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[three_lines[i]];
		}
	O->perp_of_k_points(o_rank, 3, perp, perp_sz, verbose_level);

	for (i = 0; i < perp_sz; i++) {
		perp[i] = Klein->Point_on_quadric_to_line[perp[i]];
		}

	if (f_v) {
		cout << "surface::perp_of_three_lines done" << endl;
		}
}

INT surface::perp_of_four_lines(INT *four_lines, INT *trans12, 
	INT &perp_sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[4];
	INT i;
	INT *Perp;
	INT ret = TRUE;
	
	if (f_v) {
		cout << "surface::perp_of_four_lines" << endl;
		cout << "The four lines are ";
		INT_vec_print(cout, four_lines, 4);
		cout << endl;
		}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[four_lines[i]];
		}
	//Perp = NEW_INT(O->alpha * (O->q + 1));
	O->perp_of_k_points(o_rank, 4, Perp, perp_sz, verbose_level);
	if (perp_sz != 2) {
		if (f_v) {
			cout << "perp_sz = " << perp_sz << " != 2" << endl;
			}
		ret = FALSE;
		goto finish;
		}

	trans12[0] = Klein->Point_on_quadric_to_line[Perp[0]];
	trans12[1] = Klein->Point_on_quadric_to_line[Perp[1]];

finish:
	FREE_INT(Perp);
	if (f_v) {
		cout << "surface::perp_of_four_lines done" << endl;
		}
	return ret;
}

INT surface::rank_of_four_lines_on_Klein_quadric(INT *four_lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[4];
	INT *coords;
	INT i;
	INT rk;
	
	if (f_v) {
		cout << "surface::rank_of_four_lines_on_Klein_quadric" << endl;
		cout << "The four lines are ";
		INT_vec_print(cout, four_lines, 4);
		cout << endl;
		}
	for (i = 0; i < 4; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[four_lines[i]];
		}

	coords = NEW_INT(4 * 6);
	for (i = 0; i < 4; i++) {
		O->unrank_point(coords + i * 6, 1, 
			o_rank[i], 0 /* verbose_level */);
		}
	rk = F->Gauss_easy(coords, 4, 6);
	FREE_INT(coords);
	if (f_v) {
		cout << "surface::rank_of_four_lines_on_Klein_quadric done" << endl;
		}
	return rk;
}

INT surface::create_double_six_from_five_lines_with_a_common_transversal(
	INT *five_pts, INT *double_six, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[12];
	INT i, j;
	INT ret = TRUE;
	
	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines are ";
		INT_vec_print(cout, five_pts, 5);
		cout << endl;
		}
	for (i = 0; i < 5; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[five_pts[i]];
		}
	for (i = 0; i < 5; i++) {
		for (j = i + 1; j < 5; j++) {
			if (O->evaluate_bilinear_form_by_rank(o_rank[i], o_rank[j]) == 0) {
				cout << "surface::create_double_six_from_five_lines_with_a_common_transversal two of the given lines intersect, error" << endl;
				exit(1);
				}
			}
		}
	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_a_common_transversal" << endl;
		cout << "The five lines as points are ";
		INT_vec_print(cout, o_rank, 5);
		cout << endl;
		}

	INT nb_subsets;
	INT subset[4];
	INT pts[4];
	INT rk;
	INT **Perp;
	INT *Perp_sz;
	INT lines[2];
	INT opposites[5];
	INT transversal = 0;
	
	nb_subsets = INT_n_choose_k(5, 4);
	Perp = NEW_PINT(nb_subsets);
	Perp_sz = NEW_INT(nb_subsets);
#if 0
	for (rk = 0; rk < nb_subsets; rk++) {
		Perp[rk] = NEW_INT(O->alpha * (O->q + 1));
		}
#endif
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, 5, 4);
		for (i = 0; i < 4; i++) {
			pts[i] = o_rank[subset[i]];
			}

		if (f_v) {
			cout << "subset " << rk << " / " << nb_subsets << " : " << endl;
			}
		O->perp_of_k_points(pts, 4, 
			Perp[rk], Perp_sz[rk], verbose_level - 1);
		if (f_v) {
			cout << "the perp of the subset ";
			INT_vec_print(cout, subset, 4);
			cout << " has size " << Perp_sz[rk] << " : ";
			INT_vec_print(cout, Perp[rk], Perp_sz[rk]);
			cout << endl;
			}
		if (Perp_sz[rk] != 2) {
			ret = FALSE;
			nb_subsets = rk + 1;
			goto finish;
			}
		if (rk == 0) {
			INT_vec_copy(Perp[rk], lines, 2);
			}
		else if (rk == 1) {
			if (lines[0] == Perp[rk][0]) {
				transversal = lines[0];
				opposites[0] = lines[1];
				opposites[1] = Perp[rk][1];
				}
			else if (lines[0] == Perp[rk][1]) {
				transversal = lines[0];
				opposites[0] = lines[1];
				opposites[1] = Perp[rk][0];
				}
			else if (lines[1] == Perp[rk][0]) {
				transversal = lines[1];
				opposites[0] = lines[0];
				opposites[1] = Perp[rk][1];
				}
			else if (lines[1] == Perp[rk][1]) {
				transversal = lines[1];
				opposites[0] = lines[0];
				opposites[1] = Perp[rk][0];
				}
			}
		else {
			if (transversal == Perp[rk][0]) {
				opposites[rk] = Perp[rk][1];
				}
			else {
				opposites[rk] = Perp[rk][0];
				}
			}
		}

	o_rank[11] = transversal;
	for (i = 0; i < 5; i++) {
		o_rank[10 - i] = opposites[i];
		}

	INT *Perp_opp;
	INT Perp_opp_sz;
	INT transversal_opp;
	
	//Perp_opp = NEW_INT(O->alpha * (O->q + 1));
	O->perp_of_k_points(opposites, 4, Perp_opp, Perp_opp_sz, 0 /*verbose_level*/);
	if (f_v) {
		cout << "the perp of the opposite subset ";
		INT_vec_print(cout, opposites, 4);
		cout << " has size " << Perp_opp_sz << ":";
		INT_vec_print(cout, Perp_opp, Perp_opp_sz);
		cout << endl;
		}
	if (Perp_opp_sz != 2) {
		ret = FALSE;
		FREE_INT(Perp_opp);
		goto finish;
		}

	transversal_opp = -1;
	if (Perp_opp[0] == o_rank[0]) {
		transversal_opp = Perp_opp[1];
		}
	else if (Perp_opp[1] == o_rank[0]) {
		transversal_opp = Perp_opp[0];
		}
	else {
		cout << "surface::create_double_six_from_five_lines_with_a_common_transversal something is wrong with Perp_opp" << endl;
		exit(1);
		}

	o_rank[5] = transversal_opp;

	
	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->Point_on_quadric_to_line[o_rank[i]];
		}

	
	if (f_v) {
		for (i = 0; i < 12; i++) {
			for (j = 0; j < 12; j++) {
				if (O->evaluate_bilinear_form_by_rank(
					o_rank[i], o_rank[j]) == 0) {
					cout << "1";
					}
				else {
					cout << "0";
					}
				}
			cout << endl;
			}
		}

	
	FREE_INT(Perp_opp);

finish:
	for (i = 0; i < nb_subsets; i++) {
		FREE_INT(Perp[i]);
	}
	FREE_PINT(Perp);
	//free_PINT_all(Perp, nb_subsets);
	FREE_INT(Perp_sz);

	if (f_v) {
		cout << "surface::create_double_six_from_five_lines_with_a_common_transversal done" << endl;
		}
	return ret;
}


INT surface::create_double_six_from_six_disjoint_lines(INT *single_six, 
	INT *double_six, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT o_rank[12];
	INT i, j;
	INT ret = FALSE;
	
	if (f_v) {
		cout << "surface::create_double_six_from_six_disjoint_lines" << endl;
		}
	for (i = 0; i < 6; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[single_six[i]];
		}

	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) == 0) {
				cout << "two of the given lines intersect, error" << endl;
				exit(1);
				}
			}
		}


	// compute the perp on the Klein quadric of each of the 6 given lines:
	INT **Perp_without_pt;
	INT perp_sz = 0;
	INT sz;

	sz = O->alpha * q;
	Perp_without_pt = NEW_PINT(6);
	for (i = 0; i < 6; i++) {
		Perp_without_pt[i] = NEW_INT(sz);
		O->perp(o_rank[i], Perp_without_pt[i], perp_sz, 
			0 /* verbose_level */);
		if (perp_sz != sz) {
			cout << "perp_sz != sz" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "perp_sz=" << perp_sz << endl;
		for (i = 0; i < 6; i++) {
			INT_vec_print(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
			}
		}


	// compute the intersection of all perps, five at a time:
	
	INT **I2, *I2_sz;
	INT **I3, *I3_sz;
	INT **I4, *I4_sz;
	INT **I5, *I5_sz;
	INT six2, six3, six4, six5, rk, rk2;
	INT subset[6];

	six2 = INT_n_choose_k(6, 2);
	I2 = NEW_PINT(six2);
	I2_sz = NEW_INT(six2);
	for (rk = 0; rk < six2; rk++) {
		unrank_k_subset(rk, subset, 6, 2);
		INT_vec_intersect(
			Perp_without_pt[subset[0]], 
			perp_sz, 
			Perp_without_pt[subset[1]], 
			perp_sz, 
			I2[rk], I2_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1] << " of size " << I2_sz[rk] << " = ";
			INT_vec_print(cout, I2[rk], I2_sz[rk]);
			cout << endl;
			}
		}
	six3 = INT_n_choose_k(6, 3);
	I3 = NEW_PINT(six3);
	I3_sz = NEW_INT(six3);
	for (rk = 0; rk < six3; rk++) {
		unrank_k_subset(rk, subset, 6, 3);
		rk2 = rank_k_subset(subset, 6, 2);
		unrank_k_subset(rk, subset, 6, 3);
		INT_vec_intersect(I2[rk2], I2_sz[rk2], 
			Perp_without_pt[subset[2]], 
			perp_sz, 
			I3[rk], I3_sz[rk]);
		if (f_v) {
			cout << "Perp_" << subset[0] << " \\cap Perp_" << subset[1] << " \\cap Perp_" << subset[2] << " of size " << I3_sz[rk] << " = ";
			INT_vec_print(cout, I3[rk], I3_sz[rk]);
			cout << endl;
			}
		}

	six4 = INT_n_choose_k(6, 4);
	I4 = NEW_PINT(six4);
	I4_sz = NEW_INT(six4);
	for (rk = 0; rk < six4; rk++) {
		unrank_k_subset(rk, subset, 6, 4);
		rk2 = rank_k_subset(subset, 6, 3);
		unrank_k_subset(rk, subset, 6, 4);
		INT_vec_intersect(I3[rk2], I3_sz[rk2], 
			Perp_without_pt[subset[3]], perp_sz, 
			I4[rk], I4_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six4 << " : Perp_" << subset[0] << " \\cap Perp_" << subset[1] << " \\cap Perp_" << subset[2] << " \\cap Perp_" << subset[3] << " of size " << I4_sz[rk] << " = ";
			INT_vec_print(cout, I4[rk], I4_sz[rk]);
			cout << endl;
			}
		}

	six5 = INT_n_choose_k(6, 5);
	I5 = NEW_PINT(six5);
	I5_sz = NEW_INT(six5);
	for (rk = 0; rk < six5; rk++) {
		unrank_k_subset(rk, subset, 6, 5);
		rk2 = rank_k_subset(subset, 6, 4);
		unrank_k_subset(rk, subset, 6, 5);
		INT_vec_intersect(I4[rk2], I4_sz[rk2], 
			Perp_without_pt[subset[4]], perp_sz, 
			I5[rk], I5_sz[rk]);
		if (f_v) {
			cout << rk << " / " << six5 << " : Perp_" << subset[0] << " \\cap Perp_" << subset[1] << " \\cap Perp_" << subset[2] << " \\cap Perp_" << subset[3] << " \\cap Perp_" << subset[4] << " of size " << I5_sz[rk] << " = ";
			INT_vec_print(cout, I5[rk], I5_sz[rk]);
			cout << endl;
			}

		if (I5_sz[rk] != 1) {
			cout << "surface::create_double_six I5_sz[rk] != 1" << endl;
			ret = FALSE;
			goto free_it;
			}
		}
	for (i = 0; i < 6; i++) {
		o_rank[6 + i] = I5[6 - 1 - i][0];
		}
	for (i = 0; i < 12; i++) {
		double_six[i] = Klein->Point_on_quadric_to_line[o_rank[i]];
		}

	ret = TRUE;
free_it:
	for (i = 0; i < 6; i++) {
		FREE_INT(Perp_without_pt[i]);
	}
	FREE_PINT(Perp_without_pt);
	//free_PINT_all(Perp_without_pt, 6);



	for (i = 0; i < six2; i++) {
		FREE_INT(I2[i]);
	}
	FREE_PINT(I2);
	//free_PINT_all(I2, six2);

	FREE_INT(I2_sz);


	for (i = 0; i < six3; i++) {
		FREE_INT(I3[i]);
	}
	FREE_PINT(I3);
	//free_PINT_all(I3, six3);

	FREE_INT(I3_sz);


	for (i = 0; i < six4; i++) {
		FREE_INT(I4[i]);
	}
	FREE_PINT(I4);

	//free_PINT_all(I4, six4);

	FREE_INT(I4_sz);


	for (i = 0; i < six5; i++) {
		FREE_INT(I5[i]);
	}
	FREE_PINT(I5);

	//free_PINT_all(I5, six5);

	FREE_INT(I5_sz);

	
	if (f_v) {
		cout << "surface::create_double_six_from_six_disjoint_lines done" << endl;
		}
	return ret;
}

void surface::latex_double_six(ostream &ost, INT *double_six)
{
	INT i, j, a, u, v;

	ost << "\\begin{array}{cc}" << endl;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 2; j++) {
			a = double_six[j * 6 + i];
			Gr->unrank_INT(a, 0);
			ost << "\\left[" << endl;
			ost << "\\begin{array}{*{6}{c}}" << endl;
			for (u = 0; u < 2; u++) {
				for (v = 0; v < 4; v++) {
					ost << Gr->M[u * 4 + v];
					if (v < 4 - 1) {
						ost << ", ";
						}
					}
				ost << "\\\\" << endl;
				}
			ost << "\\end{array}" << endl;
			ost << "\\right]" << endl;
			if (j < 2 - 1) {
				ost << ", " << endl;
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
}

void surface::create_the_fifteen_other_lines(INT *double_six, 
	INT *fifteen_other_lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "create_the_fifteen_other_lines" << endl;
		}
	INT *Planes;
	INT *Lines;
	INT h, k3;
	INT i, j;
	
	Planes = NEW_INT(30);
	if (f_v) {
		cout << "creating the 30 planes:" << endl;
		}
	for (h = 0; h < 30; h++) {
		i = Sets[h * 2 + 0];
		j = Sets[h * 2 + 1];
		Gr->unrank_INT_here(Basis0, double_six[i], 0/* verbose_level*/);
		Gr->unrank_INT_here(Basis0 + 8, double_six[j], 0/* verbose_level*/);
		if (F->Gauss_easy(Basis0, 4, 4) != 3) {
			cout << "the rank is not 3" << endl;
			exit(1);
			}
		Planes[h] = Gr3->rank_INT_here(Basis0, 0/* verbose_level*/);
		if (f_v) {
			cout << "plane " << h << " / " << 30 
				<< " has rank " << Planes[h] << " and basis ";
			INT_vec_print(cout, Basis0, 12);
			cout << endl;
			}
		}
	Lines = NEW_INT(15);
	if (f_v) {
		cout << "creating the 15 lines:" << endl;
		}
	for (h = 0; h < 15; h++) {
		i = Sets2[h * 2 + 0];
		j = Sets2[h * 2 + 1];
		Gr3->unrank_INT_here(Basis1, Planes[i], 0/* verbose_level*/);
		Gr3->unrank_INT_here(Basis2, Planes[j], 0/* verbose_level*/);
		F->intersect_subspaces(4, 3, Basis1, 3, Basis2, 
			k3, Basis0, 0 /* verbose_level */);
		if (k3 != 2) {
			cout << "the rank is not 2" << endl;
			exit(1);
			}
		Lines[h] = Gr->rank_INT_here(Basis0, 0/* verbose_level*/);
		for (i = 0; i < 2; i++) {
			PG_element_normalize_from_front(*F, 
				Basis0 + i * 4, 1, 4);
			}
		if (f_v) {
			cout << "line " << h << " / " << 15 
				<< " has rank " << Lines[h] 
				<< " and basis ";
			INT_vec_print(cout, Basis0, 8);
			cout << endl;
			}
		}

	INT_vec_copy(Lines, fifteen_other_lines, 15);


	FREE_INT(Planes);
	FREE_INT(Lines);

	if (f_v) {
		cout << "create_the_fifteen_other_lines done" << endl;
		}
	
}

void surface::compute_adjacency_matrix_of_line_intersection_graph(
	INT *&Adj, INT *S, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_line_intersection_graph" << endl;
		}
	if (n > 27) {
		cout << "surface::compute_adjacency_matrix_of_line_intersection_graph n > 27" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[S[i]];
		}

	Adj = NEW_INT(n * n);
	INT_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) == 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_line_intersection_graph done" << endl;
		}
}

void surface::compute_adjacency_matrix_of_line_disjointness_graph(
	INT *&Adj, INT *S, INT n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_line_disjointness_graph" << endl;
		}
	if (n > 27) {
		cout << "surface::compute_adjacency_matrix_of_line_disjointness_graph n > 27" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		o_rank[i] = Klein->Line_to_point_on_quadric[S[i]];
		}

	Adj = NEW_INT(n * n);
	INT_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (O->evaluate_bilinear_form_by_rank(
				o_rank[i], o_rank[j]) != 0) {
				Adj[i * n + j] = 1;
				Adj[j * n + i] = 1;
				}
			}
		}
	if (f_v) {
		cout << "surface::compute_adjacency_matrix_of_line_disjointness_graph done" << endl;
		}
}

void surface::compute_points_on_lines(
	INT *Pts_on_surface, INT nb_points_on_surface, 
	INT *Lines, INT nb_lines, 
	set_of_sets *&pts_on_lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, l, r;
	INT *Surf_pt_coords;
	INT Basis[8];
	INT Mtx[12];
	
	if (f_v) {
		cout << "surface::compute_points_on_lines" << endl;
		}
	pts_on_lines = NEW_OBJECT(set_of_sets);
	pts_on_lines->init_basic_constant_size(nb_points_on_surface, 
		nb_lines, q + 1, 0 /* verbose_level */);
	Surf_pt_coords = NEW_INT(nb_points_on_surface * 4);
	for (i = 0; i < nb_points_on_surface; i++) {
		P->unrank_point(Surf_pt_coords + i * 4, Pts_on_surface[i]);
		}

	INT_vec_zero(pts_on_lines->Set_size, nb_lines);
	for (i = 0; i < nb_lines; i++) {
		l = Lines[i];
		P->unrank_line(Basis, l);
		//cout << "Line " << i << " basis=";
		//INT_vec_print(cout, Basis, 8);
		//cout << " : ";
		for (j = 0; j < nb_points_on_surface; j++) {
			INT_vec_copy(Basis, Mtx, 8);
			INT_vec_copy(Surf_pt_coords + j * 4, Mtx + 8, 4);
			r = F->Gauss_easy(Mtx, 3, 4);
			if (r == 2) {
				pts_on_lines->add_element(i, j);
				//cout << j << " ";
				}
			}
		//cout << endl;
		}
	//cout << "the surface points on the set of " << nb_lines << " lines are:" << endl;
	//pts_on_lines->print_table();

	FREE_INT(Surf_pt_coords);
	if (f_v) {
		cout << "surface::compute_points_on_lines done" << endl;
		}
}

INT surface::test_special_form_alpha_beta(INT *coeff, 
	INT &alpha, INT &beta, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret = TRUE;
	INT zeroes[] = {0,1,2,4,5,7,8,10,11,13,14,15,17,18,19};
	INT alphas[] = {6,9,12};
	INT betas[] = {16};
	INT a;
	
	if (f_v) {
		cout << "surface::test_special_form_alpha_beta" << endl;
		}
	if (!INT_vec_is_constant_on_subset(coeff, 
		zeroes, sizeof(zeroes) / sizeof(INT), a)) {
		cout << "surface::test_special_form_alpha_beta not constant on zero set" << endl;
		return FALSE;
		}
	if (a != 0) {
		cout << "surface::test_special_form_alpha_beta not zero on zero set" << endl;
		return FALSE;
		}
	if (coeff[3] != 1) {
		cout << "surface::test_special_form_alpha_beta not normalized" << endl;
		exit(1);
		}
	if (!INT_vec_is_constant_on_subset(coeff, 
		alphas, sizeof(alphas) / sizeof(INT), a)) {
		cout << "surface::test_special_form_alpha_beta not constant on alpha set" << endl;
		return FALSE;
		}
	alpha = a;
	if (!INT_vec_is_constant_on_subset(coeff, 
		betas, sizeof(betas) / sizeof(INT), a)) {
		cout << "surface::test_special_form_alpha_beta not constant on beta set" << endl;
		return FALSE;
		}
	beta = a;

	if (f_v) {
		cout << "surface::test_special_form_alpha_beta done" << endl;
		}
	return ret;
}

void surface::create_special_double_six(INT *double_six, 
	INT a, INT b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT Basis[12 * 8] = {
		1,2,0,0,0,0,1,6,
		1,3,0,0,0,0,1,7,
		1,0,5,0,0,1,0,7,
		1,0,4,0,0,1,0,6,
		1,0,0,7,0,1,3,0,
		1,0,0,6,0,1,2,0,
		1,5,0,0,0,0,1,7,
		1,4,0,0,0,0,1,6,
		1,0,2,0,0,1,0,6,
		1,0,3,0,0,1,0,7,
		1,0,0,6,0,1,4,0,
		1,0,0,7,0,1,5,0
		};
	INT i, c, ma, mb, av, mav;

	if (f_v) {
		cout << "surface::create_special_double_six a=" << a << " b=" << b << endl;
		}
	ma = F->negate(a);
	mb = F->negate(b);
	av = F->inverse(a);
	mav = F->negate(av);
	for (i = 0; i < 12 * 8; i++) {
		c = Basis[i];
		if (c == 2) {
			c = a;
			}
		else if (c == 3) {
			c = ma;
			}
		else if (c == 4) {
			c = av;
			}
		else if (c == 5) {
			c = mav;
			}
		else if (c == 6) {
			c = b;
			}
		else if (c == 7) {
			c = mb;
			}
		Basis[i] = c;
		}
	for (i = 0; i < 12; i++) {
		double_six[i] = Gr->rank_INT_here(Basis + i * 8, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface::create_special_double_six done" << endl;
		}
}

void surface::create_special_fifteen_lines(INT *fifteen_lines, 
	INT a, INT b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT Basis[15 * 8] = {
		1,0,0,0,0,1,0,0, // 0 = c_12
		0,1,-1,0,2,3,0,-1, // 1 = c_13
		0,-1,-1,0,-2,-3,0,-1, // 2 = c_14
		1,0,-1,0,-3,2,0,-1, // 3 = c_15
		-1,0,-1,0,3,-2,0,-1, // 4 = c_16
		0,-1,-1,0,-2,3,0,-1, // 5 = c_23
		0,1,-1,0,2,-3,0,-1, // 6 = c_24
		-1,0,-1,0,-3,-2,0,-1, // 7 = c_25
		1,0,-1,0,3,2,0,-1, // 8 = c_26
		1,0,0,0,0,0,1,0, // 9 = c_34
		-1,0,4,5,0,-1,-4,-5, // 10 = c_35
		-1,0,4,-5,0,-1,4,-5, // 11 = c_36
		-1,0,-4,5,0,-1,-4,5, // 12 = c_45
		-1,0,-4,-5,0,-1,4,5, // 13 = c_46
		0,1,0,0,0,0,1,0 // 14 = c_56
		};
	INT i, m1, a2, a2p1, a2m1, ba2p1, /*ba2m1,*/ twoa;
	INT c, c2, cm2, c3, cm3, c4, cm4, c5, cm5;

	// 2 stands for (2a)/(b(a^2+1))
	// -2 stands for -(2a)/(b(b^2+1))
	// 3 stands for (a^2-1)/(b(a^2+1))
	// -3 stands for -(a^2-1)/(b(a^2+1))
	// 4 stands for (2a)/(a^2-1)
	// -4 stands for -(2a)/(a^2-1)
	// 5 stands for 3 inverse
	// -5 stands for -3 inverse

	if (f_v) {
		cout << "surface::create_special_fifteen_lines a=" << a << " b=" << b << endl;
		}
	m1 = F->negate(1);
	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, m1);
	twoa = F->add(a, a);
	ba2p1 = F->mult(b, a2p1);
	//ba2m1 = F->mult(b, a2m1);

	if (ba2p1 == 0) {
		cout << "surface::create_special_fifteen_lines ba2p1 = 0, cannot invert" << endl;
		exit(1);
		}
	c2 = F->mult(twoa, F->inverse(ba2p1));
	cm2 = F->negate(c2);
	c3 = F->mult(a2m1, F->inverse(ba2p1));
	cm3 = F->negate(c3);
	if (a2m1 == 0) {
		cout << "surface::create_special_fifteen_lines a2m1 = 0, cannot invert" << endl;
		exit(1);
		}
	c4 = F->mult(twoa, F->inverse(a2m1));
	cm4 = F->negate(c4);

	if (c3 == 0) {
		cout << "surface::create_special_fifteen_lines c3 = 0, cannot invert" << endl;
		exit(1);
		}
	c5 = F->inverse(c3);
	if (cm3 == 0) {
		cout << "surface::create_special_fifteen_lines cm3 = 0, cannot invert" << endl;
		exit(1);
		}
	cm5 = F->inverse(cm3);


	for (i = 0; i < 15 * 8; i++) {
		c = Basis[i];
		if (c == 0) {
			c = 0;
			}
		else if (c == 1) {
			c = 1;
			}
		else if (c == -1) {
			c = m1;
			}
		else if (c == 2) {
			c = c2;
			}
		else if (c == -2) {
			c = cm2;
			}
		else if (c == 3) {
			c = c3;
			}
		else if (c == -3) {
			c = cm3;
			}
		else if (c == 4) {
			c = c4;
			}
		else if (c == -4) {
			c = cm4;
			}
		else if (c == 5) {
			c = c5;
			}
		else if (c == -5) {
			c = cm5;
			}
		else {
			cout << "surface::create_special_fifteen_lines unknown value" << c << endl;
			exit(1);
			}
		Basis[i] = c;
		}
	for (i = 0; i < 15; i++) {
		fifteen_lines[i] = Gr->rank_INT_here(
			Basis + i * 8, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface::create_special_fifteen_lines done" << endl;
		}
}

void surface::create_remaining_fifteen_lines(INT *double_six, 
	INT *fifteen_lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, h;

	if (f_v) {
		cout << "surface::create_remaining_fifteen_lines" << endl;
		}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			if (f_vv) {
				cout << "surface::create_remaining_fifteen_lines creating line c_ij where i=" << i << " j=" << j << ":" << endl;
				}
			fifteen_lines[h++] = compute_cij(
				double_six, i, j, 0 /*verbose_level*/);
			}
		}
	if (f_v) {
		cout << "surface::create_remaining_fifteen_lines done" << endl;
		}
}

INT surface::compute_cij(INT *double_six, INT i, INT j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ai, aj, bi, bj;
	INT Basis1[16];
	INT Basis2[16];
	INT K1[16];
	INT K2[16];
	INT K[16];
	INT base_cols1[4];
	INT base_cols2[4];
	INT kernel_m, kernel_n, cij;

	if (f_v) {
		cout << "surface::compute_cij" << endl;
		}
	ai = double_six[i];
	aj = double_six[j];
	bi = double_six[6 + i];
	bj = double_six[6 + j];
	Gr->unrank_INT_here(Basis1, ai, 0 /* verbose_level */);
	Gr->unrank_INT_here(Basis1 + 2 * 4, bj, 0 /* verbose_level */);
	Gr->unrank_INT_here(Basis2, aj, 0 /* verbose_level */);
	Gr->unrank_INT_here(Basis2 + 2 * 4, bi, 0 /* verbose_level */);
	if (F->Gauss_simple(Basis1, 4, 4, base_cols1, 0 /* verbose_level */) != 3) {
		cout << "The rank of Basis1 is not 3" << endl;
		exit(1);
		}
	if (F->Gauss_simple(Basis2, 4, 4, base_cols2, 0 /* verbose_level */) != 3) {
		cout << "The rank of Basis2 is not 3" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::compute_cij before matrix_get_kernel Basis1" << endl;
		}
	F->matrix_get_kernel(Basis1, 3, 4, base_cols1, 3, 
		kernel_m, kernel_n, K1);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 when computing K1" << endl;
		exit(1);
		}
	if (kernel_n != 1) {
		cout << "surface::compute_cij kernel_1 != 1 when computing K1" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[j * 4 + i] = K1[i * kernel_n + j];
			}
		}
	if (f_v) {
		cout << "surface::compute_cij before matrix_get_kernel Basis2" << endl;
		}
	F->matrix_get_kernel(Basis2, 3, 4, base_cols2, 3, 
		kernel_m, kernel_n, K2);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 when computing K2" << endl;
		exit(1);
		}
	if (kernel_n != 1) {
		cout << "surface::compute_cij kernel_1 != 1 when computing K2" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < 4; i++) {
			K[(1 + j) * 4 + i] = K2[i * kernel_n + j];
			}
		}
	if (F->Gauss_simple(K, 2, 4, base_cols1, 0 /* verbose_level */) != 2) {
		cout << "The rank of K is not 2" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::compute_cij before matrix_get_kernel K" << endl;
		}
	F->matrix_get_kernel(K, 2, 4, base_cols1, 2, 
		kernel_m, kernel_n, K1);
	if (kernel_m != 4) {
		cout << "surface::compute_cij kernel_m != 4 when computing final kernel" << endl;
		exit(1);
		}
	if (kernel_n != 2) {
		cout << "surface::compute_cij kernel_n != 2 when computing final kernel" << endl;
		exit(1);
		}
	for (j = 0; j < kernel_n; j++) {
		for (i = 0; i < n; i++) {
			Basis1[j * n + i] = K1[i * kernel_n + j];
			}
		}
	cij = Gr->rank_INT_here(Basis1, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::compute_cij done" << endl;
		}
	return cij;
}

INT surface::compute_transversals_of_any_four(INT *&Trans, 
	INT &nb_subsets, INT *lines, INT sz, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT trans12[2];
	INT subset[4];
	INT four_lines[4];
	INT i, rk, perp_sz;
	INT ret = TRUE;
	
	if (f_v) {
		cout << "surface::compute_transversals_of_any_four" << endl;
		}
	nb_subsets = INT_n_choose_k(sz, 4);
	Trans = NEW_INT(nb_subsets * 2);
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, sz, 4);
		for (i = 0; i < 4; i++) {
			four_lines[i] = lines[subset[i]];
			}

		if (f_v) {
			cout << "testing subset " << rk << " / " 
				<< nb_subsets << " : " << endl;
			}
		if (!perp_of_four_lines(four_lines, trans12, 
			perp_sz, 0 /*verbose_level*/)) {

			if (f_v) {
				cout << "The 4-subset does not lead to two transversal lines: ";
				INT_vec_print(cout, subset, 4);
				cout << " = ";
				INT_vec_print(cout, four_lines, 4);
				cout << " perp_sz=" << perp_sz << endl;
				}
			ret = FALSE;
			//break;
			trans12[0] = -1;
			trans12[1] = -1;
			}
		INT_vec_copy(trans12, Trans + rk * 2, 2);
		}
	if (f_v) {
		cout << "Transversals:" << endl;
		INT_matrix_print(Trans, nb_subsets, 2);
		}
	if (f_v) {
		cout << "surface::compute_transversals_of_any_four done" << endl;
		}
	return ret;
}

INT surface::compute_rank_of_any_four(INT *&Rk, INT &nb_subsets, 
	INT *lines, INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT subset[4];
	INT four_lines[4];
	INT i, rk;
	INT ret = TRUE;
	
	if (f_v) {
		cout << "surface::compute_rank_of_any_four" << endl;
		}
	nb_subsets = INT_n_choose_k(sz, 4);
	Rk = NEW_INT(nb_subsets);
	for (rk = 0; rk < nb_subsets; rk++) {
		unrank_k_subset(rk, subset, sz, 4);
		for (i = 0; i < 4; i++) {
			four_lines[i] = lines[subset[i]];
			}

		if (f_v) {
			cout << "testing subset " << rk << " / " 
				<< nb_subsets << " : " << endl;
			}

		Rk[rk] = rank_of_four_lines_on_Klein_quadric(
			four_lines, 0 /* verbose_level */);
		if (Rk[rk] < 4) {
			ret = FALSE;
			}
		}
	if (f_v) {
		cout << "Ranks:" << endl;
		INT_vec_print(cout, Rk, nb_subsets);
		cout << endl;
		}
	if (f_v) {
		cout << "surface::compute_rank_of_any_four done" << endl;
		}
	return ret;
}

void surface::create_equation_Sab(INT a, INT b, INT *coeff, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT alpha, beta;

	if (f_v) {
		cout << "surface::create_equation_Sab" << endl;
		}
	alpha = F->negate(F->mult(b, b));
	beta = F->mult(F->mult(F->power(b, 3), 
		F->add(1, F->mult(a, a))), F->inverse(a));
	INT_vec_zero(coeff, nb_monomials);
	
	coeff[3] = 1;
	coeff[6] = alpha;
	coeff[9] = alpha;
	coeff[12] = alpha;
	coeff[16] = beta;
	//coeff[19] = beta;
	if (f_v) {
		cout << "surface::create_equation_Sab done" << endl;
		}
}

INT surface::create_surface_ab(INT a, INT b, INT *Lines, 
	INT &alpha, INT &beta, INT &nb_E, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT alpha0, beta0;
	//INT line_rk;
	//INT Basis[8];
	//INT Lines[27];
	INT nb, i, e, ee, nb_lines, rk, nb_pts;
	INT *coeff;
	INT *Pts;
	INT v[4];

	if (f_v) {
		cout << "surface::create_surface_ab" << endl;
		}
	alpha = -1;
	beta = -1;
	nb_E = -1;

	INT a2, a2p1, a2m1;

	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, F->negate(1));
	if (a2p1 == 0 || a2m1 == 0) {
		cout << "surface::create_surface_ab a2p1 == 0 || a2m1 == 0" << endl;
		return FALSE;
		}


	Pts = NEW_INT(nb_PG_elements(3, F->q));
	
	coeff = NEW_INT(20);
	alpha0 = F->negate(F->mult(b, b));
	beta0 = F->mult(F->mult(F->power(b, 3), 
		F->add(1, F->mult(a, a))), F->inverse(a));
	if (f_v) {
		cout << "surface::create_surface_ab a=" 
			<< a << " b=" << b << " alpha0=" << alpha0 
			<< " beta0=" << beta0 << endl;
		}
	
#if 0
	INT_vec_zero(Basis, 8);
	Basis[0 * 4 + 0] = 1;
	Basis[0 * 4 + 1] = a;
	Basis[1 * 4 + 2] = 1;
	Basis[1 * 4 + 3] = b;
	line_rk = Gr->rank_INT_here(Basis, 0);
#endif


#if 0
	//INT_vec_copy(desired_lines, Lines, 3);
	//nb = 3;

	cout << "The triangle lines are:" << endl;
	Gr->print_set(desired_lines, 3);
#endif


	INT *Oab;

	Oab = NEW_INT(12);
	create_special_double_six(Oab, a, b, 0 /* verbose_level */);

#if 0
	if (!test_if_sets_are_equal(Oab, Lines, 12)) {
		cout << "the sets are not equal" << endl;
		exit(1);
		}
#endif

	if (f_v) {
		cout << "surface::create_surface_ab The double six is:" << endl;
		Gr->print_set(Oab, 12);
		}


	INT_vec_copy(Oab, Lines, 12);
	FREE_INT(Oab);


	nb = 12;

	if (f_v) {
		cout << "surface::create_surface_ab We have a set of lines of size " << nb << ":";
		INT_vec_print(cout, Lines, nb);
		cout << endl;
		}

	create_remaining_fifteen_lines(Lines, 
		Lines + 12, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface::create_surface_ab The remaining 15 lines are:";
		INT_vec_print(cout, Lines + 12, 15);
		cout << endl;
		Gr->print_set(Lines + 12, 15);
		}


	if (f_v) {
		cout << "surface::create_surface_ab before create_special_fifteen_lines" << endl;
		}

	INT special_lines[15];

	create_special_fifteen_lines(special_lines, a, b, verbose_level);
	for (i = 0; i < 15; i++) {
		if (special_lines[i] != Lines[12 + i]) {
			cout << "surface::create_surface_ab something is wrong with the special line " << i << " / 15 " << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "surface::create_surface_ab after create_special_fifteen_lines" << endl;
		}

	rk = compute_system_in_RREF(27, Lines, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::create_surface_ab a=" << a 
			<< " b=" << b << " rk=" << rk << endl;
		}

	if (rk != 19) {
		cout << "surface::create_surface_ab rk != 19" << endl;
		FREE_INT(Pts);
		FREE_INT(coeff);
		exit(1);
		}
	build_cubic_surface_from_lines(27, Lines, coeff, 0 /* verbose_level */);
	PG_element_normalize_from_front(*F, coeff, 1, 20);



	enumerate_points(coeff, Pts, nb_pts, 0 /* verbose_level */);
	INT_vec_heapsort(Pts, nb_pts);


	if (f_v) {
		cout << "surface::create_surface_ab a=" << a << " b=" << b << " equation: ";
		print_equation(cout, coeff);
		cout << endl;
		}

	if (nb_pts != nb_pts_on_surface) {
		cout << "surface::create_surface_ab degenerate surface" << endl;
		cout << "nb_pts=" << nb_pts << endl;
		cout << "should be =" << nb_pts_on_surface << endl;
		alpha = -1;
		beta = -1;
		nb_E = -1;
		return FALSE;
		}

	if (f_v) {
		cout << "surface::create_surface_ab Pts: " << endl;
		INT_vec_print_as_table(cout, Pts, nb_pts, 10);
		}


	INT *Adj;
	INT *Intersection_pt;
	INT *Intersection_pt_idx;

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines, 27, verbose_level);
	if (f_v) {
		cout << "surface::create_surface_ab The adjacency matrix is:" << endl;
		INT_matrix_print(Adj, 27, 27);
		}



	compute_intersection_points_and_indices(
		Adj, Pts, nb_pts, Lines, 27, 
		Intersection_pt, Intersection_pt_idx, 
		verbose_level);

	if (f_v) {
		cout << "surface::create_surface_ab The intersection points are:" << endl;
		INT_matrix_print(Intersection_pt_idx, 27, 27);
		}


	classify C;

	C.init(Intersection_pt_idx, 27 * 27, FALSE, 0);
	if (f_v) {
		cout << "surface::create_surface_ab classification of points by multiplicity:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}




	if (!test_special_form_alpha_beta(coeff, alpha, beta, 
		0 /* verbose_level */)) {
		cout << "surface::create_surface_ab not of special form" << endl;
		exit(1);
		}


	if (alpha != alpha0) {
		cout << "surface::create_surface_ab alpha != alpha0" << endl;
		exit(1);
		}
	if (beta != beta0) {
		cout << "surface::create_surface_ab beta != beta0" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface::create_surface_ab determining all lines on the surface:" << endl;
		}
	{
	INT Lines2[27];
	P->find_lines_which_are_contained(Pts, nb_pts, 
		Lines2, nb_lines, 27 /* max_lines */, 
		0 /* verbose_level */);
	}
	
	if (f_v) {
		cout << "surface::create_surface_ab nb_lines = " << nb_lines << endl;
		}
	if (nb_lines != 27) {
		cout << "surface::create_surface_ab nb_lines != 27, something is wrong with the surface" << endl;
		exit(1);
		}
	set_of_sets *pts_on_lines;
	set_of_sets *lines_on_pt;
	
	compute_points_on_lines(Pts, nb_pts, 
		Lines, nb_lines, 
		pts_on_lines, 
		verbose_level);


	if (f_v) {
		cout << "surface::create_surface_ab pts_on_lines: " << endl;
		pts_on_lines->print_table();
		}

	INT *E;
	
	pts_on_lines->get_eckardt_points(E, nb_E, 0 /* verbose_level */);
	//nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface::create_surface_ab The surface contains " 
			<< nb_E << " Eckardt points" << endl;
		}

#if 0
	if (a == 2 && b == 1) {
		exit(1);
		}
#endif


	pts_on_lines->dualize(lines_on_pt, 0 /* verbose_level */);

#if 0
	cout << "lines_on_pt: " << endl;
	lines_on_pt->print_table();
#endif

	if (f_v) {
		cout << "surface::create_surface_ab The Eckardt points are:" << endl;
		for (i = 0; i < nb_E; i++) {
			e = E[i];
			ee = Pts[e];
			unrank_point(v, ee);
			cout << i << " : " << ee << " : ";
			INT_vec_print(cout, v, 4);
			cout << " on lines: ";
			INT_vec_print(cout, lines_on_pt->Sets[e], 
				lines_on_pt->Set_size[e]);
			cout << endl;
			}
		}

	
	FREE_INT(E);
	FREE_INT(coeff);
	FREE_INT(Pts);
	FREE_INT(Intersection_pt);
	FREE_INT(Intersection_pt_idx);
	FREE_OBJECT(pts_on_lines);
	FREE_OBJECT(lines_on_pt);
	if (f_v) {
		cout << "surface::create_surface_ab done" << endl;
		}
	return TRUE;
}

void surface::list_starter_configurations(INT *Lines, INT nb_lines, 
	set_of_sets *line_intersections, INT *&Table, INT &N, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT subset[5];
	INT subset2[5];
	INT S3[6];
	INT N1, nCk, h;
	INT i, j, r;
	
	if (f_v) {
		cout << "surface::list_starter_configurations" << endl;
		}

	N = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
			}
		nCk = INT_n_choose_k(line_intersections->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			unrank_k_subset(j, subset, 
				line_intersections->Set_size[i], 5);
			for (h = 0; h < 5; h++) {
				subset2[h] = 
				line_intersections->Sets[i][subset[h]];
				S3[h] = Lines[subset2[h]];
				}
			S3[5] = Lines[i];
			r = compute_system_in_RREF(6, S3, 0 /*verbose_level*/);
			if (r == 19) {
				N++;
				}
			}
		}
	if (f_v) {
		cout << "surface::list_starter_configurations We found " 
			<< N << " starter configurations on this surface" 
			<< endl;
		}
	Table = NEW_INT(N * 2);
	N1 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
			}
		nCk = INT_n_choose_k(line_intersections->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			unrank_k_subset(j, subset, 
				line_intersections->Set_size[i], 5);
			for (h = 0; h < 5; h++) {
				subset2[h] = 
				line_intersections->Sets[i][subset[h]];
				S3[h] = Lines[subset2[h]];
				}
			S3[5] = Lines[i];
			r = compute_system_in_RREF(6, S3, 0 /*verbose_level*/);
			if (r == 19) {
				Table[N1 * 2 + 0] = i;
				Table[N1 * 2 + 1] = j;
				N1++;
				}
			}
		}
	if (N1 != N) {
		cout << "N1 != N" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::list_starter_configurations done" << endl;
		}
}

void surface::create_starter_configuration(
	INT line_idx, INT subset_idx, 
	set_of_sets *line_neighbors, INT *Lines, INT *S, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT subset[5];
	INT subset2[5];
	INT h; //, nCk;
	
	if (f_v) {
		cout << "surface::create_starter_configuration" << endl;
		}
	//nCk = INT_n_choose_k(line_neighbors->Set_size[line_idx], 5);
	unrank_k_subset(subset_idx, subset, 
		line_neighbors->Set_size[line_idx], 5);
	for (h = 0; h < 5; h++) {
		subset2[h] = line_neighbors->Sets[line_idx][subset[h]];
		S[h] = Lines[subset2[h]];
		}
	S[5] = Lines[line_idx];
	if (f_v) {
		cout << "surface::create_starter_configuration done" << endl;
		}
}

void surface::wedge_to_klein(INT *W, INT *K)
{
	K[0] = W[0];
	K[1] = W[5];
	K[2] = W[1];
	K[3] = F->negate(W[4]);
	K[4] = W[2];
	K[5] = W[3];
}

void surface::klein_to_wedge(INT *K, INT *W)
{
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
}

INT surface::line_to_wedge(INT line_rk)
{
	INT a, b;
	
	a = Klein->Line_to_point_on_quadric[line_rk];
	O->unrank_point(w2, 1, a, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	PG_element_rank_modified(*F, v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface::line_to_wedge_vec(INT *Line_rk, INT *Wedge_rk, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = line_to_wedge(Line_rk[i]);
		}
}

void surface::line_to_klein_vec(INT *Line_rk, INT *Klein_rk, INT len)
{
	INT_vec_apply(Line_rk, Klein->Line_to_point_on_quadric, Klein_rk, len);
}

INT surface::klein_to_wedge(INT klein_rk)
{
	INT b;
	
	O->unrank_point(w2, 1, klein_rk, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	PG_element_rank_modified(*F, v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface::klein_to_wedge_vec(INT *Klein_rk, INT *Wedge_rk, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = klein_to_wedge(Klein_rk[i]);
		}
}

INT surface::identify_two_lines(INT *lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT iso = 0;
	INT *Adj;

	if (f_v) {
		cout << "surface::identify_two_lines" << endl;
		}
	
	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, lines, 2, 0 /* verbose_level */);
	if (Adj[0 * 2 + 1]) {
		iso = 1;
		}
	else {
		iso = 0;
		}
	FREE_INT(Adj);
	if (f_v) {
		cout << "surface::identify_two_lines done" << endl;
		}
	return iso;
}

	
INT surface::identify_three_lines(INT *lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT iso = 0;
	INT *Adj;
	INT i, j, c;
	INT a1, a2; //, a3;

	if (f_v) {
		cout << "surface::identify_three_lines" << endl;
		}
	
	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, lines, 3, 0 /* verbose_level */);

	
	c = 0;
	for (i = 0; i < 3; i++) {
		for (j = i + 1; j < 3; j++) {
			if (Adj[i * 3 + j]) {
				c++;
				}
			}
		}
	if (c == 0) {
		iso = 6;
		}
	else if (c == 1) {
		iso = 4;
		}
	else if (c == 2) {
		iso = 5;
		}
	else if (c == 3) {
		INT *Intersection_pt;
		INT rk;

		compute_intersection_points(Adj, 
			lines, 3, Intersection_pt, 
			0 /*verbose_level */);
		a1 = Intersection_pt[0 * 3 + 1];
		a2 = Intersection_pt[0 * 3 + 2];
		//a3 = Intersection_pt[1 * 3 + 2];
		if (a1 == a2) {
			INT Basis[3 * 8];

			for (i = 0; i < 3; i++) {
				Gr->unrank_INT_here(Basis + i * 8, 
					lines[i], 0 /* verbose_level */);
				}
			rk = F->Gauss_easy(Basis, 6, 4);
			if (rk == 3) {
				iso = 2;
				}
			else if (rk == 4) {
				iso = 1;
				}
			else {
				cout << "surface::identify_three_lines rk=" << rk << endl;
				exit(1);
				}
			}
		else {
			iso = 3;
			}
		FREE_INT(Intersection_pt);
		}
	

	FREE_INT(Adj);
	if (f_v) {
		cout << "surface::identify_three_lines done" << endl;
		}
	return iso;
}


void surface::make_spreadsheet_of_lines_in_three_kinds(
	spreadsheet *&Sp, 
	INT *Wedge_rk, INT *Line_rk, INT *Klein_rk, INT nb_lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, a;
	char str[1000];
	INT w[6];
	INT Basis[8];
	char **Text_wedge;
	char **Text_line;
	char **Text_klein;

	if (f_v) {
		cout << "surface::make_spreadsheet_of_lines_in_three_kinds" << endl;
		}

	Text_wedge = NEW_pchar(nb_lines);
	Text_line = NEW_pchar(nb_lines);
	Text_klein = NEW_pchar(nb_lines);

	for (i = 0; i < nb_lines; i++) {
		a = Wedge_rk[i];
		PG_element_unrank_modified(*F, w, 1, 6 /*wedge_dimension*/, a);
		INT_vec_print_to_str(str, w, 6);
		Text_wedge[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_wedge[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Line_rk[i];
		Gr->unrank_INT_here(Basis, a, 0 /* verbose_level */);
		INT_vec_print_to_str(str, Basis, 8);
		Text_line[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_line[i], str);
		}
	for (i = 0; i < nb_lines; i++) {
		a = Klein_rk[i];
		O->unrank_point(w, 1, a, 0 /* verbose_level*/);
			// error corrected: w was v which was v[4], so too short.
			// Aug 25, 2018
		INT_vec_print_to_str(str, w, 6);
			// w was v, error corrected
		Text_klein[i] = NEW_char(strlen(str) + 1);
		strcpy(Text_klein[i], str);
		}

	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_empty_table(nb_lines + 1, 7);
	Sp->fill_column_with_row_index(0, "Idx");
	Sp->fill_column_with_INT(1, Wedge_rk, "Wedge_rk");
	Sp->fill_column_with_text(2, (const char **) Text_wedge, "Wedge coords");
	Sp->fill_column_with_INT(3, Line_rk, "Line_rk");
	Sp->fill_column_with_text(4, (const char **) Text_line, "Line basis");
	Sp->fill_column_with_INT(5, Klein_rk, "Klein_rk");
	Sp->fill_column_with_text(6, (const char **) Text_klein, "Klein coords");
	
	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_wedge[i]);
		}
	FREE_pchar(Text_wedge);
	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_line[i]);
		}
	FREE_pchar(Text_line);
	for (i = 0; i < nb_lines; i++) {
		FREE_char(Text_klein[i]);
		}
	FREE_pchar(Text_klein);


	if (f_v) {
		cout << "surface::make_spreadsheet_of_lines_in_three_kinds done" << endl;
		}
}

void surface::save_lines_in_three_kinds(const char *fname_csv, 
	INT *Lines_wedge, INT *Lines, INT *Lines_klein, INT nb_lines)
{
	spreadsheet *Sp;
	
	make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Lines_wedge, Lines, Lines_klein, nb_lines, 0 /* verbose_level */);

	Sp->save(fname_csv, 0 /*verbose_level*/);
	FREE_OBJECT(Sp);
}

INT surface::line_ai(INT i)
{
	if (i >= 6) {
		cout << "surface::line_ai i >= 6" << endl;
		exit(1);
		}
	return i;
}

INT surface::line_bi(INT i)
{
	if (i >= 6) {
		cout << "surface::line_bi i >= 6" << endl;
		exit(1);
		}
	return 6 + i;
}

INT surface::line_cij(INT i, INT j)
{
	INT a;
	
	if (i > j) {
		return line_cij(j, i);
		}
	if (i == j) {
		cout << "surface::line_cij i==j" << endl;
		exit(1);
		}
	if (i >= 6) {
		cout << "surface::line_cij i >= 6" << endl;
		exit(1);
		}
	if (j >= 6) {
		cout << "surface::line_cij j >= 6" << endl;
		exit(1);
		}
	a = ij2k(i, j, 6);
	return 12 + a;
}

void surface::print_line(ostream &ost, INT rk)
{
	if (rk < 6) {
		ost << "a_" << rk + 1 << endl;
		}
	else if (rk < 12) {
		ost << "b_" << rk - 6 + 1 << endl;
		}
	else {
		INT i, j;
		
		rk -= 12;
		k2ij(rk, i, j, 6);
		ost << "c_{" << i + 1 << j + 1 << "}";
		}
}

void surface::print_Steiner_and_Eckardt(ostream &ost)
{
	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Eckardt Points}" << endl;
	latex_table_of_Eckardt_points(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Tritangent Planes}" << endl;
	latex_table_of_tritangent_planes(ost);

	ost << "\\clearpage" << endl << endl;
	ost << "\\section*{Steiner Trihedral Pairs}" << endl;
	latex_table_of_trihedral_pairs(ost);

}

void surface::latex_abstract_trihedral_pair(ostream &ost, INT t_idx)
{
	latex_trihedral_pair(ost, Trihedral_pairs + t_idx * 9, 
		Trihedral_to_Eckardt + t_idx * 6);
}

void surface::latex_trihedral_pair(ostream &ost, INT *T, INT *TE)
{
	INT i, j;
	
	ost << "\\begin{array}{*{" << 3 << "}{c}|c}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			print_line(ost, T[i * 3 + j]);
			ost << " & ";
			}
		ost << "\\pi_{";
		Eckardt_points[TE[i]].latex_index_only(ost);
		ost << "}\\\\" << endl;
		}
	ost << "\\hline" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\pi_{";
		Eckardt_points[TE[3 + j]].latex_index_only(ost);
		ost << "} & ";
		}
	ost << "\\\\" << endl;
	ost << "\\end{array}" << endl;
}

void surface::latex_table_of_trihedral_pairs(ostream &ost)
{
	INT i;
	
	cout << "surface::latex_table_of_trihedral_pairs" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << "$T_{" << i << "} = T_{" 
			<< Trihedral_pair_labels[i] << "} = \\\\" << endl;
		//ost << "\\left[" << endl;
		//ost << "\\begin{array}" << endl;
		latex_trihedral_pair(ost, Trihedral_pairs + i * 9, 
			Trihedral_to_Eckardt + i * 6);
		//ost << "\\end{array}" << endl;
		//ost << "\\right]" << endl;
		ost << "$\\\\" << endl;
#if 0
		ost << "planes: $";
		INT_vec_print(ost, Trihedral_to_Eckardt + i * 6, 6);
		ost << "$\\\\" << endl;
#endif
		}
	ost << "\\end{multicols}" << endl;

	print_trihedral_pairs(ost);
	
	cout << "surface::latex_table_of_trihedral_pairs done" << endl;
}

void surface::print_trihedral_pairs(ostream &ost)
{
	INT i, j;
	
	ost << "List of trihedral pairs:\\\\" << endl;
	for (i = 0; i < nb_trihedral_pairs; i++) {
		ost << i << " / " << nb_trihedral_pairs 
			<< ": $T_{" << i << "} =  T_{" 
			<< Trihedral_pair_labels[i] << "}=(";
		for (j = 0; j < 6; j++) {
			ost << "\\pi_{" << Trihedral_to_Eckardt[i * 6 + j] 
				<< "}";
			if (j == 2) {
				ost << "; ";
				}
			else if (j < 6 - 1) {
				ost << ", ";
				}
			}
		ost << ")$\\\\" << endl;
		}
	ost << "List of trihedral pairs numerically:\\\\" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		Trihedral_to_Eckardt, 40, 6, 0, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		Trihedral_to_Eckardt + 40 * 6, 40, 6, 40, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels_and_offset(ost, 
		Trihedral_to_Eckardt + 80 * 6, 40, 6, 80, 0, TRUE /* f_tex*/);
	ost << "$$" << endl;
}


void surface::latex_table_of_Eckardt_points(ostream &ost)
{
	INT i, j;
	INT three_lines[3];
	
	cout << "surface::latex_table_of_Eckardt_points" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(this, three_lines);

		ost << "$E_{" << i << "} = " << endl;
		Eckardt_points[i].latex(ost);
		ost << " = ";
		for (j = 0; j < 3; j++) {
			ost << Line_label_tex[three_lines[j]];
			if (j < 3 - 1) {
				ost << " \\cap ";
				}
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	cout << "surface::latex_table_of_Eckardt_points done" << endl;
}

void surface::latex_table_of_tritangent_planes(ostream &ost)
{
	INT i, j;
	INT three_lines[3];
	
	cout << "surface::latex_table_of_tritangent_planes" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	for (i = 0; i < nb_Eckardt_points; i++) {

		Eckardt_points[i].three_lines(this, three_lines);

		ost << "$\\pi_{" << i << "} = \\pi_{" << endl;
		Eckardt_points[i].latex_index_only(ost);
		ost << "} = ";
		for (j = 0; j < 3; j++) {
			ost << Line_label_tex[three_lines[j]];
			}
		ost << "$\\\\" << endl;
		}
	ost << "\\end{multicols}" << endl;
	cout << "surface::latex_table_of_tritangent_planes done" << endl;
}

void surface::find_tritangent_planes_intersecting_in_a_line(INT line_idx, 
	INT &plane1, INT &plane2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx;
	INT three_lines[3];

	if (f_v) {
		cout << "surface::find_tritangent_planes_intersecting_in_a_line" << endl;
		}
	for (plane1 = 0; plane1 < nb_Eckardt_points; plane1++) {

		Eckardt_points[plane1].three_lines(this, three_lines);
		if (INT_vec_search_linear(three_lines, 3, line_idx, idx)) {
			for (plane2 = plane1 + 1; plane2 < nb_Eckardt_points; plane2++) {

				Eckardt_points[plane2].three_lines(this, three_lines);
				if (INT_vec_search_linear(three_lines, 3, line_idx, idx)) {
					if (f_v) {
						cout << "surface::find_tritangent_planes_intersecting_in_a_line done" << endl;
						}
					return;
					}
				}
			}
		}
	cout << "surface::find_tritangent_planes_intersecting_in_a_line could not find two planes" << endl;
	exit(1);
}


void surface::make_trihedral_pairs(INT *&T, 
	char **&T_label, INT &nb_trihedral_pairs, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT h, s, idx;
	INT subset[6];
	INT second_subset[6];
	INT complement[6];
	INT size_complement;
	char label[1000];

	if (f_v) {
		cout << "surface::make_trihedral_pairs" << endl;
		}
	nb_trihedral_pairs = 120;
	T = NEW_INT(nb_trihedral_pairs * 9);
	T_label = NEW_pchar(nb_trihedral_pairs);

	idx = 0;

	// the first type (20 of them):
	for (h = 0; h < 20; h++, idx++) {
		unrank_k_subset(h, subset, 6, 3);
#if 0
		if (h == 16) {
			cout << "h=16: subset=";
			INT_vec_print(cout, subset, 3);
			cout << endl;
			}
#endif
		sprintf(label, "%ld%ld%ld", subset[0] + 1, subset[1] + 1, subset[2] + 1);

		make_Tijk(T + idx * 9, subset[0], subset[1], subset[2]);
		T_label[idx] = NEW_char(strlen(label) + 1);
		strcpy(T_label[idx], label);
#if 0
		if (h == 16) {
			cout << "h=16:T=";
			INT_vec_print(cout, T + idx * 9, 9);
			cout << endl;
			}
#endif
		}

	// the second type (90 of them):
	for (h = 0; h < 15; h++) {
		unrank_k_subset(h, subset, 6, 4);
		for (s = 0; s < 6; s++, idx++) {
			unrank_k_subset(s, second_subset, 4, 2);
			set_complement(second_subset, 2, complement, 
				size_complement, 4);
			make_Tlmnp(T + idx * 9, 
				subset[second_subset[0]], 
				subset[second_subset[1]], 
				subset[complement[0]], 
				subset[complement[1]]);
			sprintf(label, "%ld%ld,%ld%ld", 
				subset[second_subset[0]] + 1, 
				subset[second_subset[1]] + 1, 
				subset[complement[0]] + 1, 
				subset[complement[1]] + 1);
			T_label[idx] = NEW_char(strlen(label) + 1);
			strcpy(T_label[idx], label);
			}
		}

	// the third type (10 of them):
	for (h = 0; h < 10; h++, idx++) {
		unrank_k_subset(h, subset + 1, 5, 2);
		subset[0] = 0;
		subset[1]++;
		subset[2]++;
		set_complement(subset, 3, complement, 
			size_complement, 6);
		make_Tdefght(T + idx * 9, 
			subset[0], subset[1], subset[2], 
			complement[0], complement[1], complement[2]);
		sprintf(label, "%ld%ld%ld,%ld%ld%ld", 
			subset[0] + 1, 
			subset[1] + 1, 
			subset[2] + 1, 
			complement[0] + 1, 
			complement[1] + 1, 
			complement[2] + 1);
		T_label[idx] = NEW_char(strlen(label) + 1);
		strcpy(T_label[idx], label);
		}

	if (idx != 120) {
		cout << "surface::make_trihedral_pairs idx != 120" << endl;
		exit(1);
		}


	if (f_v) {
		cout << "The trihedral pairs are:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			T, 120, 9, FALSE /* f_tex */);
		print_integer_matrix_with_standard_labels(cout, 
			T, 120, 9, TRUE /* f_tex */);
		}

	if (f_v) {
		cout << "surface::make_trihedral_pairs done" << endl;
		}
}

void surface::process_trihedral_pairs(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT subset[3];
	INT i, j, h, rk, a;

	if (f_v) {
		cout << "surface::process_trihedral_pairs" << endl;
		}
	Trihedral_pairs_row_sets = NEW_INT(nb_trihedral_pairs * 3);
	Trihedral_pairs_col_sets = NEW_INT(nb_trihedral_pairs * 3);
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < 3; j++) {
			for (h = 0; h < 3; h++) {
				a = Trihedral_pairs[i * 9 + j * 3 + h];
				subset[h] = a;
				}
			INT_vec_heapsort(subset, 3);
			rk = rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_row_sets[i * 3 + j] = rk;
			}
		}
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < 3; j++) {
			for (h = 0; h < 3; h++) {
				a = Trihedral_pairs[i * 9 + h * 3 + j];
				subset[h] = a;
				}
			INT_vec_heapsort(subset, 3);
			rk = rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_col_sets[i * 3 + j] = rk;
			}
		}

	if (f_v) {
		cout << "surface::process_trihedral_pairs The trihedral pairs row sets:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Trihedral_pairs_row_sets, 120, 3, 
			FALSE /* f_tex */);
		//print_integer_matrix_with_standard_labels(cout, Trihedral_pairs_row_sets, 120, 3, TRUE /* f_tex */);
		cout << "The trihedral pairs col sets:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Trihedral_pairs_col_sets, 120, 3, 
			FALSE /* f_tex */);
		//print_integer_matrix_with_standard_labels(cout, Trihedral_pairs_col_sets, 120, 3, TRUE /* f_tex */);
		}

	Classify_trihedral_pairs_row_values = NEW_OBJECT(classify);
	Classify_trihedral_pairs_row_values->init(
		Trihedral_pairs_row_sets, 120 * 3, FALSE, 0);

	if (f_v) {
		cout << "surface::process_trihedral_pairs sorted row values:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Classify_trihedral_pairs_row_values->data_sorted, 
			120 * 3 / 10, 10, FALSE /* f_tex */);
		//INT_matrix_print(Classify_trihedral_pairs_row_values->data_sorted, 120, 3);
		//cout << endl;
		}

	Classify_trihedral_pairs_col_values = NEW_OBJECT(classify);
	Classify_trihedral_pairs_col_values->init(Trihedral_pairs_col_sets, 
		120 * 3, FALSE, 0);

	if (f_v) {
		cout << "surface::process_trihedral_pairs sorted col values:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Classify_trihedral_pairs_col_values->data_sorted, 
			120 * 3 / 10, 10, FALSE /* f_tex */);
		}
	if (f_v) {
		cout << "surface::process_trihedral_pairs done" << endl;
		}
}

void surface::make_Tijk(INT *T, INT i, INT j, INT k)
{
	T[0] = line_cij(j, k);
	T[1] = line_bi(k);
	T[2] = line_ai(j);
	T[3] = line_ai(k);
	T[4] = line_cij(i, k);
	T[5] = line_bi(i);
	T[6] = line_bi(j);
	T[7] = line_ai(i);
	T[8] = line_cij(i, j);
}

void surface::make_Tlmnp(INT *T, INT l, INT m, INT n, INT p)
{
	INT subset[4];
	INT complement[2];
	INT size_complement;
	INT r, s;

	subset[0] = l;
	subset[1] = m;
	subset[2] = n;
	subset[3] = p;
	INT_vec_heapsort(subset, 4);
	set_complement(subset, 4, complement, size_complement, 6);
	r = complement[0];
	s = complement[1];

	T[0] = line_ai(l);
	T[1] = line_bi(p);
	T[2] = line_cij(l, p);
	T[3] = line_bi(n);
	T[4] = line_ai(m);
	T[5] = line_cij(m, n);
	T[6] = line_cij(l, n);
	T[7] = line_cij(m, p);
	T[8] = line_cij(r, s);
}

void surface::make_Tdefght(INT *T, INT d, INT e, INT f, INT g, INT h, INT t)
{
	T[0] = line_cij(d, g);
	T[1] = line_cij(e, h);
	T[2] = line_cij(f, t);
	T[3] = line_cij(e, t);
	T[4] = line_cij(f, g);
	T[5] = line_cij(d, h);
	T[6] = line_cij(f, h);
	T[7] = line_cij(d, t);
	T[8] = line_cij(e, g);
}

void surface::make_Eckardt_points(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l;
	char str[1000];

	if (f_v) {
		cout << "surface::make_Eckardt_points" << endl;
		}
	nb_Eckardt_points = 45;
	Eckardt_points = NEW_OBJECTS(eckardt_point, nb_Eckardt_points);
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].init_by_rank(i);
		}
	Eckard_point_label = NEW_pchar(nb_Eckardt_points);
	Eckard_point_label_tex = NEW_pchar(nb_Eckardt_points);
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].latex_to_str_without_E(str);
		l = strlen(str);
		Eckard_point_label[i] = NEW_char(l + 1);
		strcpy(Eckard_point_label[i], str);
		Eckard_point_label_tex[i] = NEW_char(l + 1);
		strcpy(Eckard_point_label_tex[i], str);
		}
	if (f_v) {
		cout << "surface::make_Eckardt_points done" << endl;
		}
}


void surface::init_Trihedral_to_Eckardt(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t, i, j, rk;
	INT tritangent_plane[3];

	if (f_v) {
		cout << "surface::init_Trihedral_to_Eckardt" << endl;
		}
	nb_trihedral_to_Eckardt = nb_trihedral_pairs * 6;
	Trihedral_to_Eckardt = NEW_INT(nb_trihedral_to_Eckardt);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
#if 0
			if (t == 111) {
				cout << "surface::init_Trihedral_to_Eckardt t=" << t << " tritangent_plane row " << i << " = ";
				INT_vec_print(cout, tritangent_plane, 3);
				cout << endl;
				}
#endif
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
#if 0
			if (t == 111) {
				cout << "rk=" << rk << endl;
				}
#endif
			Trihedral_to_Eckardt[t * 6 + i] = rk;
			}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
#if 0
			if (t == 5) {
				cout << "surface::init_Trihedral_to_Eckardt tritangent_plane=";
				INT_vec_print(cout, tritangent_plane, 3);
				cout << endl;
				}
#endif
			rk = Eckardt_point_from_tritangent_plane(
				tritangent_plane);
#if 0
			if (t == 5) {
				cout << "rk=" << rk << endl;
				}
#endif
			Trihedral_to_Eckardt[t * 6 + 3 + j] = rk;
			}
		}
	if (f_v) {
		cout << "Trihedral_to_Eckardt:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Trihedral_to_Eckardt, nb_trihedral_pairs, 6, 
			FALSE /* f_tex */);
		}
	if (f_v) {
		cout << "surface::init_Trihedral_to_Eckardt done" << endl;
		}
}


INT surface::Eckardt_point_from_tritangent_plane(INT *tritangent_plane)
{
	INT a, b, c, rk;
	eckardt_point E;

	INT_vec_heapsort(tritangent_plane, 3);
	a = tritangent_plane[0];
	b = tritangent_plane[1];
	c = tritangent_plane[2];
	if (a < 6) {
		E.init2(a, b - 6);
		}
	else {
		if (a < 12) {
			cout << "surface::Eckardt_point_from_tritangent_plane a < 12" << endl;
			exit(1);
			}
		a -= 12;
		b -= 12;
		c -= 12;
		E.init3(a, b, c);
		}
	rk = E.rank();
	return rk;
}

void surface::init_collinear_Eckardt_triples(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t, i, rk;
	INT subset[3];

	if (f_v) {
		cout << "surface::init_collinear_Eckardt_triples" << endl;
		}
	nb_collinear_Eckardt_triples = nb_trihedral_pairs * 2;
	collinear_Eckardt_triples_rank = NEW_INT(nb_collinear_Eckardt_triples);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			INT_vec_copy(Trihedral_to_Eckardt + 6 * t + i * 3, 
				subset, 3);
			INT_vec_heapsort(subset, 3);
			rk = rank_k_subset(subset, nb_Eckardt_points, 3);
			collinear_Eckardt_triples_rank[t * 2 + i] = rk;
			}
		}
	if (f_v) {
		cout << "collinear_Eckardt_triples_rank:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			collinear_Eckardt_triples_rank, nb_trihedral_pairs, 2, 
			FALSE /* f_tex */);
		}

	Classify_collinear_Eckardt_triples = NEW_OBJECT(classify);
	Classify_collinear_Eckardt_triples->init(
		collinear_Eckardt_triples_rank, nb_collinear_Eckardt_triples, 
		FALSE, 0);
	
	if (f_v) {
		cout << "surface::init_collinear_Eckardt_triples done" << endl;
		}
}

void surface::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
	INT *E_idx, INT nb_E, 
	INT *&T_idx, INT &nb_T, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT nCk, h, k, rk, idx, i, t_idx;
	INT subset[3];
	INT set[3];
	
	if (f_v) {
		cout << "surface::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points" << endl;
		}
	nCk = INT_n_choose_k(nb_E, 3);
	T_idx = NEW_INT(nCk);
	nb_T = 0;
	for (h = 0; h < nCk; h++) {
		//cout << "subset " << h << " / " << nCk << ":";
		unrank_k_subset(h, subset, nb_E, 3);
		//INT_vec_print(cout, subset, 3);
		//cout << " = ";

		for (k = 0; k < 3; k++) {
			set[k] = E_idx[subset[k]];
			}
		//INT_vec_print(cout, set, 3);
		//cout << " = ";
		INT_vec_heapsort(set, 3);
		
		rk = rank_k_subset(set, nb_Eckardt_points, 3);


		//INT_vec_print(cout, set, 3);
		//cout << " rk=" << rk << endl;

		if (INT_vec_search(
			Classify_collinear_Eckardt_triples->data_sorted, 
			nb_collinear_Eckardt_triples, rk, idx)) {
			//cout << "idx=" << idx << endl;
			for (i = idx; i >= 0; i--) {
				//cout << "i=" << i << " value=" << Classify_collinear_Eckardt_triples->data_sorted[i] << " collinear triple index = " << Classify_collinear_Eckardt_triples->sorting_perm_inv[i] / 3 << endl;
				if (Classify_collinear_Eckardt_triples->data_sorted[i] != rk) {
					break;
					}
				t_idx = Classify_collinear_Eckardt_triples->sorting_perm_inv[i] / 2;

#if 0
				INT idx2, j;
				
				if (!INT_vec_search(T_idx, nb_T, t_idx, idx2)) {
					for (j = nb_T; j > idx2; j--) {
						T_idx[j] = T_idx[j - 1];
						}
					T_idx[idx2] = t_idx;
					nb_T++;
					}
				else {
					cout << "We already have this trihedral pair" << endl;
					}
#else
				T_idx[nb_T++] = t_idx;
#endif
				}
			}
		
		}


#if 1
	classify C;

	C.init(T_idx, nb_T, TRUE, 0);
	cout << "The trihedral pairs come in these multiplicities: ";
	C.print_naked(TRUE);
	cout << endl;

	INT t2, f2, l2, sz;
	INT t1, f1, /*l1,*/ pt;
	
	for (t2 = 0; t2 < C.second_nb_types; t2++) {
		f2 = C.second_type_first[t2];
		l2 = C.second_type_len[t2];
		sz = C.second_data_sorted[f2];
		if (sz != 1) {
			continue;
			}
		//cout << "clebsch::clebsch_map_print_fibers fibers of size " << sz << ":" << endl;
		//*fp << "There are " << l2 << " fibers of size " << sz << ":\\\\" << endl;
		for (i = 0; i < l2; i++) {
			t1 = C.second_sorting_perm_inv[f2 + i];
			f1 = C.type_first[t1];
			//l1 = C.type_len[t1];
			pt = C.data_sorted[f1];
			T_idx[i] = pt;
#if 0
			//*fp << "Arc pt " << pt << ", fiber $\\{"; // << l1 << " surface points in the list of Pts (local numbering): ";
			for (j = 0; j < l1; j++) {
				u = C.sorting_perm_inv[f1 + j];
				
				cout << u << endl;
				//*fp << u;
				//cout << Pts[u];
				if (j < l1 - 1) {
					cout << ", ";
					}
				}
#endif
			}
		nb_T = l2;
		}
#endif

	

	cout << "Found " << nb_T << " special trihedral pairs:" << endl;
	cout << "T_idx: ";
	INT_vec_print(cout, T_idx, nb_T);
	cout << endl;
	for (i = 0; i < nb_T; i++) {
		cout << i << " / " << nb_T << " T_{" 
			<< Trihedral_pair_labels[T_idx[i]] << "}" << endl;
		}
	if (f_v) {
		cout << "surface::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points done" << endl;
		}
}

void surface::multiply_conic_times_linear(INT *six_coeff, 
	INT *three_coeff, INT *ten_coeff, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, idx;
	INT M[3];

	if (f_v) {
		cout << "surface::multiply_conic_times_linear" << endl;
		}


	INT_vec_zero(ten_coeff, 10);
	for (i = 0; i < 6; i++) {
		a = six_coeff[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff[j];
			c = F->mult(a, b);
			INT_vec_add(Poly2->Monomials + i * 3, 
				Poly1->Monomials + j * 3, M, 3);
			idx = Poly3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "surface::multiply_conic_times_linear idx >= 10" << endl;
				exit(1);
				}
			ten_coeff[idx] = F->add(ten_coeff[idx], c);
			}
		}
	
	
	if (f_v) {
		cout << "surface::multiply_conic_times_linear done" << endl;
		}
}

void surface::multiply_linear_times_linear_times_linear(
	INT *three_coeff1, INT *three_coeff2, INT *three_coeff3, 
	INT *ten_coeff, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, a, b, c, d, idx;
	INT M[3];

	if (f_v) {
		cout << "surface::multiply_linear_times_linear_times_linear" << endl;
		}

	INT_vec_zero(ten_coeff, 10);
	for (i = 0; i < 3; i++) {
		a = three_coeff1[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff2[j];
			for (k = 0; k < 3; k++) {
				c = three_coeff3[k];
				d = F->mult3(a, b, c);
				INT_vec_add3(Poly1->Monomials + i * 3, 
					Poly1->Monomials + j * 3, 
					Poly1->Monomials + k * 3, 
					M, 3);
				idx = Poly3->index_of_monomial(M);
				if (idx >= 10) {
					cout << "surface::multiply_linear_times_linear_times_linear idx >= 10" << endl;
					exit(1);
					}
				ten_coeff[idx] = F->add(ten_coeff[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface::multiply_linear_times_linear_times_linear done" << endl;
		}
}

void surface::multiply_linear_times_linear_times_linear_in_space(
	INT *four_coeff1, INT *four_coeff2, INT *four_coeff3, 
	INT *twenty_coeff, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, a, b, c, d, idx;
	INT M[4];

	if (f_v) {
		cout << "surface::multiply_linear_times_linear_times_linear_in_space" << endl;
		}

	INT_vec_zero(twenty_coeff, 20);
	for (i = 0; i < 4; i++) {
		a = four_coeff1[i];
		for (j = 0; j < 4; j++) {
			b = four_coeff2[j];
			for (k = 0; k < 4; k++) {
				c = four_coeff3[k];
				d = F->mult3(a, b, c);
				INT_vec_add3(Poly1_4->Monomials + i * 4, 
					Poly1_4->Monomials + j * 4, 
					Poly1_4->Monomials + k * 4, 
					M, 4);
				idx = index_of_monomial(M);
				if (idx >= 20) {
					cout << "surface::multiply_linear_times_linear_times_linear_in_space idx >= 20" << endl;
					exit(1);
					}
				twenty_coeff[idx] = F->add(twenty_coeff[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface::multiply_linear_times_linear_times_linear_in_space done" << endl;
		}
}

void surface::multiply_Poly2_3_times_Poly2_3(INT *input1, INT *input2, 
	INT *result, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, idx;
	INT M[3];

	if (f_v) {
		cout << "surface::multiply_Poly2_3_times_Poly2_3" << endl;
		}

	INT_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly2->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			INT_vec_add(Poly2->Monomials + i * 3, 
				Poly2->Monomials + j * 3, 
				M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}
	
	
	if (f_v) {
		cout << "surface::multiply_Poly2_3_times_Poly2_3 done" << endl;
		}
}

void surface::multiply_Poly1_3_times_Poly3_3(INT *input1, INT *input2, 
	INT *result, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, idx;
	INT M[3];

	if (f_v) {
		cout << "surface::multiply_Poly1_3_times_Poly3_3" << endl;
		}

	INT_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly3->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			INT_vec_add(Poly1->Monomials + i * 3, 
				Poly3->Monomials + j * 3, M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}
	
	if (f_v) {
		cout << "surface::multiply_Poly1_3_times_Poly3_3 done" << endl;
		}
}

void surface::web_of_cubic_curves(INT *arc6, INT *&curves, 
	INT verbose_level)
// curves[45 * 10]
{
	INT f_v = (verbose_level >= 1);
	INT *bisecants;
	INT *conics;
	INT ten_coeff[10];
	INT a, rk, i, j, k, l, m, n;
	INT ij, kl, mn;

	if (f_v) {
		cout << "surface::web_of_cubic_curves" << endl;
		}
	P2->compute_bisecants_and_conics(arc6, 
		bisecants, conics, verbose_level);
	
	curves = NEW_INT(45 * 10);

	
	a = 0;

	// the first 30 curves:
	for (rk = 0; rk < 30; rk++, a++) {
		ordered_pair_unrank(rk, i, j, 6);
		ij = ij2k(i, j, 6);
		multiply_conic_times_linear(conics + j * 6, 
			bisecants + ij * 3, 
			ten_coeff, 
			0 /* verbose_level */);
		INT_vec_copy(ten_coeff, curves + a * 10, 10);
		}

	// the next 15 curves:
	for (rk = 0; rk < 15; rk++, a++) {
		unordered_triple_pair_unrank(rk, i, j, k, l, m, n);
		ij = ij2k(i, j, 6);
		kl = ij2k(k, l, 6);
		mn = ij2k(m, n, 6);
		multiply_linear_times_linear_times_linear(
			bisecants + ij * 3, 
			bisecants + kl * 3, 
			bisecants + mn * 3, 
			ten_coeff, 
			0 /* verbose_level */);
		INT_vec_copy(ten_coeff, curves + a * 10, 10);
		}

	if (a != 45) {
		cout << "surface::web_of_cubic_curves a != 45" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "The web of cubic curves is:" << endl;
		INT_matrix_print(curves, 45, 10);
		}

	FREE_INT(bisecants);
	FREE_INT(conics);

	if (f_v) {
		cout << "surface::web_of_cubic_curves done" << endl;
		}
}

void surface::print_web_of_cubic_curves(ostream &ost, 
	INT *Web_of_cubic_curves)
// curves[45 * 10]
{
	ost << "Web of cubic curves:\\\\" << endl;
	ost << "$$" << endl;
	print_integer_matrix_with_standard_labels(ost, 
		Web_of_cubic_curves, 45, 10, TRUE /* f_tex*/);
	ost << "$$" << endl;
}

void surface::create_lines_from_plane_equations(
	INT *The_plane_equations, 
	INT *Lines27, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT line_idx, plane1, plane2;
	INT Basis[16];
	
	if (f_v) {
		cout << "surface::create_lines_from_plane_equations" << endl;
		}

	for (line_idx = 0; line_idx < 27; line_idx++) {
		find_tritangent_planes_intersecting_in_a_line(
			line_idx, plane1, plane2, 0 /* verbose_level */);
		INT_vec_copy(The_plane_equations + plane1 * 4, Basis, 4);
		INT_vec_copy(The_plane_equations + plane2 * 4, Basis + 4, 4);
		F->perp_standard(4, 2, Basis, 0 /* verbose_level */);
		Lines27[line_idx] = rank_line(Basis + 8);
		}
	
	if (f_v) {
		cout << "surface::create_lines_from_plane_equations done" << endl;
		}
}

void surface::web_of_cubic_curves_rank_of_foursubsets(
	INT *Web_of_cubic_curves, 
	INT *&rk, INT &N, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT set[4], i, j, a;
	INT B[4 * 10];

	if (f_v) {
		cout << "surface::web_of_cubic_curves_rank_of_foursubsets" << endl;
		}
	if (f_v) {
		cout << "web of cubic curves:" << endl;
		INT_matrix_print(Web_of_cubic_curves, 45, 10);
		}
	N = INT_n_choose_k(45, 4);
	rk = NEW_INT(N);
	for (i = 0; i < N; i++) {
		unrank_k_subset(i, set, 45, 4);
		if (f_v) {
			cout << "subset " << i << " / " << N << " is ";
			INT_vec_print(cout, set, 4);
			cout << endl;
			}
		for (j = 0; j < 4; j++) {
			a = set[j];
			INT_vec_copy(Web_of_cubic_curves + a * 10, 
				B + j * 10, 10);
			}
		rk[i] = F->rank_of_rectangular_matrix(B, 
			4, 10, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface::web_of_cubic_curves_rank_of_foursubsets done" << endl;
		}
}

void surface::create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
	INT *arc6, INT *base_curves4, 
	INT *&Web_of_cubic_curves, INT *&The_plane_equations, 
	INT verbose_level)
// Web_of_cubic_curves[45 * 10]
{
	INT f_v = (verbose_level >= 1);
	INT h, rk, idx;
	INT *base_curves;
	INT *curves;
	INT *curves_t;

	if (f_v) {
		cout << "surface::create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes" << endl;
		}

	web_of_cubic_curves(arc6, Web_of_cubic_curves, verbose_level);

	base_curves = NEW_INT(5 * 10);
	curves = NEW_INT(5 * 10);
	curves_t = NEW_INT(10 * 5);



	for (h = 0; h < 4; h++) {
		INT_vec_copy(Web_of_cubic_curves + base_curves4[h] * 10, 
			base_curves + h * 10, 10);
		}

	if (f_v) {
		cout << "base_curves:" << endl;
		INT_matrix_print(base_curves, 4, 10);
		}

	// find the plane equations:

	The_plane_equations = NEW_INT(45 * 4);

	for (h = 0; h < 45; h++) {

		if (f_v) {
			cout << "h=" << h << " / " << 45 << ":" << endl;
			}
		
		if (INT_vec_search_linear(base_curves4, 4, h, idx)) {
			INT_vec_zero(The_plane_equations + h * 4, 4);
			The_plane_equations[h * 4 + idx] = 1;
			}
		else {
			INT_vec_copy(base_curves, curves, 4 * 10);
			INT_vec_copy(Web_of_cubic_curves + h * 10, 
				curves + 4 * 10, 10);
		
			if (f_v) {
				cout << "h=" << h << " / " << 45 
					<< " the system is:" << endl;
				INT_matrix_print(curves, 5, 10);
				}

			F->transpose_matrix(curves, curves_t, 5, 10);

			if (f_v) {
				cout << "after transpose:" << endl;
				INT_matrix_print(curves_t, 10, 5);
				}
		
			rk = F->RREF_and_kernel(5, 10, curves_t, 
				0 /* verbose_level */);
			if (rk != 4) {
				cout << "surface::create_surface_and_planes_from_trihedral_pair_and_arc the rank of the system is not equal to 4" << endl;
				cout << "rk = " << rk << endl;
				exit(1);
				}
			if (curves_t[4 * 5 + 4] != F->negate(1)) {
				cout << "h=" << h << " / " << 2 
					<< " curves_t[4 * 5 + 4] != -1" << endl;
				exit(1);
				}
			INT_vec_copy(curves_t + 4 * 5, 
				The_plane_equations + h * 4, 4);

			PG_element_normalize(*F, 
				The_plane_equations + h * 4, 1, 4);
			
			}
		if (f_v) {
			cout << "h=" << h << " / " << 45 
				<< ": the plane equation is ";
			INT_vec_print(cout, The_plane_equations + h * 4, 4);
			cout << endl;
			}
		

		}
	if (f_v) {
		cout << "the plane equations are: " << endl;
		INT_matrix_print(The_plane_equations, 45, 4);
		cout << endl;	
		}

	FREE_INT(base_curves);
	FREE_INT(curves);
	FREE_INT(curves_t);

	if (f_v) {
		cout << "surface::create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes done" << endl;
		}
}

void surface::print_equation_in_trihedral_form(ostream &ost, 
	INT *the_six_plane_equations, INT lambda, INT *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = F_0F_1F_2 + \\lambda G_0G_1G_2\\\\" << endl;
	ost << "& = " << endl;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 0 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 1 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 2 * 4);
	ost << "\\Big)";
	ost << "+ " << lambda;
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 3 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 4 * 4);
	ost << "\\Big)";
	ost << "\\Big(";
	Poly1_4->print_equation(ost, the_six_plane_equations + 5 * 4);
	ost << "\\Big)\\\\";
	ost << "& \\equiv " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface::print_equation_wrapped(ostream &ost, INT *the_equation)
{
	ost << "\\begin{align*}" << endl;
	ost << "0 & = " << endl;
	Poly3_4->print_equation(ost, the_equation);
	ost << "\\\\";
	ost << "\\end{align*}" << endl;
}

void surface::create_equations_for_pencil_of_surfaces_from_trihedral_pair(
	INT *The_six_plane_equations, INT *The_surface_equations, 
	INT verbose_level)
// The_surface_equations[(q + 1) * 20]
{
	INT f_v = (verbose_level >= 1);
	INT v[2];
	INT l;
	INT eqn_F[20];
	INT eqn_G[20];
	INT eqn_F2[20];
	INT eqn_G2[20];

	if (f_v) {
		cout << "surface::create_equations_for_pencil_of_surfaces_from_trihedral_pair" << endl;
		}
	

	for (l = 0; l < q + 1; l++) {
		PG_element_unrank_modified(*F, v, 1, 2, l);
		
		multiply_linear_times_linear_times_linear_in_space(
			The_six_plane_equations + 0 * 4, 
			The_six_plane_equations + 1 * 4, 
			The_six_plane_equations + 2 * 4, 
			eqn_F, FALSE /* verbose_level */);
		multiply_linear_times_linear_times_linear_in_space(
			The_six_plane_equations + 3 * 4, 
			The_six_plane_equations + 4 * 4, 
			The_six_plane_equations + 5 * 4, 
			eqn_G, FALSE /* verbose_level */);

		INT_vec_copy(eqn_F, eqn_F2, 20);
		F->scalar_multiply_vector_in_place(v[0], eqn_F2, 20);
		INT_vec_copy(eqn_G, eqn_G2, 20);
		F->scalar_multiply_vector_in_place(v[1], eqn_G2, 20);
		F->add_vector(eqn_F2, eqn_G2, 
			The_surface_equations + l * 20, 20);
		PG_element_normalize(*F, 
			The_surface_equations + l * 20, 1, 20);
		}

	if (f_v) {
		cout << "surface::create_equations_for_pencil_of_surfaces_from_trihedral_pair done" << endl;
		}
}

void surface::create_lambda_from_trihedral_pair_and_arc(
	INT *arc6, 
	INT *Web_of_cubic_curves, 
	INT *The_plane_equations, INT t_idx, 
	INT &lambda, INT &lambda_rk, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	INT row_col_Eckardt_points[6];
	INT six_curves[6 * 10];
	INT pt, f_point_was_found;
	INT v[3];
	INT w[2];
	INT evals[6];
	INT evals_for_point[6];
	INT pt_on_surface[4];
	INT a, b, ma, bv;

	if (f_v) {
		cout << "surface::create_lambda_from_trihedral_pair_and_arc t_idx=" << t_idx << endl;
		}
	
	if (f_v) {
		cout << "Trihedral pair T_{" << Trihedral_pair_labels[t_idx] << "}" 
			<< endl;
		}

	INT_vec_copy(Trihedral_to_Eckardt + t_idx * 6, 
		row_col_Eckardt_points, 6);
	
	if (f_v) {
		cout << "row_col_Eckardt_points = ";
		INT_vec_print(cout, row_col_Eckardt_points, 6);
		cout << endl;
		}

	

	extract_six_curves_from_web(Web_of_cubic_curves, 
		row_col_Eckardt_points, six_curves, verbose_level);

	if (f_v) {
		cout << "The six curves are:" << endl;
		INT_matrix_print(six_curves, 6, 10);
		}
		


	if (f_v) {
		cout << "surface::create_lambda_from_trihedral_pair_and_arc before find_point_not_on_six_curves" << endl;
		}
	find_point_not_on_six_curves(arc6, six_curves, 
		pt, f_point_was_found, verbose_level);
	if (!f_point_was_found) {
		lambda = 1;
		}
	else {
		if (f_v) {
			cout << "surface::create_lambda_from_trihedral_pair_and_arc after find_point_not_on_six_curves" << endl;
			cout << "pt=" << pt << endl;
			}

		Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			evals[i] = Poly3->evaluate_at_a_point(
				six_curves + i * 10, v);
			}

		if (f_v) {
			cout << "The point pt=" << pt << " = ";
			INT_vec_print(cout, v, 3);
			cout << " is nonzero on all plane sections of the trihedral pair. The values are ";
			INT_vec_print(cout, evals, 6);
			cout << endl;
			}

		if (f_v) {
			cout << "solving for lambda:" << endl;
			}
		a = F->mult3(evals[0], evals[1], evals[2]);
		b = F->mult3(evals[3], evals[4], evals[5]);
		ma = F->negate(a);
		bv = F->inverse(b);
		lambda = F->mult(ma, bv);

#if 1
		pt_on_surface[0] = evals[0];
		pt_on_surface[1] = evals[1];
		pt_on_surface[2] = evals[3];
		pt_on_surface[3] = evals[4];
#endif

		if (FALSE) {
			cout << "lambda = " << lambda << endl;
			}



		for (i = 0; i < 6; i++) {
			evals_for_point[i] = 
				Poly1_4->evaluate_at_a_point(
				The_plane_equations + 
					row_col_Eckardt_points[i] * 4, 
				pt_on_surface);
			}
		a = F->mult3(evals_for_point[0], 
			evals_for_point[1], 
			evals_for_point[2]);
		b = F->mult3(evals_for_point[3], 
			evals_for_point[4], 
			evals_for_point[5]);
		lambda = F->mult(F->negate(a), F->inverse(b));
		if (f_v) {
			cout << "lambda = " << lambda << endl;
			}
		}
	w[0] = 1;
	w[1] = lambda;
	PG_element_rank_modified(*F, w, 1, 2, lambda_rk);
	
	if (f_v) {
		cout << "surface::create_lambda_from_trihedral_pair_and_arc done" << endl;
		}
}


void surface::create_surface_equation_from_trihedral_pair(INT *arc6, 
	INT *Web_of_cubic_curves, 
	INT *The_plane_equations, INT t_idx, INT *surface_equation, 
	INT &lambda, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *The_surface_equations;
	INT row_col_Eckardt_points[6];
	INT The_six_plane_equations[6 * 4];
	INT lambda_rk;

	if (f_v) {
		cout << "surface::create_surface_equation_from_trihedral_pair t_idx=" << t_idx << endl;
		}
	

	INT_vec_copy(Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[0] * 4, The_six_plane_equations, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[1] * 4, The_six_plane_equations + 4, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[2] * 4, The_six_plane_equations + 8, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[3] * 4, The_six_plane_equations + 12, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[4] * 4, The_six_plane_equations + 16, 4);
	INT_vec_copy(The_plane_equations + row_col_Eckardt_points[5] * 4, The_six_plane_equations + 20, 4);


	The_surface_equations = NEW_INT((q + 1) * 20);

	create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations, 
		verbose_level - 2);

	create_lambda_from_trihedral_pair_and_arc(arc6, 
		Web_of_cubic_curves, 
		The_plane_equations, t_idx, lambda, lambda_rk, 
		verbose_level - 2);
	
	INT_vec_copy(The_surface_equations + lambda_rk * 20, 
		surface_equation, 20);

	FREE_INT(The_surface_equations);

	if (f_v) {
		cout << "surface::create_surface_equation_from_trihedral_pair done" << endl;
		}
}

void surface::extract_six_curves_from_web(INT *Web_of_cubic_curves, 
	INT *row_col_Eckardt_points, INT *six_curves, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "surface::extract_six_curves_from_web" << endl;
		}
	for (i = 0; i < 6; i++) {
		INT_vec_copy(Web_of_cubic_curves + row_col_Eckardt_points[i] * 10, 
			six_curves + i * 10, 10);
		}

	if (f_v) {
		cout << "The six curves are:" << endl;
		INT_matrix_print(six_curves, 6, 10);
		}
	if (f_v) {
		cout << "surface::extract_six_curves_from_web done" << endl;
		}
}

void surface::find_point_not_on_six_curves(INT *arc6, INT *six_curves, 
	INT &pt, INT &f_point_was_found, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT v[3];
	INT i;
	INT idx, a;
	
	if (f_v) {
		cout << "surface::find_point_not_on_six_curves" << endl;
		cout << "surface::find_point_not_on_six_curves P2->N_points=" 
			<< P2->N_points << endl;
		}
	pt = -1;
	for (pt = 0; pt < P2->N_points; pt++) {
		if (INT_vec_search_linear(arc6, 6, pt, idx)) {
			continue;
			}
		Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			a = Poly3->evaluate_at_a_point(six_curves + i * 10, v);
			if (a == 0) {
				break;
				}
			}
		if (i == 6) {
			break;
			}
		}
	if (pt == P2->N_points) {
		cout << "could not find a point which is not on any of the curve" << endl;
		f_point_was_found = FALSE;
		}
	else {
		f_point_was_found = TRUE;
		}
	if (f_v) {
		cout << "surface::find_point_not_on_six_curves done" << endl;
		}
}

INT surface::plane_from_three_lines(INT *three_lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT Basis[6 * 4];
	INT rk;
	
	if (f_v) {
		cout << "surface::plane_from_three_lines" << endl;
		}
	unrank_lines(Basis, three_lines, 3);
	rk = F->Gauss_easy(Basis, 6, 4);
	if (rk != 3) {
		cout << "surface::plane_from_three_lines rk != 3" << endl;
		exit(1);
		}
	rk = rank_plane(Basis);
	
	if (f_v) {
		cout << "surface::plane_from_three_lines done" << endl;
		}
	return rk;
}

void surface::Trihedral_pairs_to_planes(INT *Lines, INT *Planes, 
	INT verbose_level)
// Planes[nb_trihedral_pairs * 6]
{
	INT f_v = (verbose_level >= 1);
	INT t, i, j, rk;
	INT tritangent_plane[3];
	INT three_lines[3];

	if (f_v) {
		cout << "surface::Trihedral_pairs_to_planes" << endl;
		}
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
			for (j = 0; j < 3; j++) {
				three_lines[j] = 
					Lines[tritangent_plane[j]];
				}
			rk = plane_from_three_lines(three_lines, 
				0 /* verbose_level */);
			Planes[t * 6 + i] = rk;
			}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
			for (i = 0; i < 3; i++) {
				three_lines[i] = 
					Lines[tritangent_plane[i]];
				}
			rk = plane_from_three_lines(three_lines, 
				0 /* verbose_level */);
			Planes[t * 6 + 3 + j] = rk;
			}
		}
	if (f_v) {
		cout << "Trihedral_pairs_to_planes:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Planes, nb_trihedral_pairs, 6, FALSE /* f_tex */);
		}
	if (f_v) {
		cout << "surface::Trihedral_pairs_to_planes done" << endl;
		}
}

void surface::rearrange_lines_according_to_double_six(INT *Lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Adj;
	INT nb_lines = 27;
	INT New_lines[27];

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six" << endl;
		}
	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six before compute_adjacency_matrix_of_line_intersection_graph" << endl;
		}
	compute_adjacency_matrix_of_line_intersection_graph(Adj, 
		Lines, nb_lines, 0 /* verbose_level */);


	set_of_sets *line_intersections;
	INT *Starter_Table;
	INT nb_starter;

	line_intersections = NEW_OBJECT(set_of_sets);

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six before line_intersections->init_from_adjacency_matrix" << endl;
		}
	line_intersections->init_from_adjacency_matrix(nb_lines, Adj, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six before list_starter_configurations" << endl;
		}
	list_starter_configurations(Lines, nb_lines, 
		line_intersections, Starter_Table, nb_starter,  
		0 /*verbose_level */);

	INT l, line_idx, subset_idx;

	if (nb_starter == 0) {
		cout << "surface::rearrange_lines_according_to_double_six nb_starter == 0" << endl;
		exit(1);
		}
	l = 0;
	line_idx = Starter_Table[l * 2 + 0];
	subset_idx = Starter_Table[l * 2 + 1];

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six before rearrange_lines_according_to_starter_configuration" << endl;
		}
	rearrange_lines_according_to_starter_configuration(
		Lines, New_lines, 
		line_idx, subset_idx, Adj, 
		line_intersections, 0 /*verbose_level*/);

	INT_vec_copy(New_lines, Lines, 27);

	FREE_INT(Adj);
	FREE_INT(Starter_Table);
	FREE_OBJECT(line_intersections);
	if (f_v) {
		cout << "surface::rearrange_lines_according_to_double_six done" << endl;
		}
}

void surface::rearrange_lines_according_to_starter_configuration(
	INT *Lines, INT *New_lines, 
	INT line_idx, INT subset_idx, INT *Adj, 
	set_of_sets *line_intersections, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT S3[6];
	INT i, idx;
	INT nb_lines = 27;


	if (f_v) {
		cout << "surface::rearrange_lines_according_to_starter_configuration" << endl;
		}
	
	create_starter_configuration(line_idx, subset_idx, 
		line_intersections, Lines, S3, 0 /* verbose_level */);
	

	if (f_v) {
		cout << "line_intersections:" << endl;
		line_intersections->print_table();
		}

	INT Line_idx[27];
	for (i = 0; i < 6; i++) {
		if (!INT_vec_search_linear(Lines, nb_lines, S3[i], idx)) {
			cout << "could not find the line" << endl;
			exit(1);
			}
		Line_idx[i] = idx;
		}

	if (f_v) {
		cout << "The 5+1 lines are ";
		INT_vec_print(cout, Line_idx, 6);
		cout << endl;
		}
	
	Line_idx[11] = Line_idx[5];
	Line_idx[5] = 0;
	INT_vec_zero(New_lines, 27);
	INT_vec_copy(S3, New_lines, 5);
	New_lines[11] = S3[5];

	if (f_v) {
		cout << "computing b_j:" << endl;
		}
	for (i = 0; i < 5; i++) {
		INT four_lines[4];

		if (f_v) {
			cout << i << " / " << 5 << ":" << endl;
			}
		
		INT_vec_copy(Line_idx, four_lines, i);
		INT_vec_copy(Line_idx + i + 1, four_lines + i, 5 - i - 1);
		if (f_v) {
			cout << "four_lines=";
			INT_vec_print(cout, four_lines, 4);
			cout << endl;
			}
		
		Line_idx[6 + i] = intersection_of_four_lines_but_not_b6(
			Adj, four_lines, Line_idx[11], verbose_level);
		if (f_v) {
			cout << "b_" << i + 1 << " = " 
				<< Line_idx[6 + i] << endl;
			}
		}

	INT five_lines_idx[5];
	INT_vec_copy(Line_idx + 6, five_lines_idx, 5);
	Line_idx[5] = intersection_of_five_lines(Adj, 
		five_lines_idx, verbose_level);
	if (f_v) {
		cout << "a_" << i + 1 << " = " 
			<< Line_idx[5] << endl;
		}
	

	INT double_six[12];
	INT h, j;
	
	for (i = 0; i < 12; i++) {
		double_six[i] = Lines[Line_idx[i]];
		}
	INT_vec_copy(double_six, New_lines, 12);
	
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(
				double_six, i, j, 
				0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "New_lines:";
		INT_vec_print(cout, New_lines, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface::rearrange_lines_according_to_starter_configuration done" << endl;
		}
}

INT surface::intersection_of_four_lines_but_not_b6(INT *Adj, 
	INT *four_lines_idx, INT b6, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, i, j;

	if (f_v) {
		cout << "surface::intersection_of_four_given_line_intersections_but_not_b6" << endl;
		}
	for (i = 0; i < 27; i++) {
		if (i == b6) {
			continue;
			}
		for (j = 0; j < 4; j++) {
			if (Adj[i * 27 + four_lines_idx[j]] == 0) {
				break;
				}
			}
		if (j == 4) {
			a = i;
			break;
			}
		}
	if (i == 27) {
		cout << "surface::intersection_of_four_lines_but_not_b6 could not find the line" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "surface::intersection_of_four_given_line_intersections_but_not_b6 done" << endl;
		}
	return a;
}

INT surface::intersection_of_five_lines(INT *Adj, 
	INT *five_lines_idx, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, i, j;

	if (f_v) {
		cout << "surface::intersection_of_five_lines" << endl;
		}
	for (i = 0; i < 27; i++) {
		for (j = 0; j < 5; j++) {
			if (Adj[i * 27 + five_lines_idx[j]] == 0) {
				break;
				}
			}
		if (j == 5) {
			a = i;
			break;
			}
		}
	if (i == 27) {
		cout << "surface::intersection_of_five_lines could not find the line" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "surface::intersection_of_five_lines done" << endl;
		}
	return a;
}

void surface::create_surface_family_S(INT a, INT *Lines27, 
	INT *equation20, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface::create_surface_family_S" << endl;
		}

	INT nb_E = 0;
	INT b = 1;
	INT alpha, beta;

	if (f_v) {
		cout << "surface::create_surface_family_S creating surface for a=" << a << ":" << endl;
		}

	create_surface_ab(a, b, Lines27,
		alpha, beta, nb_E, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface::create_surface_family_S The double six is:" << endl;
		INT_matrix_print(Lines27, 2, 6);
		cout << "The lines are : ";
		INT_vec_print(cout, Lines27, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface::create_surface_family_S before create_equation_Sab" << endl;
		}
	create_equation_Sab(a, b, equation20, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::create_surface_family_S after create_equation_Sab" << endl;
		}
	
	if (f_v) {
		cout << "surface::create_surface_family_S done" << endl;
		}
}

#if 0
void surface::create_surface(INT f_iso, INT iso, INT f_S, INT a, INT f_equation, const char *eqn_txt, 
	INT f_equation_is_given, INT *given_equation, 
	INT f_has_trihedral_pair, INT t_idx, INT *nine_lines, 
	INT *Lines, INT *equation, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "surface::create_surface" << endl;
		if (f_iso) {
			cout << "surface::create_surface f_iso" << endl;
			}
		if (f_S) {
			cout << "surface::create_surface f_S" << endl;
			}
		if (f_equation) {
			cout << "surface::create_surface f_equation" << endl;
			}
		if (f_equation_is_given) {
			cout << "surface::create_surface f_equation_is_given" << endl;
			}
		if (f_has_trihedral_pair) {
			cout << "surface::create_surface f_has_trihedral_pair" << endl;
			}
		}
	
	INT_vec_zero(equation, 20);
	
	if (f_S) {
		INT nb_E = 0;
		INT b = 1;
		INT alpha, beta;

		if (f_v) {
			cout << "surface::create_surface creating surface for a=" << a << ":" << endl;
			}
	
		create_surface_ab(a, b, Lines,
			alpha, beta, nb_E, 
			0 /* verbose_level */);

		if (f_v) {
			cout << "surface::create_surface The double six is:" << endl;
			INT_matrix_print(Lines, 2, 6);
			cout << "The lines are : ";
			INT_vec_print(cout, Lines, 27);
			cout << endl;
			}

		create_equation_Sab(a, b, equation, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface::create_surface surface created for a=" << a << " nb_E = " << nb_E << endl;
			}
		}
	else if (f_iso) {
		INT *p_lines;
		INT nb_iso;
		INT nb_E = 0;

		nb_iso = cubic_surface_nb_reps(q);
		if (iso >= nb_iso) {
			cout << "iso >= nb_iso, this cubic surface does not exist" << endl;
			exit(1);
			}
		p_lines = cubic_surface_Lines(q, iso);
		INT_vec_copy(p_lines, Lines, 27);
		nb_E = cubic_surface_nb_Eckardt_points(q, iso);
		rearrange_lines_according_to_double_six(Lines, 0 /* verbose_level */);
		build_cubic_surface_from_lines(27, Lines, equation, 0 /* verbose_level */);
		}
	else if (f_equation) {
		INT *data;
		INT len;
		INT c, d;
		INT nb_lines;
		INT *Points;
		INT nb_points;
		
		INT_vec_scan(eqn_txt, data, len);
		INT_vec_zero(equation, 20);
		if (len % 2) {
			cout << "surface::create_surface description must have an even number of terms" << endl;
			exit(1);
			}
		for (i = 0; i < len; i++) {
			c = data[i];
			if (c < 0 || c >= q) {
				cout << "surface::create_surface coefficient out of range" << endl;
				exit(1);
				}
			i++;
			d = data[i];
			if (d < 0 || d >= 20) {
				cout << "surface::create_surface variable out of range" << endl;
				exit(1);
				}
			equation[d] = c;
			}
		if (f_v) {
			cout << "surface::create_surface The equation is ";
			INT_vec_print(cout, equation, 20);
			cout << endl;
			}


		Points = NEW_INT(nb_pts_on_surface);
		enumerate_points(equation, Points, nb_points, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface::create_surface The surface to be identified has " << nb_points << " points" << endl;
			}
		if (nb_points != nb_pts_on_surface) {
			cout << "nb_points != Surf->nb_pts_on_surface" << endl;
			exit(1);
			}
		
		P->find_lines_which_are_contained(Points, nb_points, Lines, nb_lines, 27 /* max_lines */, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface::create_surface The surface has " << nb_lines << " lines" << endl;
			}
		if (nb_lines != 27) {
			cout << "surface::create_surface something is wrong with the input surface, skipping" << endl;
			FREE_INT(Points);
			FREE_INT(Lines);
			return;
			}

		
		FREE_INT(Points);

		//cout << "STOP" << endl;
		//exit(1);
		
		}
	else if (f_equation_is_given) {
		INT nb_lines;
		INT *Points;
		INT nb_points;
		
		INT_vec_copy(given_equation, equation, 20);
		
		Points = NEW_INT(nb_pts_on_surface);
		enumerate_points(equation, Points, nb_points, 0 /* verbose_level */);
		if (f_v) {
			cout << "surface::create_surface The surface to be identified has " << nb_points << " points" << endl;
			}
		if (nb_points != nb_pts_on_surface) {
			cout << "nb_points != Surf->nb_pts_on_surface" << endl;
			exit(1);
			}
		
		P->find_lines_which_are_contained(Points, nb_points, Lines, nb_lines, 27 /* max_lines */, 0 /* verbose_level */);

		if (f_v) {
			cout << "surface::create_surface The surface has " << nb_lines << " lines" << endl;
			}
		if (nb_lines != 27) {
			cout << "surface::create_surface something is wrong with the input surface, skipping" << endl;
			FREE_INT(Points);
			FREE_INT(Lines);
			return;
			}

		rearrange_lines_according_to_double_six(Lines, verbose_level);
		FREE_INT(Points);

		//cout << "STOP" << endl;
		//exit(1);
		}
	else {
		cout << "surface::create_surface do not know which surface to create" << endl;
		exit(1);
		}


	rearrange_lines_according_to_double_six(Lines, verbose_level);



	if (f_has_trihedral_pair) {
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair" << endl;
			}

		if (f_v) {
			cout << "surface::create_surface rearranging lines according to the given trihedral pair with nine_lines=";
			INT_vec_print(cout, nine_lines, 9);
			cout << endl;
			}
		INT double_sixes[6 * 12]; // there are 6 double sixes for each trihedral pair

		surface_object *SO;
		INT eqn[20];

		build_cubic_surface_from_lines(27, Lines, eqn, 0 /* verbose_level */);
		SO = NEW_OBJECT(surface_object);

		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->init" << endl;
			}
		SO->init(this, Lines, eqn, verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->enumerate_points" << endl;
			}
		SO->enumerate_points(verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->compute_plane_type_by_points" << endl;
			}
		SO->compute_plane_type_by_points(verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->compute_tritangent_planes" << endl;
			}
		SO->compute_tritangent_planes(verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->compute_planes_and_dual_point_ranks" << endl;
			}
		SO->compute_planes_and_dual_point_ranks(verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before SO->identify_double_six_from_trihedral_pair" << endl;
			}
		SO->identify_double_six_from_trihedral_pair(Lines, t_idx, nine_lines, double_sixes, verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair after SO->identify_double_six_from_trihedral_pair" << endl;
			}
		

		if (f_v) {
			cout << "surface::create_surface before FREE_OBJECT SO" << endl;
			}
		FREE_OBJECT(SO);
		if (f_v) {
			cout << "surface::create_surface after FREE_OBJECT SO" << endl;
			}
		
		INT New_lines[27];

		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair before rearrange_lines_according_to_a_given_double_six" << endl;
			}
		rearrange_lines_according_to_a_given_double_six(Lines, 
			New_lines, double_sixes, verbose_level - 2);
		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair after rearrange_lines_according_to_a_given_double_six" << endl;
			}

		INT_vec_copy(New_lines, Lines, 27);

		if (f_v) {
			cout << "surface::create_surface f_has_trihedral_pair  The n e w lines are:" << endl;
			INT_vec_print(cout, Lines, 27);
			cout << endl;
			}

		
		}

	if (f_v) {
		cout << "surface::create_surface done" << endl;
		}
}
#endif

void surface::rearrange_lines_according_to_a_given_double_six(
	INT *Lines, 
	INT *New_lines, INT *double_six, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, h;


	if (f_v) {
		cout << "surface::rearrange_lines_according_to_a_given_double_six" << endl;
		}
	for (i = 0; i < 12; i++) {
		New_lines[i] = Lines[double_six[i]];
		}
	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++, h++) {
			New_lines[12 + h] = compute_cij(
				New_lines /*double_six */, 
				i, j, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "New_lines:";
		INT_vec_print(cout, New_lines, 27);
		cout << endl;
		}
	
	if (f_v) {
		cout << "surface::rearrange_lines_according_to_a_given_double_six done" << endl;
		}
}

void surface::compute_tritangent_planes(INT *Lines, 
	INT *&Tritangent_planes, INT &nb_tritangent_planes, 
	INT *&Unitangent_planes, INT &nb_unitangent_planes, 
	INT *&Lines_in_tritangent_plane, 
	INT *&Line_in_unitangent_plane, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Inc_lines_planes;
	INT *The_plane_type;
	INT nb_planes;
	INT i, j, h, c;

	if (f_v) {
		cout << "surface::compute_tritangent_planes" << endl;
		}
	if (f_v) {
		cout << "Lines=" << endl;
		INT_vec_print(cout, Lines, 27);
		cout << endl;
		}
	P->line_plane_incidence_matrix_restricted(Lines, 27, 
		Inc_lines_planes, nb_planes, 0 /* verbose_level */);

	The_plane_type = NEW_INT(nb_planes);
	INT_vec_zero(The_plane_type, nb_planes);

	for (j = 0; j < nb_planes; j++) {
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				The_plane_type[j]++;
				}
			}
		}
	classify Plane_type;

	Plane_type.init(The_plane_type, nb_planes, FALSE, 0);
	if (f_v) {
		cout << "surface::compute_tritangent_planes The plane type is: ";
		Plane_type.print_naked(TRUE);
		cout << endl;
		}


	Plane_type.get_class_by_value(Tritangent_planes, 
		nb_tritangent_planes, 3 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::compute_tritangent_planes The tritangent planes are: ";
		INT_vec_print(cout, Tritangent_planes, nb_tritangent_planes);
		cout << endl;
		}
	Lines_in_tritangent_plane = NEW_INT(nb_tritangent_planes * 3);
	for (h = 0; h < nb_tritangent_planes; h++) {
		j = Tritangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Lines_in_tritangent_plane[h * 3 + c++] = i;
				}
			}
		if (c != 3) {
			cout << "surface::compute_tritangent_planes c != 3" << endl;
			exit(1);
			}
		}


	Plane_type.get_class_by_value(Unitangent_planes, 
		nb_unitangent_planes, 1 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface::compute_tritangent_planes The unitangent planes are: ";
		INT_vec_print(cout, Unitangent_planes, nb_unitangent_planes);
		cout << endl;
		}
	Line_in_unitangent_plane = NEW_INT(nb_unitangent_planes);
	for (h = 0; h < nb_unitangent_planes; h++) {
		j = Unitangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Line_in_unitangent_plane[h * 1 + c++] = i;
				}
			}
		if (c != 1) {
			cout << "surface::compute_tritangent_planes c != 1" << endl;
			exit(1);
			}
		}

	FREE_INT(Inc_lines_planes);
	FREE_INT(The_plane_type);

	if (f_v) {
		cout << "surface::compute_tritangent_planes done" << endl;
		}
}

void surface::compute_external_lines_on_three_tritangent_planes(
	INT *Lines, INT *&External_lines, INT &nb_external_lines, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	
	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes" << endl;
		}

	INT *Tritangent_planes;
	INT nb_tritangent_planes;
	INT *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]
	
	INT *Unitangent_planes;
	INT nb_unitangent_planes;
	INT *Line_in_unitangent_plane; // [nb_unitangent_planes]
	
	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes computing tritangent planes:" << endl;
		}
	compute_tritangent_planes(Lines, 
		Tritangent_planes, nb_tritangent_planes, 
		Unitangent_planes, nb_unitangent_planes, 
		Lines_in_tritangent_plane, 
		Line_in_unitangent_plane, 
		verbose_level);

	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes Lines_in_tritangent_plane: " << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Lines_in_tritangent_plane, nb_tritangent_planes, 
			3, FALSE);
		}

	INT *Intersection_matrix; // [nb_tritangent_planes * nb_tritangent_planes]
	INT *Plane_intersections;
	INT *Plane_intersections_general;
	INT rk, idx;



	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes Computing intersection matrix of tritangent planes:" << endl;
		}
		
	P->plane_intersection_matrix_in_three_space(Tritangent_planes, 
		nb_tritangent_planes, Intersection_matrix, 
		0 /* verbose_level */);

	Plane_intersections = NEW_INT(nb_tritangent_planes * nb_tritangent_planes);
	Plane_intersections_general = NEW_INT(nb_tritangent_planes * nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		for (j = 0; j < nb_tritangent_planes; j++) {
			Plane_intersections[i * nb_tritangent_planes + j] = -1;
			Plane_intersections_general[i * nb_tritangent_planes + j] = -1;
			if (j != i) {
				rk = Intersection_matrix[i * nb_tritangent_planes + j];
				if (INT_vec_search_linear(Lines, 27, rk, idx)) {
					Plane_intersections[i * nb_tritangent_planes + j] = idx;
					}
				else {
					Plane_intersections_general[i * nb_tritangent_planes + j] = rk;
					}
				}
			}
		}

	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes The tritangent planes intersecting in surface lines:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Plane_intersections, nb_tritangent_planes, 
			nb_tritangent_planes, FALSE);
		}


	classify Plane_intersection_type;

	Plane_intersection_type.init(Plane_intersections, 
		nb_tritangent_planes * nb_tritangent_planes, TRUE, 0);
	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes The surface lines in terms of plane intersections are: ";
		Plane_intersection_type.print_naked(TRUE);
		cout << endl;
		}


	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes The tritangent planes intersecting in general lines:" << endl;
		print_integer_matrix_with_standard_labels(cout, Plane_intersections_general, nb_tritangent_planes, nb_tritangent_planes, FALSE);
		}

	classify Plane_intersection_type2;
	
	Plane_intersection_type2.init(Plane_intersections_general, 
		nb_tritangent_planes * nb_tritangent_planes, TRUE, 0);
	if (f_v) {
		cout << "The other lines in terms of plane intersections are: ";
		Plane_intersection_type2.print_naked(TRUE);
		cout << endl;
		}


	Plane_intersection_type2.get_data_by_multiplicity(
		External_lines, nb_external_lines, 6, 0 /* verbose_level */);

	INT_vec_heapsort(External_lines, nb_external_lines);

	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes The non-surface lines which are on three tritangent planes are:" << endl;
		INT_vec_print(cout, External_lines, nb_external_lines);
		cout << endl;
		cout << "surface::compute_external_lines_on_three_tritangent_planes these lines are:" << endl;
		P->Grass_lines->print_set(External_lines, nb_external_lines);
		}
	
	FREE_INT(Tritangent_planes);
	FREE_INT(Lines_in_tritangent_plane);
	FREE_INT(Unitangent_planes);
	FREE_INT(Line_in_unitangent_plane);
	FREE_INT(Intersection_matrix);
	FREE_INT(Plane_intersections);
	FREE_INT(Plane_intersections_general);

	if (f_v) {
		cout << "surface::compute_external_lines_on_three_tritangent_planes done" << endl;
		}
}

void surface::init_double_sixes(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, ij, u, v, l, m, n, h, a, b, c;
	INT set[6];
	INT size_complement;
	
	if (f_v) {
		cout << "surface::init_double_sixes" << endl;
		}
	Double_six = NEW_INT(36 * 12);
	h = 0;
	// first type: D : a_1,..., a_6; b_1, ..., b_6
	for (i = 0; i < 12; i++) {
		Double_six[h * 12 + i] = i;
		}
	h++;
	
	// second type: 
	// D_{ij} : 
	// a_1, b_1, c_23, c_24, c_25, c_26; 
	// a_2, b_2, c_13, c_14, c_15, c_16
	for (ij = 0; ij < 15; ij++, h++) {
		//cout << "second type " << ij << " / " << 15 << endl;
		k2ij(ij, i, j, 6);
		set[0] = i;
		set[1] = j;
		set_complement(set, 2 /* subset_size */, set + 2, 
			size_complement, 6 /* universal_set_size */);
		//cout << "set : ";
		//INT_vec_print(cout, set, 6);
		//cout << endl;
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_bi(i);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 2 + u] = line_cij(j, set[2 + u]);
			}
		Double_six[h * 12 + 6] = line_ai(j);
		Double_six[h * 12 + 7] = line_bi(j);
		for (u = 0; u < 4; u++) {
			Double_six[h * 12 + 8 + u] = line_cij(i, set[2 + u]);
			}
		}

	// third type: D_{ijk} : 
	// a_1, a_2, a_3, c_56, c_46, c_45; 
	// c_23, c_13, c_12, b_4, b_5, b_6 
	for (v = 0; v < 20; v++, h++) {
		//cout << "third type " << v << " / " << 20 << endl;
		unrank_k_subset(v, set, 6, 3);
		set_complement(set, 3 /* subset_size */, set + 3, 
			size_complement, 6 /* universal_set_size */);
		i = set[0];
		j = set[1];
		k = set[2];
		l = set[3];
		m = set[4];
		n = set[5];
		Double_six[h * 12 + 0] = line_ai(i);
		Double_six[h * 12 + 1] = line_ai(j);
		Double_six[h * 12 + 2] = line_ai(k);
		Double_six[h * 12 + 3] = line_cij(m, n);
		Double_six[h * 12 + 4] = line_cij(l, n);
		Double_six[h * 12 + 5] = line_cij(l, m);
		Double_six[h * 12 + 6] = line_cij(j, k);
		Double_six[h * 12 + 7] = line_cij(i, k);
		Double_six[h * 12 + 8] = line_cij(i, j);
		Double_six[h * 12 + 9] = line_bi(l);
		Double_six[h * 12 + 10] = line_bi(m);
		Double_six[h * 12 + 11] = line_bi(n);
		}

	if (h != 36) {
		cout << "surface::init_double_sixes h != 36" << endl;
		exit(1);
		}

	Double_six_label_tex = NEW_pchar(36);
	char str[1000];

	for (i = 0; i < 36; i++) {
		if (i < 1) {
			sprintf(str, "D");
			}
		else if (i < 1 + 15) {
			ij = i - 1;
			k2ij(ij, a, b, 6);
			set[0] = a;
			set[1] = b;
			set_complement(set, 2 /* subset_size */, set + 2, 
				size_complement, 6 /* universal_set_size */);
			sprintf(str, "D_{%ld%ld}", a + 1, b + 1);
			}
		else {
			v = i - 16;
			unrank_k_subset(v, set, 6, 3);
			set_complement(set, 3 /* subset_size */, set + 3, 
				size_complement, 6 /* universal_set_size */);
			a = set[0];
			b = set[1];
			c = set[2];
			sprintf(str, "D_{%ld%ld%ld}", a + 1, b + 1, c + 1);
			}
		if (f_v) {
			cout << "creating label " << str 
				<< " for Double six " << i << endl;
			}
		l = strlen(str);
		Double_six_label_tex[i] = NEW_char(l + 1);
		strcpy(Double_six_label_tex[i], str);
		}

	if (f_v) {
		cout << "surface::init_double_sixes done" << endl;
		}
}

void surface::create_half_double_sixes(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, c, ij, v, l;
	INT set[6];
	INT size_complement;
	
	if (f_v) {
		cout << "surface::create_half_double_sixes" << endl;
		}
	Half_double_sixes = NEW_INT(72 * 6);
	Half_double_six_to_double_six = NEW_INT(72);
	Half_double_six_to_double_six_row = NEW_INT(72);

	INT_vec_copy(Double_six, Half_double_sixes, 36 * 12);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			INT_vec_heapsort(
				Half_double_sixes + (2 * i + j) * 6, 6);
			Half_double_six_to_double_six[2 * i + j] = i;
			Half_double_six_to_double_six_row[2 * i + j] = j;
			}
		}
	Half_double_six_label_tex = NEW_pchar(72);
	char str[1000];

	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			if (i < 1) {
				sprintf(str, "D");
				}
			else if (i < 1 + 15) {
				ij = i - 1;
				k2ij(ij, a, b, 6);
				set[0] = a;
				set[1] = b;
				set_complement(set, 2 /* subset_size */, 
					set + 2, size_complement, 
					6 /* universal_set_size */);
				sprintf(str, "D_{%ld%ld}", a + 1, b + 1);
				}
			else {
				v = i - 16;
				unrank_k_subset(v, set, 6, 3);
				set_complement(set, 3 /* subset_size */, 
					set + 3, size_complement, 
					6 /* universal_set_size */);
				a = set[0];
				b = set[1];
				c = set[2];
				sprintf(str, "D_{%ld%ld%ld}", 
					a + 1, b + 1, c + 1);
				}
			if (j == 0) {
				sprintf(str + strlen(str), "^\\top");
				}
			else {
				sprintf(str + strlen(str), "^\\bot");
				}
			if (f_v) {
				cout << "creating label " << str 
					<< " for half double six " 
					<< 2 * i + j << endl;
				}
			l = strlen(str);
			Half_double_six_label_tex[2 * i + j] = NEW_char(l + 1);
			strcpy(Half_double_six_label_tex[2 * i + j], str);
			}
		}

	if (f_v) {
		cout << "surface::create_half_double_sixes done" << endl;
		}
}

INT surface::find_half_double_six(INT *half_double_six)
{
	INT i;

	INT_vec_heapsort(half_double_six, 6);
	for (i = 0; i < 72; i++) {
		if (INT_vec_compare(half_double_six, 
			Half_double_sixes + i * 6, 6) == 0) {
			return i;
			}
		}
	cout << "surface::find_half_double_six did not find half double six" << endl;
	exit(1);
}

INT surface::type_of_line(INT line)
// 0 = a_i, 1 = b_i, 2 = c_ij
{
	if (line < 6) {
		return 0;
		}
	else if (line < 12) {
		return 1;
		}
	else if (line < 27) {
		return 2;
		}
	else {
		cout << "surface::type_of_line error" << endl;
		exit(1);
		}
}

void surface::index_of_line(INT line, INT &i, INT &j)
// returns i for a_i, i for b_i and (i,j) for c_ij 
{
	INT a;
	
	if (line < 6) { // ai
		i = line;
		}
	else if (line < 12) { // bj
		i = line - 6;
		}
	else if (line < 27) { // c_ij
		a = line - 12;
		k2ij(a, i, j, 6);
		}
	else {
		cout << "surface::index_of_line error" << endl;
		exit(1);
		}
}

void surface::ijklm2n(INT i, INT j, INT k, INT l, INT m, INT &n)
{
	INT v[6];
	INT size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	v[4] = m;
	set_complement_safe(v, 5, v + 5, size_complement, 6);
	if (size_complement != 1) {
		cout << "surface::ijklm2n size_complement != 1" << endl;
		exit(1);
		}
	n = v[5];
}

void surface::ijkl2mn(INT i, INT j, INT k, INT l, INT &m, INT &n)
{
	INT v[6];
	INT size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	set_complement_safe(v, 4, v + 4, size_complement, 6);
	if (size_complement != 2) {
		cout << "surface::ijkl2mn size_complement != 2" << endl;
		exit(1);
		}
	m = v[4];
	n = v[5];
}

void surface::ijk2lmn(INT i, INT j, INT k, INT &l, INT &m, INT &n)
{
	INT v[6];
	INT size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	cout << "surface::ijk2lmn v=";
	INT_vec_print(cout, v, 3);
	cout << endl;
	set_complement_safe(v, 3, v + 3, size_complement, 6);
	if (size_complement != 3) {
		cout << "surface::ijk2lmn size_complement != 3" << endl;
		cout << "size_complement=" << size_complement << endl;
		exit(1);
		}
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface::ij2klmn(INT i, INT j, INT &k, INT &l, INT &m, INT &n)
{
	INT v[6];
	INT size_complement;

	v[0] = i;
	v[1] = j;
	set_complement_safe(v, 2, v + 2, size_complement, 6);
	if (size_complement != 4) {
		cout << "surface::ij2klmn size_complement != 4" << endl;
		exit(1);
		}
	k = v[2];
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface::get_half_double_six_associated_with_Clebsch_map(
	INT line1, INT line2, INT transversal, 
	INT hds[6],
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT t1, t2, t3;
	INT i, j, k, l, m, n;
	INT i1, j1;
	INT null;
	
	if (f_v) {
		cout << "surface::get_half_double_six_associated_with_Clebsch_map" << endl;
		}

	if (line1 > line2) {
		cout << "surface::get_half_double_six_associated_with_Clebsch_map line1 > line2" << endl;
		exit(1);
		}
	t1 = type_of_line(line1);
	t2 = type_of_line(line2);
	t3 = type_of_line(transversal);

	if (f_v) {
		cout << "t1=" << t1 << " t2=" << t2 << " t3=" << t3 << endl;
		}
	if (t1 == 0 && t2 == 0) { // ai and aj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 1) { // bk
			index_of_line(transversal, k, null);
			//cout << "i=" << i << " j=" << j << " k=" << k <<< endl;
			ijk2lmn(i, j, k, l, m, n);
			// bl, bm, bn, cij, cik, cjk
			hds[0] = line_bi(l);
			hds[1] = line_bi(m);
			hds[2] = line_bi(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
			}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
				// test whether {i1,j1} =  {i,j}
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// bi, bj, bk, bl, bm, bn
				hds[0] = line_bi(i);
				hds[1] = line_bi(j);
				hds[2] = line_bi(k);
				hds[3] = line_bi(l);
				hds[4] = line_bi(m);
				hds[5] = line_bi(n);
				}
			else {
				cout << "surface::get_half_doble_six_associated_with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
				}
			}
		}
	else if (t1 == 1 && t2 == 1) { // bi and bj:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (t3 == 0) { // ak
			index_of_line(transversal, k, null);
			ijk2lmn(i, j, k, l, m, n);
			// al, am, an, cij, cik, cjk
			hds[0] = line_ai(l);
			hds[1] = line_ai(m);
			hds[2] = line_ai(n);
			hds[3] = line_cij(i, j);
			hds[4] = line_cij(i, k);
			hds[5] = line_cij(j, k);
			}
		else if (t3 == 2) { // cij
			index_of_line(transversal, i1, j1);
			if ((i == i1 && j == j1) || (i == j1 && j == i1)) {
				ij2klmn(i, j, k, l, m, n);
				// ai, aj, ak, al, am, an
				hds[0] = line_ai(i);
				hds[1] = line_ai(j);
				hds[2] = line_ai(k);
				hds[3] = line_ai(l);
				hds[4] = line_ai(m);
				hds[5] = line_ai(n);
				}
			else {
				cout << "surface::get_half_doble_six_associated_with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
				}
			}
		}
	else if (t1 == 0 && t2 == 1) { // ai and bi:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (j != i) {
			cout << "surface::get_half_double_six_associated_with_Clebsch_map j != i" << endl;
			exit(1);
			}
		if (t3 != 2) {
			cout << "surface::get_half_double_six_associated_with_Clebsch_map t3 != 2" << endl;
			exit(1);
			}
		index_of_line(transversal, i1, j1);
		if (i1 == i) {
			j = j1;
			}
		else {
			j = i1;
			}
		ij2klmn(i, j, k, l, m, n);
		// cik, cil, cim, cin, aj, bj
		hds[0] = line_cij(i, k);
		hds[1] = line_cij(i, l);
		hds[2] = line_cij(i, m);
		hds[3] = line_cij(i, n);
		hds[4] = line_ai(j);
		hds[5] = line_bi(j);
		}
	else if (t1 == 1 && t2 == 2) { // bi and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
				}
			else if (j1 == i) {
				l = i1;
				}
			else {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, aj, ak, al, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_ai(j);
			hds[3] = line_ai(k);
			hds[4] = line_ai(l);
			hds[5] = line_cij(n, m);
			}
		else if (t3 == 0) { // aj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				INT tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// ak, cil, cim, cin, bk, cij
			hds[0] = line_ai(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_bi(k);
			hds[5] = line_cij(i, j);
			}
		}
	else if (t1 == 0 && t2 == 2) { // ai and cjk:
		index_of_line(line1, i, null);
		index_of_line(line2, j, k);
		if (t3 == 2) { // cil
			index_of_line(transversal, i1, j1);
			if (i1 == i) {
				l = j1;
				}
			else if (j1 == i) {
				l = i1;
				}
			else {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijkl2mn(i, j, k, l, m, n);
			// cin, cim, bj, bk, bl, cnm
			hds[0] = line_cij(i, n);
			hds[1] = line_cij(i, m);
			hds[2] = line_bi(j);
			hds[3] = line_bi(k);
			hds[4] = line_bi(l);
			hds[5] = line_cij(n, m);
			}
		else if (t3 == 1) { // bj
			index_of_line(transversal, j1, null);
			if (j1 == k) {
				// swap k and j
				INT tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// bk, cil, cim, cin, ak, cij
			hds[0] = line_bi(k);
			hds[1] = line_cij(i, l);
			hds[2] = line_cij(i, m);
			hds[3] = line_cij(i, n);
			hds[4] = line_ai(k);
			hds[5] = line_cij(i, j);
			}
		}
	else if (t1 == 2 && t2 == 2) { // cij and cik:
		index_of_line(line1, i, j);
		index_of_line(line2, i1, j1);
		if (i == i1) {
			k = j1;
			}
		else if (i == j1) {
			k = i1;
			}
		else if (j == i1) {
			j = i;
			i = i1;
			k = j1;
			}
		else if (j == j1) {
			j = i;
			i = j1;
			k = i1;
			}
		else {
			cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
			exit(1);
			}
		if (t3 == 0) { // ai
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// bi, clm, cnm, cln, bj, bk
			hds[0] = line_bi(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_bi(j);
			hds[5] = line_bi(k);
			}
		else if (t3 == 1) { // bi
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
				exit(1);
				}
			ijk2lmn(i, j, k, l, m, n);
			// ai, clm, cnm, cln, aj, ak
			hds[0] = line_ai(i);
			hds[1] = line_cij(l, m);
			hds[2] = line_cij(n, m);
			hds[3] = line_cij(l, n);
			hds[4] = line_ai(j);
			hds[5] = line_ai(k);
			}
		else if (t3 == 2) { // clm
			index_of_line(transversal, l, m);
			ijklm2n(i, j, k, l, m, n);
			// ai, bi, cmn, cln, ckn, cjn
			hds[0] = line_ai(i);
			hds[1] = line_bi(i);
			hds[2] = line_cij(m, n);
			hds[3] = line_cij(l, n);
			hds[4] = line_cij(k, n);
			hds[5] = line_cij(j, n);
			}
		}
	else {
		cout << "surface::get_half_double_six_associated_with_Clebsch_map error" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface::get_half_double_six_associated_with_Clebsch_map done" << endl;
		}
}

void surface::prepare_clebsch_map(INT ds, INT ds_row, 
	INT &line1, INT &line2, INT &transversal, 
	INT verbose_level)
{
	INT ij, i, j, k, l, m, n, size_complement;
	INT set[6];
	
	if (ds == 0) {
		if (ds_row == 0) {
			line1 = line_bi(0);
			line2 = line_bi(1);
			transversal = line_cij(0, 1);
			return;
			}
		else {
			line1 = line_ai(0);
			line2 = line_ai(1);
			transversal = line_cij(0, 1);
			return;
			}
		}
	ds--;
	if (ds < 15) {
		ij = ds;
		k2ij(ij, i, j, 6);
		
		if (ds_row == 0) {
			line1 = line_ai(j);
			line2 = line_bi(j);
			transversal = line_cij(i, j);
			return;
			}
		else {
			line1 = line_ai(i);
			line2 = line_bi(i);
			transversal = line_cij(i, j);
			return;
			}
		}
	ds -= 15;
	unrank_k_subset(ds, set, 6, 3);
	set_complement(set, 3 /* subset_size */, set + 3, 
		size_complement, 6 /* universal_set_size */);
	i = set[0];
	j = set[1];
	k = set[2];
	l = set[3];
	m = set[4];
	n = set[5];
	if (ds_row == 0) {
		line1 = line_bi(l);
		line2 = line_bi(m);
		transversal = line_ai(n);
		return;
		}
	else {
		line1 = line_ai(i);
		line2 = line_ai(j);
		transversal = line_bi(k);
		return;
		}
	
}

INT surface::clebsch_map(INT *Lines, INT *Pts, INT nb_pts, 
	INT line_idx[2], INT plane_rk, 
	INT *Image_rk, INT *Image_coeff, 
	INT verbose_level)
// assuming: 
// In:
// Lines[27]
// Pts[nb_pts]
// Out:
// Image_rk[nb_pts]  (image point in the plane in local coordinates)
//   Note Image_rk[i] is -1 if Pts[i] does not have an image.
// Image_coeff[nb_pts * 4] (image point in the plane in PG(3,q) coordinates)
{
	INT f_v = (verbose_level >= 1);
	INT Plane[4 * 4];
	INT Line_a[2 * 4];
	INT Line_b[2 * 4];
	INT Dual_planes[4 * 4]; 
		// dual coefficients of three planes:
		// the first plane is line_a together with the surface point
		// the second plane is line_b together with the surface point
		// the third plane is the plane onto which we map.
		// the fourth row is for the image point.
	INT M[4 * 4];
	INT v[4];
	INT i, h, pt, r;
	INT coefficients[3];
	INT base_cols[4];
	
	if (f_v) {
		cout << "surface::clebsch_map" << endl;
		}
	P->Grass_planes->unrank_INT_here(Plane, plane_rk, 0 /* verbose_level */);
	r = F->Gauss_simple(Plane, 3, 4, base_cols, 0 /* verbose_level */);
	if (f_v) {
		cout << "Plane rank " << plane_rk << " :" << endl;
		INT_matrix_print(Plane, 3, 4);
		}

	F->RREF_and_kernel(4, 3, Plane, 0 /* verbose_level */);

	if (f_v) {
		cout << "Plane (3 basis vectors and dual coordinates):" << endl;
		INT_matrix_print(Plane, 4, 4);
		cout << "base_cols: ";
		INT_vec_print(cout, base_cols, r);
		cout << endl;
		}

	// make sure the two lines are not contained in the plane onto which we map:

	// test line_a:
	P->Grass_lines->unrank_INT_here(Line_a, 
		Lines[line_idx[0]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line a = " << Line_label_tex[line_idx[0]] 
			<< " = " << Lines[line_idx[0]] << ":" << endl;
		INT_matrix_print(Line_a, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_a + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface::clebsch_map Line a lies inside the hyperplane" << endl;
		return FALSE;
		}

	// test line_b:
	P->Grass_lines->unrank_INT_here(Line_b, 
		Lines[line_idx[1]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line b = " << Line_label_tex[line_idx[1]] 
			<< " = " << Lines[line_idx[1]] << ":" << endl;
		INT_matrix_print(Line_b, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_b + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface::clebsch_map Line b lies inside the hyperplane" << endl;
		return FALSE;
		}

	// and now, map all surface points:
	for (h = 0; h < nb_pts; h++) {
		pt = Pts[h];

		unrank_point(v, pt);

		INT_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts << " is " << pt << " = ";
			INT_vec_print(cout, v, 4);
			cout << ":" << endl;
			}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		INT_vec_copy(Line_a, M, 2 * 4);
		INT_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_a" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		INT_vec_copy(Line_b, M, 2 * 4);
		INT_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_b" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		
		// The point is good:

		// Compute the first plane in dual coordinates:
		INT_vec_copy(Line_a, M, 2 * 4);
		INT_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		INT_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_v) {
			cout << "First plane in dual coordinates: ";
			INT_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}

		// Compute the second plane in dual coordinates:
		INT_vec_copy(Line_b, M, 2 * 4);
		INT_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		INT_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_v) {
			cout << "Second plane in dual coordinates: ";
			INT_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}


		// The third plane is the image plane, given by dual coordinates:
		INT_vec_copy(Plane + 3 * 4, Dual_planes + 8, 4);
		if (f_v) {
			cout << "Dual coordinates for all three planes: " << endl;
			INT_matrix_print(Dual_planes, 3, 4);
			cout << endl;
			}

		r = F->RREF_and_kernel(4, 3, Dual_planes, 0 /* verbose_level */);
		if (f_v) {
			cout << "Dual coordinates and perp: " << endl;
			INT_matrix_print(Dual_planes, 4, 4);
			cout << endl;
			cout << "matrix of dual coordinates has rank " << r << endl;
			}


		if (r < 3) {
			if (f_v) {
				cout << "The line is contained in the plane" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		PG_element_normalize(*F, Dual_planes + 12, 1, 4);
		if (f_v) {
			cout << "intersection point normalized: ";
			INT_vec_print(cout, Dual_planes + 12, 4);
			cout << endl;
			}
		INT_vec_copy(Dual_planes + 12, Image_coeff + h * 4, 4);
		
		// compute local coordinates of the image point:
		F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols, 
			Dual_planes + 12, coefficients, 
			0 /* verbose_level */);
		Image_rk[h] = P2->rank_point(coefficients);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts 
				<< " is " << pt << " : image = ";
			INT_vec_print(cout, Image_coeff + h * 4, 4);
			cout << " image = " << Image_rk[h] << endl;
			}
		}
	
	if (f_v) {
		cout << "surface::clebsch_map done" << endl;
		}
	return TRUE;
}

void surface::print_lines_tex(ostream &ost, INT *Lines)
{
	INT i;
	
	for (i = 0; i < 27; i++) {
		//fp << "Line " << i << " is " << v[i] << ":\\\\" << endl;
		Gr->unrank_INT(Lines[i], 0 /*verbose_level*/);
		ost << "$$" << endl;
		ost << "\\ell_{" << i << "} = " 
			<< Line_label_tex[i] << " = \\left[" << endl;
		//print_integer_matrix_width(cout, Gr->M, k, n, n, F->log10_of_q + 1);
		print_integer_matrix_tex(ost, Gr->M, 2, 4);
		ost << "\\right]_{" << Lines[i] << "}" << endl;
		ost << "$$" << endl;
		}

}

void surface::clebsch_cubics(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	


	if (f_v) {
		cout << "surface::clebsch_cubics" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::clebsch_cubics f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	INT Monomial[27];

	INT i, j, idx;

	Clebsch_Pij = NEW_INT(3 * 4 * nb_monomials2);
	Clebsch_P = NEW_PINT(3 * 4);
	Clebsch_P3 = NEW_PINT(3 * 3);

	INT_vec_zero(Clebsch_Pij, 3 * 4 * nb_monomials2);


	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			Clebsch_P[i * 4 + j] = 
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Clebsch_P3[i * 3 + j] = 
				Clebsch_Pij + (i * 4 + j) * nb_monomials2;
			}
		}
	INT coeffs[] = {
		1, 15, 2, 11,
		1, 16, 2, 12, 
		1, 17, 2, 13, 
		1, 18, 2, 14, 
		0, 3, 2, 19, 
		0, 4, 2, 20, 
		0, 5, 2, 21, 
		0, 6, 2, 22, 
		0, 23, 1, 7, 
		0, 24, 1, 8, 
		0, 25, 1, 9, 
		0, 26, 1, 10
		};
	INT c0, c1;

	if (f_v) {
		cout << "surface::clebsch_cubics Setting up the matrix P:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "i=" << i << " j=" << j << endl;
			INT_vec_zero(Monomial, 27);
			c0 = coeffs[(i * 4 + j) * 4 + 0];
			c1 = coeffs[(i * 4 + j) * 4 + 1];
			INT_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			c0 = coeffs[(i * 4 + j) * 4 + 2];
			c1 = coeffs[(i * 4 + j) * 4 + 3];
			INT_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			}
		}

	
	if (f_v) {
		cout << "surface::clebsch_cubics the matrix Clebsch_P is:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(cout, Clebsch_P[i * 4 + j]);
			cout << endl;
			}
		}

	INT *Cubics;
	INT *Adjugate;
	INT *Ad[3 * 3];
	INT *C[4];
	INT m1;


	if (f_v) {
		cout << "surface::clebsch_cubics allocating cubics" << endl;
		}

	Cubics = NEW_INT(4 * nb_monomials6);
	INT_vec_zero(Cubics, 4 * nb_monomials6);

	Adjugate = NEW_INT(3 * 3 * nb_monomials4);
	INT_vec_zero(Adjugate, 3 * 3 * nb_monomials4);

	for (i = 0; i < 4; i++) {
		C[i] = Cubics + i * nb_monomials6;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Ad[i * 3 + j] = Adjugate + (i * 3 + j) * nb_monomials4;
			}
		}

	if (f_v) {
		cout << "surface::clebsch_cubics computing C[3] = the determinant" << endl;
		}
	// compute C[3] as the negative of the determinant of the matrix of the first three columns:
	//INT_vec_zero(C[3], nb_monomials6);
	m1 = F->negate(1);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 0], Clebsch_P[1 * 4 + 1], Clebsch_P[2 * 4 + 2], m1, C[3], 0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 1], Clebsch_P[1 * 4 + 2], Clebsch_P[2 * 4 + 0], m1, C[3], 0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 2], Clebsch_P[1 * 4 + 0], Clebsch_P[2 * 4 + 1], m1, C[3], 0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 0], Clebsch_P[1 * 4 + 1], Clebsch_P[0 * 4 + 2], 1, C[3], 0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 1], Clebsch_P[1 * 4 + 2], Clebsch_P[0 * 4 + 0], 1, C[3], 0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 2], Clebsch_P[1 * 4 + 0], Clebsch_P[0 * 4 + 1], 1, C[3], 0 /* verbose_level*/);

	INT I[3];
	INT J[3];
	INT size_complement, scalar;

	if (f_v) {
		cout << "surface::clebsch_cubics computing adjugate" << endl;
		}
	// compute adjugate:
	for (i = 0; i < 3; i++) {
		I[0] = i;
		set_complement(I, 1, I + 1, size_complement, 3);
		for (j = 0; j < 3; j++) {
			J[0] = j;
			set_complement(J, 1, J + 1, size_complement, 3);
			
			if ((i + j) % 2) {
				scalar = m1;
				}
			else {
				scalar = 1;
				}
			minor22(Clebsch_P3, I[1], I[2], J[1], J[2], scalar, Ad[j * 3 + i], 0 /* verbose_level */);
			}
		}

	// multiply adjugate * last column:
	if (f_v) {
		cout << "surface::clebsch_cubics multiply adjugate times last column" << endl;
		}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			multiply42_and_add(Ad[i * 3 + j], Clebsch_P[j * 4 + 3], C[i], 0 /* verbose_level */);
			}
		}
	
	if (f_v) {
		cout << "surface::clebsch_cubics We have computed the Clebsch cubics" << endl;
		}
	

	INT Y[3];
	INT M24[24];
	INT h;
	
	Clebsch_coeffs = NEW_INT(4 * Poly3->nb_monomials * nb_monomials3);
	INT_vec_zero(Clebsch_coeffs, 4 * Poly3->nb_monomials * nb_monomials3);
	CC = NEW_PINT(4 * Poly3->nb_monomials);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < Poly3->nb_monomials; j++) {
			CC[i * Poly3->nb_monomials + j] = 
				Clebsch_coeffs + (i * Poly3->nb_monomials + j) * nb_monomials3;
			}
		}
	for (i = 0; i < Poly3->nb_monomials; i++) {
		INT_vec_copy(Poly3->Monomials + i * 3, Y, 3);
		for (j = 0; j < nb_monomials6; j++) {
			if (INT_vec_compare(Y, Poly6_27->Monomials + j * 27, 3) == 0) {
				INT_vec_copy(Poly6_27->Monomials + j * 27 + 3, M24, 24);
				idx = Poly3_24->index_of_monomial(M24);
				for (h = 0; h < 4; h++) {
					CC[h * Poly3->nb_monomials + i][idx] = 
						F->add(CC[h * Poly3->nb_monomials + i][idx], C[h][j]);
					}
				}
			}
		}

	if (f_v) {
		print_clebsch_cubics(cout);
		}

	FREE_INT(Cubics);
	FREE_INT(Adjugate);

	if (f_v) {
		cout << "surface::clebsch_cubics done" << endl;
		}
}

void surface::print_clebsch_P(ostream &ost)
{
	INT h, i, f_first;
	
	if (!f_has_large_polynomial_domains) {
		cout << "surface::print_clebsch_P f_has_large_polynomial_domains is FALSE" << endl;
		//exit(1);
		return;
		}
	ost << "\\clearpage" << endl;
	ost << "\\subsection*{The Clebsch system $P$}" << endl;

	ost << "$$" << endl;
	print_clebsch_P_matrix_only(ost);
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	ost << "\\begin{align*}" << endl;
	for (h = 0; h < 4; h++) {
		ost << "x_" << h << " &= C_" << h << "(y_0,y_1,y_2)=\\\\" << endl;
		f_first = TRUE;
		for (i = 0; i < Poly3->nb_monomials; i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->nb_monomials + i])) {
				continue;
				}
			ost << "&";

			if (f_first) {
				f_first = FALSE;
				}
			else {
				ost << "+";
				}
			ost << "\\Big(";
			Poly3_24->print_equation_with_line_breaks_tex(ost, CC[h * Poly3->nb_monomials + i], 6, "\\\\\n&");
			ost << "\\Big)" << endl;

			ost << "\\cdot" << endl;
		
			Poly3->print_monomial(ost, i);
			ost << "\\\\" << endl;
			}
		}
	ost << "\\end{align*}" << endl;
}

void surface::print_clebsch_P_matrix_only(ostream &ost)
{
	INT i, j;
	
	if (!f_has_large_polynomial_domains) {
		cout << "surface::print_clebsch_P_matrix_only f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			//cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(ost, Clebsch_P[i * 4 + j]);
			if (j < 4 - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
}

void surface::print_clebsch_cubics(ostream &ost)
{
	INT i, h;
	
	if (!f_has_large_polynomial_domains) {
		cout << "surface::print_clebsch_cubics f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	ost << "The Clebsch coefficients are:" << endl;
	for (h = 0; h < 4; h++) {
		ost << "C[" << h << "]:" << endl;
		for (i = 0; i < Poly3->nb_monomials; i++) {

			if (Poly3_24->is_zero(CC[h * Poly3->nb_monomials + i])) {
				continue;
				}
			
			Poly3->print_monomial(ost, i);
			ost << " \\cdot \\Big(";
			Poly3_24->print_equation(ost, CC[h * Poly3->nb_monomials + i]);
			ost << "\\Big)" << endl;
			}
		}
}

#if 0
INT surface::evaluate_general_cubics(INT *clebsch_coeffs_polynomial, 
	INT *variables24, INT *clebsch_coeffs_constant, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, i;

	if (f_v) {
		cout << "surface::evaluate_general_cubics" << endl;
		}

	for (i = 0; i < Poly3->nb_monomials; i++) {
		clebsch_coeffs_constant[i] = Poly3_24->evaluate_at_a_point(clebsch_coeffs + i * nb_monomials3, variables24);
		}
	if (f_v) {
		cout << "surface::evaluate_general_cubics done" << endl;
		}
}
#endif

void surface::multiply_222_27_and_add(INT *M1, INT *M2, INT *M3, 
	INT scalar, INT *MM, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k, a, b, c, d, idx;
	INT M[27];

	if (f_v) {
		cout << "surface::multiply_222_27_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::multiply_222_27_and_add f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	//INT_vec_zero(MM, nb_monomials6);
	for (i = 0; i < nb_monomials2; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
				}
			for (k = 0; k < nb_monomials2; k++) {
				c = M3[k];
				if (c == 0) {
					continue;
					}
				d = F->mult3(a, b, c);
				INT_vec_add3(Poly2_27->Monomials + i * 27, 
					Poly2_27->Monomials + j * 27, 
					Poly2_27->Monomials + k * 27, 
					M, 27);
				idx = Poly6_27->index_of_monomial(M);
				if (idx >= nb_monomials6) {
					cout << "surface::multiply_222_27_and_add idx >= nb_monomials6" << endl;
					exit(1);
					}
				d = F->mult(scalar, d);
				MM[idx] = F->add(MM[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface::multiply_222_27_and_add done" << endl;
		}
}

void surface::minor22(INT **P3, INT i1, INT i2, INT j1, INT j2, 
	INT scalar, INT *Ad, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, d, idx;
	INT M[27];

	if (f_v) {
		cout << "surface::minor22" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::minor22 f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	INT_vec_zero(Ad, nb_monomials4);
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i1 * 3 + j1][i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i2 * 3 + j2][j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			INT_vec_add(Poly2_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface::minor22 idx >= nb_monomials4" << endl;
				exit(1);
				}
			d = F->mult(scalar, d);
			Ad[idx] = F->add(Ad[idx], d);
			}
		}
	for (i = 0; i < nb_monomials2; i++) {
		a = P3[i2 * 3 + j1][i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = P3[i1 * 3 + j2][j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			INT_vec_add(Poly2_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface::minor22 idx >= nb_monomials4" << endl;
				exit(1);
				}
			d = F->mult(scalar, d);
			d = F->negate(d);
			Ad[idx] = F->add(Ad[idx], d);
			}
		}
	
	
	if (f_v) {
		cout << "surface::minor22 done" << endl;
		}
}

void surface::multiply42_and_add(INT *M1, INT *M2, INT *MM, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a, b, d, idx;
	INT M[27];

	if (f_v) {
		cout << "surface::multiply42_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::multiply42_and_add f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	for (i = 0; i < nb_monomials4; i++) {
		a = M1[i];
		if (a == 0) {
			continue;
			}
		for (j = 0; j < nb_monomials2; j++) {
			b = M2[j];
			if (b == 0) {
				continue;
				}
			d = F->mult(a, b);
			INT_vec_add(Poly4_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly6_27->index_of_monomial(M);
			if (idx >= nb_monomials6) {
				cout << "surface::multiply42_and_add idx >= nb_monomials6" << endl;
				exit(1);
				}
			MM[idx] = F->add(MM[idx], d);
			}
		}
	
	if (f_v) {
		cout << "surface::multiply42_and_add done" << endl;
		}
}

void surface::prepare_system_from_FG(INT *F_planes, INT *G_planes, 
	INT lambda, INT *&system, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;


	if (f_v) {
		cout << "surface::prepare_system_from_FG" << endl;
		}
	system = NEW_INT(3 * 4 * 3);
	INT_vec_zero(system, 3 * 4 * 3);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			INT *p = system + (i * 4 + j) * 3;
			if (i == 0) {
				p[0] = 0;
				p[1] = F->mult(lambda, G_planes[0 * 4 + j]);
				p[2] = F_planes[2 * 4 + j];
				}
			else if (i == 1) {
				p[0] = F_planes[0 * 4 + j];
				p[1] = 0;
				p[2] = G_planes[1 * 4 + j];
				}
			else if (i == 2) {
				p[0] = G_planes[2 * 4 + j];
				p[1] = F_planes[1 * 4 + j];
				p[2] = 0;
				}
			}
		}
	if (f_v) {
		cout << "surface::prepare_system_from_FG done" << endl;
		}
}

void surface::print_system(ostream &ost, INT *system)
{
	INT i, j;
	
	//ost << "The system:\\\\";
	ost << "$$" << endl;
	ost << "\\left[" << endl;
	ost << "\\begin{array}{cccc}" << endl;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			INT *p = system + (i * 4 + j) * 3;
			Poly1->print_equation(ost, p);
			if (j < 4 - 1) {
				ost << " & ";
				}
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "\\cdot \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "x_0\\\\" << endl;
	ost << "x_1\\\\" << endl;
	ost << "x_2\\\\" << endl;
	ost << "x_3\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "= \\left[" << endl;
	ost << "\\begin{array}{c}" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "0\\\\" << endl;
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	ost << "$$" << endl;
}


void surface::compute_nine_lines(INT *F_planes, INT *G_planes, 
	INT *nine_lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT Basis[16];

	if (f_v) {
		cout << "surface::compute_nine_lines" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			INT_vec_copy(F_planes + i * 4, Basis, 4);
			INT_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_INT_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		INT_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface::compute_nine_lines done" << endl;
		}
}

void surface::compute_nine_lines_by_dual_point_ranks(INT *F_planes_rank, 
	INT *G_planes_rank, INT *nine_lines, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	INT F_planes[12];
	INT G_planes[12];
	INT Basis[16];

	if (f_v) {
		cout << "surface::compute_nine_lines_by_dual_point_ranks" << endl;
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
		}
	
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			INT_vec_copy(F_planes + i * 4, Basis, 4);
			INT_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_INT_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		INT_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface::compute_nine_lines_by_dual_point_ranks done" << endl;
		}
}

void surface::print_trihedral_pair_in_dual_coordinates_in_GAP(
	INT *F_planes_rank, INT *G_planes_rank)
{
	INT i;
	INT F_planes[12];
	INT G_planes[12];

	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
		}
	cout << "[";
	for (i = 0; i < 3; i++) {
		INT_vec_print_GAP(cout, F_planes + i * 4, 4);
		cout << ", ";
		}
	for (i = 0; i < 3; i++) {
		INT_vec_print_GAP(cout, G_planes + i * 4, 4);
		if (i < 3 - 1) {
			cout << ", ";
			}
		}
	cout << "];";
}

void surface::split_nice_equation(INT *nice_equation, 
	INT *&f1, INT *&f2, INT *&f3, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface::split_nice_equation" << endl;
		}
	INT M[4];
	INT i, a, idx;

	f1 = NEW_INT(Poly1->nb_monomials);
	f2 = NEW_INT(Poly2->nb_monomials);
	f3 = NEW_INT(Poly3->nb_monomials);
	INT_vec_zero(f1, Poly1->nb_monomials);
	INT_vec_zero(f2, Poly2->nb_monomials);
	INT_vec_zero(f3, Poly3->nb_monomials);
	
	for (i = 0; i < 20; i++) {
		a = nice_equation[i];
		if (a == 0) {
			continue;
			}
		INT_vec_copy(Poly3_4->Monomials + i * 4, M, 4);
		if (M[0] == 3) {
			cout << "surface::split_nice_equation the x_0^3 term is supposed to be zero" << endl;
			exit(1);
			}
		else if (M[0] == 2) {
			idx = Poly1->index_of_monomial(M + 1);
			f1[idx] = a;
			}
		else if (M[0] == 1) {
			idx = Poly2->index_of_monomial(M + 1);
			f2[idx] = a;
			}
		else if (M[0] == 0) {
			idx = Poly3->index_of_monomial(M + 1);
			f3[idx] = a;
			}
		}
	if (f_v) {
		cout << "surface::split_nice_equation done" << endl;
		}
}

void surface::assemble_tangent_quadric(INT *f1, INT *f2, INT *f3, 
	INT *&tangent_quadric, INT verbose_level)
// 2*x_0*f_1 + f_2
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface::assemble_tangent_quadric" << endl;
		}
	INT M[4];
	INT i, a, idx, two;


	two = F->add(1, 1);
	tangent_quadric = NEW_INT(Poly2_4->nb_monomials);
	INT_vec_zero(tangent_quadric, Poly2_4->nb_monomials);
	
	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = f1[i];
		if (a == 0) {
			continue;
			}
		INT_vec_copy(Poly1->Monomials + i * 3, M + 1, 3);
		M[0] = 1;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = F->mult(two, a);
		}

	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = f2[i];
		if (a == 0) {
			continue;
			}
		INT_vec_copy(Poly2->Monomials + i * 3, M + 1, 3);
		M[0] = 0;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = a;
		}
	if (f_v) {
		cout << "surface::assemble_tangent_quadric done" << endl;
		}
}

void surface::print_polynomial_domains(ostream &ost)
{
	ost << "The polynomial domain Poly3\\_4 is:" << endl;
	Poly3_4->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly1\\_x123 is:" << endl;
	Poly1_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly2\\_x123 is:" << endl;
	Poly2_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly3\\_x123 is:" << endl;
	Poly3_x123->print_monomial_ordering(ost);

	ost << "The polynomial domain Poly4\\_x123 is:" << endl;
	Poly4_x123->print_monomial_ordering(ost);

}

void surface::print_line_labelling(ostream &ost)
{
	INT j, h;
	
	//ost << "The ordering of monomials is:\\\\" << endl;
	ost << "$$" << endl;
	for (j = 0; j < 3; j++) {
		ost << "\\begin{array}{|r|r|}" << endl;
		ost << "\\hline" << endl;
		ost << "h &  \\mbox{line} \\\\" << endl;
		ost << "\\hline" << endl;
		ost << "\\hline" << endl;
		for (h = 0; h < 9; h++) {
			ost << j * 9 + h << " & " 
				<< Line_label_tex[j * 9 + h] << "\\\\" << endl; 
			}
		ost << "\\hline" << endl;
		ost << "\\end{array}" << endl;
		if (j < 3 - 1) {
			ost << "\\qquad" << endl;
			}
		}
	ost << "$$" << endl;
}

void surface::print_set_of_lines_tex(ostream &ost, INT *v, INT len)
{
	INT i;
	
	ost << "\\{";
	for (i = 0; i < len; i++) {
		ost << Line_label_tex[v[i]];
		if (i < len - 1) {
			ost << ", ";
			}
		}
	ost << "\\}";
}

void surface::tritangent_plane_to_trihedral_pair_and_position(
	INT tritangent_plane_idx, 
	INT &trihedral_pair_idx, INT &position, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	static INT Table[] = {
		0, 2, // 0
		0, 5, // 1
		22, 0, // 2
		0, 1, // 3
		20, 0, // 4
		1, 1, // 5
		26, 0, //6
		5, 1, // 7
		32, 0, //8
		6, 1, //9
		0, 0, //10
		25, 0, // 11
		1, 0, // 12
		43, 0, //13
		2, 0, //14
		55, 0, // 15
		3, 0, // 16
		3, 3, //17
		4, 0, //18
		67, 0, // 19
		5, 0, // 20
		73, 0, // 21
		6, 0, // 22
		6, 3, // 23
		7, 0, // 24
		79, 0, // 25
		8, 0, // 26
		8, 3, // 27
		9, 0, // 28
		9, 3, // 29
		115, 0, // 30
		114, 0, // 31
		34, 2, // 32
		113, 0, // 33
		111, 0, // 34
		34, 5, // 35
		74, 2, // 36
		110, 0, // 37
		49, 2, // 38
		26, 5, // 39
		38, 5, // 40
		53, 5, // 41
		36, 5, // 42
		45, 5, // 43
		51, 5, // 44
		};

	if (f_v) {
		cout << "surface::tritangent_plane_to_trihedral_pair_and_position" << endl;
		}
	trihedral_pair_idx = Table[2 * tritangent_plane_idx + 0];
	position = Table[2 * tritangent_plane_idx + 1];
	if (f_v) {
		cout << "surface::tritangent_plane_to_trihedral_pair_and_position done" << endl;
		}
}

