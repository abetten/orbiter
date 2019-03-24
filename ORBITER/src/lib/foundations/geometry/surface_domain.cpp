// surface.cpp
// 
// Anton Betten
// Jul 25, 2016
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


surface_domain::surface_domain()
{
	null();
}

surface_domain::~surface_domain()
{
	freeself();
}

void surface_domain::freeself()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "surface_domain::freeself" << endl;
		}
	if (v) {
		FREE_int(v);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (w2) {
		FREE_int(w2);
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
		FREE_int(Sets);
		}
	if (M) {
		FREE_int(M);
		}
	if (Sets2) {
		FREE_int(Sets2);
		}
	if (Pts) {
		FREE_int(Pts);
		}
	if (pt_list) {
		FREE_int(pt_list);
		}
	if (System) {
		FREE_int(System);
		}
	if (base_cols) {
		FREE_int(base_cols);
		}
	if (f_v) {
		cout << "before FREE_pchar(Line_label);" << endl;
		}
	if (Line_label) {
		int i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label[i]);
			}
		FREE_pchar(Line_label);
		}
	if (Line_label_tex) {
		int i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label_tex[i]);
			}
		FREE_pchar(Line_label_tex);
		}
	if (Eckard_point_label) {
		int i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label[i]);
			}
		FREE_pchar(Eckard_point_label);
		}
	if (Eckard_point_label_tex) {
		int i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label_tex[i]);
			}
		FREE_pchar(Eckard_point_label_tex);
		}
	if (f_v) {
		cout << "before FREE_int(Trihedral_pairs);" << endl;
		}
	if (Trihedral_pairs) {
		FREE_int(Trihedral_pairs);
		}
	if (Trihedral_pair_labels) {
		int i;
		
		for (i = 0; i < nb_trihedral_pairs; i++) {
			FREE_char(Trihedral_pair_labels[i]);
			}
		FREE_pchar(Trihedral_pair_labels);
		}
	if (Trihedral_pairs_row_sets) {
		FREE_int(Trihedral_pairs_row_sets);
		}
	if (Trihedral_pairs_col_sets) {
		FREE_int(Trihedral_pairs_col_sets);
		}
	if (f_v) {
		cout << "before FREE_OBJECT Classify_trihedral_pairs_"
				"row_values;" << endl;
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
		FREE_int(Trihedral_to_Eckardt);
		}
	if (collinear_Eckardt_triples_rank) {
		FREE_int(collinear_Eckardt_triples_rank);
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
		FREE_int(Double_six);
		}
	if (Double_six_label_tex) {
		int i;
		
		for (i = 0; i < 36; i++) {
			FREE_char(Double_six_label_tex[i]);
			}
		FREE_pchar(Double_six_label_tex);
		}
	if (Half_double_sixes) {
		FREE_int(Half_double_sixes);
		}

	if (Half_double_six_label_tex) {
		int i;
		
		for (i = 0; i < 72; i++) {
			FREE_char(Half_double_six_label_tex[i]);
			}
		FREE_pchar(Half_double_six_label_tex);
		}

	if (Half_double_six_to_double_six) {
		FREE_int(Half_double_six_to_double_six);
		}
	if (Half_double_six_to_double_six_row) {
		FREE_int(Half_double_six_to_double_six_row);
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
		FREE_int(Clebsch_Pij);
		}
	if (Clebsch_P) {
		FREE_pint(Clebsch_P);
		}
	if (Clebsch_P3) {
		FREE_pint(Clebsch_P3);
		}
	if (Clebsch_coeffs) {
		FREE_int(Clebsch_coeffs);
		}
	if (CC) {
		FREE_pint(CC);
		}
	if (adjacency_matrix_of_lines) {
		FREE_int(adjacency_matrix_of_lines);
	}
	null();
	if (f_v) {
		cout << "surface_domain::freeself done" << endl;
		}
}

void surface_domain::null()
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

	adjacency_matrix_of_lines = NULL;
}

void surface_domain::init(finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::init" << endl;
		}
	
	n = 4;
	n2 = 2 * n;
	surface_domain::F = F;
	q = F->q;
	nb_pts_on_surface = q * q + 7 * q + 1;
	if (f_v) {
		cout << "surface::init nb_pts_on_surface = "
				<< nb_pts_on_surface << endl;
		}

	v = NEW_int(n);
	v2 = NEW_int(6);
	w2 = NEW_int(6);
	
	P = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "surface::init before P->init" << endl;
		}
	P->init(3, F, 
		TRUE /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init after P->init" << endl;
		}

	P2 = NEW_OBJECT(projective_space);
	if (f_v) {
		cout << "surface::init before P2->init" << endl;
		}
	P2->init(2, F, 
		TRUE /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init after P2->init" << endl;
		}

	Gr = NEW_OBJECT(grassmann);
	Gr->init(n, 2, F, 0 /* verbose_level */);
	nb_lines_PG_3 = Gr->nCkq.as_int();
	if (f_v) {
		cout << "surface::init nb_lines_PG_3 = "
				<< nb_lines_PG_3 << endl;
		}

	Gr3 = NEW_OBJECT(grassmann);
	Gr3->init(4, 3, F, 0 /* verbose_level*/);


	if (f_v) {
		cout << "surface::init "
				"initializing orthogonal" << endl;
		}
	O = NEW_OBJECT(orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, verbose_level - 2);
	if (f_v) {
		cout << "surface::init "
				"initializing orthogonal done" << endl;
		}

	Klein = NEW_OBJECT(klein_correspondence);

	if (f_v) {
		cout << "surface::init initializing "
				"Klein correspondence" << endl;
		}
	Klein->init(F, O, verbose_level - 2);
	if (f_v) {
		cout << "surface::init initializing "
				"Klein correspondence done" << endl;
		}



	if (f_v) {
		cout << "surface::init before "
				"init_polynomial_domains" << endl;
		}
	init_polynomial_domains(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_polynomial_domains" << endl;
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
		cout << "surface::init before "
				"init_line_data" << endl;
		}
	init_line_data(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_line_data" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"make_trihedral_pairs" << endl;
		}
	make_trihedral_pairs(Trihedral_pairs, 
		Trihedral_pair_labels, nb_trihedral_pairs, 
		verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"make_trihedral_pairs" << endl;
		}
	
	if (f_v) {
		cout << "surface::init before "
				"process_trihedral_pairs" << endl;
		}
	process_trihedral_pairs(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"process_trihedral_pairs" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"make_Eckardt_points" << endl;
		}
	make_Eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"make_Eckardt_points" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"init_Trihedral_to_Eckardt" << endl;
		}
	init_Trihedral_to_Eckardt(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_Trihedral_to_Eckardt" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"init_collinear_Eckardt_triples" << endl;
		}
	init_collinear_Eckardt_triples(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_collinear_Eckardt_triples" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"init_double_sixes" << endl;
		}
	init_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_double_sixes" << endl;
		}

	if (f_v) {
		cout << "surface::init before "
				"create_half_double_sixes" << endl;
		}
	create_half_double_sixes(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"create_half_double_sixes" << endl;
		}
	print_half_double_sixes_in_GAP();

	if (f_v) {
		cout << "surface::init before "
				"init_adjacency_matrix_of_lines" << endl;
		}
	init_adjacency_matrix_of_lines(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_adjacency_matrix_of_lines" << endl;
		}

	//clebsch_cubics(verbose_level);

	if (f_v) {
		cout << "surface::init done" << endl;
		}
}


void surface_domain::init_polynomial_domains(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains" << endl;
		}
	Poly1 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3 = NEW_OBJECT(homogeneous_polynomial_domain);

	Poly1->init(F,
			3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly2->init(F,
			3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

	Poly1_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly1_x123->init(F,
			3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly2_x123->init(F,
			3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly3_x123->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly4_x123->init(F,
			3 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);


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
	Poly1_4->init(F,
			4 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly2_4->init(F,
			4 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly3_4->init(F,
			4 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

	label_variables_4(Poly1_4, 0 /* verbose_level */);
	label_variables_4(Poly2_4, 0 /* verbose_level */);
	label_variables_4(Poly3_4, 0 /* verbose_level */);

	nb_monomials = Poly3_4->nb_monomials;

	if (f_v) {
		cout << "nb_monomials = " << nb_monomials << endl;
		}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains done" << endl;
		}
}

void surface_domain::init_large_polynomial_domains(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::init_large_polynomial_domains" << endl;
		}
	f_has_large_polynomial_domains = TRUE;
	Poly2_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly6_27 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_24 = NEW_OBJECT(homogeneous_polynomial_domain);

	Poly2_27->init(F,
			27 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly4_27->init(F,
			27 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly6_27->init(F,
			27 /* nb_vars */, 6 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	Poly3_24->init(F,
			24 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);

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
		cout << "surface_domain::init_large_polynomial_domains "
				"before clebsch_cubics" << endl;
		}
	clebsch_cubics(verbose_level - 1);
	if (f_v) {
		cout << "surface_domain::init_large_polynomial_domains "
				"after clebsch_cubics" << endl;
		}

	if (f_v) {
		cout << "surface::init_large_polynomial_domains done" << endl;
		}
}

void surface_domain::label_variables_3(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_3" << endl;
		}
	if (HPD->n != 3) {
		cout << "surface_domain::label_variables_3 HPD->n != 3" << endl;
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
		sprintf(label, "y_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_{%d}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface_domain::label_variables_3 done" << endl;
		}
	
}

void surface_domain::label_variables_x123(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_x123" << endl;
		}
	if (HPD->n != 3) {
		cout << "surface_domain::label_variables_x123 "
				"HPD->n != 3" << endl;
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
		sprintf(label, "x_%d", i + 1);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "x_{%d}", i + 1);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface_domain::label_variables_x123 done" << endl;
		}
	
}

void surface_domain::label_variables_4(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_4" << endl;
		}
	if (HPD->n != 4) {
		cout << "surface_domain::label_variables_4 HPD->n != 4" << endl;
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
		sprintf(label, "X_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 4; i++) {
		sprintf(label, "X_{%d}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	if (f_v) {
		cout << "surface::label_variables_4 done" << endl;
		}
	
}

void surface_domain::label_variables_27(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_27" << endl;
		}
	if (HPD->n != 27) {
		cout << "surface_domain::label_variables_27 HPD->n != 27" << endl;
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
		sprintf(label, "y_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[3 + 12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + 12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		sprintf(label, "y_{%d}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[3 + 12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[3 + 12 + i * 4 + j], label);
			}
		}
	if (f_v) {
		cout << "surface_domain::label_variables_27 done" << endl;
		}
	
}

void surface_domain::label_variables_24(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l;
	char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_24" << endl;
		}
	if (HPD->n != 24) {
		cout << "surface_domain::label_variables_24 HPD->n != 24" << endl;
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
			sprintf(label, "f_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "f_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			sprintf(label, "g_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[12 + i * 4 + j], label);
			}
		}
	if (f_v) {
		cout << "surface_domain::label_variables_24 done" << endl;
		}
	
}

void surface_domain::init_system(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::init_system" << endl;
		}

	max_pts = 27 * (q + 1);
	Pts = NEW_int(max_pts * n);
	pt_list = NEW_int(max_pts);
	System = NEW_int(max_pts * nb_monomials);
	base_cols = NEW_int(nb_monomials);
	
	if (f_v) {
		cout << "surface_domain::init_system done" << endl;
		}

}


int surface_domain::index_of_monomial(int *v)
{
	return Poly3_4->index_of_monomial(v);
}

void surface_domain::unrank_point(int *v, int rk)
{
	P->unrank_point(v, rk);
}

int surface_domain::rank_point(int *v)
{
	int rk;

	rk = P->rank_point(v);
	return rk;
}

void surface_domain::unrank_plane(int *v, int rk)
{
	Gr3->unrank_int_here(v, rk, 0 /* verbose_level */);
}

int surface_domain::rank_plane(int *v)
{
	int rk;

	rk = Gr3->rank_int_here(v, 0 /* verbose_level */);
	return rk;
}

int surface_domain::test(int len, int *S, int verbose_level)
{
	//verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int r, ret;

	if (f_v) {
		cout << "surface_domain::test" << endl;
		}

	r = compute_system_in_RREF(len, S, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_domain::test The system has rank " << r << endl;
		}
	if (r < nb_monomials) {
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	if (f_v) {
		cout << "surface_domain::test done ret = " << ret << endl;
		}
	return ret;
}

void surface_domain::enumerate_points(int *coeff,
	int *Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::enumerate_points" << endl;
		}

	Poly3_4->enumerate_points(coeff, Pts, nb_pts, verbose_level);
	if (f_v) {
		cout << "surface_domain::enumerate_points done" << endl;
		}
}

void surface_domain::substitute_semilinear(
	int *coeff_in, int *coeff_out,
	int f_semilinear, int frob, int *Mtx_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::substitute_semilinear" << endl;
		}
	Poly3_4->substitute_semilinear(coeff_in, coeff_out, 
		f_semilinear, frob, Mtx_inv, verbose_level);
	if (f_v) {
		cout << "surface_domain::substitute_semilinear done" << endl;
		}
}

int surface_domain::test_special_form_alpha_beta(int *coeff,
	int &alpha, int &beta, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret = TRUE;
	int zeroes[] = {0,1,2,4,5,7,8,10,11,13,14,15,17,18,19};
	int alphas[] = {6,9,12};
	int betas[] = {16};
	int a;
	
	if (f_v) {
		cout << "surface_domain::test_special_form_alpha_beta" << endl;
		}
	if (!int_vec_is_constant_on_subset(coeff, 
		zeroes, sizeof(zeroes) / sizeof(int), a)) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not constant on zero set" << endl;
		return FALSE;
		}
	if (a != 0) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not zero on zero set" << endl;
		return FALSE;
		}
	if (coeff[3] != 1) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not normalized" << endl;
		exit(1);
		}
	if (!int_vec_is_constant_on_subset(coeff, 
		alphas, sizeof(alphas) / sizeof(int), a)) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not constant on alpha set" << endl;
		return FALSE;
		}
	alpha = a;
	if (!int_vec_is_constant_on_subset(coeff, 
		betas, sizeof(betas) / sizeof(int), a)) {
		cout << "surface_domain::test_special_form_alpha_beta "
				"not constant on beta set" << endl;
		return FALSE;
		}
	beta = a;

	if (f_v) {
		cout << "surface_domain::test_special_form_alpha_beta done" << endl;
		}
	return ret;
}

void surface_domain::create_special_double_six(int *double_six,
	int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[12 * 8] = {
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
	int i, c, ma, mb, av, mav;

	if (f_v) {
		cout << "surface_domain::create_special_double_six "
				"a=" << a << " b=" << b << endl;
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
		double_six[i] = Gr->rank_int_here(Basis + i * 8,
				0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface_domain::create_special_double_six done" << endl;
		}
}

void surface_domain::create_special_fifteen_lines(int *fifteen_lines,
	int a, int b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[15 * 8] = {
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
	int i, m1, a2, a2p1, a2m1, ba2p1, /*ba2m1,*/ twoa;
	int c, c2, cm2, c3, cm3, c4, cm4, c5, cm5;

	// 2 stands for (2a)/(b(a^2+1))
	// -2 stands for -(2a)/(b(b^2+1))
	// 3 stands for (a^2-1)/(b(a^2+1))
	// -3 stands for -(a^2-1)/(b(a^2+1))
	// 4 stands for (2a)/(a^2-1)
	// -4 stands for -(2a)/(a^2-1)
	// 5 stands for 3 inverse
	// -5 stands for -3 inverse

	if (f_v) {
		cout << "surface_domain::create_special_fifteen_lines "
				"a=" << a << " b=" << b << endl;
		}
	m1 = F->negate(1);
	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, m1);
	twoa = F->add(a, a);
	ba2p1 = F->mult(b, a2p1);
	//ba2m1 = F->mult(b, a2m1);

	if (ba2p1 == 0) {
		cout << "surface_domain::create_special_fifteen_lines "
				"ba2p1 = 0, cannot invert" << endl;
		exit(1);
		}
	c2 = F->mult(twoa, F->inverse(ba2p1));
	cm2 = F->negate(c2);
	c3 = F->mult(a2m1, F->inverse(ba2p1));
	cm3 = F->negate(c3);
	if (a2m1 == 0) {
		cout << "surface_domain::create_special_fifteen_lines "
				"a2m1 = 0, cannot invert" << endl;
		exit(1);
		}
	c4 = F->mult(twoa, F->inverse(a2m1));
	cm4 = F->negate(c4);

	if (c3 == 0) {
		cout << "surface_domain::create_special_fifteen_lines "
				"c3 = 0, cannot invert" << endl;
		exit(1);
		}
	c5 = F->inverse(c3);
	if (cm3 == 0) {
		cout << "surface_domain::create_special_fifteen_lines "
				"cm3 = 0, cannot invert" << endl;
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
			cout << "surface_domain::create_special_fifteen_lines "
					"unknown value" << c << endl;
			exit(1);
			}
		Basis[i] = c;
		}
	for (i = 0; i < 15; i++) {
		fifteen_lines[i] = Gr->rank_int_here(
			Basis + i * 8, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface_domain::create_special_fifteen_lines done" << endl;
		}
}

void surface_domain::create_equation_Sab(int a, int b,
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_equation_Sab" << endl;
		}
	alpha = F->negate(F->mult(b, b));
	beta = F->mult(F->mult(F->power(b, 3), 
		F->add(1, F->mult(a, a))), F->inverse(a));
	int_vec_zero(coeff, nb_monomials);
	
	coeff[3] = 1;
	coeff[6] = alpha;
	coeff[9] = alpha;
	coeff[12] = alpha;
	coeff[16] = beta;
	//coeff[19] = beta;
	if (f_v) {
		cout << "surface_domain::create_equation_Sab done" << endl;
		}
}

int surface_domain::create_surface_ab(int a, int b,
	int *coeff20,
	int *Lines27,
	int &alpha, int &beta, int &nb_E,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int alpha0, beta0;
	//int line_rk;
	//int Basis[8];
	//int Lines[27];
	int nb, i, e, ee, nb_lines, rk, nb_pts;
	//int *coeff;
	int *Pts;
	int v[4];

	if (f_v) {
		cout << "surface_domain::create_surface_ab" << endl;
		}
	alpha = -1;
	beta = -1;
	nb_E = -1;

	int a2, a2p1, a2m1;

	a2 = F->mult(a, a);
	a2p1 = F->add(a2, 1);
	a2m1 = F->add(a2, F->negate(1));
	if (a2p1 == 0 || a2m1 == 0) {
		cout << "surface_domain::create_surface_ab "
				"a2p1 == 0 || a2m1 == 0" << endl;
		return FALSE;
		}


	Pts = NEW_int(nb_PG_elements(3, F->q));
	
	//coeff = NEW_int(20);
	alpha0 = F->negate(F->mult(b, b));
	beta0 = F->mult(F->mult(F->power(b, 3), 
		F->add(1, F->mult(a, a))), F->inverse(a));
	if (f_v) {
		cout << "surface_domain::create_surface_ab a="
			<< a << " b=" << b << " alpha0=" << alpha0 
			<< " beta0=" << beta0 << endl;
		}
	
#if 0
	int_vec_zero(Basis, 8);
	Basis[0 * 4 + 0] = 1;
	Basis[0 * 4 + 1] = a;
	Basis[1 * 4 + 2] = 1;
	Basis[1 * 4 + 3] = b;
	line_rk = Gr->rank_int_here(Basis, 0);
#endif


#if 0
	//int_vec_copy(desired_lines, Lines, 3);
	//nb = 3;

	cout << "The triangle lines are:" << endl;
	Gr->print_set(desired_lines, 3);
#endif


	int *Oab;

	Oab = NEW_int(12);
	create_special_double_six(Oab, a, b, 0 /* verbose_level */);

#if 0
	if (!test_if_sets_are_equal(Oab, Lines, 12)) {
		cout << "the sets are not equal" << endl;
		exit(1);
		}
#endif

	if (f_v) {
		cout << "surface_domain::create_surface_ab The double six is:" << endl;
		Gr->print_set(Oab, 12);
		}


	int_vec_copy(Oab, Lines27, 12);
	FREE_int(Oab);


	nb = 12;

	if (f_v) {
		cout << "surface_domain::create_surface_ab We have a set of "
				"lines of size " << nb << ":";
		int_vec_print(cout, Lines27, nb);
		cout << endl;
		}

	create_remaining_fifteen_lines(Lines27,
		Lines27 + 12, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_ab The remaining 15 lines are:";
		int_vec_print(cout, Lines27 + 12, 15);
		cout << endl;
		Gr->print_set(Lines27 + 12, 15);
		}


	if (f_v) {
		cout << "surface_domain::create_surface_ab before create_special_"
				"fifteen_lines" << endl;
		}

	int special_lines[15];

	create_special_fifteen_lines(special_lines, a, b, verbose_level);
	for (i = 0; i < 15; i++) {
		if (special_lines[i] != Lines27[12 + i]) {
			cout << "surface_domain::create_surface_ab something is wrong "
					"with the special line " << i << " / 15 " << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "surface_domain::create_surface_ab after create_special_"
				"fifteen_lines" << endl;
		}

	rk = compute_system_in_RREF(27, Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_surface_ab a=" << a
			<< " b=" << b << " rk=" << rk << endl;
		}

	if (rk != 19) {
		cout << "surface_domain::create_surface_ab rk != 19" << endl;
		FREE_int(Pts);
		//FREE_int(coeff);
		exit(1);
		}
	build_cubic_surface_from_lines(27, Lines27, coeff20,
			0 /* verbose_level */);
	F->PG_element_normalize_from_front(coeff20, 1, 20);



	enumerate_points(coeff20, Pts, nb_pts, 0 /* verbose_level */);
	int_vec_heapsort(Pts, nb_pts);


	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"a=" << a << " b=" << b << " equation: ";
		print_equation(cout, coeff20);
		cout << endl;
		}

	if (nb_pts != nb_pts_on_surface) {
		cout << "surface_domain::create_surface_ab degenerate surface" << endl;
		cout << "nb_pts=" << nb_pts << endl;
		cout << "should be =" << nb_pts_on_surface << endl;
		alpha = -1;
		beta = -1;
		nb_E = -1;
		return FALSE;
		}

	if (f_v) {
		cout << "surface_domain::create_surface_ab Pts: " << endl;
		int_vec_print_as_table(cout, Pts, nb_pts, 10);
		}


	int *Adj;
	int *Intersection_pt;
	int *Intersection_pt_idx;

	compute_adjacency_matrix_of_line_intersection_graph(
		Adj, Lines27, 27, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"The adjacency matrix is:" << endl;
		int_matrix_print(Adj, 27, 27);
		}



	compute_intersection_points_and_indices(
		Adj, Pts, nb_pts, Lines27, 27,
		Intersection_pt, Intersection_pt_idx, 
		verbose_level);

	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"The intersection points are:" << endl;
		int_matrix_print(Intersection_pt_idx, 27, 27);
		}


	classify C;

	C.init(Intersection_pt_idx, 27 * 27, FALSE, 0);
	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"classification of points by multiplicity:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		}




	if (!test_special_form_alpha_beta(coeff20, alpha, beta,
		0 /* verbose_level */)) {
		cout << "surface_domain::create_surface_ab not of special form" << endl;
		exit(1);
		}


	if (alpha != alpha0) {
		cout << "surface_domain::create_surface_ab alpha != alpha0" << endl;
		exit(1);
		}
	if (beta != beta0) {
		cout << "surface_domain::create_surface_ab beta != beta0" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"determining all lines on the surface:" << endl;
		}
	{
	int Lines2[27];
	P->find_lines_which_are_contained(Pts, nb_pts, 
		Lines2, nb_lines, 27 /* max_lines */, 
		0 /* verbose_level */);
	}
	
	if (f_v) {
		cout << "surface_domain::create_surface_ab "
				"nb_lines = " << nb_lines << endl;
		}
	if (nb_lines != 27) {
		cout << "surface_domain::create_surface_ab "
				"nb_lines != 27, something is wrong "
				"with the surface" << endl;
		exit(1);
		}
	set_of_sets *pts_on_lines;
	set_of_sets *lines_on_pt;
	
	compute_points_on_lines(Pts, nb_pts, 
		Lines27, nb_lines,
		pts_on_lines, 
		verbose_level);


	if (f_v) {
		cout << "surface_domain::create_surface_ab pts_on_lines: " << endl;
		pts_on_lines->print_table();
		}

	int *E;
	
	pts_on_lines->get_eckardt_points(E, nb_E, 0 /* verbose_level */);
	//nb_E = pts_on_lines->number_of_eckardt_points(verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_ab The surface contains "
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
		cout << "surface_domain::create_surface_ab "
				"The Eckardt points are:" << endl;
		for (i = 0; i < nb_E; i++) {
			e = E[i];
			ee = Pts[e];
			unrank_point(v, ee);
			cout << i << " : " << ee << " : ";
			int_vec_print(cout, v, 4);
			cout << " on lines: ";
			int_vec_print(cout, lines_on_pt->Sets[e], 
				lines_on_pt->Set_size[e]);
			cout << endl;
			}
		}

	
	FREE_int(E);
	//FREE_int(coeff);
	FREE_int(Pts);
	FREE_int(Intersection_pt);
	FREE_int(Intersection_pt_idx);
	FREE_OBJECT(pts_on_lines);
	FREE_OBJECT(lines_on_pt);
	if (f_v) {
		cout << "surface_domain::create_surface_ab done" << endl;
		}
	return TRUE;
}

void surface_domain::list_starter_configurations(
	int *Lines, int nb_lines,
	set_of_sets *line_intersections, int *&Table, int &N, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	int subset2[5];
	int S3[6];
	int N1, nCk, h;
	int i, j, r;
	
	if (f_v) {
		cout << "surface_domain::list_starter_configurations" << endl;
		}

	N = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
			}
		nCk = int_n_choose_k(line_intersections->Set_size[i], 5);
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
		cout << "surface_domain::list_starter_configurations We found "
			<< N << " starter configurations on this surface" 
			<< endl;
		}
	Table = NEW_int(N * 2);
	N1 = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
			}
		nCk = int_n_choose_k(line_intersections->Set_size[i], 5);
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
		cout << "surface_domain::list_starter_configurations done" << endl;
		}
}

void surface_domain::create_starter_configuration(
	int line_idx, int subset_idx, 
	set_of_sets *line_neighbors, int *Lines, int *S, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	int subset2[5];
	int h; //, nCk;
	
	if (f_v) {
		cout << "surface_domain::create_starter_configuration" << endl;
		}
	//nCk = int_n_choose_k(line_neighbors->Set_size[line_idx], 5);
	unrank_k_subset(subset_idx, subset, 
		line_neighbors->Set_size[line_idx], 5);
	for (h = 0; h < 5; h++) {
		subset2[h] = line_neighbors->Sets[line_idx][subset[h]];
		S[h] = Lines[subset2[h]];
		}
	S[5] = Lines[line_idx];
	if (f_v) {
		cout << "surface_domain::create_starter_configuration done" << endl;
		}
}

void surface_domain::wedge_to_klein(int *W, int *K)
{
	K[0] = W[0];
	K[1] = W[5];
	K[2] = W[1];
	K[3] = F->negate(W[4]);
	K[4] = W[2];
	K[5] = W[3];
}

void surface_domain::klein_to_wedge(int *K, int *W)
{
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
}

int surface_domain::line_to_wedge(int line_rk)
{
	int a, b;
	
	a = Klein->Line_to_point_on_quadric[line_rk];
	O->unrank_point(w2, 1, a, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::line_to_wedge_vec(
		int *Line_rk, int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = line_to_wedge(Line_rk[i]);
		}
}

void surface_domain::line_to_klein_vec(
		int *Line_rk, int *Klein_rk, int len)
{
	int_vec_apply(Line_rk, Klein->Line_to_point_on_quadric,
			Klein_rk, len);
}

int surface_domain::klein_to_wedge(int klein_rk)
{
	int b;
	
	O->unrank_point(w2, 1, klein_rk, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::klein_to_wedge_vec(
		int *Klein_rk, int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = klein_to_wedge(Klein_rk[i]);
		}
}

void surface_domain::save_lines_in_three_kinds(const char *fname_csv,
	int *Lines_wedge, int *Lines, int *Lines_klein, int nb_lines)
{
	spreadsheet *Sp;
	
	make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Lines_wedge, Lines, Lines_klein, nb_lines,
		0 /* verbose_level */);

	Sp->save(fname_csv, 0 /*verbose_level*/);
	FREE_OBJECT(Sp);
}

void surface_domain::find_tritangent_planes_intersecting_in_a_line(
	int line_idx,
	int &plane1, int &plane2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;
	int three_lines[3];

	if (f_v) {
		cout << "surface_domain::find_tritangent_planes_"
				"intersecting_in_a_line" << endl;
		}
	for (plane1 = 0; plane1 < nb_Eckardt_points; plane1++) {

		Eckardt_points[plane1].three_lines(this, three_lines);
		if (int_vec_search_linear(three_lines, 3, line_idx, idx)) {
			for (plane2 = plane1 + 1;
					plane2 < nb_Eckardt_points;
					plane2++) {

				Eckardt_points[plane2].three_lines(this, three_lines);
				if (int_vec_search_linear(three_lines, 3, line_idx, idx)) {
					if (f_v) {
						cout << "surface_domain::find_tritangent_planes_"
								"intersecting_in_a_line done" << endl;
						}
					return;
					}
				}
			}
		}
	cout << "surface_domain::find_tritangent_planes_"
			"intersecting_in_a_line could not find "
			"two planes" << endl;
	exit(1);
}


void surface_domain::make_trihedral_pairs(int *&T,
	char **&T_label, int &nb_trihedral_pairs, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int h, s, idx;
	int subset[6];
	int second_subset[6];
	int complement[6];
	int size_complement;
	char label[1000];

	if (f_v) {
		cout << "surface_domain::make_trihedral_pairs" << endl;
		}
	nb_trihedral_pairs = 120;
	T = NEW_int(nb_trihedral_pairs * 9);
	T_label = NEW_pchar(nb_trihedral_pairs);

	idx = 0;

	// the first type (20 of them):
	for (h = 0; h < 20; h++, idx++) {
		unrank_k_subset(h, subset, 6, 3);
#if 0
		if (h == 16) {
			cout << "h=16: subset=";
			int_vec_print(cout, subset, 3);
			cout << endl;
			}
#endif
		sprintf(label, "%d%d%d", subset[0] + 1, subset[1] + 1, subset[2] + 1);

		make_Tijk(T + idx * 9, subset[0], subset[1], subset[2]);
		T_label[idx] = NEW_char(strlen(label) + 1);
		strcpy(T_label[idx], label);
#if 0
		if (h == 16) {
			cout << "h=16:T=";
			int_vec_print(cout, T + idx * 9, 9);
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
			sprintf(label, "%d%d,%d%d",
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
		sprintf(label, "%d%d%d,%d%d%d",
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
		cout << "surface_domain::make_trihedral_pairs idx != 120" << endl;
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
		cout << "surface_domain::make_trihedral_pairs done" << endl;
		}
}

void surface_domain::process_trihedral_pairs(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[3];
	int i, j, h, rk, a;

	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs" << endl;
		}
	Trihedral_pairs_row_sets = NEW_int(nb_trihedral_pairs * 3);
	Trihedral_pairs_col_sets = NEW_int(nb_trihedral_pairs * 3);
	for (i = 0; i < nb_trihedral_pairs; i++) {
		for (j = 0; j < 3; j++) {
			for (h = 0; h < 3; h++) {
				a = Trihedral_pairs[i * 9 + j * 3 + h];
				subset[h] = a;
				}
			int_vec_heapsort(subset, 3);
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
			int_vec_heapsort(subset, 3);
			rk = rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_col_sets[i * 3 + j] = rk;
			}
		}

	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs "
				"The trihedral pairs row sets:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Trihedral_pairs_row_sets, 120, 3, 
			FALSE /* f_tex */);
		//print_integer_matrix_with_standard_labels(cout,
		//Trihedral_pairs_row_sets, 120, 3, TRUE /* f_tex */);
		cout << "The trihedral pairs col sets:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Trihedral_pairs_col_sets, 120, 3, 
			FALSE /* f_tex */);
		//print_integer_matrix_with_standard_labels(cout,
		//Trihedral_pairs_col_sets, 120, 3, TRUE /* f_tex */);
		}

	Classify_trihedral_pairs_row_values = NEW_OBJECT(classify);
	Classify_trihedral_pairs_row_values->init(
		Trihedral_pairs_row_sets, 120 * 3, FALSE, 0);

	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs "
				"sorted row values:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Classify_trihedral_pairs_row_values->data_sorted, 
			120 * 3 / 10, 10, FALSE /* f_tex */);
		//int_matrix_print(
		//Classify_trihedral_pairs_row_values->data_sorted, 120, 3);
		//cout << endl;
		}

	Classify_trihedral_pairs_col_values = NEW_OBJECT(classify);
	Classify_trihedral_pairs_col_values->init(
		Trihedral_pairs_col_sets,
		120 * 3, FALSE, 0);

	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs "
				"sorted col values:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Classify_trihedral_pairs_col_values->data_sorted, 
			120 * 3 / 10, 10, FALSE /* f_tex */);
		}
	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs done" << endl;
		}
}

void surface_domain::make_Tijk(int *T, int i, int j, int k)
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

void surface_domain::make_Tlmnp(int *T, int l, int m, int n, int p)
{
	int subset[4];
	int complement[2];
	int size_complement;
	int r, s;

	subset[0] = l;
	subset[1] = m;
	subset[2] = n;
	subset[3] = p;
	int_vec_heapsort(subset, 4);
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

void surface_domain::make_Tdefght(int *T,
		int d, int e, int f, int g, int h, int t)
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

void surface_domain::make_Eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l;
	char str[1000];

	if (f_v) {
		cout << "surface_domain::make_Eckardt_points" << endl;
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
		cout << "surface_domain::make_Eckardt_points done" << endl;
		}
}


void surface_domain::init_Trihedral_to_Eckardt(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk;
	int tritangent_plane[3];

	if (f_v) {
		cout << "surface_domain::init_Trihedral_to_Eckardt" << endl;
		}
	nb_trihedral_to_Eckardt = nb_trihedral_pairs * 6;
	Trihedral_to_Eckardt = NEW_int(nb_trihedral_to_Eckardt);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
#if 0
			if (t == 111) {
				cout << "surface_domain::init_Trihedral_to_Eckardt "
						"t=" << t << " tritangent_plane row "
						<< i << " = ";
				int_vec_print(cout, tritangent_plane, 3);
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
				cout << "surface_domain::init_Trihedral_to_Eckardt "
						"tritangent_plane=";
				int_vec_print(cout, tritangent_plane, 3);
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
		cout << "surface_domain::init_Trihedral_to_Eckardt done" << endl;
		}
}


int surface_domain::Eckardt_point_from_tritangent_plane(
		int *tritangent_plane)
{
	int a, b, c, rk;
	eckardt_point E;

	int_vec_heapsort(tritangent_plane, 3);
	a = tritangent_plane[0];
	b = tritangent_plane[1];
	c = tritangent_plane[2];
	if (a < 6) {
		E.init2(a, b - 6);
		}
	else {
		if (a < 12) {
			cout << "surface_domain::Eckardt_point_from_"
					"tritangent_plane a < 12" << endl;
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

void surface_domain::init_collinear_Eckardt_triples(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, rk;
	int subset[3];

	if (f_v) {
		cout << "surface_domain::init_collinear_Eckardt_triples" << endl;
		}
	nb_collinear_Eckardt_triples = nb_trihedral_pairs * 2;
	collinear_Eckardt_triples_rank = NEW_int(nb_collinear_Eckardt_triples);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			int_vec_copy(Trihedral_to_Eckardt + 6 * t + i * 3, 
				subset, 3);
			int_vec_heapsort(subset, 3);
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
		cout << "surface_domain::init_collinear_Eckardt_triples done" << endl;
		}
}

void surface_domain::find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(
	int *E_idx, int nb_E, 
	int *&T_idx, int &nb_T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nCk, h, k, rk, idx, i, t_idx;
	int subset[3];
	int set[3];
	
	if (f_v) {
		cout << "surface_domain::find_trihedral_pairs_from_collinear_"
				"triples_of_Eckardt_points" << endl;
		}
	nCk = int_n_choose_k(nb_E, 3);
	T_idx = NEW_int(nCk);
	nb_T = 0;
	for (h = 0; h < nCk; h++) {
		//cout << "subset " << h << " / " << nCk << ":";
		unrank_k_subset(h, subset, nb_E, 3);
		//int_vec_print(cout, subset, 3);
		//cout << " = ";

		for (k = 0; k < 3; k++) {
			set[k] = E_idx[subset[k]];
			}
		//int_vec_print(cout, set, 3);
		//cout << " = ";
		int_vec_heapsort(set, 3);
		
		rk = rank_k_subset(set, nb_Eckardt_points, 3);


		//int_vec_print(cout, set, 3);
		//cout << " rk=" << rk << endl;

		if (int_vec_search(
			Classify_collinear_Eckardt_triples->data_sorted, 
			nb_collinear_Eckardt_triples, rk, idx)) {
			//cout << "idx=" << idx << endl;
			for (i = idx; i >= 0; i--) {
				//cout << "i=" << i << " value="
				// << Classify_collinear_Eckardt_triples->data_sorted[i]
				// << " collinear triple index = "
				// << Classify_collinear_Eckardt_triples->sorting_perm_inv[
				// i] / 3 << endl;
				if (Classify_collinear_Eckardt_triples->data_sorted[i] != rk) {
					break;
					}
				t_idx =
				Classify_collinear_Eckardt_triples->sorting_perm_inv[i] / 2;

#if 0
				int idx2, j;
				
				if (!int_vec_search(T_idx, nb_T, t_idx, idx2)) {
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

	int t2, f2, l2, sz;
	int t1, f1, /*l1,*/ pt;
	
	for (t2 = 0; t2 < C.second_nb_types; t2++) {
		f2 = C.second_type_first[t2];
		l2 = C.second_type_len[t2];
		sz = C.second_data_sorted[f2];
		if (sz != 1) {
			continue;
			}
		//cout << "clebsch::clebsch_map_print_fibers fibers of size "
		// << sz << ":" << endl;
		//*fp << "There are " << l2 << " fibers of size " << sz
		// << ":\\\\" << endl;
		for (i = 0; i < l2; i++) {
			t1 = C.second_sorting_perm_inv[f2 + i];
			f1 = C.type_first[t1];
			//l1 = C.type_len[t1];
			pt = C.data_sorted[f1];
			T_idx[i] = pt;
#if 0
			//*fp << "Arc pt " << pt << ", fiber $\\{"; // << l1
			// << " surface points in the list of Pts (local numbering): ";
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
	int_vec_print(cout, T_idx, nb_T);
	cout << endl;
	for (i = 0; i < nb_T; i++) {
		cout << i << " / " << nb_T << " T_{" 
			<< Trihedral_pair_labels[T_idx[i]] << "}" << endl;
		}
	if (f_v) {
		cout << "surface_domain::find_trihedral_pairs_from_collinear_"
				"triples_of_Eckardt_points done" << endl;
		}
}

void surface_domain::multiply_conic_times_linear(int *six_coeff,
	int *three_coeff, int *ten_coeff, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear" << endl;
		}


	int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 6; i++) {
		a = six_coeff[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff[j];
			c = F->mult(a, b);
			int_vec_add(Poly2->Monomials + i * 3, 
				Poly1->Monomials + j * 3, M, 3);
			idx = Poly3->index_of_monomial(M);
			if (idx >= 10) {
				cout << "surface_domain::multiply_conic_times_linear "
						"idx >= 10" << endl;
				exit(1);
				}
			ten_coeff[idx] = F->add(ten_coeff[idx], c);
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::multiply_conic_times_linear done" << endl;
		}
}

void surface_domain::multiply_linear_times_linear_times_linear(
	int *three_coeff1, int *three_coeff2, int *three_coeff3, 
	int *ten_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear" << endl;
		}

	int_vec_zero(ten_coeff, 10);
	for (i = 0; i < 3; i++) {
		a = three_coeff1[i];
		for (j = 0; j < 3; j++) {
			b = three_coeff2[j];
			for (k = 0; k < 3; k++) {
				c = three_coeff3[k];
				d = F->mult3(a, b, c);
				int_vec_add3(Poly1->Monomials + i * 3, 
					Poly1->Monomials + j * 3, 
					Poly1->Monomials + k * 3, 
					M, 3);
				idx = Poly3->index_of_monomial(M);
				if (idx >= 10) {
					cout << "surface::multiply_linear_times_"
							"linear_times_linear idx >= 10" << endl;
					exit(1);
					}
				ten_coeff[idx] = F->add(ten_coeff[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear done" << endl;
		}
}

void surface_domain::multiply_linear_times_linear_times_linear_in_space(
	int *four_coeff1, int *four_coeff2, int *four_coeff3, 
	int *twenty_coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[4];

	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear_in_space" << endl;
		}

	int_vec_zero(twenty_coeff, 20);
	for (i = 0; i < 4; i++) {
		a = four_coeff1[i];
		for (j = 0; j < 4; j++) {
			b = four_coeff2[j];
			for (k = 0; k < 4; k++) {
				c = four_coeff3[k];
				d = F->mult3(a, b, c);
				int_vec_add3(Poly1_4->Monomials + i * 4, 
					Poly1_4->Monomials + j * 4, 
					Poly1_4->Monomials + k * 4, 
					M, 4);
				idx = index_of_monomial(M);
				if (idx >= 20) {
					cout << "surface_domain::multiply_linear_times_linear_"
							"times_linear_in_space idx >= 20" << endl;
					exit(1);
					}
				twenty_coeff[idx] = F->add(twenty_coeff[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::multiply_linear_times_linear_"
				"times_linear_in_space done" << endl;
		}
}

void surface_domain::multiply_Poly2_3_times_Poly2_3(
	int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3" << endl;
		}

	int_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly2->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			int_vec_add(Poly2->Monomials + i * 3, 
				Poly2->Monomials + j * 3, 
				M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::multiply_Poly2_3_times_Poly2_3 done" << endl;
		}
}

void surface_domain::multiply_Poly1_3_times_Poly3_3(int *input1, int *input2,
	int *result, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, idx;
	int M[3];

	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3" << endl;
		}

	int_vec_zero(result, Poly4_x123->nb_monomials);
	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = input1[i];
		for (j = 0; j < Poly3->nb_monomials; j++) {
			b = input2[j];
			c = F->mult(a, b);
			int_vec_add(Poly1->Monomials + i * 3, 
				Poly3->Monomials + j * 3, M, 3);
			idx = Poly4_x123->index_of_monomial(M);
			result[idx] = F->add(result[idx], c);
			}
		}
	
	if (f_v) {
		cout << "surface_domain::multiply_Poly1_3_times_Poly3_3 done" << endl;
		}
}

void surface_domain::web_of_cubic_curves(int *arc6, int *&curves,
	int verbose_level)
// curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int *bisecants;
	int *conics;
	int ten_coeff[10];
	int a, rk, i, j, k, l, m, n;
	int ij, kl, mn;

	if (f_v) {
		cout << "surface::web_of_cubic_curves" << endl;
		}
	P2->compute_bisecants_and_conics(arc6, 
		bisecants, conics, verbose_level);
	
	curves = NEW_int(45 * 10);

	
	a = 0;

	// the first 30 curves:
	for (rk = 0; rk < 30; rk++, a++) {
		ordered_pair_unrank(rk, i, j, 6);
		ij = ij2k(i, j, 6);
		multiply_conic_times_linear(conics + j * 6, 
			bisecants + ij * 3, 
			ten_coeff, 
			0 /* verbose_level */);
		int_vec_copy(ten_coeff, curves + a * 10, 10);
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
		int_vec_copy(ten_coeff, curves + a * 10, 10);
		}

	if (a != 45) {
		cout << "surface_domain::web_of_cubic_curves a != 45" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "The web of cubic curves is:" << endl;
		int_matrix_print(curves, 45, 10);
		}

	FREE_int(bisecants);
	FREE_int(conics);

	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves done" << endl;
		}
}

void surface_domain::web_of_cubic_curves_rank_of_foursubsets(
	int *Web_of_cubic_curves, 
	int *&rk, int &N, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int set[4], i, j, a;
	int B[4 * 10];

	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves_rank_of_foursubsets" << endl;
		}
	if (f_v) {
		cout << "web of cubic curves:" << endl;
		int_matrix_print(Web_of_cubic_curves, 45, 10);
		}
	N = int_n_choose_k(45, 4);
	rk = NEW_int(N);
	for (i = 0; i < N; i++) {
		unrank_k_subset(i, set, 45, 4);
		if (f_v) {
			cout << "subset " << i << " / " << N << " is ";
			int_vec_print(cout, set, 4);
			cout << endl;
			}
		for (j = 0; j < 4; j++) {
			a = set[j];
			int_vec_copy(Web_of_cubic_curves + a * 10, 
				B + j * 10, 10);
			}
		rk[i] = F->rank_of_rectangular_matrix(B, 
			4, 10, 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface_domain::web_of_cubic_curves_rank_of_foursubsets done" << endl;
		}
}

void
surface_domain::create_web_of_cubic_curves_and_equations_based_on_four_tritangent_planes(
	int *arc6, int *base_curves4, 
	int *&Web_of_cubic_curves, int *&The_plane_equations, 
	int verbose_level)
// Web_of_cubic_curves[45 * 10]
{
	int f_v = (verbose_level >= 1);
	int h, rk, idx;
	int *base_curves;
	int *curves;
	int *curves_t;

	if (f_v) {
		cout << "surface_domain::create_web_of_cubic_curves_and_equations_based_"
				"on_four_tritangent_planes" << endl;
		}

	web_of_cubic_curves(arc6, Web_of_cubic_curves, verbose_level);

	base_curves = NEW_int(5 * 10);
	curves = NEW_int(5 * 10);
	curves_t = NEW_int(10 * 5);



	for (h = 0; h < 4; h++) {
		int_vec_copy(Web_of_cubic_curves + base_curves4[h] * 10, 
			base_curves + h * 10, 10);
		}

	if (f_v) {
		cout << "base_curves:" << endl;
		int_matrix_print(base_curves, 4, 10);
		}

	// find the plane equations:

	The_plane_equations = NEW_int(45 * 4);

	for (h = 0; h < 45; h++) {

		if (f_v) {
			cout << "h=" << h << " / " << 45 << ":" << endl;
			}
		
		if (int_vec_search_linear(base_curves4, 4, h, idx)) {
			int_vec_zero(The_plane_equations + h * 4, 4);
			The_plane_equations[h * 4 + idx] = 1;
			}
		else {
			int_vec_copy(base_curves, curves, 4 * 10);
			int_vec_copy(Web_of_cubic_curves + h * 10, 
				curves + 4 * 10, 10);
		
			if (f_v) {
				cout << "h=" << h << " / " << 45 
					<< " the system is:" << endl;
				int_matrix_print(curves, 5, 10);
				}

			F->transpose_matrix(curves, curves_t, 5, 10);

			if (f_v) {
				cout << "after transpose:" << endl;
				int_matrix_print(curves_t, 10, 5);
				}
		
			rk = F->RREF_and_kernel(5, 10, curves_t, 
				0 /* verbose_level */);
			if (rk != 4) {
				cout << "surface::create_surface_and_planes_from_"
						"trihedral_pair_and_arc the rank of the "
						"system is not equal to 4" << endl;
				cout << "rk = " << rk << endl;
				exit(1);
				}
			if (curves_t[4 * 5 + 4] != F->negate(1)) {
				cout << "h=" << h << " / " << 2 
					<< " curves_t[4 * 5 + 4] != -1" << endl;
				exit(1);
				}
			int_vec_copy(curves_t + 4 * 5, 
				The_plane_equations + h * 4, 4);

			F->PG_element_normalize(
				The_plane_equations + h * 4, 1, 4);
			
			}
		if (f_v) {
			cout << "h=" << h << " / " << 45 
				<< ": the plane equation is ";
			int_vec_print(cout, The_plane_equations + h * 4, 4);
			cout << endl;
			}
		

		}
	if (f_v) {
		cout << "the plane equations are: " << endl;
		int_matrix_print(The_plane_equations, 45, 4);
		cout << endl;	
		}

	FREE_int(base_curves);
	FREE_int(curves);
	FREE_int(curves_t);

	if (f_v) {
		cout << "surface_domain::create_web_of_cubic_curves_and_equations_"
				"based_on_four_tritangent_planes done" << endl;
		}
}

void surface_domain::create_equations_for_pencil_of_surfaces_from_trihedral_pair(
	int *The_six_plane_equations, int *The_surface_equations, 
	int verbose_level)
// The_surface_equations[(q + 1) * 20]
{
	int f_v = (verbose_level >= 1);
	int v[2];
	int l;
	int eqn_F[20];
	int eqn_G[20];
	int eqn_F2[20];
	int eqn_G2[20];

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_"
				"from_trihedral_pair" << endl;
		}
	

	for (l = 0; l < q + 1; l++) {
		F->PG_element_unrank_modified(v, 1, 2, l);
		
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

		int_vec_copy(eqn_F, eqn_F2, 20);
		F->scalar_multiply_vector_in_place(v[0], eqn_F2, 20);
		int_vec_copy(eqn_G, eqn_G2, 20);
		F->scalar_multiply_vector_in_place(v[1], eqn_G2, 20);
		F->add_vector(eqn_F2, eqn_G2, 
			The_surface_equations + l * 20, 20);
		F->PG_element_normalize(
			The_surface_equations + l * 20, 1, 20);
		}

	if (f_v) {
		cout << "surface_domain::create_equations_for_pencil_of_surfaces_"
				"from_trihedral_pair done" << endl;
		}
}

void surface_domain::create_lambda_from_trihedral_pair_and_arc(
	int *arc6, 
	int *Web_of_cubic_curves, 
	int *The_plane_equations, int t_idx, 
	int &lambda, int &lambda_rk, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int row_col_Eckardt_points[6];
	int six_curves[6 * 10];
	int pt, f_point_was_found;
	int v[3];
	int w[2];
	int evals[6];
	int evals_for_point[6];
	int pt_on_surface[4];
	int a, b, ma, bv;

	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
				"t_idx=" << t_idx << endl;
		}
	
	if (f_v) {
		cout << "Trihedral pair T_{" << Trihedral_pair_labels[t_idx] << "}" 
			<< endl;
		}

	int_vec_copy(Trihedral_to_Eckardt + t_idx * 6, 
		row_col_Eckardt_points, 6);
	
	if (f_v) {
		cout << "row_col_Eckardt_points = ";
		int_vec_print(cout, row_col_Eckardt_points, 6);
		cout << endl;
		}

	

	extract_six_curves_from_web(Web_of_cubic_curves, 
		row_col_Eckardt_points, six_curves, verbose_level);

	if (f_v) {
		cout << "The six curves are:" << endl;
		int_matrix_print(six_curves, 6, 10);
		}
		


	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
				"before find_point_not_on_six_curves" << endl;
		}
	find_point_not_on_six_curves(arc6, six_curves, 
		pt, f_point_was_found, verbose_level);
	if (!f_point_was_found) {
		lambda = 1;
		}
	else {
		if (f_v) {
			cout << "surface_domain::create_lambda_from_trihedral_pair_and_arc "
					"after find_point_not_on_six_curves" << endl;
			cout << "pt=" << pt << endl;
			}

		Poly3->unrank_point(v, pt);
		for (i = 0; i < 6; i++) {
			evals[i] = Poly3->evaluate_at_a_point(
				six_curves + i * 10, v);
			}

		if (f_v) {
			cout << "The point pt=" << pt << " = ";
			int_vec_print(cout, v, 3);
			cout << " is nonzero on all plane sections of "
					"the trihedral pair. The values are ";
			int_vec_print(cout, evals, 6);
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
	F->PG_element_rank_modified(w, 1, 2, lambda_rk);
	
	if (f_v) {
		cout << "surface_domain::create_lambda_from_trihedral_"
				"pair_and_arc done" << endl;
		}
}


void surface_domain::create_surface_equation_from_trihedral_pair(int *arc6,
	int *Web_of_cubic_curves, 
	int *The_plane_equations, int t_idx, int *surface_equation, 
	int &lambda, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *The_surface_equations;
	int row_col_Eckardt_points[6];
	int The_six_plane_equations[6 * 4];
	int lambda_rk;

	if (f_v) {
		cout << "surface_domain::create_surface_equation_from_"
				"trihedral_pair t_idx=" << t_idx << endl;
		}
	

	int_vec_copy(Trihedral_to_Eckardt + t_idx * 6, row_col_Eckardt_points, 6);

	int_vec_copy(The_plane_equations + row_col_Eckardt_points[0] * 4,
			The_six_plane_equations, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[1] * 4,
			The_six_plane_equations + 4, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[2] * 4,
			The_six_plane_equations + 8, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[3] * 4,
			The_six_plane_equations + 12, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[4] * 4,
			The_six_plane_equations + 16, 4);
	int_vec_copy(The_plane_equations + row_col_Eckardt_points[5] * 4,
			The_six_plane_equations + 20, 4);


	The_surface_equations = NEW_int((q + 1) * 20);

	create_equations_for_pencil_of_surfaces_from_trihedral_pair(
		The_six_plane_equations, The_surface_equations, 
		verbose_level - 2);

	create_lambda_from_trihedral_pair_and_arc(arc6, 
		Web_of_cubic_curves, 
		The_plane_equations, t_idx, lambda, lambda_rk, 
		verbose_level - 2);
	
	int_vec_copy(The_surface_equations + lambda_rk * 20, 
		surface_equation, 20);

	FREE_int(The_surface_equations);

	if (f_v) {
		cout << "surface_domain::create_surface_equation_from_"
				"trihedral_pair done" << endl;
		}
}

void surface_domain::extract_six_curves_from_web(
	int *Web_of_cubic_curves,
	int *row_col_Eckardt_points, int *six_curves, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "surface_domain::extract_six_curves_from_web" << endl;
		}
	for (i = 0; i < 6; i++) {
		int_vec_copy(Web_of_cubic_curves + row_col_Eckardt_points[i] * 10, 
			six_curves + i * 10, 10);
		}

	if (f_v) {
		cout << "The six curves are:" << endl;
		int_matrix_print(six_curves, 6, 10);
		}
	if (f_v) {
		cout << "surface_domain::extract_six_curves_from_web done" << endl;
		}
}

void surface_domain::find_point_not_on_six_curves(int *arc6,
	int *six_curves,
	int &pt, int &f_point_was_found, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int v[3];
	int i;
	int idx, a;
	
	if (f_v) {
		cout << "surface_domain::find_point_not_on_six_curves" << endl;
		cout << "surface_domain::find_point_not_on_six_curves "
			"P2->N_points="
			<< P2->N_points << endl;
		}
	pt = -1;
	for (pt = 0; pt < P2->N_points; pt++) {
		if (int_vec_search_linear(arc6, 6, pt, idx)) {
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
		cout << "could not find a point which is not on "
				"any of the curve" << endl;
		f_point_was_found = FALSE;
		}
	else {
		f_point_was_found = TRUE;
		}
	if (f_v) {
		cout << "surface_domain::find_point_not_on_six_curves "
				"done" << endl;
		}
}

int surface_domain::plane_from_three_lines(int *three_lines,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Basis[6 * 4];
	int rk;
	
	if (f_v) {
		cout << "surface_domain::plane_from_three_lines" << endl;
		}
	unrank_lines(Basis, three_lines, 3);
	rk = F->Gauss_easy(Basis, 6, 4);
	if (rk != 3) {
		cout << "surface_domain::plane_from_three_lines rk != 3" << endl;
		exit(1);
		}
	rk = rank_plane(Basis);
	
	if (f_v) {
		cout << "surface_domain::plane_from_three_lines done" << endl;
		}
	return rk;
}

void surface_domain::Trihedral_pairs_to_planes(int *Lines, int *Planes,
	int verbose_level)
// Planes[nb_trihedral_pairs * 6]
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk;
	int tritangent_plane[3];
	int three_lines[3];

	if (f_v) {
		cout << "surface_domain::Trihedral_pairs_to_planes" << endl;
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
		cout << "surface_domain::Trihedral_pairs_to_planes done" << endl;
		}
}

void surface_domain::create_surface_family_S(int a,
	int *Lines27,
	int *equation20, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::create_surface_family_S" << endl;
		}

	int nb_E = 0;
	int b = 1;
	int alpha, beta;

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"creating surface for a=" << a << ":" << endl;
		}

	create_surface_ab(a, b,
		equation20,
		Lines27,
		alpha, beta, nb_E, 
		0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"The double six is:" << endl;
		int_matrix_print(Lines27, 2, 6);
		cout << "The lines are : ";
		int_vec_print(cout, Lines27, 27);
		cout << endl;
		}

	if (f_v) {
		cout << "surface_domain::create_surface_family_S "
				"done" << endl;
		}
}

void surface_domain::compute_tritangent_planes(int *Lines,
	int *&Tritangent_planes, int &nb_tritangent_planes, 
	int *&Unitangent_planes, int &nb_unitangent_planes, 
	int *&Lines_in_tritangent_plane, 
	int *&Line_in_unitangent_plane, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Inc_lines_planes;
	int *The_plane_type;
	int nb_planes;
	int i, j, h, c;

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes" << endl;
		}
	if (f_v) {
		cout << "Lines=" << endl;
		int_vec_print(cout, Lines, 27);
		cout << endl;
		}
	P->line_plane_incidence_matrix_restricted(Lines, 27, 
		Inc_lines_planes, nb_planes, 0 /* verbose_level */);

	The_plane_type = NEW_int(nb_planes);
	int_vec_zero(The_plane_type, nb_planes);

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
		cout << "surface_domain::compute_tritangent_planes The plane type is: ";
		Plane_type.print_naked(TRUE);
		cout << endl;
		}


	Plane_type.get_class_by_value(Tritangent_planes, 
		nb_tritangent_planes, 3 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes "
				"The tritangent planes are: ";
		int_vec_print(cout, Tritangent_planes, nb_tritangent_planes);
		cout << endl;
		}
	Lines_in_tritangent_plane = NEW_int(nb_tritangent_planes * 3);
	for (h = 0; h < nb_tritangent_planes; h++) {
		j = Tritangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Lines_in_tritangent_plane[h * 3 + c++] = i;
				}
			}
		if (c != 3) {
			cout << "surface_domain::compute_tritangent_planes c != 3" << endl;
			exit(1);
			}
		}


	Plane_type.get_class_by_value(Unitangent_planes, 
		nb_unitangent_planes, 1 /* value */, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes "
				"The unitangent planes are: ";
		int_vec_print(cout, Unitangent_planes, nb_unitangent_planes);
		cout << endl;
		}
	Line_in_unitangent_plane = NEW_int(nb_unitangent_planes);
	for (h = 0; h < nb_unitangent_planes; h++) {
		j = Unitangent_planes[h];
		c = 0;
		for (i = 0; i < 27; i++) {
			if (Inc_lines_planes[i * nb_planes + j]) {
				Line_in_unitangent_plane[h * 1 + c++] = i;
				}
			}
		if (c != 1) {
			cout << "surface_domain::compute_tritangent_planes c != 1" << endl;
			exit(1);
			}
		}

	FREE_int(Inc_lines_planes);
	FREE_int(The_plane_type);

	if (f_v) {
		cout << "surface_domain::compute_tritangent_planes done" << endl;
		}
}

void surface_domain::compute_external_lines_on_three_tritangent_planes(
	int *Lines, int *&External_lines, int &nb_external_lines, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes" << endl;
		}

	int *Tritangent_planes;
	int nb_tritangent_planes;
	int *Lines_in_tritangent_plane; // [nb_tritangent_planes * 3]
	
	int *Unitangent_planes;
	int nb_unitangent_planes;
	int *Line_in_unitangent_plane; // [nb_unitangent_planes]
	
	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes computing "
				"tritangent planes:" << endl;
		}
	compute_tritangent_planes(Lines, 
		Tritangent_planes, nb_tritangent_planes, 
		Unitangent_planes, nb_unitangent_planes, 
		Lines_in_tritangent_plane, 
		Line_in_unitangent_plane, 
		verbose_level);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes Lines_in_"
				"tritangent_plane: " << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Lines_in_tritangent_plane, nb_tritangent_planes, 
			3, FALSE);
		}

	int *Intersection_matrix;
		// [nb_tritangent_planes * nb_tritangent_planes]
	int *Plane_intersections;
	int *Plane_intersections_general;
	int rk, idx;



	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_"
				"three_tritangent_planes Computing intersection "
				"matrix of tritangent planes:" << endl;
		}
		
	P->plane_intersection_matrix_in_three_space(Tritangent_planes, 
		nb_tritangent_planes, Intersection_matrix, 
		0 /* verbose_level */);

	Plane_intersections =
			NEW_int(nb_tritangent_planes * nb_tritangent_planes);
	Plane_intersections_general =
			NEW_int(nb_tritangent_planes * nb_tritangent_planes);
	for (i = 0; i < nb_tritangent_planes; i++) {
		for (j = 0; j < nb_tritangent_planes; j++) {
			Plane_intersections[i * nb_tritangent_planes + j] = -1;
			Plane_intersections_general[i * nb_tritangent_planes + j] = -1;
			if (j != i) {
				rk = Intersection_matrix[i * nb_tritangent_planes + j];
				if (int_vec_search_linear(
					Lines, 27, rk, idx)) {
					Plane_intersections[i * nb_tritangent_planes + j] = idx;
					}
				else {
					Plane_intersections_general[
						i * nb_tritangent_planes + j] = rk;
					}
				}
			}
		}

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The tritangent planes intersecting "
				"in surface lines:" << endl;
		print_integer_matrix_with_standard_labels(cout, 
			Plane_intersections, nb_tritangent_planes, 
			nb_tritangent_planes, FALSE);
		}


	classify Plane_intersection_type;

	Plane_intersection_type.init(Plane_intersections, 
		nb_tritangent_planes * nb_tritangent_planes, TRUE, 0);
	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The surface lines in terms "
				"of plane intersections are: ";
		Plane_intersection_type.print_naked(TRUE);
		cout << endl;
		}


	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The tritangent planes "
				"intersecting in general lines:" << endl;
		print_integer_matrix_with_standard_labels(cout,
				Plane_intersections_general, nb_tritangent_planes,
				nb_tritangent_planes, FALSE);
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

	int_vec_heapsort(External_lines, nb_external_lines);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes The non-surface lines which are on "
				"three tritangent planes are:" << endl;
		int_vec_print(cout, External_lines, nb_external_lines);
		cout << endl;
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes these lines are:" << endl;
		P->Grass_lines->print_set(External_lines, nb_external_lines);
		}
	
	FREE_int(Tritangent_planes);
	FREE_int(Lines_in_tritangent_plane);
	FREE_int(Unitangent_planes);
	FREE_int(Line_in_unitangent_plane);
	FREE_int(Intersection_matrix);
	FREE_int(Plane_intersections);
	FREE_int(Plane_intersections_general);

	if (f_v) {
		cout << "surface_domain::compute_external_lines_on_three_"
				"tritangent_planes done" << endl;
		}
}

void surface_domain::init_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, ij, u, v, l, m, n, h, a, b, c;
	int set[6];
	int size_complement;
	
	if (f_v) {
		cout << "surface_domain::init_double_sixes" << endl;
		}
	Double_six = NEW_int(36 * 12);
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
		//int_vec_print(cout, set, 6);
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
		cout << "surface_domain::init_double_sixes h != 36" << endl;
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
			sprintf(str, "D_{%d%d}", a + 1, b + 1);
			}
		else {
			v = i - 16;
			unrank_k_subset(v, set, 6, 3);
			set_complement(set, 3 /* subset_size */, set + 3, 
				size_complement, 6 /* universal_set_size */);
			a = set[0];
			b = set[1];
			c = set[2];
			sprintf(str, "D_{%d%d%d}", a + 1, b + 1, c + 1);
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
		cout << "surface_domain::init_double_sixes done" << endl;
		}
}

void surface_domain::create_half_double_sixes(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, c, ij, v, l;
	int set[6];
	int size_complement;
	
	if (f_v) {
		cout << "surface_domain::create_half_double_sixes" << endl;
		}
	Half_double_sixes = NEW_int(72 * 6);
	Half_double_six_to_double_six = NEW_int(72);
	Half_double_six_to_double_six_row = NEW_int(72);

	int_vec_copy(Double_six, Half_double_sixes, 36 * 12);
	for (i = 0; i < 36; i++) {
		for (j = 0; j < 2; j++) {
			int_vec_heapsort(
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
				sprintf(str, "D_{%d%d}", a + 1, b + 1);
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
				sprintf(str, "D_{%d%d%d}",
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
		cout << "surface_domain::create_half_double_sixes done" << endl;
		}
}

int surface_domain::find_half_double_six(int *half_double_six)
{
	int i;

	int_vec_heapsort(half_double_six, 6);
	for (i = 0; i < 72; i++) {
		if (int_vec_compare(half_double_six, 
			Half_double_sixes + i * 6, 6) == 0) {
			return i;
			}
		}
	cout << "surface_domain::find_half_double_six did not find "
			"half double six" << endl;
	exit(1);
}

void surface_domain::ijklm2n(int i, int j, int k, int l, int m, int &n)
{
	int v[6];
	int size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	v[4] = m;
	set_complement_safe(v, 5, v + 5, size_complement, 6);
	if (size_complement != 1) {
		cout << "surface_domain::ijklm2n size_complement != 1" << endl;
		exit(1);
		}
	n = v[5];
}

void surface_domain::ijkl2mn(int i, int j, int k, int l, int &m, int &n)
{
	int v[6];
	int size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	v[3] = l;
	set_complement_safe(v, 4, v + 4, size_complement, 6);
	if (size_complement != 2) {
		cout << "surface_domain::ijkl2mn size_complement != 2" << endl;
		exit(1);
		}
	m = v[4];
	n = v[5];
}

void surface_domain::ijk2lmn(int i, int j, int k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;

	v[0] = i;
	v[1] = j;
	v[2] = k;
	cout << "surface_domain::ijk2lmn v=";
	int_vec_print(cout, v, 3);
	cout << endl;
	set_complement_safe(v, 3, v + 3, size_complement, 6);
	if (size_complement != 3) {
		cout << "surface_domain::ijk2lmn size_complement != 3" << endl;
		cout << "size_complement=" << size_complement << endl;
		exit(1);
		}
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface_domain::ij2klmn(int i, int j, int &k, int &l, int &m, int &n)
{
	int v[6];
	int size_complement;

	v[0] = i;
	v[1] = j;
	set_complement_safe(v, 2, v + 2, size_complement, 6);
	if (size_complement != 4) {
		cout << "surface_domain::ij2klmn size_complement != 4" << endl;
		exit(1);
		}
	k = v[2];
	l = v[3];
	m = v[4];
	n = v[5];
}

void surface_domain::get_half_double_six_associated_with_Clebsch_map(
	int line1, int line2, int transversal, 
	int hds[6],
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t1, t2, t3;
	int i, j, k, l, m, n;
	int i1, j1;
	int null;
	
	if (f_v) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map" << endl;
		}

	if (line1 > line2) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map line1 > line2" << endl;
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
				cout << "surface_domain::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
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
				cout << "surface_domain::get_half_doble_six_associated_"
						"with_Clebsch_map not {i,j} = {i1,j1}" << endl;
				exit(1);
				}
			}
		}
	else if (t1 == 0 && t2 == 1) { // ai and bi:
		index_of_line(line1, i, null);
		index_of_line(line2, j, null);
		if (j != i) {
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map j != i" << endl;
			exit(1);
			}
		if (t3 != 2) {
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map t3 != 2" << endl;
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
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
				int tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
				int tmp = k;
				k = j;
				j = tmp;
				}
			if (j1 != j) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
			cout << "surface_domain::get_half_double_six_associated_"
					"with_Clebsch_map error" << endl;
			exit(1);
			}
		if (t3 == 0) { // ai
			index_of_line(transversal, i1, null);
			if (i1 != i) {
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
				cout << "surface_domain::get_half_double_six_associated_"
						"with_Clebsch_map error" << endl;
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
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map error" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "surface_domain::get_half_double_six_associated_"
				"with_Clebsch_map done" << endl;
		}
}

void surface_domain::prepare_clebsch_map(int ds, int ds_row,
	int &line1, int &line2, int &transversal, 
	int verbose_level)
{
	int ij, i, j, k, l, m, n, size_complement;
	int set[6];
	
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

int surface_domain::clebsch_map(int *Lines, int *Pts, int nb_pts,
	int line_idx[2], int plane_rk, 
	int *Image_rk, int *Image_coeff, 
	int verbose_level)
// assuming: 
// In:
// Lines[27]
// Pts[nb_pts]
// Out:
// Image_rk[nb_pts]  (image point in the plane in local coordinates)
//   Note Image_rk[i] is -1 if Pts[i] does not have an image.
// Image_coeff[nb_pts * 4] (image point in the plane in PG(3,q) coordinates)
{
	int f_v = (verbose_level >= 1);
	int Plane[4 * 4];
	int Line_a[2 * 4];
	int Line_b[2 * 4];
	int Dual_planes[4 * 4]; 
		// dual coefficients of three planes:
		// the first plane is line_a together with the surface point
		// the second plane is line_b together with the surface point
		// the third plane is the plane onto which we map.
		// the fourth row is for the image point.
	int M[4 * 4];
	int v[4];
	int i, h, pt, r;
	int coefficients[3];
	int base_cols[4];
	
	if (f_v) {
		cout << "surface_domain::clebsch_map" << endl;
		}
	P->Grass_planes->unrank_int_here(Plane, plane_rk,
			0 /* verbose_level */);
	r = F->Gauss_simple(Plane, 3, 4, base_cols,
			0 /* verbose_level */);
	if (f_v) {
		cout << "Plane rank " << plane_rk << " :" << endl;
		int_matrix_print(Plane, 3, 4);
		}

	F->RREF_and_kernel(4, 3, Plane, 0 /* verbose_level */);

	if (f_v) {
		cout << "Plane (3 basis vectors and dual coordinates):" << endl;
		int_matrix_print(Plane, 4, 4);
		cout << "base_cols: ";
		int_vec_print(cout, base_cols, r);
		cout << endl;
		}

	// make sure the two lines are not contained in
	// the plane onto which we map:

	// test line_a:
	P->Grass_lines->unrank_int_here(Line_a, 
		Lines[line_idx[0]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line a = " << Line_label_tex[line_idx[0]] 
			<< " = " << Lines[line_idx[0]] << ":" << endl;
		int_matrix_print(Line_a, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_a + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface_domain::clebsch_map Line a lies "
				"inside the hyperplane" << endl;
		return FALSE;
		}

	// test line_b:
	P->Grass_lines->unrank_int_here(Line_b, 
		Lines[line_idx[1]], 0 /* verbose_level */);
	if (f_v) {
		cout << "Line b = " << Line_label_tex[line_idx[1]] 
			<< " = " << Lines[line_idx[1]] << ":" << endl;
		int_matrix_print(Line_b, 2, 4);
		}
	for (i = 0; i < 2; i++) {
		if (F->dot_product(4, Line_b + i * 4, Plane + 3 * 4)) {
			break;
			}
		}
	if (i == 2) {
		cout << "surface_domain::clebsch_map Line b lies "
				"inside the hyperplane" << endl;
		return FALSE;
		}

	// and now, map all surface points:
	for (h = 0; h < nb_pts; h++) {
		pt = Pts[h];

		unrank_point(v, pt);

		int_vec_zero(Image_coeff + h * 4, 4);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts << " is " << pt << " = ";
			int_vec_print(cout, v, 4);
			cout << ":" << endl;
			}

		// make sure the points do not lie on either line_a or line_b
		// because the map is undefined there:
		int_vec_copy(Line_a, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_a" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		int_vec_copy(Line_b, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		if (F->Gauss_easy(M, 3, 4) == 2) {
			if (f_v) {
				cout << "The point is on line_b" << endl;
				}
			Image_rk[h] = -1;
			continue;
			}
		
		// The point is good:

		// Compute the first plane in dual coordinates:
		int_vec_copy(Line_a, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		int_vec_copy(M + 3 * 4, Dual_planes, 4);
		if (f_v) {
			cout << "First plane in dual coordinates: ";
			int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}

		// Compute the second plane in dual coordinates:
		int_vec_copy(Line_b, M, 2 * 4);
		int_vec_copy(v, M + 2 * 4, 4);
		F->RREF_and_kernel(4, 3, M, 0 /* verbose_level */);
		int_vec_copy(M + 3 * 4, Dual_planes + 4, 4);
		if (f_v) {
			cout << "Second plane in dual coordinates: ";
			int_vec_print(cout, M + 3 * 4, 4);
			cout << endl;
			}


		// The third plane is the image
		// plane, given by dual coordinates:
		int_vec_copy(Plane + 3 * 4, Dual_planes + 8, 4);
		if (f_v) {
			cout << "Dual coordinates for all three planes: " << endl;
			int_matrix_print(Dual_planes, 3, 4);
			cout << endl;
			}

		r = F->RREF_and_kernel(4, 3,
				Dual_planes, 0 /* verbose_level */);
		if (f_v) {
			cout << "Dual coordinates and perp: " << endl;
			int_matrix_print(Dual_planes, 4, 4);
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
		F->PG_element_normalize(Dual_planes + 12, 1, 4);
		if (f_v) {
			cout << "intersection point normalized: ";
			int_vec_print(cout, Dual_planes + 12, 4);
			cout << endl;
			}
		int_vec_copy(Dual_planes + 12, Image_coeff + h * 4, 4);
		
		// compute local coordinates of the image point:
		F->reduce_mod_subspace_and_get_coefficient_vector(
			3, 4, Plane, base_cols, 
			Dual_planes + 12, coefficients, 
			0 /* verbose_level */);
		Image_rk[h] = P2->rank_point(coefficients);
		if (f_v) {
			cout << "pt " << h << " / " << nb_pts 
				<< " is " << pt << " : image = ";
			int_vec_print(cout, Image_coeff + h * 4, 4);
			cout << " image = " << Image_rk[h] << endl;
			}
		}
	
	if (f_v) {
		cout << "surface_domain::clebsch_map done" << endl;
		}
	return TRUE;
}

void surface_domain::clebsch_cubics(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	


	if (f_v) {
		cout << "surface_domain::clebsch_cubics" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface::clebsch_cubics f_has_large_"
				"polynomial_domains is FALSE" << endl;
		exit(1);
		}
	int Monomial[27];

	int i, j, idx;

	Clebsch_Pij = NEW_int(3 * 4 * nb_monomials2);
	Clebsch_P = NEW_pint(3 * 4);
	Clebsch_P3 = NEW_pint(3 * 3);

	int_vec_zero(Clebsch_Pij, 3 * 4 * nb_monomials2);


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
	int coeffs[] = {
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
	int c0, c1;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics "
				"Setting up the matrix P:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "i=" << i << " j=" << j << endl;
			int_vec_zero(Monomial, 27);
			c0 = coeffs[(i * 4 + j) * 4 + 0];
			c1 = coeffs[(i * 4 + j) * 4 + 1];
			int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			c0 = coeffs[(i * 4 + j) * 4 + 2];
			c1 = coeffs[(i * 4 + j) * 4 + 3];
			int_vec_zero(Monomial, 27);
			Monomial[c0] = 1;
			Monomial[c1] = 1;
			idx = Poly2_27->index_of_monomial(Monomial);
			Clebsch_P[i * 4 + j][idx] = 1;
			}
		}

	
	if (f_v) {
		cout << "surface_domain::clebsch_cubics the matrix "
				"Clebsch_P is:" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			cout << "Clebsch_P_" << i << "," << j << ":";
			Poly2_27->print_equation(cout, Clebsch_P[i * 4 + j]);
			cout << endl;
			}
		}

	int *Cubics;
	int *Adjugate;
	int *Ad[3 * 3];
	int *C[4];
	int m1;


	if (f_v) {
		cout << "surface_domain::clebsch_cubics allocating cubics" << endl;
		}

	Cubics = NEW_int(4 * nb_monomials6);
	int_vec_zero(Cubics, 4 * nb_monomials6);

	Adjugate = NEW_int(3 * 3 * nb_monomials4);
	int_vec_zero(Adjugate, 3 * 3 * nb_monomials4);

	for (i = 0; i < 4; i++) {
		C[i] = Cubics + i * nb_monomials6;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Ad[i * 3 + j] = Adjugate + (i * 3 + j) * nb_monomials4;
			}
		}

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing "
				"C[3] = the determinant" << endl;
		}
	// compute C[3] as the negative of the determinant
	// of the matrix of the first three columns:
	//int_vec_zero(C[3], nb_monomials6);
	m1 = F->negate(1);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[2 * 4 + 2], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[2 * 4 + 0], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[0 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[2 * 4 + 1], m1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 0],
			Clebsch_P[1 * 4 + 1],
			Clebsch_P[0 * 4 + 2], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 1],
			Clebsch_P[1 * 4 + 2],
			Clebsch_P[0 * 4 + 0], 1, C[3],
			0 /* verbose_level*/);
	multiply_222_27_and_add(Clebsch_P[2 * 4 + 2],
			Clebsch_P[1 * 4 + 0],
			Clebsch_P[0 * 4 + 1], 1, C[3],
			0 /* verbose_level*/);

	int I[3];
	int J[3];
	int size_complement, scalar;

	if (f_v) {
		cout << "surface_domain::clebsch_cubics computing adjugate" << endl;
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
			minor22(Clebsch_P3, I[1], I[2], J[1], J[2], scalar,
					Ad[j * 3 + i], 0 /* verbose_level */);
			}
		}

	// multiply adjugate * last column:
	if (f_v) {
		cout << "surface_domain::clebsch_cubics multiply adjugate "
				"times last column" << endl;
		}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			multiply42_and_add(Ad[i * 3 + j],
					Clebsch_P[j * 4 + 3], C[i], 0 /* verbose_level */);
			}
		}
	
	if (f_v) {
		cout << "surface_domain::clebsch_cubics We have "
				"computed the Clebsch cubics" << endl;
		}
	

	int Y[3];
	int M24[24];
	int h;
	
	Clebsch_coeffs = NEW_int(4 * Poly3->nb_monomials * nb_monomials3);
	int_vec_zero(Clebsch_coeffs,
			4 * Poly3->nb_monomials * nb_monomials3);
	CC = NEW_pint(4 * Poly3->nb_monomials);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < Poly3->nb_monomials; j++) {
			CC[i * Poly3->nb_monomials + j] = 
				Clebsch_coeffs +
					(i * Poly3->nb_monomials + j) * nb_monomials3;
			}
		}
	for (i = 0; i < Poly3->nb_monomials; i++) {
		int_vec_copy(Poly3->Monomials + i * 3, Y, 3);
		for (j = 0; j < nb_monomials6; j++) {
			if (int_vec_compare(Y, Poly6_27->Monomials + j * 27, 3) == 0) {
				int_vec_copy(Poly6_27->Monomials + j * 27 + 3, M24, 24);
				idx = Poly3_24->index_of_monomial(M24);
				for (h = 0; h < 4; h++) {
					CC[h * Poly3->nb_monomials + i][idx] = 
						F->add(CC[h * Poly3->nb_monomials + i][idx],
								C[h][j]);
					}
				}
			}
		}

	if (f_v) {
		print_clebsch_cubics(cout);
		}

	FREE_int(Cubics);
	FREE_int(Adjugate);

	if (f_v) {
		cout << "surface_domain::clebsch_cubics done" << endl;
		}
}

void surface_domain::multiply_222_27_and_add(int *M1, int *M2, int *M3,
	int scalar, int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, k, a, b, c, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply_222_27_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	//int_vec_zero(MM, nb_monomials6);
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
				int_vec_add3(Poly2_27->Monomials + i * 27, 
					Poly2_27->Monomials + j * 27, 
					Poly2_27->Monomials + k * 27, 
					M, 27);
				idx = Poly6_27->index_of_monomial(M);
				if (idx >= nb_monomials6) {
					cout << "surface_domain::multiply_222_27_and_add "
							"idx >= nb_monomials6" << endl;
					exit(1);
					}
				d = F->mult(scalar, d);
				MM[idx] = F->add(MM[idx], d);
				}
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::multiply_222_27_and_add done" << endl;
		}
}

void surface_domain::minor22(int **P3, int i1, int i2, int j1, int j2,
	int scalar, int *Ad, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::minor22" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::minor22 "
				"f_has_large_polynomial_domains is FALSE" << endl;
		exit(1);
		}
	int_vec_zero(Ad, nb_monomials4);
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
			int_vec_add(Poly2_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
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
			int_vec_add(Poly2_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly4_27->index_of_monomial(M);
			if (idx >= nb_monomials4) {
				cout << "surface_domain::minor22 "
						"idx >= nb_monomials4" << endl;
				exit(1);
				}
			d = F->mult(scalar, d);
			d = F->negate(d);
			Ad[idx] = F->add(Ad[idx], d);
			}
		}
	
	
	if (f_v) {
		cout << "surface_domain::minor22 done" << endl;
		}
}

void surface_domain::multiply42_and_add(int *M1, int *M2,
		int *MM, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a, b, d, idx;
	int M[27];

	if (f_v) {
		cout << "surface_domain::multiply42_and_add" << endl;
		}

	if (!f_has_large_polynomial_domains) {
		cout << "surface_domain::multiply42_and_add "
				"f_has_large_polynomial_domains is FALSE" << endl;
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
			int_vec_add(Poly4_27->Monomials + i * 27, 
				Poly2_27->Monomials + j * 27, 
				M, 27);
			idx = Poly6_27->index_of_monomial(M);
			if (idx >= nb_monomials6) {
				cout << "surface_domain::multiply42_and_add "
						"idx >= nb_monomials6" << endl;
				exit(1);
				}
			MM[idx] = F->add(MM[idx], d);
			}
		}
	
	if (f_v) {
		cout << "surface_domain::multiply42_and_add done" << endl;
		}
}

void surface_domain::prepare_system_from_FG(int *F_planes, int *G_planes,
	int lambda, int *&system, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;


	if (f_v) {
		cout << "surface_domain::prepare_system_from_FG" << endl;
		}
	system = NEW_int(3 * 4 * 3);
	int_vec_zero(system, 3 * 4 * 3);
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			int *p = system + (i * 4 + j) * 3;
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
		cout << "surface_domain::prepare_system_from_FG done" << endl;
		}
}


void surface_domain::compute_nine_lines(int *F_planes, int *G_planes,
	int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines" << endl;
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int_vec_copy(F_planes + i * 4, Basis, 4);
			int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_int_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		int_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines done" << endl;
		}
}

void surface_domain::compute_nine_lines_by_dual_point_ranks(
	int *F_planes_rank,
	int *G_planes_rank, int *nine_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int F_planes[12];
	int G_planes[12];
	int Basis[16];

	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_"
				"point_ranks" << endl;
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(F_planes + i * 4, F_planes_rank[i]);
		}
	for (i = 0; i < 3; i++) {
		P->unrank_point(G_planes + i * 4, G_planes_rank[i]);
		}
	
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int_vec_copy(F_planes + i * 4, Basis, 4);
			int_vec_copy(G_planes + j * 4, Basis + 4, 4);
			F->RREF_and_kernel(4, 2, Basis, 0 /* verbose_level */);
			nine_lines[i * 3 + j] = Gr->rank_int_here(
				Basis + 8, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "The nine lines are: ";
		int_vec_print(cout, nine_lines, 9);
		cout << endl;
		}
	if (f_v) {
		cout << "surface_domain::compute_nine_lines_by_dual_"
				"point_ranks done" << endl;
		}
}

void surface_domain::split_nice_equation(int *nice_equation,
	int *&f1, int *&f2, int *&f3, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::split_nice_equation" << endl;
		}
	int M[4];
	int i, a, idx;

	f1 = NEW_int(Poly1->nb_monomials);
	f2 = NEW_int(Poly2->nb_monomials);
	f3 = NEW_int(Poly3->nb_monomials);
	int_vec_zero(f1, Poly1->nb_monomials);
	int_vec_zero(f2, Poly2->nb_monomials);
	int_vec_zero(f3, Poly3->nb_monomials);
	
	for (i = 0; i < 20; i++) {
		a = nice_equation[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly3_4->Monomials + i * 4, M, 4);
		if (M[0] == 3) {
			cout << "surface_domain::split_nice_equation the x_0^3 "
				"term is supposed to be zero" << endl;
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
		cout << "surface_domain::split_nice_equation done" << endl;
		}
}

void surface_domain::assemble_tangent_quadric(
	int *f1, int *f2, int *f3,
	int *&tangent_quadric, int verbose_level)
// 2*x_0*f_1 + f_2
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric" << endl;
		}
	int M[4];
	int i, a, idx, two;


	two = F->add(1, 1);
	tangent_quadric = NEW_int(Poly2_4->nb_monomials);
	int_vec_zero(tangent_quadric, Poly2_4->nb_monomials);
	
	for (i = 0; i < Poly1->nb_monomials; i++) {
		a = f1[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly1->Monomials + i * 3, M + 1, 3);
		M[0] = 1;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = F->mult(two, a);
		}

	for (i = 0; i < Poly2->nb_monomials; i++) {
		a = f2[i];
		if (a == 0) {
			continue;
			}
		int_vec_copy(Poly2->Monomials + i * 3, M + 1, 3);
		M[0] = 0;
		idx = Poly2_4->index_of_monomial(M);
		tangent_quadric[idx] = a;
		}
	if (f_v) {
		cout << "surface_domain::assemble_tangent_quadric done" << endl;
		}
}

void surface_domain::tritangent_plane_to_trihedral_pair_and_position(
	int tritangent_plane_idx, 
	int &trihedral_pair_idx, int &position, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	static int Table[] = {
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
		cout << "surface_domain::tritangent_plane_to_trihedral_"
				"pair_and_position" << endl;
		}
	trihedral_pair_idx = Table[2 * tritangent_plane_idx + 0];
	position = Table[2 * tritangent_plane_idx + 1];
	if (f_v) {
		cout << "surface_domain::tritangent_plane_to_trihedral_"
				"pair_and_position done" << endl;
		}
}

void surface_domain::do_arc_lifting_with_two_lines(
	int *Arc6, int p1_idx, int p2_idx, int partition_rk,
	int line1, int line2,
	int *coeff20, int *lines27,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int arc[6];
	int P1, P2;


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines" << endl;
		cout << "Arc6: ";
		int_vec_print(cout, Arc6, 6);
		cout << endl;
		cout << "p1_idx=" << p1_idx << " p2_idx=" << p2_idx
				<< " partition_rk=" << partition_rk
				<< " line1=" << line1 << " line2=" << line2 << endl;
	}

	P1 = Arc6[p1_idx];
	P2 = Arc6[p2_idx];

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"P->rearrange_arc_for_lifting" << endl;
		}
	P->rearrange_arc_for_lifting(Arc6,
				P1, P2, partition_rk, arc,
				verbose_level);

	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"P->rearrange_arc_for_lifting" << endl;
		cout << "arc: ";
		int_vec_print(cout, arc, 6);
		cout << endl;
		}

	arc_lifting_with_two_lines *AL;

	AL = NEW_OBJECT(arc_lifting_with_two_lines);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines before "
				"AL->create_surface" << endl;
		}
	AL->create_surface(this, arc, line1, line2, verbose_level);
	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines after "
				"AL->create_surface" << endl;
		cout << "equation: ";
		int_vec_print(cout, AL->coeff, 20);
		cout << endl;
		cout << "lines: ";
		int_vec_print(cout, AL->lines27, 27);
		cout << endl;
		}

	int_vec_copy(AL->coeff, coeff20, 20);
	int_vec_copy(AL->lines27, lines27, 27);


	if (f_v) {
		cout << "surface_domain::do_arc_lifting_with_two_lines done" << endl;
	}
}

}
}

