// surface_domain.cpp
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
		FREE_lint(Sets);
		}
	if (M) {
		FREE_int(M);
		}
	if (Sets2) {
		FREE_lint(Sets2);
		}

	if (Pts) {
		FREE_int(Pts);
		}
	if (pt_list) {
		FREE_lint(pt_list);
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
		FREE_lint(Double_six);
		}
	if (Double_six_label_tex) {
		int i;
		
		for (i = 0; i < 36; i++) {
			FREE_char(Double_six_label_tex[i]);
			}
		FREE_pchar(Double_six_label_tex);
		}
	if (Half_double_sixes) {
		FREE_lint(Half_double_sixes);
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
	nb_lines_PG_3 = Gr->nCkq.as_lint();
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
	//print_half_double_sixes_in_GAP();

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

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1->init" << endl;
	}
	Poly1->init(F,
			3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2->init" << endl;
	}
	Poly2->init(F,
			3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3->init" << endl;
	}
	Poly3->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3->init" << endl;
	}

	Poly1_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly2_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly3_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	Poly4_x123 = NEW_OBJECT(homogeneous_polynomial_domain);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1_x123->init" << endl;
	}
	Poly1_x123->init(F,
			3 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2_x123->init" << endl;
	}
	Poly2_x123->init(F,
			3 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3_x123->init" << endl;
	}
	Poly3_x123->init(F,
			3 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3_x123->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly4_x123->init" << endl;
	}
	Poly4_x123->init(F,
			3 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly4_x123->init" << endl;
	}


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
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly1_4->init" << endl;
	}
	Poly1_4->init(F,
			4 /* nb_vars */, 1 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly1_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly2_4->init" << endl;
	}
	Poly2_4->init(F,
			4 /* nb_vars */, 2 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly2_4->init" << endl;
	}
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before Poly3_4->init" << endl;
	}
	Poly3_4->init(F,
			4 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after Poly3_4->init" << endl;
	}

	if (f_v) {
		cout << "surface_domain::init_polynomial_domains before label_variables_4" << endl;
	}
	label_variables_4(Poly1_4, 0 /* verbose_level */);
	label_variables_4(Poly2_4, 0 /* verbose_level */);
	label_variables_4(Poly3_4, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::init_polynomial_domains after label_variables_4" << endl;
	}

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
		snprintf(label, 1000, "y_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		snprintf(label, 1000, "y_{%d}", i);
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
		snprintf(label, 1000, "x_%d", i + 1);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		snprintf(label, 1000, "x_{%d}", i + 1);
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
		snprintf(label, 1000, "X_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 4; i++) {
		snprintf(label, 1000, "X_{%d}", i);
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
		snprintf(label, 1000, "y_%d", i);
		l = strlen(label);
		HPD->symbols[i] = NEW_char(l + 1);
		strcpy(HPD->symbols[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label, 1000, "f_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label, 1000, "g_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[3 + 12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[3 + 12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		snprintf(label, 1000, "y_{%d}", i);
		l = strlen(label);
		HPD->symbols_latex[i] = NEW_char(l + 1);
		strcpy(HPD->symbols_latex[i], label);
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label, 1000, "f_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[3 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[3 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label, 1000, "g_{%d%d}", i, j);
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
			snprintf(label, 1000, "f_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label, 1000, "g_%d%d", i, j);
			l = strlen(label);
			HPD->symbols[12 + i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols[12 + i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label,  1000, "f_{%d%d}", i, j);
			l = strlen(label);
			HPD->symbols_latex[i * 4 + j] = NEW_char(l + 1);
			strcpy(HPD->symbols_latex[i * 4 + j], label);
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 4; j++) {
			snprintf(label,  1000, "g_{%d%d}", i, j);
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
	pt_list = NEW_lint(max_pts);
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

void surface_domain::unrank_plane(int *v, long int rk)
{
	Gr3->unrank_lint_here(v, rk, 0 /* verbose_level */);
}

long int surface_domain::rank_plane(int *v)
{
	long int rk;

	rk = Gr3->rank_lint_here(v, 0 /* verbose_level */);
	return rk;
}

int surface_domain::test(int len, long int *S, int verbose_level)
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
	long int *Pts, int &nb_pts, int verbose_level)
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

void surface_domain::create_special_double_six(long int *double_six,
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
		double_six[i] = Gr->rank_lint_here(Basis + i * 8,
				0 /* verbose_level */);
		}
	if (f_v) {
		cout << "surface_domain::create_special_double_six done" << endl;
		}
}

void surface_domain::create_special_fifteen_lines(long int *fifteen_lines,
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
		fifteen_lines[i] = Gr->rank_lint_here(
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
	long int *Lines27,
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
	long int *Pts;
	int v[4];
	geometry_global Gg;
	sorting Sorting;

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


	Pts = NEW_lint(Gg.nb_PG_elements(3, F->q));
	
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


	long int *Oab;

	Oab = NEW_lint(12);
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


	lint_vec_copy(Oab, Lines27, 12);
	FREE_lint(Oab);


	nb = 12;

	if (f_v) {
		cout << "surface_domain::create_surface_ab We have a set of "
				"lines of size " << nb << ":";
		lint_vec_print(cout, Lines27, nb);
		cout << endl;
		}

	create_remaining_fifteen_lines(Lines27,
		Lines27 + 12, 0 /* verbose_level */);

	if (f_v) {
		cout << "surface_domain::create_surface_ab The remaining 15 lines are:";
		lint_vec_print(cout, Lines27 + 12, 15);
		cout << endl;
		Gr->print_set(Lines27 + 12, 15);
		}


	if (f_v) {
		cout << "surface_domain::create_surface_ab before create_special_"
				"fifteen_lines" << endl;
		}

	long int special_lines[15];

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
		FREE_lint(Pts);
		//FREE_int(coeff);
		exit(1);
		}
	build_cubic_surface_from_lines(27, Lines27, coeff20,
			0 /* verbose_level */);
	F->PG_element_normalize_from_front(coeff20, 1, 20);



	enumerate_points(coeff20, Pts, nb_pts, 0 /* verbose_level */);
	Sorting.lint_vec_heapsort(Pts, nb_pts);


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
		lint_vec_print_as_table(cout, Pts, nb_pts, 10);
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
	long int Lines2[27];
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
			lint_vec_print(cout, lines_on_pt->Sets[e],
				lines_on_pt->Set_size[e]);
			cout << endl;
			}
		}

	
	FREE_int(E);
	//FREE_int(coeff);
	FREE_lint(Pts);
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
	long int *Lines, int nb_lines,
	set_of_sets *line_intersections, int *&Table, int &N, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	long int subset2[5];
	long int S3[6];
	int N1, nCk, h;
	int i, j, r;
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_domain::list_starter_configurations" << endl;
		}

	N = 0;
	for (i = 0; i < nb_lines; i++) {
		if (line_intersections->Set_size[i] < 5) {
			continue;
			}
		nCk = Combi.int_n_choose_k(line_intersections->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			Combi.unrank_k_subset(j, subset,
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
		nCk = Combi.int_n_choose_k(line_intersections->Set_size[i], 5);
		for (j = 0; j < nCk; j++) {
			Combi.unrank_k_subset(j, subset,
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
	set_of_sets *line_neighbors, long int *Lines, long int *S,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	int subset2[5];
	int h; //, nCk;
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_domain::create_starter_configuration" << endl;
		}
	//nCk = int_n_choose_k(line_neighbors->Set_size[line_idx], 5);
	Combi.unrank_k_subset(subset_idx, subset,
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

long int surface_domain::line_to_wedge(long int line_rk)
{
	long int a, b;
	
	a = Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);
	O->unrank_point(w2, 1, a, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified_lint(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::line_to_wedge_vec(
		long int *Line_rk, long int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = line_to_wedge(Line_rk[i]);
		}
}

void surface_domain::line_to_klein_vec(
		long int *Line_rk, long int *Klein_rk, int len)
{
	//int_vec_apply(Line_rk, Klein->Line_to_point_on_quadric,
	//		Klein_rk, len);
	//from through to
	//for (i = 0; i < len; i++) {
	//	to[i] = through[from[i]];
	//	}
	int i;

	for (i = 0; i < len; i++) {
		Klein_rk[i] = Klein->line_to_point_on_quadric(Line_rk[i], 0 /* verbose_level*/);
		}
}

long int surface_domain::klein_to_wedge(long int klein_rk)
{
	long int b;
	
	O->unrank_point(w2, 1, klein_rk, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->PG_element_rank_modified_lint(v2, 1, 6 /*wedge_dimension*/, b);
	//b = AW->rank_point(v);
	return b;
}

void surface_domain::klein_to_wedge_vec(
		long int *Klein_rk, long int *Wedge_rk, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		Wedge_rk[i] = klein_to_wedge(Klein_rk[i]);
		}
}

void surface_domain::save_lines_in_three_kinds(const char *fname_csv,
	long int *Lines_wedge, long int *Lines, long int *Lines_klein, int nb_lines)
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
	sorting Sorting;

	if (f_v) {
		cout << "surface_domain::find_tritangent_planes_"
				"intersecting_in_a_line" << endl;
		}
	for (plane1 = 0; plane1 < nb_Eckardt_points; plane1++) {

		Eckardt_points[plane1].three_lines(this, three_lines);
		if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
			for (plane2 = plane1 + 1;
					plane2 < nb_Eckardt_points;
					plane2++) {

				Eckardt_points[plane2].three_lines(this, three_lines);
				if (Sorting.int_vec_search_linear(three_lines, 3, line_idx, idx)) {
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
	int subset_complement[6];
	int size_complement;
	char label[1000];
	combinatorics_domain Combi;
	latex_interface L;

	if (f_v) {
		cout << "surface_domain::make_trihedral_pairs" << endl;
		}
	nb_trihedral_pairs = 120;
	T = NEW_int(nb_trihedral_pairs * 9);
	T_label = NEW_pchar(nb_trihedral_pairs);

	idx = 0;

	// the first type (20 of them, 6 choose 3):
	for (h = 0; h < 20; h++, idx++) {
		Combi.unrank_k_subset(h, subset, 6, 3);
		Combi.set_complement(subset, 3, complement,
			size_complement, 6);
		snprintf(label, 1000, "%d%d%d;%d%d%d",
				subset[0] + 1, subset[1] + 1, subset[2] + 1,
				complement[0] + 1, complement[1] + 1, complement[2] + 1);

		make_Tijk(T + idx * 9, subset[0], subset[1], subset[2]);
		T_label[idx] = NEW_char(strlen(label) + 1);
		strcpy(T_label[idx], label);
		}

	// the second type (90 of them, (6 choose 2) times (4 choose 2)):
	for (h = 0; h < 15; h++) {
		Combi.unrank_k_subset(h, subset, 6, 4);
		Combi.set_complement(subset, 4, subset_complement,
			size_complement, 6);
		for (s = 0; s < 6; s++, idx++) {
			Combi.unrank_k_subset(s, second_subset, 4, 2);
			Combi.set_complement(second_subset, 2, complement,
				size_complement, 4);
			make_Tlmnp(T + idx * 9, 
				subset[second_subset[0]], 
				subset[second_subset[1]], 
				subset[complement[0]], 
				subset[complement[1]]);
			snprintf(label, 1000, "%d%d;%d%d;%d%d",
				subset[second_subset[0]] + 1, 
				subset[second_subset[1]] + 1, 
				subset[complement[0]] + 1, 
				subset[complement[1]] + 1,
				subset_complement[0] + 1,
				subset_complement[1] + 1);
			T_label[idx] = NEW_char(strlen(label) + 1);
			strcpy(T_label[idx], label);
			}
		}

	// the third type (10 of them, (6 choose 3) divide by 2):
	for (h = 0; h < 10; h++, idx++) {
		Combi.unrank_k_subset(h, subset + 1, 5, 2);
		subset[0] = 0;
		subset[1]++;
		subset[2]++;
		Combi.set_complement(subset, 3, complement,
			size_complement, 6);
		make_Tdefght(T + idx * 9, 
			subset[0], subset[1], subset[2], 
			complement[0], complement[1], complement[2]);
		snprintf(label, 1000, "%d%d%d,%d%d%d",
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
		L.print_integer_matrix_with_standard_labels(cout,
			T, 120, 9, FALSE /* f_tex */);
		L.print_integer_matrix_with_standard_labels(cout,
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
	combinatorics_domain Combi;
	sorting Sorting;
	latex_interface L;

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
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, 27, 3);
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
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, 27, 3);
			//rk = Eckardt_point_from_tritangent_plane(subset);
			Trihedral_pairs_col_sets[i * 3 + j] = rk;
			}
		}

	if (f_v) {
		cout << "surface_domain::process_trihedral_pairs "
				"The trihedral pairs row sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
			Trihedral_pairs_row_sets, 120, 3, 
			FALSE /* f_tex */);
		//print_integer_matrix_with_standard_labels(cout,
		//Trihedral_pairs_row_sets, 120, 3, TRUE /* f_tex */);
		cout << "The trihedral pairs col sets:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
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
		L.print_integer_matrix_with_standard_labels(cout,
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
		L.print_integer_matrix_with_standard_labels(cout,
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
	combinatorics_domain Combi;
	sorting Sorting;

	subset[0] = l;
	subset[1] = m;
	subset[2] = n;
	subset[3] = p;
	Sorting.int_vec_heapsort(subset, 4);
	Combi.set_complement(subset, 4, complement, size_complement, 6);
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
	latex_interface L;

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
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + i] = rk;
			}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
				}
			rk = Eckardt_point_from_tritangent_plane(
				tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + 3 + j] = rk;
			}
		}
	if (f_v) {
		cout << "Trihedral_to_Eckardt:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
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
	sorting Sorting;

	Sorting.int_vec_heapsort(tritangent_plane, 3);
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
	combinatorics_domain Combi;
	sorting Sorting;
	latex_interface L;

	if (f_v) {
		cout << "surface_domain::init_collinear_Eckardt_triples" << endl;
		}
	nb_collinear_Eckardt_triples = nb_trihedral_pairs * 2;
	collinear_Eckardt_triples_rank = NEW_int(nb_collinear_Eckardt_triples);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 2; i++) {
			int_vec_copy(Trihedral_to_Eckardt + 6 * t + i * 3, 
				subset, 3);
			Sorting.int_vec_heapsort(subset, 3);
			rk = Combi.rank_k_subset(subset, nb_Eckardt_points, 3);
			collinear_Eckardt_triples_rank[t * 2 + i] = rk;
			}
		}
	if (f_v) {
		cout << "collinear_Eckardt_triples_rank:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
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
	combinatorics_domain Combi;
	sorting Sorting;
	
	if (f_v) {
		cout << "surface_domain::find_trihedral_pairs_from_collinear_"
				"triples_of_Eckardt_points" << endl;
		}
	nCk = Combi.int_n_choose_k(nb_E, 3);
	T_idx = NEW_int(nCk);
	nb_T = 0;
	for (h = 0; h < nCk; h++) {
		//cout << "subset " << h << " / " << nCk << ":";
		Combi.unrank_k_subset(h, subset, nb_E, 3);
		//int_vec_print(cout, subset, 3);
		//cout << " = ";

		for (k = 0; k < 3; k++) {
			set[k] = E_idx[subset[k]];
			}
		//int_vec_print(cout, set, 3);
		//cout << " = ";
		Sorting.int_vec_heapsort(set, 3);
		
		rk = Combi.rank_k_subset(set, nb_Eckardt_points, 3);


		//int_vec_print(cout, set, 3);
		//cout << " rk=" << rk << endl;

		if (Sorting.int_vec_search(
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


}
}

