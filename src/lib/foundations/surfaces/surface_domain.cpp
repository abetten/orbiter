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
	incidence_lines_vs_tritangent_planes = NULL;
	Lines_in_tritangent_planes = NULL;
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
		delete [] Line_label;
#if 0
		int i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label[i]);
		}
		FREE_pchar(Line_label);
#endif
	}
	if (Line_label_tex) {
		delete [] Line_label_tex;
#if 0
		int i;
		
		for (i = 0; i < 27; i++) {
			FREE_char(Line_label_tex[i]);
		}
		FREE_pchar(Line_label_tex);
#endif
	}
	if (Eckard_point_label) {
		delete [] Eckard_point_label;
#if 0
		int i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label[i]);
		}
		FREE_pchar(Eckard_point_label);
#endif
	}
	if (Eckard_point_label_tex) {
		delete [] Eckard_point_label_tex;
#if 0
		int i;
		
		for (i = 0; i < 45; i++) {
			FREE_char(Eckard_point_label_tex[i]);
		}
		FREE_pchar(Eckard_point_label_tex);
#endif
	}
	if (f_v) {
		cout << "before FREE_int(Trihedral_pairs);" << endl;
	}
	if (Trihedral_pairs) {
		FREE_int(Trihedral_pairs);
	}
	if (Trihedral_pair_labels) {
		delete [] Trihedral_pair_labels;
#if 0
		int i;
		
		for (i = 0; i < nb_trihedral_pairs; i++) {
			FREE_char(Trihedral_pair_labels[i]);
		}
		FREE_pchar(Trihedral_pair_labels);
#endif
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
		FREE_lint(Trihedral_to_Eckardt);
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
		delete [] Double_six_label_tex;
#if 0
		int i;
		
		for (i = 0; i < 36; i++) {
			FREE_char(Double_six_label_tex[i]);
		}
		FREE_pchar(Double_six_label_tex);
#endif
	}
	if (Half_double_sixes) {
		FREE_lint(Half_double_sixes);
	}

	if (Half_double_six_label_tex) {
		delete [] Half_double_six_label_tex;
#if 0
		int i;
		
		for (i = 0; i < 72; i++) {
			FREE_char(Half_double_six_label_tex[i]);
		}
		FREE_pchar(Half_double_six_label_tex);
#endif
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
	if (incidence_lines_vs_tritangent_planes) {
		FREE_int(incidence_lines_vs_tritangent_planes);
	}
	if (Lines_in_tritangent_planes) {
		FREE_lint(Lines_in_tritangent_planes);
	}
	null();
	if (f_v) {
		cout << "surface_domain::freeself done" << endl;
	}
}

void surface_domain::null()
{
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
				"init_Schlaefli_labels" << endl;
	}
	init_Schlaefli_labels(verbose_level);
	if (f_v) {
		cout << "surface::init after "
				"init_Schlaefli_labels" << endl;
	}

	if (f_v) {
		cout << "surface::init before "
				"make_trihedral_pairs" << endl;
	}
	make_trihedral_pairs(verbose_level);
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
		cout << "surface::init before init_adjacency_matrix_of_lines" << endl;
	}
	init_adjacency_matrix_of_lines(verbose_level);
	if (f_v) {
		cout << "surface::init after init_adjacency_matrix_of_lines" << endl;
	}

	if (f_v) {
		cout << "surface::init before init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
	}
	init_incidence_matrix_of_lines_vs_tritangent_planes(verbose_level);
	if (f_v) {
		cout << "surface::init after init_incidence_matrix_of_lines_vs_tritangent_planes" << endl;
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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
			t_PART,
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

	nb_monomials = Poly3_4->get_nb_monomials();

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
			t_PART,
			verbose_level);
	Poly4_27->init(F,
			27 /* nb_vars */, 4 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	Poly6_27->init(F,
			27 /* nb_vars */, 6 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);
	Poly3_24->init(F,
			24 /* nb_vars */, 3 /* degree */,
			FALSE /* f_init_incidence_structure */,
			t_PART,
			verbose_level);

	nb_monomials2 = Poly2_27->get_nb_monomials();
	nb_monomials4 = Poly4_27->get_nb_monomials();
	nb_monomials6 = Poly6_27->get_nb_monomials();
	nb_monomials3 = Poly3_24->get_nb_monomials();

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

	if (f_v) {
		cout << "surface_domain::label_variables_3" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_domain::label_variables_3 HPD->nb_variables != 3" << endl;
		exit(1);
	}

	HPD->remake_symbols(0 /* symbol_offset */,
			"y_%d", "y_{%d}", verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_3 done" << endl;
	}
}

void surface_domain::label_variables_x123(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_x123" << endl;
	}
	if (HPD->nb_variables != 3) {
		cout << "surface_domain::label_variables_x123 "
				"HPD->nb_variables != 3" << endl;
		exit(1);
	}


	HPD->remake_symbols(0 /* symbol_offset */,
			"x_%d", "x_{%d}", verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_x123 done" << endl;
	}
}

void surface_domain::label_variables_4(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, l;
	//char label[1000];
	
	if (f_v) {
		cout << "surface_domain::label_variables_4" << endl;
		}
	if (HPD->nb_variables != 4) {
		cout << "surface_domain::label_variables_4 HPD->nb_variables != 4" << endl;
		exit(1);
		}


	HPD->remake_symbols(0 /* symbol_offset */,
			"X_%d", "X_{%d}", verbose_level);


	if (f_v) {
		cout << "surface::label_variables_4 done" << endl;
		}
	
}

void surface_domain::label_variables_27(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_27" << endl;
	}
	if (HPD->nb_variables != 27) {
		cout << "surface_domain::label_variables_27 HPD->n != 27" << endl;
		exit(1);
	}

	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 3,
			"y_%d", "y_{%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			3, 4,
			"f_0%d", "f_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			7, 4,
			"f_1%d", "f_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			11, 4,
			"f_2%d", "f_{2%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			15, 4,
			"g_0%d", "g_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			19, 4,
			"g_1%d", "g_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			23, 4,
			"g_2%d", "g_{2%d}",
			verbose_level);

	if (f_v) {
		cout << "surface_domain::label_variables_27 done" << endl;
	}
}

void surface_domain::label_variables_24(
	homogeneous_polynomial_domain *HPD,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "surface_domain::label_variables_24" << endl;
	}
	if (HPD->nb_variables != 24) {
		cout << "surface_domain::label_variables_24 HPD->n != 24" << endl;
		exit(1);
	}

	HPD->remake_symbols_interval(0 /* symbol_offset */,
			0, 4,
			"f_0%d", "f_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			4, 4,
			"f_1%d", "f_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			8, 4,
			"f_2%d", "f_{2%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			12, 4,
			"g_0%d", "g_{0%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			16, 4,
			"g_1%d", "g_{1%d}",
			verbose_level);
	HPD->remake_symbols_interval(0 /* symbol_offset */,
			20, 4,
			"g_2%d", "g_{2%d}",
			verbose_level);

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
	F->wedge_to_klein(W, K);
#if 0
	K[0] = W[0];
	K[1] = W[5];
	K[2] = W[1];
	K[3] = F->negate(W[4]);
	K[4] = W[2];
	K[5] = W[3];
#endif
}

void surface_domain::klein_to_wedge(int *K, int *W)
{
	F->klein_to_wedge(K, W);
#if 0
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
#endif
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

void surface_domain::save_lines_in_three_kinds(std::string &fname_csv,
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
		cout << "surface_domain::find_tritangent_planes_intersecting_in_a_line" << endl;
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
	cout << "surface_domain::find_tritangent_planes_intersecting_in_a_line could not find "
			"two planes" << endl;
	exit(1);
}


void surface_domain::make_trihedral_pairs(int verbose_level)
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
	Trihedral_pairs = NEW_int(nb_trihedral_pairs * 9);
	Trihedral_pair_labels = new std::string [nb_trihedral_pairs];

	idx = 0;

	// the first type (20 of them, 6 choose 3):
	for (h = 0; h < 20; h++, idx++) {
		Combi.unrank_k_subset(h, subset, 6, 3);
		Combi.set_complement(subset, 3, complement,
			size_complement, 6);
		snprintf(label, 1000, "%d%d%d;%d%d%d",
				subset[0] + 1, subset[1] + 1, subset[2] + 1,
				complement[0] + 1, complement[1] + 1, complement[2] + 1);

		make_Tijk(Trihedral_pairs + idx * 9, subset[0], subset[1], subset[2]);
		//T_label[idx] = NEW_char(strlen(label) + 1);
		//strcpy(T_label[idx], label);
		Trihedral_pair_labels[idx].assign(label);
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
			make_Tlmnp(Trihedral_pairs + idx * 9,
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
			//T_label[idx] = NEW_char(strlen(label) + 1);
			//strcpy(T_label[idx], label);
			Trihedral_pair_labels[idx].assign(label);
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
		make_Tdefght(Trihedral_pairs + idx * 9,
			subset[0], subset[1], subset[2], 
			complement[0], complement[1], complement[2]);
		snprintf(label, 1000, "%d%d%d,%d%d%d",
			subset[0] + 1, 
			subset[1] + 1, 
			subset[2] + 1, 
			complement[0] + 1, 
			complement[1] + 1, 
			complement[2] + 1);
		//T_label[idx] = NEW_char(strlen(label) + 1);
		//strcpy(T_label[idx], label);
		Trihedral_pair_labels[idx].assign(label);
	}

	if (idx != 120) {
		cout << "surface_domain::make_trihedral_pairs idx != 120" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "The trihedral pairs are:" << endl;
		L.print_integer_matrix_with_standard_labels(cout,
				Trihedral_pairs, 120, 9, FALSE /* f_tex */);
		L.print_integer_matrix_with_standard_labels(cout,
				Trihedral_pairs, 120, 9, TRUE /* f_tex */);
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

	Classify_trihedral_pairs_row_values = NEW_OBJECT(tally);
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

	Classify_trihedral_pairs_col_values = NEW_OBJECT(tally);
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
	int i;
	char str[1000];

	if (f_v) {
		cout << "surface_domain::make_Eckardt_points" << endl;
	}
	nb_Eckardt_points = 45;
	Eckardt_points = NEW_OBJECTS(eckardt_point, nb_Eckardt_points);
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].init_by_rank(i);
	}
	Eckard_point_label = new string [nb_Eckardt_points];
	Eckard_point_label_tex = new string [nb_Eckardt_points];
	for (i = 0; i < nb_Eckardt_points; i++) {
		Eckardt_points[i].latex_to_str_without_E(str);
		//l = strlen(str);
		//Eckard_point_label[i] = NEW_char(l + 1);
		//strcpy(Eckard_point_label[i], str);
		Eckard_point_label[i].assign(str);
		//Eckard_point_label_tex[i] = NEW_char(l + 1);
		//strcpy(Eckard_point_label_tex[i], str);
		Eckard_point_label_tex[i].assign(str);
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
	Trihedral_to_Eckardt = NEW_lint(nb_trihedral_to_Eckardt);
	for (t = 0; t < nb_trihedral_pairs; t++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				tritangent_plane[j] = Trihedral_pairs[t * 9 + i * 3 + j];
				}
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + i] = rk;
		}
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) {
				tritangent_plane[i] = 
					Trihedral_pairs[t * 9 + i * 3 + j];
			}
			rk = Eckardt_point_from_tritangent_plane(tritangent_plane);
			Trihedral_to_Eckardt[t * 6 + 3 + j] = rk;
		}
	}
	if (f_v) {
		cout << "Trihedral_to_Eckardt:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
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
			cout << "surface_domain::Eckardt_point_from_tritangent_plane a < 12" << endl;
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
	int t, i, rk, h;
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
			for (h = 0; h < 3; h++) {
				subset[h] = Trihedral_to_Eckardt[6 * t + i * 3 + h];
			}
			//int_vec_copy(Trihedral_to_Eckardt + 6 * t + i * 3, subset, 3);
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

	Classify_collinear_Eckardt_triples = NEW_OBJECT(tally);
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
	tally C;

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

