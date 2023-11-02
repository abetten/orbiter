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
namespace layer1_foundations {
namespace algebraic_geometry {


surface_domain::surface_domain()
{
	q = 0;
	n = 0;
	n2 = 0;

	F = NULL;
	P = NULL;
	P2 = NULL;
	Gr = NULL;
	Gr3 = NULL;
	nb_lines_PG_3 = 0;
	nb_pts_on_surface_with_27_lines = 0;

	O = NULL;
	Klein = NULL;

	//int Basis0[16];
	//int Basis1[16];
	//int Basis2[16];

	v = NULL;
	v2 = NULL;
	w2 = NULL;


	Schlaefli = NULL;

	PolynomialDomains = NULL;

}





surface_domain::~surface_domain()
{
	int f_v = false;

	if (f_v) {
		cout << "surface_domain::~surface_domain" << endl;
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

	if (Schlaefli) {
		FREE_OBJECT(Schlaefli);
	}

	if (PolynomialDomains) {
		FREE_OBJECT(PolynomialDomains);
	}

	if (f_v) {
		cout << "surface_domain::~surface_domain done" << endl;
	}
}

void surface_domain::init_surface_domain(
		field_theory::finite_field *F,
		int verbose_level)
// allocates projective_space objects for a PG(3,q) and PG(2,q)
// allocates grassmann objects for lines and hyperplanes
// allocates orthogonal and klein objects for the Klein correspondence
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::init_surface_domain" << endl;
	}
	
	n = 4;
	n2 = 2 * n;
	surface_domain::F = F;
	q = F->q;
	nb_pts_on_surface_with_27_lines = q * q + 7 * q + 1;
	if (f_v) {
		cout << "surface::init_surface_domain "
				"nb_pts_on_surface_with_27_lines = "
				<< nb_pts_on_surface_with_27_lines << endl;
	}

	v = NEW_int(n);
	v2 = NEW_int(6);
	w2 = NEW_int(6);
	
	P = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"before P->projective_space_init" << endl;
	}
	P->projective_space_init(3, F,
		true /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"after P->projective_space_init" << endl;
	}

	P2 = NEW_OBJECT(geometry::projective_space);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"before P2->projective_space_init" << endl;
	}
	P2->projective_space_init(2, F,
		true /*f_init_incidence_structure */, 
		verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"after P2->projective_space_init" << endl;
	}

	Gr = NEW_OBJECT(geometry::grassmann);
	Gr->init(n, 2, F, 0 /* verbose_level */);
	nb_lines_PG_3 = Gr->nCkq->as_lint();
	if (f_v) {
		cout << "surface::init_surface_domain nb_lines_PG_3 = "
				<< nb_lines_PG_3 << endl;
	}

	Gr3 = NEW_OBJECT(geometry::grassmann);
	Gr3->init(4, 3, F, 0 /* verbose_level*/);


	if (f_v) {
		cout << "surface::init_surface_domain "
				"initializing orthogonal" << endl;
	}
	O = NEW_OBJECT(orthogonal_geometry::orthogonal);
	O->init(1 /* epsilon */, 6 /* n */, F, verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"initializing orthogonal done" << endl;
	}

	Klein = NEW_OBJECT(geometry::klein_correspondence);

	if (f_v) {
		cout << "surface::init_surface_domain before Klein->init" << endl;
	}
	Klein->init(F, O, verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain after Klein->init" << endl;
	}




	PolynomialDomains = NEW_OBJECT(surface_polynomial_domains);

	if (f_v) {
		cout << "surface::init_surface_domain "
				"before PolynomialDomains->init" << endl;
	}
	PolynomialDomains->init(this, verbose_level);
	//init_polynomial_domains(verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain "
				"after PolynomialDomains->init" << endl;
	}

	if (f_v) {
		cout << "surface::init_surface_domain "
				"polynomial domains are:" << endl;
		PolynomialDomains->print_polynomial_domains_latex(cout);
		PolynomialDomains->Poly3_4->print_monomial_ordering_latex(cout);
	}





	if (f_v) {
		cout << "surface::init_surface_domain before init_Schlaefli" << endl;
	}
	init_Schlaefli(verbose_level - 2);
	if (f_v) {
		cout << "surface::init_surface_domain after init_Schlaefli" << endl;
	}



	//clebsch_cubics(verbose_level);

	if (f_v) {
		cout << "surface::init_surface_domain done" << endl;
	}
}


void surface_domain::unrank_point(
		int *v, long int rk)
{
	P->unrank_point(v, rk);
}

long int surface_domain::rank_point(
		int *v)
{
	long int rk;

	rk = P->rank_point(v);
	return rk;
}

void surface_domain::unrank_plane(
		int *v, long int rk)
{
	Gr3->unrank_lint_here(v, rk, 0 /* verbose_level */);
}

long int surface_domain::rank_plane(
		int *v)
{
	long int rk;

	rk = Gr3->rank_lint_here(v, 0 /* verbose_level */);
	return rk;
}

void surface_domain::enumerate_points(
		int *coeff,
		std::vector<long int> &Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::enumerate_points" << endl;
	}

	PolynomialDomains->Poly3_4->enumerate_points(coeff, Pts, verbose_level);
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
	PolynomialDomains->Poly3_4->substitute_semilinear(
			coeff_in, coeff_out,
		f_semilinear, frob, Mtx_inv, verbose_level);
	if (f_v) {
		cout << "surface_domain::substitute_semilinear done" << endl;
	}
}

void surface_domain::list_starter_configurations(
	long int *Lines, int nb_lines,
	data_structures::set_of_sets *line_intersections,
	int *&Table, int &N,
	int verbose_level)
// Goes over all lines of the surface which intersect at least 5 others.
// Then filters those 5-subsets which together
// with the original line give 19 linearly independent conditions
// uses line_intersections to find the lines which intersect at least 5 others.
// Table will have two columns.
// The first column is the index of the line,
// the second column is the lex-index of the 5-subset
// which creates 19 linearly independent conditions.
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "surface_domain::list_starter_configurations" << endl;
	}

	vector<vector<int>> V;

	{
		int subset[5];
		long int subset2[5];
		long int S6[6];
		int nCk, h;
		int i, j, r;
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
					S6[h] = Lines[subset2[h]];
				}
				S6[5] = Lines[i];

				r = rank_of_system(6, S6, 0 /*verbose_level*/);

				if (r == 19) {
					vector<int> v;

					v.push_back(i);
					v.push_back(j);
					V.push_back(v);
					N++;
				}
			}
		}
	}

	if (f_v) {
		cout << "surface_domain::list_starter_configurations We found "
			<< N << " starter configurations on this surface" 
			<< endl;
	}

	if (N != V.size()) {
		cout << "surface_domain::list_starter_configurations N != V.size()" << endl;
		exit(1);
	}

	Table = NEW_int(N * 2);
	int i;

	for (i = 0; i < V.size(); i++) {
		Table[i * 2 + 0] = V[i][0];
		Table[i * 2 + 1] = V[i][1];
	}
#if 0
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
				S6[h] = Lines[subset2[h]];
			}
			S6[5] = Lines[i];
			r = rank_of_system(6, S6, 0 /*verbose_level*/);
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
#endif
	if (f_v) {
		cout << "surface_domain::list_starter_configurations done" << endl;
	}
}

void surface_domain::create_starter_configuration(
	int line_idx, int subset_idx, 
	data_structures::set_of_sets *line_neighbors,
	long int *Lines, long int *S,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int subset[5];
	int subset2[5];
	int h; //, nCk;
	combinatorics::combinatorics_domain Combi;
	
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

void surface_domain::wedge_to_klein(
		int *W, int *K)
{
	geometry::geometry_global Geo;


	Geo.wedge_to_klein(F, W, K);
#if 0
	K[0] = W[0];
	K[1] = W[5];
	K[2] = W[1];
	K[3] = F->negate(W[4]);
	K[4] = W[2];
	K[5] = W[3];
#endif
}

void surface_domain::klein_to_wedge(
		int *K, int *W)
{
	geometry::geometry_global Geo;

	Geo.klein_to_wedge(F, K, W);
#if 0
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = F->negate(K[3]);
	W[5] = K[1];
#endif
}

long int surface_domain::line_to_wedge(
		long int line_rk)
{
	long int a, b;
	
	a = Klein->line_to_point_on_quadric(line_rk, 0 /* verbose_level*/);
	O->Hyperbolic_pair->unrank_point(w2, 1, a, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->Projective_space_basic->PG_element_rank_modified_lint(
			v2, 1, 6 /*wedge_dimension*/, b);
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
	int i;

	for (i = 0; i < len; i++) {
		Klein_rk[i] = Klein->line_to_point_on_quadric(Line_rk[i], 0 /* verbose_level*/);
	}
}

long int surface_domain::klein_to_wedge(
		long int klein_rk)
{
	long int b;
	
	O->Hyperbolic_pair->unrank_point(w2, 1, klein_rk, 0 /* verbose_level*/);
	klein_to_wedge(w2, v2);
	F->Projective_space_basic->PG_element_rank_modified_lint(
			v2, 1, 6 /*wedge_dimension*/, b);
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

void surface_domain::save_lines_in_three_kinds(
		std::string &fname_csv,
	long int *Lines_wedge,
	long int *Lines,
	long int *Lines_klein, int nb_lines)
{
	data_structures::spreadsheet *Sp;
	
	make_spreadsheet_of_lines_in_three_kinds(Sp, 
		Lines_wedge, Lines, Lines_klein, nb_lines,
		0 /* verbose_level */);

	Sp->save(fname_csv, 0 /*verbose_level*/);
	FREE_OBJECT(Sp);
}


int surface_domain::build_surface_from_double_six_and_count_Eckardt_points(
		long int *double_six,
		std::string &label_txt,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points" << endl;
	}

	int coeffs20[20];
	long int Lines27[27];
	int nb_E;


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"before Surf->build_cubic_surface_from_lines" << endl;
	}
	build_cubic_surface_from_lines(
		12, double_six,
		coeffs20, 0/* verbose_level*/);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"coeffs20:" << endl;
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;

		PolynomialDomains->Poly3_4->print_equation(cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"before Surf->create_the_fifteen_other_lines" << endl;
	}
	create_the_fifteen_other_lines(Lines27,
			Lines27 + 12, verbose_level);
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"after Surf->create_the_fifteen_other_lines" << endl;
	}



	surface_object *SO;

	SO = NEW_OBJECT(surface_object);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(this,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"after SO->init_with_27_lines" << endl;
	}

	nb_E = SO->SOP->nb_Eckardt_points;
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"the surface has " << nb_E << " Eckardt points" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"deleting SO" << endl;
	}
	FREE_OBJECT(SO);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six_and_count_Eckardt_points "
				"done" << endl;
	}
	return nb_E;

}

void surface_domain::build_surface_from_double_six(
		long int *double_six,
		std::string &label_txt,
		std::string &label_tex,
		algebraic_geometry::surface_object *&SO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six" << endl;
	}

	int coeffs20[20];
	long int Lines27[27];


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"double_six=";
		Lint_vec_print(cout, double_six, 12);
		cout << endl;
	}


	if (!test_double_six_property(double_six, 0 /* verbose_level*/)) {
		cout << "surface_domain::build_surface_from_double_six The double six is wrong" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"passes the double six property test" << endl;
	}


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"before Surf->build_cubic_surface_from_lines" << endl;
	}

	build_cubic_surface_from_lines(
		12, double_six,
		coeffs20, 0/* verbose_level*/);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"after Surf->build_cubic_surface_from_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"coeffs20:" << endl;
		Int_vec_print(cout, coeffs20, 20);
		cout << endl;

		PolynomialDomains->Poly3_4->print_equation(cout, coeffs20);
		cout << endl;
	}


	Lint_vec_copy(double_six, Lines27, 12);


	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"before Surf->create_the_fifteen_other_lines" << endl;
	}
	create_the_fifteen_other_lines(Lines27,
			Lines27 + 12, verbose_level);
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"after Surf->create_the_fifteen_other_lines" << endl;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);

#if 0
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"before SO->init_equation_points_and_lines_only" << endl;
	}

	SO->init_equation_points_and_lines_only(Surf, coeffs20, verbose_level);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"after SO->init_equation_points_and_lines_only" << endl;
	}
#else
	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"before SO->init_with_27_lines" << endl;
	}

	SO->init_with_27_lines(this,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six "
				"after SO->init_with_27_lines" << endl;
	}


#endif

	if (f_v) {
		cout << "surface_domain::build_surface_from_double_six done" << endl;
	}


}

int surface_domain::create_surface_by_equation(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		std::string &name_of_formula_tex,
		std::string &managed_variables,
		std::string &equation_text,
		std::string &equation_parameters,
		std::string &equation_parameters_tex,
		std::string &equation_parameter_values,
		std::vector<std::string> &select_double_six_string,
		algebraic_geometry::surface_object *&SO,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation" << endl;
		cout << "surface_domain::create_surface_by_equation "
				"name_of_formula=" << name_of_formula << endl;
		cout << "surface_domain::create_surface_by_equation "
				"name_of_formula_tex=" << name_of_formula_tex << endl;
		cout << "surface_domain::create_surface_by_equation "
				"managed_variables=" << managed_variables << endl;
		cout << "surface_domain::create_surface_by_equation "
				"equation_text=" << equation_text << endl;
		cout << "surface_domain::create_surface_by_equation "
				"equation_parameters=" << equation_parameters << endl;
		cout << "surface_domain::create_surface_by_equation "
				"equation_parameters_tex=" << equation_parameters_tex << endl;
		cout << "surface_domain::create_surface_by_equation "
				"equation_parameter_values=" << equation_parameter_values << endl;
	}


	string label_txt;
	string label_tex;

	label_txt = name_of_formula + "_" + equation_parameters;
	label_tex = name_of_formula_tex + "\\_" + equation_parameters_tex;

	// create a symbolic object containing the general formula:

	expression_parser::symbolic_object_builder_description *Descr1;


	Descr1 = NEW_OBJECT(expression_parser::symbolic_object_builder_description);
	Descr1->f_field_pointer = true;
	Descr1->field_pointer = F;
	Descr1->f_text = true;
	Descr1->text_txt = equation_text;




	expression_parser::symbolic_object_builder *SB1;

	SB1 = NEW_OBJECT(expression_parser::symbolic_object_builder);



	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before SB1->init" << endl;
	}

	string s1;

	s1 = name_of_formula + "_raw";

	SB1->init(Descr1, s1, verbose_level);

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after SB1->init" << endl;
	}



	// create a second symbolic object containing the specific values
	// to be substituted.

	expression_parser::symbolic_object_builder_description *Descr2;


	Descr2 = NEW_OBJECT(expression_parser::symbolic_object_builder_description);
	Descr2->f_field_pointer = true;
	Descr2->field_pointer = F;
	Descr2->f_text = true;
	Descr2->text_txt = equation_parameter_values;



	expression_parser::symbolic_object_builder *SB2;

	SB2 = NEW_OBJECT(expression_parser::symbolic_object_builder);

	string s2;

	s2 = name_of_formula + "_param_values";


	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before SB2->init" << endl;
	}

	SB2->init(Descr2, s2, verbose_level);

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after SB2->init" << endl;
	}


	// Perform the substitution.
	// Create temporary object Formula_vector_after_sub

	expression_parser::symbolic_object_builder *O_target = SB1;
	expression_parser::symbolic_object_builder *O_source = SB2;

	//O_target = Get_symbol(Descr->substitute_target);
	//O_source = Get_symbol(Descr->substitute_source);


	expression_parser::formula_vector *Formula_vector_after_sub;


	Formula_vector_after_sub = NEW_OBJECT(expression_parser::formula_vector);

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before Formula_vector_after_sub->substitute" << endl;
	}
	Formula_vector_after_sub->substitute(
			O_source->Formula_vector,
			O_target->Formula_vector,
			equation_parameters /*Descr->substitute_variables*/,
			name_of_formula, name_of_formula_tex,
			managed_variables,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after Formula_vector_after_sub->substitute" << endl;
	}


	// Perform simplification

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before Formula_vector_after_sub->V[0].simplify" << endl;
	}
	Formula_vector_after_sub->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after Formula_vector_after_sub->V[0].simplify" << endl;
	}

	// Perform expansion.
	// The result will be in the temporary object Formula_vector_after_expand


	expression_parser::formula_vector *Formula_vector_after_expand;

	Formula_vector_after_expand = NEW_OBJECT(expression_parser::formula_vector);

	int f_write_trees_during_expand = false;

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before Formula_vector->expand" << endl;
	}
	Formula_vector_after_expand->expand(
			Formula_vector_after_sub,
			F,
			name_of_formula, name_of_formula_tex,
			managed_variables,
			f_write_trees_during_expand,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after Formula_vector->expand" << endl;
	}

	// Perform simplification



	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before Formula_vector_after_expand->V[0].simplify" << endl;
	}
	Formula_vector_after_expand->V[0].simplify(verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after Formula_vector_after_expand->V[0].simplify" << endl;
	}



	// assemble the equation as a vector of coefficients
	// in the ordering of the polynomial ring:

	int *coeffs;
	int nb_coeffs;

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before Formula_vector_after_expand->V[0].collect_coefficients_of_equation" << endl;
	}
	Formula_vector_after_expand->V[0].collect_coefficients_of_equation(
			Poly,
			coeffs, nb_coeffs,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after Formula_vector_after_expand->V[0].collect_coefficients_of_equation" << endl;
	}

	if (nb_coeffs != 20) {
		cout << "surface_domain::create_surface_by_equation nb_coeffs != 20" << endl;
		exit(1);
	}
	// build a surface_object and compute properties of the surface:


	if (Int_vec_is_zero(coeffs, nb_coeffs)) {
		return false;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"before create_surface_by_coefficient_vector" << endl;
	}

	create_surface_by_coefficient_vector(coeffs,
			select_double_six_string,
			label_txt, label_tex,
			SO,
			verbose_level);


	if (f_v) {
		cout << "surface_domain::create_surface_by_equation "
				"after create_surface_by_coefficient_vector" << endl;
	}

	FREE_int(coeffs);

	if (f_v) {
		cout << "surface_domain::create_surface_by_equation done" << endl;
	}
	return true;
}



int surface_domain::create_surface_by_symbolic_object(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::string &name_of_formula,
		std::vector<std::string> &select_double_six_string,
		algebraic_geometry::surface_object *&SO,
		int verbose_level)
// returns false if the equation is zero
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object" << endl;
		cout << "surface_domain::create_surface_by_symbolic_object "
				"name_of_formula=" << name_of_formula << endl;
	}



	expression_parser::symbolic_object_builder *Symbol;

	Symbol = Get_symbol(name_of_formula);

	// assemble the equation as a vector of coefficients
	// in the ordering of the polynomial ring:

	int *coeffs;
	int nb_coeffs;

	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object "
				"before Symbol->Formula_vector->V[0].collect_coefficients_of_equation" << endl;
	}
	Symbol->Formula_vector->V[0].collect_coefficients_of_equation(
			Poly,
			coeffs, nb_coeffs,
			verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object "
				"after Symbol->Formula_vector->V[0].collect_coefficients_of_equation" << endl;
	}

	if (nb_coeffs != 20) {
		cout << "surface_domain::create_surface_by_symbolic_object nb_coeffs != 20" << endl;
		exit(1);
	}
	// build a surface_object and compute properties of the surface:


	if (Int_vec_is_zero(coeffs, nb_coeffs)) {
		return false;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);


	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object "
				"before create_surface_by_coefficient_vector" << endl;
	}

	create_surface_by_coefficient_vector(coeffs,
			select_double_six_string,
			name_of_formula, name_of_formula,
			SO,
			verbose_level);


	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object "
				"after create_surface_by_coefficient_vector" << endl;
	}

	FREE_int(coeffs);

	if (f_v) {
		cout << "surface_domain::create_surface_by_symbolic_object done" << endl;
	}
	return true;
}





void surface_domain::create_surface_by_coefficient_vector(
		int *coeffs20,
		std::vector<std::string> &select_double_six_string,
		std::string &label_txt,
		std::string &label_tex,
		algebraic_geometry::surface_object *&SO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_by_coefficient_vector" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_by_coefficient_vector "
				"surface is given by the coefficients" << endl;
	}



	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_by_coefficient_vector "
				"before SO->init_equation" << endl;
	}
	SO->init_equation(this, coeffs20, label_txt, label_tex, verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_by_coefficient_vector "
				"after SO->init_equation" << endl;
	}



	int nb_select_double_six;

	nb_select_double_six = select_double_six_string.size();

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_domain::create_surface_by_coefficient_vector "
						"selecting double six " << i << " / "
						<< nb_select_double_six << endl;
			}

			data_structures::string_tools ST;

			ST.read_string_of_schlaefli_labels(select_double_six_string[i],
					select_double_six, sz, verbose_level);


			//Orbiter->Int_vec.scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_domain::create_surface_by_coefficient_vector "
						"f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_domain::create_surface_by_coefficient_vector "
						"select_double_six = ";
				Int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_domain::create_surface_by_coefficient_vector "
						"before "
						"rearrange_lines_according_to_a_given_double_six" << endl;
			}
			rearrange_lines_according_to_a_given_double_six(
					SO->Lines, select_double_six, New_lines, 0 /* verbose_level */);

			Lint_vec_copy(New_lines, SO->Lines, 27);
			FREE_int(select_double_six);


		}


		if (f_v) {
			cout << "surface_domain::create_surface_by_coefficient_vector "
					"before compute_properties" << endl;
		}
		SO->compute_properties(verbose_level - 2);
		if (f_v) {
			cout << "surface_domain::create_surface_by_coefficient_vector "
					"after compute_properties" << endl;
		}


	}




	if (f_v) {
		cout << "surface_domain::create_surface_by_coefficient_vector done" << endl;
	}

}

void surface_domain::create_surface_from_catalogue(
		int iso,
		std::vector<std::string> &select_double_six_string,
		algebraic_geometry::surface_object *&SO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue" << endl;
	}
	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"iso = " << iso << endl;
	}

	int nb_select_double_six;

	nb_select_double_six = select_double_six_string.size();
	long int *p_lines;
	long int Lines27[27];
	int nb_iso;
	//int nb_E = 0;
	knowledge_base::knowledge_base K;

	nb_iso = K.cubic_surface_nb_reps(q);
	if (iso >= nb_iso) {
		cout << "surface_domain::create_surface_from_catalogue "
				"iso >= nb_iso, "
				"this cubic surface does not exist" << endl;
		exit(1);
	}
	p_lines = K.cubic_surface_Lines(q, iso);
	Lint_vec_copy(p_lines, Lines27, 27);
	//nb_E = cubic_surface_nb_Eckardt_points(q, Descr->iso);

	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"before rearrange_lines_according_to_double_six" << endl;
	}
	rearrange_lines_according_to_double_six(
			Lines27, 0 /* verbose_level */);
	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"after rearrange_lines_according_to_double_six" << endl;
	}

	if (nb_select_double_six) {
		int i;

		for (i = 0; i < nb_select_double_six; i++) {
			int *select_double_six;
			int sz;
			long int New_lines[27];

			if (f_v) {
				cout << "surface_domain::create_surface_from_catalogue "
						"selecting double six " << i << " / " << nb_select_double_six << endl;
			}
			Int_vec_scan(select_double_six_string[i], select_double_six, sz);
			if (sz != 12) {
				cout << "surface_domain::create_surface_from_catalogue "
						"f_select_double_six double six must consist of 12 numbers" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "surface_domain::create_surface_from_catalogue "
						"select_double_six = ";
				Int_vec_print(cout, select_double_six, 12);
				cout << endl;
			}


			if (f_v) {
				cout << "surface_domain::create_surface_from_catalogue "
						"before rearrange_lines_according_to_a_given_double_six" << endl;
			}
			rearrange_lines_according_to_a_given_double_six(
					Lines27, select_double_six, New_lines,
					0 /* verbose_level */);

			Lint_vec_copy(New_lines, Lines27, 27);
			FREE_int(select_double_six);
		}
	}

	int coeffs20[20];

	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"before build_cubic_surface_from_lines" << endl;
	}
	build_cubic_surface_from_lines(
			27, Lines27, coeffs20,
			0 /* verbose_level */);
	if (f_v) {
		cout << "surface_create::create_surface_from_catalogue "
				"after build_cubic_surface_from_lines" << endl;
	}

	string label_txt;
	string label_tex;

	label_txt = "catalogue_q" + std::to_string(q) + "_iso" + std::to_string(iso);
	label_tex = "catalogue_q" + std::to_string(q) + "_iso" + std::to_string(iso);

	SO = NEW_OBJECT(algebraic_geometry::surface_object);

	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"before SO->init_with_27_lines" << endl;
	}
	SO->init_with_27_lines(this,
		Lines27, coeffs20,
		label_txt, label_tex,
		false /* f_find_double_six_and_rearrange_lines */,
		verbose_level);
	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue "
				"after SO->init_with_27_lines" << endl;
	}

	if (f_v) {
		cout << "surface_domain::create_surface_from_catalogue done" << endl;
	}
}

std::string surface_domain::stringify_eqn_maple(int *eqn)
{
	stringstream sstr;
	string str;
	print_equation_maple(sstr, eqn);
	str.assign(sstr.str());
	return str;
}





}}}

