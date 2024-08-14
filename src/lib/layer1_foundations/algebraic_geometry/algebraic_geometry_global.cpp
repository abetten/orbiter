/*
 * algebraic_geometry_global.cpp
 *
 *  Created on: Dec 22, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


algebraic_geometry_global::algebraic_geometry_global()
{
}

algebraic_geometry_global::~algebraic_geometry_global()
{
}


void algebraic_geometry_global::analyze_del_Pezzo_surface(
		geometry::projective_space *P,
		expression_parser::formula *Formula,
		std::string &evaluate_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface" << endl;
	}

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface" << endl;
		cout << "formula:" << endl;
		Formula->print(cout);
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
#if 0
	if (Formula->degree != 4) {
		cout << "Formula is not of degree 4. "
				"Degree is " << Formula->degree << endl;
		exit(1);
	}
#endif
	if (Formula->nb_managed_vars != 3) {
		cout << "Formula should have 3 managed variables. "
				"Has " << Formula->nb_managed_vars << endl;
		exit(1);
	}

	ring_theory::homogeneous_polynomial_domain *Poly4_3;

	Poly4_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface "
				"before Poly->init" << endl;
	}
	Poly4_3->init(P->Subspaces->F,
			Formula->nb_managed_vars /* nb_vars */, 4 /*Formula->degree*/,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface "
				"after Poly->init" << endl;
	}


	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface "
				"before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly4_3, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface "
				"after Formula->get_subtrees" << endl;
	}

#if 0
	int i;

	for (i = 0; i < nb_monomials; i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			Subtrees[i]->print_expression(cout);
			cout << " * ";
			Poly4_3->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
		}
	}
#endif

	int *Coefficient_vector;

	Coefficient_vector = NEW_int(Poly4_3->get_nb_monomials());

	Formula->evaluate(Poly4_3,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface "
				"coefficient vector:" << endl;
		Int_vec_print(cout, Coefficient_vector, Poly4_3->get_nb_monomials());
		cout << endl;
	}

	algebraic_geometry::del_pezzo_surface_of_degree_two_domain *del_Pezzo;

	del_Pezzo = NEW_OBJECT(algebraic_geometry::del_pezzo_surface_of_degree_two_domain);

	del_Pezzo->init(P, Poly4_3, verbose_level);

	algebraic_geometry::del_pezzo_surface_of_degree_two_object *del_Pezzo_surface;

	del_Pezzo_surface = NEW_OBJECT(algebraic_geometry::del_pezzo_surface_of_degree_two_object);

	del_Pezzo_surface->init(del_Pezzo,
			Formula, Subtrees, Coefficient_vector,
			verbose_level);

	del_Pezzo_surface->enumerate_points_and_lines(verbose_level);

#if 0
	del_Pezzo_surface->pal->write_points_to_txt_file(
			Formula->name_of_formula, verbose_level);
#endif

	del_Pezzo_surface->create_latex_report(
			Formula->name_of_formula,
			Formula->name_of_formula_latex,
			verbose_level);

	FREE_OBJECT(del_Pezzo_surface);
	FREE_OBJECT(del_Pezzo);

	FREE_int(Coefficient_vector);
	FREE_OBJECT(Poly4_3);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface done" << endl;
	}
}

void algebraic_geometry_global::report_grassmannian(
		geometry::projective_space *P,
		int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::report_grassmannian" << endl;
	}
	string fname;
	string author;
	string title;
	string extra_praeamble;



	fname = "Gr_" + std::to_string(P->Subspaces->n + 1) + "_"
			+ std::to_string(k) + "_"
			+ std::to_string(P->Subspaces->F->q) + ".tex";
	title = "Cheat Sheet Gr($" + std::to_string(P->Subspaces->n + 1)
			+ "," + std::to_string(k) + ","
			+ std::to_string(P->Subspaces->F->q) + "$)";




	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;

		L.head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);


		if (f_v) {
			cout << "algebraic_geometry_global::report_grassmannian "
					"before P->create_latex_report_for_Grassmannian" << endl;
		}
		P->Reporting->report_subspaces_of_dimension(ost, k, verbose_level);
		if (f_v) {
			cout << "algebraic_geometry_global::report_grassmannian "
					"after P->create_latex_report_for_Grassmannian" << endl;
		}


		L.foot(ost);

	}
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "algebraic_geometry_global::report_grassmannian "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "algebraic_geometry_global::report_grassmannian done" << endl;
	}

}

void algebraic_geometry_global::map(
		geometry::projective_space *P,
		std::string &ring_label,
		std::string &formula_label,
		std::string &evaluate_text,
		long int *&Image_pts,
		long int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::map" << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::map "
				"n = " << P->Subspaces->n << endl;
	}



	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_ring(ring_label);


	expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(formula_label);


	if (Ring->get_nb_variables() != P->Subspaces->n + 1) {
		cout << "algebraic_geometry_global::map "
				"number of variables is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::map "
				"before evaluate_regular_map" << endl;
	}
	evaluate_regular_map(
			Ring,
			P,
			Object,
			evaluate_text,
			Image_pts, N_points,
			verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::map "
				"after evaluate_regular_map" << endl;
	}






	if (f_v) {
		cout << "algebraic_geometry_global::map done" << endl;
	}
}

void algebraic_geometry_global::affine_map(
		geometry::projective_space *P,
		std::string &ring_label,
		std::string &formula_label,
		std::string &evaluate_text,
		long int *&Image_pts,
		long int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::affine_map" << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::affine_map "
				"n = " << P->Subspaces->n << endl;
	}



	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_ring(ring_label);


	expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(formula_label);


	if (Ring->get_nb_variables() != P->Subspaces->n + 1) {
		cout << "algebraic_geometry_global::affine_map "
				"number of variables is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::affine_map "
				"before evaluate_regular_map" << endl;
	}
	evaluate_affine_map(
			Ring,
			P,
			Object,
			evaluate_text,
			Image_pts, N_points,
			verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::affine_map "
				"after evaluate_regular_map" << endl;
	}






	if (f_v) {
		cout << "algebraic_geometry_global::affine_map done" << endl;
	}
}

void algebraic_geometry_global::projective_variety(
		geometry::projective_space *P,
		std::string &ring_label,
		std::string &formula_label,
		std::string &evaluate_text,
		long int *&Image_pts,
		long int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::projective_variety" << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::projective_variety "
				"n = " << P->Subspaces->n << endl;
	}



	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_ring(ring_label);


	expression_parser::symbolic_object_builder *Object;

	Object = Get_symbol(formula_label);


	if (Ring->get_nb_variables() != P->Subspaces->n + 1) {
		cout << "algebraic_geometry_global::variety "
				"number of variables is wrong" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::projective_variety "
				"before compute_projective_variety" << endl;
	}
	compute_projective_variety(
			Ring,
			P,
			Object,
			evaluate_text,
			Image_pts, N_points,
			verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::projective_variety "
				"after compute_projective_variety" << endl;
	}






	if (f_v) {
		cout << "algebraic_geometry_global::projective_variety done" << endl;
	}
}

void algebraic_geometry_global::evaluate_regular_map(
		ring_theory::homogeneous_polynomial_domain *Ring,
		geometry::projective_space *P,
		expression_parser::symbolic_object_builder *Object,
		std::string &evaluate_text,
		long int *&Image_pts, long int &N_points_input,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_regular_map" << endl;
	}

	int *v;
	int *w;
	int *w2;
	int h;
	long int i, j;

	N_points_input = P->Subspaces->N_points;

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_regular_map "
				"N_points_input = " << N_points_input << endl;
	}

	int len;

	len = Object->Formula_vector->len;

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_regular_map "
				"len = " << len << endl;
	}


	Image_pts = NEW_lint(N_points_input);

	v = NEW_int(P->Subspaces->n + 1);
	w = NEW_int(len);
	w2 = NEW_int(len);


	data_structures::string_tools ST;
	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(
			symbol_table,
				evaluate_text, verbose_level - 1);

	for (i = 0; i < N_points_input; i++) {

		P->unrank_point(v, i);

		if (f_v) {
			cout << "algebraic_geometry_global::evaluate_regular_map "
					"point " << i << " / " << N_points_input << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << endl;
		}

		for (h = 0; h < P->Subspaces->n + 1; h++) {

			symbol_table[Ring->get_symbol(h)] = std::to_string(v[h]);

		}

		for (h = 0; h < len; h++) {

			w[h] = Object->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					0 /*verbose_level*/);

		}

		Int_vec_copy(w, w2, len);



		if (!Int_vec_is_zero(w, len)) {
			P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
					w, 1 /* stride */, len, j);
		}
		else {
			j = -1;
		}

		if (f_v) {
			cout << "algebraic_geometry_global::evaluate_regular_map "
					"point " << i << " / " << N_points_input << " is ";
			Int_vec_print(cout, v, P->Subspaces->n + 1);
			cout << " maps to ";
			Int_vec_print(cout, w2, len);
			cout << " image rank = " << j;
			cout << endl;
		}


		Image_pts[i] = j;
	}

	FREE_int(v);
	FREE_int(w);
	FREE_int(w2);

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_regular_map done" << endl;
	}
}


void algebraic_geometry_global::evaluate_affine_map(
		ring_theory::homogeneous_polynomial_domain *Ring,
		geometry::projective_space *P,
		expression_parser::symbolic_object_builder *Object,
		std::string &evaluate_text,
		long int *&Image_pts, long int &N_points_input,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_affine_map" << endl;
	}

	int h;
	long int i, j;
	int k;
	int *input;
	int *output;

	k = P->Subspaces->n;

	int len;

	len = Object->Formula_vector->len;

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_affine_map "
				"len = " << len << endl;
	}


	geometry::geometry_global Gg;

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_affine_map "
				"before Gg.nb_AG_elements" << endl;
	}
	N_points_input = Gg.nb_AG_elements(k, P->Subspaces->F->q);
	if (f_v) {
		cout << N_points_input << " elements in the domain" << endl;
	}

	Image_pts = NEW_lint(N_points_input);


	input = NEW_int(k);
	output = NEW_int(len);


	data_structures::string_tools ST;
	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(symbol_table,
				evaluate_text, verbose_level - 1);

	for (i = 0; i < N_points_input; i++) {

		//P->unrank_point(v, i);
		Gg.AG_element_unrank(P->Subspaces->F->q, input, 1, k, i);

		if (f_vv) {
			cout << "algebraic_geometry_global::evaluate_affine_map "
					"point " << i << " is ";
			Int_vec_print(cout, input, k);
			cout << endl;
		}

		for (h = 0; h < k; h++) {

			symbol_table[Ring->get_symbol(h)] = std::to_string(input[h]);

		}

		for (h = 0; h < len; h++) {

			output[h] = Object->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					verbose_level - 2);

		}

#if 0
		if (!Int_vec_is_zero(w, len)) {
			P->Subspaces->F->Projective_space_basic->PG_element_rank_modified_lint(
					w, 1 /* stride */, len, j);
		}
		else {
			j = -1;
		}
#endif
		j = Gg.AG_element_rank(P->Subspaces->F->q, output, 1, len);

		if (f_vv) {
			cout << "algebraic_geometry_global::evaluate_affine_map point " << i << " = ";
			Int_vec_print(cout, input, k);
			cout << " and maps to ";
			Int_vec_print(cout, output, len);
			cout << " = " << j << endl;
		}

		Image_pts[i] = j;
	}

	FREE_int(input);
	FREE_int(output);

	if (f_v) {
		cout << "algebraic_geometry_global::evaluate_affine_map done" << endl;
	}
}



void algebraic_geometry_global::compute_projective_variety(
		ring_theory::homogeneous_polynomial_domain *Ring,
		geometry::projective_space *P,
		expression_parser::symbolic_object_builder *Object,
		std::string &evaluate_text,
		long int *&Variety, long int &Variety_nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebraic_geometry_global::compute_projective_variety" << endl;
	}

	int h;
	long int i;
	int k;
	int input_len;
	int output_len;
	int *input;
	int *output;
	long int N_points_input;

	k = P->Subspaces->n;

	input_len = k + 1;
	output_len = Object->Formula_vector->len;

	if (f_v) {
		cout << "algebraic_geometry_global::compute_projective_variety "
				"output_len = " << output_len << endl;
	}


	geometry::geometry_global Gg;

	N_points_input = P->Subspaces->N_points;
	if (f_v) {
		cout << "algebraic_geometry_global::compute_projective_variety "
				"There are " << N_points_input
				<< " elements in the domain" << endl;
	}



	Variety = NEW_lint(N_points_input);
	Variety_nb_points = 0;


	input = NEW_int(input_len);
	output = NEW_int(output_len);


	data_structures::string_tools ST;
	std::map<std::string, std::string> symbol_table;

	ST.parse_value_pairs(symbol_table,
				evaluate_text, verbose_level - 1);


	for (h = 0; h < output_len; h++) {

		cout << "Formula " << h << " : ";
		Object->Formula_vector->V[h].tree->print_easy(cout);
		cout << endl;

	}

	for (i = 0; i < N_points_input; i++) {

		P->unrank_point(input, i);
		//Gg.AG_element_unrank(P->Subspaces->F->q, input, 1, k, i);

		if (f_vv) {
			cout << "algebraic_geometry_global::compute_projective_variety "
					"point " << i << " is ";
			Int_vec_print(cout, input, input_len);
			cout << endl;
		}

		for (h = 0; h < input_len; h++) {

			symbol_table[Ring->get_symbol(h)] = std::to_string(input[h]);

		}

		for (h = 0; h < output_len; h++) {

			output[h] = Object->Formula_vector->V[h].tree->evaluate(
					symbol_table,
					verbose_level - 2);

		}

		if (f_vv) {
			cout << "algebraic_geometry_global::compute_projective_variety "
					"point " << i << " = ";
			Int_vec_print(cout, input, input_len);
			cout << " evaluates to ";
			Int_vec_print(cout, output, output_len);
			cout << endl;
		}

		if (Int_vec_is_zero(output, output_len)) {
			Variety[Variety_nb_points++] = i;
			if (f_vv) {
				cout << "algebraic_geometry_global::compute_projective_variety "
						"point on the variety " << i << " = ";
				Int_vec_print(cout, input, input_len);
				cout << endl;
			}
		}


	}

	FREE_int(input);
	FREE_int(output);

	if (f_v) {
		cout << "algebraic_geometry_global::compute_projective_variety done" << endl;
	}
}




void algebraic_geometry_global::make_evaluation_matrix_wrt_ring(
		ring_theory::homogeneous_polynomial_domain *Ring,
		geometry::projective_space *P,
		int *&M, int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "algebraic_geometry_global::make_evaluation_matrix_wrt_ring" << endl;
	}

	int nb_vars;
	int c;
	long int i, j, k;
	int input_len;
	int *pt_vec;
	long int N_points_input;

	k = P->Subspaces->n;

	input_len = k + 1;


	//geometry::geometry_global Gg;

	N_points_input = P->Subspaces->N_points;
	if (f_v) {
		cout << "algebraic_geometry_global::make_evaluation_matrix_wrt_ring "
				"There are " << N_points_input
				<< " elements in the domain" << endl;
	}


	nb_vars = Ring->nb_variables;
	nb_rows = Ring->get_nb_monomials();
	nb_cols = N_points_input;
	M = NEW_int(nb_rows * nb_cols);
	Int_vec_zero(M, nb_rows * nb_cols);


	pt_vec = NEW_int(input_len);


	for (j = 0; j < nb_cols; j++) {

		P->unrank_point(pt_vec, j);
		//Gg.AG_element_unrank(P->Subspaces->F->q, input, 1, k, i);

		if (f_vv) {
			cout << "algebraic_geometry_global::make_evaluation_matrix_wrt_ring "
					"point " << j << " is ";
			Int_vec_print(cout, pt_vec, input_len);
			cout << endl;
		}

		for (i = 0; i < nb_rows; i++) {

			c = Ring->get_F()->Linear_algebra->evaluate_monomial(
					Ring->get_monomial_pointer(i),
					pt_vec,
					nb_vars);

			M[i * nb_cols + j] = c;
		}


	}

	FREE_int(pt_vec);

	if (f_v) {
		cout << "algebraic_geometry_global::make_evaluation_matrix_wrt_ring done" << endl;
	}
}



void algebraic_geometry_global::cubic_surface_family_24_generators(
		field_theory::finite_field *F,
	int f_with_normalizer,
	int f_semilinear,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m_one;

	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_24_generators" << endl;
	}
	m_one = F->minus_one();
	nb_gens = 3;
	data_size = 16;
	if (f_semilinear) {
		data_size++;
	}
	if (EVEN(F->q)) {
		group_order = 6;
	}
	else {
		group_order = 24;
	}
	if (f_with_normalizer) {
		nb_gens++;
		group_order *= F->q - 1;
	}
	gens = NEW_int(nb_gens * data_size);
	Int_vec_zero(gens, nb_gens * data_size);
		// this sets the field automorphism index
		// to zero if we are semilinear

	gens[0 * data_size + 0 * 4 + 0] = 1;
	gens[0 * data_size + 1 * 4 + 2] = 1;
	gens[0 * data_size + 2 * 4 + 1] = 1;
	gens[0 * data_size + 3 * 4 + 3] = 1;
	gens[1 * data_size + 0 * 4 + 1] = 1;
	gens[1 * data_size + 1 * 4 + 0] = 1;
	gens[1 * data_size + 2 * 4 + 2] = 1;
	gens[1 * data_size + 3 * 4 + 3] = 1;
	gens[2 * data_size + 0 * 4 + 0] = m_one;
	gens[2 * data_size + 1 * 4 + 2] = 1;
	gens[2 * data_size + 2 * 4 + 1] = 1;
	gens[2 * data_size + 3 * 4 + 3] = m_one;
	if (f_with_normalizer) {
		gens[3 * data_size + 0 * 4 + 0] = 1;
		gens[3 * data_size + 1 * 4 + 1] = 1;
		gens[3 * data_size + 2 * 4 + 2] = 1;
		gens[3 * data_size + 3 * 4 + 3] = F->primitive_root();
	}
	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_24_generators "
				"done" << endl;
	}
}

void algebraic_geometry_global::cubic_surface_family_G13_generators(
		field_theory::finite_field *F,
	int a,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int data[] = {
			// A1:
			1,0,0,0,
			0,1,0,0,
			3,2,1,0,
			3,0,0,1,

			// A2:
			1,0,0,0,
			0,1,0,0,
			1,1,1,0,
			1,0,0,1,

			// A3:
			0,1,0,0,
			1,0,0,0,
			0,0,1,0,
			1,1,1,1,

			// A4:
			0,1,0,0,
			1,0,0,0,
			1,1,1,0,
			7,6,1,1,

			// A5:
			2,3,0,0,
			3,2,0,0,
			0,0,1,0,
			3,3,5,1,

			// A6:
			2,2,1,0,
			3,3,1,0,
			1,0,1,0,
			1,4,2,1,

	};

	data_size = 16 + 1;
	nb_gens = 6;
	group_order = 192;

	gens = NEW_int(nb_gens * data_size);
	Int_vec_zero(gens, nb_gens * data_size);

	int h, i, j, c, m, l;
	int *v;
	geometry::geometry_global Gg;
	number_theory::number_theory_domain NT;

	m = Int_vec_maximum(data, nb_gens * data_size);
	l = NT.int_log2(m) + 1;

	v = NEW_int(l);


	for (h = 0; h < nb_gens; h++) {
		for (i = 0; i < 16; i++) {
			Int_vec_zero(v, l);
			Gg.AG_element_unrank(
					F->p, v, 1, l, data[h * 16 + i]);
			c = 0;
			for (j = 0; j < l; j++) {
				c = F->mult(c, a);
				if (v[l - 1 - j]) {
					c = F->add(c, v[l - 1 - j]);
				}
			}
			gens[h * data_size + i] = c;
		}
		gens[h * data_size + 16] = 0;
	}
	FREE_int(v);

	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_G13_generators" << endl;
		for (h = 0; h < nb_gens; h++) {
			cout << "generator " << h << ":" << endl;
			Int_matrix_print(gens + h * data_size, 4, 4);
		}
	}
	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_G13_generators done" << endl;
	}
}

void algebraic_geometry_global::cubic_surface_family_F13_generators(
		field_theory::finite_field *F,
	int a,
	int *&gens, int &nb_gens, int &data_size,
	int &group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	// 2 = a
	// 3 = a+1
	// 4 = a^2
	// 5 = a^2+1
	// 6 = a^2 + a
	// 7 = a^2 + a + 1
	// 8 = a^3
	// 9 = a^3 + 1
	// 10 = a^3 + a
	// 11 = a^3 + a + 1
	// 12 = a^3 + a^2
	// 13 = a^3 + a^2 + 1
	// 14 = a^3 + a^2 + a
	// 15 = a^3 + a^2 + a + 1 = (a+1)^3
	// 16 = a^4
	// 17 = a^4 + 1 = (a+1)^4
	// 18 = a^4 + a
	// 19 = a^4 + a + 1
	// 20 = a^4 + a^2 = a^2(a+1)^2
	// 23 = (a+1)(a^3+a^2+1)
	// 34 = a(a+1)^4
	// 45 = (a+1)^3(a^2+a+1)
	// 52 = a^2(a^3+a^2+1)
	// 54 = a(a+1)^2(a^2+a+1)
	// 57 = (a+1)^2(a^3+a^2+1)
	// 60 = a^2(a+1)^3
	// 63 = (a+1)(a^2+a+1)^2
	// 75 = (a+1)^3(a^3+a^2+1)
	// 90 = a(a+1)^3(a^2+a+1) = a^6 + a^4 + a^3 + a
	// 170 = a(a+1)^6
	int data[] = {
			// A1:
			10,0,0,0,
			0,10,0,0,
			4,10,10,0,
			0,17,0,10,

			// A2:
			10,0,0,0,
			0,10,0,0,
			2,0,10,0,
			0,15,0,10,

			// A3:
			10,0,0,0,
			2,10,0,0,
			0,0,10,0,
			0,0,15,10,

			// A4:
			60,0,0,0,
			12,60,0,0,
			12,0,60,0,
			54,34,34,60,

			// A5:
			12,0,0,0,
			4,12,0,0,
			0,0,12,0,
			0,0,34,12,

			// A6:
			10,0,0,0,
			4,0,10,0,
			0,10,10,0,
			10,60,17,10,

	};

	data_size = 16 + 1;
	nb_gens = 6;
	group_order = 192;

	gens = NEW_int(nb_gens * data_size);
	Int_vec_zero(gens, nb_gens * data_size);

	int h, i, j, c, m, l;
	int *v;
	geometry::geometry_global Gg;
	number_theory::number_theory_domain NT;

	m = Int_vec_maximum(data, nb_gens * data_size);
	l = NT.int_log2(m) + 1;

	v = NEW_int(l);


	for (h = 0; h < nb_gens; h++) {
		for (i = 0; i < 16; i++) {
			Int_vec_zero(v, l);
			Gg.AG_element_unrank(F->p, v, 1, l, data[h * 16 + i]);
			c = 0;
			for (j = 0; j < l; j++) {
				c = F->mult(c, a);
				if (v[l - 1 - j]) {
					c = F->add(c, v[l - 1 - j]);
				}
			}
			gens[h * data_size + i] = c;
		}
		gens[h * data_size + 16] = 0;
	}
	FREE_int(v);

	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_F13_generators" << endl;
		for (h = 0; h < nb_gens; h++) {
			cout << "generator " << h << ":" << endl;
			Int_matrix_print(gens + h * data_size, 4, 4);
		}
	}
	if (f_v) {
		cout << "algebraic_geometry_global::cubic_surface_family_F13_generators done" << endl;
	}
}

int algebraic_geometry_global::nonconical_six_arc_get_nb_Eckardt_points(
		geometry::projective_space *P2,
		long int *Arc6, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::nonconical_six_arc_get_nb_Eckardt_points" << endl;
	}

	algebraic_geometry::eckardt_point_info *E;
	int nb_E;

	E = compute_eckardt_point_info(P2, Arc6, 0/*verbose_level*/);

	nb_E = E->nb_E;

	FREE_OBJECT(E);
	return nb_E;
}

algebraic_geometry::eckardt_point_info *algebraic_geometry_global::compute_eckardt_point_info(
		geometry::projective_space *P2,
	long int *arc6,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	algebraic_geometry::eckardt_point_info *E;

	if (f_v) {
		cout << "algebraic_geometry_global::compute_eckardt_point_info" << endl;
	}
	if (P2->Subspaces->n != 2) {
		cout << "algebraic_geometry_global::compute_eckardt_point_info "
				"P2->n != 2" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "arc: ";
		Lint_vec_print(cout, arc6, 6);
		cout << endl;
	}

	E = NEW_OBJECT(algebraic_geometry::eckardt_point_info);
	E->init(P2, arc6, verbose_level);

	if (f_v) {
		cout << "algebraic_geometry_global::compute_eckardt_point_info done" << endl;
	}
	return E;
}

int algebraic_geometry_global::test_nb_Eckardt_points(
		geometry::projective_space *P2,
		long int *S, int len, int pt, int nb_E, int verbose_level)
// input: S[5] and pt, which form a 6-arc.
// we assume that len = 5.
// nb_E the predicted number of Eckardt points
// output true if nb_E is correct, false otherwise.
{
	int f_v = (verbose_level >= 1);
	int ret = true;
	long int Arc6[6];

	if (f_v) {
		cout << "algebraic_geometry_global::test_nb_Eckardt_points" << endl;
	}
	if (len != 5) {
		return true;
	}

	Lint_vec_copy(S, Arc6, 5);
	Arc6[5] = pt;

	algebraic_geometry::eckardt_point_info *E;

	if (f_v) {
		cout << "algebraic_geometry_global::test_nb_Eckardt_points "
				"before compute_eckardt_point_info" << endl;
	}
	E = compute_eckardt_point_info(
			P2, Arc6, 0 /*verbose_level*/);
	if (f_v) {
		cout << "algebraic_geometry_global::test_nb_Eckardt_points "
				"after compute_eckardt_point_info" << endl;
	}


	if (E->nb_E != nb_E) {
		ret = false;
	}

	FREE_OBJECT(E);

	if (f_v) {
		cout << "algebraic_geometry_global::test_nb_Eckardt_points done" << endl;
	}
	return ret;
}

void algebraic_geometry_global::rearrange_arc_for_lifting(
		long int *Arc6,
		long int P1, long int P2, int partition_rk, long int *arc,
		int verbose_level)
// P1 and P2 are points on the arc.
// Find them and remove them
// so we can find the remaining four point of the arc:
{
	int f_v = (verbose_level >= 1);
	long int i, a, h;
	int part[4];
	long int pts[4];
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "algebraic_geometry_global::rearrange_arc_for_lifting" << endl;
	}
	arc[0] = P1;
	arc[1] = P2;
	h = 2;
	for (i = 0; i < 6; i++) {
		a = Arc6[i];
		if (a == P1 || a == P2) {
			continue;
		}
		arc[h++] = a;
	}
	if (h != 6) {
		cout << "algebraic_geometry_global::rearrange_arc_for_lifting "
				"h != 6" << endl;
		exit(1);
	}
	// now arc[2], arc[3], arc[4], arc[5] are the remaining four points
	// of the arc.

	Combi.set_partition_4_into_2_unrank(partition_rk, part);

	Lint_vec_copy(arc + 2, pts, 4);

	for (i = 0; i < 4; i++) {
		a = part[i];
		arc[2 + i] = pts[a];
	}

	if (f_v) {
		cout << "algebraic_geometry_global::rearrange_arc_for_lifting done" << endl;
	}
}

void algebraic_geometry_global::find_two_lines_for_arc_lifting(
		geometry::projective_space *P,
		long int P1, long int P2, long int &line1, long int &line2,
		int verbose_level)
// P1 and P2 are points on the arc and in the plane W=0.
// Note the points are points in PG(3,q), not in local coordinates in W=0.
// We find two skew lines in space through P1 and P2, respectively.
{
	int f_v = (verbose_level >= 1);
	int Basis[16];
	int Basis2[16];
	int Basis_search[16];
	int Basis_search_copy[16];
	int base_cols[4];
	int i, N, rk;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting" << endl;
	}
	if (P->Subspaces->n != 3) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"P->Subspaces->n != 3" << endl;
		exit(1);
	}
	// unrank points P1 and P2 in the plane W=3:
	// Note the points are points in PG(3,q), not in local coordinates.
	P->unrank_point(Basis, P1);
	P->unrank_point(Basis + 4, P2);
	if (Basis[3]) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"Basis[3] != 0, the point P1 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	if (Basis[7]) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"Basis[7] != 0, the point P2 does not lie "
				"in the hyperplane W = 0" << endl;
		exit(1);
	}
	Int_vec_zero(Basis + 8, 8);

	N = Gg.nb_PG_elements(3, P->Subspaces->q);
	// N = the number of points in PG(3,q)

	// Find the first line.
	// Loop over all points P.
	// Make sure the point does not belong to the hyperplane,
	// i.e. the last coordinate is nonzero.
	// Make sure the rank of the subspace spanned by P1, P2 and P is three.

	for (i = 0; i < N; i++) {
		Int_vec_copy(Basis, Basis_search, 4);
		Int_vec_copy(Basis + 4, Basis_search + 4, 4);

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				Basis_search + 8, 1, 4, i);

		if (Basis_search[11] == 0) {
			continue;
		}

		Int_vec_copy(Basis_search, Basis_search_copy, 12);

		rk = P->Subspaces->F->Linear_algebra->Gauss_easy_memory_given(
				Basis_search_copy, 3, 4, base_cols);

		if (rk == 3) {
			break;
		}
	}
	if (i == N) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"i == N, could not find line1" << endl;
		exit(1);
	}
	int p0, p1;

	p0 = i;

	// Find the second line.
	// Loop over all points Q after the first P.
	// Make sure the point does not belong to the hyperplane,
	// i.e. the last coordinate is nonzero.
	// Make sure the rank of the subspace spanned by P1, P2 and P and Q is four.

	for (i = p0 + 1; i < N; i++) {
		Int_vec_copy(Basis, Basis_search, 4);
		Int_vec_copy(Basis + 4, Basis_search + 4, 4);
		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				Basis_search + 8, 1, 4, p0);
		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				Basis_search + 12, 1, 4, i);
		if (Basis_search[15] == 0) {
			continue;
		}
		Int_vec_copy(Basis_search, Basis_search_copy, 16);
		rk = P->Subspaces->F->Linear_algebra->Gauss_easy_memory_given(
				Basis_search_copy, 4, 4, base_cols);
		if (rk == 4) {
			break;
		}
	}
	if (i == N) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"i == N, could not find line2" << endl;
		exit(1);
	}
	p1 = i;

	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"p0=" << p0 << " p1=" << p1 << endl;
	}
	P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
			Basis + 8, 1, 4, p0);
	P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
			Basis + 12, 1, 4, p1);
	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting " << endl;
		cout << "Basis:" << endl;
		Int_matrix_print(Basis, 4, 4);
	}
	Int_vec_copy(Basis, Basis2, 4);
	Int_vec_copy(Basis + 8, Basis2 + 4, 4);
	Int_vec_copy(Basis + 4, Basis2 + 8, 4);
	Int_vec_copy(Basis + 12, Basis2 + 12, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"Basis2:" << endl;
		Int_matrix_print(Basis2, 4, 4);
	}
	line1 = P->rank_line(Basis2);
	line2 = P->rank_line(Basis2 + 8);
	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"line1=" << line1 << " line2=" << line2 << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::find_two_lines_for_arc_lifting "
				"done" << endl;
	}
}

void algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed(
		geometry::projective_space *P,
		int *A3, int f_semilinear, int frobenius,
		long int line1, long int line2,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1[8];
	int Line2[8];
	int P1A[3];
	int P2A[3];
	int A3t[9];
	int x[3];
	int y[3];
	int xmy[4];
	int Mt[16];
	int M[16];
	int Mv[16];
	int v[4];
	int w[4];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;
	int f_swap; // does A3 swap P1 and P2?
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed" << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A3:" << endl;
		Int_matrix_print(A3, 3, 3);
		cout << "f_semilinear = " << f_semilinear
				<< " frobenius=" << frobenius << endl;
	}
	m1 = P->Subspaces->F->negate(1);
	P->unrank_line(Line1, line1);
	P->unrank_line(Line2, line2);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"input Line1:" << endl;
		Int_matrix_print(Line1, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"input Line2:" << endl;
		Int_matrix_print(Line2, 2, 4);
	}
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line1 + 4, Line1,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line2 + 4, Line2,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"modified Line1:" << endl;
		Int_matrix_print(Line1, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"modified Line2:" << endl;
		Int_matrix_print(Line2, 2, 4);
	}

	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			Line1, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			Line2, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			Line1 + 4, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			Line2 + 4, 1, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 = first point on Line1:" << endl;
		Int_matrix_print(Line1, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 = first point on Line2:" << endl;
		Int_matrix_print(Line2, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"x = second point on Line1:" << endl;
		Int_matrix_print(Line1 + 4, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"y = second point on Line2:" << endl;
		Int_matrix_print(Line2 + 4, 1, 4);
	}
	// compute P1 * A3 to figure out if A switches P1 and P2 or not:
	P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
			Line1, A3, P1A, 3, 3);
	P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
			Line2, A3, P2A, 3, 3);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 * A = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 * A = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	if (f_semilinear) {
		if (f_v) {
			cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
					"applying frobenius" << endl;
		}
		P->Subspaces->F->Linear_algebra->vector_frobenius_power_in_place(
				P1A, 3, frobenius);
		P->Subspaces->F->Linear_algebra->vector_frobenius_power_in_place(
				P2A, 3, frobenius);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P1 * A ^Phi^frobenius = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"P2 * A ^Phi^frobenius = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			P1A, 1, 3);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(
			P2A, 1, 3);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"normalized P1 * A = " << endl;
		Int_matrix_print(P1A, 1, 3);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"normalized P2 * A = " << endl;
		Int_matrix_print(P2A, 1, 3);
	}
	if (Sorting.int_vec_compare(P1A, Line1, 3) == 0) {
		f_swap = false;
		if (Sorting.int_vec_compare(P2A, Line2, 3)) {
			cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"We don't have a swap but A3 does not stabilize P2" << endl;
			exit(1);
		}
	}
	else if (Sorting.int_vec_compare(P1A, Line2, 3) == 0) {
		f_swap = true;
		if (Sorting.int_vec_compare(P2A, Line1, 3)) {
			cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"We have a swap but A3 does not map P2 to P1" << endl;
			exit(1);
		}
	}
	else {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"unable to determine if we have a swap or not." << endl;
		exit(1);
	}

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"f_swap=" << f_swap << endl;
	}

	Int_vec_copy(Line1 + 4, x, 3);
	Int_vec_copy(Line2 + 4, y, 3);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"x:" << endl;
		Int_matrix_print(x, 1, 3);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"y:" << endl;
		Int_matrix_print(y, 1, 3);
	}

	P->Subspaces->F->Linear_algebra->linear_combination_of_vectors(
			1, x, m1, y, xmy, 3);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"xmy:" << endl;
		Int_matrix_print(xmy, 1, 3);
	}

	P->Subspaces->F->Linear_algebra->transpose_matrix(
			A3, A3t, 3, 3);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A3t:" << endl;
		Int_matrix_print(A3t, 3, 3);
	}


	P->Subspaces->F->Linear_algebra->mult_vector_from_the_right(
			A3t, xmy, v, 3, 3);
	if (f_semilinear) {
		P->Subspaces->F->Linear_algebra->vector_frobenius_power_in_place(
				v, 3, frobenius);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"v:" << endl;
		Int_matrix_print(v, 1, 3);
	}
	P->Subspaces->F->Linear_algebra->mult_vector_from_the_right(
			A3t, x, w, 3, 3);
	if (f_semilinear) {
		P->Subspaces->F->Linear_algebra->vector_frobenius_power_in_place(
				w, 3, frobenius);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"w:" << endl;
		Int_matrix_print(w, 1, 3);
	}

	if (f_swap) {
		Int_vec_copy(Line2 + 4, Mt + 0, 4);
		Int_vec_copy(Line2 + 0, Mt + 4, 4);
		Int_vec_copy(Line1 + 4, Mt + 8, 4);
		Int_vec_copy(Line1 + 0, Mt + 12, 4);
	}
	else {
		Int_vec_copy(Line1 + 4, Mt + 0, 4);
		Int_vec_copy(Line1 + 0, Mt + 4, 4);
		Int_vec_copy(Line2 + 4, Mt + 8, 4);
		Int_vec_copy(Line2 + 0, Mt + 12, 4);
	}

	P->Subspaces->F->Linear_algebra->negate_vector_in_place(
			Mt + 8, 8);
	P->Subspaces->F->Linear_algebra->transpose_matrix(
			Mt, M, 4, 4);
	//int_vec_copy(Mt, M, 16);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"M:" << endl;
		Int_matrix_print(M, 4, 4);
	}

	P->Subspaces->F->Linear_algebra->invert_matrix_memory_given(
			M,
			Mv, 4, M_tmp, tmp_basecols,
			0 /* verbose_level */);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"Mv:" << endl;
		Int_matrix_print(Mv, 4, 4);
	}

	v[3] = 0;
	w[3] = 0;
	P->Subspaces->F->Linear_algebra->mult_vector_from_the_right(
			Mv, v, lmei, 4, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"lmei:" << endl;
		Int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	//epsilon = lmei[2];
	//iota = lmei[3];

	if (f_swap) {
		P->Subspaces->F->Linear_algebra->linear_combination_of_three_vectors(
				lambda, y, mu, Line2, m1, w, abgd, 3);
	}
	else {
		P->Subspaces->F->Linear_algebra->linear_combination_of_three_vectors(
				lambda, x, mu, Line1, m1, w, abgd, 3);
	}
	abgd[3] = lambda;
	if (f_semilinear) {
		P->Subspaces->F->Linear_algebra->vector_frobenius_power_in_place(
				abgd, 4, P->Subspaces->F->e - frobenius);
	}
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"abgd:" << endl;
		Int_matrix_print(abgd, 1, 4);
	}
	// make an identity matrix:
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			A4[i * 4 + j] = A3[i * 3 + j];
		}
		A4[i * 4 + 3] = 0;
	}
	// fill in the last row:
	Int_vec_copy(abgd, A4 + 4 * 3, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed "
				"A4:" << endl;
		Int_matrix_print(A4, 4, 4);
	}

	if (f_semilinear) {
		A4[16] = frobenius;
	}

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_fixed done" << endl;
	}
}

void algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved(
		geometry::projective_space *P,
		long int line1_from, long int line1_to,
		long int line2_from, long int line2_to,
		int *A4,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int Line1_from[8];
	int Line2_from[8];
	int Line1_to[8];
	int Line2_to[8];
	int P1[4];
	int P2[4];
	int x[4];
	int y[4];
	int u[4];
	int v[4];
	int umv[4];
	int M[16];
	int Mv[16];
	int lmei[4];
	int m1;
	int M_tmp[16];
	int tmp_basecols[4];
	int lambda, mu; //, epsilon, iota;
	int abgd[4];
	int i, j;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved" << endl;
	}
	m1 = P->Subspaces->F->negate(1);

	P->unrank_line(Line1_from, line1_from);
	P->unrank_line(Line2_from, line2_from);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line1_from:" << endl;
		Int_matrix_print(Line1_from, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line2_from:" << endl;
		Int_matrix_print(Line2_from, 2, 4);
	}
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line1_from + 4, Line1_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line2_from + 4, Line2_from,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line1_from:" << endl;
		Int_matrix_print(Line1_from, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line2_from:" << endl;
		Int_matrix_print(Line2_from, 2, 4);
	}

	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line1_from, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line2_from, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line1_from + 4, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line2_from + 4, 1, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P1 = first point on Line1_from:" << endl;
		Int_matrix_print(Line1_from, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P2 = first point on Line2_from:" << endl;
		Int_matrix_print(Line2_from, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"u = second point on Line1_from:" << endl;
		Int_matrix_print(Line1_from + 4, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"v = second point on Line2_from:" << endl;
		Int_matrix_print(Line2_from + 4, 1, 4);
	}
	Int_vec_copy(Line1_from + 4, u, 4);
	Int_vec_copy(Line1_from, P1, 4);
	Int_vec_copy(Line2_from + 4, v, 4);
	Int_vec_copy(Line2_from, P2, 4);


	P->unrank_line(Line1_to, line1_to);
	P->unrank_line(Line2_to, line2_to);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line1_to:" << endl;
		Int_matrix_print(Line1_to, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"input Line2_to:" << endl;
		Int_matrix_print(Line2_to, 2, 4);
	}
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line1_to + 4, Line1_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line1[3] = 0 and Line1[7] = 1
	P->Subspaces->F->Linear_algebra->Gauss_step_make_pivot_one(
			Line2_to + 4, Line2_to,
		4 /* len */, 3 /* idx */, 0 /* verbose_level*/);
		// afterwards:  v1,v2 span the same space as before
		// v2[idx] = 0, v1[idx] = 1,
		// So, now Line2[3] = 0 and Line2[7] = 1
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line1_to:" << endl;
		Int_matrix_print(Line1_to, 2, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"modified Line2_to:" << endl;
		Int_matrix_print(Line2_to, 2, 4);
	}

	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line1_to, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line2_to, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line1_to + 4, 1, 4);
	P->Subspaces->F->Projective_space_basic->PG_element_normalize(Line2_to + 4, 1, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P1 = first point on Line1_to:" << endl;
		Int_matrix_print(Line1_to, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"P2 = first point on Line2_to:" << endl;
		Int_matrix_print(Line2_to, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"x = second point on Line1_to:" << endl;
		Int_matrix_print(Line1_to + 4, 1, 4);
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"y = second point on Line2_to:" << endl;
		Int_matrix_print(Line2_to + 4, 1, 4);
	}


	Int_vec_copy(Line1_to + 4, x, 4);
	//int_vec_copy(Line1_to, P1, 4);
	if (Sorting.int_vec_compare(P1, Line1_to, 4)) {
		cout << "Line1_from and Line1_to must intersect in W=0" << endl;
		exit(1);
	}
	Int_vec_copy(Line2_to + 4, y, 4);
	//int_vec_copy(Line2_to, P2, 4);
	if (Sorting.int_vec_compare(P2, Line2_to, 4)) {
		cout << "Line2_from and Line2_to must intersect in W=0" << endl;
		exit(1);
	}


	P->Subspaces->F->Linear_algebra->linear_combination_of_vectors(
			1, u, m1, v, umv, 3);
	umv[3] = 0;
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"umv:" << endl;
		Int_matrix_print(umv, 1, 4);
	}

	Int_vec_copy(x, M + 0, 4);
	Int_vec_copy(P1, M + 4, 4);
	Int_vec_copy(y, M + 8, 4);
	Int_vec_copy(P2, M + 12, 4);

	P->Subspaces->F->Linear_algebra->negate_vector_in_place(
			M + 8, 8);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"M:" << endl;
		Int_matrix_print(M, 4, 4);
	}

	P->Subspaces->F->Linear_algebra->invert_matrix_memory_given(
			M, Mv, 4, M_tmp, tmp_basecols, 0 /* verbose_level */);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"Mv:" << endl;
		Int_matrix_print(Mv, 4, 4);
	}

	P->Subspaces->F->Linear_algebra->mult_vector_from_the_left(
			umv, Mv, lmei, 4, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"lmei=" << endl;
		Int_matrix_print(lmei, 1, 4);
	}
	lambda = lmei[0];
	mu = lmei[1];
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"lambda=" << lambda << " mu=" << mu << endl;
	}

	P->Subspaces->F->Linear_algebra->linear_combination_of_three_vectors(
			lambda, x, mu, P1, m1, u, abgd, 3);
	// abgd = lambda * x + mu * P1 - u, with a lambda in the 4th coordinate.

	abgd[3] = lambda;

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"abgd:" << endl;
		Int_matrix_print(abgd, 1, 4);
	}

	// make an identity matrix:
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			if (i == j) {
				A4[i * 4 + j] = 1;
			}
			else {
				A4[i * 4 + j] = 0;
			}
		}
	}
	// fill in the last row:
	Int_vec_copy(abgd, A4 + 3 * 4, 4);
	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved "
				"A4:" << endl;
		Int_matrix_print(A4, 4, 4);

		P->Subspaces->F->Io->print_matrix_latex(cout, A4, 4, 4);

	}

	if (f_v) {
		cout << "algebraic_geometry_global::hyperplane_lifting_with_two_lines_moved done" << endl;
	}

}

void algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer(
		geometry::projective_space *P3,
		long int line1_from, long int line2_from,
		long int line1_to, long int line2_to,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer" << endl;
	}

	if (P3->Subspaces->n != 3) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer "
				"n != 3" << endl;
		exit(1);
	}
	int A4[16];


	hyperplane_lifting_with_two_lines_moved(P3,
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer "
				"A4=" << endl;
		Int_matrix_print(A4, 4, 4);
	}

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer done" << endl;
	}
}

void algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text(
		geometry::projective_space *P3,
		std::string &line1_from_text, std::string &line2_from_text,
		std::string &line1_to_text, std::string &line2_to_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text" << endl;
	}
	if (P3->Subspaces->n != 3) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer n != 3" << endl;
		exit(1);
	}

	//geometry::geometry_global Gg;
	int A4[16];


	int *line1_from_data;
	int *line2_from_data;
	int *line1_to_data;
	int *line2_to_data;
	int sz;

	Int_vec_scan(line1_from_text, line1_from_data, sz);
	if (sz != 8) {
		cout << "line1_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line2_from_text, line2_from_data, sz);
	if (sz != 8) {
		cout << "line2_from_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line1_to_text, line1_to_data, sz);
	if (sz != 8) {
		cout << "line1_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}
	Int_vec_scan(line2_to_text, line2_to_data, sz);
	if (sz != 8) {
		cout << "line2_to_text must contain exactly 8 integers" << endl;
		exit(1);
	}

	long int line1_from;
	long int line2_from;
	long int line1_to;
	long int line2_to;

	line1_from = P3->rank_line(line1_from_data);
	line2_from = P3->rank_line(line2_from_data);
	line1_to = P3->rank_line(line1_to_data);
	line2_to = P3->rank_line(line2_to_data);


	hyperplane_lifting_with_two_lines_moved(
			P3,
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text A4=" << endl;
		Int_matrix_print(A4, 4, 4);
	}

	if (f_v) {
		cout << "algebraic_geometry_global::do_move_two_lines_in_hyperplane_stabilizer_text done" << endl;
	}
}



}}}

