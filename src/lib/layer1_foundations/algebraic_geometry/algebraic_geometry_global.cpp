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

	del_Pezzo_surface->pal->write_points_to_txt_file(
			Formula->name_of_formula, verbose_level);

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


	char str[1000];

	snprintf(str, 1000, "Gr_%d_%d_%d.tex",
			P->Subspaces->n + 1, k, P->Subspaces->F->q);
	fname.assign(str);
	snprintf(str, 1000, "Cheat Sheet Gr($%d,%d,%d$)",
			P->Subspaces->n + 1, k, P->Subspaces->F->q);
	title.assign(str);




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
		int &N_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "algebraic_geometry_global::map" << endl;
	}
	if (f_v) {
		cout << "algebraic_geometry_global::map n = " << P->Subspaces->n << endl;
	}



	int idx;
	ring_theory::homogeneous_polynomial_domain *Ring;

	Ring = Get_ring(ring_label);

	idx = orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol(formula_label);

	if (idx < 0) {
		cout << "could not find symbol " << formula_label << endl;
		exit(1);
	}
	orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object(idx);

	if (orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx].type != orbiter_kernel_system::t_object) {
		cout << "symbol table entry must be of type t_object" << endl;
		exit(1);
	}

	// ToDo: use Symbolic Object instead of formula

	if (orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx].object_type == t_collection) {
		cout << "symbol table entry is a collection" << endl;

		vector<string> *List;

		List = (vector<string> *) orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx].ptr;
		int i;

		int *coefficient_vector; // [List->size() * Ring->get_nb_monomials()]

		coefficient_vector = NEW_int(List->size() * Ring->get_nb_monomials());

		for (i = 0; i < List->size(); i++) {
			int idx1;

			idx1 = orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol((*List)[i]);
			if (idx1 < 0) {
				cout << "could not find symbol " << (*List)[i] << endl;
				exit(1);
			}
			expression_parser::formula *Formula;
			Formula = (expression_parser::formula *)
					orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx1].ptr;

			if (f_v) {
				cout << "projective_space_global::map i=" << i << " / " << List->size()
						<< " before Ring->get_coefficient_vector" << endl;
			}
			Ring->get_coefficient_vector(Formula,
					evaluate_text,
					coefficient_vector + i * Ring->get_nb_monomials(),
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::map i=" << i << " / " << List->size()
						<< " after Ring->get_coefficient_vector" << endl;
			}
		}

		if (f_v) {
			cout << "projective_space_global::map coefficient_vector:" << endl;
			Int_matrix_print(coefficient_vector, List->size(), Ring->get_nb_monomials());
		}



		Ring->evaluate_regular_map(
				coefficient_vector,
				List->size(),
				P,
				Image_pts, N_points,
				verbose_level);


		if (f_v) {
			cout << "projective_space_global::map permutation:" << endl;
			Lint_vec_print(cout, Image_pts, N_points);
			cout << endl;
		}



	}
#if 0
	else if (orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx].object_type == t_formula) {
		cout << "symbol table entry is a formula" << endl;

		expression_parser::formula *Formula;
		Formula = (expression_parser::formula *)
				orbiter_kernel_system::Orbiter->Orbiter_symbol_table->Table[idx].ptr;

		int *coefficient_vector; // [Ring->get_nb_monomials()]

		coefficient_vector = NEW_int(Ring->get_nb_monomials());

		Ring->get_coefficient_vector(Formula,
				evaluate_text,
				coefficient_vector,
				verbose_level);


		Ring->evaluate_regular_map(
				coefficient_vector,
				1,
				P,
				Image_pts, N_points,
				verbose_level);


		FREE_int(coefficient_vector);



	}
#endif
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "algebraic_geometry_global::map done" << endl;
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



}}}

