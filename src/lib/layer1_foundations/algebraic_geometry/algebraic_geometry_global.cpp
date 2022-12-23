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
		Formula->print();
	}

	if (!Formula->f_is_homogeneous) {
		cout << "Formula is not homogeneous" << endl;
		exit(1);
	}
	if (Formula->degree != 4) {
		cout << "Formula is not of degree 4. Degree is " << Formula->degree << endl;
		exit(1);
	}
	if (Formula->nb_managed_vars != 3) {
		cout << "Formula should have 3 managed variables. Has " << Formula->nb_managed_vars << endl;
		exit(1);
	}

	ring_theory::homogeneous_polynomial_domain *Poly4_3;

	Poly4_3 = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface before Poly->init" << endl;
	}
	Poly4_3->init(P->F,
			Formula->nb_managed_vars /* nb_vars */, Formula->degree,
			t_PART,
			verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface after Poly->init" << endl;
	}


	expression_parser::syntax_tree_node **Subtrees;
	int nb_monomials;

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface before Formula->get_subtrees" << endl;
	}
	Formula->get_subtrees(Poly4_3, Subtrees, nb_monomials, verbose_level);
	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface after Formula->get_subtrees" << endl;
	}

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


	int *Coefficient_vector;

	Coefficient_vector = NEW_int(nb_monomials);

	Formula->evaluate(Poly4_3,
			Subtrees, evaluate_text, Coefficient_vector,
			verbose_level);

	if (f_v) {
		cout << "algebraic_geometry_global::analyze_del_Pezzo_surface coefficient vector:" << endl;
		Int_vec_print(cout, Coefficient_vector, nb_monomials);
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

	del_Pezzo_surface->pal->write_points_to_txt_file(Formula->name_of_formula, verbose_level);

	del_Pezzo_surface->create_latex_report(Formula->name_of_formula, Formula->name_of_formula_latex, verbose_level);

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
			P->n + 1, k, P->F->q);
	fname.assign(str);
	snprintf(str, 1000, "Cheat Sheet Gr($%d,%d,%d$)",
			P->n + 1, k, P->F->q);
	title.assign(str);




	{
		ofstream ost(fname);
		orbiter_kernel_system::latex_interface L;

		L.head(ost,
				FALSE /* f_book*/,
				TRUE /* f_title */,
				title, author,
				FALSE /* f_toc */,
				FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
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
		cout << "algebraic_geometry_global::report_grassmannian written file " << fname << " of size "
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
		cout << "algebraic_geometry_global::map n = " << P->n << endl;
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
	else {
		cout << "symbol table entry must be either a formula or a collection" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "algebraic_geometry_global::map done" << endl;
	}
}


}}}

