/*
 * polynomial_ring_activity.cpp
 *
 *  Created on: Feb 26, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {



polynomial_ring_activity::polynomial_ring_activity()
{
	Descr = NULL;
	HPD = NULL;

}


polynomial_ring_activity::~polynomial_ring_activity()
{

}

void polynomial_ring_activity::init(
		ring_theory::polynomial_ring_activity_description *Descr,
		ring_theory::homogeneous_polynomial_domain *HPD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polynomial_ring_activity::init" << endl;
	}


	polynomial_ring_activity::Descr = Descr;
	polynomial_ring_activity::HPD = HPD;

	if (f_v) {
		cout << "polynomial_ring_activity::init done" << endl;
	}
}

void polynomial_ring_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polynomial_ring_activity::perform_activity" << endl;
	}


	if (Descr->f_cheat_sheet) {

		if (f_v) {
			cout << "polynomial_ring_activity::perform_activity f_cheat_sheet" << endl;
		}
		algebra::algebra_global Algebra;

		//Algebra.do_cheat_sheet_GF(F, verbose_level);
		//HPD->print_monomial_ordering_latex(cout);

		Algebra.do_cheat_sheet_ring(HPD, verbose_level);


	}
	else if (Descr->f_ideal) {

		if (f_v) {
			cout << "polynomial_ring_activity::perform_activity f_ideal" << endl;
		}


		//ring_theory::homogeneous_polynomial_domain *HPD;


		//HPD = Get_ring(Descr->ideal_ring_label);

		int dim_kernel;
		int nb_monomials;
		int *Kernel;

		HPD->create_ideal(
				Descr->ideal_label_txt,
				Descr->ideal_label_tex,
				Descr->ideal_point_set_label,
				dim_kernel, nb_monomials, Kernel,
				verbose_level - 2);

		if (f_v) {
			cout << "The ideal has dimension " << dim_kernel << endl;
			cout << "generators for the ideal:" << endl;
			Int_matrix_print(Kernel, dim_kernel, nb_monomials);

			int i;

			for (i = 0; i < dim_kernel; i++) {
				HPD->print_equation_relaxed(cout, Kernel + i * nb_monomials);
				cout << endl;
			}
		}

	}
	else if (Descr->f_apply_transformation) {
		if (f_v) {
			cout << "polynomial_ring_activity::perform_activity f_apply_transformation" << endl;
			cout << "polynomial_ring_activity::perform_activity vector of group elements " << Descr->apply_transformation_vector_ge_label << endl;
		}

		apps_algebra::vector_ge_builder *VB;
		data_structures_groups::vector_ge *V;
			actions::action *A;

		int i;

		VB = Get_object_of_type_vector_ge(Descr->apply_transformation_vector_ge_label);

		V = VB->V;

		if (f_v) {
			cout << "polynomial_ring_activity::perform_activity the input vector has length " << V->len << endl;
		}

		A = V->A;

		if (f_v) {
			cout << "polynomial_ring_activity::perform_activity the input vector has length " << V->len << endl;
			cout << "Group elements:" << endl;
			for (i = 0; i < V->len; i++) {
				cout << i << " / " << V->len << ":" << endl;
				A->Group_element->element_print_quick(V->ith(i), cout);
			}
		}



		int *Eqn_in;
		int *Eqn_out;
		int sz;


		Get_int_vector_from_label(Descr->apply_transformation_Eqn_in_label, Eqn_in, sz, 0 /* verbose_level */);
		//Int_vec_scan(Descr->apply_transformation_Eqn_in_label, Eqn_in, sz);

		if (f_v) {
			cout << "The input equation is:";
			Int_vec_print(cout, Eqn_in, sz);
			cout << endl;
		}

		if (sz != HPD->get_nb_monomials()) {
			cout << "polynomial_ring_activity::perform_activity the equation does not have the right amount of coefficients" << endl;
			cout << "have: " << sz << endl;
			cout << "need: " << HPD->get_nb_monomials() << endl;
			exit(1);
		}

		if (f_v) {
			cout << "The input equation is:";
			HPD->print_equation_simple(cout, Eqn_in);
			cout << endl;
		}

		int *Elt_inv;


		Elt_inv = NEW_int(A->elt_size_in_int);
		Eqn_out = NEW_int(HPD->get_nb_monomials());


		int *Diagonal_part = NULL;

		if (HPD->degree == 2) {
			Diagonal_part = NEW_int(V->len * HPD->nb_variables);
		}

		for (i = 0; i < V->len; i++) {
			if (f_v) {
				cout << "polynomial_ring_activity::perform_activity i=" << i << " / " << V->len << endl;
				cout << "Group element:" << endl;
				A->Group_element->element_print_quick(V->ith(i), cout);
			}

			if (f_v) {
				cout << "polynomial_ring_activity::perform_activity before element_invert" << endl;
			}

			A->Group_element->element_invert(V->ith(i), Elt_inv, 0);
			if (f_v) {
				cout << "polynomial_ring_activity::perform_activity after element_invert" << endl;
			}

			if (f_v) {
				cout << "polynomial_ring_activity::perform_activity before substitute_linear" << endl;
			}
			HPD->substitute_linear(Eqn_in /* coeff_in */, Eqn_out /* coeff_out */,
					Elt_inv /* Mtx_inv */, 0/*verbose_level*/);
			if (f_v) {
				cout << "polynomial_ring_activity::perform_activity after substitute_linear" << endl;
			}

			HPD->get_F()->Projective_space_basic->PG_element_normalize_from_front(
					Eqn_out, 1, HPD->get_nb_monomials());


			if (f_v) {
				cout << "The mapped equation is:";
				HPD->print_equation_simple(cout, Eqn_out);
				cout << endl;
				cout << "The mapped equation is:";
				HPD->print_equation_tex(cout, Eqn_out);
				cout << endl;
				cout << "The mapped equation is:";
				HPD->print_equation_relaxed(cout, Eqn_out);
				cout << endl;
			}

			if (HPD->degree == 2) {
				int n = HPD->nb_variables;
				int *M;

				M = NEW_int(n * n);
				HPD->get_quadratic_form_matrix(Eqn_out, M);
				cout << "quadratic form matrix:" << endl;
				Int_matrix_print(M, n, n);
				FREE_int(M);
				int h;
				for (h = 0; h < n; h++) {
					Diagonal_part[i * n + h] = M[h * n + h];
				}
			}
		}
		if (HPD->degree == 2) {
			cout << "Diagonal part:" << endl;
			Int_matrix_print(Diagonal_part, V->len, HPD->nb_variables);
		}

		FREE_int(Elt_inv);
		FREE_int(Eqn_out);


	}
	else if (Descr->f_set_variable_names) {
		if (f_v) {
			cout << "-set_variable_names "
					<< Descr->set_variable_names_txt << " "
					<< Descr->set_variable_names_tex << " "
					<< endl;
		}

		HPD->remake_symbols(0 /* symbol_offset */,
				Descr->set_variable_names_txt,
				Descr->set_variable_names_tex,
					verbose_level);

	}
	else if (Descr->f_print_equation) {
		if (f_v) {
			cout << "-print_equation "
					<< Descr->print_equation_input << " "
					<< endl;
		}

		int *eqn;
		int sz;

		Get_int_vector_from_label(Descr->print_equation_input, eqn, sz, 0 /* verbose_level */);

		HPD->print_equation_tex(cout, eqn);
		cout << endl;
	}

	else if (Descr->f_parse_equation_wo_parameters) {
		if (f_v) {
			cout << "-parse_equation_wo_parameters "
					<< Descr->parse_equation_wo_parameters_name_of_formula << " "
					<< Descr->parse_equation_wo_parameters_name_of_formula_tex << " "
					<< Descr->parse_equation_wo_parameters_equation_text << " "
					<< endl;
		}


		int *eqn;
		int eqn_size;

		HPD->parse_equation_wo_parameters(
				Descr->parse_equation_wo_parameters_name_of_formula,
				Descr->parse_equation_wo_parameters_name_of_formula_tex,
				Descr->parse_equation_wo_parameters_equation_text,
				eqn, eqn_size,
				verbose_level);
		if (f_v) {
			cout << "-parse_equation_wo_parameters The equation is:" << endl;
			Int_vec_print(cout, eqn, eqn_size);
			cout << endl;
		}

	}


	else if (Descr->f_parse_equation) {
		if (f_v) {
			cout << "-parse_equation "
					<< Descr->parse_equation_name_of_formula << " "
					<< Descr->parse_equation_name_of_formula_tex << " "
					<< Descr->parse_equation_equation_text << " "
					<< Descr->parse_equation_equation_parameters << " "
					<< Descr->parse_equation_equation_parameter_values << " "
					<< endl;
		}


		int *eqn;
		int eqn_size;

		HPD->parse_equation_and_substitute_parameters(
				Descr->parse_equation_name_of_formula,
				Descr->parse_equation_name_of_formula_tex,
				Descr->parse_equation_equation_text,
				Descr->parse_equation_equation_parameters,
				Descr->parse_equation_equation_parameter_values,
				eqn, eqn_size,
				verbose_level);
		if (f_v) {
			cout << "-parse_equation The equation is:" << endl;
			Int_vec_print(cout, eqn, eqn_size);
			cout << endl;
		}

	}

	else if (Descr->f_table_of_monomials_write_csv) {
		if (f_v) {
			cout << "-table_of_monomials_write_csv "
					<< Descr->table_of_monomials_write_csv_label << " "
					<< endl;
		}

		ring_theory::ring_theory_global RT;

		RT.make_table_of_monomials(
				HPD,
				Descr->table_of_monomials_write_csv_label,
				verbose_level);

	}


	//int f_table_of_monomials_write_csv;
	//std::string table_of_monomials_write_csv_label;


	if (f_v) {
		cout << "polynomial_ring_activity::perform_activity done" << endl;
	}

}



}}}


