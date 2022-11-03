/*
 * formula_activity.cpp
 *
 *  Created on: Feb 4, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {


formula_activity::formula_activity()
{
	Descr = NULL;
	f = NULL;
}

formula_activity::~formula_activity()
{
}

void formula_activity::init(formula_activity_description *Descr,
		formula *f,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_activity::init" << endl;
	}

	formula_activity::Descr = Descr;
	formula_activity::f = f;
	if (f_v) {
		cout << "formula_activity::init done" << endl;
	}
}

void formula_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_activity::perform_activity" << endl;
	}


	if (Descr->f_export) {

		std::string fname;
		fname.assign(f->name_of_formula);
		fname.append(".gv");

		{
			std::ofstream ost(fname);

			cout << "formula " << f->name_of_formula << " = ";
			f->tree->print_easy(cout);
			cout << endl;



			cout << "formula " << f->name_of_formula << " = " << endl;
			f->tree->print(cout);
			cout << endl;

			f->tree->Root->export_graphviz(f->name_of_formula, ost);
		}

	}
	else if (Descr->f_evaluate) {

		cout << "before evaluate" << endl;

		field_theory::finite_field *F;

		F = orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(Descr->evaluate_finite_field_label);

		expression_parser_domain ED;
		//int a;

		//a = ;
		ED.evaluate_formula(
				f,
				F,
				Descr->evaluate_assignment,
				verbose_level);



	}
	else if (Descr->f_print_over_Fq) {

		cout << "before f_print_over_Fq" << endl;

		field_theory::finite_field *F;

		F = orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(Descr->print_over_Fq_field_label);

		f->print_easy(F, cout);
		cout << endl;

	}
	else if (Descr->f_sweep) {

		cout << "before f_seep" << endl;

		field_theory::finite_field *F;

		F = orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(Descr->sweep_field_label);

		do_sweep(FALSE /* f_affine */,
				f,
				F, Descr->sweep_variables,
				verbose_level);

	}
	else if (Descr->f_sweep_affine) {

		cout << "before f_seep_affine" << endl;

		field_theory::finite_field *F;

		F = orbiter_kernel_system::Orbiter->get_object_of_type_finite_field(Descr->sweep_affine_field_label);

		do_sweep(TRUE /* f_affine */,
				f,
				F, Descr->sweep_affine_variables,
				verbose_level);

	}



	if (f_v) {
		cout << "formula_activity::perform_activity done" << endl;
	}

}


void formula_activity::do_sweep(int f_affine,
		formula *f,
		field_theory::finite_field *F, std::string &sweep_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_activity::do_sweep" << endl;
	}


	f->print_easy(F, cout);
	cout << endl;

	data_structures::string_tools ST;
	std::vector<std::string> symbol_table;

	ST.parse_comma_separated_values(symbol_table,
			sweep_variables,
			verbose_level);

	expression_parser_domain ED;
	geometry::geometry_global Gg;
	int n, N;
	int *v;
	char str[1000];


	n = symbol_table.size();
	v = NEW_int(n);

	N = Gg.nb_AG_elements(n, F->q);

	orbiter_kernel_system::file_io Fio;

	snprintf(str, sizeof(str), "_q%d", F->q);

	string fname;
	fname.assign("sweep_");
	fname.append(f->name_of_formula);
	fname.append(str);
	fname.append(".csv");

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	int degree;

	if (!f->is_homogeneous(degree, verbose_level - 3)) {
		cout << "not homogeneous" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "homogeneous of degree " << degree << endl;
	}

	ring_theory::homogeneous_polynomial_domain *Poly;

	Poly = NEW_OBJECT(ring_theory::homogeneous_polynomial_domain);

	if (f_v) {
		cout << "before Poly->init" << endl;
	}
	Poly->init(F,
			f->nb_managed_vars /* nb_vars */, degree,
			t_PART,
			0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "after Poly->init" << endl;
	}

	int *fun;
	int N_points;

	if (f_affine) {
		number_theory::number_theory_domain NT;

		N_points = NT.i_power_j(F->q, Poly->nb_variables - 1);
	}
	else {
		geometry::geometry_global Gg;

		N_points = Gg.nb_PG_elements(Poly->nb_variables - 1, F->q);
	}
	fun = NEW_int(N_points);


	{
		ofstream ost(fname);
		int i, j, cnt;

		ost << "Row,index,parameters,coefficients,evaluation_vector" << endl;


		cnt = 0;

		for (i = 0; i < N; i++) {


			cout << "sweep is at " << i << " / " << N << endl;
			string values;

			Gg.AG_element_unrank(F->q, v, 1, n, i);

			if (f_affine) {
				if (v[0] == 0) {
					continue;
				}
			}

			values.assign("");
			for (j = 0; j < n; j++) {
				values.append(symbol_table[j]);
				values.append("=");
				snprintf(str, sizeof(str), "%d", v[j]);
				values.append(str);
				if (j < n - 1) {
					values.append(",");
				}
			}

			int *Values;
			int nb_monomials;

			ED.evaluate_managed_formula(
					f,
					F,
					values,
					Values, nb_monomials,
					verbose_level - 2);


			int c;

			c = Int_vec_is_zero(Values, nb_monomials);

			if (!c) {


				if (f_affine) {
					Poly->polynomial_function_affine(Values, fun, verbose_level);
				}
				else {
					Poly->polynomial_function(Values, fun, verbose_level);
				}


				ost << cnt << "," << i;
				{
					string str;
					ost << ",";
					//ost << values;
					orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, v, n);
					ost << str;
					ost << ",";
					orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, Values, nb_monomials);
					ost << str;
					ost << ",";
					orbiter_kernel_system::Orbiter->Int_vec->create_string_with_quotes(str, fun, N_points);
					ost << str;
				}
				ost << endl;

				cnt++;

			}

			FREE_int(Values);

		} // next i
		ost << "END" << endl;
	}
	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	FREE_int(fun);
	FREE_OBJECT(Poly);

	if (f_v) {
		cout << "formula_activity::do_sweep done" << endl;
	}

}


}}}

