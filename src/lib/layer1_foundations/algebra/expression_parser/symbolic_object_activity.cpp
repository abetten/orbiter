/*
 * symbolic_object_activity.cpp
 *
 *  Created on: Feb 4, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace expression_parser {


symbolic_object_activity::symbolic_object_activity()
{
	Record_birth();
	Descr = NULL;
	f = NULL;
}

symbolic_object_activity::~symbolic_object_activity()
{
	Record_death();
}

void symbolic_object_activity::init(
		symbolic_object_activity_description *Descr,
		symbolic_object_builder *f,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::init" << endl;
	}

	symbolic_object_activity::Descr = Descr;
	symbolic_object_activity::f = f;
	if (f_v) {
		cout << "symbolic_object_activity::init done" << endl;
	}
}

void symbolic_object_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::perform_activity" << endl;
	}


	if (Descr->f_print) {

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_print" << endl;
		}

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before print" << endl;
		}
		print(verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after print" << endl;
		}

	}
	else if (Descr->f_latex) {

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_latex" << endl;
		}

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before latex" << endl;
		}
		latex(verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after latex" << endl;
		}

	}
	else if (Descr->f_evaluate_affine) {
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_evaluate_affine" << endl;
		}

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before evaluate_affine" << endl;
		}
		evaluate_affine(
				verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after evaluate_affine" << endl;
		}
	}
	else if (Descr->f_collect_monomials_binary) {
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_collect_monomials_binary" << endl;
		}

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before collect_monomials_binary" << endl;
		}
		collect_monomials_binary(
				verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after collect_monomials_binary" << endl;
		}
	}


#if 0
	if (Descr->f_export) {

		std::string fname;
		fname = f->name_of_formula + ".gv";

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

		//field_theory::finite_field *F;

		//F = Get_finite_field(Descr->evaluate_finite_field_label);

		expression_parser_domain ED;
		//int a;

		//a = ;
		ED.evaluate_formula(
				f,
				Descr->evaluate_assignment,
				verbose_level);



	}
	else if (Descr->f_print) {

		cout << "before f_print" << endl;

		//field_theory::finite_field *F;

		//F = Get_finite_field(Descr->print_over_Fq_field_label);

		f->print_easy(cout);
		cout << endl;

	}
	else if (Descr->f_sweep) {

		cout << "before f_seep" << endl;

		//field_theory::finite_field *F;

		//F = Get_finite_field(Descr->sweep_field_label);

		do_sweep(false /* f_affine */,
				f,
				Descr->sweep_variables,
				verbose_level);

	}
	else if (Descr->f_sweep_affine) {

		cout << "before f_seep_affine" << endl;

		//field_theory::finite_field *F;

		//F = Get_finite_field(Descr->sweep_affine_field_label);

		do_sweep(true /* f_affine */,
				f,
				Descr->sweep_affine_variables,
				verbose_level);

	}
#endif


	if (f_v) {
		cout << "symbolic_object_activity::perform_activity done" << endl;
	}

}

void symbolic_object_activity::print(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::print" << endl;
	}


	expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	int i;

	for (i = 0; i < Vec->len; i++) {
		cout << i << " : ";
		Vec->V[i].print(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "symbolic_object_activity::print done" << endl;
	}
}

void symbolic_object_activity::latex(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::latex" << endl;
	}

	int f_latex = true;

	expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	int i;

	for (i = 0; i < Vec->len; i++) {
		cout << i << " : ";

		string s;


		s = Vec->V[i].string_representation(
				f_latex, 0 /* verbose_level*/);

		cout << s;
		cout << endl;
	}

	if (f_v) {
		cout << "symbolic_object_activity::latex done" << endl;
	}
}



void symbolic_object_activity::evaluate_affine(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::evaluate_affine" << endl;
	}


	expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	if (Vec->len != 1) {
		cout << "symbolic_object_activity::evaluate_affine len != 1" << endl;
		exit(1);
	}


	if (Vec->V[0].tree->f_has_managed_variables) {
		cout << "symbolic_object_activity::evaluate_affine "
				"Vec->V[0].tree->f_has_managed_variables" << endl;
		exit(1);
	}

	int nb_variables;
	int i;

	nb_variables = Vec->V[0].tree->variables.size();

	cout << "symbolic_object_activity::evaluate_affine "
			"nb_variables = " << nb_variables << endl;

	for (i = 0; i < nb_variables; i++) {
		cout << i << " : " << Vec->V[0].tree->variables[i] << endl;
	}


#if 0

	int f_has_managed_variables;
	std::string managed_variables_text;
	std::vector<std::string> managed_variables;

	algebra::field_theory::finite_field *Fq;

	syntax_tree_node *Root;

	std::vector<std::string> variables;
#endif

	std::map<std::string, std::string> symbol_table;
	int *Values_in;
	int *Values_out;
	int *Index_set;
	long int N;
	int q;
	long int rk;

	q = Vec->V[0].tree->Fq->q;

	geometry::other_geometry::geometry_global Geometry_global;

	N = Geometry_global.nb_AG_elements(nb_variables, q);

	Values_in = NEW_int(nb_variables);
	Values_out = NEW_int(N);
	Index_set = NEW_int(N);


	for (rk = 0; rk < N; rk++) {

		Geometry_global.AG_element_unrank(q, Values_in, 1, nb_variables, rk);

		for (i = 0; i < nb_variables; i++) {
			symbol_table[Vec->V[0].tree->variables[i]] = std::to_string(Values_in[i]);
		}


		Values_out[rk] = Vec->V[0].evaluate_with_symbol_table(
				symbol_table,
				0 /*verbose_level*/);



	}


	cout << "symbolic_object_activity::evaluate_affine Values_out:" << endl;
	Int_vec_print(cout, Values_out, N);
	cout << endl;
	cout << "N=" << N << endl;

	int sz;

	sz = 0;
	for (i = 0; i < N; i++) {
		if (Values_out[i]) {
			Index_set[sz++] = i;
		}
	}

	cout << "symbolic_object_activity::evaluate_affine Index_set:" << endl;
	Int_vec_print(cout, Index_set, sz);
	cout << endl;

	cout << "Size of index set = " << sz << endl;

	FREE_int(Values_in);
	FREE_int(Values_out);
	FREE_int(Index_set);

	if (f_v) {
		cout << "symbolic_object_activity::evaluate_affine done" << endl;
	}
}

void symbolic_object_activity::collect_monomials_binary(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary" << endl;
	}


	expression_parser::formula_vector *Formula_vector;

	Formula_vector = f->Formula_vector;


	other::data_structures::int_matrix *I;
	int *Coeff;

	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary "
				"before Formula_vector->collect_terms_and_coefficients" << endl;
	}
	Formula_vector->collect_terms_and_coefficients(
			I, Coeff,
			verbose_level);
	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary "
				"after Formula_vector->collect_terms_and_coefficients" << endl;
	}

	if (f_v) {

		cout << "monomial table:" << endl;
		int i;
		int m, n;

		m = I->m;
		n = I->n;

		for (i = 0; i < m; i++) {

			cout << i << " : ";
			Int_vec_print(cout, I->M + i * n, n);
			cout << " : " << Coeff[i] << endl;
		}

	}
	other::orbiter_kernel_system::file_io Fio;

	string fname_monomial_table;

	fname_monomial_table = Formula_vector->label_txt + "_monomial_table_binary.csv";

	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary "
				"before I->write_index_set_csv" << endl;
	}
	I->write_index_set_csv(fname_monomial_table, verbose_level);
	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary "
				"after I->write_index_set_csv" << endl;
	}

	cout << "symbolic_object_activity::collect_monomials_binary "
			"written file " << fname_monomial_table
			<< " of size " << Fio.file_size(fname_monomial_table) << endl;

	FREE_OBJECT(I);

	FREE_int(Coeff);

	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary done" << endl;
	}
}



#if 0
void symbolic_object_activity::do_sweep(
		int f_affine,
		formula *f,
		std::string &sweep_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::do_sweep" << endl;
	}


	f->print_easy(cout);
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


	n = symbol_table.size();
	v = NEW_int(n);

	N = Gg.nb_AG_elements(n, f->Fq->q);

	orbiter_kernel_system::file_io Fio;

	string fname;
	fname = "sweep_" + f->name_of_formula + "_q" + std::to_string(f->Fq->q) + ".csv";

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
	Poly->init(f->Fq,
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

		N_points = NT.i_power_j(f->Fq->q, Poly->nb_variables - 1);
	}
	else {
		geometry::geometry_global Gg;

		N_points = Gg.nb_PG_elements(Poly->nb_variables - 1, f->Fq->q);
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

			Gg.AG_element_unrank(f->Fq->q, v, 1, n, i);

			if (f_affine) {
				if (v[0] == 0) {
					continue;
				}
			}

			values.assign("");
			for (j = 0; j < n; j++) {
				values += symbol_table[j] + "=" + std::to_string(v[j]);
				if (j < n - 1) {
					values += ",";
				}
			}

			int *Values;
			int nb_monomials;

			ED.evaluate_managed_formula(
					f,
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
		cout << "symbolic_object_activity::do_sweep done" << endl;
	}

}
#endif

}}}}

