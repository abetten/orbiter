/*
 * symbolic_object_activity.cpp
 *
 *  Created on: Feb 4, 2022
 *      Author: betten
 */





#include "orbiter_user_interface.h"

using namespace std;


namespace orbiter {
namespace layer6_user_interface {
namespace activities_layer1 {


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
		layer6_user_interface::activities_layer1::symbolic_object_activity_description *Descr,
		algebra::expression_parser::symbolic_object_builder *f,
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
	else if (Descr->f_save) {

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_save" << endl;
		}

		string fname;

		fname = f->label + ".txt";
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"fname = " << fname << endl;
		}

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before save" << endl;
		}
		save(fname, verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after save" << endl;
		}

	}
	else if (Descr->f_as_vector) {

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_as_vector" << endl;
		}

		int *v;
		int len;

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before as_vector" << endl;
		}
		as_vector(
				v, len,
				verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after as_vector" << endl;
		}

		cout << "vector of length " << len << " : ";
		Int_vec_print(cout, v, len);
		cout << endl;

		FREE_int(v);

	}
	else if (Descr->f_homogenize) {

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity f_homogenize" << endl;
		}

		int *v;
		int len;

		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"before as_vector" << endl;
		}
		as_vector(
				v, len,
				verbose_level);
		if (f_v) {
			cout << "symbolic_object_activity::perform_activity "
					"after as_vector" << endl;
		}

		cout << "vector of length " << len << " : ";
		Int_vec_print(cout, v, len);
		cout << endl;

		other::data_structures::algorithms Algorithms;

		Algorithms.print_homogenized(
				v, len,
				cout, verbose_level);


		FREE_int(v);

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


	algebra::expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	int i, j;
	int f_latex = false;

	for (i = 0; i < Vec->len; i++) {
		cout << i << " : ";

		std::vector<std::string> rep;
		string s;

		Vec->V[i].print_to_vector(
				rep, f_latex, 0 /*verbose_level */);

		if (rep.size() == 0) {
			string zero;

			zero = "0";
			rep.push_back(zero);
		}

		for (j = 0; j < rep.size(); j++) {
			s += rep[j];
		}

		cout << s;

		//Vec->V[i].print(cout);
		cout << endl;
	}

	if (f_v) {
		cout << "symbolic_object_activity::print done" << endl;
	}
}

void symbolic_object_activity::save(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::save" << endl;
	}


	algebra::expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	Vec->save_ascii(
			fname, verbose_level);



	if (f_v) {
		cout << "symbolic_object_activity::save done" << endl;
	}
}




void symbolic_object_activity::as_vector(
		int *&v, int &len,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::as_vector" << endl;
	}

	std::vector<std::string> String_rep;

	stringify(
			String_rep,
			0 /* verbose_level */);

	len = String_rep.size();

	int i;

	v = NEW_int(len);

	for (i = 0; i < len; i++) {

		v[i] = std::stoi(String_rep[i]);

	}


	if (f_v) {
		cout << "symbolic_object_activity::as_vector done" << endl;
	}
}




void symbolic_object_activity::stringify(
		std::vector<std::string> &String_rep,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::stringify" << endl;
	}

	algebra::expression_parser::formula_vector *Vec;

	Vec = f->Formula_vector;

	int i, j;
	int f_latex = false;

	for (i = 0; i < Vec->len; i++) {
		//cout << i << " : ";

		std::vector<std::string> rep;
		string s;

		Vec->V[i].print_to_vector(
				rep, f_latex, 0 /*verbose_level */);

		if (rep.size() == 0) {
			string zero;

			zero = "0";
			rep.push_back(zero);
		}

		for (j = 0; j < rep.size(); j++) {
			s += rep[j];
		}

		String_rep.push_back(s);

		//cout << s;

		//Vec->V[i].print(cout);
		//cout << endl;
	}

	if (f_v) {
		cout << "symbolic_object_activity::stringify done" << endl;
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

	algebra::expression_parser::formula_vector *Vec;

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


	algebra::expression_parser::formula_vector *Vec;

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



	if (f_v) {
		cout << "symbolic_object_activity::evaluate_affine nb_variables=" << nb_variables << endl;
		cout << "symbolic_object_activity::evaluate_affine q=" << q << endl;
		cout << "symbolic_object_activity::evaluate_affine N=" << N << endl;

		cout << "symbolic_object_activity::evaluate_affine Values_out:" << endl;
		Int_vec_print(cout, Values_out, N);
		cout << endl;
	}



	string fname;

	fname = f->label + "_evaluation.csv";

	other::orbiter_kernel_system::file_io Fio;

	std::string col_heading;

	col_heading = "evaluation";

	Fio.Csv_file_support->int_matrix_write_csv_tabulated(
			fname, col_heading,
			Values_out, 1, N, verbose_level);

	if (f_v) {
		cout << "symbolic_object_activity::evaluate_affine "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}






	if (true) {

		other::data_structures::algorithms Algorithms;

		Algorithms.content_analysis_by_Hamming_weight(
				Values_out, N,
				verbose_level);


	}

	// compute the true set of Values_out, viewed as a boolean function:

	int sz;

	sz = 0;
	for (i = 0; i < N; i++) {
		if (Values_out[i]) {
			Index_set[sz++] = i;
		}
	}

	cout << "symbolic_object_activity::evaluate_affine true set = " << endl;
	Int_vec_print(cout, Index_set, sz);
	cout << endl;

	cout << "Size of true set = " << sz << endl;

	FREE_int(Values_in);
	FREE_int(Values_out);
	FREE_int(Index_set);

	if (f_v) {
		cout << "symbolic_object_activity::evaluate_affine done" << endl;
	}
}

void symbolic_object_activity::collect_monomials_binary(
		int verbose_level)
// writes the monomial table to a csv file.
// The filename is composed of the label and the suffix "_monomial_table_binary.csv"
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_activity::collect_monomials_binary" << endl;
	}


	algebra::expression_parser::formula_vector *Formula_vector;

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




}}}


