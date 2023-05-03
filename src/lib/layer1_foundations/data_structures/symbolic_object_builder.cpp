/*
 * symbolic_object_builder.cpp
 *
 *  Created on: Apr 7, 2023
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {


symbolic_object_builder::symbolic_object_builder()
{
	Descr = NULL;

	Fq = NULL;

	Ring = NULL;

	V = NULL;
	len = 0;

	f_matrix = false;
	nb_rows = 0;
	nb_cols = 0;
}

symbolic_object_builder::~symbolic_object_builder()
{
	if (V) {
		FREE_OBJECTS(V);
		V = NULL;
	}
}

void symbolic_object_builder::init(
		symbolic_object_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::init" << endl;
	}

	symbolic_object_builder::Descr = Descr;
	string managed_variables;


	if (Descr->f_field) {
		if (f_v) {
			cout << "symbolic_object_builder::init -field " << Descr->field_label << endl;
		}
		Fq = Get_finite_field(Descr->field_label);
	}

	if (Descr->f_ring) {
		if (f_v) {
			cout << "symbolic_object_builder::init -ring " << Descr->ring_label << endl;
		}
		Ring = Get_ring(Descr->ring_label);
		int i;
		for (i = 0; i < Ring->nb_variables; i++) {
			managed_variables += Ring->get_symbol(i);
			if (i < Ring->nb_variables - 1) {
				managed_variables += ",";
			}
		}
	}

	if (Descr->f_ring && !Descr->f_field) {
		Fq = Ring->get_F();
	}

	if (!Descr->f_ring && !Descr->f_field) {
		cout << "symbolic_object_builder::init please use either "
				"-ring <ring> or -field <field> "
				"to specify the domain of coefficients" << endl;
		exit(1);
	}

	if (Descr->f_text) {
		if (f_v) {
			cout << "symbolic_object_builder::init -text" << Descr->text_txt << endl;
		}


		data_structures::string_tools ST;
		std::vector<std::string> input;

		ST.parse_comma_separated_list(
				Descr->text_txt, input,
				verbose_level);


		len = input.size();

		V = NEW_OBJECTS(expression_parser::formula, len);

		if (f_v) {
			cout << "symbol_definition::read_definition "
					"before loop" << endl;
		}

		int i;
		for (i = 0; i < len; i++) {

			string label;
			label = "V" + std::to_string(i);

			if (f_v) {
				cout << "symbol_definition::read_definition "
						"before V[i].init_formula_Sajeeb" << endl;
			}

			V[i].init_formula_Sajeeb(
					label, label,
					managed_variables, input[i],
					Fq,
					verbose_level);

			if (f_v) {
				cout << "symbol_definition::read_definition "
						"after V[i].init_formula_Sajeeb" << endl;
			}

			if (f_v) {
				cout << "symbol_definition::read_definition "
						"before V[i].export_graphviz" << endl;
			}

			V[i].export_graphviz(
					label);

			if (f_v) {
				cout << "symbol_definition::read_definition "
						"after V[i].export_graphviz" << endl;
			}

		}

		if (f_v) {
			cout << "symbol_definition::read_definition "
					"after loop" << endl;
		}

		if (Descr->f_matrix) {
			f_matrix = true;
			nb_rows = Descr->nb_rows;
			nb_cols = len / Descr->nb_rows;
		}

	}
	else if (Descr->f_test) {
		if (f_v) {
			cout << "symbolic_object_builder::init -test" << Descr->test_object1 << endl;
		}

		data_structures::symbolic_object_builder *O1;

		O1 = Get_symbol(Descr->test_object1);

		if (!O1->f_matrix) {
			cout << "symbolic_object_builder::init "
					"the object is not of type matrix" << endl;
			exit(1);
		}

		int n;

		n = O1->nb_rows;

		if (O1->nb_rows != O1->nb_cols) {
			cout << "symbolic_object_builder::init the matrix is not square" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "symbolic_object_builder::init "
					"we found a square matrix of size " << n << endl;
		}

		int N;
		int a;
		geometry::geometry_global GG;
		expression_parser::formula **terms;
		int stage_counter = 0;

		combinatorics::combinatorics_domain Combi;
		ring_theory::longinteger_domain Long;
		ring_theory::longinteger_object result;
		int *lehmer_code;
		int *perm;

		lehmer_code = NEW_int(n);
		perm = NEW_int(n);

		Long.factorial(result, n);
		N = result.as_lint();

		if (f_v) {
			cout << "symbolic_object_builder::init N = " << N << endl;
		}

		for (a = 0; a < N; a++) {

			if (a == 0) {
				Combi.first_lehmercode(n, lehmer_code);
			}
			else {
				Combi.next_lehmercode(n, lehmer_code);
			}
			Combi.lehmercode_to_permutation(
					n, lehmer_code, perm);


			if (f_v) {
				cout << "symbolic_object_builder::init a = " << a << " / " << N
						<< " stage_counter=" << stage_counter << " perm=";
				Int_vec_print(cout, perm, n);
				cout << endl;
			}



			terms = (expression_parser::formula **) new pvoid [n];


			int i;
			for (i = 0; i < n; i++) {


				terms[i] = &O1->V[i * nb_cols + perm[i]];

			}

			if (f_v) {
				cout << "symbolic_object_builder::init a = " << a << " / " << N
						<< " before multiply_terms" << endl;
			}
			multiply_terms(terms, n, stage_counter, verbose_level);
			if (f_v) {
				cout << "symbolic_object_builder::init a = " << a << " / " << N
						<< " after multiply_terms" << endl;
			}

			if (f_v) {
				cout << "symbolic_object_builder::init before delete [] terms" << endl;
			}

			delete [] terms;

			if (f_v) {
				cout << "symbolic_object_builder::init after delete [] terms" << endl;
			}

		}

		if (f_v) {
			cout << "symbolic_object_builder::init before FREE_int" << endl;
		}
		FREE_int(perm);
		FREE_int(lehmer_code);
		if (f_v) {
			cout << "symbolic_object_builder::init after FREE_int" << endl;
		}

	}

	if (Descr->f_file) {
		if (f_v) {
			cout << "symbolic_object_builder::init -file" << Descr->file_name << endl;
		}

		//V = NEW_OBJECT(expression_parser::formula);

	}




	if (f_v) {
		cout << "symbolic_object_builder::init "
				"symbolic object vector of length " << len << endl;
		//Lint_vec_print(cout, v, len);
		cout << endl;

		cout << "ir tree:" << endl;
		print_Sajeeb(cout);

		cout << "final tree:" << endl;
		print_formula(cout);

	}


	if (f_v) {
		cout << "symbolic_object_builder::init done" << endl;
	}
}

void symbolic_object_builder::get_string_representation_Sajeeb(std::vector<std::string> &S)
{
	int i, j;

	if (f_matrix) {
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				string s;

				s = V[i * nb_cols + j].string_representation_Sajeeb();
				S.push_back(s);
			}
		}
	}
	else {
		int i;
		for (i = 0; i < len; i++) {
			string s;

			s = V[i].string_representation_Sajeeb();
			S.push_back(s);
		}

	}
}

void symbolic_object_builder::get_string_representation_formula(std::vector<std::string> &S)
{
	int i, j;

	if (f_matrix) {
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				string s;

				s = V[i * nb_cols + j].string_representation_formula();
				S.push_back(s);
			}
		}
	}
	else {
		int i;
		for (i = 0; i < len; i++) {
			string s;

			s = V[i].string_representation_formula();
			S.push_back(s);
		}
	}
}

void symbolic_object_builder::print_Sajeeb(std::ostream &ost)
{

	if (f_matrix) {
		ost << "symbolic matrix of size "
				<< nb_rows << " x " << nb_cols << endl;
		vector<string> S;

		get_string_representation_Sajeeb(S);
		//get_string_representation_formula(S);

		print_matrix(S, ost);
	}
	else {
		ost << "symbolic vector of size "
				<< len << endl;
		vector<string> S;

		get_string_representation_Sajeeb(S);
		//get_string_representation_formula(S);

		print_vector(S, ost);

	}
}

void symbolic_object_builder::print_formula(std::ostream &ost)
{

	if (f_matrix) {
		ost << "symbolic matrix of size "
				<< nb_rows << " x " << nb_cols << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S);

		print_matrix(S, ost);
	}
	else {
		ost << "symbolic vector of size "
				<< len << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S);

		print_vector(S, ost);
	}
}


void symbolic_object_builder::print_matrix(
		std::vector<std::string> &S, std::ostream &ost)
{
	int i, j;
	int *W;
	int l;

	W = NEW_int(nb_cols);
	for (j = 0; j < nb_cols; j++) {
		W[j] = 0;
		for (i = 0; i < nb_rows; i++) {
			l = S[i * nb_cols + j].length();
			W[j] = MAXIMUM(W[j], l);
		}
	}

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			ost << setw(W[j]) << S[i * nb_cols + j];
			if (j < nb_cols - 1) {
				ost << ", ";
			}
		}
		ost << endl;
	}

}

void symbolic_object_builder::print_vector(
		std::vector<std::string> &S, std::ostream &ost)
{
	int i, j;
	int W;
	int l;

	W = 0;
	for (i = 0; i < len; i++) {
		l = S[i].length();
		W = MAXIMUM(W, l);
	}

	for (i = 0; i < len; i++) {
		ost << setw(3) << i << " : " << setw(W) << S[i];
		ost << endl;
	}

}


void symbolic_object_builder::multiply_terms(
		expression_parser::formula **terms,
		int n,
		int &stage_counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms" << endl;
	}

	void **Terms;


	Terms = new pvoid [n];

	int i;

	for (i = 0; i < n; i++) {
		if (!terms[i]->f_Sajeeb) {
			cout << "The formula must be of type Sajeeb" << endl;
			exit(1);
		}
		Terms[i] = terms[i]->Expression_parser_sajeeb;
	}

	l1_interfaces::expression_parser_sajeeb E;

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"before E.multiply" << endl;
	}
	E.multiply(
			(l1_interfaces::expression_parser_sajeeb **) Terms,
			n,
			stage_counter,
			verbose_level);
	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"after E.multiply" << endl;
	}


	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"before delete [] Terms" << endl;
	}

	delete [] Terms;

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms "
				"after delete [] Terms" << endl;
	}

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms done" << endl;
	}
}



}}}



