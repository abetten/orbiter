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


		int i;
		for (i = 0; i < len; i++) {

			string label;
			label = "V" + std::to_string(i);

			if (f_v) {
				cout << "symbol_definition::read_definition "
						"before Formula->init_formula_Sajeeb" << endl;
			}

			V[i].init_formula_Sajeeb(
					label, label,
					managed_variables, input[i],
					verbose_level);

		}

		if (f_v) {
			cout << "symbol_definition::read_definition "
					"after Formula->init_formula_Sajeeb" << endl;
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
			cout << "symbolic_object_builder::init the object is not of type matrix" << endl;
			exit(1);
		}

		int n;

		n = O1->nb_rows;

		if (O1->nb_rows != O1->nb_cols) {
			cout << "symbolic_object_builder::init the matrix is not square" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "symbolic_object_builder::init we found a square matrix of size " << n << endl;
		}

		int *I;
		int N;
		int a;
		geometry::geometry_global GG;
		expression_parser::formula **terms;


		N = 1 << n;

		I = NEW_int(n);
		for (a = 0; a < N; a++) {

			GG.AG_element_unrank(
					2 /* q */, I, 1, n, a);

			terms = (expression_parser::formula **) new pvoid [n];


			int i;
			for (i = 0; i < n; i++) {


				terms[i] = &O1->V[i * nb_cols + I[i]];

			}

			multiply_terms(terms, n, verbose_level);


			delete [] terms;


		}

		FREE_int(I);

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

		print(cout);

	}


	if (f_v) {
		cout << "symbolic_object_builder::init done" << endl;
	}
}


void symbolic_object_builder::print(std::ostream &ost)
{

	if (f_matrix) {
		ost << "symbolic matrix of size "
				<< nb_rows << " x " << nb_cols << endl;
		int i, j;
		vector<string> S;
		int *W;
		int l;

		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				string s;

				s = V[i * nb_cols + j].string_representation();
				S.push_back(s);
			}
		}
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

		//Lint_matrix_print(v, k, len / k);
	}
}


void symbolic_object_builder::multiply_terms(
		expression_parser::formula **terms,
		int n,
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
		cout << "symbolic_object_builder::multiply_terms before E.multiply" << endl;
	}
	E.multiply(
			(l1_interfaces::expression_parser_sajeeb **) Terms,
			n,
			verbose_level);
	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms after E.multiply" << endl;
	}



	delete [] Terms;

	if (f_v) {
		cout << "symbolic_object_builder::multiply_terms done" << endl;
	}
}



}}}



