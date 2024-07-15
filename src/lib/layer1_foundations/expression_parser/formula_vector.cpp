/*
 * formula_vector.cpp
 *
 *  Created on: May 10, 2023
 *      Author: betten
 */




#include "foundations.h"



using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {




formula_vector::formula_vector()
{
	//std::string label_txt;
	//std::string label_tex;

	f_has_managed_variables = false;
	//std::string managed_variables_text;

	V = NULL;
	len = 0;

	f_matrix = false;
	nb_rows = 0;
	nb_cols = 0;
}



formula_vector::~formula_vector()
{
	if (V) {
		FREE_OBJECTS(V);
		V = NULL;
	}
}

void formula_vector::init_from_text(
		std::string &label_txt,
		std::string &label_tex,
		std::string &text,
		field_theory::finite_field *Fq,
		int f_managed_variables,
		std::string &managed_variables_text,
		int f_matrix, int nb_rows,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::init_from_text" << endl;
	}

	formula_vector::label_txt = label_txt;
	formula_vector::label_tex = label_tex;

	if (f_managed_variables) {
		f_has_managed_variables = true;
		formula_vector::managed_variables_text = managed_variables_text;
		if (f_v) {
			cout << "formula_vector::init_from_text "
					"managed_variables_text = "
					<< managed_variables_text << endl;
		}
	}
	else {
		f_has_managed_variables = false;
		if (f_v) {
			cout << "formula_vector::init_from_text "
					"no managed variables" << endl;
		}
	}

	data_structures::string_tools ST;
	std::vector<std::string> input;

	ST.parse_comma_separated_list(
			text, input,
			verbose_level);

	if (f_v) {
		cout << "formula_vector::init_from_text "
				"the input has size " << input.size() << endl;
		cout << "formula_vector::init_from_text "
				"the input is:" << endl;
		int i;
		for (i = 0; i < input.size(); i++) {
			cout << i << " : " << input[i] << endl;
		}
	}

	init_and_allocate(
			label_txt,
			label_tex,
			f_managed_variables,
			managed_variables_text,
			input.size(), verbose_level);

	if (f_v) {
		cout << "formula_vector::init_from_text "
				"before loop" << endl;
	}

	int i;
	for (i = 0; i < len; i++) {

		string element_label_txt;
		string element_label_tex;
		element_label_txt = label_txt + "_V" + std::to_string(i);
		element_label_tex = label_tex + "V" + std::to_string(i);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"before V[i].init_formula_Sajeeb" << endl;
		}
		if (f_v) {
			cout << "formula_vector::init_from_text "
					<< i << " / " << len << " input = " << input[i] << endl;
		}

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"before V[i].init_formula_Sajeeb" << endl;
		}

		V[i].init_formula_Sajeeb(
				element_label_txt, element_label_tex,
				f_managed_variables,
				managed_variables_text,
				input[i],
				Fq,
				verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"after V[i].init_formula_Sajeeb" << endl;
		}

		string s;

		s = V[i].string_representation_formula(false, verbose_level);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"formula " << i << " / " << len << " is: " << s << endl;
		}

#if 0
		if (f_v) {
			cout << "formula_vector::init_from_text "
					"before V[i].export_graphviz" << endl;
		}

		V[i].export_graphviz(
				label_txt);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"after V[i].export_graphviz" << endl;
		}
#endif

	}

	if (f_v) {
		cout << "formula_vector::init_from_text "
				"after loop" << endl;
	}

	if (f_matrix) {
		formula_vector::f_matrix = true;
		formula_vector::nb_rows = nb_rows;
		formula_vector::nb_cols = len / nb_rows;
	}
	if (f_v) {
		cout << "formula_vector::init_from_text done" << endl;
	}

}

void formula_vector::init_and_allocate(
		std::string &label_txt, std::string &label_tex,
		int f_has_managed_variables,
		std::string managed_variables_text,
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::init_and_allocate" << endl;
	}
	formula_vector::label_txt = label_txt;
	formula_vector::label_tex = label_tex;

	formula_vector::f_has_managed_variables = f_has_managed_variables;
	formula_vector::managed_variables_text = managed_variables_text;
	if (f_v) {
		cout << "formula_vector::init_and_allocate" << endl;
		if (f_has_managed_variables) {
			cout << "formula_vector::init_and_allocate "
					"managed_variables_text=" << managed_variables_text << endl;
		}
		else {
			cout << "formula_vector::init_and_allocate no managed variables" << endl;

		}
	}


	V = NEW_OBJECTS(formula, len);
	formula_vector::len = len;

	int i;

	for (i = 0; i < len; i++) {
		if (f_has_managed_variables) {
			V[i].f_has_managed_variables = f_has_managed_variables;
			V[i].managed_variables = managed_variables_text;
		}
		V[i].name_of_formula = label_txt + "_" + std::to_string(i);
		V[i].name_of_formula_latex = label_tex + "_{" + std::to_string(i) + "}";
	}

	if (f_v) {
		cout << "formula_vector::init_and_allocate done" << endl;
	}
}

int formula_vector::is_integer_matrix()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::is_integer_matrix" << endl;
	}

	if (!f_matrix) {
		cout << "formula_vector::is_integer_matrix "
				"the object is not a matrix" << endl;
		exit(1);
	}
	int i, j;

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			if (f_v) {
				cout << "reading matrix entry " << i << "," << j << endl;
			}
			if (!V[i * nb_cols + j].tree->Root->f_terminal) {
				if (f_v) {
					cout << "entry (" << i << "," << j << ") "
						"is not a terminal node" << endl;
				}
				return false;
			}
			if (!V[i * nb_cols + j].tree->Root->T->f_int) {
				if (f_v) {
					cout << "entry (" << i << "," << j << ") "
						"is not an integer terminal node" << endl;
				}
				return false;
			}
		}
	}
	if (f_v) {
		cout << "formula_vector::is_integer_matrix done" << endl;
	}
	return true;

}

void formula_vector::get_integer_matrix(
		int *&M, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::get_integer_matrix" << endl;
	}

	int i, j, a;

	M = NEW_int(nb_rows * nb_cols);

	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			a = V[i * nb_cols + j].tree->Root->T->value_int;
			M[i * nb_cols + j] = a;
		}
	}

	if (f_v) {
		cout << "formula_vector::get_integer_matrix done" << endl;
	}
}

void formula_vector::get_string_representation_Sajeeb(
		std::vector<std::string> &S)
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

void formula_vector::get_string_representation_formula(
		std::vector<std::string> &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::get_string_representation_formula" << endl;
	}
	int i, j;

	if (f_matrix) {
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				string s;

				if (f_v) {
					cout << "formula_vector::get_string_representation_formula "
							"i=" << i << " j=" << j << endl;
				}
				s = V[i * nb_cols + j].string_representation_formula(false, verbose_level);
				if (f_v) {
					cout << "formula_vector::get_string_representation_formula "
							"s=" << s << endl;
				}
				S.push_back(s);
			}
		}
	}
	else {
		int i;
		for (i = 0; i < len; i++) {
			string s;

			s = V[i].string_representation_formula(false, verbose_level);
			S.push_back(s);
		}
	}
}


void formula_vector::print_Sajeeb(
		std::ostream &ost)
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

void formula_vector::print_formula_size(
		std::ostream &ost, int verbose_level)
{
	int i;
	long int *Nb;
	long int total;


	Nb = NEW_lint(len);

	total = 0;
	for (i = 0; i < len; i++) {


		Nb[i] = V[i].tree->nb_nodes_total();
		total += Nb[i];
	}

	cout << "Number of nodes: ";
	Lint_vec_print(cout, Nb, len);
	cout << endl;
	cout << "total: " << total << endl;
}

void formula_vector::print_formula(
		std::ostream &ost, int verbose_level)
{

	data_structures::string_tools String;

	if (f_matrix) {
		ost << "symbolic matrix of size "
				<< nb_rows << " x " << nb_cols << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S, 0 /*verbose_level*/);

		print_matrix(S, ost);
		ost << endl;


#if 0
		String.make_latex_friendly_vector(
					S, 0 /*verbose_level*/);

		ost << "latex friendly matrix:" << endl;
		print_matrix(S, ost);
		ost << endl;

		ost << "number of nodes:";
		int i, j, nb;
		vector<string> S2;


		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				nb = V[i * nb_cols + j].tree->nb_nodes_total();
				string s;

				s = std::to_string(nb);
				S2.push_back(s);
			}
		}

		print_matrix(S2, ost);
		ost << endl;
#endif

	}
	else {
		ost << "symbolic vector of size "
				<< len << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S, 0 /*verbose_level*/);

		print_vector(S, ost);

		ost << endl;

#if 0
		String.make_latex_friendly_vector(
					S, 0 /*verbose_level*/);

		ost << "latex friendly vector of length " << len << ":" << endl;

		if (len > 20) {
			ost << "too large to print" << endl;
		}
		else {
			print_vector_latex(S, ost);

			ost << endl;
		}

		int h;
		for (h = 0; h < len; h++) {

			cout << "entry " << h << " / " << len << ":" << endl;
			data_structures::int_matrix *I;
			int *Coeff;

			V[h].collect_monomial_terms(
					I, Coeff,
					verbose_level);

			if (I == NULL) {

				cout << "nothing" << endl;
			}
			else {

				int i;

				for (i = 0; i < I->m; i++) {
					cout << Coeff[i] << " : ";
					Int_vec_print(cout, I->M + i * I->n, I->n);
					cout << endl;
				}

				FREE_OBJECT(I);
				FREE_int(Coeff);
			}

			cout << endl;

		}
#endif


	}


	ost << "number of nodes:" << endl;
	int i, nb;
	vector<string> S2;


	for (i = 0; i < len; i++) {


		nb = V[i].tree->nb_nodes_total();


		string s;

		s = std::to_string(nb);
		S2.push_back(s);

		if (i < len - 1) {
			s = ",";
			S2.push_back(s);
		}
	}

	print_vector(S2, ost);
	ost << endl;

	ost << "number of nodes in detail:" << endl;
	vector<string> S3;


	for (i = 0; i < len; i++) {


		nb = V[i].tree->nb_nodes_total();

		int nb_add, nb_mult, nb_int, nb_text, max_degree;

		V[i].tree->count_nodes(
				nb_add, nb_mult, nb_int, nb_text,
				max_degree);

		string s;

		s = std::to_string(nb) + "=" +
				std::to_string(nb_add) + "," +
				std::to_string(nb_mult) + ";" +
				std::to_string(nb_int) + "," +
				std::to_string(nb_text) + ";" +
				std::to_string(max_degree);
		S3.push_back(s);
	}

	print_vector(S3, ost);
	ost << endl;

}


void formula_vector::print_matrix(
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

	print_variables(ost);

}

void formula_vector::print_vector(
		std::vector<std::string> &S, std::ostream &ost)
{
	int i;
	int W;
	int l;

	W = 0;
	for (i = 0; i < len; i++) {
		l = S[i].length();
		W = MAXIMUM(W, l);
	}

	for (i = 0; i < len; i++) {
		ost << setw(3) << i << " : " << S[i] << endl;
	}

	print_variables(ost);

}

void formula_vector::print_vector_latex(
		std::vector<std::string> &S, std::ostream &ost)
{
	int i;
	int W;
	int l;

	W = 0;
	for (i = 0; i < len; i++) {
		l = S[i].length();
		W = MAXIMUM(W, l);
	}

	ost << "{\\tt" << endl;
	for (i = 0; i < len; i++) {
		ost << setw(3) << i << " : \\\\" << endl;
		ost << S[i];
		ost << "\\\\" << endl;
	}
	ost << "}" << endl;

	print_variables(ost);

}


void formula_vector::print_latex(
		std::ostream &ost, std::string &label)
{
	int i;
	vector<string> v;

	for (i = 0; i < len; i++) {
		string s;

		s = V[i].string_representation_formula(true, 0 /*verbose_level*/);
		v.push_back(s);
	}
	if (f_matrix) {
		int j;

		ost << "$$" << endl;
		ost << label << " = " << endl;
		ost << "\\left[" << endl;
		ost << "\\begin{array}{*{" << nb_cols << "}c}" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {

				if (v[i * nb_cols + j].length() < 1000) {
					ost << v[i * nb_cols + j];
				}
				else {
					ost << "formula_vector::print_latex "
							"too large to print v[i * nb_cols + j].length() = " << v[i].length();
				}
				if (j < nb_cols - 1) {
					ost << "  & ";
				}
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		ost << "$$" << endl;

	}
	else {

		if (len > 1) {

			l1_interfaces::latex_interface L;
			int f_brackets = true;
			int f_enter_math_mode = true;

			L.print_vector_vertically_with_label(
					ost,
					label,
					v,
					f_brackets,
					f_enter_math_mode);

			L.print_vector_horizontally_with_label(
					ost,
					label,
					v,
					f_brackets,
					f_enter_math_mode);

#if 0
			ost << "$$" << endl;
			ost << label << " = " << endl;
			ost << "\\left[" << endl;
			ost << "\\begin{array}{*{" << 1 << "}c}" << endl;
			for (i = 0; i < len; i++) {
				if (v[i].length() < 100) {
					ost << v[i];
				}
				else {
					ost << "formula_vector::print_latex "
							"too large to print v[i].length() = " << v[i].length();
				}
				ost << "\\\\" << endl;
			}
			ost << "\\end{array}" << endl;
			ost << "\\right]" << endl;
			ost << "$$" << endl;
#endif

		}
		else {

			l1_interfaces::latex_interface L;
			int f_brackets = false;
			int f_enter_math_mode = true;

			L.print_vector_vertically_with_label(
					ost,
					label,
					v,
					f_brackets,
					f_enter_math_mode);

			L.print_vector_horizontally_with_label(
					ost,
					label,
					v,
					f_brackets,
					f_enter_math_mode);


#if 0
			ost << "$$" << endl;
			//ost << "\\left[" << endl;
			ost << label << " = " << endl;
			ost << "\\begin{array}{*{" << 1 << "}c}" << endl;
			for (i = 0; i < len; i++) {
				if (v[i].length() < 100) {
					ost << v[i];
				}
				else {
					ost << "formula_vector::print_latex "
							"too large to print v[i].length() = " << v[i].length();
				}
				ost << "\\\\" << endl;
				}
			ost << "\\end{array}" << endl;
			//ost << "\\right]" << endl;
			ost << "$$" << endl;
#endif

		}
	}

	//print_variables(ost);
}

void formula_vector::make_A_minus_lambda_Identity(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &variable,
		std::string &label_txt,
		std::string &label_tex,
		int f_managed_variables,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::make_A_minus_lambda_Identity" << endl;
	}


	if (!A->f_matrix) {
		cout << "formula_vector::make_A_minus_lambda_Identity "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int n;

	n = A->nb_rows;

	if (A->nb_rows != A->nb_cols) {
		cout << "formula_vector::make_A_minus_lambda_Identity "
				"the matrix is not square" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "formula_vector::make_A_minus_lambda_Identity "
				"we found a square matrix of size " << n << endl;
	}

	int n2;
	int i, j;
	int minus_one;

	minus_one = Fq->negate(1);

	n2 = n * n;

	init_and_allocate(
			label_txt,
			label_tex,
			f_managed_variables,
			managed_variables_text,
			n2, verbose_level);


	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {

			// copy the matrix entry:

			A->V[i * n + j].copy_to(
					&V[i * n + j],
					verbose_level);


			if (i == j) {

				// subtract lambda on the diagonal:


				syntax_tree_node *node;
				syntax_tree_node *add_node;
				syntax_tree_node *mult_node;
				syntax_tree_node *lambda_node;
				syntax_tree *Tree;

				Tree = A->V[i * n + j].tree;


				// remember the old root node:

				node = V[i * n + j].tree->Root;


				// Create a new addition node:

				add_node = NEW_OBJECT(syntax_tree_node);

				add_node->init_empty_plus_node_with_exponent(
						Tree,
						1 /* exponent */,
						verbose_level);

				// The old node becomes the first summand in the addition:

				add_node->Nodes[add_node->nb_nodes++] = node;

				// the second summand will be a multiplication node
				// to contain (-1) * lambda:

				add_node->add_empty_multiplication_node(Tree, verbose_level);

				mult_node = add_node->Nodes[add_node->nb_nodes - 1];

				// Insert a factor of minus one:
				mult_node->add_numerical_factor(
							minus_one, verbose_level);

				// Insert a factor of lambda.
				// This will be a terminal node of type text:

				mult_node->add_empty_node(Tree, verbose_level);
				lambda_node = mult_node->Nodes[mult_node->nb_nodes - 1];
				lambda_node->init_terminal_node_text(
							Tree,
							variable,
							verbose_level);


				// set the root node to be the new add node:

				V[i * n + j].tree->Root = add_node;
			}

		}
	}

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::make_A_minus_lambda_Identity done" << endl;
	}
}


void formula_vector::substitute(
		formula_vector *Source,
		formula_vector *Target,
		std::string &substitution_variables,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::substitute" << endl;
	}
	data_structures::string_tools ST;
	std::vector<std::string> variables;

	ST.parse_comma_separated_list(
			substitution_variables, variables,
			verbose_level - 2);

	int N;
	N = variables.size();
	if (f_v) {
		int i;
		for (i = 0; i < N; i++) {
			cout << setw(3) << i << " : " << variables[i] << endl;
		}
	}
	int t, s, i;
	int nb, len;

	if (f_v) {
		cout << "formula_vector::substitute "
				"Source->len=" << Source->len << endl;
		cout << "formula_vector::substitute N=" << N << endl;
	}


	nb = Source->len / N;

	if (f_v) {
		cout << "formula_vector::substitute nb=" << nb << endl;
	}

	if (nb == 0) {
		nb = 1;
	}

	len = nb * Target->len;


	init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables,
			len, verbose_level - 2);

	f_matrix = true;
	nb_rows = nb;
	nb_cols = Target->len;

	if (f_v) {
		cout << "formula_vector::substitute len=" << len << endl;
	}
	if (f_v) {
		cout << "formula_vector::substitute "
				"Target->len=" << Target->len << endl;
	}


	formula **S;



	S = (formula **) NEW_pvoid(N);

	if (f_v) {
		cout << "formula_vector::substitute "
				"nb = " << nb << endl;
	}


	for (s = 0; s < nb; s++) {

		if (f_v) {
			cout << "formula_vector::substitute "
					"s=" << s << " / " << nb << endl;
		}
		for (i = 0; i < N; i++) {
			S[i] = &Source->V[s * N + i];
		}

		for (t = 0; t < Target->len; t++) {

			if (f_v) {
				cout << "formula_vector::substitute "
						"s=" << s << " / " << nb
						<< " t=" << t << " / " << Target->len << endl;
			}

			formula *T;

			T = &Target->V[t];

			formula *Out;

			Out = V + s * Target->len + t;


			if (f_v) {
				cout << "formula_vector::substitute "
						"s=" << s << " / " << nb << " t=" << t << " / " << Target->len
						<< " before T->substitute" << endl;
			}

			T->substitute(
					variables, managed_variables,
					S, Out, verbose_level);

			if (f_v) {
				cout << "formula_vector::substitute "
						"s=" << s << " / " << nb << " t=" << t << " / " << Target->len
						<< " after T->substitute" << endl;
			}


		}
	}

	FREE_pvoid((void **) S);

	if (f_v) {
		cout << "formula_vector::substitute "
				"before collect_variables" << endl;
	}
	collect_variables(verbose_level);
	if (f_v) {
		cout << "formula_vector::substitute "
				"after collect_variables" << endl;
	}

	if (f_v) {
		cout << "formula_vector::substitute done" << endl;
	}

}

void formula_vector::simplify(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::simplify" << endl;
	}
	init_and_allocate(
			label_txt,
			label_tex,
			A->f_has_managed_variables,
			A->managed_variables_text,
			A->len, verbose_level);


	f_matrix = A->f_matrix;
	nb_rows = A->nb_rows;
	nb_cols = A->nb_cols;

	int i;

	for (i = 0; i < A->len; i++) {
		if (f_v) {
			cout << "formula_vector::simplify "
					"node " << i << " / " << A->len
					<< " before copy_to" << endl;
		}

		A->V[i].copy_to(
				&V[i], verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::simplify "
					"node " << i << " / " << A->len
					<< " after copy_to" << endl;
		}

		if (f_v) {
			cout << "formula_vector::simplify "
					"node " << i << " / " << A->len
					<< " before V[i].simplify" << endl;
		}

		V[i].simplify(verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::simplify "
					"node " << i << " / " << A->len
					<< " after V[i].simplify" << endl;
			cout << "formula_vector::simplify ";
			V[i].tree->Root->print_subtree_easy(cout);
			cout << endl;
		}

		if (f_v) {
			cout << "formula_vector::simplify "
					"node " << i << " / " << A->len
					<< " done" << endl;
		}
	}

	if (f_v) {
		cout << "formula_vector::simplify before collect_variables" << endl;
	}
	collect_variables(verbose_level);
	if (f_v) {
		cout << "formula_vector::simplify after collect_variables" << endl;
	}

	if (f_v) {
		cout << "formula_vector::simplify done" << endl;
	}
}

void formula_vector::expand(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		int f_write_trees,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::expand" << endl;
	}
	init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables,
			A->len, verbose_level);


	f_matrix = A->f_matrix;
	nb_rows = A->nb_rows;
	nb_cols = A->nb_cols;

	int i;

	for (i = 0; i < A->len; i++) {
		if (f_v) {
			cout << "formula_vector::expand "
					"node " << i << " / " << A->len
					<< " before copy_to" << endl;
		}

		A->V[i].copy_to(
				&V[i], verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::expand "
					"node " << i << " / " << A->len
					<< " after copy_to" << endl;
		}

		if (f_v) {
			cout << "formula_vector::expand "
					"node " << i << " / " << A->len
					<< " before V[i].expand_in_place" << endl;
		}

		V[i].expand_in_place(f_write_trees, verbose_level);

		if (f_v) {
			cout << "formula_vector::expand "
					"node " << i << " / " << A->len
					<< " after V[i].expand_in_place" << endl;
		}

		if (f_v) {
			cout << "formula_vector::expand "
					"node " << i << " / " << A->len
					<< " done" << endl;
		}
	}

	if (f_v) {
		cout << "formula_vector::expand "
				"before collect_variables" << endl;
	}
	collect_variables(verbose_level);
	if (f_v) {
		cout << "formula_vector::expand "
				"after collect_variables" << endl;
	}

	if (f_v) {
		cout << "formula_vector::expand done" << endl;
	}

}

void formula_vector::characteristic_polynomial(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &variable,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial" << endl;
	}
	formula_vector A_minus_lambda_Id;

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"before A_minus_lambda_Id.make_A_minus_lambda_Identity" << endl;
	}
	A_minus_lambda_Id.make_A_minus_lambda_Identity(
			A,
			Fq,
			variable,
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"after A_minus_lambda_Id.make_A_minus_lambda_Identity" << endl;
	}


	int n;

	n = A->nb_rows;


	syntax_tree *Tree;

	Tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"before Tree->init" << endl;
	}
	Tree->init(Fq, f_has_managed_variables, managed_variables_text, verbose_level);
	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"after Tree->init" << endl;
	}

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"before Tree->make_determinant" << endl;
	}
	Tree->make_determinant(
			Fq,
			A_minus_lambda_Id.V,
			n,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::characteristic_polynomial "
				"after Tree->make_determinant" << endl;
	}




	init_and_allocate(
			label_txt,
			label_tex,
			true,
			managed_variables_text,
			1, verbose_level);

	V[0].init_formula_from_tree(
				label_txt, label_tex,
				Fq,
				Tree,
				verbose_level);

	f_matrix = false;
	nb_rows = 0;
	nb_cols = 0;

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial done" << endl;
	}
}

void formula_vector::determinant(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::determinant" << endl;
	}
	if (!A->f_matrix) {
		cout << "formula_vector::determinant "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int n;

	n = A->nb_rows;

	if (A->nb_rows != A->nb_cols) {
		cout << "formula_vector::determinant "
				"the matrix is not square" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "formula_vector::determinant "
				"we found a square matrix of size " << n << endl;
	}




	syntax_tree *Tree;

	Tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "formula_vector::determinant "
				"before Tree->init" << endl;
	}
	Tree->init(Fq, f_has_managed_variables, managed_variables_text, verbose_level);
	if (f_v) {
		cout << "formula_vector::determinant "
				"after Tree->init" << endl;
	}

	if (f_v) {
		cout << "formula_vector::determinant "
				"before Tree->make_determinant" << endl;
	}
	Tree->make_determinant(
			Fq,
			A->V,
			n,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::determinant "
				"after Tree->make_determinant" << endl;
	}



	init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			1 /*len*/, verbose_level);

	V[0].init_formula_from_tree(
				label_txt, label_tex,
				Fq,
				Tree,
				verbose_level);

	f_matrix = false;

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::determinant done" << endl;
	}
}

void formula_vector::right_nullspace(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::right_nullspace" << endl;
	}

	if (!A->f_matrix) {
		cout << "formula_vector::right_nullspace "
				"-right_nullspace input must be a matrix" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix of size " << A->nb_rows
				<< " x " << A->nb_cols << endl;
	}



	if (!A->is_integer_matrix()) {
		cout << "formula_vector::right_nullspace "
				"not a matrix of constant integers" << endl;
		exit(1);
	}

	int *M;

	A->get_integer_matrix(M, verbose_level);


	if (f_v) {
		cout << "formula_vector::right_nullspace "
				"of the matrix"
				<< endl;
		Int_matrix_print(M,
				A->nb_rows,
				A->nb_cols);
	}

	linear_algebra::linear_algebra_global LA;

	int *Nullspace;
	int nullspace_m, nullspace_n;
	int *M2;
	int rk_M;
	int *base_cols;

	if (f_v) {
		cout << "formula_vector::right_nullspace "
				"before LA.nullspace" << endl;
	}
	LA.nullspace(
			Fq,
			M, A->nb_rows, A->nb_cols,
			Nullspace, nullspace_m, nullspace_n,
			M2, rk_M, base_cols,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::right_nullspace "
				"after LA.nullspace" << endl;
	}


	init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			nullspace_m * nullspace_n, verbose_level);

	int i;

	for (i = 0; i < len; i++) {

		int value;

		value = Nullspace[i];

		V[i].init_formula_int(
				label_txt, label_tex,
				value,
				Fq,
				f_has_managed_variables,
				managed_variables,
				verbose_level);
	}

	f_matrix = true;
	nb_rows = nullspace_m;
	nb_cols = nullspace_n;

	FREE_int(M);
	FREE_int(Nullspace);
	FREE_int(M2);
	FREE_int(base_cols);

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::right_nullspace done" << endl;
	}

}


void formula_vector::matrix_minor(
		formula_vector *A,
		field_theory::finite_field *Fq,
		int i, int j,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::matrix_minor" << endl;
	}


	if (!A->f_matrix) {
		cout << "formula_vector::matrix_minor "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int m, n;

	m = A->nb_rows;
	n = A->nb_cols;

	if (i >= m) {
		cout << "row index out of range." << endl;
		exit(1);
	}
	if (i < 0) {
		cout << "row index out of range." << endl;
		exit(1);
	}
	if (j >= n) {
		cout << "column index out of range." << endl;
		exit(1);
	}
	if (j < 0) {
		cout << "column index out of range." << endl;
		exit(1);
	}

	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"we found a matrix of size " << m << " x " << n << endl;
	}
	if (m != n) {
		cout << "formula_vector::matrix_minor "
				"the input matrix must be square" << endl;
		exit(1);
	}

	int u, v;
	int c, d;


	formula_vector *M;

	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"before computing temporary matrix" << endl;
	}

	M = NEW_OBJECT(formula_vector);

	M->init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			(n - 1) * (n - 1), verbose_level);

	M->f_matrix = true;
	M->nb_rows = n - 1;
	M->nb_cols = n - 1;

	for (u = 0, c = 0; u < n; u++, c++) {

		if (u == i) {
			c--;
			continue;
		}
		for (v = 0, d = 0; v < n; v++, d++) {

			if (v == j) {
				d--;
				continue;
			}

			// copy the matrix entry (u,v) to (c,d):

			if (f_v) {
				cout << "formula_vector::matrix_minor copying "
						"(" << u << "," << v << ") to "
						"(" << c << "," << d << ")" << endl;
			}

			A->V[u * n + v].copy_to(
					&M->V[c * (n - 1) + d],
					verbose_level);

		}
	}
	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"after computing temporary matrix" << endl;
	}


	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"before determinant" << endl;
	}

	formula_vector *Det;

	Det = NEW_OBJECT(formula_vector);

	Det->determinant(
			M,
			Fq,
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			verbose_level);

	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"after determinant" << endl;
	}

	FREE_OBJECT(M);


	if (ODD(i + j)) {
		if (f_v) {
			cout << "formula_vector::matrix_minor "
					"before multiplying by -1" << endl;
		}

		Det->V[0].tree->multiply_by_minus_one(
				Fq,
				verbose_level);

		if (f_v) {
			cout << "formula_vector::matrix_minor "
					"after multiplying by -1" << endl;
		}
	}


	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"before simplify" << endl;
	}
	simplify(
			Det,
			Fq,
			label_txt,
			label_tex,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::matrix_minor "
				"after simplify" << endl;
	}

	FREE_OBJECT(Det);

	collect_variables(verbose_level);


	if (f_v) {
		cout << "formula_vector::matrix_minor done" << endl;
	}
}

void formula_vector::symbolic_nullspace(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::symbolic_nullspace" << endl;
	}


	if (!A->f_matrix) {
		cout << "formula_vector::symbolic_nullspace "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int m, n;

	m = A->nb_rows;
	n = A->nb_cols;

	if (f_v) {
		cout << "formula_vector::symbolic_nullspace "
				"we found a matrix of size " << m << " x " << n << endl;
	}

	if (m != n - 1) {
		cout << "formula_vector::symbolic_nullspace need m == n - 1" << endl;
		exit(1);
	}

	init_and_allocate(
			label_txt,
			label_tex,
			true,
			managed_variables_text,
			n, verbose_level);

	f_matrix = true;
	nb_rows = 1;
	nb_cols = n;


	formula_vector *B; // a temporary object
	int i, j;

	B = NEW_OBJECT(formula_vector);

	B->init_and_allocate(
				label_txt, label_tex,
				true,
				managed_variables_text,
				n * n, verbose_level);
	B->f_matrix = true;
	B->nb_rows = n;
	B->nb_cols = n;

	if (f_v) {
		cout << "formula_vector::symbolic_nullspace "
				"before copy matrix" << endl;
	}
	for (i = 0; i < m * n; i++) {
		A->V[i].copy_to(&B->V[i], verbose_level);
	}
	if (f_v) {
		cout << "formula_vector::symbolic_nullspace "
				"after copy matrix" << endl;
	}

	// fill the last row of B with ones:
	// The value of the entry does not matter since
	// we will do minors w.r.t. the last row of B:

	i = n - 1;
	for (j = 0; j < n; j++) {

		string label_txt;
		string label_tex;

		label_txt = "dummy";
		label_tex = "dummy";

		if (f_v) {
			cout << "formula_vector::symbolic_nullspace "
					"before init_formula_int j = " << j << endl;
		}
		B->V[i * n + j].init_formula_int(
				label_txt, label_tex,
				1,
				Fq,
				f_has_managed_variables,
				managed_variables,
				verbose_level);
		if (f_v) {
			cout << "formula_vector::symbolic_nullspace "
					"after init_formula_int" << endl;
		}
	}

	// compute the minors w.r.t. the entries in the last row of B:

	for (j = 0; j < n; j++) {

		formula_vector *T; // temporary object to compute the minor

		T = NEW_OBJECT(formula_vector);

		T->matrix_minor(
				B,
				Fq,
				i, j,
				label_txt,
				label_tex,
				f_has_managed_variables,
				managed_variables_text,
				verbose_level);

		// copy the partial result into the output vector:

		T->V[0].copy_to(
				&V[j],
				verbose_level);

		FREE_OBJECT(T);

	}

	FREE_OBJECT(B);

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::symbolic_nullspace done" << endl;
	}
}


void formula_vector::multiply_2by2_from_the_left(
		formula_vector *M,
		formula_vector *A2,
		int i, int j,
		field_theory::finite_field *Fq,
		std::string &label_txt,
		std::string &label_tex,
		int f_has_managed_variables,
		std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::multiply_2by2_from_the_left" << endl;
	}


	if (!M->f_matrix) {
		cout << "formula_vector::multiply_2by2_from_the_left "
				"the object M is not of type matrix" << endl;
		exit(1);
	}

	int m, n;

	m = M->nb_rows;
	n = M->nb_cols;

	if (f_v) {
		cout << "formula_vector::multiply_2by2_from_the_left "
				"we found a matrix of size " << m << " x " << n << endl;
	}


	init_and_allocate(
			label_txt,
			label_tex,
			f_has_managed_variables,
			managed_variables_text,
			m * n, verbose_level);

	f_matrix = true;
	nb_rows = m;
	nb_cols = n;


	if (!A2->f_matrix) {
		cout << "formula_vector::multiply_2by2_from_the_left "
				"the object A2 is not of type matrix" << endl;
		exit(1);
	}
	if (A2->nb_rows != 2) {
		cout << "formula_vector::multiply_2by2_from_the_left "
				"the object A2 is not 2 x 2" << endl;
		exit(1);
	}
	if (A2->nb_cols != 2) {
		cout << "formula_vector::multiply_2by2_from_the_left "
				"the object A2 is not 2 x 2" << endl;
		exit(1);
	}

	int u, v;

	for (u = 0; u < m; u++) {
		if (u == i) {
			if (f_v) {
				cout << "formula_vector::multiply_2by2_from_the_left "
						"u = " << u << endl;
			}
			// row i is a linear combination of row i and row j:
			for (v = 0; v < n; v++) {
				if (f_v) {
					cout << "formula_vector::multiply_2by2_from_the_left "
							"u = " << u << " v=" << v << endl;
				}
				V[u * n + v].make_linear_combination(
						&A2->V[0 * 2 + 0],
						&M->V[i * n + v],
						&A2->V[0 * 2 + 1],
						&M->V[j * n + v],
						Fq,
						label_txt,
						label_tex,
						managed_variables_text,
						verbose_level);
				if (f_v) {
					cout << "formula_vector::multiply_2by2_from_the_left "
							"u = " << u << " v=" << v << endl;
					V[u * n + v].print_easy(cout);
					cout << endl;
				}
			}
		}
		else if (u == j) {
			if (f_v) {
				cout << "formula_vector::multiply_2by2_from_the_left "
						"u = " << u << endl;
			}
			// row j is a linear combination of row i and row j:
			for (v = 0; v < n; v++) {
				if (f_v) {
					cout << "formula_vector::multiply_2by2_from_the_left "
							"u = " << u << " v=" << v << endl;
				}
				V[u * n + v].make_linear_combination(
						&A2->V[1 * 2 + 0],
						&M->V[i * n + v],
						&A2->V[1 * 2 + 1],
						&M->V[j * n + v],
						Fq,
						label_txt,
						label_tex,
						managed_variables_text,
						verbose_level);
				if (f_v) {
					cout << "formula_vector::multiply_2by2_from_the_left "
							"u = " << u << " v=" << v << endl;
					V[u * n + v].print_easy(cout);
					cout << endl;
				}
			}
		}
		else {
			// row u is an unmodified copy of row u:
			for (v = 0; v < n; v++) {
				M->V[u * n + v].copy_to(&V[u * n + v], verbose_level - 3);
			}
		}
	}

	collect_variables(verbose_level);

	if (f_v) {
		cout << "formula_vector::multiply_2by2_from_the_left" << endl;
	}

}

void formula_vector::latex_tree(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::latex_tree" << endl;
	}
	int i;

	if (len > 1) {
		for (i = 0; i < len; i++) {
			string s;

			s = label_txt + "_" + std::to_string(i);
			V[i].latex_tree(s, verbose_level);
		}
	}
	else if (len == 1) {
		string s;
		string s2;
		data_structures::string_tools String;

		s = label_txt;
		s2 = "M_split";

		if (String.compare_string_string(s, s2) == 0) {
			//V[0].latex(s, verbose_level);
			V[0].latex_tree_split(
					s, 0 /* split_level */, 10 /* split_mod */,
					verbose_level);
		}
		else {
			V[0].latex_tree(s, verbose_level);
		}

	}
	if (f_v) {
		cout << "formula_vector::latex_tree done" << endl;
	}
}

void formula_vector::export_tree(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::export_tree" << endl;
	}
	int i;

	if (len > 1) {
		for (i = 0; i < len; i++) {
			string s;

			s = label_txt + "_" + std::to_string(i);
			V[i].export_tree(s, verbose_level);
		}
	}
	else if (len == 1) {
		V[0].export_tree(label_txt, verbose_level);

	}
	if (f_v) {
		cout << "formula_vector::export_tree done" << endl;
	}
}

void formula_vector::collect_variables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::collect_variables" << endl;
	}
	int i;

	for (i = 0; i < len; i++) {
		if (f_v) {
			cout << "formula_vector::collect_variables "
					"node i=" << i << " / " << len << endl;
		}
		V[i].collect_variables(verbose_level);
		if (f_v) {
			cout << "formula_vector::collect_variables "
					"node i=" << i << " / " << len << " done" << endl;
		}
	}

	if (f_v) {
		cout << "formula_vector::collect_variables print_variables" << endl;
		print_variables(cout);
	}


	if (f_v) {
		cout << "formula_vector::collect_variables done" << endl;
	}

}

void formula_vector::print_variables(
		std::ostream &ost)
{
	int i;

	for (i = 0; i < len; i++) {
		cout << i << " : ";
		V[i].tree->print_variables_in_line(ost);
		cout << endl;
	}

}


}}}


