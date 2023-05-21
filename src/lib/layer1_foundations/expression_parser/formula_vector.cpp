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
		std::string &label_txt, std::string &label_tex,
		std::string &text,
		field_theory::finite_field *Fq,
		std::string &managed_variables,
		int f_matrix, int nb_rows,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::init_from_text" << endl;
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
			input.size(), verbose_level);

	if (f_v) {
		cout << "formula_vector::init_from_text "
				"before loop" << endl;
	}

	int i;
	for (i = 0; i < len; i++) {

		string label_txt;
		string label_tex;
		label_txt = label_txt + "_V" + std::to_string(i);
		label_tex = label_tex + "V" + std::to_string(i);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"before V[i].init_formula_Sajeeb" << endl;
		}
		if (f_v) {
			cout << "formula_vector::init_from_text " << i << " / " << len << " input = " << input[i] << endl;
		}

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"before V[i].init_formula_Sajeeb" << endl;
		}

		V[i].init_formula_Sajeeb(
				label_txt, label_tex,
				managed_variables, input[i],
				Fq,
				verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"after V[i].init_formula_Sajeeb" << endl;
		}

		string s;

		s = V[i].string_representation_formula(verbose_level);

		if (f_v) {
			cout << "formula_vector::init_from_text "
					"formula " << i << " / " << len << " is: " << s << endl;
		}


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
		int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::init_and_allocate" << endl;
	}
	formula_vector::label_txt = label_txt;
	formula_vector::label_tex = label_tex;

	V = NEW_OBJECTS(formula, len);
	formula_vector::len = len;



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

void formula_vector::get_integer_matrix(int *&M, int verbose_level)
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

void formula_vector::get_string_representation_Sajeeb(std::vector<std::string> &S)
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
					cout << "formula_vector::get_string_representation_formula i=" << i << " j=" << j << endl;
				}
				s = V[i * nb_cols + j].string_representation_formula(verbose_level);
				if (f_v) {
					cout << "formula_vector::get_string_representation_formula s=" << s << endl;
				}
				S.push_back(s);
			}
		}
	}
	else {
		int i;
		for (i = 0; i < len; i++) {
			string s;

			s = V[i].string_representation_formula(verbose_level);
			S.push_back(s);
		}
	}
}

void formula_vector::print_Sajeeb(std::ostream &ost)
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

void formula_vector::print_formula(std::ostream &ost, int verbose_level)
{

	if (f_matrix) {
		ost << "symbolic matrix of size "
				<< nb_rows << " x " << nb_cols << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S, verbose_level);

		print_matrix(S, ost);
	}
	else {
		ost << "symbolic vector of size "
				<< len << endl;
		vector<string> S;

		//get_string_representation_Sajeeb(S);
		get_string_representation_formula(S, verbose_level);

		print_vector(S, ost);
	}
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
		ost << setw(3) << i << " : " << setw(W) << S[i];
		ost << endl;
	}

}

void formula_vector::make_A_minus_lambda_Identity(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &variable,
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

	string label_txt = A->label_txt + "minuslambdaId";
	string label_tex = A->label_tex + "minuslambdaId";

	init_and_allocate(
			label_txt,
			label_tex,
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
				lambda_node->init_empty_terminal_node_text(
							Tree,
							variable,
							verbose_level);


				// set the root node to be the new add node:

				V[i * n + j].tree->Root = add_node;
			}

		}
	}


	if (f_v) {
		cout << "formula_vector::make_A_minus_lambda_Identity done" << endl;
	}
}


void formula_vector::substitute(formula_vector *Source,
		formula_vector *Target,
		std::string &substitution_variables,
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
			verbose_level);

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
	std::string label_txt;
	std::string label_tex;


	nb = Source->len / N;

	len = nb * Target->len;

	label_txt = Source->label_txt + "_sub";
	label_tex = Source->label_tex + "Sub";

	init_and_allocate(
			label_txt,
			label_tex,
			len, verbose_level);

	f_matrix = true;
	nb_rows = nb;
	nb_cols = Target->len;


	formula **S;



	S = (formula **) NEW_pvoid(N);


	for (s = 0; s < nb; s++) {

		if (f_v) {
			cout << "formula_vector::substitute "
					"s=" << s << endl;
		}
		for (i = 0; i < N; i++) {
			S[i] = &Source->V[s * N + i];
		}

		for (t = 0; t < Target->len; t++) {

			if (f_v) {
				cout << "formula_vector::substitute "
						"s=" << s << " t=" << t << endl;
			}

			formula *T;

			T = &Target->V[t];

			formula *Out;

			Out = V + s * Target->len + t;

			T->substitute(variables, S, Out, verbose_level - 2);

		}
	}

	FREE_pvoid((void **) S);

	if (f_v) {
		cout << "formula_vector::substitute done" << endl;
	}

}

void formula_vector::simplify(
		formula_vector *A,
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::simplify" << endl;
	}
	init_and_allocate(
			A->label_txt,
			A->label_tex,
			A->len, verbose_level);


	f_matrix = A->f_matrix;
	nb_rows = A->nb_rows;
	nb_cols = A->nb_cols;

	int i;

	for (i = 0; i < A->len; i++) {
		if (f_v) {
			cout << "formula_vector::simplify "
					"-simplify " << i << " / " << A->len
					<< " before copy_to" << endl;
		}

		A->V[i].copy_to(
				&V[i], verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::simplify "
					"-simplify " << i << " / " << A->len
					<< " after copy_to" << endl;
		}

		if (f_v) {
			cout << "formula_vector::simplify "
					"-simplify " << i << " / " << A->len
					<< " before simplify" << endl;
		}

		V[i].simplify(verbose_level - 2);

		if (f_v) {
			cout << "formula_vector::simplify "
					"-simplify " << i << " / " << A->len
					<< " after simplify" << endl;
		}

		if (f_v) {
			cout << "formula_vector::simplify "
					"-simplify " << i << " / " << A->len
					<< " done" << endl;
		}
	}

}

void formula_vector::characteristic_polynomial(
		formula_vector *A,
		field_theory::finite_field *Fq,
		std::string &variable,
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
	Tree->init(Fq, verbose_level);
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



	std::string label_txt;
	std::string label_tex;

	label_txt = A->label_txt + "_charpoly";
	label_tex = A->label_tex + "Charpoly";

	init_and_allocate(
			label_txt,
			label_tex,
			1, verbose_level);

	V[0].init_formula_from_tree(
				label_txt, label_tex,
				Fq,
				Tree,
				verbose_level);

	f_matrix = false;
	nb_rows = 0;
	nb_cols = 0;

	if (f_v) {
		cout << "formula_vector::characteristic_polynomial done" << endl;
	}
}

void formula_vector::determinant(
		formula_vector *A,
		field_theory::finite_field *Fq,
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
	Tree->init(Fq, verbose_level);
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



	std::string label;
	std::string label_tex;

	label = A->label_txt + "_det";
	label_tex = A->label_tex + "det";

	init_and_allocate(
			label,
			label_tex,
			1 /*len*/, verbose_level);

	V[0].init_formula_from_tree(
				label, label_tex,
				Fq,
				Tree,
				verbose_level);

	f_matrix = false;

	if (f_v) {
		cout << "formula_vector::determinant done" << endl;
	}
}

void formula_vector::right_nullspace(
		formula_vector *A,
		field_theory::finite_field *Fq,
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


	if (f_v) {
		cout << "formula_vector::right_nullspace "
				"before LA.do_nullspace" << endl;
	}
	LA.do_nullspace(
			Fq,
			M,
			A->nb_rows,
			A->nb_cols,
			false /* Descr->f_normalize_from_the_left */,
			false /* Descr->f_normalize_from_the_right */,
			Nullspace, nullspace_m, nullspace_n,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::right_nullspace "
				"after LA.do_nullspace" << endl;
	}


	init_and_allocate(
			A->label_txt,
			A->label_tex,
			nullspace_m * nullspace_n, verbose_level);

	int i;

	for (i = 0; i < len; i++) {
		std::string label;
		std::string label_tex;

		label = A->label_txt + "_Nullspace_" + std::to_string(i);
		label_tex = A->label_tex + "Nullspace" + std::to_string(i);

		int value;

		value = Nullspace[i];

		V[i].init_formula_int(
				label, label_tex,
				value,
				Fq,
				verbose_level);
	}

	f_matrix = true;
	nb_rows = nullspace_m;
	nb_cols = nullspace_n;

	FREE_int(M);
	FREE_int(Nullspace);

	if (f_v) {
		cout << "formula_vector::right_nullspace done" << endl;
	}

}


void formula_vector::minor(
		formula_vector *A,
		field_theory::finite_field *Fq,
		int i, int j,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "formula_vector::minor" << endl;
	}


	if (!A->f_matrix) {
		cout << "formula_vector::minor "
				"the object is not of type matrix" << endl;
		exit(1);
	}

	int m, n;

	m = A->nb_rows;
	n = A->nb_cols;

	if (f_v) {
		cout << "formula_vector::minor "
				"we found a matrix of size " << m << " x " << n << endl;
	}

	int u, v;
	int c, d;


	string label_txt = A->label_txt + "minor_" + std::to_string(i) + "_" + std::to_string(j);
	string label_tex = A->label_tex + "minor" + std::to_string(i) + std::to_string(j);

	formula_vector *M;

	if (f_v) {
		cout << "formula_vector::minor before computing temporary matrix" << endl;
	}

	M = NEW_OBJECT(formula_vector);

	M->init_and_allocate(
			label_txt,
			label_tex,
			(m - 1) * (n - 1), verbose_level);

	M->f_matrix = true;
	M->nb_rows = m - 1;
	M->nb_cols = n - 1;

	for (u = 0, c = 0; u < m; u++, c++) {

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
				cout << "formula_vector::minor copying (" << u << "," << v << ") to (" << c << "," << d << ")" << endl;
			}

			A->V[u * n + v].copy_to(
					&M->V[c * (n - 1) + d],
					verbose_level);

		}
	}
	if (f_v) {
		cout << "formula_vector::minor after computing temporary matrix" << endl;
	}


	if (f_v) {
		cout << "formula_vector::minor before determinant" << endl;
	}

	formula_vector *Det;

	Det = NEW_OBJECT(formula_vector);

	Det->determinant(
			M,
			Fq,
			verbose_level);

	if (f_v) {
		cout << "formula_vector::minor after determinant" << endl;
	}

	FREE_OBJECT(M);


	if (ODD(i + j)) {
		if (f_v) {
			cout << "formula_vector::minor before multiplying by -1" << endl;
		}

		Det->V[0].tree->multiply_by_minus_one(
				Fq,
				verbose_level);

		if (f_v) {
			cout << "formula_vector::minor after multiplying by -1" << endl;
		}
	}


	if (f_v) {
		cout << "formula_vector::minor "
				"before simplify" << endl;
	}
	simplify(
			Det,
			Fq,
			verbose_level);
	if (f_v) {
		cout << "formula_vector::minor "
				"after simplify" << endl;
	}

	FREE_OBJECT(Det);


	if (f_v) {
		cout << "formula_vector::minor done" << endl;
	}
}

void formula_vector::symbolic_nullspace(
		formula_vector *A,
		field_theory::finite_field *Fq,
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

	string label_txt = A->label_txt + "perp";
	string label_tex = A->label_tex + "perp";

	init_and_allocate(
			label_txt,
			label_tex,
			n, verbose_level);

	f_matrix = true;
	nb_rows = 1;
	nb_cols = n;


	formula_vector *B;
	int i, j;

	B = NEW_OBJECT(formula_vector);

	B->init_and_allocate(
				A->label_txt, A->label_tex,
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
				verbose_level);
		if (f_v) {
			cout << "formula_vector::symbolic_nullspace "
					"after init_formula_int" << endl;
		}
	}

	for (j = 0; j < n; j++) {

		formula_vector *T;

		T = NEW_OBJECT(formula_vector);

		T->minor(
				B,
				Fq,
				i, j,
				verbose_level);

		T->V[0].copy_to(
				&V[j],
				verbose_level);

		FREE_OBJECT(T);

	}

	FREE_OBJECT(B);


	if (f_v) {
		cout << "formula_vector::symbolic_nullspace done" << endl;
	}
}



}}}


