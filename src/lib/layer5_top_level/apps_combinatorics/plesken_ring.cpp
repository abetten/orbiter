/*
 * plesken_ring.cpp
 *
 *  Created on: Mar 2, 2026
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {




plesken_ring::plesken_ring()
{
	Record_birth();

	PC = NULL;

	Matrix_stack = NULL;
	matrix_stack_size = 0;

	Pup = NULL;
	Pdown = NULL;
	N = 0;

	Pup_inv = NULL;
	Pdown_inv = NULL;

	output2 = NULL;

}




plesken_ring::~plesken_ring()
{
	Record_death();

	if (Matrix_stack) {

		int i;

		for (i = 0; i < matrix_stack_size; i++) {
			FREE_OBJECT(Matrix_stack[i]);
		}
		FREE_pvoid((void **) Matrix_stack);
	}

	if (Pup_inv) {
		FREE_int(Pup_inv);
	}
	if (Pdown_inv) {
		FREE_int(Pdown_inv);
	}
	if (output2) {
		FREE_int(output2);
	}
}


void plesken_ring::init(
		poset_classification::poset_classification *PC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring::init" << endl;
	}

	plesken_ring::PC = PC;

	poset_classification::pc_combinatorics Pc_combinatorics;

	Pc_combinatorics.init(
			PC,
			0 /* verbose_level*/);

	// compute Stack of neighboring Kramer Mesner matrices


	if (f_v) {
		cout << "pc_combinatorics::compute_Kramer_Mesner_matrix "
				"before compute_incidence_matrix_stack" << endl;
	}
	Pc_combinatorics.compute_incidence_matrix_stack(
			Matrix_stack,
			PC->get_depth(),
			verbose_level);
	if (f_v) {
		cout << "pc_combinatorics::compute_Kramer_Mesner_matrix "
				"after compute_incidence_matrix_stack" << endl;
	}
	matrix_stack_size = PC->get_depth();


	if (f_v) {
		cout << "plesken_ring::init "
				"before Pc_combinatorics.Plesken_matrices" << endl;
	}
	Pc_combinatorics.Plesken_matrices(
			Matrix_stack,
			Pup,
			Pdown,
			N,
			verbose_level);
	if (f_v) {
		cout << "plesken_ring::init "
				"after Pc_combinatorics.Plesken_matrices" << endl;
	}

	if (f_v) {
		cout << "plesken_ring::init Pup = " << endl;
		Int_matrix_print(Pup, N, N);
	}

	if (f_v) {
		cout << "plesken_ring::init Pdown = " << endl;
		Int_matrix_print(Pdown, N, N);
	}

	int *Tmp;

	Tmp = NEW_int(N * N);

	Pup_inv = NEW_int(N * N);
	Pdown_inv = NEW_int(N * N);

	Int_vec_zero(Pup_inv, N * N);
	Int_vec_zero(Pdown_inv, N * N);

	int i;
	for (i = 0; i < N; i++) {
		Pup_inv[i * N + i] = 1;
	}
	for (i = 0; i < N; i++) {
		Pdown_inv[i * N + i] = 1;
	}

	Int_vec_copy(Pup, Tmp, N * N);
	if (f_v) {
		cout << "plesken_ring::init "
				"before RREF_elimination_above" << endl;
	}
	RREF_elimination_above(
			Tmp, Pup_inv, N,
			verbose_level - 2);
	if (f_v) {
		cout << "plesken_ring::init "
				"after RREF_elimination_above" << endl;
	}

	if (f_v) {
		cout << "plesken_ring::init Pup_inv = " << endl;
		Int_matrix_print(Pup_inv, N, N);
	}

	Int_vec_copy(Pdown, Tmp, N * N);
	if (f_v) {
		cout << "plesken_ring::init "
				"before RREF_elimination_above" << endl;
	}
	RREF_elimination_above(
			Tmp, Pdown_inv, N,
			verbose_level - 2);
	if (f_v) {
		cout << "plesken_ring::init "
				"after RREF_elimination_above" << endl;
	}

	if (f_v) {
		cout << "plesken_ring::init Pdown_inv = " << endl;
		Int_matrix_print(Pdown_inv, N, N);
	}


	output2 = NEW_int(N);

	FREE_int(Tmp);

	if (f_v) {
		cout << "plesken_ring::init done" << endl;
	}
}


void plesken_ring::RREF_elimination_above(
		int *A, int *B, int n,
		int verbose_level)
// A is a n x n matrix,
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "linear_algebra::RREF_elimination_above" << endl;
	}

	int i, j, k, jj, z; //, a, b, c;

	for (i = n - 1; i > 0; i--) {
		j = i; //base_cols[i];
		//a = A[i * n + j];
		// do the gaussian elimination in the upper part:
		for (k = i - 1; k >= 0; k--) {
			z = A[k * n + j];
			if (z == 0) {
				continue;
			}
			A[k * n + j] = 0;
			for (jj = j + 1; jj < n; jj++) {

				A[k * n + jj] -= z * A[i * n + jj];

#if 0
				a = A[i * n + jj];
				b = A[k * n + jj];
				c = F->mult(z, a);
				c = F->negate(c);
				c = F->add(c, b);
				A[k * n + jj] = c;
#endif
			}
			for (jj = 0; jj < n; jj++) {

				B[k * n + jj] -= z * B[i * n + jj];
			}
		} // next k
	}
	if (f_v) {
		cout << "linear_algebra::RREF_elimination_above done" << endl;
	}
}



void plesken_ring::do_report(
		other::graphics::draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring::do_report" << endl;
	}


	{
		string fname_report;

		fname_report = PC->get_problem_label() + "_plesken_ring.tex";


		{
			ofstream ost(fname_report);


			string title, author, extra_praeamble;

			title = "Plesken Ring ";


			other::l1_interfaces::latex_interface L;

			//latex_head_easy(fp);
			L.head(ost,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				true /*f_enlarged_page */,
				true /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);




			//ost << "\\subsection*{The surface $" << SC->label_tex << "$}" << endl;
			if (f_v) {
				cout << "plesken_ring::do_report "
						"before do_report2" << endl;
			}
			do_report2(
					ost, Draw_options, verbose_level - 2);
			if (f_v) {
				cout << "plesken_ring::do_report "
						"after do_report2" << endl;
			}


			L.foot(ost);
		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	}
	if (f_v) {
		cout << "plesken_ring::do_report done" << endl;
	}

}


void plesken_ring::do_report2(
		std::ostream &ost,
		other::graphics::draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring::do_report2" << endl;
	}

	int *r, *c;
	int *C;

	r = NEW_int(N);
	c = NEW_int(N);
	C = NEW_int(N * N);

	other::l1_interfaces::latex_interface Latex_interface;

	ost << "$$" << endl;
	ost << "N=" << N << endl;
	ost << "$$" << endl;

	ost << "$$" << endl;
	ost << "P^{\\vee}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		Pup, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	row_sum(Pup, r);
	col_sum(Pup, c);

	ost << "$$" << endl;
	ost << "r=";
	Int_vec_print(ost, r, N);
	ost << endl;
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "c=";
	Int_vec_print(ost, c, N);
	ost << endl;
	ost << "$$" << endl;

	ost << "$$" << endl;
	ost << "(P^{\\vee})^{-1}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		Pup_inv, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	mult_matrix(
			Pup, Pup_inv, C);

	ost << "$$" << endl;
	ost << "P^{\\vee} \\cdot (P^{\\vee})^{-1}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		C, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	ost << "$$" << endl;
	ost << "P^{\\wedge}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		Pdown, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;


	row_sum(Pdown, r);
	col_sum(Pdown, c);

	ost << "$$" << endl;
	ost << "r=";
	Int_vec_print(ost, r, N);
	ost << endl;
	ost << "$$" << endl;
	ost << "$$" << endl;
	ost << "c=";
	Int_vec_print(ost, c, N);
	ost << endl;
	ost << "$$" << endl;


	ost << "$$" << endl;
	ost << "(P^{\\wedge})^{-1}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		Pdown_inv, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;

	mult_matrix(
			Pdown, Pdown_inv, C);


	ost << "$$" << endl;
	ost << "P^{\\wedge} \\cdot (P^{\\wedge})^{-1}=" << endl;
	ost << "\\left[" << endl;
	Latex_interface.print_integer_matrix_tex(
			ost,
		C, N, N);
	ost << "\\right]" << endl;
	ost << "$$" << endl;




	FREE_int(r);
	FREE_int(c);
	FREE_int(C);

	if (f_v) {
		cout << "plesken_ring::do_report2 done" << endl;
	}

}

void plesken_ring::row_sum(
		int *M, int *v)
{
	int i, j;

	for (i = 0; i < N; i++) {
		v[i] = 0;
		for (j = 0; j < N; j++) {
			v[i] += M[i * N + j];
		}
	}
}

void plesken_ring::col_sum(
		int *M, int *v)
{
	int i, j;

	for (j = 0; j < N; j++) {
		v[j] = 0;
		for (i = 0; i < N; i++) {
			v[j] += M[i * N + j];
		}
	}
}

void plesken_ring::mult_matrix(
		int *A, int *B, int *C)
{
	int i, j, k, c;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			c = 0;
			for (k = 0; k < N; k++) {
				c += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = c;
		}
	}
}


void plesken_ring::evaluate_expression(
		int f_sup,
		algebra::ring_theory::homogeneous_polynomial_domain *Ring,
		algebra::expression_parser::symbolic_object_builder *Object,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "plesken_ring::evaluate_expression" << endl;
	}


	int len;

	len = Object->Formula_vector->len;

	if (f_v) {
		cout << "plesken_ring::evaluate_expression "
				"len = " << len << endl;
	}

	int nb_vars;

	nb_vars = Ring->get_nb_variables();


	std::map<std::string, int> symbol_table;

#if 0
	other::data_structures::string_tools ST;

	ST.parse_value_pairs(
			symbol_table,
				evaluate_text, verbose_level - 1);
#endif

	int h;


	for (h = 0; h < nb_vars; h++) {

		symbol_table[Ring->get_symbol(h)] = h;

	}

	int *output;

	output = NEW_int(N);

	for (h = 0; h < len; h++) {

		syntax_tree_evaluate(
				f_sup,
				Object->Formula_vector->V[h].tree,
				symbol_table,
				output,
				verbose_level);

		if (f_v) {
			cout << "plesken_ring::evaluate_expression "
					" term " << h << " / " << len <<
					" output = ";
			Int_vec_print(cout, output, N);
			cout << endl;
		}


		int *Pv;

		if (f_sup) {
			Pv = Pdown_inv;

			int i, j;


			// multiply the inverse from the right
			// output2 = output * P^-1

			for (i = 0; i < N; i++) {
				output2[i] = 0;
				for (j = 0; j < N; j++) {
					output2[i] += output[j] * Pv[j * N + i];
				}
			}


		}
		else {
			Pv = Pup_inv;

			int i, j;

			// multiply the inverse from the left
			// output2 = P^-1 * output

			for (i = 0; i < N; i++) {
				output2[i] = 0;
				for (j = 0; j < N; j++) {
					output2[i] += Pv[i * N + j] * output[j];
				}
			}


		}

		if (f_v) {
			cout << "plesken_ring::evaluate_expression "
					" term " << h << " / " << len <<
					" output2 = ";
			Int_vec_print(cout, output2, N);
			cout << endl;
		}

	}




	FREE_int(output);



	if (f_v) {
		cout << "plesken_ring::evaluate_expression done" << endl;
	}
}


void plesken_ring::syntax_tree_evaluate(
		int f_sup,
		algebra::expression_parser::syntax_tree *tree,
		std::map<std::string, int> &symbol_table,
		int *output,
		int verbose_level)
// output[N]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "plesken_ring::syntax_tree_evaluate" << endl;
	}

	syntax_tree_node_evaluate(
			f_sup,
			tree->Root,
			symbol_table,
			output,
			verbose_level - 1);

	if (f_v) {
		cout << "plesken_ring::syntax_tree_evaluate done" << endl;
	}
}

void plesken_ring::syntax_tree_node_evaluate(
		int f_sup,
		algebra::expression_parser::syntax_tree_node *node,
		std::map<std::string, int> &symbol_table,
		int *output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "plesken_ring::syntax_tree_node_evaluate" << endl;
	}
	if (node->f_terminal) {
		syntax_tree_node_terminal_evaluate(f_sup, node->T, symbol_table, output, verbose_level - 2);
		if (node->f_has_minus) {
			cout << "plesken_ring::syntax_tree_node_evaluate "
					" minus is not yet implemented" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "plesken_ring::syntax_tree_node_evaluate "
					" output = ";
			Int_vec_print(cout, output, N);
			cout << endl;
		}
	}
	else {
		if (node->nb_nodes == 1) {
			syntax_tree_node_evaluate(f_sup, node->Nodes[0], symbol_table, output, verbose_level - 2);
			if (node->f_has_minus) {
				cout << "plesken_ring::syntax_tree_node_evaluate "
						" minus is not yet implemented" << endl;
				exit(1);
			}
			if (f_v) {
				cout << "plesken_ring::syntax_tree_node_evaluate "
						" output = ";
				Int_vec_print(cout, output, N);
				cout << endl;
			}
		}
		else {
			if (node->type == operation_type_mult) {
				//a = 1;
				syntax_tree_node_evaluate(f_sup, node->Nodes[0], symbol_table, output, verbose_level - 2);
				for (i = 1; i < node->nb_nodes; i++) {
					syntax_tree_node_evaluate(f_sup, node->Nodes[i], symbol_table, output2, verbose_level - 2);
					int j;
					for (j = 0; j < N; j++) {
						output[j] *= output2[j];
					}
					//a = node->Tree->Fq->mult(a, b);
				}
				if (node->f_has_minus) {
					//a = node->Tree->Fq->negate(a);
					cout << "plesken_ring::syntax_tree_node_evaluate "
							" minus is not yet implemented" << endl;
					exit(1);
				}
				if (f_v) {
					cout << "plesken_ring::syntax_tree_node_evaluate "
							" output = ";
					Int_vec_print(cout, output, N);
					cout << endl;
				}
			}
			else if (node->type == operation_type_add) {
				//a = 0;
				syntax_tree_node_evaluate(f_sup, node->Nodes[0], symbol_table, output, verbose_level - 2);
				for (i = 0; i < node->nb_nodes; i++) {
					syntax_tree_node_evaluate(f_sup, node->Nodes[i], symbol_table, output2, verbose_level - 2);
					int j;
					for (j = 0; j < N; j++) {
						output[j] += output2[j];
					}
					//a = node->Tree->Fq->add(a, b);
				}
				if (f_v) {
					cout << "plesken_ring::syntax_tree_node_evaluate "
							" output = ";
					Int_vec_print(cout, output, N);
					cout << endl;
				}
			}
			else {
				cout << "plesken_ring::syntax_tree_node_evaluate unknown operation" << endl;
				exit(1);
			}
		}
	}
	if (node->f_has_exponent) {
		if (f_v) {
			cout << "plesken_ring::syntax_tree_node_evaluate "
					"before raising to the power of "
					<< node->exponent << ", a=" << endl;
		}
		//a = node->Tree->Fq->power(a, node->exponent);
		if (node->exponent > 1) {
			int j;
			for (j = 0; j < N; j++) {
				output[j] = power(output[j], node->exponent);
			}
		}
		if (f_v) {
			cout << "plesken_ring::syntax_tree_node_evaluate "
					"after raising to the power of "
					<< node->exponent << ", output = ";
			Int_vec_print(cout, output, N);
			cout << endl;
		}
	}

	if (f_v) {
		cout << "plesken_ring::syntax_tree_node_evaluate done" << endl;
	}
	//return a;
}

long int plesken_ring::power(
		long int a, long int n)
// for the algorithm, compare longinteger_domain::power_longint_mod
{
	long int b, c;
	long int N;

	b = a;
	c = 1;

	N = n;
	while (N) {
		if (N % 2) {
			c *= b;
		}
		b *= b;
		N >>= 1;
	}
	return c;
}



void plesken_ring::syntax_tree_node_terminal_evaluate(
		int f_sup,
		algebra::expression_parser::syntax_tree_node_terminal *Terminal,
		std::map<std::string, int> &symbol_table,
		int *output,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "plesken_ring::syntax_tree_node_terminal_evaluate" << endl;
	}
	if (Terminal->f_int) {
		int a;
		a = Terminal->value_int;
		int i;
		for (i = 0; i < N; i++) {
			output[i] = a;
		}
	}
	else if (Terminal->f_double) {
		int a;
		a = Terminal->value_double;
		int i;
		for (i = 0; i < N; i++) {
			output[i] = a;
		}
		//cout << "plesken_ring::syntax_tree_node_terminal_evaluate f_double" << endl;
		//exit(1);
	}
	else if (Terminal->f_text) {
		//a = strtoi(value_text);
		int idx;

		idx = symbol_table[Terminal->value_text];

		if (f_sup) {

			int i;

			// get the idx-th row:
			for (i = 0; i < N; i++) {
				output[i] = Pdown[idx * N + i];
			}
		}
		else {

			int i;

			// get the idx-th column:
			for (i = 0; i < N; i++) {
				output[i] = Pup[i * N + idx];
			}
		}
	}
	else {
		cout << "plesken_ring::syntax_tree_node_terminal_evaluate unknown type" << endl;
		exit(1);
	}

#if 0
	if (a < 0) {
		cout << "plesken_ring::syntax_tree_node_terminal_evaluate a < 0" << endl;
		exit(1);
	}
	if (a >= F->q) {
		cout << "plesken_ring::syntax_tree_node_terminal_evaluate a >= F->q" << endl;
		exit(1);
	}
#endif

	if (f_v) {
		cout << "plesken_ring::syntax_tree_node_terminal_evaluate done output = ";
		Int_vec_print(cout, output, N);
		cout << endl;
	}
	//return a;
}



}}}

