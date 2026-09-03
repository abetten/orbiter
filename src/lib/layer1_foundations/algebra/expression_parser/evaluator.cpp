/*
 * evaluator.cpp
 *
 *  Created on: Aug 22, 2026
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace expression_parser {




evaluator::evaluator()
{
	Record_birth();

	f_evaluation_mode_int = false;

	f_evaluation_mode_algebra = false;

	algebra_dimension = 0;

	f_characteristic_zero = false;

	f_characteristic_p = 0;

	F = NULL;

}

evaluator::~evaluator()
{
	Record_death();

}


void evaluator::init_mode_int(
		int verbose_level)
{
	int f_v;

	f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::init_mode_int" << endl;
	}

	f_evaluation_mode_int = true;

	if (f_v) {
		cout << "evaluator::init_mode_int done" << endl;
	}

}

void evaluator::init_mode_algebra_characteristic_zero(
		int algebra_dimension,
		int verbose_level)
{
	int f_v;

	f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::init_mode_algebra_characteristic_zero" << endl;
	}

	f_evaluation_mode_algebra = true;

	f_characteristic_zero = true;

	evaluator::algebra_dimension = algebra_dimension;

	if (f_v) {
		cout << "evaluator::init_mode_algebra_characteristic_zero done" << endl;
	}

}

void evaluator::init_mode_algebra_characteristic_p(
		int algebra_dimension,
		algebra::field_theory::finite_field *F,
		int verbose_level)
{
	int f_v;

	f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::init_mode_algebra_characteristic_p" << endl;
	}

	f_evaluation_mode_algebra = true;

	f_characteristic_p = true;
	evaluator::F = F;

	evaluator::algebra_dimension = algebra_dimension;

	if (f_v) {
		cout << "evaluator::init_mode_algebra_characteristic_p done" << endl;
	}

}

void evaluator::algebra_element_print(
		int *Mtx)
{
	Int_matrix_print(Mtx, algebra_dimension, algebra_dimension);
}


void evaluator::algebra_element_make_zero(
		int *Mtx)
{

	Int_vec_zero(Mtx, algebra_dimension * algebra_dimension);
}


void evaluator::algebra_element_make_identity(
		int *Mtx)
{
	int i;

	Int_vec_zero(Mtx, algebra_dimension * algebra_dimension);

	for (i = 0; i < algebra_dimension; i++) {
		Mtx[i * algebra_dimension + i] = 1;
	}
}

void evaluator::algebra_element_make_scalar(
		int a,
		int *Mtx)
{
	int i;

	if (a < 0) {
		cout << "evaluator::algebra_element_make_scalar a < 0" << endl;
		exit(1);
	}
	if (a >= F->q) {
		cout << "evaluator::algebra_element_make_scalar a >= F->q" << endl;
		exit(1);
	}

	Int_vec_zero(Mtx, algebra_dimension * algebra_dimension);

	for (i = 0; i < algebra_dimension; i++) {
		Mtx[i * algebra_dimension + i] = a;
	}
}


void evaluator::algebra_element_minus(
		int *Mtx)
{
	int i;

	for (i = 0; i < algebra_dimension * algebra_dimension; i++) {
		Mtx[i] = F->negate(Mtx[i]);
	}
}

void evaluator::algebra_element_mult(
		int *Mtx_A, int *Mtx_B, int *Mtx_C, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::algebra_element_mult" << endl;
	}

	if (f_v) {
		cout << "evaluator::algebra_element_mult "
				"Mtx_A = " << endl;
		algebra_element_print(Mtx_A);
		cout << "evaluator::algebra_element_mult "
				"Mtx_B = " << endl;
		algebra_element_print(Mtx_B);
	}

	algebra::linear_algebra::linear_algebra Linear_algebra;

	Linear_algebra.init(F, verbose_level);

	if (f_v) {
		cout << "evaluator::algebra_element_mult before Linear_algebra.mult_matrix_matrix" << endl;
	}
	Linear_algebra.mult_matrix_matrix(
			Mtx_A, Mtx_B, Mtx_C,
			algebra_dimension, algebra_dimension, algebra_dimension,
			0 /* verbose_level*/);

	if (f_v) {
		cout << "evaluator::algebra_element_mult "
				"Mtx_C = " << endl;
		algebra_element_print(Mtx_C);
	}

}

void evaluator::algebra_element_add_apply(
		int *Mtx_A, int *Mtx_B)
{
	int i;

	for (i = 0; i < algebra_dimension * algebra_dimension; i++) {
		Mtx_A[i] = F->add(Mtx_A[i], Mtx_B[i]);
	}
}


void evaluator::algebra_element_move(
		int *Mtx_A, int *Mtx_B)
{
	Int_vec_copy(Mtx_A, Mtx_B, algebra_dimension * algebra_dimension);
}

void evaluator::algebra_element_inverse(
		int *Mtx_A, int *Mtx_Av)
{

	algebra::linear_algebra::linear_algebra Linear_algebra;

	Linear_algebra.init(F, 0 /*verbose_level*/);

	Linear_algebra.matrix_inverse(
			Mtx_A, Mtx_Av, algebra_dimension, 0 /* verbose_level */);
}


void evaluator::algebra_element_power(
		int *Mtx_A, int n, int *Mtx_return, int verbose_level)
// computes A^n
{
	int f_v = (verbose_level >= 1);
	//int b, c;

	if (f_v) {
		cout << "evaluator::algebra_element_power "
				"A=" << endl;
		algebra_element_print(Mtx_A);
		cout << "evaluator::algebra_element_power n=" << n << endl;
	}

	int *Mtx_B;
	int *Mtx_C;
	int *Mtx_D;


	Mtx_B = NEW_int(algebra_dimension * algebra_dimension);
	Mtx_C = NEW_int(algebra_dimension * algebra_dimension);
	Mtx_D = NEW_int(algebra_dimension * algebra_dimension);

	if (n < 0) {
		if (f_v) {
			cout << "evaluator::algebra_element_power "
					"exponent is negative" << endl;
		}
		algebra_element_inverse(Mtx_A, Mtx_B);
		algebra_element_move(Mtx_B, Mtx_A);
		//a = inverse(a);
		n = -n;
	}
	algebra_element_move(Mtx_A, Mtx_B);
	//b = a;

	algebra_element_make_identity(Mtx_C);
	//c = 1;

	while (n) {
		if (f_v) {
			cout << "evaluator::algebra_element_power n=" << n << endl;
					//<< " a=" << a << " b=" << b << " c=" << c << endl;
		}
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";
			algebra_element_mult(Mtx_B, Mtx_C, Mtx_D, verbose_level);
			algebra_element_move(Mtx_D, Mtx_C);

			//c = mult(b, c);
			//cout << c << endl;
		}
		algebra_element_mult(Mtx_B, Mtx_B, Mtx_D, verbose_level);
		algebra_element_move(Mtx_D, Mtx_B);
		//b = mult_verbose(b, b, verbose_level);

		n >>= 1;
		//cout << "finite_field::power: " << b << "^"
		//<< n << " * " << c << endl;
	}
	if (f_v) {
		cout << "evaluator::algebra_element_power A=" << endl;
		algebra_element_print(Mtx_A);
		cout << "evaluator::algebra_element_power n=" << n << endl;
		cout << "evaluator::algebra_element_power C=" << endl;
		algebra_element_print(Mtx_C);
	}

	algebra_element_move(Mtx_C, Mtx_return);

	FREE_int(Mtx_B);
	FREE_int(Mtx_C);
	FREE_int(Mtx_D);

	if (f_v) {
		cout << "evaluator::algebra_element_power done" << endl;
	}
	//return c;
}

void evaluator::algebra_evaluate_formula(
		algebra::expression_parser::formula *V,
		int *Mtx_return,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula" << endl;
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula before algebra_evaluate_syntax_tree" << endl;
	}
	algebra_evaluate_syntax_tree(V->tree, Mtx_return, verbose_level);
	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula after algebra_evaluate_syntax_tree" << endl;
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula done" << endl;
	}
}

void evaluator::algebra_evaluate_syntax_tree(
		syntax_tree *tree,
		int *Mtx_return,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::algebra_evaluate_syntax_tree" << endl;
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula before algebra_evaluate_syntax_tree_node" << endl;
	}
	algebra_evaluate_syntax_tree_node(tree->Root, Mtx_return, verbose_level);
	if (f_v) {
		cout << "evaluator::algebra_evaluate_formula after algebra_evaluate_syntax_tree_node" << endl;
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_syntax_tree done" << endl;
	}
}


void evaluator::algebra_evaluate_syntax_tree_node(
		syntax_tree_node *Node,
		int *Mtx_return,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	//int a, b;

	if (f_v) {
		cout << "evaluator::algebra_evaluate_syntax_tree_node" << endl;
	}
	if (Node->f_terminal) {

		if (f_v) {
			cout << "evaluator::algebra_evaluate_syntax_tree_node terminal node" << endl;
		}

		algebra_evaluate_terminal_node(Node->T, Mtx_return, verbose_level - 2);
		//a = Node->T->evaluate(symbol_table, Node->Tree->Fq, verbose_level - 2);

		if (Node->f_has_minus) {
			algebra_element_minus(Mtx_return);
			//a = Node->Tree->Fq->negate(a);
		}
		if (f_v) {
			cout << "evaluator::algebra_evaluate_syntax_tree_node "
					"terminal node evaluates to:" << endl;
			algebra_element_print(Mtx_return);
		}
	}
	else {
		if (Node->nb_nodes == 1) {

			if (f_v) {
				cout << "evaluator::algebra_evaluate_syntax_tree_node nb_nodes == 1" << endl;
			}

			algebra_evaluate_syntax_tree_node(Node->Nodes[0], Mtx_return, verbose_level - 2);

			//a = Node->Nodes[0]->evaluate(symbol_table, verbose_level - 2);


			if (Node->f_has_minus) {
				algebra_element_minus(Mtx_return);
				//a = Node->Tree->Fq->negate(a);
			}
			if (f_v) {
				cout << "evaluator::algebra_evaluate_syntax_tree_node "
						"single node evaluates to " << endl;
				algebra_element_print(Mtx_return);
			}
		}
		else {
			if (f_v) {
				cout << "evaluator::algebra_evaluate_syntax_tree_node nb_nodes > 1" << endl;
			}

			if (Node->type == operation_type_mult) {

				if (f_v) {
					cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node" << endl;
				}
				int *Mtx_A, *Mtx_B, *Mtx_C;

				Mtx_A = NEW_int(algebra_dimension * algebra_dimension);
				Mtx_B = NEW_int(algebra_dimension * algebra_dimension);
				Mtx_C = NEW_int(algebra_dimension * algebra_dimension);


				algebra_element_make_identity(Mtx_A);
				//a = 1;
				if (f_v) {
					cout << "evaluator::algebra_evaluate_syntax_tree_node "
							"Mtx_A = " << endl;
					algebra_element_print(Mtx_A);
				}

				for (i = 0; i < Node->nb_nodes; i++) {

					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node " << i << " / " << Node->nb_nodes << endl;
					}

					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node " << i << " / " << Node->nb_nodes << " performing recursion" << endl;
					}

					algebra_evaluate_syntax_tree_node(Node->Nodes[i], Mtx_B, verbose_level - 2);
					//b = Node->Nodes[i]->evaluate(symbol_table, verbose_level - 2);

					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node " << i << " / " << Node->nb_nodes << " after recursion" << endl;
					}
					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node "
								"single node evaluates to " << endl;
						algebra_element_print(Mtx_B);
					}


					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node " << i << " / " << Node->nb_nodes << " performing multiplication" << endl;
					}

					algebra_element_mult(Mtx_A, Mtx_B, Mtx_C, verbose_level);

					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node multiplication node " << i << " / " << Node->nb_nodes << " performing move" << endl;
					}

					algebra_element_move(Mtx_C, Mtx_A);
					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node "
								"Mtx_A = " << endl;
						algebra_element_print(Mtx_A);
					}

					//a = Node->Tree->Fq->mult(a, b);
				}
				if (Node->f_has_minus) {
					algebra_element_minus(Mtx_A);
					//a = Node->Tree->Fq->negate(a);
				}
				algebra_element_move(Mtx_A, Mtx_return);

				FREE_int(Mtx_A);
				FREE_int(Mtx_B);
				FREE_int(Mtx_C);
				if (f_v) {
					cout << "evaluator::algebra_evaluate_syntax_tree_node "
							"product evaluates to "<< endl;
					algebra_element_print(Mtx_return);
				}
			}
			else if (Node->type == operation_type_add) {

				if (f_v) {
					cout << "evaluator::algebra_evaluate_syntax_tree_node addition node" << endl;
				}

				int *Mtx_A, *Mtx_B;

				Mtx_A = NEW_int(algebra_dimension * algebra_dimension);
				Mtx_B = NEW_int(algebra_dimension * algebra_dimension);

				algebra_element_make_zero(Mtx_A);
				//a = 0;
				for (i = 0; i < Node->nb_nodes; i++) {

					if (f_v) {
						cout << "evaluator::algebra_evaluate_syntax_tree_node addition node " << i << " / " << Node->nb_nodes << endl;
					}


					algebra_evaluate_syntax_tree_node(Node->Nodes[i], Mtx_B, verbose_level - 2);
					//b = Node->Nodes[i]->evaluate(symbol_table, verbose_level - 2);

					algebra_element_add_apply(Mtx_A, Mtx_B);
					//a = Node->Tree->Fq->add(a, b);
				}
				algebra_element_move(Mtx_A, Mtx_return);

				FREE_int(Mtx_A);
				FREE_int(Mtx_B);

				if (f_v) {
					cout << "evaluator::algebra_evaluate_syntax_tree_node "
							"sum evaluates to " << endl;
					algebra_element_print(Mtx_return);
				}

			}
			else {
				cout << "evaluator::algebra_evaluate_syntax_tree_node unknown operation" << endl;
				exit(1);
			}
		}
	}


	if (Node->f_has_exponent) {
		if (f_v) {
			cout << "evaluator::algebra_evaluate_syntax_tree_node "
					"before raising to the power of "
					<< Node->exponent << endl;
			cout << "evaluator::algebra_evaluate_syntax_tree_node "
					"Mtx_return = " << endl;
			algebra_element_print(Mtx_return);
		}

		int *Mtx_D;

		Mtx_D = NEW_int(algebra_dimension * algebra_dimension);

		algebra_element_power(
				Mtx_return, Node->exponent, Mtx_D, 0 /* verbose_level*/);
		algebra_element_move(Mtx_D, Mtx_return);

		FREE_int(Mtx_D);

		//a = Node->Tree->Fq->power(a, Node->exponent);
		if (f_v) {
			cout << "evaluator::algebra_evaluate_syntax_tree_node "
					"after raising to the power of "
					<< Node->exponent << ", result = " << endl;
			algebra_element_print(Mtx_return);
		}
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_syntax_tree_node done, result = " << endl;
		algebra_element_print(Mtx_return);
	}
	//return a;
}

void evaluator::algebra_evaluate_terminal_node(
		syntax_tree_node_terminal *T_Node,
		int *Mtx_return,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "evaluator::algebra_evaluate_terminal_node" << endl;
	}

	if (T_Node->f_int) {
		int a;

		a = T_Node->value_int;
		algebra_element_make_scalar(a, Mtx_return);
	}
	else if (T_Node->f_double) {
		int a;

		a = T_Node->value_double;
		algebra_element_make_scalar(a, Mtx_return);
		//cout << "syntax_tree_node_terminal::evaluate f_double" << endl;
		//exit(1);
	}
	else if (T_Node->f_text) {
		//a = strtoi(value_text);
		//a = ST.strtoi(symbol_table[T_Node->value_text]);

		// look up global symbol table to retrieve a matrix:
		int idx;

		idx = other::orbiter_kernel_system::Orbiter->Orbiter_symbol_table->find_symbol(
				T_Node->value_text);

		other::orbiter_kernel_system::symbol_table_object_type type;

		type = other::orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object_type(
				idx);

		if (type == other::orbiter_kernel_system::t_vector) {
			other::data_structures::vector_builder *V;

			V = Get_vector(T_Node->value_text);
			//other::orbiter_kernel_system::Orbiter->Orbiter_symbol_table->get_object(idx);

			if (V->f_has_k) {
				if (V->k != algebra_dimension) {
					cout << "evaluator::algebra_evaluate_terminal_node the size of the matrix does not match" << endl;
					cout << "expected = " << algebra_dimension << endl;
					cout << "found = " << V->k << endl;
					cout << "evaluator::algebra_evaluate_terminal_node label = " << T_Node->value_text << endl;
				}
				else {
					if (V->len != algebra_dimension * algebra_dimension) {
						cout << "evaluator::algebra_evaluate_terminal_node the size of the vector representing the matrix does not match" << endl;
						cout << "expected = " << algebra_dimension * algebra_dimension << endl;
						cout << "found = " << V->len << endl;
						exit(1);
					}
					else {
						Lint_vec_copy_to_int(V->v, Mtx_return, algebra_dimension * algebra_dimension);
					}
				}
			}
			else {
				cout << "evaluator::algebra_evaluate_terminal_node expecting a matrix" << endl;
				cout << "evaluator::algebra_evaluate_terminal_node label = " << T_Node->value_text << endl;
				exit(1);
			}
		}
		else {
			cout << "evaluator::algebra_evaluate_terminal_node the object is not of type vector" << endl;
			cout << "evaluator::algebra_evaluate_terminal_node label = " << T_Node->value_text << endl;
			exit(1);
		}

	}
	else {
		cout << "evaluator::algebra_evaluate_terminal_node unknown type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "evaluator::algebra_evaluate_terminal_node done, result = " << endl;
		algebra_element_print(Mtx_return);
	}
	//return a;

	if (f_v) {
		cout << "evaluator::algebra_evaluate_terminal_node done" << endl;
	}
}







}}}}


