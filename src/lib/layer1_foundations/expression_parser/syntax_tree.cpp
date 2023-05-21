/*
 * syntax_tree.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {


syntax_tree::syntax_tree()
{
	f_has_managed_variables = false;
	//std::vector<std::string> managed_variables;

	Fq = NULL;

	Root = NULL;
}

syntax_tree::~syntax_tree()
{
	if (Root) {
		FREE_OBJECT(Root);
	}
}

void syntax_tree::init(
		field_theory::finite_field *Fq, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init" << endl;
	}

	syntax_tree::Fq = Fq;

	if (f_v) {
		cout << "syntax_tree::init done" << endl;
	}
}

void syntax_tree::init_root_node(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init_root_node" << endl;
	}

	Root = NEW_OBJECT(syntax_tree_node);
	Root->Tree = this;

	if (f_v) {
		cout << "syntax_tree::init_root_node done" << endl;
	}
}

void syntax_tree::init_int(
		field_theory::finite_field *Fq, int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init_int" << endl;
	}

	syntax_tree::Fq = Fq;

	init_root_node(verbose_level);


	//Root = NEW_OBJECT(syntax_tree_node);
	Root->init_empty_terminal_node_int(
			this,
			value,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::init_int done" << endl;
	}
}


void syntax_tree::print_to_vector(
		std::vector<std::string> &rep, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::print_to_vector "
				"before Root->print_subtree_to_vector" << endl;
	}
	Root->print_subtree_to_vector(rep, verbose_level);
	if (f_v) {
		cout << "syntax_tree::print_to_vector "
				"after Root->print_subtree_to_vector" << endl;
	}

}

void syntax_tree::print(std::ostream &ost)
{
	Root->print_subtree(ost);
}


void syntax_tree::print_easy(std::ostream &ost)
{
	Root->print_subtree_easy(ost);
}

void syntax_tree::print_monomial(
		std::ostream &ost, int *monomial)
{
	int i;

	for (i = 0; i < managed_variables.size(); i++) {
		if (monomial[i]) {
			ost << "*" << managed_variables[i];
			if (monomial[i] > 1) {
				ost << "^" << monomial[i];
			}
		}
	}
}

void syntax_tree::export_graphviz(
		std::string &name, std::ostream &ost)
{

	Root->export_graphviz(name, ost);

}


int syntax_tree::identify_single_literal(
		std::string &single_literal)
{
	int i;

	//cout << "syntax_tree::identify_single_literal trying to identify " << single_literal << endl;
	for (i = 0; i < managed_variables.size(); i++) {
		if (strcmp(single_literal.c_str(),
				managed_variables[i].c_str()) == 0) {
			//cout << "syntax_tree::identify_single_literal literal identified as managed variable " << i << endl;
			return i;
		}
	}
	return -1;
}

int syntax_tree::is_homogeneous(
		int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::is_homogeneous" << endl;
	}
	int ret;

	if (!f_has_managed_variables) {
		ret = false;
	}
	else {
		degree = -1;

		ret = Root->is_homogeneous(degree, verbose_level);
	}
	if (f_v) {
		cout << "syntax_tree::is_homogeneous done" << endl;
	}
	return ret;
}


void syntax_tree::split_by_monomials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **&Subtrees, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::split_by_monomials" << endl;
	}
	if (!f_has_managed_variables) {
		cout << "syntax_tree::split_by_monomials !f_has_managed_variables" << endl;
		exit(1);
	}


	int nb_monomials;
	int i;

	nb_monomials = Poly->get_nb_monomials();
	Subtrees = (syntax_tree_node **) NEW_pvoid(nb_monomials);

	for (i = 0; i < nb_monomials; i++) {
		Subtrees[i] = NULL;
	}
	Root->split_by_monomials(Poly, Subtrees, verbose_level);

	if (f_v) {
		for (i = 0; i < nb_monomials; i++) {
			cout << "Monomial " << i << " has subtree:" << endl;
			if (Subtrees[i]) {
				Subtrees[i]->print_subtree_easy(cout);
			}
			else {
				cout << "no subtree" << endl;
			}
		}
	}
	if (f_v) {
		cout << "syntax_tree::split_by_monomials done" << endl;
	}

}


void syntax_tree::substitute(
		std::vector<std::string> &variables,
		formula *Target,
		formula **S,
		syntax_tree *Output_tree,
		syntax_tree_node *Output_root_node,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::substitute" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::substitute before Root->substitute" << endl;
	}


	Root->substitute(
			variables,
			Target,
			S,
			this,
			Output_tree,
			Output_root_node,
			verbose_level);


	if (f_v) {
		cout << "syntax_tree::substitute after Root->substitute" << endl;
		cout << "syntax_tree::substitute Output_root_node=";
		Output_root_node->print_subtree_easy(cout);
		cout << endl;
	}



	if (f_v) {
		cout << "syntax_tree::substitute done" << endl;
	}
}

void syntax_tree::copy_to(
		syntax_tree *Output_tree,
		syntax_tree_node *Output_root_node,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::copy_to" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::copy_to "
				"before Root->copy_to" << endl;
	}
	Root->copy_to(
			Output_tree,
			Output_root_node,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::copy_to "
				"after Root->copy_to" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::copy_to done" << endl;
	}
}

void syntax_tree::simplify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::simplify" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::simplify "
				"before Root->simplify" << endl;
	}
	Root->simplify(
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::simplify "
				"after Root->simplify" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::simplify done" << endl;
	}
}


void syntax_tree::multiply_by_minus_one(
		field_theory::finite_field *Fq,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::multiply_by_minus_one" << endl;
	}

	int minus_one;

	minus_one = Fq->negate(1);

	syntax_tree_node *mult_node;

	mult_node = NEW_OBJECT(syntax_tree_node);

	mult_node->init_empty_multiplication_node(this, verbose_level);

	mult_node->add_numerical_factor(
				minus_one, verbose_level);

	mult_node->Nodes[mult_node->nb_nodes] = Root;
	mult_node->nb_nodes++;

	Root = mult_node;

	if (f_v) {
		cout << "syntax_tree::multiply_by_minus_one done" << endl;
	}

}


void syntax_tree::make_determinant(
		field_theory::finite_field *Fq,
		formula *V_in,
		int n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::make_determinant" << endl;
	}



	// For creating the set of permutations of n:
	combinatorics::combinatorics_domain Combi;
	long int a, N; // N = n factorial as long int
	int sgn;
	int *lehmer_code; // [n]
	int *perm; // [n]

	// For computing the number n factorial:
	ring_theory::longinteger_domain Long;
	ring_theory::longinteger_object result;


	lehmer_code = NEW_int(n);
	perm = NEW_int(n);

	Long.factorial(result, n);
	N = result.as_lint();

	if (f_v) {
		cout << "syntax_tree::make_determinant N = " << N << endl;
	}


	// We will use the determinantal formula
	// as the sum over all permutations
	// with a sign.
	// For this, we need the field element minus one:

	int minus_one;

	minus_one = Fq->negate(1);
		// minus one is the additive inverse of one.
		// In Orbiter, it is NOT represented as -1.
		// The representation depends on whether q is a prime or not.

	syntax_tree_node *add_node;

	// This add_node will be the root node of the determinantal expression.
	// It will have n factorial many summands,
	// one for each permutation of n.


	add_node = NEW_OBJECT(syntax_tree_node);

	add_node->init_empty_plus_node_with_exponent(
			this, 1 /* exponent */, verbose_level);

	Root = add_node;

	for (a = 0; a < N; a++) {

		// create the permutations in the order determined by the Lehmercode:

		if (a == 0) {
			Combi.first_lehmercode(n, lehmer_code);
		}
		else {
			Combi.next_lehmercode(n, lehmer_code);
		}
		Combi.lehmercode_to_permutation(
				n, lehmer_code, perm);


		if (f_v) {
			cout << "syntax_tree::make_determinant "
					"a = " << a << " / " << N
					<< " perm=";
			Int_vec_print(cout, perm, n);
			cout << endl;
		}

		sgn = Combi.perm_signum(perm, n);
			// sgn is either +1 or -1.


		syntax_tree_node *mult_node;

		mult_node = NEW_OBJECT(syntax_tree_node);

		mult_node->init_empty_multiplication_node(this, verbose_level);

		if (sgn == -1) {
			mult_node->add_numerical_factor(
					minus_one, verbose_level);
		}

		int i;
		for (i = 0; i < n; i++) {


			syntax_tree_node *node;

			node = NEW_OBJECT(syntax_tree_node);

			// Get the entry (i, pi(i)) from the input matrix
			// and add it as a factor:

			V_in[i * n + perm[i]].tree->Root->copy_to(
					this,
					node, verbose_level);

			mult_node->Nodes[mult_node->nb_nodes++] = node;

		}


		// add another summand to the addition:

		add_node->Nodes[add_node->nb_nodes++] = mult_node;


	}

	if (f_v) {
		cout << "syntax_tree::make_determinant before FREE_int" << endl;
	}
	FREE_int(perm);
	FREE_int(lehmer_code);
	if (f_v) {
		cout << "syntax_tree::make_determinant after FREE_int" << endl;
	}
	if (f_v) {
		cout << "syntax_tree::make_determinant done" << endl;
	}

}

int syntax_tree::compare_nodes(
		syntax_tree_node *Node1,
		syntax_tree_node *Node2,
		int verbose_level)
// -1 = the order of nodes is good.
// 0 = the nodes are of the same precedence
// 1 = the order of the nodes should be reversed
{
	int f_v = (verbose_level >= 1);
	int ret = 0;

	if (f_v) {
		cout << "syntax_tree::compare_nodes" << endl;
	}
	if (Node1->f_terminal) {

		if (Node2->f_terminal) {

			if (Node1->T->f_int) {
				if (Node1->T->f_int && Node2->T->f_int ) {
					return 0;
				}
				else {
					return -1;
				}
			}
			else if (Node2->T->f_int) {
				return 1;
			}

			if (Node1->T->f_text && Node2->T->f_text) {

				data_structures::string_tools ST;

				ret = ST.compare_string_string(Node1->T->value_text, Node2->T->value_text);

				return ret;
			}
		}
		return -1;
	}
	else if (Node2->f_terminal) {
		ret = 1;
	}
	else {
		ret = 0;
	}

	if (f_v) {
		cout << "syntax_tree::compare_nodes done" << endl;
	}
	return ret;
}



}}}


