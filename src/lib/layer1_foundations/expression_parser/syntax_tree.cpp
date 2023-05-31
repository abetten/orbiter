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
		field_theory::finite_field *Fq,
		int verbose_level)
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
		field_theory::finite_field *Fq, int value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init_int" << endl;
	}

	syntax_tree::Fq = Fq;

	init_root_node(verbose_level);


	Root->init_terminal_node_int(
			this,
			value,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::init_int done" << endl;
	}
}


void syntax_tree::print_to_vector(
		std::vector<std::string> &rep, int f_latex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::print_to_vector "
				"before Root->print_subtree_to_vector" << endl;
	}
	Root->print_subtree_to_vector(rep, f_latex, verbose_level);
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
		formula **Substitutions,
		syntax_tree *Output_tree,
		syntax_tree_node *Output_root_node,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::substitute" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::substitute "
				"before Root->substitute" << endl;
	}


	Root->substitute(
			variables,
			Target,
			Substitutions,
			this,
			Output_tree,
			Output_root_node,
			verbose_level);


	if (f_v) {
		cout << "syntax_tree::substitute "
				"after Root->substitute" << endl;
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


void syntax_tree::expand_in_place(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::expand_in_place" << endl;
	}



	if (f_v) {
		cout << "syntax_tree::expand_in_place "
				"before Root->expand_in_place_handle_exponents" << endl;
	}

	Root->expand_in_place_handle_exponents(
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::expand_in_place "
				"after Root->expand_in_place_handle_exponents" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::expand_in_place "
				"before Root->expand_in_place" << endl;
	}

	Root->expand_in_place(
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::expand_in_place "
				"after Root->expand_in_place" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::expand_in_place done" << endl;
	}
}

int syntax_tree::highest_order_term(
		std::string &variable, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::highest_order_term" << endl;
	}

	int d;

	if (f_v) {
		cout << "syntax_tree::highest_order_term before Root->highest_order_term" << endl;
	}
	d = Root->highest_order_term(variable, verbose_level);
	if (f_v) {
		cout << "syntax_tree::highest_order_term after Root->highest_order_term" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::highest_order_term done" << endl;
	}
	return d;
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

	mult_node->init_empty_multiplication_node(
			this, verbose_level);

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

				if (ret == 0) {
					int e1, e2;

					e1 = Node1->get_exponent();
					e2 = Node2->get_exponent();
					if (e1 > e2) {
						return 1;
					}
					else if (e1 < e2) {
						return -1;
					}
					ret = 0;
				}
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

void syntax_tree::make_linear_combination(
		syntax_tree_node *Node1a,
		syntax_tree_node *Node1b,
		syntax_tree_node *Node2a,
		syntax_tree_node *Node2b,
		int verbose_level)
// Creates Root = Node1a * Node1b + Node2a * Node2b
// All input nodes are copied.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::make_linear_combination" << endl;
	}

	syntax_tree_node *add_node;
	syntax_tree_node *mult_node1;
	syntax_tree_node *mult_node2;


	add_node = NEW_OBJECT(syntax_tree_node);

	add_node->init_empty_plus_node_with_exponent(
			this, 1 /* exponent */, verbose_level);


	mult_node1 = NEW_OBJECT(syntax_tree_node);
	mult_node2 = NEW_OBJECT(syntax_tree_node);


	mult_node1->init_empty_multiplication_node(this, verbose_level);
	mult_node2->init_empty_multiplication_node(this, verbose_level);


	add_node->Nodes[add_node->nb_nodes++] = mult_node1;
	add_node->Nodes[add_node->nb_nodes++] = mult_node2;


	syntax_tree_node *Node1a_copy;
	syntax_tree_node *Node1b_copy;
	syntax_tree_node *Node2a_copy;
	syntax_tree_node *Node2b_copy;

	Node1a_copy = NEW_OBJECT(syntax_tree_node);
	Node1b_copy = NEW_OBJECT(syntax_tree_node);
	Node2a_copy = NEW_OBJECT(syntax_tree_node);
	Node2b_copy = NEW_OBJECT(syntax_tree_node);

	Node1a->copy_to(this, Node1a_copy, verbose_level);
	Node1b->copy_to(this, Node1b_copy, verbose_level);
	Node2a->copy_to(this, Node2a_copy, verbose_level);
	Node2b->copy_to(this, Node2b_copy, verbose_level);

	mult_node1->Nodes[mult_node1->nb_nodes++] = Node1a_copy;
	mult_node1->Nodes[mult_node1->nb_nodes++] = Node1b_copy;

	mult_node2->Nodes[mult_node2->nb_nodes++] = Node2a_copy;
	mult_node2->Nodes[mult_node2->nb_nodes++] = Node2b_copy;


	Root = add_node;


	if (f_v) {
		cout << "syntax_tree::make_linear_combination done" << endl;
	}

}




void syntax_tree::latex_tree(
		std::string &name, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::latex_tree" << endl;
	}

	syntax_tree_latex L;

	L.latex_tree(name, Root, verbose_level);

	if (f_v) {
		cout << "syntax_tree::latex_tree done" << endl;
	}
}

void syntax_tree::latex_tree_split(
		std::string &name, int split_level, int split_mod,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::latex_tree_split" << endl;
	}

	syntax_tree_latex L;

	L.latex_tree_split(name, Root,
			split_level, split_mod,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::latex_tree_split done" << endl;
	}
}

void syntax_tree::collect_variables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::collect_variables" << endl;
	}

	Root->collect_variables(verbose_level);


	if (f_v) {
		cout << "syntax_tree::collect_variables the variables are:" << endl;
		print_variables(cout, verbose_level);
	}



	if (f_v) {
		cout << "syntax_tree::collect_variables done" << endl;
	}
}

void syntax_tree::print_variables(std::ostream &ost,
		int verbose_level)
{
	int i;

	for (i = 0; i < variables.size(); i++) {
		ost << i << " : " << variables[i] << endl;
	}
}

void syntax_tree::print_variables_in_line(std::ostream &ost)
{
	int i;

	for (i = 0; i < variables.size(); i++) {
		ost << variables[i];
		if (i < variables.size() - 1) {
			ost << ",";
		}
	}
}

int syntax_tree::find_variable(
		std::string var,
		int verbose_level)
{
	data_structures::string_tools String;


	int i, cmp;

	for (i = 0; i < variables.size(); i++) {
		cmp = String.compare_string_string(variables[i], var);
		if (cmp == 0) {
			return i;
		}
	}
	return -1;
}

void syntax_tree::add_variable(std::string &var)
{
	data_structures::string_tools String;
	int cmp;
	std::vector<std::string>::iterator it;


	for (it = variables.begin(); it < variables.end(); it++) {
		cmp = String.compare_string_string(*it, var);
		if (cmp > 0) {
			variables.insert(it, var);
			return;
		}
	}
	variables.push_back(var);
}


}}}


