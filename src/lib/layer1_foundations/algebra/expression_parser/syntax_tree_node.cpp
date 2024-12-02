/*
 * syntax_tree_node.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace expression_parser {


static int syntax_tree_node_compare_func(
		void *data,
		int i, int j, void *extra_data);
static void syntax_tree_node_swap_func(
		void *data,
		int i, int j, void *extra_data);


syntax_tree_node::syntax_tree_node()
{
	Record_birth();
	Tree = NULL;
	idx = other::orbiter_kernel_system::Orbiter->syntax_tree_node_index;

	f_has_exponent = false;
	exponent = 0;

	other::orbiter_kernel_system::Orbiter->syntax_tree_node_index++;

	if ((other::orbiter_kernel_system::Orbiter->syntax_tree_node_index % 1000000) == 0) {
		cout << "Number of algebra nodes = "
				<< other::orbiter_kernel_system::Orbiter->syntax_tree_node_index << endl;
	}

	f_terminal = false;
	T = NULL;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;

	f_has_monomial = false;
	monomial = NULL;

	f_has_minus = false;

	nb_nodes_allocated = 0;
	Nodes = NULL;
#if 0
	Nodes = (syntax_tree_node **) NEW_pvoid(nb_nodes_allocated);
	int i;
	for (i = 0; i < nb_nodes_allocated; i++) {
		Nodes[i] = NULL;
	}
#endif
}

syntax_tree_node::~syntax_tree_node()
{
	Record_death();

	if (f_terminal) {
		FREE_OBJECT(T);
		T = 0L;
	}
	else {
		if (monomial) {
			FREE_int(monomial);
		}
	}
	if (nb_nodes_allocated) {

		int i;

		for (i = 0; i < nb_nodes; i++) {
			FREE_OBJECT(Nodes[i]);
		}
		FREE_pvoid((void **) Nodes);
		Nodes = NULL;
		nb_nodes_allocated = 0;
	}
}


void syntax_tree_node::null()
{
	idx = 0;

	f_terminal = false;
	T = 0L;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;
	nb_nodes_allocated = 0;
	Nodes = NULL;
}

void syntax_tree_node::add_numerical_factor(
		int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_factor" << endl;
	}

	syntax_tree_node *fresh_node;

	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_terminal_node_int(
			Tree, value, verbose_level - 2);

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_factor "
				"before append_node" << endl;
	}
	append_node(fresh_node, 0 /* verbose_level */);
	if (f_v) {
		cout << "syntax_tree_node::add_numerical_factor "
				"after append_node" << endl;
	}


	if (f_v) {
		cout << "syntax_tree_node::add_numerical_factor done" << endl;
	}
}

void syntax_tree_node::add_numerical_summand(
		int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_summand" << endl;
	}
	syntax_tree_node *fresh_node;

	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_terminal_node_int(
			Tree, value, verbose_level - 2);

	append_node(fresh_node, 0 /* verbose_level */);


	if (f_v) {
		cout << "syntax_tree_node::add_numerical_summand done" << endl;
	}
}

int syntax_tree_node::text_value_match(
		std::string &factor)
{
	other::data_structures::string_tools ST;

	if (f_terminal && T->f_text &&
				ST.compare_string_string(T->value_text, factor) == 0) {
		return true;
	}
	else {
		return false;
	}
}


// entrance point 4 -> init_terminal_node_text_with_exponent


void syntax_tree_node::add_factor(
		std::string &factor, int exponent,
		int verbose_level)
// We assume that the current node is a multiplication node.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_factor" << endl;
	}

	if (type != operation_type_mult) {
		cout << "syntax_tree_node::add_factor "
				"must be multiplication node" << endl;
		exit(1);
	}

	int i;
	other::data_structures::string_tools ST;

	for (i = 0; i < nb_nodes; i++) {
		if (Nodes[i]->text_value_match(factor)) {
			if (Nodes[i]->f_has_exponent) {
				Nodes[i]->exponent += exponent;
			}
			else {
				Nodes[i]->f_has_exponent = true;
				Nodes[i]->exponent = 1 + exponent;
			}
			break;
		}
	}
	if (i == nb_nodes) {

		syntax_tree_node *fresh_node;

		fresh_node = NEW_OBJECT(syntax_tree_node);

		fresh_node->init_terminal_node_text_with_exponent(
				Tree, factor, exponent, verbose_level - 2);

		append_node(fresh_node, 0 /* verbose_level */);

		if (f_v) {
			cout << "syntax_tree_node::add_factor "
					"new nb_nodes = " << nb_nodes << endl;
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::add_factor done" << endl;
	}
}


// entrance point 3 -> init_terminal_node_text


void syntax_tree_node::add_summand(
		std::string &summand, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_summand" << endl;
	}
	if (type != operation_type_add) {
		cout << "syntax_tree_node::add_summand "
				"must be addition node" << endl;
		exit(1);
	}

	syntax_tree_node *fresh_node;


	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_terminal_node_text(
			Tree, summand, verbose_level - 2);

	append_node(fresh_node, 0 /* verbose_level */);

	if (f_v) {
		cout << "syntax_tree_node::add_summand done" << endl;
	}
}

void syntax_tree_node::init_terminal_node_int(
		syntax_tree *Tree,
		int value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_int" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = true;
	f_has_exponent = false;
	T = NEW_OBJECT(syntax_tree_node_terminal);
	T->f_int = true;
	T->value_int = value;
	type = operation_type_nothing;
	nb_nodes = 0;
	nb_nodes_allocated = 0;
	Nodes = NULL;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_int done" << endl;
	}
}

void syntax_tree_node::init_monopoly(
		syntax_tree *Tree,
		std::string &variable,
		int *coeffs, int nb_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_monopoly" << endl;
	}

	init_empty_plus_node_with_exponent(
				Tree,
				1 /* exponent */,
				verbose_level - 2);

	int i, c;

	for (i = 0; i < nb_coeffs; i++) {
		c = coeffs[i];
		if (c == 0) {
			continue;
		}
		if (i == 0) {
			syntax_tree_node *fresh_node;

			fresh_node = NEW_OBJECT(syntax_tree_node);
			fresh_node->init_terminal_node_int(
					Tree,
					c,
					0 /* verbose_level */);
			append_node(fresh_node, verbose_level - 2);
		}
		else {

			if (c == 1) {
				syntax_tree_node *fresh_node;

				fresh_node = NEW_OBJECT(syntax_tree_node);
				fresh_node->init_terminal_node_text_with_exponent(
						Tree,
						variable,
						i /* exp */,
						verbose_level - 2);

				append_node(fresh_node, verbose_level - 2);

			}
			else {
				syntax_tree_node *mult_node;

				mult_node = NEW_OBJECT(syntax_tree_node);
				mult_node->init_empty_multiplication_node(
						Tree,
						0 /* verbose_level */);


				syntax_tree_node *fresh_node1;
				syntax_tree_node *fresh_node2;

				fresh_node1 = NEW_OBJECT(syntax_tree_node);
				fresh_node1->init_terminal_node_int(
						Tree, c, verbose_level - 2);
				mult_node->append_node(fresh_node1, verbose_level - 2);


				fresh_node2 = NEW_OBJECT(syntax_tree_node);
				fresh_node2->init_terminal_node_text_with_exponent(
						Tree,
						variable,
						i /* exp */,
						verbose_level - 2);
				mult_node->append_node(fresh_node2, verbose_level - 2);

				append_node(mult_node, verbose_level - 2);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::init_monopoly done" << endl;
	}
}




// entrance point 1 -> init_terminal_node_text_with_exponent


void syntax_tree_node::init_terminal_node_text(
		syntax_tree *Tree,
		std::string &value_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_text" << endl;
	}

	init_terminal_node_text_with_exponent(
			Tree, value_text, 1 /* exponent */, verbose_level - 2);

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_text done" << endl;
	}
}


// entrance point 2


void syntax_tree_node::init_terminal_node_text_with_exponent(
		syntax_tree *Tree,
		std::string &value_text,
		int exp,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_text_with_exponent" << endl;
	}

	if (Find_symbol(value_text) == -1) {
		if (f_v) {
			cout << "syntax_tree_node::init_terminal_node_text_with_exponent "
					"the object does not exist: name = " << value_text << endl;
		}
		syntax_tree_node::Tree = Tree;
		f_terminal = true;

		if (exp != 1) {
			f_has_exponent = true;
			syntax_tree_node::exponent = exp;
		}
		else {
			f_has_exponent = false;
			syntax_tree_node::exponent = 0;
		}
		T = NEW_OBJECT(syntax_tree_node_terminal);
		T->f_text = true;
		T->value_text.assign(value_text);
		type = operation_type_nothing;
		nb_nodes = 0;
		Nodes = NULL;
		f_has_monomial = false;
		f_has_minus = false;
	}
	else {
		if (f_v) {
			cout << "syntax_tree_node::init_terminal_node_text_with_exponent "
					"the object exists: name = " << value_text << endl;
		}

		expression_parser::symbolic_object_builder *Builder;

		Builder = Get_symbol(value_text);

		formula_vector *Formula_vector;

		Formula_vector = Builder->Formula_vector;

		if (Formula_vector->len > 1) {
			cout << "syntax_tree_node::init_terminal_node_text_with_exponent "
					"the object exists: name = " << value_text
					<< " but it is not of scalar type" << endl;
			exit(1);
		}

		formula *V;

		V = &Formula_vector->V[0];

		syntax_tree *tree;

		tree = V->tree;


		if (f_v) {
			cout << "syntax_tree_node::init_terminal_node_text_with_exponent "
					"before tree->copy_to" << endl;
		}

		tree->copy_to(
				Tree,
				this,
				verbose_level - 2);

		if (f_v) {
			cout << "syntax_tree_node::init_terminal_node_text_with_exponent "
					"after tree->copy_to" << endl;
			//print_subtree_easy(cout);
			//cout << endl;
		}

		if (exp != 1) {
			if (f_has_exponent) {
				exponent *= exp;
			}
			else {
				f_has_exponent = true;
				exponent = exp;
			}
		}

	}

	if (f_v) {
		cout << "syntax_tree_node::init_terminal_node_text_with_exponent done" << endl;
	}
}


void syntax_tree_node::add_empty_plus_node_with_exponent(
		syntax_tree *Tree,
		int exponent, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent" << endl;
	}

	syntax_tree_node *fresh_node;

	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_empty_plus_node_with_exponent(
			Tree, exponent, verbose_level - 2);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent "
				"before append_node" << endl;
	}
	append_node(fresh_node, verbose_level - 2);
	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent "
				"after append_node" << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent "
				"done" << endl;
	}
}

void syntax_tree_node::init_empty_plus_node_with_exponent(
		syntax_tree *Tree,
		int exponent,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_plus_node_with_exponent" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = false;
	if (exponent != 1) {
		f_has_exponent = true;
		syntax_tree_node::exponent = exponent;
	}
	else {
		f_has_exponent = false;
		syntax_tree_node::exponent = 0;
	}
	type = operation_type_add;
	nb_nodes = 0;
	nb_nodes_allocated = 0;
	Nodes = NULL;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_plus_node_with_exponent "
				"before reallocate" << endl;
	}
	reallocate(1, verbose_level - 2);
	if (f_v) {
		cout << "syntax_tree_node::init_empty_plus_node_with_exponent "
				"after reallocate" << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::init_empty_plus_node_with_exponent "
				"done" << endl;
	}
}



void syntax_tree_node::add_empty_multiplication_node(
		syntax_tree *Tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node" << endl;
	}

	syntax_tree_node *fresh_node;


	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_empty_multiplication_node(
			Tree, verbose_level - 2);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node "
				"before append_node" << endl;
	}
	append_node(fresh_node, 0 /* verbose_level */);
	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node "
				"after append_node" << endl;
	}


	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node done" << endl;
	}
}

void syntax_tree_node::init_empty_multiplication_node(
		syntax_tree *Tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_multiplication_node" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = false;
	f_has_exponent = false;
	exponent = 0;
	type = operation_type_mult;
	nb_nodes = 0;
	nb_nodes_allocated = 0;
	Nodes = NULL;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_multiplication_node done" << endl;
	}
}

void syntax_tree_node::add_empty_node(
		syntax_tree *Tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_node" << endl;
	}


	syntax_tree_node *fresh_node;

	fresh_node = NEW_OBJECT(syntax_tree_node);

	fresh_node->init_empty_node(
			Tree, verbose_level - 2);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_node "
				"before append_node" << endl;
	}
	append_node(fresh_node, 0 /* verbose_level */);
	if (f_v) {
		cout << "syntax_tree_node::add_empty_node "
				"after append_node" << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::add_empty_node done" << endl;
	}
}

void syntax_tree_node::init_empty_node(
		syntax_tree *Tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_node" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = false;
	f_has_exponent = false;
	exponent = 0;
	type = operation_type_nothing;
	nb_nodes = 0;
	nb_nodes_allocated = 0;
	Nodes = NULL;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_node done" << endl;
	}
}




void syntax_tree_node::split_by_monomials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **Subtrees,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::split_by_monomials" << endl;
		cout << "syntax_tree_node::split_by_monomials Node " << idx << endl;
	}
	if (f_terminal) {
		return;
	}
	else {
		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::split_by_monomials "
						"checking multiplication node" << endl;
			}
			idx = Poly->index_of_monomial(monomial);
			Subtrees[idx] = this;
		}
		else {
			int i;

			if (f_v) {
				cout << "syntax_tree_node::split_by_monomials "
						"splitting subtree" << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->split_by_monomials(Poly, Subtrees, verbose_level - 2);
			}
		}
	}
}

int syntax_tree_node::is_homogeneous(
		int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int deg, i;

	if (f_v) {
		cout << "syntax_tree_node::is_homogeneous Node " << idx << endl;
	}
	if (f_terminal) {
		return true;
	}
	else {
		if (type == operation_type_mult) {
			if (f_v) {
				cout << "checking multiplication node" << endl;
			}
			deg = 0;
			for (i = 0; i < Tree->managed_variables.size(); i++) {
				deg += monomial[i];
			}
			if (f_v) {
				cout << "syntax_tree_node::is_homogeneous "
						"node " << idx << " has degree " << deg << endl;
			}
			if (degree == -1) {
				degree = deg;
				if (f_v) {
					cout << "syntax_tree_node::is_homogeneous "
							"node " << idx << " setting degree to " << degree << endl;
				}
			}
			else {
				if (deg != degree) {
					if (f_v) {
						cout << "syntax_tree_node::is_homogeneous "
								"node " << idx << " has degree " << deg
								<< " which is different from " << degree
								<< ", so not homogeneous" << endl;
					}
					return false;
				}
			}
			return true;
		}
		else {
			int i, ret;

			if (f_v) {
				cout << "checking subtree" << endl;
			}
			ret = true;
			for (i = 0; i < nb_nodes; i++) {
				ret = Nodes[i]->is_homogeneous(degree, verbose_level - 2);
				if (ret == false) {
					return false;
				}
			}
			return ret;
		}
	}

}




int syntax_tree_node::is_mult_node()
{
	if (type == operation_type_mult) {
		return true;
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_add_node()
{
	if (type == operation_type_add) {
		return true;
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_int_node()
{
	if (f_terminal && T->f_int) {
		return true;
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_text_node()
{
	if (f_terminal && T->f_text) {
		return true;
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_monomial()
{

	if (f_terminal) {
		return true;
	}
	if (type == operation_type_add) {
		return false;
	}

	int i;

	for (i = 0; i < nb_nodes; i++) {
		if (!Nodes[i]->f_terminal) {
			return false;
		}
	}
	return true;
}

int syntax_tree_node::is_this_variable(
		std::string &variable)
{
	other::data_structures::string_tools ST;

	if (!is_text_node()) {
		return false;
	}
	if (ST.compare_string_string(T->value_text, variable) == 0) {
		return true;
	}
	return false;
}


int syntax_tree_node::highest_order_term(
		std::string &variable, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::highest_order_term" << endl;
	}
	int d = 0, i, d1;

	if (f_terminal) {
		d = exponent_of_variable(variable, verbose_level - 2);
	}
	else {
		if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::highest_order_term add node, "
						"nb_nodes = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				if (f_v) {
					cout << "syntax_tree_node::highest_order_term "
							"child " << i << " / " << nb_nodes << endl;
				}
				d1 = Nodes[i]->exponent_of_variable(
						variable, verbose_level - 2);
				if (f_v) {
					cout << "syntax_tree_node::highest_order_term "
							"child " << i << " / " << nb_nodes
							<< " d1=" << d1 << endl;
				}
				if (d1 > d) {
					d = d1;
				}
			}
		}
		else {
			cout << "syntax_tree_node::highest_order_term "
					"multiplication node is not allowed" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "syntax_tree_node::highest_order_term done d=" << d << endl;
	}
	return d;
}

void syntax_tree_node::get_monopoly(
		std::string &variable,
		std::vector<int> &Coeff, std::vector<int> &Exp,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::get_monopoly" << endl;
	}
	int i;

	int coeff, exp;

	if (f_terminal) {
		coeff = 1;
		exp = exponent_of_variable(variable, verbose_level - 2);
		Coeff.push_back(coeff);
		Exp.push_back(exp);
	}
	else {
		if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::get_monopoly add node, "
						"nb_nodes = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				if (f_v) {
					cout << "syntax_tree_node::get_monopoly "
							"child " << i << " / " << nb_nodes << endl;
				}
				Nodes[i]->get_exponent_and_coefficient_of_variable(
						variable, coeff, exp, verbose_level - 2);
				Coeff.push_back(coeff);
				Exp.push_back(exp);
			}
		}
		else {
			cout << "syntax_tree_node::get_monopoly "
					"multiplication node is not allowed" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "syntax_tree_node::get_monopoly done" << endl;
	}
}

void syntax_tree_node::get_multipoly(
		ring_theory::homogeneous_polynomial_domain *HPD,
		int *&eqn, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::get_multipoly" << endl;
	}
	int i;

	sz = HPD->get_nb_monomials();
	eqn = NEW_int(HPD->get_nb_monomials());
	Int_vec_zero(eqn, HPD->get_nb_monomials());

	std::map<std::string,int> variable_idx;
	int *monomial;
	int coeff;


	monomial = NEW_int(HPD->nb_variables);

	for (i = 0; i < HPD->nb_variables; i++) {
		variable_idx[HPD->get_symbol(i)] = i;
	}


	if (type == operation_type_add) {
		if (f_v) {
			cout << "syntax_tree_node::get_multipoly "
					"addition node" << endl;
		}
		if (f_v) {
			cout << "syntax_tree_node::get_monomial add node, "
					"nb_nodes = " << nb_nodes << endl;
		}
		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::get_monomial "
						"child " << i << " / " << nb_nodes << endl;
			}
			Int_vec_zero(monomial, HPD->nb_variables);
			coeff = 1;
			Nodes[i]->get_monomial(variable_idx, HPD->nb_variables,
					monomial, coeff, verbose_level);
			idx = HPD->index_of_monomial(monomial);
			if (f_v) {
				cout << "syntax_tree_node::get_monomial "
						"child " << i << " / " << nb_nodes << " monomial = ";
				Int_vec_print(cout, monomial, HPD->nb_variables);
				cout << " : coeff = " << coeff << " idx = " << idx;
				cout << endl;
			}
			eqn[idx] = HPD->get_F()->add(eqn[idx], coeff);
		}
	}
	else if (type == operation_type_mult) {
		cout << "syntax_tree_node::get_multipoly "
				"multiplication node is not allowed" << endl;
		exit(1);
	}

	FREE_int(monomial);

	if (f_v) {
		cout << "syntax_tree_node::get_multipoly done" << endl;
	}
}

void syntax_tree_node::get_monomial(
		std::map<std::string,int> &variable_idx, int nb_variables,
		int *exponent_vector, int &coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::get_monomial" << endl;
	}
	int i;




	if (f_terminal) {
		if (T->f_text) {
			exponent_vector[variable_idx[T->value_text]] += get_exponent();
		}
		else if (T->f_int) {
			coeff = Tree->Fq->mult(coeff, T->value_int);
		}
		else {
			cout << "syntax_tree_node::get_monomial "
					"illegal type of terminal node" << endl;
			exit(1);
		}
	}
	else {
		if (type == operation_type_add) {
			cout << "syntax_tree_node::get_monomial "
					"addition node is not allowed" << endl;
			exit(1);
		}
		else {
			if (f_v) {
				cout << "syntax_tree_node::get_monomial add node, "
						"nb_nodes = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				if (f_v) {
					cout << "syntax_tree_node::get_monomial "
							"child " << i << " / " << nb_nodes << endl;
				}
				Nodes[i]->get_monomial(
						variable_idx, nb_variables, exponent_vector,
						coeff,
						verbose_level);
			}
		}
	}
	if (f_v) {
		cout << "syntax_tree_node::get_monomial done" << endl;
	}
}


void syntax_tree_node::get_exponent_and_coefficient_of_variable(
		std::string &variable, int &coeff, int &exp,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable" << endl;
	}


	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
					"terminal node" << endl;
		}
		if (is_this_variable(variable)) {
			exp = get_exponent();
			coeff = 1;
		}
		else {
			if (is_int_node()) {
				exp = 0;
				coeff = T->value_int;

				if (f_has_exponent) {
					coeff = Tree->Fq->power(coeff, get_exponent());
				}
			}
			else {
				cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
						"unkown node type" << endl;
				exit(1);
			}
		}
	}
	else {
		if (type == operation_type_add) {
			cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
					"cannot be add node" << endl;
			exit(1);
		}

		int i;

		coeff = -1;
		exp = -1;

		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
						"child " << i << " / " << nb_nodes << endl;
			}

			if (Nodes[i]->f_terminal) {
				if (f_v) {
					cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
							"child " << i << " / " << nb_nodes << " is terminal node" << endl;
				}
				if (Nodes[i]->is_int_node()) {
					if (f_v) {
						cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
								"child " << i << " / " << nb_nodes << " is int node" << endl;
					}
					if (coeff != -1) {
						cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
								"malformed, cannot have multiple int nodes" << endl;
						exit(1);
					}
					coeff = Nodes[i]->T->value_int;
					if (Nodes[i]->f_has_exponent) {
						coeff = Tree->Fq->power(coeff, Nodes[i]->get_exponent());
					}

				}
				else if (Nodes[i]->is_this_variable(variable)) {
					if (exp != -1) {
						cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
								"malformed, cannot have multiple int nodes" << endl;
						exit(1);
					}
					exp = Nodes[i]->get_exponent();
				}
				else {
					cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
							"malformed, cannot have multiple int nodes" << endl;
					exit(1);
				}
			}
			else {
				cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
						"malformed, cannot have multiple int nodes" << endl;
				exit(1);
			}
		}
		if (coeff == -1 && exp != -1) {
			coeff = 1;
		}
		if (exp == -1 && coeff != -1) {
			exp = 0;
		}
		if (f_has_exponent) {
			coeff = Tree->Fq->power(coeff, get_exponent());
			exp *= get_exponent();
		}

	}
	if (f_v) {
		cout << "syntax_tree_node::get_exponent_and_coefficient_of_variable "
				"done, coeff = " << coeff << " exp = " << exp << endl;
	}
}

int syntax_tree_node::exponent_of_variable(
		std::string &variable, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::exponent_of_variable" << endl;
	}

	int exp = 0;

	if (f_terminal) {
		if (is_this_variable(variable)) {
			exp = get_exponent();
		}
		else {
			exp = 0;
		}
	}
	else {
		if (type == operation_type_add) {
			cout << "syntax_tree_node::exponent_of_variable "
					"cannot be add node" << endl;
			exit(1);
		}

		int i, e;

		exp = 0;

		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::exponent_of_variable" << endl;
			}
			e = Nodes[i]->exponent_of_variable(
					variable,
					0 /* verbose_level */);
			if (f_v) {
				cout << "syntax_tree_node::exponent_of_variable "
						"child " << i << " has degree " << e << endl;
			}
			exp += e;
		}

		exp *= get_exponent();
	}
	if (f_v) {
		cout << "syntax_tree_node::exponent_of_variable "
				"done, exp = " << exp << endl;
	}
	return exp;
}

int syntax_tree_node::exponent_of_variable_destructive(
		std::string &variable)
{

	int exp = 0;

	if (f_terminal) {
		if (is_this_variable(variable)) {
			exp = get_exponent();
			T->f_int = true;
			T->f_text = false;
			T->value_int = 1;
			f_has_exponent = false;
			exponent = 0;
			return exp;
		}
		else {
			return 0;
		}
	}
	if (type == operation_type_add) {
		cout << "syntax_tree_node::exponent_of_variable_destructive "
				"cannot be add node" << endl;
		exit(1);
	}

	int i, e;

	exp = 0;

	for (i = 0; i < nb_nodes; i++) {
		e = Nodes[i]->exponent_of_variable(
				variable, 0 /* verbose_level */);
		if (e) {

			if (nb_nodes > 1) {
				delete_one_child(i, 0 /*verbose_level*/);
				i--;
			}
			else {
				if (!Nodes[i]->f_terminal) {
					cout << "syntax_tree_node::exponent_of_variable_destructive "
							"!Nodes[i]->f_terminal" << endl;
					exit(1);
				}
				else {
					// it is a terminal node:

					Nodes[i]->f_terminal = true; // redundant
					Nodes[i]->T->f_int = true;
					Nodes[i]->T->value_int = 1;
				}
			}
		}
		exp += e;
	}

	exp *= get_exponent();
	return exp;
}


int syntax_tree_node::get_exponent()
{
	if (f_has_exponent) {
		return exponent;
	}
	else {
		return 1;
	}
}

int syntax_tree_node::evaluate(
		std::map<std::string, std::string> &symbol_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int a, b;

	if (f_v) {
		cout << "syntax_tree_node::evaluate" << endl;
	}
	if (f_terminal) {
		a = T->evaluate(symbol_table, Tree->Fq, verbose_level - 2);
		if (f_has_minus) {
			a = Tree->Fq->negate(a);
		}
		if (f_v) {
			cout << "syntax_tree_node::evaluate "
					"terminal node evaluates to " << a << endl;
		}
	}
	else {
		if (nb_nodes == 1) {
			a = Nodes[0]->evaluate(symbol_table, verbose_level - 2);
			if (f_has_minus) {
				a = Tree->Fq->negate(a);
			}
			if (f_v) {
				cout << "syntax_tree_node::evaluate "
						"single node evaluates to " << a << endl;
			}
		}
		else {
			if (type == operation_type_mult) {
				a = 1;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, verbose_level - 2);
					a = Tree->Fq->mult(a, b);
				}
				if (f_has_minus) {
					a = Tree->Fq->negate(a);
				}
				if (f_v) {
					cout << "syntax_tree_node::evaluate "
							"product evaluates to " << a << endl;
				}
			}
			else if (type == operation_type_add) {
				a = 0;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, verbose_level - 2);
					a = Tree->Fq->add(a, b);
				}
				if (f_v) {
					cout << "syntax_tree_node::evaluate "
							"sum evaluates to " << a << endl;
				}
			}
			else {
				cout << "syntax_tree_node::evaluate unknown operation" << endl;
				exit(1);
			}
		}
	}
	if (f_has_exponent) {
		if (f_v) {
			cout << "syntax_tree_node::evaluate "
					"before raising to the power of "
					<< exponent << ", a=" << a << endl;
		}
		a = Tree->Fq->power(a, exponent);
		if (f_v) {
			cout << "syntax_tree_node::evaluate "
					"after raising to the power of "
					<< exponent << ", a=" << a << endl;
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::evaluate done, value = " << a << endl;
	}
	return a;
}


void syntax_tree_node::push_a_minus_sign()
{
	if (f_has_minus) {
		f_has_minus = false;
	}
	else {
		f_has_minus = true;
	}

}


void syntax_tree_node::copy_to(
		algebra::expression_parser::syntax_tree *Output_tree,
		algebra::expression_parser::syntax_tree_node *Output_node,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::copy_to" << endl;
	}

	Output_node->Tree = Output_tree;

	Output_node->f_has_exponent = f_has_exponent;
	Output_node->exponent = exponent;

	Output_node->f_has_monomial = f_has_monomial;
	Output_node->monomial = monomial;

	Output_node->f_has_minus = f_has_minus;
	Output_node->nb_nodes = 0;

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::copy_to "
					"terminal node" << endl;
		}
		Output_node->f_terminal = true;
		Output_node->T = NEW_OBJECT(syntax_tree_node_terminal);

		if (T->f_int) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"terminal node int" << endl;
			}
			Output_node->T->f_int = true;
			Output_node->T->value_int = T->value_int;
		}
		else if (T->f_double) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"terminal node double" << endl;
			}
			Output_node->T->f_double = true;
			Output_node->T->value_double = T->value_double;
		}
		else if (T->f_text) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"terminal node text" << endl;
			}
			Output_node->T->f_text = true;
			Output_node->T->value_text = T->value_text;
		}

	}
	else {
		int i;


		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"multiplication node" << endl;
			}
			Output_node->type = operation_type_mult;
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"addition node" << endl;
			}
			Output_node->type = operation_type_add;
		}
		else if (type == operation_type_nothing) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to "
						"nothing node" << endl;
			}
			Output_node->type = operation_type_nothing;
		}
		else {
			cout << "syntax_tree_node::copy_to "
					"unknown operation" << endl;
			exit(1);
		}

		Output_node->nb_nodes = 0;
		Output_node->nb_nodes_allocated = 0;
		Output_node->Nodes = NULL;

		if (f_v) {
			cout << "syntax_tree_node::copy_to "
					"before Output_node->reallocate" << endl;
		}
		Output_node->reallocate(nb_nodes, verbose_level - 2);
		if (f_v) {
			cout << "syntax_tree_node::copy_to "
					"after Output_node->reallocate" << endl;
		}

		for (i = 0; i < nb_nodes; i++) {

			syntax_tree_node *Output_node2;


			Output_node2 = NEW_OBJECT(syntax_tree_node);

			if (f_v) {
				cout << "syntax_tree_node::copy_to child "
						<< i << " / " << nb_nodes << endl;
			}
			Nodes[i]->copy_to(
					Output_tree,
					Output_node2,
					verbose_level - 2);

			//Output_node2->Tree = Output_tree;


			Output_node->append_node(Output_node2, 0 /* verbose_level */);

			if (f_v) {
				cout << "syntax_tree_node::copy_to child "
						<< i << " / " << nb_nodes << " done" << endl;
			}
		}

		//Output_node->nb_nodes = nb_nodes;
	}


	if (f_v) {
		cout << "syntax_tree_node::copy_to done" << endl;
	}
}

void syntax_tree_node::substitute(
		std::vector<std::string> &variables,
		formula *Target,
		formula **Substitutions,
		expression_parser::syntax_tree *Input_tree,
		expression_parser::syntax_tree *Output_tree,
		expression_parser::syntax_tree_node *Output_node,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::substitute" << endl;
	}

	Output_node->Tree = Output_tree;

	Output_node->f_has_exponent = f_has_exponent;
	Output_node->exponent = exponent;

	Output_node->f_has_monomial = f_has_monomial;
	Output_node->monomial = monomial;

	Output_node->f_has_minus = f_has_minus;


	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::substitute terminal node" << endl;
		}
		Output_node->f_terminal = true;
		Output_node->T = NEW_OBJECT(syntax_tree_node_terminal);

		if (T->f_int) {
			Output_node->T->f_int = true;
			Output_node->T->value_int = T->value_int;
		}
		else if (T->f_double) {
			Output_node->T->f_double = true;
			Output_node->T->value_double = T->value_double;
		}
		else if (T->f_text) {
			Output_node->T->f_text = true;
			Output_node->T->value_text = T->value_text;
			int i;
			for (i = 0; i < variables.size(); i++) {
				if (strcmp(T->value_text.c_str(), variables[i].c_str()) == 0) {
					if (f_v) {
						cout << "syntax_tree_node::substitute "
								"seeing variable " << i << " = " << variables[i] << endl;
					}
					break;
				}
			}
			if (i < variables.size()) {
				Output_node->f_terminal = false;
				FREE_OBJECT(Output_node->T);
				Output_node->T = NULL;

				if (f_v) {
					cout << "syntax_tree_node::substitute "
							"before Substitutions[i]->tree->Root->copy_to" << endl;
				}
				Substitutions[i]->tree->Root->copy_to(
						Output_tree,
						Output_node,
						0 /*verbose_level*/);
				if (f_v) {
					cout << "syntax_tree_node::substitute "
							"after Substitutions[i]->tree->Root->copy_to" << endl;
				}

			}
		}

	}
	else {
		int i;
		if (f_v) {
			cout << "syntax_tree_node::substitute "
					"not a terminal node" << endl;
		}


		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"multiplication node" << endl;
			}
			Output_node->type = operation_type_mult;
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"addition node" << endl;
			}
			Output_node->type = operation_type_add;
		}
		else if (type == operation_type_nothing) {
			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"nothing node" << endl;
			}
			cout << "syntax_tree_node::substitute "
					"operation_type_nothing is not allowed" << endl;
			exit(1);
		}
		else {
			cout << "syntax_tree_node::substitute "
					"unknown operation" << endl;
			exit(1);
		}


		Output_node->nb_nodes = 0;
		Output_node->nb_nodes_allocated = 0;
		Output_node->Nodes = NULL;

		for (i = 0; i < nb_nodes; i++) {

			syntax_tree_node *Output_node2;


			Output_node2 = NEW_OBJECT(syntax_tree_node);

			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"child " << i << " / " << nb_nodes << endl;
			}
			Nodes[i]->substitute(
					variables,
					Target,
					Substitutions,
					Input_tree,
					Output_tree,
					Output_node2,
					verbose_level - 2);

			//Output_node2->Tree = Output_tree;

			Output_node2->f_has_exponent = Nodes[i]->f_has_exponent;
			Output_node2->exponent = Nodes[i]->exponent;

			Output_node2->f_has_monomial = Nodes[i]->f_has_monomial;
			Output_node2->monomial = Nodes[i]->monomial; // ToDo

			Output_node2->f_has_minus = Nodes[i]->f_has_minus;


			Output_node->append_node(Output_node2, 0 /* verbose_level */);
			//Output_node->Nodes[i] = Output_node2;

			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"child " << i << " / " << nb_nodes << " done" << endl;
			}
		}

		//Output_node->nb_nodes = nb_nodes;
	}

	if (f_has_exponent) {
		Output_node->f_has_exponent = true;
		Output_node->exponent = exponent;
	}


	if (f_v) {
		cout << "syntax_tree_node::substitute done" << endl;
	}
}

void syntax_tree_node::simplify(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::simplify" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::simplify terminal node" << endl;
		}
		if (T->f_int) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"terminal node of type int" << endl;
			}
			int val1, val2;
			val1 = T->value_int;
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"terminal node of type int val1=" << val1 << endl;
			}
			if (f_has_exponent) {
				int exp = exponent;
				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"exponent=" << exp << endl;
				}

				if (Tree == NULL) {
					cout << "syntax_tree_node::simplify "
							"Tree == NULL" << endl;
					exit(1);
				}
				if (Tree->Fq == NULL) {
					cout << "syntax_tree_node::simplify "
							"Tree->Fq == NULL" << endl;
					exit(1);
				}
				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"before Tree->Fq->power_verbose" << endl;
				}
				val2 = Tree->Fq->power_verbose(val1, exp, verbose_level - 2);
				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"after Tree->Fq->power_verbose" << endl;
				}
				T->value_int = val2;
				f_has_exponent = false;
				exponent = 0;
				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"terminal node of type int to "
							"the power of " << exp << " = " << val2 << endl;
				}
			}
		}

	}
	else {
		int i;

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
		}



		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"child " << i << " / " << nb_nodes
						<< " before simplify" << endl;
			}
			Nodes[i]->simplify(verbose_level - 2);
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"child " << i << " / " << nb_nodes
						<< " after simplify" << endl;
			}
		}

		if (f_v) {
			display_children_by_type();
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes - 1; i++) {
				if (Nodes[i]->is_int_node()
						&& Nodes[i + 1]->is_int_node()) {


					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"combining children " << i
								<< " and " << i + 1
								<< " using multiplication" << endl;
					}
					int val1, val2, val3;

					val1 = Nodes[i]->T->value_int;

					if (Nodes[i]->f_has_exponent) {
						int exp = Nodes[i]->exponent;
						int val;

						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"node[" << i << "] "
											"raising to the power of "
									<< exp << endl;
						}
						val = Tree->Fq->power(val1, exp);
						Nodes[i]->T->value_int = val;
						Nodes[i]->f_has_exponent = false;
						Nodes[i]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val1
									<< " to the power of "
									<< exp << " = " << val << endl;
						}
						val1 = val;
					}

					val2 = Nodes[i + 1]->T->value_int;

					if (Nodes[i + 1]->f_has_exponent) {
						int exp = Nodes[i + 1]->exponent;
						int val;

						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"node[" << i + 1 << "] raising "
											"to the power of "
									<< exp << endl;
						}
						val = Tree->Fq->power(val2, exp);
						Nodes[i + 1]->T->value_int = val;
						Nodes[i + 1]->f_has_exponent = false;
						Nodes[i + 1]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val2 << " to the "
											"power of " << exp
											<< " = " << val << endl;
						}
						val2 = val;
					}


					val3 = Tree->Fq->mult(val1, val2);
					if (f_v) {
						cout << "syntax_tree_node::simplify "
								<< val1 << " * " << val2
								<< " = " << val3 << endl;
					}
					Nodes[i]->T->value_int = val3;
					delete_one_child(i + 1, verbose_level - 2);

					i--;
					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"nb_nodes=" << nb_nodes << endl;
					}
				}
			}
			for (i = 0; i < nb_nodes - 1; i++) {
				if (Nodes[i]->is_text_node()
						&& Nodes[i + 1]->is_text_node()) {
					if (Nodes[i]->text_value_match(Nodes[i + 1]->T->value_text)) {
						int exp1, exp2, exp3;

						exp1 = Nodes[i]->get_exponent();
						exp2 = Nodes[i + 1]->get_exponent();
						exp3 = exp1 + exp2;
						Nodes[i]->f_has_exponent = true;
						Nodes[i]->exponent = exp3;

						delete_one_child(i + 1, verbose_level - 2);

						i--;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"nb_nodes=" << nb_nodes << endl;
						}

					}
				}
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes - 1; i++) {
				if (Nodes[i]->is_int_node()
						&& Nodes[i + 1]->is_int_node()) {


					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"combining children " << i
								<< " and " << i + 1
								<< " using addition" << endl;
					}

					int val1, val2, val3;

					val1 = Nodes[i]->T->value_int;

					if (Nodes[i]->f_has_exponent) {
						int exp = Nodes[i]->exponent;
						int val;

						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"node[" << i << "] "
											"raising to the power of "
									<< exp << endl;
						}
						val = Tree->Fq->power(val1, exp);
						Nodes[i]->T->value_int = val;
						Nodes[i]->f_has_exponent = false;
						Nodes[i]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val1
									<< " to the power of " << exp
									<< " = " << val << endl;
						}
						val1 = val;
					}


					val2 = Nodes[i + 1]->T->value_int;

					if (Nodes[i + 1]->f_has_exponent) {
						int exp = Nodes[i + 1]->exponent;
						int val;

						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"node[" << i + 1 << "] raising "
											"to the power of " << exp << endl;
						}
						val = Tree->Fq->power(val2, exp);
						Nodes[i + 1]->T->value_int = val;
						Nodes[i + 1]->f_has_exponent = false;
						Nodes[i + 1]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val2
									<< " to the power of " << exp
									<< " = " << val << endl;
						}
						val2 = val;
					}


					val3 = Tree->Fq->add(val1, val2);
					Nodes[i]->T->value_int = val3;
					delete_one_child(i + 1, verbose_level - 2);

					i--;
					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"nb_nodes=" << nb_nodes << endl;
					}
				}
			}
		}


		if (nb_nodes == 1
				&& Nodes[0]->f_terminal
				&& Nodes[0]->T->f_int) {

			int val;

			val = Nodes[0]->T->value_int;

			if (Nodes[0]->f_has_exponent) {
				int exp = Nodes[0]->exponent;
				int val1;

				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"node[" << 0 << "] raising to the "
									"power of " << exp << endl;
				}
				val1 = Tree->Fq->power(val, exp);
				Nodes[0]->T->value_int = val1;
				Nodes[0]->f_has_exponent = false;
				Nodes[0]->exponent = 0;
				if (f_v) {
					cout << "syntax_tree_node::simplify "
							"terminal " << val << " to the "
									"power of " << exp << " = "
									<< val1 << endl;
				}
				val = val1;
			}


			FREE_OBJECT(Nodes[0]);
			Nodes[0] = NULL;
			nb_nodes = 0;


			type = operation_type_nothing;
			f_terminal = true;
			T = NEW_OBJECT(syntax_tree_node_terminal);
			T->f_int = true;
			T->value_int = val;

		}

	}


	if (f_v) {
		cout << "syntax_tree_node::simplify done" << endl;
	}
}

void syntax_tree_node::expand_in_place(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::expand_in_place" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::expand_in_place "
					"terminal node" << endl;
		}
		if (T->f_int) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"terminal node of type int" << endl;
			}
			int val1, val2;
			val1 = T->value_int;
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"terminal node of type int "
						"val1=" << val1 << endl;
			}
			if (f_has_exponent) {
				int exp = exponent;
				if (f_v) {
					cout << "syntax_tree_node::expand_in_place "
							"exponent=" << exp << endl;
				}

				if (Tree == NULL) {
					cout << "syntax_tree_node::expand_in_place "
							"Tree == NULL" << endl;
					exit(1);
				}
				if (Tree->Fq == NULL) {
					cout << "syntax_tree_node::expand_in_place "
							"Tree->Fq == NULL" << endl;
					exit(1);
				}
				if (f_v) {
					cout << "syntax_tree_node::expand_in_place "
							"before Tree->Fq->power_verbose" << endl;
				}
				val2 = Tree->Fq->power_verbose(val1, exp, verbose_level - 2);
				if (f_v) {
					cout << "syntax_tree_node::expand_in_place "
							"after Tree->Fq->power_verbose" << endl;
				}
				T->value_int = val2;
				f_has_exponent = false;
				exponent = 0;
				if (f_v) {
					cout << "syntax_tree_node::expand_in_place "
							"terminal node of type int to "
							"the power of " << exp << " = " << val2 << endl;
				}
			}
		}

	}
	else {
		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
		}


		if (f_v) {
			display_children_by_type();
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"multiplication node, "
						"before expand_in_place_handle_multiplication_node" << endl;
			}

			expand_in_place_handle_multiplication_node(verbose_level - 2);

			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"multiplication node, "
						"after expand_in_place_handle_multiplication_node" << endl;
			}



		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"addition node, "
						"nb_children = " << nb_nodes << endl;
			}

		}

		int i;

		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"child " << i << " / " << nb_nodes
						<< " before expand_in_place" << endl;
			}
			Nodes[i]->expand_in_place(0 /*verbose_level*/);
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place "
						"child " << i << " / " << nb_nodes
						<< " after expand_in_place" << endl;
			}
		}


	}


	if (f_v) {
		cout << "syntax_tree_node::expand_in_place done" << endl;
	}
}


void syntax_tree_node::expand_in_place_handle_multiplication_node(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::expand_in_place_handle_multiplication_node" << endl;
	}


	if (f_v) {
		cout << "syntax_tree_node::expand_in_place_handle_multiplication_node "
				"nb_children = " << nb_nodes << endl;
	}

	int i;


	for (i = 0; i < nb_nodes; i++) {

		if (f_v) {
			cout << "syntax_tree_node::expand_in_place_handle_multiplication_node "
					"child " << i << " / " << nb_nodes << endl;
		}

		if (Nodes[i]->is_add_node() &&
				Nodes[i]->get_exponent() == 1) {


			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_multiplication_node "
						"child " << i << " is addition node "
								"with exponent 1" << endl;
			}

			syntax_tree_node *mult_node;
			syntax_tree_node *add_node;
			int nb_a, nb_m, j, u;

			mult_node = this;
			add_node = Nodes[i];
			nb_m = mult_node->nb_nodes;
			nb_a = add_node->nb_nodes;

			syntax_tree_node *fresh_add_node;

			fresh_add_node = NEW_OBJECT(syntax_tree_node);

			fresh_add_node->init_empty_plus_node_with_exponent(
						Tree,
						1 /* exponent */, verbose_level - 2);

			for (j = 0; j < nb_a; j++) {

				syntax_tree_node *fresh_mult_node;

				fresh_mult_node = NEW_OBJECT(syntax_tree_node);

				fresh_mult_node->init_empty_multiplication_node(Tree,
						verbose_level - 2);

				for (u = 0; u < nb_m; u++) {


					syntax_tree_node *fresh_node;

					fresh_node = NEW_OBJECT(syntax_tree_node);

					if (u == i) {


						// copy the j-th child of add_node:

						add_node->Nodes[j]->copy_to(
									Tree /* Output_tree */,
									fresh_node,
									0 /*verbose_level*/);

					}
					else {


						// copy the u-th child of mult_node:

						mult_node->Nodes[u]->copy_to(
									Tree /* Output_tree */,
									fresh_node,
									0 /*verbose_level*/);

					}

					fresh_mult_node->append_node(fresh_node, 0 /*verbose_level*/);
					//fresh_mult_node->Nodes[fresh_mult_node->nb_nodes++] = fresh_node;

				}

				fresh_add_node->append_node(fresh_mult_node, 0 /*verbose_level*/);
				//fresh_add_node->Nodes[fresh_add_node->nb_nodes++] = fresh_mult_node;

			}

			for (u = 0; u < nb_m; u++) {

				if (u == i) {
					continue;
				}

				FREE_OBJECT(mult_node->Nodes[u]);
				mult_node->Nodes[u] = NULL;

			}
			mult_node->nb_nodes = 0;
			FREE_OBJECT(add_node);


			// ToDo: repurposing node

			mult_node->reallocate(fresh_add_node->nb_nodes, verbose_level - 2);

			mult_node->type = operation_type_add;
			for (u = 0; u < fresh_add_node->nb_nodes; u++) {
				mult_node->Nodes[u] = fresh_add_node->Nodes[u];
				fresh_add_node->Nodes[u] = NULL;
			}
			mult_node->nb_nodes = fresh_add_node->nb_nodes;
			fresh_add_node->nb_nodes = 0;

			FREE_OBJECT(fresh_add_node);

		}
	}



	if (f_v) {
		cout << "syntax_tree_node::expand_in_place_handle_multiplication_node done" << endl;
	}
}



void syntax_tree_node::expand_in_place_handle_exponents(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::expand_in_place_handle_exponents" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::expand_in_place_handle_exponents "
					"terminal node" << endl;
		}
		if (T->f_int) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"terminal node of type int" << endl;
			}
		}
	}
	else {
		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
		}


		if (f_v) {
			display_children_by_type();
		}

		if (type == operation_type_add && get_exponent() != 1) {

			int exp;

			exp = get_exponent();

			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"expanding addition node, "
						"exp = " << exp
						<< " nb_nodes = " << nb_nodes
						<< endl;
			}

			if (exp < 0) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"expanding addition node, "
						"exponent = " << exp << " cannot be negative" << endl;
				exit(1);
			}

			syntax_tree_node *fresh_mult_node;

			fresh_mult_node = NEW_OBJECT(syntax_tree_node);

			fresh_mult_node->init_empty_multiplication_node(
						Tree,
						verbose_level - 2);

			int i;

			// remove the exponent before copying to avoid infinite recursion:

			f_has_exponent = false;
			exponent = 0;

			for (i = 0; i < exp; i++) {

				syntax_tree_node *fresh_node;

				fresh_node = NEW_OBJECT(syntax_tree_node);

				if (f_v) {
					cout << "syntax_tree_node::expand_in_place_handle_exponents "
							"expanding addition node, "
							"exp = " << exp
							<< " nb_nodes = " << nb_nodes
							<< " copying " << i << " times"
							<< endl;
				}

				copy_to(
							Tree /* Output_tree */,
							fresh_node,
							verbose_level - 2);

				fresh_mult_node->append_node(fresh_node, 0 /* verbose_level */);
				//fresh_mult_node->Nodes[fresh_mult_node->nb_nodes++] = fresh_node;

			}


			for (i = 0; i < nb_nodes; i++) {

				FREE_OBJECT(Nodes[i]);
				Nodes[i] = NULL;

			}
			nb_nodes = 0;

			// make space:

			if (fresh_mult_node->nb_nodes > nb_nodes_allocated) {
				reallocate(fresh_mult_node->nb_nodes, verbose_level - 2);
			}

			for (i = 0; i < fresh_mult_node->nb_nodes; i++) {
				Nodes[i] = fresh_mult_node->Nodes[i];
				fresh_mult_node->Nodes[i] = NULL;
			}
			nb_nodes = fresh_mult_node->nb_nodes;
			fresh_mult_node->nb_nodes = 0;

			type = operation_type_mult;

			FREE_OBJECT(fresh_mult_node);

		}
#if 1
		int i;

		for (i = 0; i < nb_nodes; i++) {

			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"multiplication node, "
						"child " << i << " / " << nb_nodes << endl;
			}

			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"multiplication node, "
						"child " << i << " / " << nb_nodes
						<< " before expand_in_place_handle_exponents" << endl;
			}
			Nodes[i]->expand_in_place_handle_exponents(verbose_level - 2);
			if (f_v) {
				cout << "syntax_tree_node::expand_in_place_handle_exponents "
						"multiplication node, "
						"child " << i << " / " << nb_nodes
						<< " after expand_in_place_handle_exponents" << endl;
			}
		}
#endif
	}
	if (f_v) {
		cout << "syntax_tree_node::expand_in_place_handle_exponents done" << endl;
	}
}



void syntax_tree_node::simplify_exponents(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::simplify_exponents" << endl;
	}

	if (f_terminal) {
		if (f_has_exponent) {
			if (T->f_int) {
				int exp = exponent;
				int val1, val2;

				if (f_v) {
					cout << "syntax_tree_node::simplify_exponents "
							"raising to the power of " << exp << endl;
				}
				val1 = T->value_int;

				if (Tree == NULL) {
					cout << "syntax_tree_node::simplify_exponents "
							"Tree == NULL" << endl;
					exit(1);
				}
				if (Tree->Fq == NULL) {
					cout << "syntax_tree_node::simplify_exponents "
							"Tree->Fq == NULL" << endl;
					exit(1);
				}
				if (f_v) {
					cout << "syntax_tree_node::simplify_exponents "
							"before Tree->Fq->power" << endl;
				}
				val2 = Tree->Fq->power(val1, exp);
				if (f_v) {
					cout << "syntax_tree_node::simplify_exponents "
							"after Tree->Fq->power" << endl;
				}
				T->value_int = val2;
				f_has_exponent = false;
				exponent = 0;
				if (f_v) {
					cout << "syntax_tree_node::simplify_exponents "
							"terminal " << val1 << " to the power of "
							<< exp << " = " << val2 << endl;
				}
			}
			else if (T->f_text) {
				if (exponent == 1) {
					f_has_exponent = false;
					exponent = 0;
					if (f_v) {
						cout << "syntax_tree_node::simplify_exponents "
								"eliminating exponent of 1 of "
								"literal terminal node" << endl;
					}
				}
			}
		}
	}
	else {
		if (f_has_exponent && exponent == 1) {
			f_has_exponent = false;
			exponent = 0;
			if (f_v) {
				cout << "syntax_tree_node::simplify_exponents "
						"eliminating exponent of 1" << endl;
			}
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::simplify_exponents "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_exponents(verbose_level - 2);
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::simplify_exponents "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_exponents(verbose_level - 2);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::simplify_exponents done" << endl;
	}

}

void syntax_tree_node::sort_terms(
		int verbose_level)
// called from formula::simplify
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::sort_terms" << endl;
	}

	if (f_terminal) {
	}
	else {

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::sort_terms "
						"multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}

			other::data_structures::sorting Sorting;

			if (f_v) {
				cout << "syntax_tree_node::sort_terms "
						"before Sorting.Heapsort_general" << endl;
			}
			Sorting.Heapsort_general(
					Nodes, nb_nodes,
					syntax_tree_node_compare_func,
					syntax_tree_node_swap_func,
					this);

			if (f_v) {
				cout << "syntax_tree_node::sort_terms "
						"after Sorting.Heapsort_general" << endl;
			}
			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->sort_terms(verbose_level);
			}
			for (i = 0; i < nb_nodes - 1; i++) {

				if (syntax_tree_node_compare_func(Nodes,
						i, i + 1, this) == 0) {


					if (Nodes[i]->is_int_node() &&
							Nodes[i + 1]->is_int_node()) {
						if (!Nodes[i]->f_has_exponent && !Nodes[i + 1]->f_has_exponent) {
							int val1, val2, val3;

							val1 = Nodes[i]->T->value_int;
							val2 = Nodes[i + 1]->T->value_int;
							val3 = Tree->Fq->mult(val1, val2);
							Nodes[i]->T->value_int = val3;
							delete_one_child(i + 1, verbose_level - 2);
							i--;
						}
					}
					else if (Nodes[i]->is_text_node() &&
								Nodes[i + 1]->is_text_node()) {
						int exp1, exp2, exp3;

						exp1 = Nodes[i]->get_exponent();
						exp2 = Nodes[i + 1]->get_exponent();
						exp3 = exp1 + exp2;
						if (exp3 != 1) {
							Nodes[i]->f_has_exponent = true;
							Nodes[i]->exponent = exp3;
						}
						else {
							Nodes[i]->f_has_exponent = false;
							Nodes[i]->exponent = 0;
						}
						delete_one_child(i + 1, verbose_level - 2);
						i--;
					}
				}
			}

		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::sort_terms "
						"addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			other::data_structures::sorting Sorting;

			Sorting.Heapsort_general(
					Nodes, nb_nodes,
					syntax_tree_node_compare_func,
					syntax_tree_node_swap_func,
					this);
			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->sort_terms(verbose_level - 2);
			}
		}
	}


	if (f_v) {
		cout << "syntax_tree_node::sort_terms done" << endl;
	}
}



void syntax_tree_node::collect_like_terms(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::collect_like_terms "
					"cannot be a terminal node" << endl;
		}
		// do nothing
	}
	else {

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}

			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->collect_like_terms(verbose_level - 2);
			}

		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"addition node, "
						"nb_children = " << nb_nodes << endl;
			}

			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"before collect_like_terms_addition" << endl;
			}
			collect_like_terms_addition(verbose_level - 2);
			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"after collect_like_terms_addition" << endl;
				cout << "syntax_tree_node::collect_like_terms "
						"subtree: " << endl;
				print_subtree_easy(cout);
				cout << endl;
			}

		}
	}


	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms done" << endl;
	}
}

void syntax_tree_node::collect_like_terms_addition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition" << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"nb_children = " << nb_nodes << endl;
	}

	int i;
	other::data_structures::int_matrix *I;
	int *Coeff;
	int N;

	N = Tree->get_number_of_variables();


	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"before collect_terms_and_coefficients" << endl;
	}
	collect_terms_and_coefficients(
			I, Coeff,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"after collect_terms_and_coefficients" << endl;
	}




	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"data collected:" << endl;
		for (i = 0; i < nb_nodes; i++) {
			cout << setw(3) << i << " / " << nb_nodes << " : ";
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * N, N);
			cout << endl;
		}
	}


	I->sort_rows(verbose_level);

	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"after sorting:" << endl;
		for (i = 0; i < nb_nodes; i++) {
			cout << setw(3) << i << " / " << nb_nodes << " : ";
			cout << setw(3) << Coeff[i] << " : ";
			cout << setw(4) << I->perm_inv[i] << " : ";
			Int_vec_print(cout, I->M + i * N, N);
			cout << endl;
		}
	}

	int Nb_nodes;

	Nb_nodes = nb_nodes;

	for (i = 0; i < Nb_nodes; i++) {

		FREE_OBJECT(Nodes[i]);
		Nodes[i] = NULL;
	}
	nb_nodes = 0;

	int j, u; //, v, exp;
	int coeff;
	other::data_structures::sorting Sorting;

	for (i = 0; i < Nb_nodes; i++) {

		for (j = i + 1; j < Nb_nodes; j++) {
			if (Sorting.integer_vec_compare(
					I->M + i * N, I->M + j * N, N)) {
				break;
			}
		}
		coeff = Coeff[I->perm_inv[i]];
		for (u = i + 1; u < j; u++) {
			coeff = Tree->Fq->add(coeff, Coeff[I->perm_inv[u]]);
		}

		if (coeff) {

#if 0
			syntax_tree_node *node;

			node = NEW_OBJECT(syntax_tree_node);

			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms_addition "
						"adding term ";
				cout << setw(3) << nb_nodes << " : ";
				Int_vec_print(cout, I->M + i * N, N);
				cout << " with coeff " << coeff << endl;
			}
			node->init_empty_multiplication_node(
					Tree,
					0 /*verbose_level*/);

			if (coeff != 1) {
				node->add_numerical_factor(
						coeff, 0 /*verbose_level*/);
			}
			else if (Int_vec_is_zero(I->M + i * N, N)) {
				node->add_numerical_factor(
						coeff, 0 /*verbose_level*/);
			}

			for (v = 0; v < N; v++) {
				exp = I->M[i * N + v];
				if (exp) {
					node->add_factor(
							Tree->get_variable_name(v), exp,
							0 /*verbose_level*/);
				}
			}

			append_node(node, 0 /* verbose_level */);
			//Nodes[nb_nodes++] = node;
#else
			append_monomial(
					I->M + i * I->n, I->n, coeff,
					verbose_level - 1);
#endif
		}
		i = j - 1;

	}
	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"reduced from " << Nb_nodes << " to "
				<< nb_nodes << " terms" << endl;
	}
	if (nb_nodes == 0) {
		cout << "syntax_tree_node::collect_like_terms_addition "
				"reduced from " << Nb_nodes << " to "
				<< nb_nodes << " terms" << endl;
		cout << "syntax_tree_node::collect_like_terms_addition "
				"adding a zero term:" << endl;
		syntax_tree_node *node;

		node = NEW_OBJECT(syntax_tree_node);
		node->init_terminal_node_int(
				Tree,
				0 /* value */,
				verbose_level);
		append_node(node, 0 /* verbose_level */);

	}
	FREE_OBJECT(I);


}

void syntax_tree_node::append_monomial(
		int *exponent_vector, int nb_vars, int coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::append_monomial" << endl;
	}

	//int N;

	//N = Tree->get_number_of_variables();

	syntax_tree_node *node;

	node = NEW_OBJECT(syntax_tree_node);

	if (f_v) {
		cout << "syntax_tree_node::append_monomial "
				"adding term ";
		cout << setw(3) << nb_nodes << " : ";
		Int_vec_print(cout, exponent_vector, nb_vars);
		cout << " with coeff " << coeff << endl;
	}
	node->init_empty_multiplication_node(
			Tree,
			0 /*verbose_level*/);

	if (coeff != 1) {
		node->add_numerical_factor(
				coeff, 0 /*verbose_level*/);
	}
	else if (Int_vec_is_zero(exponent_vector, nb_vars)) {
		node->add_numerical_factor(
				coeff, 0 /*verbose_level*/);
	}

	int v, exp;

	for (v = 0; v < nb_vars; v++) {
		exp = exponent_vector[v];
		if (exp) {
			node->add_factor(
					Tree->get_variable_name(v), exp,
					0 /*verbose_level*/);
		}
	}

	append_node(node, 0 /* verbose_level */);

	if (f_v) {
		cout << "syntax_tree_node::append_monomial done" << endl;
	}

}

void syntax_tree_node::collect_monomial_terms(
		other::data_structures::int_matrix *&I, int *&Coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_monomial_terms" << endl;
	}

	if (f_terminal) {
		cout << "syntax_tree_node::collect_monomial_terms "
				"cannot be a terminal node" << endl;
		I = NULL;
		Coeff = NULL;
	}
	else {

		if (type == operation_type_mult) {

			cout << "syntax_tree_node::collect_monomial_terms "
					"cannot be a multiplication node" << endl;
			I = NULL;
			Coeff = NULL;

		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::collect_monomial_terms "
						"addition node, "
						"nb_children = " << nb_nodes << endl;
			}


			//int N;

			//N = Tree->variables.size();


			if (f_v) {
				cout << "syntax_tree_node::collect_monomial_terms "
						"before collect_terms_and_coefficients" << endl;
			}
			collect_terms_and_coefficients(
					I, Coeff,
					verbose_level - 2);
			if (f_v) {
				cout << "syntax_tree_node::collect_monomial_terms "
						"after collect_terms_and_coefficients" << endl;
			}

		}
		else {
			cout << "syntax_tree_node::collect_monomial_terms "
					"unknown node type" << endl;
			exit(1);

		}

	}

	if (f_v) {
		cout << "syntax_tree_node::collect_monomial_terms done" << endl;
	}
}


void syntax_tree_node::collect_terms_and_coefficients(
		other::data_structures::int_matrix *&I, int *&Coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_terms_and_coefficients" << endl;
	}

	int i;
	int index;
	int exp;
	int N;
	int *monomial_exponents;



	N = Tree->get_number_of_variables();

	Coeff = NEW_int(nb_nodes);
	I = NEW_OBJECT(other::data_structures::int_matrix);
	I->allocate(nb_nodes, N);

	monomial_exponents = NEW_int(N);

	for (i = 0; i < nb_nodes; i++) {

		if (f_v) {
			cout << "syntax_tree_node::collect_terms_and_coefficients "
					"addition node, collecting "
					"child " << i << " / " << nb_nodes << endl;
		}

		Int_vec_zero(monomial_exponents, N);

		syntax_tree_node *node;

		node = Nodes[i];

		int coeff = 1;

		if (node->f_terminal && node->T->f_text) {

			index = node->terminal_node_get_variable_index();
			exp = node->get_exponent();
			monomial_exponents[index] = exp;

		}
		else if (node->f_terminal && node->T->f_int) {

			coeff = node->T->value_int;

		}
		else if (node->type == operation_type_mult) {

			int j;
			int node_exp;

			node_exp = node->get_exponent();

			for (j = 0; j < node->nb_nodes; j++) {

				if (node->Nodes[j]->f_terminal) {
					if (node->Nodes[j]->T->f_int) {
						if (coeff != 1) {
							cout << "syntax_tree_node::collect_terms_and_coefficients "
									"we cannot have more than "
									"one coefficient" << endl;
							exit(1);
						}

						coeff = Tree->Fq->mult(coeff, node->Nodes[j]->T->value_int);
						coeff = Tree->Fq->power(coeff, node_exp);
					}
					else {
						index = node->Nodes[j]->terminal_node_get_variable_index();

						if (index < 0) {
							cout << "syntax_tree_node::collect_terms_and_coefficients "
									"cannot find variable "
									<< node->Nodes[j]->T->value_text << endl;
							exit(1);
						}
						exp = node->Nodes[j]->get_exponent();
						monomial_exponents[index] += exp * node_exp;

					}
				}


			}

		}
		else if (node->type == operation_type_add) {
			cout << "syntax_tree_node::collect_terms_and_coefficients "
					"cannot have add node as a child "
					"of an add node" << endl;
			exit(1);
		}
		else {
			cout << "syntax_tree_node::collect_terms_and_coefficients "
					"unknown node type" << endl;
			exit(1);

		}

		Int_vec_copy(monomial_exponents, I->M + i * N, N);

		Coeff[i] = coeff;

	}

	if (false) {
		cout << "syntax_tree_node::collect_terms_and_coefficients "
				"data collected:" << endl;
		for (i = 0; i < nb_nodes; i++) {
			cout << Coeff[i] << " : ";
			Int_vec_print(cout, I->M + i * N, N);
			cout << endl;
		}
	}
	FREE_int(monomial_exponents);

	if (f_v) {
		cout << "syntax_tree_node::collect_terms_and_coefficients done" << endl;
	}
}

void syntax_tree_node::simplify_constants(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::simplify_constants" << endl;
	}

	if (f_terminal) {
	}
	else {

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::simplify_constants "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
			int i;

			for (i = 0; i < nb_nodes; i++) {

				if (Nodes[i]->is_constant_one(verbose_level - 2)) {
					if (nb_nodes > 1) {

						delete_one_child(i, verbose_level - 2);

					}
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				if (Nodes[i]->is_constant_zero(verbose_level - 2)) {

					delete_all_but_one_child(i, verbose_level - 2);
					break;
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_constants(verbose_level - 2);
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::simplify_constants "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			int i;

			for (i = 0; i < nb_nodes; i++) {

				if (Nodes[i]->is_constant_zero(verbose_level - 2)) {
					if (nb_nodes > 1) {

						delete_one_child(i, verbose_level - 2);

					}
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_constants(verbose_level - 2);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::simplify_constants done" << endl;
	}

}

void syntax_tree_node::flatten(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::flatten "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::flatten before ";

		print_subtree_easy(cout);
	}


	flatten_with_depth(0 /* depth */,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree_node::flatten after  ";

		print_subtree_easy(cout);
	}


	if (f_v) {
		cout << "syntax_tree_node::flatten done" << endl;
	}
}


void syntax_tree_node::flatten_with_depth(
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::flatten_with_depth "
				"verbose_level = " << verbose_level << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::flatten_with_depth before ";

		other::data_structures::algorithms Algo;

		Algo.print_repeated_character(cout, ' ', depth);
		print_subtree_easy(cout);
	}


	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::flatten_with_depth terminal node" << endl;
		}
	}
	else {
		int i;

		if (nb_nodes == 0) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth nb_nodes == 0" << endl;
				exit(1);
			}
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
		}


		if (f_v) {
			display_children_by_type();
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {

				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth "
							"simplifying multiplication node, "
							"child = " << i << " / " << nb_nodes << " before flatten" << endl;
				}
				Nodes[i]->flatten_with_depth(depth + 1, verbose_level - 2);
				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth "
							"simplifying multiplication node, "
							"child = " << i << " / " << nb_nodes << " after flatten" << endl;
				}

				if (Nodes[i]->is_mult_node() || (Nodes[i]->is_add_node() && Nodes[i]->nb_nodes == 1)) {


					if (f_v) {
						cout << "syntax_tree_node::flatten_with_depth "
								"simplifying multiplication node, "
								"folding children of " << i << endl;
					}

					flatten_at(i, depth + 1, verbose_level - 2);

					i--; // do the present child again, because it has changed

				} // if (Nodes[i]->is_mult_node())
				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth after round " << i << " ";

					other::data_structures::algorithms Algo;

					Algo.print_repeated_character(cout, ' ', depth);

					print_subtree_easy(cout);
				}
			} // next i

		} // if (type == operation_type_mult)
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {

				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth "
							"simplifying addition node, "
							"child = " << i << " / " << nb_nodes << " before flatten" << endl;
				}
				Nodes[i]->flatten_with_depth(depth + 1, verbose_level - 2);
				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth "
							"simplifying addition node, "
							"child = " << i << " / " << nb_nodes << " after flatten" << endl;
				}

				if ((Nodes[i]->is_add_node() && Nodes[i]->get_exponent() == 1)
						|| (Nodes[i]->is_mult_node() && Nodes[i]->nb_nodes == 1)) {
					if (f_v) {
						cout << "syntax_tree_node::flatten_with_depth "
								"simplifying addition node, "
								"child " << i << " / " << nb_nodes
								<< " before flatten_at" << endl;
					}
					flatten_at(i, depth + 1, verbose_level - 2);
					i--;
					if (f_v) {
						cout << "syntax_tree_node::flatten_with_depth "
								"simplifying addition node, "
								"child " << i << " / " << nb_nodes
								<< " after flatten_at" << endl;
					}
				}
				if (f_v) {
					cout << "syntax_tree_node::flatten_with_depth after round " << i << " ";

					other::data_structures::algorithms Algo;

					Algo.print_repeated_character(cout, ' ', depth);

					print_subtree_easy(cout);
				}
			}

		}
	}


	if (f_v) {
		cout << "syntax_tree_node::flatten_with_depth "
				"before flatten_post_process ";
		other::data_structures::algorithms Algo;

		Algo.print_repeated_character(cout, ' ', depth);
		print_subtree_easy(cout);
	}
	flatten_post_process(depth, verbose_level);
	if (f_v) {
		cout << "syntax_tree_node::flatten_with_depth "
				"after flatten_post_process ";
		other::data_structures::algorithms Algo;

		Algo.print_repeated_character(cout, ' ', depth);
		print_subtree_easy(cout);
	}




	if (f_v) {
		cout << "syntax_tree_node::flatten_with_depth done" << endl;
	}
}

void syntax_tree_node::flatten_post_process(
		int depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::flatten_post_process "
				"verbose_level = " << verbose_level << endl;
	}

	if (nb_nodes == 1
			&& Nodes[0]->f_terminal
			&& Nodes[0]->T->f_int) {

		int val;

		val = Nodes[0]->T->value_int;

		if (Nodes[0]->f_has_exponent) {
			int exp = Nodes[0]->exponent;
			int val1;

			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"node[" << 0 << "] raising to the "
								"power of " << exp << endl;
			}
			val1 = Tree->Fq->power(val, exp);
			Nodes[0]->T->value_int = val1;
			Nodes[0]->f_has_exponent = false;
			Nodes[0]->exponent = 0;
			if (f_v) {
				cout << "syntax_tree_node::flatten_with_depth "
						"terminal " << val << " to the "
								"power of " << exp
								<< " = " << val1 << endl;
			}
			val = val1;
		}


		FREE_OBJECT(Nodes[0]);
		Nodes[0] = NULL;
		nb_nodes = 0;


		// ToDo: repurposing a node

		type = operation_type_nothing;
		f_terminal = true;
		T = NEW_OBJECT(syntax_tree_node_terminal);
		T->f_int = true;
		T->value_int = val;

	}

	if (f_v) {
		cout << "syntax_tree_node::flatten_post_process done" << endl;
	}
}


void syntax_tree_node::flatten_at(
		int i, int depth, int verbose_level)
// we are assuming the child is a multiplication node
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "syntax_tree_node::flatten_at "
				"child " << i << " / " << nb_nodes << endl;
	}

	if (f_v) {
		cout << "syntax_tree_node::flatten_at i=" << i << " before ";
		other::data_structures::algorithms Algo;

		Algo.print_repeated_character(cout, ' ', depth);
		Nodes[i]->print_subtree_easy(cout);
	}

#if 0
	if (!Nodes[i]->is_mult_node()) {
		cout << "syntax_tree_node::flatten_at i=" << i << " the sub node is not a multiplication node" << endl;
		cout << "syntax_tree_node::flatten_at i=" << i << " sub node type: ";
		Nodes[i]->print_node_type(cout);
		cout << endl;
		cout << "syntax_tree_node::flatten_at i=" << i << " subtree: ";
		Nodes[i]->print_subtree_easy(cout);
		exit(1);
	}
#endif


	if (Nodes[i]->f_has_exponent) {

		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" we have an exponent " << endl;
		}

		if (Nodes[i]->is_add_node() && Nodes[i]->nb_nodes > 1) {
			if (f_v) {
				cout << "syntax_tree_node::flatten_at i=" << i <<
						" we have an exponent and Nodes[i]->is_add_node()"
						" && Nodes[i]->nb_nodes > 1" << endl;
				exit(1);
			}

		}
		syntax_tree_node *old_node;

		old_node = Nodes[i];

		int exp;

		exp = old_node->exponent;

		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" exp =  " << exp << endl;
		}
		int nb_n;

		nb_n = old_node->nb_nodes;

		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" nb_nodes =  " << nb_n << endl;
		}
		int j;


		// Make space for the children's nodes.
		//
		// Move up by nb_n  - 1 because the old node is deleted:

		// insert nodes:

		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" before insert_nodes_at i=" << i
					<< " nb_n - 1 = " << nb_n - 1 << endl;
		}
		insert_nodes_at(i, nb_n - 1, verbose_level - 2);
		if (f_v) {
			cout << "syntax_tree_node::flatten_at "
					" after insert_nodes_at i=" << i << endl;
		}
#if 0
		for (j = nb_nodes - 1; j > i; j--) {
			Nodes[j + nb_n - 1] = Nodes[j];
		}
		nb_nodes += nb_n - 1;
#endif
		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" after insert_nodes_at nb_n=" << nb_n << endl;
		}
		for (j = 0; j < nb_n; j++) {
			Nodes[i + j] = old_node->Nodes[j];
			old_node->Nodes[j] = NULL;
			if (Nodes[i + j]->f_has_exponent) {
				Nodes[i + j]->exponent *= exp;
			}
			else {
				Nodes[i + j]->f_has_exponent = true;
				Nodes[i + j]->exponent = exp;
			}
		}
		old_node->nb_nodes = 0;
		FREE_OBJECT(old_node);

	}
	else {

		if (f_v) {
			cout << "syntax_tree_node::flatten_at i=" << i <<
					" we don't have an exponent " << endl;
		}
		syntax_tree_node *old_node;

		old_node = Nodes[i];


		int nb_n;

		nb_n = old_node->nb_nodes;

		int j;


		// insert nodes:

		insert_nodes_at(i, nb_n - 1, verbose_level - 2);

#if 0
		for (j = nb_nodes - 1; j > i; j--) {
			Nodes[j + nb_n - 1] = Nodes[j];
		}
		nb_nodes += nb_n - 1;
#endif

		for (j = 0; j < nb_n; j++) {
			Nodes[i + j] = old_node->Nodes[j];
			old_node->Nodes[j] = NULL;
		}
		old_node->nb_nodes = 0;
		FREE_OBJECT(old_node);

	}
	if (f_v) {
		cout << "syntax_tree_node::flatten_at i=" << i << " after ";
		other::data_structures::algorithms Algo;

		Algo.print_repeated_character(cout, ' ', depth);
		Nodes[i]->print_subtree_easy(cout);
	}

	if (f_v) {
		cout << "syntax_tree_node::flatten_at i=" << i <<
				" nb_nodes=" << nb_nodes << " done" << endl;
	}
}

void syntax_tree_node::delete_all_but_one_child(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::delete_all_but_one_child" << endl;
	}
	int j;


	for (j = 0; j < i; j++) {
		FREE_OBJECT(Nodes[j]);
		Nodes[j] = NULL;
	}
	for (j = i + 1; j < nb_nodes; j++) {
		FREE_OBJECT(Nodes[j]);
		Nodes[j] = NULL;
	}
	if (i) {
		Nodes[0] = Nodes[i];
		Nodes[i] = NULL;
	}
	nb_nodes = 1;

	if (f_v) {
		cout << "syntax_tree_node::delete_all_but_one_child done" << endl;
	}
}

void syntax_tree_node::delete_one_child(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::delete_one_child" << endl;
	}

	int j;

	FREE_OBJECT(Nodes[i]);
	for (j = i + 1; j < nb_nodes; j++) {
		Nodes[j - 1] = Nodes[j];
	}
	Nodes[nb_nodes - 1] = NULL;
	nb_nodes--;

	if (f_v) {
		cout << "syntax_tree_node::delete_one_child done" << endl;
	}

}

int syntax_tree_node::is_constant_one(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::is_constant_one" << endl;
	}
	if (f_terminal) {
		if (T->f_int) {
			int val;

			val = T->value_int;
			if (val == 1) {
				if (f_v) {
					cout << "syntax_tree_node::is_constant_one "
							"returns true" << endl;
				}
				return true;
			}
			else {
				return false;
			}
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_constant_zero(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::is_constant_zero" << endl;
	}
	if (f_terminal) {
		if (T->f_int) {
			int val;

			val = T->value_int;
			if (val == 0) {
				if (f_v) {
					cout << "syntax_tree_node::is_constant_zero "
							"returns true" << endl;
				}
				return true;
			}
			else {
				if (f_v) {
					cout << "syntax_tree_node::is_constant_zero "
							"returns false. "
							"f_terminal, f_int are true "
							"but value is not zero " << endl;
				}
				return false;
			}
		}
		else {
			if (f_v) {
				cout << "syntax_tree_node::is_constant_zero "
						"returns false. "
						"f_terminal is true but f_int is not " << endl;
			}
			return false;
		}
	}
	else {
		if (f_v) {
			cout << "syntax_tree_node::is_constant_zero "
					"returns false. "
					"f_terminal is false. Node type is ";
			print_node_type(cout);
			cout << endl;
		}
		return false;
	}
}



void syntax_tree_node::collect_variables(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_variables" << endl;
	}
	if (f_terminal) {
		if (T->f_text) {

			if (Tree->find_variable(
					T->value_text,
					verbose_level) == -1) {

				if (f_v) {
					cout << "syntax_tree_node::collect_variables "
							"before Tree->add_variable value = " << T->value_text << endl;
				}
				Tree->add_variable(T->value_text, verbose_level);
				if (f_v) {
					cout << "syntax_tree_node::collect_variables "
							"after Tree->add_variable value = " << T->value_text << endl;
				}

			}
		}
	}
	else {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			if (f_v) {
				cout << "syntax_tree_node::collect_variables "
						"node " << i << " / " << nb_nodes << endl;
			}
			Nodes[i]->collect_variables(verbose_level);
			if (f_v) {
				cout << "syntax_tree_node::collect_variables "
						"node " << i << " / " << nb_nodes << " done" << endl;
			}
		}
	}
	if (f_v) {
		cout << "syntax_tree_node::collect_variables done" << endl;
	}
}


int syntax_tree_node::terminal_node_get_variable_index()
{
	if (f_terminal) {
		if (T->f_text) {

			int idx;

			idx = Tree->find_variable(
					T->value_text,
					0 /*verbose_level*/);

			if (idx == -1) {
				cout << "syntax_tree_node::terminal_node_get_variable_index "
						"variable not found" << endl;
				exit(1);
			}
			return idx;
		}
		cout << "syntax_tree_node::terminal_node_get_variable_index "
				"node must be text node" << endl;
		exit(1);
	}
	cout << "syntax_tree_node::terminal_node_get_variable_index "
			"node must be terminal node" << endl;
	exit(1);
}



void syntax_tree_node::count_nodes(
		int &nb_add, int &nb_mult, int &nb_int,
		int &nb_text, int &max_degree)
{
	if (f_terminal) {
		if (T->f_int) {
			nb_int++;
		}
		else if (T->f_text) {
			nb_text++;
		}
	}
	else if (type == operation_type_mult) {
		nb_mult++;
	}
	else if (type == operation_type_add) {
		nb_add++;
	}

	int i;

	max_degree = MAXIMUM(max_degree, nb_nodes);

	for (i = 0; i < nb_nodes; i++) {
		Nodes[i]->count_nodes(
				nb_add, nb_mult, nb_int, nb_text, max_degree);
	}
}

void syntax_tree_node::reallocate(
		int nb_nodes_needed, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::reallocate" << endl;
	}

	if (nb_nodes > nb_nodes_allocated) {
		cout << "syntax_tree_node::reallocate "
				"nb_nodes > nb_nodes_allocated" << endl;
		cout << "syntax_tree_node::reallocate "
				"nb_nodes = " << nb_nodes << endl;
		cout << "syntax_tree_node::reallocate "
				"nb_nodes_allocated = " << nb_nodes_allocated << endl;
		exit(1);
	}

	if (nb_nodes_needed < nb_nodes_allocated) {
		if (f_v) {
			cout << "syntax_tree_node::reallocate "
					"nb_nodes_needed < nb_nodes_have" << endl;
			cout << "syntax_tree_node::reallocate "
					"nb_nodes_allocated = " << nb_nodes_allocated << endl;
			cout << "syntax_tree_node::reallocate "
					"nb_nodes_needed = " << nb_nodes_needed << endl;
		}
		return;
	}
	int nb_nodes_have;
	syntax_tree_node **Fresh_nodes;
	int i;

	nb_nodes_have = nb_nodes_allocated;
	nb_nodes_allocated =
			nb_nodes_needed + 2 * (nb_nodes_needed - nb_nodes_have) + 5;

	if (f_v) {
		cout << "syntax_tree_node::reallocate "
				"from " << nb_nodes_have
				<< " to " << nb_nodes_allocated << endl;
	}
	Fresh_nodes = (syntax_tree_node **) NEW_pvoid(nb_nodes_allocated);

	for (i = 0; i < nb_nodes_have; i++) {
		Fresh_nodes[i] = Nodes[i];
	}
	for (; i < nb_nodes_allocated; i++) {
		Fresh_nodes[i] = NULL;
	}

	if (nb_nodes_have && Nodes) {
		FREE_pvoid((void **) Nodes);
	}

	Nodes = Fresh_nodes;

	if (f_v) {
		cout << "syntax_tree_node::reallocate done" << endl;
	}
}


void syntax_tree_node::append_node(
		syntax_tree_node *child, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::append_node" << endl;
	}

	if (nb_nodes == nb_nodes_allocated) {

		reallocate(nb_nodes + 1 /* nb_nodes_needed */, verbose_level);
		//cout << "syntax_tree_node::append_node too many nodes" << endl;
		//cout << "syntax_tree_node::append_node nb_nodes = " << nb_nodes << endl;
		//exit(1);
	}

	Nodes[nb_nodes++] = child;

}

void syntax_tree_node::insert_nodes_at(
		int idx, int nb_to_insert, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::insert_nodes_at" << endl;
	}

	if (nb_nodes + nb_to_insert >= nb_nodes_allocated) {

		reallocate(nb_nodes + nb_to_insert /* nb_nodes_needed */, verbose_level);

		//cout << "syntax_tree_node::insert_nodes_at too many nodes" << endl;
		//cout << "syntax_tree_node::insert_nodes_at nb_nodes = " << nb_nodes << endl;
		//exit(1);
	}

	int j;

	for (j = nb_nodes - 1; j > idx; j--) {
		Nodes[j + nb_to_insert] = Nodes[j];
	}
	nb_nodes += nb_to_insert;
}

int syntax_tree_node::needs_to_be_expanded()
// true if there is a multiplication node on top of an addition node
{
	if (f_terminal) {
		return false;
	}
	else if (type == operation_type_mult) {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			if (Nodes[i]->is_add_node()) {
				return true;
			}
			if (Nodes[i]->needs_to_be_expanded()) {
				return true;
			}
		}
	}
	else if (type == operation_type_add) {

		int i;

		for (i = 0; i < nb_nodes; i++) {

			if (Nodes[i]->needs_to_be_expanded()) {
				return true;
			}
		}
	}
	return false;
}

void syntax_tree_node::make_determinant(
		syntax_tree *Output_tree,
		algebra::field_theory::finite_field *Fq,
		formula *V_in,
		int n,
		int verbose_level)
// we assume that the node is an empty plus node
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::make_determinant" << endl;
	}

	// For creating the set of permutations of n:
	combinatorics::other_combinatorics::combinatorics_domain Combi;
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
		cout << "syntax_tree_node::make_determinant "
				"N = " << N << endl;
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



	for (a = 0; a < N; a++) {

		// create the permutations in the order determined by the Lehmercode:

		if (a == 0) {
			Combi.Permutations->first_lehmercode(n, lehmer_code);
		}
		else {
			Combi.Permutations->next_lehmercode(n, lehmer_code);
		}
		Combi.Permutations->lehmercode_to_permutation(
				n, lehmer_code, perm);


		if (f_v) {
			cout << "syntax_tree_node::make_determinant "
					"a = " << a << " / " << N
					<< " perm=";
			Int_vec_print(cout, perm, n);
			cout << endl;
		}

		sgn = Combi.Permutations->perm_signum(perm, n);
			// sgn is either +1 or -1.


		syntax_tree_node *mult_node;

		mult_node = NEW_OBJECT(syntax_tree_node);

		if (f_v) {
			cout << "syntax_tree_node::make_determinant "
					"a = " << a << " / " << N
					<< " before mult_node->init_empty_multiplication_node" << endl;
		}

		mult_node->init_empty_multiplication_node(
				Output_tree, 0 /*verbose_level*/);

		if (f_v) {
			cout << "syntax_tree_node::make_determinant "
					"a = " << a << " / " << N
					<< " after mult_node->init_empty_multiplication_node" << endl;
		}

		if (sgn == -1) {
			if (f_v) {
				cout << "syntax_tree_node::make_determinant "
						"a = " << a << " / " << N
						<< " before mult_node->add_numerical_factor" << endl;
			}
			mult_node->add_numerical_factor(
					minus_one, 0 /*verbose_level*/);
			if (f_v) {
				cout << "syntax_tree_node::make_determinant "
						"a = " << a << " / " << N
						<< " after mult_node->add_numerical_factor" << endl;
			}
		}

		int i;
		for (i = 0; i < n; i++) {


			syntax_tree_node *node;

			node = NEW_OBJECT(syntax_tree_node);

			// Get the entry (i, pi(i)) from the input matrix
			// and add it as a factor:

			V_in[i * n + perm[i]].tree->Root->copy_to(
					Output_tree,
					node, verbose_level - 2);

			mult_node->append_node(node, verbose_level - 2);

		}


		// add another summand to the addition:

		append_node(mult_node, verbose_level - 2);


	}

	FREE_int(perm);
	FREE_int(lehmer_code);

	if (f_v) {
		cout << "syntax_tree_node::make_determinant done" << endl;
	}
}

void syntax_tree_node::make_linear_combination(
		syntax_tree *Output_tree,
		syntax_tree_node *Node1a,
		syntax_tree_node *Node1b,
		syntax_tree_node *Node2a,
		syntax_tree_node *Node2b,
		int verbose_level)
// we assume that the node is an empty plus node
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::make_linear_combination" << endl;
	}


	syntax_tree_node *mult_node1;
	syntax_tree_node *mult_node2;

	mult_node1 = NEW_OBJECT(syntax_tree_node);
	mult_node2 = NEW_OBJECT(syntax_tree_node);


	mult_node1->init_empty_multiplication_node(Output_tree, verbose_level);
	mult_node2->init_empty_multiplication_node(Output_tree, verbose_level);


	append_node(mult_node1, 0 /*verbose_level*/);
	append_node(mult_node2, 0 /*verbose_level*/);


	syntax_tree_node *Node1a_copy;
	syntax_tree_node *Node1b_copy;
	syntax_tree_node *Node2a_copy;
	syntax_tree_node *Node2b_copy;

	Node1a_copy = NEW_OBJECT(syntax_tree_node);
	Node1b_copy = NEW_OBJECT(syntax_tree_node);
	Node2a_copy = NEW_OBJECT(syntax_tree_node);
	Node2b_copy = NEW_OBJECT(syntax_tree_node);

	Node1a->copy_to(Output_tree, Node1a_copy, 0 /*verbose_level*/);
	Node1b->copy_to(Output_tree, Node1b_copy, 0 /*verbose_level*/);
	Node2a->copy_to(Output_tree, Node2a_copy, 0 /*verbose_level*/);
	Node2b->copy_to(Output_tree, Node2b_copy, 0 /*verbose_level*/);

	mult_node1->append_node(Node1a_copy, 0 /*verbose_level*/);
	mult_node1->append_node(Node1b_copy, 0 /*verbose_level*/);

	mult_node2->append_node(Node2a_copy, 0 /*verbose_level*/);
	mult_node2->append_node(Node2b_copy, 0 /*verbose_level*/);


	if (f_v) {
		cout << "syntax_tree_node::make_linear_combination" << endl;
		cout << "syntax_tree_node::make_linear_combination Root=";
		print_subtree_easy(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "syntax_tree_node::make_linear_combination done" << endl;
	}

}

static int syntax_tree_node_compare_func(
		void *data,
		int i, int j, void *extra_data)
{
	int verbose_level = 0;

	syntax_tree_node *node = (syntax_tree_node *)extra_data;

	return node->Tree->compare_nodes(
			node->Nodes[i],
			node->Nodes[j],
			verbose_level);
}

static void syntax_tree_node_swap_func(
		void *data,
		int i, int j, void *extra_data)
{
	//int verbose_level = 0;

	syntax_tree_node *node = (syntax_tree_node *)extra_data;

	syntax_tree_node *node1;

	node1 = node->Nodes[i];
	node->Nodes[i] = node->Nodes[j];
	node->Nodes[j] = node1;
}




}}}}



