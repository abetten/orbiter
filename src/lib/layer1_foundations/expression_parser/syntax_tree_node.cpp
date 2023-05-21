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
namespace expression_parser {


static int syntax_tree_node_compare_func(void *data,
		int i, int j, void *extra_data);
static void syntax_tree_node_swap_func(void *data,
		int i, int j, void *extra_data);


syntax_tree_node::syntax_tree_node()
{
	Tree = NULL;
	idx = orbiter_kernel_system::Orbiter->syntax_tree_node_index;

	f_has_exponent = false;
	exponent = 0;

	orbiter_kernel_system::Orbiter->syntax_tree_node_index++;

	f_terminal = false;
	T = NULL;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;

	f_has_monomial = false;
	monomial = NULL;

	f_has_minus = false;
}

syntax_tree_node::~syntax_tree_node()
{

	if (f_terminal) {
		FREE_OBJECT(T);
		T = 0L;
	}
	else {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			FREE_OBJECT(Nodes[i]);
		}
		if (monomial) {
			FREE_int(monomial);
		}
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

}

void syntax_tree_node::add_numerical_factor(
		int value, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_factor" << endl;
	}

	int i;

	for (i = 0; i < nb_nodes; i++) {
		if (Nodes[i]->f_terminal && Nodes[i]->T->f_int) {
			if (value >= Nodes[i]->Tree->Fq->q) {
				cout << "syntax_tree_node::add_numerical_factor "
						"value is out of range: value = " << value
						<< " q=" << Nodes[i]->Tree->Fq->q << endl;
				exit(1);
			}
			if (value < 0) {
				cout << "syntax_tree_node::add_numerical_factor "
						"value is out of range: value = " << value << endl;
				exit(1);
			}
			Nodes[i]->T->value_int =
					Nodes[i]->Tree->Fq->mult(Nodes[i]->T->value_int, value);
			break;
		}
		if (Nodes[i]->f_terminal && Nodes[i]->T->f_double) {
			Nodes[i]->T->value_double *= value;
			break;
		}
	}
	if (i == nb_nodes) {
		int j;

		for (j = nb_nodes; j > 0; j--) {
			Nodes[j] = Nodes[i - 1];
		}
		Nodes[0] = NEW_OBJECT(syntax_tree_node);

		Nodes[0]->init_empty_terminal_node_int(
				Tree, value, verbose_level);
		nb_nodes++;

#if 0
		Nodes[0]->Tree = Tree;
		Nodes[0]->f_terminal = true;
		Nodes[0]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[0]->T->f_int = true;
		Nodes[0]->T->value_int = value;
#endif

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

	int i;

	for (i = 0; i < nb_nodes; i++) {
		if (Nodes[i]->f_terminal && Nodes[i]->T->f_int) {
			if (value >= Nodes[i]->Tree->Fq->q) {
				cout << "syntax_tree_node::add_numerical_summand "
						"value is out of range: value = " << value
						<< " q=" << Nodes[i]->Tree->Fq->q << endl;
				exit(1);
			}
			if (value < 0) {
				cout << "syntax_tree_node::add_numerical_summand "
						"value is out of range: value = " << value << endl;
				exit(1);
			}
			if (f_v) {
				cout << "syntax_tree_node::add_numerical_summand "
						"adding two numerical values: "
						<< Nodes[i]->T->value_int << " + " << value << endl;
			}
			Nodes[i]->T->value_int =
					Nodes[i]->Tree->Fq->add(Nodes[i]->T->value_int, value);
			break;
		}
		else if (Nodes[i]->f_terminal && Nodes[i]->T->f_double) {
			Nodes[i]->T->value_double += value;
			break;
		}
	}
	if (i == nb_nodes) {
		int j;

		for (j = nb_nodes; j > 0; j--) {
			Nodes[j] = Nodes[i - 1];
		}
		Nodes[0] = NEW_OBJECT(syntax_tree_node);

		Nodes[0]->init_empty_terminal_node_int(
				Tree, value, verbose_level);
#if 0
		Nodes[0]->Tree = Tree;
		Nodes[0]->f_terminal = true;
		Nodes[0]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[0]->T->f_int = true;
		Nodes[0]->T->value_int = value;
#endif
		nb_nodes++;
	}

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_summand done" << endl;
	}
}

int syntax_tree_node::text_value_match(std::string &factor)
{
	data_structures::string_tools ST;

	if (f_terminal && T->f_text &&
				ST.compare_string_string(T->value_text, factor) == 0) {
		return true;
	}
	else {
		return false;
	}
}

void syntax_tree_node::add_factor(
		std::string &factor, int exponent, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_factor" << endl;
	}

	int i;
	data_structures::string_tools ST;

	for (i = 0; i < nb_nodes; i++) {
		if (Nodes[i]->text_value_match(factor) /*f_terminal && Nodes[i]->T->f_text &&
				ST.compare_string_string(Nodes[i]->T->value_text, factor) == 0*/) {
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

		Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);

		Nodes[nb_nodes]->init_empty_terminal_node_text_with_exponent(
				Tree, factor, exponent, verbose_level);
#if 0
		Nodes[nb_nodes]->f_terminal = true;
		Nodes[nb_nodes]->Tree = Tree;
		Nodes[nb_nodes]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[nb_nodes]->T->f_text = true;
		Nodes[nb_nodes]->T->value_text.assign(factor);
		Nodes[nb_nodes]->f_has_exponent = true;
		Nodes[nb_nodes]->exponent = exponent;
#endif
		nb_nodes++;
		if (f_v) {
			cout << "syntax_tree_node::add_factor new nb_nodes = " << nb_nodes << endl;
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::add_factor done" << endl;
	}
}

#if 1
void syntax_tree_node::add_summand(
		std::string &summand, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_summand" << endl;
	}

#if 0
	int i;
	data_structures::string_tools ST;

	for (i = 0; i < nb_nodes; i++) {
		if (Nodes[i]->f_terminal && Nodes[i]->T->f_text &&
				ST.compare_string_string(Nodes[i]->T->value_text, summand) == 0) {
			break;
		}
	}
	if (i == nb_nodes) {
#endif

		Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);

		Nodes[nb_nodes]->init_empty_terminal_node_text(
				Tree, summand, verbose_level);
#if 0
		Nodes[nb_nodes]->f_terminal = true;
		Nodes[nb_nodes]->Tree = Tree;
		Nodes[nb_nodes]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[nb_nodes]->T->f_text = true;
		Nodes[nb_nodes]->T->value_text.assign(summand);
		Nodes[nb_nodes]->f_has_exponent = false;
		Nodes[nb_nodes]->exponent = 0;
#endif
		nb_nodes++;
	//}

	if (f_v) {
		cout << "syntax_tree_node::add_summand done" << endl;
	}
}
#endif

void syntax_tree_node::init_empty_terminal_node_int(
		syntax_tree *Tree,
		int value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_int" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = true;
	f_has_exponent = false;
	T = NEW_OBJECT(syntax_tree_node_terminal);
	T->f_int = true;
	T->value_int = value;
	type = operation_type_nothing;
	nb_nodes = 0;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_int done" << endl;
	}
}


void syntax_tree_node::init_empty_terminal_node_text(
		syntax_tree *Tree,
		std::string &value_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_text" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = true;
	f_has_exponent = false;
	T = NEW_OBJECT(syntax_tree_node_terminal);
	T->f_text = true;
	T->value_text.assign(value_text);
	type = operation_type_nothing;
	nb_nodes = 0;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_text done" << endl;
	}
}

void syntax_tree_node::init_empty_terminal_node_text_with_exponent(
		syntax_tree *Tree,
		std::string &value_text,
		int exponent,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_text_with_exponent" << endl;
	}

	syntax_tree_node::Tree = Tree;
	f_terminal = true;

	if (exponent != 1) {
		f_has_exponent = true;
		syntax_tree_node::exponent = exponent;
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
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_terminal_node_text_with_exponent done" << endl;
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

	data_structures::string_tools ST;

	Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
	Nodes[nb_nodes]->init_empty_plus_node_with_exponent(
			Tree, exponent, verbose_level);
#if 0
	Nodes[nb_nodes]->Tree = Tree;
	Nodes[nb_nodes]->f_terminal = false;
	Nodes[nb_nodes]->f_has_exponent = true;
	Nodes[nb_nodes]->exponent = exponent;
	Nodes[nb_nodes]->type = operation_type_add;
	Nodes[nb_nodes]->nb_nodes = 0;
#endif
	nb_nodes++;

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent done" << endl;
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
	//f_has_exponent = true;
	//syntax_tree_node::exponent = exponent;
	type = operation_type_add;
	nb_nodes = 0;
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_plus_node_with_exponent done" << endl;
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

	data_structures::string_tools ST;

	Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
	Nodes[nb_nodes]->init_empty_multiplication_node(
			Tree, verbose_level);
#if 0
	Nodes[nb_nodes]->Tree = Tree;
	Nodes[nb_nodes]->f_terminal = false;
	Nodes[nb_nodes]->f_has_exponent = false;
	Nodes[nb_nodes]->exponent = 0;
	Nodes[nb_nodes]->type = operation_type_mult;
	Nodes[nb_nodes]->nb_nodes = 0;
#endif
	nb_nodes++;

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

	data_structures::string_tools ST;

	Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
	Nodes[nb_nodes]->init_empty_node(
			Tree, verbose_level);
	nb_nodes++;

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
	f_has_monomial = false;
	f_has_minus = false;

	if (f_v) {
		cout << "syntax_tree_node::init_empty_node done" << endl;
	}
}




void syntax_tree_node::split_by_monomials(
		ring_theory::homogeneous_polynomial_domain *Poly,
		syntax_tree_node **Subtrees, int verbose_level)
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
				Nodes[i]->split_by_monomials(Poly, Subtrees, verbose_level);
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
				ret = Nodes[i]->is_homogeneous(degree, verbose_level);
				if (ret == false) {
					return false;
				}
			}
			return ret;
		}
	}

}

void syntax_tree_node::print_subtree_to_vector(
		std::vector<std::string> &rep, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::print_subtree_to_vector" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::print_subtree_to_vector is terminal" << endl;

		}
		//ost << "is terminal" << std::endl;
		T->print_to_vector(rep, verbose_level);
	}
	else {
		//ost << "with " << nb_nodes << " descendants" << std::endl;
		//ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;
		if (f_v) {
			cout << "syntax_tree_node::print_subtree_to_vector "
					"nb_nodes = " << nb_nodes<< endl;

		}
		for (i = 0; i < nb_nodes; i++) {
			string s;

			if (f_v) {
				cout << "syntax_tree_node::print_subtree_to_vector "
						"node " << i << " / " << nb_nodes<< endl;

			}

			s = "(";
			rep.push_back(s);
			Nodes[i]->print_subtree_to_vector(rep, verbose_level);
			s = ")";
			rep.push_back(s);

			if (i < nb_nodes - 1) {
				if (type == operation_type_mult) {
					s = " * ";
					rep.push_back(s);
				}
				else if (type == operation_type_add) {
					s = " + ";
					rep.push_back(s);
				}
			}
		}
	}

	if (f_has_exponent) {
		string s = " ^ " + std::to_string(exponent);
		rep.push_back(s);
	}
#if 0
	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}
#endif
	if (f_v) {
		cout << "syntax_tree_node::print_subtree_to_vector done" << endl;
	}

}


void syntax_tree_node::print_subtree(
		std::ostream &ost)
{

	ost << "Node " << idx << ": ";

	if (f_terminal) {
		ost << "is terminal" << std::endl;
		T->print(ost);
	}
	else {
		//ost << "with " << nb_nodes << " descendants" << std::endl;
		//ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i
					<< " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
		ost << "detailed list:" << std::endl;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i
					<< " is node " << Nodes[i]->idx << std::endl;
			Nodes[i]->print_subtree(ost);
		}
	}

	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
	}

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}


void syntax_tree_node::print_subtree_easy(
		std::ostream &ost)
{
	print_subtree_easy_without_monomial(ost);

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}
	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
	}

}

void syntax_tree_node::print_subtree_easy_without_monomial(
		std::ostream &ost)
{
	int i;

	//ost << "Node " << idx << ": ";

	if (f_terminal) {
		//ost << "is terminal" << std::endl;
		T->print_easy(ost);
	}
	else {
		if (nb_nodes == 1) {
			if (!is_mult_node() && !is_add_node()) {
				ost << "(neither add nor mult node):";
			}

			if (f_has_minus) {
				ost << "-";
			}
			Nodes[0]->print_subtree_easy(ost);

		}
		else {
			if (is_mult_node()) {
				if (f_has_minus) {
					ost << "-";
				}

				for (i = 0; i < nb_nodes; i++) {
					ost << "(";
					Nodes[i]->print_subtree_easy(ost);
					ost << ")";
					if (i < nb_nodes - 1) {
						ost << " * ";
					}
				}
			}
			else if (is_add_node()) {
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_subtree_easy(ost);
					if (i < nb_nodes - 1) {
						ost << " + ";
					}
				}
			}
			else {
				cout << "syntax_tree_node::print_easy_without_monomial "
						"unknown operation" << endl;
				exit(1);
			}
		}
	}
#if 0
	if (f_has_exponent) {
		ost << " ^ " << exponent << endl;
	}
#endif

}


void syntax_tree_node::print_node_type(
		std::ostream &ost)
{
	if (f_terminal) {
		ost << "terminal";
	}
	else {
		if (is_mult_node()) {
			ost << "mult";
		}
		else if (is_add_node()) {
			ost << "add";
		}
		else {
			cout << "syntax_tree_node::print_node_type unknown operation" << endl;
			exit(1);
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
		a = T->evaluate(symbol_table, Tree->Fq, verbose_level);
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
			a = Nodes[0]->evaluate(symbol_table, verbose_level);
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
					b = Nodes[i]->evaluate(symbol_table, verbose_level);
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
					b = Nodes[i]->evaluate(symbol_table, verbose_level);
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

void syntax_tree_node::print_expression(
		std::ostream &ost)
{
	int i;

	if (f_terminal) {
		if (f_has_minus) {
			ost << "-";
		}
		//T->print_expression(ost);
		T->print_graphviz(ost);
	}

	else {
		if (nb_nodes == 1) {
			if (f_has_minus) {
				ost << "-";
			}
			Nodes[0]->print_expression(ost);
		}
		else {
			ost << "(";
			if (type == operation_type_mult) {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
					if (i < nb_nodes - 1) {
						ost << "*";
					}
				}
			}
			else if (type == operation_type_add) {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
					if (i < nb_nodes - 1) {
						ost << "+";
					}
				}
			}
			else {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
				}
			}
			ost << ")";
		}
	}
	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
	}

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

void syntax_tree_node::print_without_recursion(
		std::ostream &ost)
{

	ost << "Node " << idx << ": ";

	if (f_terminal) {
		ost << "is terminal" << std::endl;
		//T->print(ost);
	}

	else {
		ost << "with " << nb_nodes << " descendants" << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i
					<< " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
	}

}


void syntax_tree_node::export_graphviz(
		std::string &name, std::ostream &ost)
{
	ost << "graph " << name << " {" << std::endl;

	export_graphviz_recursion(ost);

	ost << "}" << std::endl;

}

void syntax_tree_node::export_graphviz_recursion(
		std::ostream &ost)
{
	//ost << "Node " << idx << " nb_nodes=" << nb_nodes << endl;

	if (f_terminal) {
		//ost << "Node " << idx << " is terminal node" << endl;
		ost << idx << " [label=\"";
		T->print_graphviz(ost);
		ost << "\" ] ;" << std::endl;
	}
	else {
		int i;
		ost << idx << " [label=\"";


		if (nb_nodes == 1) {
			ost << "(...)";
		}
		else {
			if (type == operation_type_mult) {
				ost << "*";
			}
			else if (type == operation_type_add) {
				ost << "+";
			}
			else if (type == operation_type_nothing) {
				ost << " nop ";
			}
			else {
				cout << "syntax_tree_node::export_graphviz_recursion "
						"unknown operation" << endl;
				exit(1);
			}
		}
		ost << "\" ] ;" << std::endl;


		for (i = 0; i < nb_nodes; i++) {
			ost << idx << " -- " << Nodes[i]->idx << std::endl;
		}
		for (i = 0; i < nb_nodes; i++) {
			//ost << "recursing into node " << Nodes[i]->idx << endl;
			Nodes[i]->export_graphviz_recursion(ost);
			//ost << "recursing into node " << Nodes[i]->idx << " finished" << endl;
		}
	}

}

void syntax_tree_node::copy_to(
		syntax_tree *Output_tree,
		syntax_tree_node *Output_node,
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

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::copy_to terminal node" << endl;
		}
		Output_node->f_terminal = true;
		Output_node->T = NEW_OBJECT(syntax_tree_node_terminal);

		if (T->f_int) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to terminal node int" << endl;
			}
			Output_node->T->f_int = true;
			Output_node->T->value_int = T->value_int;
		}
		else if (T->f_double) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to terminal node double" << endl;
			}
			Output_node->T->f_double = true;
			Output_node->T->value_double = T->value_double;
		}
		else if (T->f_text) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to terminal node text" << endl;
			}
			Output_node->T->f_text = true;
			Output_node->T->value_text = T->value_text;
		}

	}
	else {
		int i;


		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to multiplication node" << endl;
			}
			Output_node->type = operation_type_mult;
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to addition node" << endl;
			}
			Output_node->type = operation_type_add;
		}
		else if (type == operation_type_nothing) {
			if (f_v) {
				cout << "syntax_tree_node::copy_to nothing node" << endl;
			}
			Output_node->type = operation_type_nothing;
		}
		else {
			cout << "syntax_tree_node::copy_to unknown operation" << endl;
			exit(1);
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
					verbose_level);

			Output_node2->Tree = Output_tree;
			//Output_node2->idx;

			Output_node2->f_has_exponent = Nodes[i]->f_has_exponent;
			Output_node2->exponent = Nodes[i]->exponent;

			Output_node2->f_has_monomial = Nodes[i]->f_has_monomial;
			Output_node2->monomial = Nodes[i]->monomial; // ToDo

			Output_node2->f_has_minus = Nodes[i]->f_has_minus;


			Output_node->Nodes[i] = Output_node2;

			if (f_v) {
				cout << "syntax_tree_node::copy_to child "
						<< i << " / " << nb_nodes << " done" << endl;
			}
		}

		Output_node->nb_nodes = nb_nodes;
	}


	if (f_v) {
		cout << "syntax_tree_node::copy_to done" << endl;
	}
}

void syntax_tree_node::substitute(
		std::vector<std::string> &variables,
		formula *Target,
		formula **Substitutions,
		syntax_tree *Input_tree,
		syntax_tree *Output_tree,
		syntax_tree_node *Output_node,
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
						verbose_level);
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
			cout << "syntax_tree_node::substitute not a terminal node" << endl;
		}


		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::substitute multiplication node" << endl;
			}
			Output_node->type = operation_type_mult;
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::substitute addition node" << endl;
			}
			Output_node->type = operation_type_add;
		}
		else if (type == operation_type_nothing) {
			if (f_v) {
				cout << "syntax_tree_node::substitute nothing node" << endl;
			}
			cout << "syntax_tree_node::substitute operation_type_nothing is not allowed" << endl;
			exit(1);
		}
		else {
			cout << "syntax_tree_node::substitute unknown operation" << endl;
			exit(1);
		}


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
					verbose_level);

			Output_node2->Tree = Output_tree;

			Output_node2->f_has_exponent = Nodes[i]->f_has_exponent;
			Output_node2->exponent = Nodes[i]->exponent;

			Output_node2->f_has_monomial = Nodes[i]->f_has_monomial;
			Output_node2->monomial = Nodes[i]->monomial; // ToDo

			Output_node2->f_has_minus = Nodes[i]->f_has_minus;


			Output_node->Nodes[i] = Output_node2;

			if (f_v) {
				cout << "syntax_tree_node::substitute "
						"child " << i << " / " << nb_nodes << " done" << endl;
			}
		}

		Output_node->nb_nodes = nb_nodes;
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
				val2 = Tree->Fq->power_verbose(val1, exp, verbose_level);
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
						"child " << i << " / " << nb_nodes << " before simplify" << endl;
			}
			Nodes[i]->simplify(verbose_level);
			if (f_v) {
				cout << "syntax_tree_node::simplify "
						"child " << i << " / " << nb_nodes << " after simplify" << endl;
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
				if (Nodes[i]->f_terminal
						&& Nodes[i]->T->f_int
						&& Nodes[i + 1]->f_terminal
						&& Nodes[i + 1]->T->f_int) {


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
											"raising to the power of " << exp << endl;
						}
						val = Tree->Fq->power(val1, exp);
						Nodes[i]->T->value_int = val;
						Nodes[i]->f_has_exponent = false;
						Nodes[i]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val1
									<< " to the power of " << exp << " = " << val << endl;
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
									"terminal " << val2 << " to the "
											"power of " << exp << " = " << val << endl;
						}
						val2 = val;
					}


					val3 = Tree->Fq->mult(val1, val2);
					if (f_v) {
						cout << "syntax_tree_node::simplify "
								<< val1 << " * " << val2 << " = " << val3 << endl;
					}
					Nodes[i]->T->value_int = val3;
					delete_one_child(i + 1, verbose_level);

					i--;
					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"nb_nodes=" << nb_nodes << endl;
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
				if (Nodes[i]->f_terminal
						&& Nodes[i]->T->f_int
						&& Nodes[i + 1]->f_terminal
						&& Nodes[i + 1]->T->f_int) {


					if (f_v) {
						cout << "syntax_tree_node::simplify "
								"combining children " << i
								<< " and " << i + 1 << " using addition" << endl;
					}

					int val1, val2, val3;

					val1 = Nodes[i]->T->value_int;

					if (Nodes[i]->f_has_exponent) {
						int exp = Nodes[i]->exponent;
						int val;

						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"node[" << i << "] "
											"raising to the power of " << exp << endl;
						}
						val = Tree->Fq->power(val1, exp);
						Nodes[i]->T->value_int = val;
						Nodes[i]->f_has_exponent = false;
						Nodes[i]->exponent = 0;
						if (f_v) {
							cout << "syntax_tree_node::simplify "
									"terminal " << val1
									<< " to the power of " << exp << " = " << val << endl;
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
									<< " to the power of " << exp << " = " << val << endl;
						}
						val2 = val;
					}


					val3 = Tree->Fq->add(val1, val2);
					Nodes[i]->T->value_int = val3;
					delete_one_child(i + 1, verbose_level);

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
									"power of " << exp << " = " << val1 << endl;
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

void syntax_tree_node::simplify_exponents(int verbose_level)
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
					cout << "syntax_tree_node::simplify_exponents Tree == NULL" << endl;
					exit(1);
				}
				if (Tree->Fq == NULL) {
					cout << "syntax_tree_node::simplify_exponents Tree->Fq == NULL" << endl;
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
								"eliminating exponent of 1 of literal terminal node" << endl;
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
				Nodes[i]->simplify_exponents(verbose_level);
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
				Nodes[i]->simplify_exponents(verbose_level);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::simplify_exponents done" << endl;
	}

}

void syntax_tree_node::collect_like_terms(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms" << endl;
	}

	if (f_terminal) {
	}
	else {

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}

			data_structures::sorting Sorting;

			Sorting.Heapsort_general(
					Nodes, nb_nodes,
					syntax_tree_node_compare_func,
					syntax_tree_node_swap_func,
					this);

			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->collect_like_terms(verbose_level);
			}

		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::collect_like_terms "
						"addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			data_structures::sorting Sorting;

			Sorting.Heapsort_general(
					Nodes, nb_nodes,
					syntax_tree_node_compare_func,
					syntax_tree_node_swap_func,
					this);
			int i;

			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->collect_like_terms(verbose_level);
			}
		}
	}


	if (f_v) {
		cout << "syntax_tree_node::collect_like_terms done" << endl;
	}
}




void syntax_tree_node::simplify_constants(int verbose_level)
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

				if (Nodes[i]->is_constant_one(verbose_level)) {
					if (nb_nodes > 1) {

						delete_one_child(i, verbose_level);

					}
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				if (Nodes[i]->is_constant_zero(verbose_level)) {

					delete_all_but_one_child(i, verbose_level);
					break;
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_constants(verbose_level);
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

				if (Nodes[i]->is_constant_zero(verbose_level)) {
					if (nb_nodes > 1) {

						delete_one_child(i, verbose_level);

					}
				}
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->simplify_constants(verbose_level);
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
		cout << "syntax_tree_node::flatten" << endl;
	}

	if (f_terminal) {
		if (f_v) {
			cout << "syntax_tree_node::flatten terminal node" << endl;
		}
	}
	else {
		int i;

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::flatten "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
		}
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::flatten "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
		}


		if (f_v) {
			display_children_by_type();
		}

		if (type == operation_type_mult) {
			if (f_v) {
				cout << "syntax_tree_node::flatten "
						"simplifying multiplication node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				if (Nodes[i]->is_mult_node() || Nodes[i]->nb_nodes == 1) {


					if (f_v) {
						cout << "syntax_tree_node::flatten "
								"folding children of " << i << endl;
					}

					if (Nodes[i]->f_has_exponent) {

						syntax_tree_node *old_node;

						old_node = Nodes[i];

						int exp;

						exp = old_node->exponent;

						int nb_n;

						nb_n = old_node->nb_nodes;

						int j;


						// Make space for the children's nodes.
						//
						// Move up by nb_n  - 1 because the old node is deleted:

						for (j = nb_nodes - 1; j > i; j--) {
							Nodes[j + nb_n - 1] = Nodes[j];
						}
						nb_nodes += nb_n - 1;
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

						syntax_tree_node *old_node;

						old_node = Nodes[i];


						int nb_n;

						nb_n = old_node->nb_nodes;

						int j;



						for (j = nb_nodes - 1; j > i; j--) {
							Nodes[j + nb_n - 1] = Nodes[j];
						}
						nb_nodes += nb_n - 1;
						for (j = 0; j < nb_n; j++) {
							Nodes[i + j] = old_node->Nodes[j];
							old_node->Nodes[j] = NULL;
						}
						old_node->nb_nodes = 0;
						FREE_OBJECT(old_node);

					}
					if (f_v) {
						cout << "syntax_tree_node::flatten i=" << i <<
								" nb_nodes=" << nb_nodes << endl;
					}
					i--; // do the present child again, because it has changed

				} // if (Nodes[i]->is_mult_node())
			} // next i

		} // if (type == operation_type_mult)
		else if (type == operation_type_add) {
			if (f_v) {
				cout << "syntax_tree_node::flatten "
						"simplifying addition node, "
						"nb_children = " << nb_nodes << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				if (f_v) {
					cout << "syntax_tree_node::flatten "
							"child " << i << " / " << nb_nodes << " before flatten" << endl;
				}
				Nodes[i]->flatten(verbose_level);
				if (f_v) {
					cout << "syntax_tree_node::flatten "
							"child " << i << " / " << nb_nodes << " after flatten" << endl;
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
					cout << "syntax_tree_node::flatten "
							"node[" << 0 << "] raising to the "
									"power of " << exp << endl;
				}
				val1 = Tree->Fq->power(val, exp);
				Nodes[0]->T->value_int = val1;
				Nodes[0]->f_has_exponent = false;
				Nodes[0]->exponent = 0;
				if (f_v) {
					cout << "syntax_tree_node::flatten "
							"terminal " << val << " to the "
									"power of " << exp << " = " << val1 << endl;
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
		cout << "syntax_tree_node::flatten done" << endl;
	}
}

void syntax_tree_node::display_children_by_type()
{
	int i;

	for (i = 0; i < nb_nodes; i++) {

		cout << "child " << i << " is ";
		if (Nodes[i]->f_terminal) {
			cout << "terminal node of type ";
			if (Nodes[i]->T->f_int) {
				cout << "int, value = " << Nodes[i]->T->value_int;
			}
			else if (Nodes[i]->T->f_double) {
				cout << "double";
			}
			else if (Nodes[i]->T->f_text) {
				cout << "text";
			}
			cout << endl;
		}
		else if (Nodes[i]->type == operation_type_mult) {
			cout << "multiplication node" << endl;
		}
		else if (Nodes[i]->type == operation_type_add) {
			cout << "addition node" << endl;
		}
	}

}

void syntax_tree_node::delete_all_but_one_child(int i, int verbose_level)
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

void syntax_tree_node::delete_one_child(int i, int verbose_level)
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

int syntax_tree_node::is_constant_one(int verbose_level)
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

int syntax_tree_node::is_constant_zero(int verbose_level)
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
					cout << "syntax_tree_node::is_constant_zero returns false. "
							"f_terminal, f_int are true but value is not zero " << endl;
				}
				return false;
			}
		}
		else {
			if (f_v) {
				cout << "syntax_tree_node::is_constant_zero returns false. "
						"f_terminal is true but f_int is not " << endl;
			}
			return false;
		}
	}
	else {
		if (f_v) {
			cout << "syntax_tree_node::is_constant_zero returns false. "
					"f_terminal is false. Node type is ";
			print_node_type(cout);
			cout << endl;
		}
		return false;
	}
}




static int syntax_tree_node_compare_func(void *data,
		int i, int j, void *extra_data)
{
	int verbose_level = 0;

	syntax_tree_node *node = (syntax_tree_node *)extra_data;

	return node->Tree->compare_nodes(
			node->Nodes[i],
			node->Nodes[j],
			verbose_level);
}

static void syntax_tree_node_swap_func(void *data,
		int i, int j, void *extra_data)
{
	int verbose_level = 0;

	syntax_tree_node *node = (syntax_tree_node *)extra_data;

	syntax_tree_node *node1;

	node1 = node->Nodes[i];
	node->Nodes[i] = node->Nodes[j];
	node->Nodes[j] = node1;
}




}}}



