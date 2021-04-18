/*
 * syntax_tree_node.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

int syntax_tree_node_index = 0;

syntax_tree_node::syntax_tree_node()
{
	Tree = NULL;
	idx = syntax_tree_node_index;
	syntax_tree_node_index++;

	f_terminal = false;
	T = NULL;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;

	f_has_monomial = FALSE;
	monomial = NULL;

	f_has_minus = FALSE;
}

syntax_tree_node::~syntax_tree_node()
{

	if (f_terminal) {
		delete T;
		T = 0L;
	}
	else {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			delete Nodes[i];
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

void syntax_tree_node::split_by_monomials(homogeneous_polynomial_domain *Poly,
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
				cout << "syntax_tree_node::split_by_monomials checking multiplication node" << endl;
			}
			idx = Poly->index_of_monomial(monomial);
			Subtrees[idx] = this;
		}
		else {
			int i;

			if (f_v) {
				cout << "syntax_tree_node::split_by_monomials splitting subtree" << endl;
			}
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->split_by_monomials(Poly, Subtrees, verbose_level);
			}
		}
	}
}

int syntax_tree_node::is_homogeneous(int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int deg, i;

	if (f_v) {
		cout << "syntax_tree_node::is_homogeneous Node " << idx << endl;
	}
	if (f_terminal) {
		return TRUE;
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
				cout << "syntax_tree_node::is_homogeneous node " << idx << " has degree " << deg << endl;
			}
			if (degree == -1) {
				degree = deg;
				if (f_v) {
					cout << "syntax_tree_node::is_homogeneous node " << idx << " setting degree to " << degree << endl;
				}
			}
			else {
				if (deg != degree) {
					if (f_v) {
						cout << "syntax_tree_node::is_homogeneous node " << idx << " has degree " << deg << " which is different from " << degree << ", so not homogeneous" << endl;
					}
					return FALSE;
				}
			}
			return TRUE;
		}
		else {
			int i, ret;

			if (f_v) {
				cout << "checking subtree" << endl;
			}
			ret = TRUE;
			for (i = 0; i < nb_nodes; i++) {
				ret = Nodes[i]->is_homogeneous(degree, verbose_level);
				if (ret == FALSE) {
					return FALSE;
				}
			}
			return ret;
		}
	}

}

void syntax_tree_node::print(std::ostream &ost)
{

	ost << "Node " << idx << ": ";

	if (f_terminal) {
		ost << "is terminal" << std::endl;
		T->print(ost);
	}

	else {
		ost << "with " << nb_nodes << " descendants" << std::endl;
		ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
		ost << "detailed list:" << std::endl;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			Nodes[i]->print(ost);
		}
	}
	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}



int syntax_tree_node::evaluate(std::map<std::string, std::string> &symbol_table,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int a, b;

	if (f_v) {
		cout << "syntax_tree_node::evaluate" << endl;
	}
	if (f_terminal) {
		a = T->evaluate(symbol_table, F, verbose_level);
		if (f_has_minus) {
			a = F->negate(a);
		}
	}
	else {
		if (nb_nodes == 1) {
			a = Nodes[0]->evaluate(symbol_table, F, verbose_level);
			if (f_has_minus) {
				a = F->negate(a);
			}
		}
		else {
			if (type == operation_type_mult) {
				a = 1;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->mult(a, b);
				}
				if (f_has_minus) {
					a = F->negate(a);
				}
			}
			else if (type == operation_type_add) {
				a = 0;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->add(a, b);
				}
			}
			else {
				cout << "syntax_tree_node::evaluate unknown operation" << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::evaluate done, value = " << a << endl;
	}
	return a;
}

void syntax_tree_node::print_expression(std::ostream &ost)
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

void syntax_tree_node::print_without_recursion(std::ostream &ost)
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
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
	}

}


void syntax_tree_node::export_graphviz(std::string &name, std::ostream &ost)
{
	ost << "graph " << name << " {" << std::endl;

	export_graphviz_recursion(ost);

	ost << "}" << std::endl;

}

void syntax_tree_node::export_graphviz_recursion(std::ostream &ost)
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
			else {
				cout << "syntax_tree_node::export_graphviz_recursion unknown operation" << endl;
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



}}

