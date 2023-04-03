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


syntax_tree_node::syntax_tree_node()
{
	Tree = NULL;
	idx = orbiter_kernel_system::Orbiter->syntax_tree_node_index;
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

void syntax_tree_node::split_by_monomials(ring_theory::homogeneous_polynomial_domain *Poly,
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

void syntax_tree_node::print(std::ostream &ost)
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


void syntax_tree_node::print_easy(std::ostream &ost)
{
	print_easy_without_monomial(ost);

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}

void syntax_tree_node::print_easy_without_monomial(std::ostream &ost)
{
	int i;

	//ost << "Node " << idx << ": ";

	if (f_terminal) {
		//ost << "is terminal" << std::endl;
		T->print_easy(ost);
	}
	else {
		if (nb_nodes == 1) {
			if (f_has_minus) {
				ost << "-";
			}
			Nodes[0]->print_easy(ost);
		}
		else {
			if (is_mult()) {
				if (f_has_minus) {
					ost << "-";
				}

				for (i = 0; i < nb_nodes; i++) {
					ost << "(";
					Nodes[i]->print_easy(ost);
					ost << ")";
					if (i < nb_nodes - 1) {
						ost << " * ";
					}
				}
			}
			else if (is_add()) {
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_easy(ost);
					if (i < nb_nodes - 1) {
						ost << " + ";
					}
				}
			}
			else {
				cout << "syntax_tree_node::evaluate unknown operation" << endl;
				exit(1);
			}
		}
	}

}

int syntax_tree_node::is_mult()
{
	if (type == operation_type_mult) {
		return true;
	}
	else {
		return false;
	}
}

int syntax_tree_node::is_add()
{
	if (type == operation_type_add) {
		return true;
	}
	else {
		return false;
	}
}


int syntax_tree_node::evaluate(std::map<std::string, std::string> &symbol_table,
		field_theory::finite_field *F, int verbose_level)
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
		if (f_v) {
			cout << "syntax_tree_node::evaluate "
					"terminal node evaluates to " << a << endl;
		}
	}
	else {
		if (nb_nodes == 1) {
			a = Nodes[0]->evaluate(symbol_table, F, verbose_level);
			if (f_has_minus) {
				a = F->negate(a);
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
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->mult(a, b);
				}
				if (f_has_minus) {
					a = F->negate(a);
				}
				if (f_v) {
					cout << "syntax_tree_node::evaluate "
							"product evaluates to " << a << endl;
				}
			}
			else if (type == operation_type_add) {
				a = 0;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->add(a, b);
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



}}}



