/*
 * syntax_tree_node_io.cpp
 *
 *  Created on: Oct 4, 2023
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {


void syntax_tree_node::print_subtree_to_vector(
		std::vector<std::string> &rep,
		int f_latex,
		int verbose_level)
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
		if (f_has_exponent) {
			string s;
			if (f_latex) {
				s = "^{" + std::to_string(exponent) + "}";
			}
			else {
				s = " ^ " + std::to_string(exponent);
			}
			rep.push_back(s);
		}
	}
	else {

		if (f_has_exponent) {
			string s;
			s = "(";
			rep.push_back(s);
		}

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

			int f_need_parens;

			if (Nodes[i]->f_terminal) {
				f_need_parens = false;
			}
			else {
				f_need_parens = true;
			}

			if (f_need_parens) {
				s = "(";
			}
			else {
				s = "";
			}
			rep.push_back(s);
			Nodes[i]->print_subtree_to_vector(rep, f_latex, verbose_level);
			if (f_need_parens) {
				s = ")";
			}
			else {
				s = "";
			}
			rep.push_back(s);

			if (i < nb_nodes - 1) {
				if (type == operation_type_mult) {
					if (f_latex) {
						s = " ";
					}
					else {
						s = " * ";
					}
					rep.push_back(s);
				}
				else if (type == operation_type_add) {
					if (f_latex) {
						s = " + ";
					}
					else {
						s = " + ";
					}
					rep.push_back(s);
				}
			}
		}
		if (f_has_exponent) {
			string s;
			s = ")";
			rep.push_back(s);
			if (f_latex) {
				s = "^{" + std::to_string(exponent) + "}";
			}
			else {
				s = " ^ " + std::to_string(exponent);
			}
			rep.push_back(s);
		}
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
		if (f_has_exponent) {
			cout << " ^ " << exponent << endl;
		}

	}
	else {
		//ost << "with " << nb_nodes << " descendants" << std::endl;
		//ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;


		if (f_has_exponent) {
			cout << "(" << endl;
		}


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
		if (f_has_exponent) {
			cout << " ) ^ " << exponent << endl;
		}
	}


	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}


void syntax_tree_node::print_subtree_easy(
		std::ostream &ost)
{

	print_subtree_easy_no_lf(ost);
	ost << endl;

}

void syntax_tree_node::print_subtree_easy_no_lf(
		std::ostream &ost)
{

	if (f_has_exponent) {
		cout << " ( ";
	}
	print_subtree_easy_without_monomial(ost);

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}
	if (f_has_exponent) {
		cout << " ) ^ " << exponent << " ";
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
			Nodes[0]->print_subtree_easy_no_lf(ost);

		}
		else {
			if (is_mult_node()) {
				if (f_has_minus) {
					ost << "-";
				}

				ost << " ( ";
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_subtree_easy_no_lf(ost);
					if (i < nb_nodes - 1) {
						ost << " * ";
					}
				}
				ost << " ) ";
			}
			else if (is_add_node()) {
				ost << " ( ";
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_subtree_easy_no_lf(ost);
					if (i < nb_nodes - 1) {
						ost << " + ";
					}
				}
				ost << " ) ";
			}
			else {
				cout << "syntax_tree_node::print_subtree_easy_without_monomial "
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

void syntax_tree_node::display_children_by_type()
{
	int i;

	cout << "The node has " << nb_nodes << " children:" << endl;
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





}}}


