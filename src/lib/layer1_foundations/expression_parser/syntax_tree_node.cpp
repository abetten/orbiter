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
			Nodes[i]->T->value_int  = Nodes[i]->Tree->Fq->mult(Nodes[i]->T->value_int, value);
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
		nb_nodes++;
		Nodes[0]->Tree = Tree;
		Nodes[0]->f_terminal = true;
		Nodes[0]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[0]->T->f_int = true;
		Nodes[0]->T->value_int = value;
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
			Nodes[i]->T->value_int = Nodes[i]->Tree->Fq->add(Nodes[i]->T->value_int, value);
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
		nb_nodes++;
		Nodes[0]->Tree = Tree;
		Nodes[0]->f_terminal = true;
		Nodes[0]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[0]->T->f_int = true;
		Nodes[0]->T->value_int = value;
	}

	if (f_v) {
		cout << "syntax_tree_node::add_numerical_summand done" << endl;
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
		if (Nodes[i]->f_terminal && Nodes[i]->T->f_text &&
				ST.compare_string_string(Nodes[i]->T->value_text, factor) == 0) {
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
		int j;

		Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
		Nodes[nb_nodes]->f_terminal = true;
		Nodes[nb_nodes]->Tree = Tree;
		Nodes[nb_nodes]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[nb_nodes]->T->f_text = true;
		Nodes[nb_nodes]->T->value_text.assign(factor);
		Nodes[nb_nodes]->f_has_exponent = true;
		Nodes[nb_nodes]->exponent = exponent;
		nb_nodes++;
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
		Nodes[nb_nodes]->f_terminal = true;
		Nodes[nb_nodes]->Tree = Tree;
		Nodes[nb_nodes]->T = NEW_OBJECT(syntax_tree_node_terminal);
		Nodes[nb_nodes]->T->f_text = true;
		Nodes[nb_nodes]->T->value_text.assign(summand);
		Nodes[nb_nodes]->f_has_exponent = false;
		Nodes[nb_nodes]->exponent = 0;
		nb_nodes++;
	//}

	if (f_v) {
		cout << "syntax_tree_node::add_summand done" << endl;
	}
}
#endif

void syntax_tree_node::add_empty_plus_node_with_exponent(
		int exponent, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent" << endl;
	}

	int i;
	data_structures::string_tools ST;

	Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
	Nodes[nb_nodes]->Tree = Tree;
	Nodes[nb_nodes]->f_terminal = false;
	Nodes[nb_nodes]->f_has_exponent = true;
	Nodes[nb_nodes]->exponent = exponent;
	Nodes[nb_nodes]->type = operation_type_add;
	Nodes[nb_nodes]->nb_nodes = 0;
	nb_nodes++;

	if (f_v) {
		cout << "syntax_tree_node::add_empty_plus_node_with_exponent done" << endl;
	}
}

void syntax_tree_node::add_empty_multiplication_node(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node" << endl;
	}

	int i;
	data_structures::string_tools ST;

	Nodes[nb_nodes] = NEW_OBJECT(syntax_tree_node);
	Nodes[nb_nodes]->Tree = Tree;
	Nodes[nb_nodes]->f_terminal = false;
	Nodes[nb_nodes]->f_has_exponent = false;
	Nodes[nb_nodes]->exponent = 0;
	Nodes[nb_nodes]->type = operation_type_mult;
	Nodes[nb_nodes]->nb_nodes = 0;
	nb_nodes++;

	if (f_v) {
		cout << "syntax_tree_node::add_empty_multiplication_node done" << endl;
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

void syntax_tree_node::print_to_vector(std::vector<std::string> &rep)
{
	if (f_terminal) {
		//ost << "is terminal" << std::endl;
		T->print_to_vector(rep);
	}
	else {
		//ost << "with " << nb_nodes << " descendants" << std::endl;
		//ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			string s;

			s = "(";
			rep.push_back(s);
			Nodes[i]->print_to_vector(rep);
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

}


void syntax_tree_node::print(
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
			Nodes[i]->print(ost);
		}
	}

	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
	}

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}


void syntax_tree_node::print_easy(
		std::ostream &ost)
{
	print_easy_without_monomial(ost);

	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}
	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
	}

}

void syntax_tree_node::print_easy_without_monomial(
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
	if (f_has_exponent) {
		cout << " ^ " << exponent << endl;
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
					"before raising to the power of " << exponent << ", a=" << a << endl;
		}
		a = Tree->Fq->power(a, exponent);
		if (f_v) {
			cout << "syntax_tree_node::evaluate "
					"after raising to the power of " << exponent << ", a=" << a << endl;
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



