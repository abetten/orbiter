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
	f_has_managed_variables = FALSE;
	//std::vector<std::string> managed_variables;
	Root = NULL;
}

void syntax_tree::print(std::ostream &ost)
{
	Root->print(ost);
}

void syntax_tree::print_easy(std::ostream &ost)
{
	Root->print_easy(ost);
}

void syntax_tree::print_monomial(std::ostream &ost, int *monomial)
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

int syntax_tree::identify_single_literal(std::string &single_literal)
{
	int i;

	//cout << "syntax_tree::identify_single_literal trying to identify " << single_literal << endl;
	for (i = 0; i < managed_variables.size(); i++) {
		if (strcmp(single_literal.c_str(), managed_variables[i].c_str()) == 0) {
			//cout << "syntax_tree::identify_single_literal literal identified as managed variable " << i << endl;
			return i;
		}
	}
	return -1;
}

int syntax_tree::is_homogeneous(int &degree, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::is_homogeneous" << endl;
	}
	int ret;

	if (!f_has_managed_variables) {
		ret = FALSE;
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
				Subtrees[i]->print_easy(cout);
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



}}}


