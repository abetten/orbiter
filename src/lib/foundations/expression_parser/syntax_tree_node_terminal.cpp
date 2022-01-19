/*
 * syntax_tree_node_terminal.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {

syntax_tree_node_terminal::syntax_tree_node_terminal()
{
	f_int = false;
	f_double = false;
	f_text = false;
	value_int = 0;
	value_double = 0.;
	//value_text;

}


void syntax_tree_node_terminal::print(std::ostream &ost)
{
	ost << "terminal node, ";
	if (f_int) {
		ost << "int=" << value_int << std::endl;
	}
	else if (f_double) {
		ost << "double=" << value_double << std::endl;
	}
	else if (f_text) {
		ost << "text=" << value_text << std::endl;
	}
}

void syntax_tree_node_terminal::print_expression(std::ostream &ost)
{
	if (f_int) {
		ost << value_int;
	}
	else if (f_double) {
		ost << value_double;
	}
	else if (f_text) {
		ost << value_text;
	}
}

void syntax_tree_node_terminal::print_graphviz(std::ostream &ost)
{
	if (f_int) {
		ost << value_int;
	}
	else if (f_double) {
		ost << value_double;
	}
	else if (f_text) {
		ost << value_text;
	}
	else {
		cout << "syntax_tree_node_terminal::print_graphviz unknown type" << endl;
		exit(1);
	}

}
int syntax_tree_node_terminal::evaluate(std::map<std::string, std::string> &symbol_table,
		field_theory::finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "syntax_tree_node_terminal::evaluate" << endl;
	}
	if (f_int) {
		a = value_int;
	}
	else if (f_double) {
		a = value_double;
		//cout << "syntax_tree_node_terminal::evaluate f_double" << endl;
		//exit(1);
	}
	else if (f_text) {
		//a = strtoi(value_text);
		a = ST.strtoi(symbol_table[value_text]);
	}
	else {
		cout << "syntax_tree_node_terminal::evaluate unknown type" << endl;
		exit(1);
	}

	if (a < 0) {
		cout << "syntax_tree_node_terminal::evaluate a < 0" << endl;
		exit(1);
	}
	if (a >= F->q) {
		cout << "syntax_tree_node_terminal::evaluate a >= F->q" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "syntax_tree_node_terminal::evaluate done, value = " << a << endl;
	}
	return a;
}



}}

