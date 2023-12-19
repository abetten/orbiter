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
	//std::string managed_variables_text
	//std::vector<std::string> managed_variables;

	Fq = NULL;

	Root = NULL;

	//std::vector<std::string> variables;

}

syntax_tree::~syntax_tree()
{
	if (Root) {
		FREE_OBJECT(Root);
	}
}

void syntax_tree::init(
		field_theory::finite_field *Fq,
		int f_managed_variables, std::string &managed_variables_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init" << endl;
	}
	if (f_v) {
		cout << "syntax_tree::init f_managed_variables = " << f_managed_variables << endl;
		if (f_managed_variables) {
			cout << "syntax_tree::init managed_variables_text = " << managed_variables_text << endl;
		}
	}

	syntax_tree::Fq = Fq;

	if (f_managed_variables) {

		f_has_managed_variables = true;

		if (f_v) {
			cout << "syntax_tree::init with managed variables: " << managed_variables_text << endl;
		}

		syntax_tree::managed_variables_text.assign(managed_variables_text);
		data_structures::string_tools ST;


		ST.parse_comma_separated_list(
				managed_variables_text, managed_variables,
				verbose_level);

		if (f_v) {
			cout << "syntax_tree::init "
					"number of managed variables = " << managed_variables.size() << endl;
		}
	}
	else {
		f_has_managed_variables = false;

	}

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

void syntax_tree::init_monopoly(
		field_theory::finite_field *Fq,
		std::string &variable,
		int *coeffs, int nb_coeffs,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::init_monopoly" << endl;
	}

	syntax_tree::Fq = Fq;

	init_root_node(verbose_level);


	Root->init_monopoly(
			this,
			variable,
			coeffs, nb_coeffs,
			verbose_level);

	if (f_v) {
		cout << "syntax_tree::init_monopoly done" << endl;
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

void syntax_tree::count_nodes(
		int &nb_add, int &nb_mult, int &nb_int,
		int &nb_text, int &max_degree)
{
	nb_add = 0;
	nb_mult = 0;
	nb_int = 0;
	nb_text = 0;
	max_degree = 0;
	Root->count_nodes(nb_add, nb_mult, nb_int, nb_text, max_degree);
}

int syntax_tree::nb_nodes_total()
{
	int nb_add, nb_mult, nb_int, nb_text, max_degree, nb_total;

	count_nodes(nb_add, nb_mult, nb_int, nb_text, max_degree);

	nb_total = nb_add + nb_mult + nb_int + nb_text;

	return nb_total;
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

	if (Root == NULL) {
		cout << "syntax_tree::copy_to Root == NULL" << endl;
		exit(1);
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
		std::string &variable,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::highest_order_term" << endl;
	}

	int d;

	if (f_v) {
		cout << "syntax_tree::highest_order_term "
				"before Root->highest_order_term" << endl;
	}
	d = Root->highest_order_term(variable, verbose_level);
	if (f_v) {
		cout << "syntax_tree::highest_order_term "
				"after Root->highest_order_term" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::highest_order_term done" << endl;
	}
	return d;
}

void syntax_tree::get_monopoly(
		std::string &variable, int *&coeff, int &nb_coeff,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::get_monopoly" << endl;
	}

	std::vector<int> Coeff;
	std::vector<int> Exp;

	if (f_v) {
		cout << "syntax_tree::get_monopoly "
				"before Root->get_monopoly" << endl;
	}
	Root->get_monopoly(variable,
			Coeff, Exp,
			verbose_level);
	if (f_v) {
		cout << "syntax_tree::get_monopoly "
				"after Root->get_monopoly" << endl;
	}

	int i, d, c, e;

	d = -1;
	for (i = 0; i < Coeff.size(); i++) {
		d = MAXIMUM(d, Exp[i]);
	}
	nb_coeff = d + 1;

	coeff = NEW_int(nb_coeff);
	Int_vec_zero(coeff, nb_coeff);

	for (i = 0; i < Coeff.size(); i++) {
		c = Coeff[i];
		e = Exp[i];
		coeff[e] = c;
	}
	if (f_v) {
		cout << "syntax_tree::get_monopoly "
				"found a monopoly of degree " << d << " : ";
		Int_vec_print(cout, coeff, nb_coeff);
		cout << endl;
	}


	if (f_v) {
		cout << "syntax_tree::get_monopoly" << endl;
	}
}

void syntax_tree::get_multipoly(
		ring_theory::homogeneous_polynomial_domain *HPD,
		int *&eqn, int &sz,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::get_multipoly" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::get_multipoly "
				"before Root->get_multipoly" << endl;
	}
	Root->get_multipoly(HPD,
			eqn, sz,
			verbose_level);
	if (f_v) {
		cout << "syntax_tree::get_multipoly "
				"after Root->get_multipoly" << endl;
	}

	if (f_v) {
		cout << "syntax_tree::get_multipoly done" << endl;
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

	mult_node->init_empty_multiplication_node(
			this, verbose_level - 2);

	mult_node->add_numerical_factor(
				minus_one, verbose_level - 2);

	mult_node->append_node(Root, 0 /* verbose_level */);
	//Nodes[mult_node->nb_nodes] = Root;
	//mult_node->nb_nodes++;

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

	syntax_tree_node *add_node;

	// This add_node will be the root node of the determinantal expression.
	// It will have n factorial many summands,
	// one for each permutation of n.

	add_node = NEW_OBJECT(syntax_tree_node);

	if (f_v) {
		cout << "syntax_tree::make_determinant "
				"before add_node->init_empty_plus_node_with_exponent" << endl;
	}
	add_node->init_empty_plus_node_with_exponent(
			this, 1 /* exponent */, verbose_level - 2);
	if (f_v) {
		cout << "syntax_tree::make_determinant "
				"after add_node->init_empty_plus_node_with_exponent" << endl;
	}

	Root = add_node;


	if (f_v) {
		cout << "syntax_tree::make_determinant "
				"before add_node->make_determinant" << endl;
	}
	add_node->make_determinant(
			this /* Output_tree */,
			Fq,
			V_in,
			n,
			verbose_level - 2);
	if (f_v) {
		cout << "syntax_tree::make_determinant "
				"after add_node->make_determinant" << endl;
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

				ret = ST.compare_string_string(
						Node1->T->value_text,
						Node2->T->value_text);

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


	add_node = NEW_OBJECT(syntax_tree_node);

	add_node->init_empty_plus_node_with_exponent(
			this, 1 /* exponent */, verbose_level);

	Root = add_node;

	if (f_v) {
		cout << "syntax_tree::make_linear_combination "
				"before add_node->make_linear_combination" << endl;
	}
	add_node->make_linear_combination(
			this /* *Output_tree */,
			Node1a,
			Node1b,
			Node2a,
			Node2b,
			verbose_level - 2);
	if (f_v) {
		cout << "syntax_tree::make_linear_combination "
				"after add_node->make_linear_combination" << endl;
	}

#if 0
	syntax_tree_node *mult_node1;
	syntax_tree_node *mult_node2;

	mult_node1 = NEW_OBJECT(syntax_tree_node);
	mult_node2 = NEW_OBJECT(syntax_tree_node);


	mult_node1->init_empty_multiplication_node(this, verbose_level);
	mult_node2->init_empty_multiplication_node(this, verbose_level);


	add_node->append_node(mult_node1, 0 /*verbose_level*/);
	add_node->append_node(mult_node2, 0 /*verbose_level*/);
	//add_node->Nodes[add_node->nb_nodes++] = mult_node1;
	//add_node->Nodes[add_node->nb_nodes++] = mult_node2;


	syntax_tree_node *Node1a_copy;
	syntax_tree_node *Node1b_copy;
	syntax_tree_node *Node2a_copy;
	syntax_tree_node *Node2b_copy;

	Node1a_copy = NEW_OBJECT(syntax_tree_node);
	Node1b_copy = NEW_OBJECT(syntax_tree_node);
	Node2a_copy = NEW_OBJECT(syntax_tree_node);
	Node2b_copy = NEW_OBJECT(syntax_tree_node);

	Node1a->copy_to(this, Node1a_copy, 0 /*verbose_level*/);
	Node1b->copy_to(this, Node1b_copy, 0 /*verbose_level*/);
	Node2a->copy_to(this, Node2a_copy, 0 /*verbose_level*/);
	Node2b->copy_to(this, Node2b_copy, 0 /*verbose_level*/);

	mult_node1->append_node(Node1a_copy, 0 /*verbose_level*/);
	mult_node1->append_node(Node1b_copy, 0 /*verbose_level*/);
	//mult_node1->Nodes[mult_node1->nb_nodes++] = Node1a_copy;
	//mult_node1->Nodes[mult_node1->nb_nodes++] = Node1b_copy;

	mult_node2->append_node(Node2a_copy, 0 /*verbose_level*/);
	mult_node2->append_node(Node2b_copy, 0 /*verbose_level*/);
	//mult_node2->Nodes[mult_node2->nb_nodes++] = Node2a_copy;
	//mult_node2->Nodes[mult_node2->nb_nodes++] = Node2b_copy;


	if (f_v) {
		cout << "syntax_tree::make_linear_combination" << endl;
		cout << "syntax_tree::make_linear_combination Root=";
		Root->print_subtree_easy(cout);
		cout << endl;
	}
#endif


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

void syntax_tree::export_tree(
		std::string &name, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::export_tree" << endl;
	}

	syntax_tree_latex L;

	L.export_tree(name, Root, verbose_level);

	if (f_v) {
		cout << "syntax_tree::export_tree done" << endl;
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
		cout << "syntax_tree::collect_variables "
				"f_has_managed_variables = " << f_has_managed_variables << endl;
		cout << "syntax_tree::collect_variables "
				"number of managed variables = " << managed_variables.size() << endl;
		int i;
		for (i = 0; i < managed_variables.size(); i++) {
			cout << i << " : " << managed_variables[i] << endl;
		}
	}

	if (f_v) {
		cout << "syntax_tree::collect_variables "
				"before Root->collect_variables" << endl;
	}
	Root->collect_variables(verbose_level);
	if (f_v) {
		cout << "syntax_tree::collect_variables "
				"after Root->collect_variables" << endl;
	}


	if (f_v) {
		cout << "syntax_tree::collect_variables "
				"the variables are:" << endl;
		print_variables(cout, verbose_level);
	}



	if (f_v) {
		cout << "syntax_tree::collect_variables done" << endl;
	}
}

void syntax_tree::print_variables(
		std::ostream &ost,
		int verbose_level)
{
	int i;

	if (f_has_managed_variables) {
		if (managed_variables.size() == 0) {
			cout << "even though there are managed variables, none is listed" << endl;
			exit(1);
		}
		else {
			for (i = 0; i < managed_variables.size(); i++) {
				ost << i << " : " << managed_variables[i] << endl;
			}
		}
	}
	else {
		cout << "there are no managed variables" << endl;
	}
	ost << "-" << endl;
	cout << "ordinary variables:" << endl;
	for (i = 0; i < variables.size(); i++) {
		ost << managed_variables.size() + i << " : " << variables[i] << endl;
	}
}

void syntax_tree::print_variables_in_line(
		std::ostream &ost)
{
	int i;

	if (f_has_managed_variables) {
		for (i = 0; i < managed_variables.size(); i++) {
			ost << managed_variables[i];
			if (i < managed_variables.size() - 1) {
				ost << ",";
			}
		}
		ost << ";";
	}
	for (i = 0; i < variables.size(); i++) {
		ost << variables[i];
		if (i < variables.size() - 1) {
			ost << ",";
		}
	}
}

int syntax_tree::find_variable(
		std::string &var,
		int verbose_level)
{
	data_structures::string_tools String;


	int i, cmp, idx;

	idx = find_managed_variable(var,
			0 /* verbose_level */);

	if (idx >= 0) {
		return idx;
	}
	else {
		for (i = 0; i < variables.size(); i++) {
			cmp = String.compare_string_string(variables[i], var);
			if (cmp == 0) {
				return managed_variables.size() + i;
			}
		}
	}
	return -1;
}

int syntax_tree::find_managed_variable(
		std::string &var,
		int verbose_level)
{
	data_structures::string_tools String;


	if (!f_has_managed_variables) {
		return -1;
	}

	int i, cmp;

	for (i = 0; i < managed_variables.size(); i++) {
		cmp = String.compare_string_string(
				managed_variables[i], var);
		if (cmp == 0) {
			return i;
		}
	}
	return -1;
}

void syntax_tree::add_variable(
		std::string &var, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::add_variable var=" << var << endl;
	}
	data_structures::string_tools String;
	int cmp;
	std::vector<std::string>::iterator it;
	int idx;

	idx = find_managed_variable(var,
			0 /* verbose_level */);
	if (idx >= 0) {
		if (f_v) {
			cout << "syntax_tree::add_variable "
					"var=" << var << " is managed variable at idx=" << idx << endl;
		}
		return;
	}
	else {
		if (f_v) {
			cout << "syntax_tree::add_variable "
					"var=" << var << " is not a managed variable" << endl;
		}
		for (it = variables.begin(); it < variables.end(); it++) {
			cmp = String.compare_string_string(*it, var);
			if (cmp > 0) {
				variables.insert(it, var);
				return;
			}
		}
		variables.push_back(var);
	}
}

int syntax_tree::get_number_of_variables()
{
	int nb = 0;

	if (f_has_managed_variables) {
		nb += managed_variables.size();
	}
	nb += variables.size();
	return nb;

}

std::string &syntax_tree::get_variable_name(int index)
{
	if (index < managed_variables.size()) {
		return managed_variables[index];
	}
	index -= managed_variables.size();
	return variables[index];
}


int syntax_tree::needs_to_be_expanded()
{
	return Root->needs_to_be_expanded();
}

long int syntax_tree::evaluate(
		std::map<std::string, std::string> &symbol_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::evaluate" << endl;
	}
	long int a;

	a = Root->evaluate(
			symbol_table,
			verbose_level - 1);
	if (f_v) {
		cout << "syntax_tree::evaluate done" << endl;
	}
	return a;
}



}}}


