/*
 * expression_parser_sajeeb.cpp
 *
 *  Created on: Apr 1, 2023
 *      Author: betten
 */



#include "foundations.h"

#include <map>
#include <unordered_map>

// This always needs to be included
#include "../expression_parser_sajeeb/parser.h"

//#include "../expression_parser_sajeeb/parser.tab.hpp"
//#include "../expression_parser_sajeeb/lexer.yy.h"

// This only needs to be included if the tree is to be visited
#include "../expression_parser_sajeeb/Visitors/dispatcher.h"

#include "../expression_parser_sajeeb/Visitors/PrintVisitors/ir_tree_pretty_print_visitor.h"
#include "../expression_parser_sajeeb/Visitors/uminus_distribute_and_reduce_visitor.h"
#include "../expression_parser_sajeeb/Visitors/merge_nodes_visitor.h"
#include "../expression_parser_sajeeb/Visitors/LatexVisitors/ir_tree_latex_visitor_strategy.h"
#include "../expression_parser_sajeeb/Visitors/ToStringVisitors/ir_tree_to_string_visitor.h"
#include "../expression_parser_sajeeb/Visitors/remove_minus_nodes_visitor.h"
#include "../expression_parser_sajeeb/Visitors/ExpansionVisitors/multiplication_expansion_visitor.h"
#include "../expression_parser_sajeeb/Visitors/CopyVisitors/deep_copy_visitor.h"
#include "../expression_parser_sajeeb/Visitors/exponent_vector_visitor.h"
#include "../expression_parser_sajeeb/Visitors/ReductionVisitors/simplify_numerical_visitor.h"
#include "../expression_parser_sajeeb/Visitors/EvaluateVisitors/eval_visitor.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace l1_interfaces {


static void convert_to_orbiter_recursion(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		const shared_ptr<irtree_node> current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
static void node_type_as_string(
		shared_ptr<irtree_node> current_node, std::string &node_type_as_string);
static void handle_unary_negate_node(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		unary_negate_node *node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
static void handle_exponent_node(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		exponent_node *node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
static int get_exponent(
		shared_ptr<irtree_node> child2,
		int verbose_level);
static void collect_factors(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		non_terminal_node *current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
static void collect_summands(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		plus_node *current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
#if 0
static void collect_unary_negate_node(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		const shared_ptr<irtree_node>& child,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level);
#endif


#if 0
class get_latex_staged_visitor_functor final {
    int stage_counter = 0;
    std::ofstream latex_output_stream;
    std::string directory;
    ir_tree_latex_visitor_strategy::type tree_type;
    ir_tree_latex_visitor_strategy::wrapper strategy_wrapper;

public:
    get_latex_staged_visitor_functor(std::string directory,
                                     ir_tree_latex_visitor_strategy::type tree_type)
                                     : directory(directory),
                                       tree_type(tree_type),
                                       strategy_wrapper(ir_tree_latex_visitor_strategy::get(tree_type)) {}

    virtual ~get_latex_staged_visitor_functor() {
        if (latex_output_stream.is_open()) latex_output_stream.close();
    }

    IRTreeVoidReturnTypeVisitorInterface& operator()() {
        if (latex_output_stream.is_open()) latex_output_stream.close();
        latex_output_stream.open(directory + "stage" + std::to_string(stage_counter++) + ".tex");

        strategy_wrapper.set_output_stream(latex_output_stream);
        return *strategy_wrapper.get_visitor();
    }
};
#endif



class expression_parser_sajeeb_private_data {

public:

	managed_variables_index_table managed_variables_table;

	shared_ptr<irtree_node> ir_tree_root;

	exponent_vector_visitor *evv;

	int nb_monomials;
	int *table_of_monomials;
};

expression_parser_sajeeb::expression_parser_sajeeb()
{
	Formula = NULL;

	private_data = NULL;
}

expression_parser_sajeeb::~expression_parser_sajeeb()
{
}

static int stage_counter = 0;

void expression_parser_sajeeb::init_formula(
		expression_parser::formula *Formula,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula" << endl;
	}

#if 0
	name_of_formula.assign(label);
	name_of_formula_latex.assign(label_tex);
	formula::managed_variables.assign(managed_variables);
	formula::formula_text.assign(formula_text);
#endif

	expression_parser_sajeeb::Formula = Formula;

	//expression_parser Parser;
	data_structures::string_tools ST;
	int i;

	//tree = NEW_OBJECT(syntax_tree);

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Formula " << Formula->name_of_formula
				<< " is " << Formula->formula_text << endl;
		cout << "expression_parser_sajeeb::init_formula "
				"Managed variables: " << Formula->managed_variables << endl;
	}


	expression_parser_sajeeb_private_data *PD;

	PD = new expression_parser_sajeeb_private_data;

	private_data = PD;


	//managed_variables_index_table managed_variables_table;

	vector<string> vec_managed_variables;

	ST.parse_comma_separated_list(
			Formula->managed_variables, vec_managed_variables,
			verbose_level);

	for (i = 0; i < vec_managed_variables.size(); i++) {
		PD->managed_variables_table.insert(vec_managed_variables[i]);

	}


	Formula->nb_managed_vars = PD->managed_variables_table.size();
	//nb_managed_vars = tree->managed_variables.size();

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Managed variables are: " << endl;
		for (i = 0; i < Formula->nb_managed_vars; i++) {
			//cout << i << " : " << tree->managed_variables[i] << endl;
			//cout << i << " : " << managed_variables_table[i] << endl;
		}
	}


	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Starting to parse " << Formula->name_of_formula << endl;
	}


	PD->ir_tree_root =
    		parser::parse_expression(
    				Formula->formula_text, PD->managed_variables_table);



	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"Parsing " << Formula->name_of_formula << " finished" << endl;
	}

	{
		string prefix;

		prefix = "./sajeeb_tree" + std::to_string(stage_counter) + ".tex";
		{
		ofstream ost(prefix);

		dispatcher::visit(PD->ir_tree_root, ir_tree_latex_visitor_simple_tree(ost));
		}

		stage_counter++;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"before remove_minus_nodes_visitor" << endl;
	}
    dispatcher::visit(PD->ir_tree_root, remove_minus_nodes_visitor());
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"after remove_minus_nodes_visitor" << endl;
	}

#if 0
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"before merge_nodes_visitor" << endl;
	}
    dispatcher::visit(PD->ir_tree_root, merge_nodes_visitor());
	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula "
				"after merge_nodes_visitor" << endl;
	}
#endif


	Formula->f_is_homogeneous = true;

#if 0
    //exponent_vector_visitor evv;

	PD->evv = new exponent_vector_visitor;

    dispatcher::visit(PD->ir_tree_root, (*PD->evv)(PD->managed_variables_table));



	{
		string prefix;

		prefix = Formula->name_of_formula + "_tree.tex";
		{
		ofstream ost(prefix);

		dispatcher::visit(PD->ir_tree_root, ir_tree_latex_visitor_simple_tree(ost));
		}

	}
#endif


	#if 0
	unordered_map<string, int> assignemnt = {
			{"a", 4},
			{"b", 2},
			{"c", 2},
			{"d", 4}
	};
	for (auto& it : evv.monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: ";

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, &Fq, assignemnt);
			val += tmp;
		}
		cout << val << endl;
	}
#endif


#if 0
	if (false) {
		cout << "expression_parser_sajeeb::init_formula Syntax tree:" << endl;
		tree->print(cout);
	}

	std::string fname;
	fname = ame_of_formula + ".gv";

	{
		std::ofstream ost(fname);
		tree->Root->export_graphviz(name_of_formula, ost);
	}

	if (f_is_homogeneous) {
		cout << "expression_parser_sajeeb::init_formula before tree->is_homogeneous" << endl;
	}
	f_is_homogeneous = tree->is_homogeneous(degree, verbose_level - 3);
	if (f_is_homogeneous) {
		cout << "expression_parser_sajeeb::init_formula after tree->is_homogeneous" << endl;
	}

	if (f_is_homogeneous) {
		cout << "expression_parser_sajeeb::init_formula the formula is homogeneous of degree " << degree << endl;
	}
#endif

	if (f_v) {
		cout << "expression_parser_sajeeb::init_formula done" << endl;
	}
}

string expression_parser_sajeeb::string_representation()
{
	int f_v = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::string_representation" << endl;
	}

	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;

	ir_tree_to_string_visitor to_string_visitor;


	dispatcher::visit(PD->ir_tree_root, to_string_visitor);
	return to_string_visitor.get_string_representation();

}


void expression_parser_sajeeb::get_subtrees(
		ring_theory::homogeneous_polynomial_domain *Poly,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees" << endl;
	}

	//int i;

	//nb_monomials = Poly->get_nb_monomials();

	//tree->split_by_monomials(Poly, Subtrees, verbose_level);



	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;


	int deg = -1, d = 0;
	int j;

	PD->nb_monomials = 0;

	for (auto& it : PD->evv->monomial_coefficient_table_) {

		const vector<unsigned int>& vec = it.first;

		if (vec.size() != Poly->nb_variables) {
			cout << "expression_parser_sajeeb::get_subtrees "
					"vec.size() != Poly->nb_variables" << endl;
			exit(1);
		}
		d = 0;
		for (j = 0; j < vec.size(); j++) {
			d += vec[j];
		}
		if (deg == -1) {
			deg = d;
		}
		else {
			if (d != deg) {
				cout << "expression_parser_sajeeb::get_subtrees "
						"The polynomial is not homogeneous" << endl;
				exit(1);
			}
		}

		cout << PD->nb_monomials << " : ";
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: " << endl;
		PD->nb_monomials++;
	}

	Formula->degree = deg;


	Formula->f_is_homogeneous = true;


	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"homogeneous of degree " << Formula->degree << endl;
	}

	if (Formula->degree != Poly->degree) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"degree != Poly->degree" << endl;
		exit(1);
	}


	PD->table_of_monomials = NEW_int(PD->nb_monomials * Poly->nb_variables);

	int i;

	i = 0;
	for (auto& it : PD->evv->monomial_coefficient_table_) {


		const vector<unsigned int>& vec = it.first;
		for (j = 0; j < Poly->nb_variables; j++) {
			PD->table_of_monomials[i * Poly->nb_variables + j] = vec[j];
		}
		i++;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees "
				"table_of_monomials=" << endl;
		Int_matrix_print(
				PD->table_of_monomials,
				PD->nb_monomials,
				Poly->nb_variables);
	}


#if 0
	unordered_map<string, int> assignemnt = {
			{"a", 4},
			{"b", 2},
			{"c", 2},
			{"d", 4}
	};
	for (auto& it : evv.monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		std::cout << "[";
		for (const auto& itit : vec) std::cout << itit << " ";
		std::cout << "]: ";

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, &Fq, assignemnt);
			val += tmp;
		}
		cout << val << endl;
	}
#endif


	if (f_v) {
		cout << "expression_parser_sajeeb::get_subtrees done" << endl;
	}
}

void expression_parser_sajeeb::evaluate(
		ring_theory::homogeneous_polynomial_domain *Poly,
		std::map<std::string, std::string> &symbol_table, int *Values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::evaluate" << endl;
	}


#if 0
	cout << "symbol table:" << endl;
	for (i = 0; i < symbol_table.size(); i++) {
		cout << i << " : " << symbol_table[i] << " = " << values[i] << endl;
	}
#endif


	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;


	int i;

	Int_vec_zero(Values, Poly->get_nb_monomials());


	//Values = NEW_int(nb_monomials);

#if 0
	for (i = 0; i < Poly->get_nb_monomials(); i++) {
		cout << "Monomial " << i << " : ";
		if (Subtrees[i]) {
			//Subtrees[i]->print_expression(cout);
			a = Subtrees[i]->evaluate(symbol_table, Poly->get_F(), verbose_level);
			Values[i] = a;
			cout << a << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		else {
			cout << "no subtree" << endl;
			Values[i] = 0;
		}
	}
#endif

	unordered_map<string, int> assignment;

	//= {
	//		{"a", 4},
	//		{"b", 2},
	//		{"c", 2},
	//		{"d", 4}
	//};


	{
		std::map<std::string, std::string>::iterator it = symbol_table.begin();

		// Iterate through the map and print the elements
		while (it != symbol_table.end()) {
			int a;
			string label;
			string val;

			label = it->first;
			val = it->second;
			a = stoi(val);
			//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
			assignment.insert(std::make_pair(label, a));
			++it;
		}
	}

	eval_visitor evalVisitor;

	i = 0;
	for (auto& it : PD->evv->monomial_coefficient_table_) {
		const vector<unsigned int>& vec = it.first;
		if (f_v) {
			std::cout << "[";
			for (const auto& itit : vec) std::cout << itit << " ";
			std::cout << "]: ";
		}

		auto root_nodes = it.second;
		int val = 0;
		for (auto& node : root_nodes) {
			auto tmp = dispatcher::visit(node, evalVisitor, Poly->get_F(), assignment);

			val += tmp; // we should never have to add anything here. Why do we need this line?

		}

		int idx;

		idx = Poly->index_of_monomial(
				PD->table_of_monomials + i * Poly->nb_variables);

		Values[idx] = Poly->get_F()->add(Values[idx], val);

		if (f_v) {
			cout << i << " : " << val << " : ";
			Int_vec_print(cout,
					PD->table_of_monomials + i * Poly->nb_variables,
					Poly->nb_variables);
			cout << " : " << val << " : " << idx;
			cout << endl;
		}
		i++;
	}


	if (f_v) {
		cout << "evaluated polynomial:" << endl;
		for (i = 0; i < Poly->get_nb_monomials(); i++) {
			cout << Values[i] << " * ";
			Poly->print_monomial(cout, i);
			cout << endl;
		}
		cout << "coefficient vector: ";
		Int_vec_print(cout, Values, Poly->get_nb_monomials());
		cout << endl;
	}


	if (f_v) {
		cout << "expression_parser_sajeeb::evaluate done" << endl;
	}
}

void expression_parser_sajeeb::multiply(
		expression_parser_sajeeb **terms,
		int n,
		int &stage_counter,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::multiply" << endl;
	}
	if (f_v) {
		cout << "expression_parser_sajeeb::multiply n=" << n << endl;
	}


#if 1
	int i;

	vector<shared_ptr<irtree_node>> Ir_tree_root(n);

	if (f_v) {
		cout << "expression_parser_sajeeb::multiply setting Ir_tree_root" << endl;
	}

	for (i = 0; i < n; i++) {
		expression_parser_sajeeb_private_data *PD;

		PD = (expression_parser_sajeeb_private_data *) terms[i]->private_data;

		Ir_tree_root[i] = PD->ir_tree_root;
	}

	vector<shared_ptr<irtree_node>> Ir_tree_root_copy(n);

	//Ir_tree_root_copy = shared_ptr<irtree_node> [n];


	if (f_v) {
		cout << "expression_parser_sajeeb::multiply before deepCopyVisitor" << endl;
	}


	for (i = 0; i < n; i++) {


		Ir_tree_root_copy[i] = dispatcher::visit(Ir_tree_root[i], deep_copy_visitor());
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::multiply after deepCopyVisitor" << endl;
	}


	{
		shared_ptr<multiply_node> M = make_shared<multiply_node>();


		for (i = 0; i < n; i++) {
			if (f_v) {
				cout << "expression_parser_sajeeb::multiply i=" << i << " before M->add_child" << endl;
			}
			M->add_child((static_cast<non_terminal_node*> (Ir_tree_root_copy[i].get()))->children.front());
			//  need to remove the sentinel node

		}

		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before new sentinel_node" << endl;
		}

		shared_ptr<irtree_node> root = make_shared<sentinel_node>();
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before add_child" << endl;
		}

		(static_cast<non_terminal_node *>(root.get()))->add_child(M);


		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before latex" << endl;
		}
		{
			string prefix;

			prefix = "./tree" + std::to_string(stage_counter) + ".tex";
			{
			ofstream ost(prefix);

			dispatcher::visit(root, ir_tree_latex_visitor_simple_tree(ost));
			}

			stage_counter++;
		}
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply after latex" << endl;
		}
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before leaving scope" << endl;
		}
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::multiply after leaving scope" << endl;
	}

#endif

#if 0
	{
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply testing tree" << endl;
		}
		string a, b, c, d;

		a.assign("a");
		b.assign("b");
		c.assign("c");
		d.assign("d");
		//variable_node *A = new variable_node(a);
		//variable_node *B = new variable_node(b);
		//variable_node *C = new variable_node(c);
		//variable_node *D = new variable_node(d);
		//plus_node *P1;
		//plus_node *P2;
		shared_ptr<plus_node> P1 = make_shared<plus_node>();
		shared_ptr<plus_node> P2 = make_shared<plus_node>();
		shared_ptr<multiply_node> M = make_shared<multiply_node>();


		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before P1->add_child" << endl;
		}
		//P1 = new plus_node;
		//P2 = new plus_node;
		P1->add_child(make_shared<variable_node>(a), make_shared<variable_node>(b));
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before P2->add_child" << endl;
		}
		P2->add_child(make_shared<variable_node>(c), make_shared<variable_node>(d));

		//M = new multiply_node;

		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before M->add_child" << endl;
		}
		M->add_child(P1, P2);
		//M->add_child(P2);

		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before make_shared<sentinel_node>" << endl;
		}
		shared_ptr<irtree_node> root = make_shared<sentinel_node>();
		if (f_v) {
			cout << "expression_parser_sajeeb::multiply before add_child" << endl;
		}
		static_cast<non_terminal_node *>(root.get())->add_child(M);

		//shared_ptr<irtree_node> root;

		//root = std::shared_ptr<irtree_node>(node);

		get_latex_staged_visitor_functor
			get_latex_staged_visitor("./",
					ir_tree_latex_visitor_strategy::type::SIMPLE_TREE);
		dispatcher::visit(root, get_latex_staged_visitor());


		deep_copy_visitor deepCopyVisitor;
		shared_ptr<irtree_node> root_copy = dispatcher::visit(root, deepCopyVisitor);


		dispatcher::visit(root_copy, get_latex_staged_visitor());



	    if (f_v) {
			cout << "expression_parser_sajeeb::multiply testing tree done, before deleting object" << endl;
		}
	}
	if (f_v) {
		cout << "expression_parser_sajeeb::multiply testing tree done, after deleting object" << endl;
	}
#endif


#if 0
	ir_tree_to_string_visitor to_string_visitor;
	dispatcher::visit(M, to_string_visitor);
	cout << "in:  " << exp << endl;
	cout << "out: " << to_string_visitor.get_string_representation() << endl;
#endif


	if (f_v) {
		cout << "expression_parser_sajeeb::multiply "
				"before delete [] Ir_tree_root" << endl;
	}
	//delete [] Ir_tree_root;
	if (f_v) {
		cout << "expression_parser_sajeeb::multiply "
				"after delete [] Ir_tree_root" << endl;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::multiply done" << endl;
	}
}


void expression_parser_sajeeb::convert_to_orbiter(
		expression_parser::syntax_tree *&Tree,
		field_theory::finite_field *F,
		int f_has_managed_variables,
		std::string &managed_variables,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::convert_to_orbiter" << endl;
		cout << "expression_parser_sajeeb::convert_to_orbiter "
				"verbose_level = " << verbose_level << endl;
		cout << "expression_parser_sajeeb::convert_to_orbiter "
				"f_has_managed_variables = " << f_has_managed_variables << endl;
	}

	expression_parser_sajeeb_private_data *PD;

	PD = (expression_parser_sajeeb_private_data *) private_data;

	shared_ptr<irtree_node> root_node = PD->ir_tree_root;

	if (root_node->type == irtree_node::node_type::SENTINEL_NODE) {

		shared_ptr<irtree_node> current_node;

		current_node = (static_cast<non_terminal_node *>(root_node.get()))->children.front();

		Tree = NEW_OBJECT(expression_parser::syntax_tree);

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter "
					"before Tree->init" << endl;
		}
		Tree->init(F, f_has_managed_variables, managed_variables, verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter "
					"after Tree->init" << endl;
		}

		Tree->Root = NEW_OBJECT(expression_parser::syntax_tree_node);

		Tree->Root->Tree = Tree;


		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter "
					"before convert_to_orbiter_recursion" << endl;
		}
		convert_to_orbiter_recursion(
				this,
				Tree, current_node, Tree->Root,
				verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter "
					"after convert_to_orbiter_recursion" << endl;
		}

	}
	else {
		cout << "expression_parser_sajeeb::convert_to_orbiter "
				"top level node must be sentinel node" << endl;
	}

	if (f_v) {
		cout << "expression_parser_sajeeb::convert_to_orbiter done" << endl;
	}
}

static void convert_to_orbiter_recursion(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		const shared_ptr<irtree_node> current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "expression_parser_sajeeb::convert_to_orbiter_recursion" << endl;
	}
	if (current_node->type == irtree_node::node_type::SENTINEL_NODE) {
		cout << "convert_to_orbiter_recursion sentinel node is not allowed" << endl;
		exit(1);
	}
	else if (current_node->type == irtree_node::node_type::MULTIPLY_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion MULTIPLY_NODE" << endl;
		}

		current_node_copy->type = operation_type_mult;
		current_node_copy->nb_nodes = 0;

		current_node_copy->f_has_monomial = false;
		current_node_copy->f_has_minus = false;

		multiply_node *node = static_cast<multiply_node *>(current_node.get());

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before collect_factors" << endl;
		}
		collect_factors(
				root, Tree, node, current_node_copy,
				verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"after collect_factors" << endl;
		}
	}

	else if (current_node->type == irtree_node::node_type::PLUS_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion PLUS_NODE" << endl;
		}


		current_node_copy->type = operation_type_add;

		current_node_copy->f_has_monomial = false;
		current_node_copy->f_has_minus = false;

		plus_node *node = static_cast<plus_node *>(current_node.get());

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before collect_summands" << endl;
		}
		collect_summands(
				root, Tree, node, current_node_copy,
				verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"after collect_summands" << endl;
		}
	}
	else if (current_node->type == irtree_node::node_type::NUMBER_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion NUMBER_NODE" << endl;
		}

		number_node *node = static_cast<number_node *>(current_node.get());

		int value = node->value;

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"NUMBER_NODE value = " << value << endl;
		}

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_int" << endl;
		}
		current_node_copy->init_terminal_node_int(
				Tree,
				value,
				verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_int" << endl;
		}

	}
	else if (current_node->type == irtree_node::node_type::PARAMETER_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion PARAMETER_NODE" << endl;
		}


		parameter_node *node = static_cast<parameter_node *>(current_node.get());

		string name = node->name;

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"PARAMETER_NODE name = " << name << endl;
		}

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_text" << endl;
		}
		current_node_copy->init_terminal_node_text(
				Tree,
				name,
				verbose_level);  // ToDo

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_text" << endl;
		}

	}
	else if (current_node->type == irtree_node::node_type::VARIABLE_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion VARIABLE_NODE" << endl;
		}


		variable_node *node = static_cast<variable_node *>(current_node.get());

		string name = node->name;

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"VARIABLE_NODE name = " << name << endl;
		}

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_text" << endl;
		}
		current_node_copy->init_terminal_node_text(
				Tree,
				name,
				verbose_level);  // ToDo

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before current_node_copy->init_terminal_node_text" << endl;
		}

	}
	else if (current_node->type == irtree_node::node_type::UNARY_NEGATE_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion UNARY_NEGATE_NODE" << endl;
		}

		unary_negate_node *node = static_cast<unary_negate_node *>(current_node.get());

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion UNARY_NEGATE_NODE before handle_unary_negate_node" << endl;
		}

		handle_unary_negate_node(
				root,
				Tree,
				node,
				current_node_copy,
				verbose_level);

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion UNARY_NEGATE_NODE after handle_unary_negate_node" << endl;
		}



	}
	else if (current_node->type == irtree_node::node_type::EXPONENT_NODE) {

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion EXPONENT_NODE" << endl;
		}


		exponent_node *node = static_cast<exponent_node *>(current_node.get());


		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"before handle_exponent_node" << endl;
		}
		handle_exponent_node(
				root,
				Tree,
				node,
				current_node_copy,
				verbose_level);
		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"after handle_exponent_node" << endl;
		}


	}
	else {
		string s;

		node_type_as_string(current_node, s);
		cout << "unknown node type: " << s << endl;

		if (f_v) {
			cout << "expression_parser_sajeeb::convert_to_orbiter_recursion "
					"unknown node type " << s << endl;
		}

		exit(1);


	}


	if (f_v) {
		cout << "expression_parser_sajeeb::convert_to_orbiter_recursion done" << endl;
	}
}

static void node_type_as_string(shared_ptr<irtree_node> current_node, std::string &node_type_as_string)
{
	if (current_node->type == irtree_node::node_type::SENTINEL_NODE) {
		node_type_as_string = "SENTINEL_NODE";
	}
	else if (current_node->type == irtree_node::node_type::NUMBER_NODE) {
		node_type_as_string = "NUMBER_NODE";
	}
	else if (current_node->type == irtree_node::node_type::PARAMETER_NODE) {
		node_type_as_string = "PARAMETER_NODE";
	}
	else if (current_node->type == irtree_node::node_type::VARIABLE_NODE) {
		node_type_as_string = "VARIABLE_NODE";
	}
	else if (current_node->type == irtree_node::node_type::UNARY_NEGATE_NODE) {
		node_type_as_string = "UNARY_NEGATE_NODE";
	}
	else if (current_node->type == irtree_node::node_type::EXPONENT_NODE) {
		node_type_as_string = "EXPONENT_NODE";
	}
	else if (current_node->type == irtree_node::node_type::MULTIPLY_NODE) {
		node_type_as_string = "MULTIPLY_NODE";
	}
	else if (current_node->type == irtree_node::node_type::MINUS_NODE) {
		node_type_as_string = "MINUS_NODE";
	}
	else if (current_node->type == irtree_node::node_type::PLUS_NODE) {
		node_type_as_string = "PLUS_NODE";
	}
	else {
		node_type_as_string = "UNKNOWN";
	}

}

static void handle_unary_negate_node(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		unary_negate_node *node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "handle_unary_negate_node" << endl;
	}

	if (f_v) {
		cout << "handle_unary_negate_node "
				"before current_node_copy->init_empty_multiplication_node" << endl;
	}
	current_node_copy->init_empty_multiplication_node(
			Tree,
			verbose_level);

	if (f_v) {
		cout << "handle_unary_negate_node "
				"after current_node_copy->init_empty_multiplication_node" << endl;
	}


	int value_minus_one;

	value_minus_one = Tree->Fq->negate(1);

	if (f_v) {
		cout << "handle_unary_negate_node "
				"before current_node_copy->add_numerical_factor "
				"value_minus_one=" << value_minus_one << endl;
	}
	current_node_copy->add_numerical_factor(
			value_minus_one, verbose_level);
	if (f_v) {
		cout << "handle_unary_negate_node "
				"after current_node_copy->add_numerical_factor "
				"value_minus_one=" << value_minus_one << endl;
	}


	if (f_v) {
		cout << "handle_unary_negate_node "
				"before current_node_copy->add_empty_node" << endl;
	}
	current_node_copy->add_empty_node(Tree,
			verbose_level);
	if (f_v) {
		cout << "handle_unary_negate_node "
				"after current_node_copy->add_empty_node" << endl;
	}


	shared_ptr<irtree_node> sub_node;

	sub_node = node->children.front();

	if (f_v) {
		cout << "handle_unary_negate_node "
				"before convert_to_orbiter_recursion" << endl;
	}

	convert_to_orbiter_recursion(
			root,
			Tree,
			sub_node,
			current_node_copy->Nodes[current_node_copy->nb_nodes - 1],
			verbose_level);

	if (f_v) {
		cout << "handle_unary_negate_node "
				"after convert_to_orbiter_recursion" << endl;
	}

	if (f_v) {
		cout << "handle_unary_negate_node done" << endl;
	}
}

static void handle_exponent_node(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		exponent_node *node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "handle_exponent_node" << endl;
	}

	shared_ptr<irtree_node> child1;
	shared_ptr<irtree_node> child2;
	int cnt;

	list<shared_ptr<irtree_node>>::iterator it;
	cnt = 0;
	for (it = node->children.begin(); it != node->children.end(); ++it) {
		if (cnt == 0) {
			child1 = *it;
		}
		else if (cnt == 1) {
			child2 = *it;
		}
		else {
			cout << "handle_exponent_node EXPONENT_NODE too many children" << endl;
			exit(1);
		}
		cnt++;
	}

	string t1, t2;


	node_type_as_string(child1, t1);
	node_type_as_string(child2, t2);

	if (f_v) {
		cout << "handle_exponent_node type1=" << t1 << " type2=" << t2 << endl;
	}

	if (child1.get()->type == irtree_node::node_type::PARAMETER_NODE) {

		if (f_v) {
			cout << "handle_exponent_node PARAMETER_NODE" << endl;
		}

		parameter_node *node1 = static_cast<parameter_node *>(child1.get());


		std::string factor;

		factor.assign(node1->name);

		int exp = 0;


		exp = get_exponent(
				child2,
				verbose_level);




		if (f_v) {
			cout << "handle_exponent_node before "
					"current_node_copy->init_terminal_node_text_with_exponent "
					"factor=" << factor << " exponent=" << exp << endl;
		}
		current_node_copy->init_terminal_node_text_with_exponent(
						Tree, factor, exp /* exponent */, verbose_level);  // ToDo
		if (f_v) {
			cout << "handle_exponent_node after "
					"current_node_copy->init_terminal_node_text_with_exponent "
					"factor=" << factor << " exponent=" << exp << endl;
		}

	}
	else if (child1.get()->type == irtree_node::node_type::VARIABLE_NODE) {

		if (f_v) {
			cout << "handle_exponent_node VARIABLE_NODE" << endl;
		}

		variable_node *node1 = static_cast<variable_node *>(child1.get());


		std::string factor;

		factor.assign(node1->name);

		int exp = 0;


		exp = get_exponent(
				child2,
				verbose_level);




		if (f_v) {
			cout << "handle_exponent_node before "
					"current_node_copy->init_terminal_node_text_with_exponent "
					"factor=" << factor << " exponent=" << exp << endl;
		}
		current_node_copy->init_terminal_node_text_with_exponent(
						Tree, factor, exp /* exponent */, verbose_level);  // ToDo
		if (f_v) {
			cout << "handle_exponent_node after "
					"current_node_copy->init_terminal_node_text_with_exponent "
					"factor=" << factor << " exponent=" << exp << endl;
		}


	}
	else if (child1.get()->type == irtree_node::node_type::PLUS_NODE) {

		plus_node *node1 = static_cast<plus_node *>(child1.get());


		int exponent = 0;

		exponent = get_exponent(
				child2,
				verbose_level);




		if (f_v) {
			cout << "handle_exponent_node before "
					"current_node_copy->init_empty_plus_node_with_exponent" << endl;
		}
		current_node_copy->init_empty_plus_node_with_exponent(
				Tree, exponent /* exponent */, verbose_level);

		if (f_v) {
			cout << "handle_exponent_node after "
					"current_node_copy->init_empty_plus_node_with_exponent" << endl;
		}

		if (f_v) {
			cout << "handle_exponent_node PLUS_NODE before collect_summands" << endl;
		}
		collect_summands(root, Tree, node1,
				current_node_copy,
				verbose_level);
		if (f_v) {
			cout << "handle_exponent_node PLUS_NODE after collect_summands" << endl;
			cout << "handle_exponent_node nb_nodes=" << current_node_copy->nb_nodes << endl;
		}


	}
	else if (child1.get()->type == irtree_node::node_type::MULTIPLY_NODE) {

		multiply_node *node1 = static_cast<multiply_node *>(child1.get());


		int exp = 0;



		exp = get_exponent(
				child2,
				verbose_level);




		if (f_v) {
			cout << "handle_exponent_node before "
					"current_node_copy->init_empty_multiplication_node" << endl;
		}
		current_node_copy->init_empty_multiplication_node(
				Tree,
				verbose_level);

		if (f_v) {
			cout << "handle_exponent_node after "
					"current_node_copy->init_empty_multiplication_node" << endl;
		}

		if (exp != 1) {
			current_node_copy->f_has_exponent = true;
			current_node_copy->exponent = exp;
		}


		if (f_v) {
			cout << "handle_exponent_node MULTIPLY_NODE before collect_factors" << endl;
		}
		collect_factors(root, Tree, node1,
				current_node_copy,
				verbose_level);
		if (f_v) {
			cout << "handle_exponent_node PLUS_NODE after collect_summands" << endl;
			cout << "handle_exponent_node nb_nodes=" << current_node_copy->nb_nodes << endl;
		}


	}
	else {
		cout << " do not know how to handle this node" << endl;
		string s;

		node_type_as_string(child1, s);
		cout << "unknown node child1 of type: " << s << endl;

		node_type_as_string(child2, s);
		cout << "unknown node child2 of type: " << s << endl;
		exit(1);
	}

	if (f_v) {
		cout << "handle_exponent_node done" << endl;
	}
}


static int get_exponent(
		shared_ptr<irtree_node> child2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "get_exponent" << endl;
	}
	int exp = 0;

	if (child2.get()->type == irtree_node::node_type::NUMBER_NODE) {
		number_node *node2 = static_cast<number_node *>(child2.get());

		exp = node2->value;
	}
	else if (child2.get()->type == irtree_node::node_type::UNARY_NEGATE_NODE) {


		unary_negate_node *node = static_cast<unary_negate_node *>(child2.get());

		shared_ptr<irtree_node> child1;
		int cnt;

		list<shared_ptr<irtree_node>>::iterator it;
		cnt = 0;
		for (it = node->children.begin(); it != node->children.end(); ++it) {
			if (cnt == 0) {
				child1 = *it;
			}
			else {
				cout << "get_exponent too many children after unary minus" << endl;
				exit(1);
			}
			cnt++;
		}

		if (child1.get()->type == irtree_node::node_type::NUMBER_NODE) {
			number_node *node1 = static_cast<number_node *>(child1.get());

			exp = - node1->value;
		}
		else {
			cout << "handle_exponent_node unknown type of exponent node" << endl;
			string s;
			node_type_as_string(child2, s);
			cout << "unknown node child2 of type: " << s << endl;
			exit(1);
		}


	}
	else {

		cout << "handle_exponent_node unknown type of exponent node" << endl;
		string s;
		node_type_as_string(child2, s);
		cout << "unknown node child2 of type: " << s << endl;
		exit(1);
	}
	if (f_v) {
		cout << "get_exponent done, exp = " << exp << endl;
	}
	return exp;
}

static void collect_factors(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		non_terminal_node *current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "collect_factors" << endl;
	}

	int nb_children = 0;
	for (const shared_ptr<irtree_node>& child : current_node->children) {
		nb_children++;
	}

	current_node_copy->f_terminal = false;
	int child_counter = 0;
	for (const shared_ptr<irtree_node>& child : current_node->children) {

		if (child.get()->type == irtree_node::node_type::SENTINEL_NODE) {
			cout << "collect_factors sentinel node is not allowed" << endl;
			exit(1);
		}
		else if (child.get()->type == irtree_node::node_type::NUMBER_NODE) {
			if (f_v) {
				cout << "collect_factors NUMBER_NODE" << endl;
			}
			number_node *node = static_cast<number_node *>(child.get());
			int value;

			value = node->value;

			if (f_v) {
				cout << "collect_factors before "
						"current_node_copy->add_numerical_factor value=" << value << endl;
			}
			current_node_copy->add_numerical_factor(
					value, verbose_level);
			if (f_v) {
				cout << "collect_factors after "
						"current_node_copy->add_numerical_factor value=" << value << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}

		}
		else if (child.get()->type == irtree_node::node_type::PARAMETER_NODE) {
			if (f_v) {
				cout << "collect_factors PARAMETER_NODE" << endl;
			}

			parameter_node *node = static_cast<parameter_node *>(child.get());

			std::string factor;

			factor.assign(node->name);

			if (f_v) {
				cout << "collect_factors before "
						"current_node_copy->add_factor factor=" << factor << endl;
			}
			current_node_copy->add_factor(
					factor, 1 /* exponent */, verbose_level); // ToDo

			if (f_v) {
				cout << "collect_factors after "
						"current_node_copy->add_factor factor=" << factor << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::VARIABLE_NODE) {
			if (f_v) {
				cout << "collect_factors VARIABLE_NODE" << endl;
			}

			variable_node *node = static_cast<variable_node *>(child.get());

			std::string factor;

			factor.assign(node->name);

			if (f_v) {
				cout << "collect_factors before "
						"current_node_copy->add_factor factor=" << factor << endl;
			}
			current_node_copy->add_factor(
					factor, 1 /* exponent */, verbose_level); // ToDo

			if (f_v) {
				cout << "collect_factors after "
						"current_node_copy->add_factor factor=" << factor << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::UNARY_NEGATE_NODE) {
			if (f_v) {
				cout << "collect_factors UNARY_NEGATE_NODE" << endl;
			}


			unary_negate_node *node = static_cast<unary_negate_node *>(child.get());

			if (f_v) {
				cout << "collect_factors UNARY_NEGATE_NODE before "
						"current_node_copy->add_empty_node" << endl;
			}
			current_node_copy->add_empty_node(
					Tree, verbose_level);

			if (f_v) {
				cout << "collect_factors UNARY_NEGATE_NODE after "
						"current_node_copy->add_empty_node" << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}

			expression_parser::syntax_tree_node *fresh_node;

			fresh_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];

			if (f_v) {
				cout << "collect_factors UNARY_NEGATE_NODE before handle_unary_negate_node" << endl;
			}

			handle_unary_negate_node(
					root,
					Tree,
					node,
					fresh_node,
					verbose_level);

			if (f_v) {
				cout << "collect_factors UNARY_NEGATE_NODE after handle_unary_negate_node" << endl;
			}

		}
		else if (child.get()->type == irtree_node::node_type::EXPONENT_NODE) {
			if (f_v) {
				cout << "collect_factors EXPONENT_NODE" << endl;
			}
			exponent_node *node = static_cast<exponent_node *>(child.get());


			if (f_v) {
				cout << "collect_factors EXPONENT_NODE "
						"before current_node_copy->add_empty_node" << endl;
			}
			current_node_copy->add_empty_node(Tree,
					verbose_level);
			if (f_v) {
				cout << "collect_factors EXPONENT_NODE "
						"after current_node_copy->add_empty_node" << endl;
			}


			expression_parser::syntax_tree_node *fresh_node;

			fresh_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];

			if (f_v) {
				cout << "collect_factors EXPONENT_NODE "
						"before handle_exponent_node" << endl;
			}
			handle_exponent_node(
					root,
					Tree,
					node,
					fresh_node,
					verbose_level);
			if (f_v) {
				cout << "collect_factors EXPONENT_NODE "
						"after handle_exponent_node" << endl;
			}

		}
		else if (child.get()->type == irtree_node::node_type::MULTIPLY_NODE) {

			if (f_v) {
				cout << "collect_factors MULTIPLY_NODE" << endl;
			}
			multiply_node *node = static_cast<multiply_node *>(child.get());


			if (f_v) {
				cout << "collect_factors MULTIPLY_NODE before collect_factors" << endl;
			}
			collect_factors(root, Tree, node, current_node_copy, verbose_level);
			if (f_v) {
				cout << "collect_factors MULTIPLY_NODE after collect_factors" << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::MINUS_NODE) {
			cout << "collect_factors MINUS_NODE" << endl;
			cout << " do not know how to handle this node" << endl;
			exit(1);
		}
		else if (child.get()->type == irtree_node::node_type::PLUS_NODE) {
			if (f_v) {
				cout << "collect_factors PLUS_NODE" << endl;
			}
			plus_node *node = static_cast<plus_node *>(child.get());

			if (f_v) {
				cout << "collect_factors before "
						"current_node_copy->add_empty_plus_node_with_exponent" << endl;
			}
			current_node_copy->add_empty_plus_node_with_exponent(
					Tree, 1 /* exponent */, verbose_level);

			if (f_v) {
				cout << "collect_factors after "
						"current_node_copy->add_empty_plus_node_with_exponent" << endl;
			}


			expression_parser::syntax_tree_node *fresh_node;

			fresh_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];


			if (f_v) {
				cout << "collect_factors PLUS_NODE before collect_summands" << endl;
			}
			collect_summands(root, Tree, node,
					fresh_node,
					verbose_level);
			if (f_v) {
				cout << "collect_factors PLUS_NODE after collect_summands" << endl;
				cout << "collect_factors nb_nodes=" << current_node_copy->nb_nodes << endl;
			}
		}
		else {
			cout << "collect_factors unknown node type" << endl;
			cout << " do not know how to handle this node" << endl;
			string s;

			node_type_as_string(child, s);
			cout << "unknown node child of type: " << s << endl;
			exit(1);
		}

		child_counter++;

	}

	if (f_v) {
		cout << "collect_factors finished" << endl;
	}
}


static void collect_summands(
		expression_parser_sajeeb *root,
		expression_parser::syntax_tree *Tree,
		plus_node *current_node,
		expression_parser::syntax_tree_node *current_node_copy,
		int verbose_level)
// current_node_copy is addition node
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "collect_summands" << endl;
	}


	int nb_children = 0;

	for (const shared_ptr<irtree_node>& child : current_node->children) {
		nb_children++;
	}

	current_node_copy->f_terminal = false;
	int child_counter = 0;
	for (const shared_ptr<irtree_node>& child : current_node->children) {

		if (f_v) {
			cout << "collect_summands child_counter=" << child_counter << endl;
		}
		if (child.get()->type == irtree_node::node_type::SENTINEL_NODE) {
			cout << "collect_summands sentinel node is not allowed" << endl;
			exit(1);
		}
		else if (child.get()->type == irtree_node::node_type::NUMBER_NODE) {
			if (f_v) {
				cout << "collect_summands NUMBER_NODE" << endl;
			}
			number_node *node = static_cast<number_node *>(child.get());
			int value;

			value = node->value;

			if (f_v) {
				cout << "collect_summands before "
						"current_node_copy->add_numerical_summand "
						"value=" << value << endl;
			}
			current_node_copy->add_numerical_summand(
					value, verbose_level);
			if (f_v) {
				cout << "collect_summands after "
						"current_node_copy->add_numerical_summand "
						"value=" << value << endl;
			}

		}
		else if (child.get()->type == irtree_node::node_type::PARAMETER_NODE) {
			if (f_v) {
				cout << "collect_summands PARAMETER_NODE" << endl;
			}

			parameter_node *node = static_cast<parameter_node *>(child.get());

			std::string summand;

			summand.assign(node->name);

			if (f_v) {
				cout << "collect_summands "
						"before current_node_copy->add_summand "
						"summand=" << summand << endl;
			}
			current_node_copy->add_summand(
					summand, verbose_level); // ToDo

			if (f_v) {
				cout << "collect_summands "
						"after current_node_copy->add_factor "
						"summand=" << summand << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::VARIABLE_NODE) {
			if (f_v) {
				cout << "collect_summands VARIABLE_NODE" << endl;
			}

			variable_node *node = static_cast<variable_node *>(child.get());

			std::string summand;

			summand.assign(node->name);

			if (f_v) {
				cout << "collect_summands "
						"before current_node_copy->add_summand "
						"summand=" << summand << endl;
			}
			current_node_copy->add_summand(
					summand, verbose_level); // ToDo

			if (f_v) {
				cout << "collect_summands "
						"after current_node_copy->add_summand "
						"summand=" << summand << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::MULTIPLY_NODE) {

			if (f_v) {
				cout << "collect_summands MULTIPLY_NODE" << endl;
			}
			multiply_node *node = static_cast<multiply_node *>(child.get());


			if (f_v) {
				cout << "collect_summands before "
						"current_node_copy->add_empty_multiplication_node" << endl;
			}
			current_node_copy->add_empty_multiplication_node(Tree,
					verbose_level);

			if (f_v) {
				cout << "collect_summands after "
						"current_node_copy->add_empty_multiplication_node" << endl;
			}

			expression_parser::syntax_tree_node *artificial_mult_node;


			artificial_mult_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];

			if (f_v) {
				cout << "collect_summands MULTIPLY_NODE "
						"before collect_factors" << endl;
			}
			collect_factors(root, Tree, node,
					artificial_mult_node,
					verbose_level);
			if (f_v) {
				cout << "collect_summands MULTIPLY_NODE "
						"after collect_factors" << endl;
			}

		}
		else if (child.get()->type == irtree_node::node_type::MINUS_NODE) {
			cout << "collect_summands MINUS_NODE" << endl;
			cout << " do not know how to handle this node" << endl;
			exit(1);
		}
		else if (child.get()->type == irtree_node::node_type::PLUS_NODE) {
			if (f_v) {
				cout << "collect_summands PLUS_NODE" << endl;
			}
			plus_node *node = static_cast<plus_node *>(child.get());


			if (f_v) {
				cout << "collect_summands PLUS_NODE "
						"before collect_summands" << endl;
			}
			collect_summands(root, Tree, node, current_node_copy, verbose_level);
			if (f_v) {
				cout << "collect_summands PLUS_NODE "
						"after collect_summands" << endl;
			}
		}
		else if (child.get()->type == irtree_node::node_type::UNARY_NEGATE_NODE) {
			if (f_v) {
				cout << "collect_summands UNARY_NEGATE_NODE" << endl;
			}

			unary_negate_node *node = static_cast<unary_negate_node *>(child.get());

			if (f_v) {
				cout << "collect_summands UNARY_NEGATE_NODE "
						"before current_node_copy->add_empty_node" << endl;
			}
			current_node_copy->add_empty_node(Tree,
					verbose_level);
			if (f_v) {
				cout << "collect_summands UNARY_NEGATE_NODE "
						"after current_node_copy->add_empty_node" << endl;
			}


			expression_parser::syntax_tree_node *fresh_node;

			fresh_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];


			if (f_v) {
				cout << "collect_summands UNARY_NEGATE_NODE "
						"before handle_unary_negate_node" << endl;
			}

			handle_unary_negate_node(
					root,
					Tree,
					node,
					fresh_node,
					verbose_level);

			if (f_v) {
				cout << "collect_summands UNARY_NEGATE_NODE "
						"after handle_unary_negate_node" << endl;
			}




		}

		else if (child.get()->type == irtree_node::node_type::EXPONENT_NODE) {
			if (f_v) {
				cout << "collect_summands EXPONENT_NODE" << endl;
			}
			exponent_node *node = static_cast<exponent_node *>(child.get());


			if (f_v) {
				cout << "collect_summands EXPONENT_NODE "
						"before current_node_copy->add_empty_node" << endl;
			}
			current_node_copy->add_empty_node(Tree,
					verbose_level);
			if (f_v) {
				cout << "collect_summands EXPONENT_NODE "
						"after current_node_copy->add_empty_node" << endl;
			}


			expression_parser::syntax_tree_node *fresh_node;

			fresh_node = current_node_copy->Nodes[current_node_copy->nb_nodes - 1];

			if (f_v) {
				cout << "collect_summands EXPONENT_NODE "
						"before handle_exponent_node" << endl;
			}
			handle_exponent_node(
					root,
					Tree,
					node,
					fresh_node,
					verbose_level);
			if (f_v) {
				cout << "collect_summands EXPONENT_NODE "
						"after handle_exponent_node" << endl;
			}

		}

		else {
			cout << "collect_summands do not know how to handle this node" << endl;
			string s;

			node_type_as_string(child, s);
			cout << "unknown node child of type: " << s << endl;
			exit(1);
		}

		child_counter++;
		if (f_v) {
			cout << "collect_summands child_counter = " << child_counter << endl;
		}


	}

	if (f_v) {
		cout << "collect_summands finished" << endl;
	}
}





}}}



