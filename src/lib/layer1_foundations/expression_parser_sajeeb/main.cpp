
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
// This always needs to be included    
#include "parser.h"

#include <iostream>
#include <unordered_map>
#include <string>

// This only needs to be included if the tree is to be visited
#include "Visitors/dispatcher.h"

// Provided visitors
#include "Visitors/PrintVisitors/ir_tree_pretty_print_visitor.h"
#include "Visitors/uminus_distribute_and_reduce_visitor.h"
#include "Visitors/merge_nodes_visitor.h"
#include "Visitors/LatexVisitors/ir_tree_latex_visitor_strategy.h"
#include "Visitors/ToStringVisitors/ir_tree_to_string_visitor.h"
#include "Visitors/remove_minus_nodes_visitor.h"
#include "Visitors/ExpansionVisitors/multiplication_expansion_visitor.h"
#include "Visitors/CopyVisitors/deep_copy_visitor.h"
#include "Visitors/exponent_vector_visitor.h"
#include "Visitors/ReductionVisitors/simplify_numerical_visitor.h"
#include "Visitors/EvaluateVisitors/eval_visitor.h"

#include "orbiter.h"
//#include "layer1_foundations/foundations.h"

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

using std::shared_ptr;

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


int main(int argc, const char** argv) {
	// std::string exp = "a-(-b)^(c*j*i-d*-9*-(-1+7))*e+f+g"; //a + --b^(c+d)*e + f + g
//	 std::string exp = "a*b*c*-(d*-(e*-f*-g)*-i) * -k^-(d*-(e*-f*-g)*-i) -x -y -x -x -z + a+b+c-(d-e-(f+g))";
//    std::string exp = "-(-(-a+-b) + -(c+d))"; // -a + -b + c + d
//    std::string exp = "a-b-c-d";
//	 std::string exp = "-(a*-b*-c)";
//	 std::string exp = "-(a*-b*--k^(c+d))"; // -a * b * -k^(c+d)
//    std::string exp = "(a+k^i) * (c+d) * (e+f) * (g+h) * (1+1+1) * (1*2*1)";
//    std::string exp = "1+2+-(3*-2)+(4*5*6*2^3^2)";
//    std::string exp = "(a*b*c)*x0*x2 + 1*x1 + 1*x2 + x0^2*x1 + x1^2*x2 + x2*x0*x1 + a^2*x2^2";

    std::string exp = "-(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*(b - d)*X0^2*X2 \
+ (a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*(a + b - c - d)*X0*X1*X2 \
+ (a^2*c - a^2*d - a*c^2 + b*c^2 + a*d - b*c)*(b - d)*X0*X1*X3 \
- (a*d - b*c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X0*X2^2 \
- (a^2*c*d - a*b*c^2 - a^2*d + a*b*d + b*c^2 - b*c*d)*(b - d)*X0*X2*X3 \
- (a - c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1^2*X2 \
- (a - c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1^2*X3 \
+ (a*d - b*c)*(a*b*c - a*b*d - a*c*d + b*c*d + a*d - b*c)*X1*X2^2 \
+ ((1+1)*a^2*b*c*d - a^2*b*d^2 - (1+1)*a^2*c*d^2 \
- (1+1)*a*b^2*c^2 + a*b^2*c*d + (1+1)*a*b*c^2*d + a*b*c*d^2 \
- b^2*c^2*d - a^2*b*c + a^2*c*d + a^2*d^2 + a*b^2*c + a*b*c^2 \
- (1+1+1+1)*a*b*c*d - a*c^2*d + a*c*d^2 + b^2*c^2)*X1*X2*X3 \
+ c*a*(a*d - b*c - a + b + c - d)*(b - d)*X1*X3^2";


    managed_variables_index_table managed_variables_table;
    for (int i=0; i<4; ++i)
        managed_variables_table.insert("X"+std::to_string(i));
    cout << "managed_variables_table:\n" << managed_variables_table << endl;


    shared_ptr<irtree_node> ir_tree_root = parser::parse_expression(exp, managed_variables_table);

    get_latex_staged_visitor_functor
        get_latex_staged_visitor("visitor_result/",
                                 ir_tree_latex_visitor_strategy::type::SIMPLE_TREE);


    // print the AST
    dispatcher::visit(ir_tree_root, get_latex_staged_visitor());

    //
//    simplify_numerical_visitor simplify;
//    dispatcher::visit(ir_tree_root, &simplify);
//    dispatcher::visit(ir_tree_root, get_latex_staged_visitor());


    // remove minus nodes
    dispatcher::visit(ir_tree_root, remove_minus_nodes_visitor());
    dispatcher::visit(ir_tree_root, merge_nodes_visitor());
    dispatcher::visit(ir_tree_root, get_latex_staged_visitor());

   // distribute and reduce unary minus nodes
    dispatcher::visit(ir_tree_root, uminus_distribute_and_reduce_visitor());
    dispatcher::visit(ir_tree_root, merge_nodes_visitor());
    dispatcher::visit(ir_tree_root, get_latex_staged_visitor());

    //
    // multiplication_expansion_visitor mev;
    // dispatcher::visit(ir_tree_root, &mev);
    // dispatcher::visit(ir_tree_root, get_latex_staged_visitor());

    //
//    deep_copy_visitor deepCopyVisitor;
//    shared_ptr<irtree_node> ir_tree_root_cpy = dispatcher::visit(ir_tree_root, &deepCopyVisitor);
//    ir_tree_root_cpy->accept(get_latex_staged_visitor());

   //
    exponent_vector_visitor evv;
    dispatcher::visit(ir_tree_root, evv(managed_variables_table));
    dispatcher::visit(ir_tree_root, get_latex_staged_visitor());
    eval_visitor evalVisitor;



    // orbiter stuff
    orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
    orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;

    //std::string *Argv;
    //data_structures::string_tools ST;
    //ST.convert_arguments(argc, argv, Argv);
    // argc has changed!
    cout << "after ST.convert_arguments, argc=" << argc << endl;
    cout << "before Top_level_session.startup_and_read_arguments" << endl;
    //static_cast<void>(Top_level_session.startup_and_read_arguments(argc, Argv, 1));

    {
		orbiter::layer1_foundations::field_theory::finite_field_description Descr;
		orbiter::layer1_foundations::field_theory::finite_field Fq;
		Descr.f_q = TRUE;
		Descr.q_text.assign("5");
		;

		LOG("before Fq.init");
		Fq.init(&Descr, 5);
		LOG("after Fq.init");
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
	   dispatcher::visit(ir_tree_root, get_latex_staged_visitor());




		// print string representation of the IR tree
		ir_tree_to_string_visitor to_string_visitor;
		dispatcher::visit(ir_tree_root, to_string_visitor);
		cout << "in:  " << exp << endl;
		cout << "out: " << to_string_visitor.get_string_representation() << endl;
	    LOG("before deleting orbiter objects")
    }

    LOG("after deleting orbiter objects")
}
