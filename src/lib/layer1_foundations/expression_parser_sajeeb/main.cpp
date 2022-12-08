
/**
* Author:    Sajeeb Roy Chowdhury
* Created:   09.22.2022
* Email:     sajeeb.roy.chow@gmail.com
*
**/
    
#include <iostream>
#include <unordered_map>
#include <string>

#include "parser.tab.hpp"
#include "lexer.yy.h"

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

#include "orbiter.h"

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

    IRTreeVoidReturnTypeVisitorInterface* operator()() {
        if (latex_output_stream.is_open()) latex_output_stream.close();
        latex_output_stream.open(directory + "stage" + std::to_string(stage_counter++) + ".tex");

        strategy_wrapper.set_output_stream(latex_output_stream);
        return strategy_wrapper.get_visitor();
    }
};

void remove_minus_nodes(shared_ptr<irtree_node>& root) {
    static remove_minus_nodes_visitor remove_minus_nodes;
    root->accept(&remove_minus_nodes);
}

void merge_redundant_nodes(shared_ptr<irtree_node>& root) {
    static merge_nodes_visitor merge_redundant_nodes;
    root->accept(&merge_redundant_nodes);
}

shared_ptr<irtree_node> generate_abstract_syntax_tree(std::string& exp, managed_variables_index_table managed_variables_table) {
    shared_ptr<irtree_node> ir_tree_root;
    YY_BUFFER_STATE buffer = yy_scan_string( exp.c_str() );
    yy_switch_to_buffer(buffer);
    int result = yyparse(ir_tree_root, managed_variables_table);
    yy_delete_buffer(buffer);
    yylex_destroy();
    return ir_tree_root;
}

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
    std::string exp = "X0*X8^7 + X8^8 + X1*X8^7 + X2*X8^7 + X3*X8^7 + X4*X8^7 + X5*X8^7 + X6*X8^7 + X7*X8^7 + X0*X1*X8^6 + X0*X2*X8^6 + X0*X3*X8^6 + X0*X4*X8^6 + X0*X5*X8^6 + X0*X6*X8^6 + X0*X7*X8^6 + X1*X2*X8^6 + X1*X3*X8^6 + X1*X4*X8^6 + X1*X5*X8^6 + X1*X6*X8^6 + X1*X7*X8^6 + X2*X3*X8^6 + X2*X4*X8^6 + X2*X5*X8^6 + X2*X6*X8^6 + X2*X7*X8^6 + X3*X4*X8^6 + X3*X5*X8^6 + X3*X6*X8^6 + X3*X7*X8^6 + X4*X5*X8^6 + X4*X6*X8^6 + X4*X7*X8^6 + X5*X6*X8^6 + X5*X7*X8^6 + X6*X7*X8^6 + X0*X1*X2*X8^5 + X0*X1*X3*X8^5 + X0*X1*X4*X8^5 + X0*X1*X5*X8^5 + X0*X1*X6*X8^5 + X0*X1*X7*X8^5 + X0*X2*X3*X8^5 + X0*X2*X4*X8^5 + X0*X2*X5*X8^5 + X0*X2*X6*X8^5 + X0*X2*X7*X8^5 + X0*X3*X4*X8^5 + X0*X3*X5*X8^5 + X0*X3*X6*X8^5 + X0*X3*X7*X8^5 + X0*X4*X5*X8^5 + X0*X4*X6*X8^5 + X0*X4*X7*X8^5 + X0*X5*X6*X8^5 + X0*X5*X7*X8^5 + X0*X6*X7*X8^5 + X1*X2*X3*X8^5 + X1*X2*X4*X8^5 + X1*X2*X5*X8^5 + X1*X2*X6*X8^5 + X1*X2*X7*X8^5 + X1*X3*X4*X8^5 + X1*X3*X5*X8^5 + X1*X3*X6*X8^5 + X1*X3*X7*X8^5 + X1*X4*X5*X8^5 + X1*X4*X6*X8^5 + X1*X4*X7*X8^5 + X1*X5*X6*X8^5 + X1*X5*X7*X8^5 + X1*X6*X7*X8^5 + X2*X3*X4*X8^5 + X2*X3*X5*X8^5 + X2*X3*X6*X8^5 + X2*X3*X7*X8^5 + X2*X4*X5*X8^5 + X2*X4*X6*X8^5 + X2*X4*X7*X8^5 + X2*X5*X6*X8^5 + X2*X5*X7*X8^5 + X2*X6*X7*X8^5 + X3*X4*X5*X8^5 + X3*X4*X6*X8^5 + X3*X4*X7*X8^5 + X3*X5*X6*X8^5 + X3*X5*X7*X8^5 + X3*X6*X7*X8^5 + X4*X5*X6*X8^5 + X4*X5*X7*X8^5 + X4*X6*X7*X8^5 + X5*X6*X7*X8^5 + X0*X1*X2*X4*X8^4 + X0*X1*X2*X5*X8^4 + X0*X1*X2*X6*X8^4 + X0*X1*X2*X7*X8^4 + X0*X1*X3*X4*X8^4 + X0*X1*X3*X5*X8^4 + X0*X1*X3*X6*X8^4 + X0*X1*X3*X7*X8^4 + X0*X1*X4*X6*X8^4 + X0*X1*X4*X7*X8^4 + X0*X1*X5*X6*X8^4 + X0*X1*X5*X7*X8^4 + X0*X2*X3*X4*X8^4 + X0*X2*X3*X5*X8^4 + X0*X2*X3*X6*X8^4 + X0*X2*X3*X7*X8^4 + X0*X2*X4*X5*X8^4 + X0*X2*X4*X7*X8^4 + X0*X2*X5*X6*X8^4 + X0*X2*X6*X7*X8^4 + X0*X3*X4*X5*X8^4 + X0*X3*X4*X6*X8^4 + X0*X3*X5*X7*X8^4 + X0*X3*X6*X7*X8^4 + X0*X4*X5*X6*X8^4 + X0*X4*X5*X7*X8^4 + X0*X4*X6*X7*X8^4 + X0*X5*X6*X7*X8^4 + X1*X2*X3*X4*X8^4 + X1*X2*X3*X5*X8^4 + X1*X2*X3*X6*X8^4 + X1*X2*X3*X7*X8^4 + X1*X2*X4*X5*X8^4 + X1*X2*X4*X6*X8^4 + X1*X2*X5*X7*X8^4 + X1*X2*X6*X7*X8^4 + X1*X3*X4*X5*X8^4 + X1*X3*X4*X7*X8^4 + X1*X3*X5*X6*X8^4 + X1*X3*X6*X7*X8^4 + X1*X4*X5*X6*X8^4 + X1*X4*X5*X7*X8^4 + X1*X4*X6*X7*X8^4 + X1*X5*X6*X7*X8^4 + X2*X3*X4*X6*X8^4 + X2*X3*X4*X7*X8^4 + X2*X3*X5*X6*X8^4 + X2*X3*X5*X7*X8^4 + X2*X4*X5*X6*X8^4 + X2*X4*X5*X7*X8^4 + X2*X4*X6*X7*X8^4 + X2*X5*X6*X7*X8^4 + X3*X4*X5*X6*X8^4 + X3*X4*X5*X7*X8^4 + X3*X4*X6*X7*X8^4 + X3*X5*X6*X7*X8^4";


    // (a+b)(c-d) = a*b - a*d + b*c - b*d

    //
    LOG("");
    managed_variables_index_table managed_variables_table;
    for (int i=0; i<9; ++i)
        managed_variables_table.insert("X"+std::to_string(i));
    cout << "managed_variables_table:\n" << managed_variables_table << endl;


    LOG("");
    shared_ptr<irtree_node> ir_tree_root = generate_abstract_syntax_tree(exp, managed_variables_table);


    LOG("");
    get_latex_staged_visitor_functor
        get_latex_staged_visitor("visitor_result/",
                                 ir_tree_latex_visitor_strategy::type::SIMPLE_TREE);


    // print the AST
//    LOG(ir_tree_root.get());
//    ir_tree_root->accept(get_latex_staged_visitor());
//    LOG("");

    //
//    simplify_numerical_visitor simplify;
//    ir_tree_root->accept(&simplify);
//    ir_tree_root->accept(get_latex_staged_visitor());


    LOG("");
    // remove minus nodes
    remove_minus_nodes(ir_tree_root);
    ir_tree_root->accept(get_latex_staged_visitor());

    LOG("");
   // merge redundant nodes
    merge_redundant_nodes(ir_tree_root);
    ir_tree_root->accept(get_latex_staged_visitor());

    LOG("");
   // distribute and reduce unary minus nodes
     uminus_distribute_and_reduce_visitor distribute_uminus_visitor;
     LOG("");
     ir_tree_root->accept(&distribute_uminus_visitor);
     LOG("");
     ir_tree_root->accept(get_latex_staged_visitor());

     LOG("");
     // merge redundant nodes
     merge_redundant_nodes(ir_tree_root);
     ir_tree_root->accept(get_latex_staged_visitor());

    //
    // multiplication_expansion_visitor mev;
    // ir_tree_root->accept(&mev);
    // ir_tree_root->accept(get_latex_staged_visitor());

    //
//    deep_copy_visitor deepCopyVisitor;
//    shared_ptr<irtree_node> ir_tree_root_cpy = ir_tree_root->accept(&deepCopyVisitor);
//    ir_tree_root_cpy->accept(get_latex_staged_visitor());

     LOG("");
   //
    exponent_vector_visitor evv;
    ir_tree_root->accept(evv(managed_variables_table));
    eval_visitor evalVisitor;
    orbiter::layer5_applications::user_interface::orbiter_top_level_session Top_level_session;
   orbiter::layer5_applications::user_interface::The_Orbiter_top_level_session = &Top_level_session;

    std::string *Argv;
    data_structures::string_tools ST;
    LOG("");
   ST.convert_arguments(argc, argv, Argv);
    // argc has changed!
    cout << "after ST.convert_arguments, argc=" << argc << endl;
    cout << "before Top_level_session.startup_and_read_arguments" << endl;
    static_cast<void>(Top_level_session.startup_and_read_arguments(argc, Argv, 1));
    orbiter::layer1_foundations::field_theory::finite_field_description Descr;
    LOG("");
    orbiter::layer1_foundations::field_theory::finite_field Fq;
    Descr.f_q = TRUE;
    Descr.q = 2;
    Fq.init(&Descr, 1);
    LOG("");
    unordered_map<string, int> assignemnt = {
            {"a", 4},
            {"b", 2},
            {"c", 2},
            {"d", 4}
    };
    for (auto& it : evv.monomial_coefficient_table_) {
        const vector<unsigned int>& vec = it.first;
        vector<irtree_node*> root_nodes = it.second;
        int val = 0;
        for (auto& node : root_nodes) val += node->accept(&evalVisitor, &Fq, assignemnt);
        std::cout << val << ":  [";
        for (const auto& itit : vec) std::cout << itit << " ";
        std::cout << "]" << std::endl;
    }
   LOG("");
   ir_tree_root->accept(get_latex_staged_visitor());

    // print string representation of the IR tree
    ir_tree_to_string_visitor to_string_visitor;
    ir_tree_root->accept(&to_string_visitor);
    cout << "in:  " << exp << endl;
    cout << "out: " << to_string_visitor.get_string_representation() << endl;

}
