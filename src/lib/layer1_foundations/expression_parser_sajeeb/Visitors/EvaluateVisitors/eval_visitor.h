/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include "../IRTreeVisitor.h"
#include "../../../foundations.h"
#include <unordered_map>
#include <string>

#ifndef EVAL_VISITOR_H
#define EVAL_VISITOR_H

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

using finite_field = orbiter::layer1_foundations::field_theory::finite_field;
using std::unordered_map;
using std::string;

class eval_visitor final :
        public IRTreeTemplateReturnTypeVariadicArgumentConstantVisitorInterface<int,
                                                                                finite_field*,
                                                                                unordered_map<string, int>&> {
public:
    
    int visit(const plus_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const minus_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const multiply_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const exponent_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const unary_negate_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const variable_node* num_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const parameter_node* node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const number_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
    int visit(const sentinel_node* op_node, finite_field* Fq, unordered_map<string, int>& assignment_table) override;
};

#endif // EVAL_VISITOR_H
