/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include <vector>
#include "IRTreeVisitor.h"
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include "IRTreeChildLinkArgumentVisitor.h"
#include "../exception.h"
#include "../../foundations.h"

#ifndef EXPONENT_VECTOR_VISITOR_H
#define EXPONENT_VECTOR_VISITOR_H

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

using std::unordered_set;
using std::vector;
using std::unordered_map;
using std::string;

class managed_variables_index_table : public unordered_map<string, size_t> {
    typedef unordered_map<string, size_t> base_t;

public:
    template<class T>
    void insert(const T& name) {
        base_t::insert({name, size()});
    }

    template<class T, class... Args>
    void insert(const T& name, const Args&... args) {
        base_t::insert({name, size()});
        insert(args...);
    }

    size_t index(const string& key) const {
        return base_t::at(key);
    }
};
std::ostream& operator<< (std::ostream& os, const managed_variables_index_table& obj);

struct monomial_coefficient_table_hash_function final {
    using algorithms = orbiter::layer1_foundations::data_structures::algorithms;
    uint32_t operator()(const vector<unsigned int>& exponent_vector) const {
        static algorithms algo_instance;
        return algo_instance.SuperFastHash_uint(exponent_vector.data(), exponent_vector.size());
    }
};
struct monomial_coefficient_table_key_equal_function final {
    bool operator()(const vector<unsigned int>& a, const vector<unsigned int>& b) const {
        using sorting = orbiter::layer1_foundations::data_structures::sorting;
        static sorting keyEqualFn;
        return !keyEqualFn.integer_vec_std_compare(a, b);
    }
};
class monomial_coefficient_table final : public unordered_map<vector<unsigned int>,
                                                              vector<irtree_node*>,
                                                              monomial_coefficient_table_hash_function,
                                                              monomial_coefficient_table_key_equal_function> {};


class exponent_vector_visitor final : public IRTreeVoidReturnTypeVisitorInterface,
                                      public IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<vector<unsigned int>&,
                                              list<shared_ptr<irtree_node> >::iterator&,
                                              irtree_node*> {
    typedef size_t index_t;

    void visit(plus_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(minus_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(multiply_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(exponent_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(unary_negate_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(variable_node* num_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(parameter_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(number_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;
    void visit(sentinel_node* op_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               irtree_node* parent_node) override;

    managed_variables_index_table* symbol_table;

    using IRTreeVoidReturnTypeVisitorInterface::visit;

public:

    monomial_coefficient_table monomial_coefficient_table_;

    void visit(multiply_node* op_node) override;
    void visit(plus_node* op_node) override;
    void visit(sentinel_node* op_node) override;

    exponent_vector_visitor* operator()(managed_variables_index_table& symbol_table) {
        this->symbol_table = &symbol_table;
        return this;
    }

    void print() const {
        for (const auto& it : monomial_coefficient_table_) {
            const vector<unsigned int>& vec = it.first;
            const vector<irtree_node*> root_nodes = it.second;
            std::cout << "[";
            for (const auto& node : root_nodes) std::cout << node << " ";
            std::cout << "]:  [";
            for (const auto& itit : vec) std::cout << itit << " ";
            std::cout << "]" << std::endl;
        }
    }

    friend class irtree_node;
    friend class plus_node;
    friend class minus_node;
    friend class multiply_node;
    friend class exponent_node;
    friend class unary_negate_node;
    friend class variable_node;
    friend class parameter_node;
    friend class number_node;
    friend class sentinel_node;
};


#endif // EXPONENT_VECTOR_VISITOR_H
