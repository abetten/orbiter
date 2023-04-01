/**
* Author:    Sajeeb Roy Chowdhury
* Created:   10/8/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>

#include "IRTreeVisitor.h"
#include "dispatcher.h"
#include "../exception.h"
#include "../../foundations.h"

#ifndef EXPONENT_VECTOR_VISITOR_H
#define EXPONENT_VECTOR_VISITOR_H

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

using std::shared_ptr;
using std::list;
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
                                                              vector<shared_ptr<irtree_node>>,
                                                              monomial_coefficient_table_hash_function,
                                                              monomial_coefficient_table_key_equal_function> {};


class exponent_vector_visitor final : public IRTreeVoidReturnTypeVisitorInterface,
                                      public IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<shared_ptr<irtree_node>&>,
                                      public IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<vector<unsigned int>&,
                                              list<shared_ptr<irtree_node> >::iterator&,
                                              shared_ptr<irtree_node>&,
                                              shared_ptr<irtree_node>&> {
    using IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<shared_ptr<irtree_node>&>::visit;
    using IRTreeVoidReturnTypeVariadicArgumentVisitorInterface<vector<unsigned int>&,
                                              list<shared_ptr<irtree_node> >::iterator&,
                                              shared_ptr<irtree_node>&,
                                              shared_ptr<irtree_node>&>::visit;
    using IRTreeVoidReturnTypeVisitorInterface::visit;
    typedef size_t index_t;

    void visit(plus_node* node, shared_ptr<irtree_node>& node_self) override;
    void visit(minus_node* node, shared_ptr<irtree_node>& node_self) override;
    void visit(multiply_node* node, shared_ptr<irtree_node>& node_self) override;
    void visit(exponent_node* node, shared_ptr<irtree_node>& node_self) override;
    void visit(unary_negate_node* node, shared_ptr<irtree_node>& node_self) override;
    void visit(sentinel_node* node, shared_ptr<irtree_node>& node_self) override;
    
    void visit(plus_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(minus_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(multiply_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(exponent_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(unary_negate_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(variable_node* num_node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(parameter_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(number_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;
    void visit(sentinel_node* node,
               vector<unsigned int>& exponent_vector,
               list<shared_ptr<irtree_node> >::iterator& link,
               shared_ptr<irtree_node>& node_self,
               shared_ptr<irtree_node>& parent_node) override;

    managed_variables_index_table* symbol_table;

public:

    monomial_coefficient_table monomial_coefficient_table_;

    void visit(sentinel_node* node) override;

    exponent_vector_visitor& operator()(managed_variables_index_table& symbol_table) {
        this->symbol_table = &symbol_table;
        return *this;
    }

    void print() const {
        for (const auto& it : monomial_coefficient_table_) {
            const vector<unsigned int>& vec = it.first;
            const vector<shared_ptr<irtree_node>>& root_nodes = it.second;
            std::cout << "[";
            for (const auto& node : root_nodes) std::cout << node << " ";
            std::cout << "]:  [";
            for (const auto& itit : vec) std::cout << itit << " ";
            std::cout << "]" << std::endl;
        }
    }

    friend dispatcher;
};


#endif // EXPONENT_VECTOR_VISITOR_H
