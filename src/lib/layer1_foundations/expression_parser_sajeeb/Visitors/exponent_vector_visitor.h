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

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

using std::unordered_set;
using std::vector;
using std::unordered_map;
using std::string;

class managed_variables_index_table {
    unordered_map<string, size_t> index_table;
    size_t current_index = 0;

public:
    template<class T>
    void insert(const T& name) {
        index_table.insert({name, current_index++});
    }

    template<class T, class... Args>
    void insert(const T& name, const Args&... args) {
        index_table.insert({name, current_index++});
        insert(args...);
    }

    size_t index(const string& key) const {
        return index_table.at(key);
    }

    size_t size() const noexcept {
        return current_index;
    }

    decltype(index_table.begin()) find(const string& key)  {
        return index_table.find(key);
    }

    decltype(index_table.end()) end() noexcept {
        return index_table.end();
    }
};


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
                                                              irtree_node*,
                                                              monomial_coefficient_table_hash_function,
                                                              monomial_coefficient_table_key_equal_function> {};


class exponent_vector_visitor final : public IRTreeChildLinkArgumentVisitor,
                                      public IRTreeVoidReturnTypeVariadicArgumentVisitor<vector<unsigned int>&> {
    typedef size_t index_t;

    void visit(plus_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(minus_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(multiply_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(exponent_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(unary_negate_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(variable_node* num_node, vector<unsigned int>& exponent_vector) override;
    void visit(parameter_node* node, vector<unsigned int>& exponent_vector) override;
    void visit(number_node* op_node, vector<unsigned int>& exponent_vector) override;
    void visit(sentinel_node* op_node, vector<unsigned int>& exponent_vector) override;

    managed_variables_index_table* symbol_table;
    monomial_coefficient_table monomial_coefficient_table_;

    using IRTreeVoidReturnTypeVisitor::visit;

public:
    void visit(multiply_node* op_node) override;
    exponent_vector_visitor* operator()(managed_variables_index_table& symbol_table) {
        this->symbol_table = &symbol_table;
        return this;
    }

    void print() const {
        for (const auto& it : monomial_coefficient_table_) {
            const vector<unsigned int>& vec = it.first;
            const irtree_node* root = it.second;
            std::cout << root << ":  [";
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
