#include <list>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include "../Visitors/IRTreeVisitor.h"
#include "../Visitors/IRTreeChildLinkArgumentVisitor.h"
#include "../Visitors/CopyVisitors/deep_copy_visitor.h"
#include "../Visitors/exponent_vector_visitor.h"
#include "../Visitors/ReductionVisitors/simplify_visitor.h"
#include "node_forward_declaration.h"
#include "../Visitors/EvaluateVisitors/eval_visitor.h"

#ifndef IR_TREE_NODE
#define IR_TREE_NODE

using std::string;
using std::list;
using std::shared_ptr;
using std::vector;

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

#define DEFINE_ACCEPT_VISITOR_FUNCTION() \
    void accept(IRTreeVoidReturnTypeVisitorInterface* visitor) override {visitor->visit(this);} \
    void accept(IRTreeChildLinkArgumentVisitor* visitor) override {visitor->visit(this);} \
    void accept(IRTreeChildLinkArgumentVisitor* visitor, list<shared_ptr<irtree_node>>::iterator& link) override \
        {visitor->visit(this, link);}    \
    shared_ptr<irtree_node> accept(deep_copy_visitor* visitor) override {\
        return visitor->visit(this);                                     \
    }                                    \
    void accept(exponent_vector_visitor* visitor) override {             \
        visitor->visit(this);                                 \
    }                                    \
    void accept(simplify_visitor* visitor) override { visitor->visit(this); }                                    \
    int accept_simplify_numerical_visitor(IRTreeTemplateReturnTypeVariadicArgumentVisitorInterface<int, irtree_node*>* visitor,                     \
               irtree_node* parent_node) override {             \
        return visitor->visit(this, parent_node);                       \
    }                               \
    int accept(eval_visitor* visitor, finite_field* Fq, unordered_map<string, int>& assignment_table) override {    \
        return visitor->visit(this, Fq, assignment_table);                                   \
    }\
                       \
    private:                             \
    void accept(exponent_vector_visitor* visitor, \
                vector<unsigned int>& exponent_vector, \
                list<shared_ptr<irtree_node> >::iterator& link, \
                irtree_node* parent_node) override {      \
        visitor->visit(this, exponent_vector, link, parent_node);                                 \
    }\
    void accept(deep_copy_visitor* visitor, shared_ptr<irtree_node> root) override {          \
        return visitor->visit(this, root);  \
    }                                    \
                                         \
    public:                                     \
    friend class deep_copy_visitor;             \
    friend class exponent_vector_visitor;



class irtree_node {
    virtual void accept(deep_copy_visitor* visitor, shared_ptr<irtree_node> root) = 0;
    virtual void accept(exponent_vector_visitor* visitor,
                        vector<unsigned int>& exponent_vector,
                        list<shared_ptr<irtree_node> >::iterator& link,
                        irtree_node* parent_node) = 0;

public:
	enum class node_type {
		SENTINEL_NODE, NUMBER_NODE, PARAMETER_NODE, VARIABLE_NODE,
		UNARY_NEGATE_NODE, EXPONENT_NODE, MULTIPLY_NODE, MINUS_NODE, 
		PLUS_NODE
	};

	irtree_node(node_type _type): type(_type) {}
    irtree_node(const irtree_node& other) : type(other.type) {}

    inline
    bool is_terminal() const {
        return (type == node_type::NUMBER_NODE) ||
               (type == node_type::PARAMETER_NODE) ||
               (type == node_type::VARIABLE_NODE);
    }

    virtual ~irtree_node();
    virtual void accept(simplify_visitor* visitor) = 0;
    virtual int accept_simplify_numerical_visitor(
                        IRTreeTemplateReturnTypeVariadicArgumentVisitorInterface<int, irtree_node*>* visitor,
                        irtree_node* parent_node) = 0;
    virtual void accept(IRTreeVoidReturnTypeVisitorInterface* visitor) = 0;
	virtual void accept(IRTreeChildLinkArgumentVisitor* visitor) = 0;
	virtual void accept(IRTreeChildLinkArgumentVisitor* visitor,
						list<shared_ptr<irtree_node>>::iterator& link) = 0;
    virtual shared_ptr<irtree_node> accept(deep_copy_visitor* visitor) = 0;
    virtual void accept(exponent_vector_visitor* visitor) = 0;
    virtual int accept(eval_visitor* visitor, finite_field* Fq, unordered_map<string, int>& assignment_table) = 0;


	node_type type;

    friend class deep_copy_visitor;
    friend class exponent_vector_visitor;
};
std::ostream& operator<< (std::ostream& os, const irtree_node::node_type& obj);

class non_terminal_node : public irtree_node {
public:
	list<shared_ptr<irtree_node>> children;
	non_terminal_node(node_type _type) : irtree_node(_type) {}
    non_terminal_node(const non_terminal_node& other): irtree_node(other) {}

	inline
    void add_child() {
		// DO NOT IMPLEMENT
	}

	inline
    void add_child(shared_ptr<irtree_node>&& node) {
		children.emplace_back(node) ;
	}

	template<typename T>
	inline
    void add_child(T& node) {
		children.emplace_back(node) ;
	}

    template <typename T, typename... Args>
	inline
    void add_child(T& arg, Args&... args) {
		children.emplace_back(arg);
		add_child(args...);
	}

	inline
    void add_child(irtree_node*& node) {
		children.emplace_back(shared_ptr<irtree_node>(node)) ;
	}

    virtual ~non_terminal_node();
};

class terminal_node : public irtree_node {
public:
	terminal_node(node_type _type) : irtree_node(_type) {}
    terminal_node(const terminal_node& other) : irtree_node(other) {}
    virtual ~terminal_node();
};

class plus_node: public non_terminal_node {
public:
	plus_node() : non_terminal_node(irtree_node::node_type::PLUS_NODE) {}
    plus_node(const plus_node& other) : non_terminal_node(other) {}
    virtual ~plus_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

class minus_node: public non_terminal_node {
public:
	minus_node() : non_terminal_node(irtree_node::node_type::MINUS_NODE) {}
    minus_node(const minus_node& other) : non_terminal_node(other) {}
    virtual ~minus_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

class multiply_node: public non_terminal_node {
public:
	multiply_node() : non_terminal_node(irtree_node::node_type::MULTIPLY_NODE) {}
    virtual ~multiply_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

class exponent_node: public non_terminal_node {
public:
	exponent_node() : non_terminal_node(irtree_node::node_type::EXPONENT_NODE) {}
    virtual ~exponent_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

class unary_negate_node: public non_terminal_node {
public:
	unary_negate_node() : non_terminal_node(irtree_node::node_type::UNARY_NEGATE_NODE) {}
    virtual ~unary_negate_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

class variable_node: public terminal_node {
public:
	variable_node(string name) : terminal_node(irtree_node::node_type::VARIABLE_NODE), name(name) {}
    virtual ~variable_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
    
	string name;
};

class parameter_node: public terminal_node {
public:
	parameter_node(string name) : terminal_node(irtree_node::node_type::PARAMETER_NODE), name(name) {}
    virtual ~parameter_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
    
	string name;
};

class number_node: public terminal_node {
public:
    number_node(int val) : terminal_node(irtree_node::node_type::NUMBER_NODE), value(val) {}
    virtual ~number_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();

    int value;
};

class sentinel_node: public non_terminal_node {
public:
	sentinel_node() : non_terminal_node(irtree_node::node_type::SENTINEL_NODE) {}
    virtual ~sentinel_node();
    DEFINE_ACCEPT_VISITOR_FUNCTION();
};

#endif /* IR_TREE_NODE */
