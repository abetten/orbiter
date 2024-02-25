#include <list>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <variant>


#include "node_forward_declaration.h"

#ifndef IR_TREE_NODE
#define IR_TREE_NODE

using std::string;
using std::list;
using std::shared_ptr;
using std::vector;

#ifndef LOG
#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;
#endif

//! base class for a node in Sajeeb's abstract syntax tree

class irtree_node {
public:
	enum class node_type {
		SENTINEL_NODE, NUMBER_NODE, PARAMETER_NODE, VARIABLE_NODE,
		UNARY_NEGATE_NODE, EXPONENT_NODE, MULTIPLY_NODE, MINUS_NODE, 
		PLUS_NODE
	};

	irtree_node(node_type _type): type(_type) {

    }
    irtree_node(const irtree_node& other) : type(other.type) {}

    inline
    bool is_terminal() const {
        return (type == node_type::NUMBER_NODE) ||
               (type == node_type::PARAMETER_NODE) ||
               (type == node_type::VARIABLE_NODE);
    }

    virtual ~irtree_node();
	node_type type;
};
std::ostream& operator<< (std::ostream& os, const irtree_node::node_type& obj);

//! derived class for non terminal nodes in Sajeeb's abstract syntax tree

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
    void add_child(T&& node) {
		children.emplace_back(std::forward<T>(node)) ;
	}

    template <typename T, typename... Args>
	inline
    void add_child(T&& arg, Args&&... args) {
		children.emplace_back(std::forward<T>(arg));
		add_child(std::forward<T>(args)...);
	}

	inline
    void add_child(irtree_node*& node) {
		children.emplace_back(shared_ptr<irtree_node>(node)) ;
	}

    virtual ~non_terminal_node();
};

//! derived class for terminal nodes in Sajeeb's abstract syntax tree


class terminal_node : public irtree_node {
public:
	terminal_node(node_type _type) : irtree_node(_type) {}
    terminal_node(const terminal_node& other) : irtree_node(other) {}
    virtual ~terminal_node();
};


//! derived class for plus nodes in Sajeeb's abstract syntax tree

class plus_node: public non_terminal_node {
public:
	plus_node() : non_terminal_node(irtree_node::node_type::PLUS_NODE) {}
    plus_node(const plus_node& other) : non_terminal_node(other) {}
    virtual ~plus_node();
};

//! derived class for minus nodes in Sajeeb's abstract syntax tree

class minus_node: public non_terminal_node {
public:
	minus_node() : non_terminal_node(irtree_node::node_type::MINUS_NODE) {}
    minus_node(const minus_node& other) : non_terminal_node(other) {}
    virtual ~minus_node();
};


//! derived class for multiplication nodes in Sajeeb's abstract syntax tree

class multiply_node: public non_terminal_node {
public:
	multiply_node() : non_terminal_node(irtree_node::node_type::MULTIPLY_NODE) {}
    virtual ~multiply_node();
};

//! derived class for exponent nodes in Sajeeb's abstract syntax tree

class exponent_node: public non_terminal_node {
public:
	exponent_node() : non_terminal_node(irtree_node::node_type::EXPONENT_NODE) {}
    virtual ~exponent_node();
};

//! derived class for unary negate nodes in Sajeeb's abstract syntax tree

class unary_negate_node: public non_terminal_node {
public:
	unary_negate_node() : non_terminal_node(irtree_node::node_type::UNARY_NEGATE_NODE) {}
    virtual ~unary_negate_node();
};

//! derived class for variable nodes in Sajeeb's abstract syntax tree

class variable_node: public terminal_node {
public:
	variable_node(string name) : terminal_node(irtree_node::node_type::VARIABLE_NODE), name(name) {}
    virtual ~variable_node();
    
	string name;
};

//! derived class for parameter nodes in Sajeeb's abstract syntax tree

class parameter_node: public terminal_node {
public:
	parameter_node(string name) : terminal_node(irtree_node::node_type::PARAMETER_NODE), name(name) {}
    virtual ~parameter_node();
    
	string name;
};

//! derived class for number nodes in Sajeeb's abstract syntax tree

class number_node: public terminal_node {
public:
    number_node(int val) : terminal_node(irtree_node::node_type::NUMBER_NODE), value(val) {}
    virtual ~number_node();

    int value;
};

//! derived class for the root node in Sajeeb's abstract syntax tree

class sentinel_node: public non_terminal_node {
public:
	sentinel_node() : non_terminal_node(irtree_node::node_type::SENTINEL_NODE) {}
    virtual ~sentinel_node();
};

#endif /* IR_TREE_NODE */
