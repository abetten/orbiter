#include "multiplication_expansion_visitor.h"
#include "../../IRTree/node.h"
#include <functional>

#define LOG(x) std::cout << __FILE__ << ":" << __LINE__ << ": " << x << std::endl;

using std::make_shared;
using std::cout;
using std::endl;

void multiplication_expansion_visitor::expand_multiplication_node(multiply_node*& op_node, iterator_t& link) {
    shared_ptr<non_terminal_node> substitution_node = make_shared<plus_node>();
    shared_ptr<non_terminal_node> multiplication_node = make_shared<multiply_node>();

    iterator_t child_iter = op_node->children.begin();
    iterator_t child_iter_end = op_node->children.end();

    auto get_grandchild_iterator = [](const iterator_t& child_iter,
                                      iterator_t& grandchild_iter,
                                      iterator_t& grandchild_iter_end){
        if ((*child_iter)->is_terminal()) grandchild_iter = child_iter, (grandchild_iter_end = child_iter)++;
        else {
            non_terminal_node* child_raw_ptr = static_cast<non_terminal_node*>(child_iter->get());
            grandchild_iter = child_raw_ptr->children.begin();
            grandchild_iter_end = child_raw_ptr->children.end();
        }
    };

    std::function<void(iterator_t)> combination_sequence = [&child_iter_end,
                                                            &substitution_node,
                                                            &multiplication_node,
                                                            &get_grandchild_iterator,
                                                            &combination_sequence](iterator_t child_iter){
        ++child_iter;
        if (child_iter == child_iter_end) {
            substitution_node->add_child(multiplication_node);
            // TODO: implement copy constructor for AST nodes
            cout << endl;
            return;
        }

        iterator_t grandchild_iter, grandchild_iter_end;
        get_grandchild_iterator(child_iter, grandchild_iter, grandchild_iter_end);

        for (; grandchild_iter != grandchild_iter_end; ++grandchild_iter) {        
            multiplication_node->add_child(*grandchild_iter);
            cout << static_cast<variable_node*>(grandchild_iter->get())->name << " -> ";
            combination_sequence(child_iter);
            multiplication_node->children.pop_back();
        }
    };

    iterator_t grandchild_iter, grandchild_iter_end;
    get_grandchild_iterator(child_iter, grandchild_iter, grandchild_iter_end);

    for (; grandchild_iter != grandchild_iter_end; ++grandchild_iter) {
        LOG("");
        multiplication_node->add_child(*grandchild_iter);
        cout << "--> " << static_cast<variable_node*>(grandchild_iter->get())->name << " ";
        combination_sequence(child_iter);
        multiplication_node->children.pop_back();
    }
}

void multiplication_expansion_visitor::visit(multiply_node* op_node, iterator_t& link) {
    LOG("");
    bool expand = false;
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it) {
        if (!(*it)->is_terminal() || (*it)->type == irtree_node::node_type::EXPONENT_NODE) expand = true;
        if ((*it)->type == irtree_node::node_type::MINUS_NODE)
            throw not_implemented("Expansion sequence with following MINUS node not implemented.");
    }
    if (expand) expand_multiplication_node(op_node, link);
    for (auto it=op_node->children.begin(); it != op_node->children.end(); ++it)
        dispatcher::visit(*it, *this, link);
}
