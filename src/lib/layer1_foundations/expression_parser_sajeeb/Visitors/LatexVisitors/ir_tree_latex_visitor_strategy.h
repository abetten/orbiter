//
// Created by Sajeeb Roy Chowdhury on 9/29/22.
//

#include <unordered_map>
#include <vector>
#include "ir_tree_latex_visitor.h"
#include "../IRTreeVisitor.h"
#include "ir_tree_latex_visitor_family_tree.h"
#include "ir_tree_latex_visitor_simple_tree.h"
#include <memory>

#ifndef IR_TREE_LATEX_VISITOR_STRATEGY
#define IR_TREE_LATEX_VISITOR_STRATEGY

using std::unordered_map;
using std::shared_ptr;
using std::make_shared;

class ir_tree_latex_visitor_strategy final {
public:
    enum class type {
        FAMILY_TREE=0, SIMPLE_TREE
    };
    static vector<shared_ptr<IRTreeVoidReturnTypeVisitorInterface>> concrete_strategy;

    class wrapper final {
        shared_ptr<IRTreeVoidReturnTypeVisitorInterface> concrete_strategy;
        ir_tree_latex_visitor_strategy::type strategy_type;

    public:
        wrapper(ir_tree_latex_visitor_strategy::type _type) : strategy_type(_type) {
            switch (_type) {
                case ir_tree_latex_visitor_strategy::type::FAMILY_TREE: {
                    concrete_strategy = make_shared<ir_tree_latex_visitor_family_tree>();
                    break;
                }

                case ir_tree_latex_visitor_strategy::type::SIMPLE_TREE: {
                    concrete_strategy = make_shared<ir_tree_latex_visitor_simple_tree>();
                    break;
                }

                default:
                    break;
            }
        }

        IRTreeVoidReturnTypeVisitorInterface* get_visitor() {
            return concrete_strategy.get();
        }

        void set_output_stream(std::ostream& ostream) {
            IRTreeVoidReturnTypeVisitorInterface* raw_ptr = concrete_strategy.get();
            switch (strategy_type) {
                case ir_tree_latex_visitor_strategy::type::FAMILY_TREE: {
                    static_cast<ir_tree_latex_visitor_family_tree*>(raw_ptr)->set_output_stream(ostream);
                    break;
                }

                case ir_tree_latex_visitor_strategy::type::SIMPLE_TREE: {
                    static_cast<ir_tree_latex_visitor_simple_tree*>(raw_ptr)->set_output_stream(ostream);
                    break;
                }

                default:
                    break;
            }
        }
    };

    static const wrapper get(const ir_tree_latex_visitor_strategy::type _type) {
        return wrapper(_type);
    }
};

#endif //IR_TREE_LATEX_VISITOR_STRATEGY
