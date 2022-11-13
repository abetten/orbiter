//
// Created by Sajeeb Roy Chowdhury on 9/30/22.
//

#include <iostream>
#include <string>

#ifndef GRAMMAR_TEST_IR_TREE_LATEX_VISITOR_H
#define GRAMMAR_TEST_IR_TREE_LATEX_VISITOR_H

class ir_tree_latex_visitor {
protected:
    std::ostream* output_stream;
    std::string indentation;
    std::string delimiter = "  ";

    virtual void add_epilogue() = 0;
    virtual void add_prologue() = 0;

    void add_indentation() {
        indentation.append(delimiter);
    }

    void remove_indentation() {
        indentation.erase(indentation.size() - delimiter.size());
    }

public:
    ir_tree_latex_visitor() {}
    ir_tree_latex_visitor(std::ostream& ostream) : output_stream(&ostream) {}

    void set_output_stream(std::ostream& ostream) {
        output_stream = &ostream;
    }

    void unset_output_stream() {
        output_stream = NULL;
    }
};


#endif //GRAMMAR_TEST_IR_TREE_LATEX_VISITOR_H
