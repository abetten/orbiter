/*
 * syntax_tree_latex.cpp
 *
 *  Created on: May 23, 2023
 *      Author: betten
 */





#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace expression_parser {


syntax_tree_latex::syntax_tree_latex()
{
	Record_birth();

}

syntax_tree_latex::~syntax_tree_latex()
{
	Record_death();

}

void syntax_tree_latex::latex_tree(
		std::string &name, syntax_tree_node *Root,
		int verbose_level)
{


	string fname_wrapper;
	string fname_body;

	fname_wrapper = name + ".tex";
	fname_body = name + "_body.tex";
	f_split = false;
	split_level = 0;
	split_r = 0;
	split_mod = 1;
	indentation = "";
	delimiter = " ";



	std::ofstream *ost = new std::ofstream(fname_wrapper);

	*ost << "\\documentclass{standalone}" << "\n"
        << "\\usepackage{forest}" << "\n"
        << "\\usetikzlibrary{arrows.meta}" << "\n"
        << "\\begin{document}" << endl;
	*ost << "\\input " << fname_body << endl;
	*ost <<  "\\end{document}" << endl;

	delete ost;


	output_stream = new std::ofstream(fname_body);
	add_prologue();
	add_indentation();
	*output_stream << indentation << "[";
	add_indentation();
	latex_tree_recursion(Root, 0, verbose_level);
	remove_indentation();
	*output_stream << indentation << "]";
	remove_indentation();
	add_epilogue();
	delete output_stream;

}

void syntax_tree_latex::export_tree(
		std::string &name, syntax_tree_node *Root,
		int verbose_level)
{

	string fname_tree;

	fname_tree = name + ".tree";
	f_split = false;
	split_level = 0;
	split_r = 0;
	split_mod = 1;
	indentation = "";
	delimiter = " ";




	{
		output_stream = new std::ofstream(fname_tree);


		std::vector<int> path;


		export_tree_recursion(Root, 0, path, verbose_level);
		*output_stream << -1 << endl;

		delete output_stream;
		output_stream = NULL;
	}

}


void syntax_tree_latex::latex_tree_split(
		std::string &name, syntax_tree_node *Root,
		int split_level, int split_mod,
		int verbose_level)
{

	syntax_tree_latex L;
	string fname_wrapper;
	string fname_body;

	f_split = true;
	syntax_tree_latex::split_level = split_level;
	split_r = 0;
	syntax_tree_latex::split_mod = split_mod;

	for (split_r = 0; split_r < split_mod; split_r++) {

		fname_wrapper = name + "_case" + std::to_string(split_r) + ".tex";
		fname_body = name + "_case" + std::to_string(split_r) + "_body.tex";

		indentation = "";
		delimiter = " ";


		std::ofstream *ost = new std::ofstream(fname_wrapper);

		*ost << "\\documentclass{standalone}" << "\n"
	        << "\\usepackage{forest}" << "\n"
	        << "\\usetikzlibrary{arrows.meta}" << "\n"
	        << "\\begin{document}" << endl;
		*ost << "\\input " << fname_body << endl;
		*ost <<  "\\end{document}" << endl;

		delete ost;


		output_stream = new std::ofstream(fname_body);

		add_prologue();
		add_indentation();
		*output_stream << indentation << "[";
		add_indentation();
		latex_tree_recursion(Root, 0, 0 /*verbose_level*/);
		remove_indentation();
		*output_stream << indentation << "]";
		remove_indentation();
		add_epilogue();
		delete output_stream;
	}

}


void syntax_tree_latex::add_prologue()
{
#if 0
	*output_stream << "\\documentclass{standalone}" << "\n"
        << "\\usepackage{forest}" << "\n"
        << "\\usetikzlibrary{arrows.meta}" << "\n"
        << "\\begin{document}" << "\n";
#endif

	*output_stream << "\\begin{forest}" << "\n"
        << "for tree={" << "\n"
        << "grow'=0, treenode/.style = {align=center, inner sep=2.5pt," << "\n"
        << "    text centered, font=\\sffamily}," << "\n"
        << "arn_n/.style = {treenode, rectangle, text width=1.5em}," << "\n"
        << "arn_x/.style = {treenode}," << "\n"
        << "gray-arrow/.style = {draw=gray}," << "\n"
        << "edge=-{Stealth}}" << "\n"
        << "[\\fbox{R}, arn_n" << "\n";
}

void syntax_tree_latex::add_epilogue()
{
	*output_stream << "]" << "\n"
        << "\\end{forest}" << endl;
    //    <<  "\\end{document}" << "\n";
}

void syntax_tree_latex::add_indentation()
{
	indentation += delimiter;
}

void syntax_tree_latex::remove_indentation()
{
	indentation.erase(indentation.size() - delimiter.size());
}

void syntax_tree_latex::latex_tree_recursion(
		syntax_tree_node *node, int depth,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_latex::latex_tree_recursion" << endl;
	}
	if (node->f_terminal) {
		if (node->T->f_int) {
			if (node->f_has_exponent) {
				*output_stream << node->T->value_int
						<< "\\^{}{" << node->exponent << "}, arn_x";
			}
			else {
				*output_stream << node->T->value_int << ", arn_x";
			}
		}
		else if (node->T->f_double) {
			*output_stream << node->T->value_double << ", arn_x";
		}
		else if (node->T->f_text) {
			if (node->f_has_exponent) {
				*output_stream << node->T->value_text
						<< "\\^{}{" << node->exponent << "}, arn_x";
			}
			else {
				*output_stream << node->T->value_text << ", arn_x";
			}
		}
	}
	else {
		int i;

		if (node->type == operation_type_mult) {
			if (node->f_has_exponent) {
				*output_stream << "\\fbox{\\^{}{" << node->exponent << "}}, arn_n" << "\n";
				add_indentation();
				*output_stream << indentation << "[";
			}
			*output_stream << "\\fbox{*}, arn_n" << "\n";
			add_indentation();
			for (i = 0; i < node->nb_nodes; i++) {
				if (f_split && depth == split_level && (i % split_mod) != split_r) {
					continue;
				}
				*output_stream << indentation << "[";
				latex_tree_recursion(node->Nodes[i], depth + 1, verbose_level);
				*output_stream << indentation << "]\n";
			}
			remove_indentation();
			if (node->f_has_exponent) {
				*output_stream << indentation << "]";
				remove_indentation();
			}
		}
		else if (node->type == operation_type_add) {
			if (node->f_has_exponent) {
				*output_stream << "\\fbox{\\^{}{" << node->exponent << "}}, arn_n" << "\n";
				add_indentation();
				*output_stream << indentation << "[";
			}
			*output_stream << "\\fbox{+}, arn_n" << "\n";
			add_indentation();
			for (i = 0; i < node->nb_nodes; i++) {
				if (f_split && depth == split_level && (i % split_mod) != split_r) {
					continue;
				}
				*output_stream << indentation << "[";
				latex_tree_recursion(node->Nodes[i], depth + 1, verbose_level);
				*output_stream << indentation << "]\n";
			}
			remove_indentation();
			if (node->f_has_exponent) {
				*output_stream << indentation << "]";
				remove_indentation();
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_latex::latex_tree_recursion done" << endl;
	}
}


void syntax_tree_latex::export_tree_recursion(
		syntax_tree_node *node, int depth,
		std::vector<int> &path,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_latex::latex_tree_recursion" << endl;
	}

	int i;
	*output_stream << path.size() + 1 << " ";
	for (i = 0; i < path.size(); i++) {
		*output_stream << path[i] << " ";
	}
	*output_stream << node->idx << " ";
	if (node->f_terminal) {
		*output_stream << "$t$";
	}
	else {
		if (node->type == operation_type_mult) {
			*output_stream << "$+$";

		}
		else if (node->type == operation_type_add) {
			*output_stream << "$-$";
		}
	}
	*output_stream << endl;

	if (node->f_terminal) {

	}
	else {
		int i;

		if (node->type == operation_type_mult) {
			for (i = 0; i < node->nb_nodes; i++) {
				path.push_back(node->idx);

				export_tree_recursion(node->Nodes[i], depth + 1, path, verbose_level);

				path.pop_back();

			}
		}
		else if (node->type == operation_type_add) {
			for (i = 0; i < node->nb_nodes; i++) {
				path.push_back(node->idx);

				export_tree_recursion(node->Nodes[i], depth + 1, path, verbose_level);

				path.pop_back();

			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_latex::latex_tree_recursion done" << endl;
	}
}


}}}}


