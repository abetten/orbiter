/*
 * lexer.cpp
 *
 *  Created on: Feb 14, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


syntax_tree::syntax_tree()
{
	f_has_managed_variables = FALSE;
	//std::vector<std::string> managed_variables;
	Root = NULL;
}

void syntax_tree::print(std::ostream &ost)
{
	Root->print(ost);
}

void syntax_tree::print_monomial(std::ostream &ost, int *monomial)
{
	int i;

	for (i = 0; i < managed_variables.size(); i++) {
		if (monomial[i]) {
			ost << "*" << managed_variables[i];
			if (monomial[i] > 1) {
				ost << "^" << monomial[i];
			}
		}
	}
}

int syntax_tree::identify_single_literal(std::string &single_literal)
{
	int i;

	cout << "syntax_tree::identify_single_literal trying to identify " << single_literal << endl;
	for (i = 0; i < managed_variables.size(); i++) {
		if (strcmp(single_literal.c_str(), managed_variables[i].c_str()) == 0) {
			cout << "syntax_tree::identify_single_literal literal identified as managed variable " << i << endl;
			return i;
		}
	}
	return -1;
}

int syntax_tree::is_homoegeneous(int &degree)
{
	int ret;

	if (!f_has_managed_variables) {
		return FALSE;
	}
	else {
		degree = -1;

		ret = Root->is_homogeneous(degree);
		return ret;
	}
}


void syntax_tree::split_by_monomials(homogeneous_polynomial_domain *Poly,
		syntax_tree_node **&Subtrees, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree::split_by_monomials" << endl;
	}
	if (!f_has_managed_variables) {
		cout << "syntax_tree::split_by_monomials !f_has_managed_variables" << endl;
		exit(1);
	}


	int nb_monomials;
	int i;

	nb_monomials = Poly->get_nb_monomials();
	Subtrees = (syntax_tree_node **) NEW_pvoid(nb_monomials);

	for (i = 0; i < nb_monomials; i++) {
		Subtrees[i] = NULL;
	}
	Root->split_by_monomials(Poly, Subtrees, verbose_level);

	if (f_v) {
		for (i = 0; i < nb_monomials; i++) {
			cout << "Monomial " << i << " has subtree:" << endl;
			if (Subtrees[i]) {
				Subtrees[i]->print(cout);
			}
			else {
				cout << "no subtree" << endl;
			}
		}
	}
	if (f_v) {
		cout << "syntax_tree::split_by_monomials done" << endl;
	}

}

syntax_tree_node_terminal::syntax_tree_node_terminal()
{
	f_int = false;
	f_double = false;
	f_text = false;
	value_int = 0;
	value_double = 0.;
	//value_text;

}


void syntax_tree_node_terminal::print(std::ostream &ost)
{
	ost << "terminal node, ";
	if (f_int) {
		ost << "int=" << value_int << std::endl;
	}
	else if (f_double) {
		ost << "double=" << value_double << std::endl;
	}
	else if (f_text) {
		ost << "text=" << value_text << std::endl;
	}
}

void syntax_tree_node_terminal::print_expression(std::ostream &ost)
{
	if (f_int) {
		ost << value_int;
	}
	else if (f_double) {
		ost << value_double;
	}
	else if (f_text) {
		ost << value_text;
	}
}

int syntax_tree_node_terminal::evaluate(std::map<std::string, std::string> &symbol_table,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;

	if (f_v) {
		cout << "syntax_tree_node_terminal::evaluate" << endl;
	}
	if (f_int) {
		a = value_int;
	}
	else if (f_double) {
		a = value_double;
		//cout << "syntax_tree_node_terminal::evaluate f_double" << endl;
		//exit(1);
	}
	else if (f_text) {
		//a = strtoi(value_text);
		a = strtoi(symbol_table[value_text]);
	}
	else {
		cout << "syntax_tree_node_terminal::evaluate unknown type" << endl;
		exit(1);
	}

	if (a < 0) {
		cout << "syntax_tree_node_terminal::evaluate a < 0" << endl;
		exit(1);
	}
	if (a >= F->q) {
		cout << "syntax_tree_node_terminal::evaluate a >= F->q" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "syntax_tree_node_terminal::evaluate done, value = " << a << endl;
	}
	return a;
}



int syntax_tree_node_index = 0;

syntax_tree_node::syntax_tree_node()
{
	Tree = NULL;
	idx = syntax_tree_node_index;
	syntax_tree_node_index++;

	f_terminal = false;
	T = 0L;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;

	f_has_monomial = FALSE;
	monomial = NULL;

	f_has_minus = FALSE;
}

syntax_tree_node::~syntax_tree_node()
{

	if (f_terminal) {
		delete T;
		T = 0L;
	}
	else {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			delete Nodes[i];
		}
		if (monomial) {
			FREE_int(monomial);
		}
	}
}


void syntax_tree_node::null()
{
	idx = 0;

	f_terminal = false;
	T = 0L;

	type = operation_type_nothing;

	nb_nodes = 0;
	//Nodes = 0L;

}

void syntax_tree_node::split_by_monomials(homogeneous_polynomial_domain *Poly,
		syntax_tree_node **Subtrees, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "syntax_tree_node::split_by_monomials" << endl;
	}
	cout << "syntax_tree_node::split_by_monomials Node " << idx << endl;
	if (f_terminal) {
		return;
	}
	else {
		if (type == operation_type_mult) {
			cout << "checking multiplication node" << endl;
			idx = Poly->index_of_monomial(monomial);
			Subtrees[idx] = this;
		}
		else {
			int i;

			cout << "splitting subtree" << endl;
			for (i = 0; i < nb_nodes; i++) {
				Nodes[i]->split_by_monomials(Poly, Subtrees, verbose_level);
			}
		}
	}
}

int syntax_tree_node::is_homogeneous(int &degree)
{
	int deg, i;

	cout << "syntax_tree_node::is_homogeneous Node " << idx << endl;
	if (f_terminal) {
		return TRUE;
	}
	else {
		if (type == operation_type_mult) {
			cout << "checking multiplication node" << endl;
			deg = 0;
			for (i = 0; i < Tree->managed_variables.size(); i++) {
				deg += monomial[i];
			}
			cout << "syntax_tree_node::is_homogeneous node " << idx << " has degree " << deg << endl;
			if (degree == -1) {
				degree = deg;
				cout << "syntax_tree_node::is_homogeneous node " << idx << " setting degree to " << degree << endl;
			}
			else {
				if (deg != degree) {
					cout << "syntax_tree_node::is_homogeneous node " << idx << " has degree " << deg << " which is different from " << degree << ", so not homogeneous" << endl;
					return FALSE;
				}
			}
			return TRUE;
		}
		else {
			int i, ret;

			cout << "checking subtree" << endl;
			ret = TRUE;
			for (i = 0; i < nb_nodes; i++) {
				ret = Nodes[i]->is_homogeneous(degree);
				if (ret == FALSE) {
					return FALSE;
				}
			}
			return ret;
		}
	}

}

void syntax_tree_node::print(std::ostream &ost)
{

	ost << "Node " << idx << ": ";

	if (f_terminal) {
		ost << "is terminal" << std::endl;
		T->print(ost);
	}

	else {
		ost << "with " << nb_nodes << " descendants" << std::endl;
		ost << "f_has_minus = " << f_has_minus << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
		ost << "detailed list:" << std::endl;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			Nodes[i]->print(ost);
		}
	}
	if (f_has_monomial) {
		Tree->print_monomial(ost, monomial);
	}

}


int syntax_tree_node::evaluate(std::map<std::string, std::string> &symbol_table,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int a, b;

	if (f_v) {
		cout << "syntax_tree_node::evaluate" << endl;
	}
	if (f_terminal) {
		a = T->evaluate(symbol_table, F, verbose_level);
		if (f_has_minus) {
			a = F->negate(a);
		}
	}
	else {
		if (nb_nodes == 1) {
			a = Nodes[0]->evaluate(symbol_table, F, verbose_level);
			if (f_has_minus) {
				a = F->negate(a);
			}
		}
		else {
			if (type == operation_type_mult) {
				a = 1;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->mult(a, b);
				}
				if (f_has_minus) {
					a = F->negate(a);
				}
			}
			else if (type == operation_type_add) {
				a = 0;
				for (i = 0; i < nb_nodes; i++) {
					b = Nodes[i]->evaluate(symbol_table, F, verbose_level);
					a = F->add(a, b);
				}
			}
			else {
				cout << "syntax_tree_node::evaluate unknown operation" << endl;
				exit(1);
			}
		}
	}

	if (f_v) {
		cout << "syntax_tree_node::evaluate done, value = " << a << endl;
	}
	return a;
}

void syntax_tree_node::print_expression(std::ostream &ost)
{
	int i;

	if (f_terminal) {
		if (f_has_minus) {
			ost << "-";
		}
		T->print_expression(ost);
	}

	else {
		if (nb_nodes == 1) {
			if (f_has_minus) {
				ost << "-";
			}
			Nodes[0]->print_expression(ost);
		}
		else {
			ost << "(";
			if (type == operation_type_mult) {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
					if (i < nb_nodes - 1) {
						ost << "*";
					}
				}
			}
			else if (type == operation_type_add) {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
					if (i < nb_nodes - 1) {
						ost << "+";
					}
				}
			}
			else {
				if (f_has_minus) {
					ost << "-";
				}
				for (i = 0; i < nb_nodes; i++) {
					Nodes[i]->print_expression(ost);
				}
			}
			ost << ")";
		}
	}

}

void syntax_tree_node::push_a_minus_sign()
{
	if (f_has_minus) {
		f_has_minus = false;
	}
	else {
		f_has_minus = true;
	}

}

void syntax_tree_node::print_without_recursion(std::ostream &ost)
{

	ost << "Node " << idx << ": ";

	if (f_terminal) {
		ost << "is terminal" << std::endl;
		//T->print(ost);
	}

	else {
		ost << "with " << nb_nodes << " descendants" << std::endl;
		int i;
		for (i = 0; i < nb_nodes; i++) {
			ost << "Node " << idx << ", descendant " << i << " is node " << Nodes[i]->idx << std::endl;
			//Nodes[i]->print(ost);
		}
	}

}


void syntax_tree_node::export_graphviz(std::string &name, std::ostream &ost)
{
	ost << "graph " << name << " {" << std::endl;

	export_graphviz_recursion(ost);

	ost << "}" << std::endl;

}

void syntax_tree_node::export_graphviz_recursion(std::ostream &ost)
{

	if (f_terminal) {
		//ost << "is terminal" << std::endl;
		//T->print(ost);
	}

	else {
		int i;

		for (i = 0; i < nb_nodes; i++) {
			ost << idx << " -- " << Nodes[i]->idx << std::endl;
		}
		for (i = 0; i < nb_nodes; i++) {
			Nodes[i]->export_graphviz_recursion(ost);
		}
	}

}

lexer::lexer()
{
	  //std::string program_;

	  pWord_ = 0L;
	  pWordStart_ = 0L;
	  // last token parsed
	  type_ = NONE;
	  //std::string word_;
	  value_ = 0.;
	  T = 0L;
}


void lexer::print_token(std::ostream &ost, TokenType t)
{
	std::string s;

	token_as_string(s, t);
	ost << s;
}

void lexer::token_as_string(std::string &s, TokenType t)
{
	if (t == NONE) {
		s.assign("NONE");
	}
	else if (t == NAME) {
		s.assign("NAME");
	}
	else if (t == NUMBER) {
		s.assign("NUMBER");
	}
	else if (t == END) {
		s.assign("END");
	}
	else if (t == PLUS) {
		s.assign("PLUS");
	}
	else if (t == MINUS) {
		s.assign("MINUS");
	}
	else if (t == MULTIPLY) {
		s.assign("MULTIPLY");
	}
	else if (t == DIVIDE) {
		s.assign("DIVIDE");
	}
	else if (t == ASSIGN) {
		s.assign("ASSIGN");
	}
	else if (t == LHPAREN) {
		s.assign("LHPAREN");
	}
	else if (t == RHPAREN) {
		s.assign("RHPAREN");
	}
	else if (t == COMMA) {
		s.assign("COMMA");
	}
	else if (t == NOT) {
		s.assign("NOT");
	}
	else if (t == LT) {
		s.assign("LT");
	}
	else if (t == GT) {
		s.assign("GT");
	}
	else if (t == LE) {
		s.assign("LE");
	}
	else if (t == GE) {
		s.assign("GE");
	}
	else if (t == EQ) {
		s.assign("EQ");
	}
	else if (t == NE) {
		s.assign("NE");
	}
	else if (t == AND) {
		s.assign("AND");
	}
	else if (t == OR) {
		s.assign("OR");
	}
	else if (t == ASSIGN_ADD) {
		s.assign("ASSIGN_ADD");
	}
	else if (t == ASSIGN_SUB) {
		s.assign("ASSIGN_SUB");
	}
	else if (t == ASSIGN_MUL) {
		s.assign("ASSIGN_MUL");
	}
	else if (t == ASSIGN_DIV) {
		s.assign("ASSIGN_DIV");
	}
}


TokenType lexer::GetToken (int verbose_level, const bool ignoreSign)
{
	int f_v = (verbose_level >= 1);


	  word_.erase (0, std::string::npos);

	  // skip spaces
	  while (*pWord_ && isspace (*pWord_))
	    ++pWord_;

	  pWordStart_ = pWord_;   // remember where word_ starts *now*

	  // look out for unterminated statements and things
	  if (*pWord_ == 0 &&  // we have EOF
	      type_ == END)  // after already detecting it
	    throw std::runtime_error ("Unexpected end of expression.");

	  unsigned char cFirstCharacter = *pWord_;        // first character in new word_

	  if (cFirstCharacter == 0)    // stop at end of file
	    {
	    word_ = "<end of expression>";
	    if (f_v) {
	  	  std::cout << "token END " << word_ << std::endl;
	    }
		create_text_token(word_);
	    return type_ = END;
	    }

	  unsigned char cNextCharacter  = *(pWord_ + 1);  // 2nd character in new word_

	  // look for number
	  // can be: + or - followed by a decimal point
	  // or: + or - followed by a digit
	  // or: starting with a digit
	  // or: decimal point followed by a digit
	  if ((!ignoreSign &&
		   (cFirstCharacter == '+' || cFirstCharacter == '-') &&
		   (isdigit (cNextCharacter) || cNextCharacter == '.')
		   )
		  || isdigit (cFirstCharacter)
		  // allow decimal numbers without a leading 0. e.g. ".5"
		  // Dennis Jones 01-30-2009
		  || (cFirstCharacter == '.' && isdigit (cNextCharacter)) )
		  {
	    // skip sign for now
	    if ((cFirstCharacter == '+' || cFirstCharacter == '-'))
	      pWord_++;
	    while (isdigit (*pWord_) || *pWord_ == '.')
	      pWord_++;

	    // allow for 1.53158e+15
	    if (*pWord_ == 'e' || *pWord_ == 'E')
	      {
	      pWord_++; // skip 'e'
	      if ((*pWord_  == '+' || *pWord_  == '-'))
	        pWord_++; // skip sign after e
	      while (isdigit (*pWord_))  // now digits after e
	        pWord_++;
	      }

	    word_ = std::string (pWordStart_, pWord_ - pWordStart_);

	    std::istringstream is (word_);
	    // parse std::string into double value
	    is >> value_;

	    if (is.fail () && !is.eof ())
	      throw std::runtime_error ("Bad numeric literal: " + word_);
	    if (f_v) {
	  	  std::cout << "token NUMBER " << value_ << std::endl;
	    }

		create_double_token(value_);
	    return type_ = NUMBER;
	    }   // end of number found

	  // special test for 2-character sequences: <= >= == !=
	  // also +=, -=, /=, *=
	  if (cNextCharacter == '=')
	    {
	    switch (cFirstCharacter)
	      {
	      // comparisons
	      case '=': type_ = EQ;   break;
	      case '<': type_ = LE;   break;
	      case '>': type_ = GE;   break;
	      case '!': type_ = NE;   break;
	      // assignments
	      case '+': type_ = ASSIGN_ADD;   break;
	      case '-': type_ = ASSIGN_SUB;   break;
	      case '*': type_ = ASSIGN_MUL;   break;
	      case '/': type_ = ASSIGN_DIV;   break;
	      // none of the above
	      default:  type_ = NONE; break;
	      } // end of switch on cFirstCharacter

	    if (type_ != NONE)
	      {
	      word_ = std::string (pWordStart_, 2);
	      pWord_ += 2;   // skip both characters
	      if (f_v) {
	    	  std::cout << "token operator ";
	    	  print_token(std::cout, type_);
	    	  std::cout << std::endl;
	      }
		  create_text_token(word_);
	      return type_;
	      } // end of found one
	    } // end of *=

	  switch (cFirstCharacter)
	    {
	    case '&': if (cNextCharacter == '&')    // &&
	                {
	                word_ = std::string (pWordStart_, 2);
	                pWord_ += 2;   // skip both characters
	                if (f_v) {
	              	  std::cout << "token AND " << std::endl;
	                }
	    		    T = new syntax_tree_node_terminal;
	    		    T->f_text = true;
	    		    T->value_text.assign(word_);
	                return type_ = AND;
	                }
	              break;
	   case '|': if (cNextCharacter == '|')   // ||
	                {
	                word_ = std::string (pWordStart_, 2);
	                pWord_ += 2;   // skip both characters
	                if (f_v) {
	              	  std::cout << "token OR " << std::endl;
	                }
	    		    T = new syntax_tree_node_terminal;
	    		    T->f_text = true;
	    		    T->value_text.assign(word_);
	                return type_ = OR;
	                }
	              break;
	    // single-character symbols
	    case '=':
	    case '<':
	    case '>':
	    case '+':
	    case '-':
	    case '/':
	    case '*':
	    case '(':
	    case ')':
	    case ',':
	    case '!':
	      word_ = std::string (pWordStart_, 1);
	      ++pWord_;   // skip it
	      type_ = TokenType (cFirstCharacter);
	      if (f_v) {
	    	  std::cout << "token operator ";
	    	  print_token(std::cout, type_);
	    	  std::cout << std::endl;
	      }

		  create_text_token(word_);
	      return type_;
	    } // end of switch on cFirstCharacter

	  if (!isalpha (cFirstCharacter))
	    {
	    if (cFirstCharacter < ' ')
	      {
	      std::ostringstream s;
	      s << "Unexpected character (decimal " << int (cFirstCharacter) << ")";
	      throw std::runtime_error (s.str ());
	      }
	    else
	      throw std::runtime_error ("Unexpected character: " + std::string (1, cFirstCharacter));
	    }

	  // we have a word (starting with A-Z) - pull it out
	  while (isalnum (*pWord_) || *pWord_ == '_')
	    ++pWord_;

	  word_ = std::string (pWordStart_, pWord_ - pWordStart_);

	  if (f_v) {
		  std::cout << "token NAME " << word_ << std::endl;
	  }
	  create_text_token(word_);
	  return type_ = NAME;

}

void lexer::create_text_token(std::string &txt)
{
    if (T) {
    	delete T;
    }
    T = new syntax_tree_node_terminal;
    T->f_text = true;
    T->value_text.assign(txt);
    std::cout << "lexer::create_text_token text=" << txt << std::endl;

}

void lexer::create_double_token(double dbl)
{
    if (T) {
    	delete T;
    }
    T = new syntax_tree_node_terminal;
    T->f_double = true;
    T->value_double = dbl;
    std::cout << "lexer::create_double_token value=" << dbl << std::endl;

}



void lexer::CheckToken (TokenType wanted)
{
  if (type_ != wanted)
    {
    std::ostringstream s;
    s << "'" << static_cast <char> (wanted) << "' expected.";
    throw std::runtime_error (s.str ());
    }
}








}}


