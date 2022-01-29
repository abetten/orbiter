/*
 * lexer.cpp
 *
 *  Created on: Feb 14, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {




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
    //std::cout << "lexer::create_text_token text=" << txt << std::endl;

}

void lexer::create_double_token(double dbl)
{
    if (T) {
    	delete T;
    }
    T = new syntax_tree_node_terminal;
    T->f_double = true;
    T->value_double = dbl;
    //std::cout << "lexer::create_double_token value=" << dbl << std::endl;

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





}}}



