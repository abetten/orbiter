/*
 * expression_parser.cpp
 *
 *  Created on: Feb 14, 2021
 *      Author: betten
 */

/*

 Parser - an expression parser

 Author:  Nick Gammon
          http://www.gammon.com.au/

(C) Copyright Nick Gammon 2004. Permission to copy, use, modify, sell and
distribute this software is granted provided this copyright notice appears
in all copies. This software is provided "as is" without express or implied
warranty, and with no claim as to its suitability for any purpose.

Modified 24 October 2005 by Nick Gammon.

  1. Changed use of "abs" to "fabs"
  2. Changed inclues from math.h and time.h to fmath and ftime
  3. Rewrote DoMin and DoMax to inline the computation because of some problems with some libraries.
  4. Removed "using namespace std;" and put "std::" in front of std namespace names where appropriate
  5. Removed MAKE_STRING macro and inlined the functionality where required.
  6. Changed Evaluate function to take its argument by reference.

Modified 13 January 2010 by Nick Gammon.

  1. Changed getrandom to work more reliably (see page 2 of discussion thread)
  2. Changed recognition of numbers to allow for .5 (eg. "a + .5" where there is no leading 0)
     Also recognises -.5 (so you don't have to write -0.5)
  3. Fixed problem where (2+3)-1 would not parse correctly (- sign directly after parentheses)
  4. Fixed problem where changing a parameter and calling p.Evaluate again would fail because the
     initial token type was not reset to NONE.

Modified 16 February 2010 by Nick Gammon

  1. Fixed bug where if you called Evaluate () twice, the original expression would not be reprocessed.

Modified 27 November 2014 by Nick Gammon

  1. Fixed bug where a literal number followed by EOF would throw an error.

Thanks to various posters on my forum for suggestions. The relevant post is currently at:

  http://www.gammon.com.au/forum/?id=4649

*/

/*

Expression-evaluator
--------------------

Author: Nick Gammon
-------------------


Example usage:

    Parser p ("2 + 2 * (3 * 5) + nick");

    p.symbols_ ["nick"] = 42;

    double v = p.Evaluate ();

    double v1 = p.Evaluate ("5 + 6");   // supply new expression and evaluate it

Syntax:

  You can use normal algebraic syntax.

  Multiply and divide has higher precedence than add and subtract.

  You can use parentheses (eg. (2 + 3) * 5 )

  Variables can be assigned, and tested. eg. a=24+a*2

  Variables can be preloaded:

    p.symbols_ ["abc"] = 42;
    p.symbols_ ["def"] = 42;

  Afterwards they can be retrieved:

    x = p.symbols_ ["abc"];

  There are 2 predefined symbols, "pi" and "e".

  You can use the comma operator to load variables and then use them, eg.

    a=42, b=a+6

  You can use predefined functions, see below for examples of writing your own.

    42 + sqrt (64)


  Comparisons
  -----------

  Comparisons work by returning 1.0 if true, 0.0 if false.

  Thus, 2 > 3 would return 0.0
        3 > 2 would return 1.0

  Similarly, tests for truth (eg. a && b) test whether the values are 0.0 or not.

  If test
  -------

  There is a ternary function: if (truth-test, true-value, false-value)

  eg.  if (1 < 2, 22, 33)  returns 22


  Precedence
  ----------

  ( )  =   - nested brackets, including function calls like sqrt (x), and assignment
  * /      - multiply, divide
  + -      - add and subtract
  < <= > >= == !=  - comparisons
  && ||    - AND and OR
  ,        - comma operator

    Credits:

    Based in part on a simple calculator described in "The C++ Programming Language"
    by Bjarne Stroustrup, however with considerable enhancements by me, and also based
    on my earlier experience in writing Pascal compilers, which had a similar structure.

*/


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace expression_parser {



typedef double (*OneArgFunction)  (double arg);
typedef const double (*TwoArgFunction)  (const double arg1, const double arg2);
typedef const double (*ThreeArgFunction)  (const double arg1, const double arg2, const double arg3);

// maps of function names to functions
static std::map<std::string, OneArgFunction>    OneArgumentFunctions;
static std::map<std::string, TwoArgFunction>    TwoArgumentFunctions;
static std::map<std::string, ThreeArgFunction>  ThreeArgumentFunctions;

// for standard library functions
#define STD_FUNCTION(arg) OneArgumentFunctions [#arg] = arg

static int LoadOneArgumentFunctions ()
  {
  OneArgumentFunctions ["abs"] = fabs;
  STD_FUNCTION (acos);
  STD_FUNCTION (asin);
  STD_FUNCTION (atan);
#ifndef WIN32   // doesn't seem to exist under Visual C++ 6
  STD_FUNCTION (atanh);
#endif
  STD_FUNCTION (ceil);
  STD_FUNCTION (cos);
  STD_FUNCTION (cosh);
  STD_FUNCTION (exp);
  STD_FUNCTION (exp);
  STD_FUNCTION (floor);
  STD_FUNCTION (log);
  STD_FUNCTION (log10);
  STD_FUNCTION (sin);
  STD_FUNCTION (sinh);
  STD_FUNCTION (sqrt);
  STD_FUNCTION (tan);
  STD_FUNCTION (tanh);

#if 0
  OneArgumentFunctions ["int"] = DoInt;
  OneArgumentFunctions ["rand"] = DoRandom;
  OneArgumentFunctions ["rand"] = DoRandom;
  OneArgumentFunctions ["percent"] = DoPercent;
#endif
  return 0;
  } // end of LoadOneArgumentFunctions

static int LoadTwoArgumentFunctions ()
  {
#if 0
  TwoArgumentFunctions ["min"]  = DoMin;
  TwoArgumentFunctions ["max"]  = DoMax;
  TwoArgumentFunctions ["mod"]  = DoFmod;
  TwoArgumentFunctions ["pow"]  = DoPow;     //   x to the power y
  TwoArgumentFunctions ["roll"] = DoRoll;   // dice roll
#endif
  return 0;
  } // end of LoadTwoArgumentFunctions

static int LoadThreeArgumentFunctions ()
  {
#if 0
 ThreeArgumentFunctions ["if"]  = DoIf;
#endif
  return 0;
  } // end of LoadThreeArgumentFunctions


expression_parser::expression_parser()
{
	Lexer = NULL;
	Tree = NULL;
    // insert pre-defined names:
    symbols_ ["pi"] = 3.1415926535897932385;
    symbols_ ["e"]  = 2.7182818284590452354;
    LoadOneArgumentFunctions ();
    LoadTwoArgumentFunctions ();
    LoadThreeArgumentFunctions ();
}

expression_parser::~expression_parser()
{
	if (Lexer) {
		FREE_OBJECT(Lexer);
	}
}

#if 0
double expression_parser::Function(int verbose_level, std::string &word)
{
	int f_v = (verbose_level >= 1);

	syntax_tree_node_terminal *T;

	// might be single-argument function (eg. abs (x) )
    std::map<std::string, OneArgFunction>::const_iterator si;
    si = OneArgumentFunctions.find (word);
    if (si != OneArgumentFunctions.end ())
      {
		if (f_v) {
			std::cout << "expression_parser::Primary one argument function" << std::endl;
		}
      double v = Expression (verbose_level, true);   // get argument
      Lexer.CheckToken (RHPAREN);
      Lexer.GetToken (verbose_level, true);        // get next one (one-token lookahead)
		if (f_v) {
			std::cout << "expression_parser::Primary one argument function done" << std::endl;
		}
      return si->second (v);  // evaluate function
      }

    // might be double-argument function (eg. roll (6, 2) )
    std::map<std::string, TwoArgFunction>::const_iterator di;
    di = TwoArgumentFunctions.find (word);
    if (di != TwoArgumentFunctions.end ())
      {
		if (f_v) {
			std::cout << "expression_parser::Primary two argument function" << std::endl;
		}
      double v1 = Expression (verbose_level, true);   // get argument 1 (not commalist)
      Lexer.CheckToken (COMMA);
      double v2 = Expression (verbose_level, true);   // get argument 2 (not commalist)
      Lexer.CheckToken (RHPAREN);
      Lexer.GetToken (verbose_level, true);            // get next one (one-token lookahead)
		if (f_v) {
			std::cout << "expression_parser::Primary two argument function dibe" << std::endl;
		}
      return di->second (v1, v2); // evaluate function
      }

   // might be double-argument function (eg. roll (6, 2) )
    std::map<std::string, ThreeArgFunction>::const_iterator ti;
    ti = ThreeArgumentFunctions.find (word);
    if (ti != ThreeArgumentFunctions.end ())
      {
		if (f_v) {
			std::cout << "expression_parser::Primary three argument function" << std::endl;
		}
      double v1 = Expression (verbose_level, true);   // get argument 1 (not commalist)
      Lexer.CheckToken (COMMA);
      double v2 = Expression (verbose_level, true);   // get argument 2 (not commalist)
      Lexer.CheckToken (COMMA);
      double v3 = Expression (verbose_level, true);   // get argument 3 (not commalist)
      Lexer.CheckToken (RHPAREN);
      Lexer.GetToken (verbose_level, true);  // get next one (one-token lookahead)
		if (f_v) {
			std::cout << "expression_parser::Primary three argument function done" << std::endl;
		}
      return ti->second (v1, v2, v3); // evaluate function
      }

    throw std::runtime_error ("Function '" + word + "' not implemented.");

}
#endif


syntax_tree_node *expression_parser::Primary (int verbose_level,
		int &f_single_literal, std::string &single_literal, int &f_has_seen_minus, const bool get)
  {
	int f_v = (verbose_level >= 1);

	syntax_tree_node *N;


	if (f_v) {
		std::cout << "expression_parser::Primary" << std::endl;
	}
	f_single_literal = false;
	f_has_seen_minus = false;

	N = new syntax_tree_node;
	N->Tree = Tree;
	if (f_v) {
		std::cout << "expression_parser::Primary opening node " << N->idx << std::endl;
	}

  if (get)
    Lexer->GetToken(verbose_level);    // one-token lookahead

  if (Lexer->type_ == NUMBER) {
      double v = Lexer->value_;
		N->f_terminal = true;
		N->T = Lexer->T;
		Lexer->T = 0L;
      Lexer->GetToken (verbose_level, true);  // get next one (one-token lookahead)
		if (f_v) {
			std::cout << "expression_parser::Primary NUMBER " << v << " done" << std::endl;
		}
		if (f_v) {
			N->print(std::cout);
		}
      return N;
  }
  else if (Lexer->type_ == NAME) {
      std::string word = Lexer->word_;

		N->f_terminal = true;
		N->T = Lexer->T;
		Lexer->T = 0L;

		Lexer->GetToken (verbose_level, true);
      if (Lexer->type_ == LHPAREN)
        {
    	  exit(1);
    	  //return Function(verbose_level, word);
        }

      // not a function? must be a symbol in the symbol table
      f_single_literal = true;
      single_literal.assign(word);
      //double & v = symbols_ [word];  // get REFERENCE to symbol table entry
      // change table entry with expression? (eg. a = 22, or a = 22)
#if 0
      switch (Lexer.type_)
        {
        // maybe check for NaN or Inf here (see: isinf, isnan functions)
        case ASSIGN:     v  = Expression (verbose_level, true); break;
        case ASSIGN_ADD: v += Expression (verbose_level, true); break;
        case ASSIGN_SUB: v -= Expression (verbose_level, true); break;
        case ASSIGN_MUL: v *= Expression (verbose_level, true); break;
        case ASSIGN_DIV:
            {
        		if (f_v) {
        			std::cout << "Parser::Primary assignment" << std::endl;
        		}
            double d = Expression (verbose_level, true);
            if (d == 0.0)
              throw std::runtime_error ("Divide by zero");
            v /= d;
            break;   // change table entry with expression
            } // end of ASSIGN_DIV
        default: break;   // do nothing for others
        } // end of switch on type_
#endif
		if (f_v) {
			std::cout << "expression_parser::Primary symbol " << word << std::endl;
		}
		if (f_v) {
			N->print(std::cout);
		}
 		return N;               // and return new value
      }
  else if (Lexer->type_ == MINUS) {
    	// unary minus
		if (f_v) {
			std::cout << "expression_parser::Primary unary minus" << std::endl;
		}
		delete N;
		N = Primary(verbose_level, f_single_literal, single_literal, f_has_seen_minus, true);
		if (f_has_seen_minus) {
			std::cout << "expression_parser::Primary double minus" << std::endl;
			exit(1);
		}
		f_has_seen_minus = true;
		return N;
  }
#if 0
  else if (Lexer.type_ == NOT) {
    // unary not
		if (f_v) {
			std::cout << "expression_parser::Primary unary not" << std::endl;
		}
		f_single_literal = true;
      double val2 = (Primary (verbose_level, f_single_literal, single_literal, true) == 0.0) ? 1.0 : 0.0;
      return N;
  }
#endif
  else if (Lexer->type_ == LHPAREN) {
  		if (f_v) {
  			std::cout << "expression_parser::Primary left hand parenthesis, calling CommaList" << std::endl;
  		}
  		syntax_tree_node * N1 = CommaList (verbose_level, true);    // inside parens, you could have commas
		if (f_v) {
			std::cout << "expression_parser::Primary left hand parenthesis, after CommaList" << std::endl;
		}
		Lexer->CheckToken (RHPAREN);
      Lexer->GetToken (verbose_level, true);                // eat the ')'
		if (f_v) {
			std::cout << "expression_parser::Primary left hand parenthesis done" << std::endl;
		}
		delete N;
		if (f_v) {
			N1->print(std::cout);
		}
      return N1;
   }
  else {
	  std::string remainder;

	  remainder.assign(Lexer->pWord_);
      throw std::runtime_error ("Unexpected token: " + Lexer->word_ + " at " + remainder);
   } // end of if on type

  } // end of Parser::Primary

#if 0
syntax_tree_node *expression_parser::Term (int verbose_level, const bool get)
// multiply and divide
  {
	int f_v = (verbose_level >= 1);
	int f_single_literal;
	std::string single_literal;
	int *monomial;
	int i;
	syntax_tree_node *N;
	int f_has_seen_minus;
	int nb_minus_signs = 0;

	if (f_v) {
		std::cout << "expression_parser::Term" << std::endl;
	}

	if (Tree->managed_variables.size()) {
		monomial = NEW_int(Tree->managed_variables.size());
		int_vec_zero(monomial, Tree->managed_variables.size());
	}

	N = new syntax_tree_node;
	N->Tree = Tree;
	N->type = operation_type_mult;
	if (f_v) {
		std::cout << "expression_parser::Term opening node " << N->idx << std::endl;
	}
	if (f_v) {
		std::cout << "expression_parser::Term before Primary" << std::endl;
	}
	syntax_tree_node *left = Primary (verbose_level, f_single_literal, single_literal, f_has_seen_minus, get);
	if (f_has_seen_minus) {
		nb_minus_signs++;
	}
	if (f_v) {
		std::cout << "expression_parser::Term after Primary, f_single_literal = " << f_single_literal << std::endl;
	}
	int f_done;
	f_done = false;
	if (Tree->managed_variables.size() && f_single_literal) {
		i = Tree->identify_single_literal(single_literal);
		if (i >= 0) {
			monomial[i]++;
			delete left;
			f_done = true;
		}
	}
	if (!f_done) {
		N->nb_nodes = 1;
		N->Nodes[0] = left;
		if (f_v) {
			std::cout << "expression_parser::Term first descendant of " << N->idx << " is node " << N->Nodes[0]->idx << std::endl;
		}
	}
  while (true)
    {
    switch (Lexer.type_)
      {
      case MULTIPLY:
    		if (f_v) {
    			std::cout << "expression_parser::Term MULTIPLY, calling Primary" << std::endl;
    		}
    	syntax_tree_node * N2;
        N2 = Primary (verbose_level, f_single_literal, single_literal, f_has_seen_minus, true);
		if (f_v) {
			std::cout << "Parser::Term MULTIPLY, after Primary, f_single_literal=" << f_single_literal << std::endl;
		}
		if (f_has_seen_minus) {
			nb_minus_signs++;
		}
		f_done = false;
		if (f_single_literal) {
			std::cout << "single_literal = " << single_literal << std::endl;
			if (Tree->managed_variables.size() && f_single_literal) {
				i = Tree->identify_single_literal(single_literal);
				if (i >= 0) {
					monomial[i]++;
					delete N2;
					f_done = true;
				}
			}
		}
		if (!f_done) {
			std::cout << "not a single_literal, N->nb_nodes=" << N->nb_nodes << std::endl;
			{
				N->Nodes[N->nb_nodes] = N2;
				N->nb_nodes++;
				if (ODD(nb_minus_signs)) {
					N->f_has_minus = TRUE;
				}
			}
		}
		break;
#if 0
      case DIVIDE:
          {
      		if (f_v) {
      			std::cout << "expression_parser::Term DIVIDE, calling Primary" << std::endl;
      		}
          double d = Primary (verbose_level, f_single_literal, single_literal, true);
    		if (f_v) {
    			std::cout << "expression_parser::Term DIVIDE, after Primary" << std::endl;
    		}
          if (d == 0.0)
            throw std::runtime_error ("Divide by zero");
          left /= d;
          break;
          }
#endif
      default:
    	 if (Tree->managed_variables.size()) {
    		 N->f_has_monomial = TRUE;
    		 N->monomial = monomial;
    	 }
 		if (f_v) {
  			std::cout << "expression_parser::Term before return, ";
  	   		N->print_without_recursion(std::cout);
  			if (N->f_has_monomial) {
  				Tree->print_monomial(std::cout, N->monomial);
  			}
  			std::cout << std::endl;
  		}
 		return N;
      } // end of switch on type
    }   // end of loop
	if (f_v) {
		std::cout << "expression_parser::Term done" << std::endl;
	}
  } // end of Parser::Term
#else
syntax_tree_node *expression_parser::Term (int verbose_level, const bool get)
// multiply and divide
  {
	int f_v = (verbose_level >= 1);
	int f_single_literal;
	std::string single_literal;
	int *monomial;
	int i;
	syntax_tree_node *N;
	int f_has_seen_minus;
	int nb_minus_signs = 0;

	if (f_v) {
		std::cout << "expression_parser::Term" << std::endl;
	}

	monomial = NEW_int(Tree->managed_variables.size());
	Int_vec_zero(monomial, Tree->managed_variables.size());

	N = new syntax_tree_node;
	N->Tree = Tree;
	N->type = operation_type_mult;
	if (f_v) {
		std::cout << "expression_parser::Term opening node " << N->idx << std::endl;
	}
	if (f_v) {
		std::cout << "expression_parser::Term before Primary" << std::endl;
	}
	syntax_tree_node *left = Primary (verbose_level, f_single_literal, single_literal, f_has_seen_minus, get);
	if (f_has_seen_minus) {
		nb_minus_signs++;
	}
	if (f_v) {
		std::cout << "expression_parser::Term after Primary, f_single_literal = " << f_single_literal << std::endl;
	}
	int f_done;
	f_done = false;
	if (f_single_literal) {
		i = Tree->identify_single_literal(single_literal);
		if (i >= 0) {
			monomial[i]++;
			delete left;
			f_done = true;
		}
	}
	if (!f_done) {
		N->nb_nodes = 1;
		N->Nodes[0] = left;
		if (f_v) {
			std::cout << "expression_parser::Term first descendant of " << N->idx << " is node " << N->Nodes[0]->idx << std::endl;
		}
	}
  while (true)
    {
    switch (Lexer->type_)
      {
      case MULTIPLY:
    		if (f_v) {
    			std::cout << "expression_parser::Term MULTIPLY, calling Primary" << std::endl;
    		}
    	syntax_tree_node * N2;
        N2 = Primary (verbose_level, f_single_literal, single_literal, f_has_seen_minus, true);
		if (f_v) {
			std::cout << "Parser::Term MULTIPLY, after Primary, f_single_literal=" << f_single_literal << std::endl;
		}
		if (f_has_seen_minus) {
			nb_minus_signs++;
		}
		f_done = false;
		if (f_single_literal) {
			if (f_v) {
				std::cout << "single_literal = " << single_literal << std::endl;
			}
			if (f_single_literal) {
				i = Tree->identify_single_literal(single_literal);
				if (i >= 0) {
					monomial[i]++;
					delete N2;
					f_done = true;
				}
			}
		}
		if (!f_done) {
			if (f_v) {
				std::cout << "not a single_literal, N->nb_nodes=" << N->nb_nodes << std::endl;
			}
			{
				N->Nodes[N->nb_nodes] = N2;
				N->nb_nodes++;
				if (ODD(nb_minus_signs)) {
					N->f_has_minus = TRUE;
				}
			}
		}
		break;
#if 0
      case DIVIDE:
          {
      		if (f_v) {
      			std::cout << "expression_parser::Term DIVIDE, calling Primary" << std::endl;
      		}
          double d = Primary (verbose_level, f_single_literal, single_literal, true);
    		if (f_v) {
    			std::cout << "expression_parser::Term DIVIDE, after Primary" << std::endl;
    		}
          if (d == 0.0)
            throw std::runtime_error ("Divide by zero");
          left /= d;
          break;
          }
#endif
      default:
    	N->monomial = monomial;
    	N->f_has_monomial = TRUE;
 		if (f_v) {
  			std::cout << "expression_parser::Term before return ";
  			if (N->f_has_monomial) {
  				Tree->print_monomial(std::cout, N->monomial);
  			}
  			std::cout << std::endl;
  			std::cout << "expression_parser::Term created the following node:" << endl;
  	   		N->print_without_recursion(std::cout);
  			std::cout << "expression_parser::Term done" << endl;
  		}
 		return N;
      } // end of switch on type
    }   // end of loop
	if (f_v) {
		std::cout << "expression_parser::Term done" << std::endl;
	}
  } // end of Parser::Term

#endif

syntax_tree_node *expression_parser::AddSubtract (int verbose_level, const bool get)
// add and subtract
  {
	int f_v = (verbose_level >= 1);
	syntax_tree_node *N;

	if (f_v) {
		std::cout << "expression_parser::AddSubtract" << std::endl;
	}
	N = new syntax_tree_node;
	N->type = operation_type_add;
	N->Tree = Tree;
	if (f_v) {
		std::cout << "expression_parser::AddSubtract opening node " << N->idx << std::endl;
	}
	if (f_v) {
		std::cout << "expression_parser::AddSubtract before Term" << std::endl;
	}
	syntax_tree_node *left = Term (verbose_level, get);
	N->nb_nodes = 1;
	N->Nodes[0] = left;
	if (f_v) {
		std::cout << "expression_parser::AddSubtract after Term" << std::endl;
	}
  while (true)
    {
    switch (Lexer->type_)
      {
      case PLUS:
    		if (f_v) {
    			std::cout << "expression_parser::AddSubtract PLUS before Term" << std::endl;
    		}
    		//syntax_tree_node *N2;
    		left = Term (verbose_level, true);
    		{
				N->Nodes[N->nb_nodes] = left;
				N->nb_nodes++;
    		}
		if (f_v) {
			std::cout << "expression_parser::AddSubtract PLUS after Term" << std::endl;
		}
   	  break;
      case MINUS:
  		if (f_v) {
  			std::cout << "expression_parser::AddSubtract MINUS before Term" << std::endl;
  		}
    	  left = Term (verbose_level, true);
    		if (f_v) {
    			std::cout << "expression_parser::AddSubtract MINUS after Term" << std::endl;
    		}
    		{
				N->Nodes[N->nb_nodes] = left;
				N->nb_nodes++;
		  		if (f_v) {
		  			std::cout << "expression_parser::AddSubtract pushing a minus" << std::endl;
		  		}
				left->push_a_minus_sign();
    		}
    	  break;
      default:
    		if (f_v) {
    			std::cout << "expression_parser::AddSubtract before return" << std::endl;
    		}
    		if (f_v) {
    			N->print_without_recursion(std::cout);
    		}
    		return N;
      } // end of switch on type
    }   // end of loop
	if (f_v) {
		std::cout << "expression_parser::AddSubtract done" << std::endl;
	}
  } // end of Parser::AddSubtract

syntax_tree_node *expression_parser::Comparison (int verbose_level, const bool get)  // LT, GT, LE, EQ etc.
  {
	syntax_tree_node *left = AddSubtract (verbose_level, get);
#if 0
  while (true)
    {
    switch (Lexer.type_)
      {
      case LT:  left = left <  AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
      case GT:  left = left >  AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
      case LE:  left = left <= AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
      case GE:  left = left >= AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
      case EQ:  left = left == AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
      case NE:  left = left != AddSubtract (verbose_level, true) ? 1.0 : 0.0; break;
           default:    return left;
      } // end of switch on type
    }   // end of loop
#endif
  return left;
  } // end of Parser::Comparison

syntax_tree_node *expression_parser::Expression (int verbose_level, const bool get)  // AND and OR
  {
	int f_v = (verbose_level >= 1);
	//syntax_tree_node *N;

	if (f_v) {
		std::cout << "expression_parser::Expression" << std::endl;
	}
	if (f_v) {
		std::cout << "expression_parser::Expression before Comparison" << std::endl;
	}
	syntax_tree_node *left = Comparison (verbose_level, get);
	if (f_v) {
		std::cout << "expression_parser::Expression after Comparison" << std::endl;
	}
#if 0
	if (f_v) {
		std::cout << "expression_parser::Expression after Comparison" << std::endl;
	}
  while (true)
    {
    switch (Lexer.type_)
      {
      case AND:
            {
            	if (f_v) {
            		std::cout << "expression_parser::Expression AND before Comparison" << std::endl;
            	}
            syntax_tree_node *N2 = Comparison (verbose_level, true);   // don't want short-circuit evaluation
        	if (f_v) {
        		std::cout << "expression_parser::Expression AND after Comparison" << std::endl;
        	}
            //left = (left != 0.0) && (d != 0.0);
        	N = new syntax_tree_node;
        	N->nb_nodes = 2;
        	N->Nodes = new syntax_tree_node [2];
        	N->Nodes[0] = *left;
        	left->null();
        	delete left;
            }
          break;
      case OR:
            {
            	if (f_v) {
            		std::cout << "expression_parser::Expression OR before Comparison" << std::endl;
            	}
            syntax_tree_node *N2 = Comparison (verbose_level, true);   // don't want short-circuit evaluation
        	if (f_v) {
        		std::cout << "expression_parser::Expression OR after Comparison" << std::endl;
        	}
            //left = (left != 0.0) || (d != 0.0);
            }
          break;
      default:
      	if (f_v) {
      		std::cout << "expression_parser::Expression before return" << std::endl;
      	}
    	  return left;
      } // end of switch on type
    }   // end of loop
#endif

  	  if (f_v) {
  		  std::cout << "expression_parser::Expression:" << std::endl;
  		  left->print_without_recursion(std::cout);
  	  }


	return left;
  } // end of Parser::Expression

syntax_tree_node *expression_parser::CommaList (int verbose_level, const bool get)  // expr1, expr2
  {
	int f_v = (verbose_level >= 1);

	if (f_v) {
		std::cout << "expression_parser::CommaList" << std::endl;
	}
	if (f_v) {
		std::cout << "expression_parser::CommaList before Expression" << std::endl;
	}
	syntax_tree_node *left = Expression (verbose_level, get);
	if (f_v) {
		std::cout << "expression_parser::CommaList after Expression" << std::endl;
	}

	if (f_v) {
		std::cout << "expression_parser::CommaList:" << std::endl;
		left->print_without_recursion(std::cout);
	}
	return left;
#if 0
  while (true)
    {
    switch (Lexer.type_)
      {
      case COMMA:
    		if (f_v) {
    			std::cout << "expression_parser::CommaList COMMA, before Expression" << std::endl;
    		}
    	  left = Expression (verbose_level, true);
    		if (f_v) {
    			std::cout << "expression_parser::CommaList COMMA, after Expression" << std::endl;
    		}
    	  break; // discard previous value
      default:
    		if (f_v) {
    			std::cout << "expression_parser::CommaList done" << std::endl;
    		}
    	  return left;
      } // end of switch on type
    }   // end of loop
#endif
  } // end of Parser::CommaList

void expression_parser::parse (syntax_tree *tree, std::string & program, int verbose_level)
  {
	int f_v = (verbose_level >= 1);

	if (f_v) {
		std::cout << "expression_parser::parse" << std::endl;
	}
	Lexer = NEW_OBJECT(lexer);
	Tree = tree;
	Lexer->program_ = program;
	Lexer->pWord_    = Lexer->program_.c_str ();
	Lexer->type_     = NONE;
	if (f_v) {
		std::cout << "expression_parser::parse before CommaList" << std::endl;
	}
	Tree->Root = CommaList (verbose_level, true);
	if (f_v) {
		std::cout << "expression_parser::parse after CommaList" << std::endl;
	}
	if (Lexer->type_ != END)
		throw std::runtime_error ("Unexpected text at end of expression: " + std::string (Lexer->pWordStart_));
	}


}}}


