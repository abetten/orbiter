/*
 * function_polish.cpp
 *
 *  Created on: Apr 18, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


function_polish::function_polish()
{
	Descr = NULL;
}

function_polish::~function_polish()
{
}


void function_polish::init(
		function_polish_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double val;
	int entry, len;
	int i;


	if (f_v) {
		cout << "function_polish::init" << endl;
		cout << "function_polish::nb_variables = " << Descr->nb_variables << endl;
		cout << "function_polish::nb_constants = " << Descr->nb_constants << endl;
		cout << "function_polish::nb_commands = " << Descr->code_sz << endl;
		cout << "variables:" << endl;
		for (i = 0; i < Descr->nb_variables; i++) {
			cout << i << " : " << Descr->variable_names[i] << endl;
		}
		cout << "constants:" << endl;
		for (i = 0; i < Descr->nb_constants; i++) {
			cout << i << " : " << Descr->const_names[i] << " = " << Descr->const_values[i] << endl;
		}
		cout << "commands:" << endl;
		for (i = 0; i < Descr->code_sz; i++) {
			cout << i << " : " << Descr->code[i] << endl;
		}
	}

	for (i = 0; i < Descr->nb_variables; i++) {
		string S(Descr->variable_names[i]);
		Variables.push_back(S);
	}

	for (i = 0; i < Descr->nb_constants; i++) {
		double f;

		f = strtof(Descr->const_values[i]);
		pair<string, double> P(Descr->const_names[i], f);
		Constants.push_back(P);
	}


	if (f_v) {
		cout << "function_polish::init start parsing" << endl;
	}
	entry = 0;
	len = 0;
	for (i = 0; i < Descr->code_sz; i++) {
		function_command cmd;

		len++;

		if (stringcmp(Descr->code[i], "push") == 0) {

			string S(Descr->code[++i]);

			int j;

			for (j = 0; j < (int) Variables.size(); j++) {
				if (strcmp(Variables[j].c_str(), S.c_str()) == 0) {
					break;
				}
			}
			if (j < (int) Variables.size()) {
				cmd.init_with_argument(3, j);
				if (f_v) {
					cout << "push variable " << S << " (= " << j << ")" << endl;
				}
			}
			else {
				for (j = 0; j < (int) Constants.size(); j++) {

					if (strcmp(Constants[j].first.c_str(), S.c_str()) == 0) {
						break;
					}
				}
				if (j < (int) Constants.size()) {
					cmd.init_with_argument(1, j);
					if (f_v) {
						cout << "push constant " << S << " (= " << j << ")" << endl;
					}
				}
				else {
					val = atof(S.c_str());
					cmd.init_push_immediate_constant(val);
					if (f_v) {
						cout << "push immediate " << val << endl;
					}
					//cout << "function_polish::init unknown label " << S << endl;
					//exit(1);
				}
			}
		}
		else if (stringcmp(Descr->code[i], "store") == 0) {
			int j;
			string S(Descr->code[++i]);

			for (j = 0; j < (int) Variables.size(); j++) {
				if (strcmp(Variables[j].c_str(), S.c_str()) == 0) {
					break;
				}
			}
			if (j < (int) Variables.size()) {
				cmd.init_with_argument(4, j);
				if (f_v) {
					cout << "store " << S << " (= " << j << ")" << endl;
				}
			}
			else {
				cout << "store command, cannot find variable with label " << S << endl;
			}
		}
		else if (stringcmp(Descr->code[i], "mult") == 0) {
			cmd.init_simple(5);
			if (f_v) {
				cout << "mult" << endl;
			}
		}
		else if (stringcmp(Descr->code[i], "add") == 0) {
			cmd.init_simple(6);
			if (f_v) {
				cout << "add" << endl;
			}
		}
		else if (stringcmp(Descr->code[i], "cos") == 0) {
			cmd.init_simple(7);
		}
		else if (stringcmp(Descr->code[i], "sin") == 0) {
			cmd.init_simple(8);
			if (f_v) {
				cout << "cos" << endl;
			}
		}
		else if (stringcmp(Descr->code[i], "sqrt") == 0) {
			cmd.init_simple(10);
			if (f_v) {
				cout << "sqrt" << endl;
			}
		}
		else if (stringcmp(Descr->code[i], "return") == 0) {
			cmd.init_simple(9);
			if (f_v) {
				cout << "return" << endl;
			}
			Entry.push_back(entry);
			Len.push_back(len);
			entry = Code.size() + 1; // next entry point
			len = 0;
		}
		else {
			cout << "unrecognized command " << Descr->code[i] << endl;
			exit(1);
		}
		Code.push_back(cmd);
	}

	if (f_v) {
		cout << "function_polish::init Parsed " << Entry.size() << " functions" << endl;

		print_code_complete(verbose_level);

	}


	if (f_v) {
		cout << "function_polish::init done" << endl;
	}
}

void function_polish::print_code_complete(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "function_polish::print_code_complete" << endl;
	}

	int h, i0, len;

	for (h = 0; h < Entry.size(); h++) {
		i0 = Entry[h];
		len = Len[h];
		cout << "subroutine " << h << " starts at " << i0 << " and has length " << len << ":" << endl;
		print_code(i0, len, verbose_level);
	}
}

void function_polish::print_code(
		int i0,  int len,
		int verbose_level)
{
	int i, j, t;
	double f;

	for (j = 0; j < len; j++) {
		i = i0 + j;
		if (i >= Code.size()) {
			break;
		}
		t = Code[i].type;
		if (t == 1) {
			// push a labeled constant
			cout << i << " : push " << Constants[Code[i].arg].first << endl;
		}
		else if (t == 2) {
			// push a immediate constant
			f = Code[i].val;
			cout << i << " : push " << f << endl;
		}
		else if (t == 3) {
			// push a variable
			cout << i << " : push " << Variables[Code[i].arg] << endl;
		}
		else if (t == 4) {
			// store to a variable, pop the stack
			cout << i << " : store " << Variables[Code[i].arg] << endl;
		}
		else if (t == 5) {
			// mult
			cout << i << " : mult" << endl;
		}
		else if (t == 6) {
			// add
			cout << i << " : add" << endl;
		}
		else if (t == 7) {
			// cos
			cout << i << " : cos" << endl;
		}
		else if (t == 8) {
			// sin
			cout << i << " : sin" << endl;
		}
		else if (t == 10) {
			// sqrt
			cout << i << " : sqrt" << endl;
		}
		else if (t == 9) {
			cout << i << " : return" << endl;
		}
		else {
			cout << "unknown type, error. t = " << t << endl;
			exit(1);
		}
	} //next j
}

void function_polish::evaluate(
		double *variable_values,
		double *output_values,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	vector<double > Stack;
	int h, fst, len, i, j, t;
	double f, a, b;

	if (f_v) {
		cout << "function_polish::evaluate" << endl;
	}
	for (h = 0; h < (int) Entry.size(); h++) {
		fst = Entry[h];
		len = Len[h];
		if (f_v) {
			cout << "function_polish::evaluate evaluation function " << h << " / " << Entry.size() << endl;
			cout << "function_polish::evaluate fst=" << fst << endl;
			cout << "function_polish::evaluate len=" << len << endl;
		}
		if (Stack.size() != 0) {
			cout << "stack must be empty before we start" << endl;
			exit(1);
		}
		for (j = 0; j < len; j++) {
			if (f_v) {
				cout << "h=" << h << " j=" << j << " : ";
			}
			i = fst + j;
			t = Code[i].type;
			if (t == 1) {
				// push a labeled constant
				f = Constants[Code[i].arg].second;
				if (f_v) {
					cout << "pushing constant " << f << " from constant " << Code[i].arg << endl;
				}
				Stack.push_back(f);
			}
			else if (t == 2) {
				// push a immediate constant
				f = Code[i].val;
				if (f_v) {
					cout << "pushing immediate constant " << f << endl;
				}
				Stack.push_back(f);
			}
			else if (t == 3) {
				// push a variable
				f = variable_values[Code[i].arg];
				if (f_v) {
					cout << "pushing variable " << f << " from variable " << Code[i].arg << endl;
				}
				Stack.push_back(f);
			}
			else if (t == 4) {
				// store to a variable, pop the stack
				f = Stack[Stack.size() - 1];
				variable_values[Code[i].arg] = f;
				Stack.pop_back();
				if (f_v || TRUE) {
					cout << "storing value " << f << " to variable " << Variables[Code[i].arg] << " : stacksize = " << Stack.size() << endl;
				}
			}
			else if (t == 5) {
				// mult
				if (Stack.size() < 2) {
					cout << "multiplication needs at least two elements on the stack; stack size = " << Stack.size() << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				a = Stack[Stack.size() - 1];
				Stack.pop_back();
				b = Stack[Stack.size() - 1];
				Stack.pop_back();
				f = a * b;
				if (f_v) {
					cout << "mult: " << a << " * " << b << " = " << f << endl;
				}
				Stack.push_back(f);
			}
			else if (t == 6) {
				// add
				if (Stack.size() < 2) {
					cout << "addition needs at least two elements on the stack; stack size = " << Stack.size() << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				a = Stack[Stack.size() - 1];
				Stack.pop_back();
				b = Stack[Stack.size() - 1];
				Stack.pop_back();
				f = a + b;
				if (f_v) {
					cout << "add: " << a << " + " << b << " = " << f << endl;
				}
				Stack.push_back(f);
			}
			else if (t == 7) {
				// cos
				if (Stack.size() < 1) {
					cout << "cos needs at least one element on the stack; stack size = " << Stack.size() << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				a = Stack[Stack.size() - 1];
				f = cos(a);
				if (f_v) {
					cout << "cos " << a << " = " << f << endl;
				}
				Stack[Stack.size() - 1] = f;
			}
			else if (t == 8) {
				// sin
				if (Stack.size() < 1) {
					cout << "sin needs at least one element on the stack; stack size = " << Stack.size() << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				a = Stack[Stack.size() - 1];
				f = sin(a);
				if (f_v) {
					cout << "sin " << a << " = " << f << endl;
				}
				Stack[Stack.size() - 1] = f;
			}
			else if (t == 10) {
				// sqrt
				if (Stack.size() < 1) {
					cout << "sin needs at least one element on the stack; stack size = " << Stack.size() << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				a = Stack[Stack.size() - 1];
				if (a < 0) {
					cout << "sqrt: the argument is negative, a = " << a << endl;
					print_code(fst + j,  20, verbose_level);
					exit(1);
				}
				f = sqrt(a);
				if (f_v) {
					cout << "sqrt " << a << " = " << f << endl;
				}
				Stack[Stack.size() - 1] = f;
			}
			else if (t == 9) {
				if (f_v) {
					cout << "return, stacksize = " << Stack.size() << endl;
				}
				if (Stack.size() != 1) {
					cout << "Stack size is not 1 at the end of the execution of function " << h << endl;
					exit(1);
				}
				output_values[h] = Stack[0];
				Stack.pop_back();
				break;
			}
			else {
				cout << "unknown type, error. t = " << t << endl;
				exit(1);
			}
		} //next j
		if (f_v) {
			cout << "function_polish::evaluate function " << h << " / " << Entry.size()
					<< " evaluates to " << output_values[h] << endl;
		}
	} // next h

	if (f_v) {
		for (h = 0; h < (int) Entry.size(); h++) {
			cout << "function " << h << " evaluates to " << output_values[h] << endl;
		}
	}
	if (f_v) {
		cout << "function_polish::evaluate done" << endl;
	}
}

}}
