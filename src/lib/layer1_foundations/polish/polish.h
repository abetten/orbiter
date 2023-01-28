// globals.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005


#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_




namespace orbiter {
namespace layer1_foundations {
namespace polish {


// #############################################################################
// function_command.cpp
// #############################################################################

//! an individual command which is part of a function expressed in reverse polish notation

class function_command {
public:

	int type;
	// 1 = push labeled constant
	// 2 = push immediate constant
	// 3 = push variable
	// 4 = store variable
	// 5 = mult
	// 6 = add
	// 7 = cos
	// 8 = sin
	// 9 = return
	// 10 = sqrt

	int f_has_argument;
	int arg;
	double val; // for push immediate constant

	function_command();
	~function_command();
	void init_with_argument(int type, int arg);
	void init_push_immediate_constant(double val);
	void init_simple(int type);

};

// #############################################################################
// function_polish_description.cpp
// #############################################################################

//! description of a function in reverse polish notation from the command line


class function_polish_description {
public:
	int nb_constants;
	std::vector<std::string> const_names;
	std::vector<std::string> const_values;
	int nb_variables;
	std::vector<std::string> variable_names;
	int code_sz;
	std::vector<std::string> code;

	function_polish_description();
	~function_polish_description();
	int read_arguments(int argc, std::string *argv,
		int verbose_level);
	void print();
};


// #############################################################################
// function_polish.cpp
// #############################################################################

//! a set of functions in reverse polish notation

class function_polish {
public:

	function_polish_description *Descr;

	std::vector<std::string > Variables;

	std::vector<std::pair<std::string, double> > Constants;

	std::vector<int> Entry;
	std::vector<int> Len;

	std::vector<function_command> Code;


	function_polish();
	~function_polish();
	void init(
			function_polish_description *Descr,
			int verbose_level);
	void print_code_complete(
			int verbose_level);
	void print_code(
			int i0,  int len,
			int verbose_level);
	void evaluate(
			double *variable_values,
			double *output_values,
			int verbose_level);

};







}}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GLOBALS_GLOBALS_H_ */



