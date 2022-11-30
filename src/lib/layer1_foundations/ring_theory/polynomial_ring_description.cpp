/*
 * polynomial_ring_description.cpp
 *
 *  Created on: Feb 25, 2022
 *      Author: betten
 */





#include "foundations.h"


using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {


polynomial_ring_description::polynomial_ring_description()
{

	f_field = FALSE;
	//std::string finite_field_label;

	f_homogeneous = FALSE;
	homogeneous_of_degree = 0;

	f_number_of_variables = FALSE;
	number_of_variables = 0;

	Monomial_ordering_type = t_PART;

	f_variables = FALSE;
	//std::string variables_txt;
	//std::string variables_tex;

}

polynomial_ring_description::~polynomial_ring_description()
{

}

int polynomial_ring_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "polynomial_ring_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {
		if (ST.stringcmp(argv[i], "-field") == 0) {
			f_field = TRUE;
			finite_field_label.assign(argv[++i]);
			if (f_v) {
				cout << "-field " << finite_field_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-homogeneous_of_degree") == 0) {
			f_homogeneous = TRUE;
			homogeneous_of_degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-homogeneous_of_degree " << homogeneous_of_degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-number_of_variables") == 0) {
			f_number_of_variables = TRUE;
			number_of_variables = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-number_of_variables " << number_of_variables << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-monomial_ordering_partition") == 0) {
			Monomial_ordering_type = t_PART;
			if (f_v) {
				cout << "-monomial_ordering_partition " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-monomial_ordering_lex") == 0) {
			Monomial_ordering_type = t_LEX;
			if (f_v) {
				cout << "-monomial_ordering_lex " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-variables") == 0) {
			f_variables = TRUE;
			variables_txt.assign(argv[++i]);
			variables_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-variables " << variables_txt << " " << variables_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "polynomial_ring_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "polynomial_ring_description::read_arguments done" << endl;
	}
	return i + 1;
}

void polynomial_ring_description::print()
{
	//cout << "polynomial_ring_description::print:" << endl;

	if (f_field) {
		cout << "-field " << finite_field_label << endl;
	}
	if (f_homogeneous) {
		cout << "-homogeneous_of_degree " << homogeneous_of_degree << endl;
	}
	if (f_number_of_variables) {
		cout << "-number_of_variables " << number_of_variables << endl;
	}
	if (Monomial_ordering_type == t_PART) {
		cout << "-monomial_ordering_partition" << endl;
	}
	if (Monomial_ordering_type == t_LEX) {
		cout << "-monomial_ordering_lex" << endl;
	}
	if (f_variables) {
		cout << "-variables " << variables_txt << " " << variables_tex << endl;
	}
}




}}}


