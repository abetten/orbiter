/*
 * variety_description.cpp
 *
 *  Created on: Jul 13, 2024
 *      Author: betten
 */






#include "foundations.h"


using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {



variety_description::variety_description()
{
	f_label_txt = false;
	//std::string label_txt;

	f_label_tex = false;
	//std::string label_tex;

	f_projective_space = false;
	//std::string projective_space_label;

	f_has_projective_space_pointer = false;
	Projective_space_pointer = NULL;

	f_ring = false;
	//std::string ring_label;

	f_has_ring_pointer = false;
	Ring_pointer = NULL;


	f_has_equation_in_algebraic_form = false;
	//std::string equation_in_algebraic_form_text;

	f_has_equation_by_coefficients = false;
	//std::string equation_by_coefficients_text;

	f_has_second_equation_in_algebraic_form = false;
	//std::string second_equation_in_algebraic_form_text;

	f_has_second_equation_by_coefficients = false;
	//std::string second_equation_by_coefficients_text;

	f_has_points = false;
	//std::string points_txt;

	f_has_bitangents = false;
	//std::string bitangents_txt;

	//std::vector<std::string> transformations;
}



variety_description::~variety_description()
{
}


int variety_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i = 0;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "variety_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-label_txt") == 0) {
			f_label_txt = true;
			label_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-label_txt " << label_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_tex") == 0) {
			f_label_tex = true;
			label_tex.assign(argv[++i]);
			if (f_v) {
				cout << "-label_tex " << label_tex << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-projective_space") == 0) {
			f_projective_space = true;
			projective_space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-projective_space " << projective_space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ring") == 0) {
			f_ring = true;
			ring_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ring " << ring_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-equation_in_algebraic_form") == 0) {
			f_has_equation_in_algebraic_form = true;
			equation_in_algebraic_form_text.assign(argv[++i]);
			if (f_v) {
				cout << "-equation_in_algebraic_form " << equation_in_algebraic_form_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-equation_by_coefficients") == 0) {
			f_has_equation_by_coefficients = true;
			equation_by_coefficients_text.assign(argv[++i]);
			if (f_v) {
				cout << "-equation_by_coefficients " << equation_by_coefficients_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-second_equation_in_algebraic_form") == 0) {
			f_has_second_equation_in_algebraic_form = true;
			second_equation_in_algebraic_form_text.assign(argv[++i]);
			if (f_v) {
				cout << "-second_equation_in_algebraic_form " << second_equation_in_algebraic_form_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-second_equation_by_coefficients") == 0) {
			f_has_second_equation_by_coefficients = true;
			second_equation_by_coefficients_text.assign(argv[++i]);
			if (f_v) {
				cout << "-second_equation_by_coefficients " << second_equation_by_coefficients_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-points") == 0) {
			f_has_points = true;
			points_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-points " << points_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-bitangents") == 0) {
			f_has_bitangents = true;
			bitangents_txt.assign(argv[++i]);
			if (f_v) {
				cout << "-bitangents " << bitangents_txt << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-transform") == 0) {

			string s;

			s.assign(argv[++i]);
			transformations.push_back(s);
			if (f_v) {
				cout << "-transform " << transformations[transformations.size() - 1] << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "variety_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "variety_description::read_arguments done" << endl;
	}
	return i + 1;
}

void variety_description::print()
{
	if (f_label_txt) {
		cout << "-label_txt " << label_txt << endl;
	}
	if (f_label_tex) {
		cout << "-label_tex " << label_tex << endl;
	}
	if (f_projective_space) {
		cout << "-projective_space " << projective_space_label << endl;
	}
	if (f_ring) {
		cout << "-ring " << ring_label << endl;
	}
	if (f_has_equation_in_algebraic_form) {
		cout << "-equation_in_algebraic_form " << equation_in_algebraic_form_text << endl;
	}
	if (f_has_equation_by_coefficients) {
		cout << "-equation_by_coefficients " << equation_by_coefficients_text << endl;
	}
	if (f_has_second_equation_in_algebraic_form) {
		cout << "-second_equation_in_algebraic_form " << second_equation_in_algebraic_form_text << endl;
	}
	if (f_has_second_equation_by_coefficients) {
		cout << "-second_equation_by_coefficients " << second_equation_by_coefficients_text << endl;
	}
	if (f_has_points) {
		cout << "-points " << points_txt << endl;
	}
	if (f_has_bitangents) {
		cout << "-bitangents " << bitangents_txt << endl;
	}
	int i;
	for (i = 0; i < transformations.size(); i++) {
		cout << "-transform " << transformations[i] << endl;

	}

}




}}}
