/*
 * draw_projective_curve_description.cpp
 *
 *  Created on: May 2, 2021
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace graphics {


draw_projective_curve_description::draw_projective_curve_description()
{
	f_number = FALSE;
	number = 0;

	f_file = FALSE;
	//std::string fname;

	f_animate = FALSE;
	animate_nb_of_steps = 0;

	f_animate_with_transition = FALSE;
	animate_transition_nb_of_steps = 0;

	f_title_page = FALSE;
	f_trailer_page = FALSE;


}

draw_projective_curve_description::~draw_projective_curve_description()
{

}



int draw_projective_curve_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "draw_projective_curve_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-number") == 0) {
			f_number = TRUE;
			number = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-number " << number << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname.assign(argv[++i]);
			if (f_v) {
				cout << "-file " << fname << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-animate") == 0) {
			f_animate = TRUE;
			animate_nb_of_steps = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-animate " << animate_nb_of_steps << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-animate_with_transition") == 0) {
			f_animate_with_transition = TRUE;
			animate_nb_of_steps = ST.strtoi(argv[++i]);
			animate_transition_nb_of_steps = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-animate_with_transition " << animate_nb_of_steps
						<< " " << animate_transition_nb_of_steps << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-title_page") == 0) {
			f_title_page = TRUE;
			if (f_v) {
				cout << "-title_page " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-trailer_page") == 0) {
			f_trailer_page = TRUE;
			if (f_v) {
				cout << "-trailer_page " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			if (f_v) {
				cout << "-end" << endl;
			}
			break;
		}
		else {
			cout << "draw_projective_curve_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "draw_projective_curve_description::read_arguments done" << endl;
	}
	return i + 1;
}

void draw_projective_curve_description::print()
{
	if (f_number) {
		cout << "-number " << number << endl;
	}
	if (f_file) {
		cout << "-file " << fname << endl;
	}
	if (f_animate) {
		cout << "-animate " << animate_nb_of_steps << endl;
	}
	if (f_animate_with_transition) {
		cout << "-animate_with_transition " << animate_nb_of_steps
				<< " " << animate_transition_nb_of_steps << endl;
	}
	if (f_title_page) {
		cout << "-title_page " << endl;
	}
	if (f_trailer_page) {
		cout << "-trailer_page " << endl;
	}
}



}}}


