// surface_create_description.cpp
// 
// Anton Betten
//
// December 8, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


surface_create_description::surface_create_description()
{
	null();
}

surface_create_description::~surface_create_description()
{
	freeself();
}

void surface_create_description::null()
{
	f_q = FALSE;
	q = 0;
	f_catalogue = FALSE;
	iso = 0;
	f_by_coefficients = FALSE;
	coefficients_text = NULL;
	f_family_HCV = FALSE;
	family_HCV_a = 0;
	f_family_G13 = FALSE;
	family_G13_a = 0;
	f_family_F13 = FALSE;
	family_F13_a = 0;
	f_arc_lifting = FALSE;
	arc_lifting_text = NULL;
	arc_lifting_two_lines_text = NULL;
	f_arc_lifting_with_two_lines = FALSE;
	nb_select_double_six = 0;
	//select_double_six_string[];
}

void surface_create_description::freeself()
{
	null();
}

int surface_create_description::read_arguments(int argc, const char **argv, 
	int verbose_level)
{
	int i;

	cout << "surface_create_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (argv[i][0] != '-') {
			continue;
		}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (strcmp(argv[i], "-catalogue") == 0) {
			f_catalogue = TRUE;
			iso = atoi(argv[++i]);
			cout << "-catalogue " << iso << endl;
		}
		else if (strcmp(argv[i], "-by_coefficients") == 0) {
			f_by_coefficients = TRUE;
			coefficients_text = argv[++i];
			cout << "-by_coefficients " << coefficients_text << endl;
		}
		else if (strcmp(argv[i], "-family_HCV") == 0) {
			f_family_HCV = TRUE;
			family_HCV_a = atoi(argv[++i]);
			cout << "-family_HCV " << family_HCV_a << endl;
		}
		else if (strcmp(argv[i], "-family_G13") == 0) {
			f_family_G13 = TRUE;
			family_G13_a = atoi(argv[++i]);
			cout << "-family_G13 " << family_G13_a << endl;
		}
		else if (strcmp(argv[i], "-family_F13") == 0) {
			f_family_F13 = TRUE;
			family_F13_a = atoi(argv[++i]);
			cout << "-family_F13 " << family_F13_a << endl;
		}
		else if (strcmp(argv[i], "-arc_lifting") == 0) {
			f_arc_lifting = TRUE;
			arc_lifting_text = argv[++i];
			cout << "-arc_lifting " << arc_lifting_text << endl;
		}
		else if (strcmp(argv[i], "-arc_lifting_with_two_lines") == 0) {
			f_arc_lifting_with_two_lines = TRUE;
			arc_lifting_text = argv[++i];
			arc_lifting_two_lines_text = argv[++i];
			cout << "-arc_lifting_with_two_lines " << arc_lifting_text << " " << arc_lifting_two_lines_text << endl;
		}
		else if (strcmp(argv[i], "-select_double_six") == 0) {
			//f_select_double_six = TRUE;
			if (nb_select_double_six == SURFACE_CREATE_MAX_SELECT_DOUBLE_SIX) {
				cout << "too many -select_double_six options" << endl;
				exit(1);
			}
			select_double_six_string[nb_select_double_six++] = argv[++i];
			cout << "-select_double_six " << select_double_six_string[nb_select_double_six - 1] << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
	} // next i
	cout << "surface_create_description::read_arguments done" << endl;
	return i;
}


int surface_create_description::get_q()
{
	if (!f_q) {
		cout << "surface_create_description::get_q "
				"q has not been set yet" << endl;
		exit(1);
	}
	return q;
}

}}



