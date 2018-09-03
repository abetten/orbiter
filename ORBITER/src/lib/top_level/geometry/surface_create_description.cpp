// surface_create_description.C
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
	f_family_S = FALSE;
	parameter_a = 0;
	f_arc_lifting = FALSE;
	arc_lifting_text = NULL;
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
		else if (strcmp(argv[i], "-family_S") == 0) {
			f_family_S = TRUE;
			parameter_a = atoi(argv[++i]);
			cout << "-family_S " << parameter_a << endl;
			}
		else if (strcmp(argv[i], "-arc_lifting") == 0) {
			f_arc_lifting = TRUE;
			arc_lifting_text = argv[++i];
			cout << "-arc_lifting " << arc_lifting_text << endl;
			}
		else if (strcmp(argv[i], "-end") == 0) {
			return i;
			}
		} // next i
	cout << "surface_create_description::read_arguments done" << endl;
	return i;
}


