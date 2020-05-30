/*
 * diophant_description.cpp
 *
 *  Created on: May 28, 2020
 *      Author: betten
 */



#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {



diophant_description::diophant_description()
{
	f_q = FALSE;
	input_q = 0;
	f_override_polynomial = FALSE;
	override_polynomial = NULL;
	F = NULL;

	f_maximal_arc = FALSE;
	maximal_arc_sz = 0;
	maximal_arc_d = 0;
	maximal_arc_secants_text = NULL;
	external_lines_as_subset_of_secants_text = NULL;
}

diophant_description::~diophant_description()
{
}

int diophant_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "diophant_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {


		if (strcmp(argv[i], "-maximal_arc") == 0) {
			f_maximal_arc = TRUE;
			maximal_arc_sz = atoi(argv[++i]);
			maximal_arc_d = atoi(argv[++i]);
			maximal_arc_secants_text = argv[++i];
			external_lines_as_subset_of_secants_text = argv[++i];
			cout << "-maximal_arc " << maximal_arc_sz << " " << maximal_arc_d
					<< " " << maximal_arc_secants_text
					<< " " << external_lines_as_subset_of_secants_text << endl;
		}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			input_q = atoi(argv[++i]);
			cout << "-q" << input_q << endl;
		}
		else if (strcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial = argv[++i];
			cout << "-override_polynomial" << override_polynomial << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "diophant_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "diophant_description::read_arguments done" << endl;
	return i;
}

}}

