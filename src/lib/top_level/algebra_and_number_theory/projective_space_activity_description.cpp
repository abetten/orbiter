/*
 * projective_space_activity_description.cpp
 *
 *  Created on: Jan 5, 2021
 *      Author: betten
 */





#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


projective_space_activity_description::projective_space_activity_description()
{

	f_input = FALSE;
	Data = NULL;

	f_fname_base_out = FALSE;
	//fname_base_out;

	f_canonical_form_PG = FALSE;
	//canonical_form_PG_n = 0;
	Canonical_form_PG_Descr = NULL;


}

projective_space_activity_description::~projective_space_activity_description()
{

}


int projective_space_activity_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "projective_space_activity_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			cout << "-input" << endl;
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "projective_space_activity_description::read_arguments finished reading -input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out.assign(argv[++i]);
			cout << "-fname_base_out " << fname_base_out << endl;
		}
		else if (stringcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			//canonical_form_PG_n = strtoi(argv[++i]);
			cout << "-canonical_form_PG, reading extra arguments" << endl;

			Canonical_form_PG_Descr = NEW_OBJECT(projective_space_object_classifier_description);

			i += Canonical_form_PG_Descr->read_arguments(argc - (i + 1), argv + i + 1, verbose_level);
			cout << "done reading -canonical_form_PG " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}


		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "projective_space_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "projective_space_activity_description::read_arguments looping, i=" << i << endl;
	} // next i

	cout << "projective_space_activity_description::read_arguments done" << endl;
	return i + 1;
}


}}
