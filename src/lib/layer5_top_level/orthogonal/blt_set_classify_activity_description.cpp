/*
 * blt_set_classify_activity_description.cpp
 *
 *  Created on: Aug 2, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {

blt_set_classify_activity_description::blt_set_classify_activity_description()
{
	f_compute_starter = FALSE;
	starter_control = NULL;

	f_create_graphs = FALSE;

	f_split = FALSE;
	split_r = 0;
	split_m = 1;

	f_isomorph = FALSE;
	Isomorph_arguments = NULL;

}

blt_set_classify_activity_description::~blt_set_classify_activity_description()
{
}

int blt_set_classify_activity_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	if (f_v) {
		cout << "blt_set_classify_activity_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-compute_starter") == 0) {
			f_compute_starter = TRUE;
			if (f_v) {
				cout << "-compute_starter " << endl;
			}
			starter_control = NEW_OBJECT(poset_classification::poset_classification_control);

			i += starter_control->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

		}
		else if (ST.stringcmp(argv[i], "-create_graphs") == 0) {
			f_create_graphs = TRUE;
			if (f_v) {
				cout << "-create_graphs " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_r = ST.strtoi(argv[++i]);
			split_m = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-split " << split_r << " " << split_m << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = TRUE;
			//prefix_classify.assign(argv[++i]);
			//prefix_iso.assign(argv[++i]);
			Isomorph_arguments = NEW_OBJECT(layer4_classification::isomorph::isomorph_arguments);

			i += Isomorph_arguments->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			if (f_v) {
				cout << "-isomorph " << endl; //prefix_classify << " " << prefix_iso << endl;
				Isomorph_arguments->print();
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "blt_set_classify_activity_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	if (f_v) {
		cout << "blt_set_classify_activity_description::read_arguments done" << endl;
	}
	return i + 1;
}

void blt_set_classify_activity_description::print()
{
	if (f_compute_starter) {
		cout << "-compute_starter " << endl;
	}
	if (f_create_graphs) {
		cout << "-create_graphs " << endl;
	}
	if (f_split) {
		cout << "-split " << split_r << " " << split_m << endl;
	}
	if (f_isomorph) {
		cout << "-isomorph " << endl; //prefix_classify << " " << prefix_iso << endl;
		Isomorph_arguments->print();
	}
}


}}}




