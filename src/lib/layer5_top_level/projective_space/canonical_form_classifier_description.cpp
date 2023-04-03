/*
 * canonical_form_classifier_description.cpp
 *
 *  Created on: Apr 24, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace projective_geometry {



canonical_form_classifier_description::canonical_form_classifier_description()
{

	f_space = false;
	//std::string space_label;

	f_input_fname_mask = false;
	std::string fname_mask;

	f_nb_files = false;
	nb_files = 0;


	f_output_fname = false;
	//std::string fname_base_out;

	f_label_po = false;
	column_label_po.assign("PO");
	f_label_so = false;
	column_label_so.assign("orbit");
	f_label_equation = false;
	column_label_eqn.assign("curve");
	f_label_points = false;
	column_label_pts.assign("pts_on_curve");
	f_label_lines = false;
	column_label_bitangents.assign("bitangents");


	f_degree = false;
	degree = 0;

	f_algorithm_nauty = false;
	f_algorithm_substructure = false;

	f_substructure_size = false;
	substructure_size = 0;

	f_skip = false;
	//std::string skip_label;

	PA = NULL;

	Canon_substructure = NULL;


}


canonical_form_classifier_description::~canonical_form_classifier_description()
{
}

int canonical_form_classifier_description::read_arguments(int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	data_structures::string_tools ST;

	cout << "canonical_form_classifier_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-space") == 0) {
			f_space = true;
			space_label.assign(argv[++i]);
			if (f_v) {
				cout << "-space " << space_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-input_fname_mask") == 0) {
			f_input_fname_mask = true;
			fname_mask.assign(argv[++i]);
			if (f_v) {
				cout << "-input_fname_mask " << fname_mask << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-nb_files") == 0) {
			f_nb_files = true;
			nb_files = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-nb_files " << nb_files << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-output_fname") == 0) {
			f_output_fname = true;
			fname_base_out.assign(argv[++i]);
			if (f_v) {
				cout << "-output_fname " << fname_base_out << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_po") == 0) {
			f_label_po = true;
			column_label_po.assign(argv[++i]);
			if (f_v) {
				cout << "-label_po " << column_label_po << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_so") == 0) {
			f_label_so = true;
			column_label_so.assign(argv[++i]);
			if (f_v) {
				cout << "-label_so " << column_label_so << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_equation") == 0) {
			f_label_equation = true;
			column_label_eqn.assign(argv[++i]);
			if (f_v) {
				cout << "-label_equation " << column_label_eqn << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_points") == 0) {
			f_label_points = true;
			column_label_pts.assign(argv[++i]);
			if (f_v) {
				cout << "-label_points " << column_label_pts << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-label_lines") == 0) {
			f_label_lines = true;
			column_label_bitangents.assign(argv[++i]);
			if (f_v) {
				cout << "-label_lines " << column_label_bitangents << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-degree") == 0) {
			f_degree = true;
			degree = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-degree " << degree << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-algorithm_nauty") == 0) {
			f_algorithm_nauty = true;
			if (f_v) {
				cout << "-algorithm_nauty" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-algorithm_substructure") == 0) {
			f_algorithm_substructure = true;
			if (f_v) {
				cout << "-algorithm_substructure" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-substructure_size") == 0) {
			f_substructure_size = true;
			substructure_size = ST.strtoi(argv[++i]);
			if (f_v) {
				cout << "-substructure_size" << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-skip") == 0) {
			f_skip = true;
			skip_label.assign(argv[++i]);
			if (f_v) {
				cout << "-skip " << skip_label << endl;
			}
		}

		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "canonical_form_classifier_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "canonical_form_classifier_description::read_arguments done" << endl;
	return i + 1;
}




void canonical_form_classifier_description::print()
{
	if (f_space) {
		cout << "-space " << space_label << endl;
	}
	if (f_input_fname_mask) {
		cout << "-input_fname_mask " << fname_mask << endl;
	}
	if (f_nb_files) {
		cout << "-nb_files " << nb_files << endl;
	}
	if (f_output_fname) {
		cout << "-output_fname " << fname_base_out << endl;
	}
	if (f_label_po) {
		cout << "-label_po " << column_label_po << endl;
	}
	if (f_label_so) {
		cout << "-label_so " << column_label_so << endl;
	}
	if (f_label_equation) {
		cout << "-label_equation " << column_label_eqn << endl;
	}
	if (f_label_points) {
		cout << "-label_points " << column_label_pts << endl;
	}
	if (f_label_lines) {
		cout << "-label_lines " << column_label_bitangents << endl;
	}
	if (f_degree) {
		cout << "-degree " << degree << endl;
	}
	if (f_algorithm_nauty) {
		cout << "-algorithm_nauty" << endl;
	}
	if (f_algorithm_substructure) {
		cout << "-algorithm_substructure" << endl;
	}
	if (f_substructure_size) {
		cout << "-substructure_size" << endl;
	}
	if (f_skip) {
		cout << "-skip " << skip_label << endl;
	}
}




}}}

