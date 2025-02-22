/*
 * mapping_description.cpp
 *
 *  Created on: Sep 23, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {


mapping_description::mapping_description()
{
	Record_birth();
	f_domain = false;
	//std::string domain_label;

	f_codomain = false;
	//std::string codomain_label;

	f_ring = false;
	//std::string ring_label;

	f_formula = false;
	//std::string formula_label;

	f_substitute = false;
	//std::string substitute_text;

	f_affine = false;

	f_object_in_codomain_cubic_surface = false;
	//std::string object_in_codomain_cubic_surface_label;

}

mapping_description::~mapping_description()
{
	Record_death();
}

int mapping_description::read_arguments(
		int argc, std::string *argv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	other::data_structures::string_tools ST;


	if (f_v) {
		cout << "mapping_description::read_arguments" << endl;
	}
	for (i = 0; i < argc; i++) {

		if (ST.stringcmp(argv[i], "-domain") == 0) {
			f_domain = true;
			domain_label.assign(argv[++i]);
			if (f_v) {
				cout << "-domain " << domain_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-codomain") == 0) {
			f_codomain = true;
			codomain_label.assign(argv[++i]);
			if (f_v) {
				cout << "-codomain " << codomain_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-ring") == 0) {
			f_ring = true;
			ring_label.assign(argv[++i]);
			if (f_v) {
				cout << "-ring " << ring_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-formula") == 0) {
			f_formula = true;
			formula_label.assign(argv[++i]);
			if (f_v) {
				cout << "-formula " << formula_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-substitute") == 0) {
			f_substitute = true;
			substitute_text.assign(argv[++i]);
			if (f_v) {
				cout << "-substitute " << substitute_text << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-affine") == 0) {
			f_affine = true;
			if (f_v) {
				cout << "-affine " << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-object_in_codomain_cubic_surface") == 0) {
			f_object_in_codomain_cubic_surface = true;
			object_in_codomain_cubic_surface_label.assign(argv[++i]);
			if (f_v) {
				cout << "-object_in_codomain_cubic_surface " << object_in_codomain_cubic_surface_label << endl;
			}
		}
		else if (ST.stringcmp(argv[i], "-end") == 0) {
			break;
		}
		else {
			cout << "mapping_description::read_arguments unknown argument " << argv[i] << endl;
			exit(1);
		}
	} // next i

	if (f_v) {
		cout << "mapping_description::read_arguments done" << endl;
	}
	return i + 1;
}

void mapping_description::print()
{
	if (f_domain) {
			cout << "-domain " << domain_label << endl;
	}
	if (f_codomain) {
			cout << "-codomain " << codomain_label << endl;
	}
	if (f_ring) {
			cout << "-ring " << ring_label << endl;
	}
	if (f_formula) {
			cout << "-formula " << formula_label << endl;
	}
	if (f_substitute) {
			cout << "-substitute " << substitute_text << endl;
	}
	if (f_affine) {
			cout << "-affine " << endl;
	}
	if (f_object_in_codomain_cubic_surface) {
		cout << "-object_in_codomain_cubic_surface " << object_in_codomain_cubic_surface_label << endl;
	}
}


}}}



