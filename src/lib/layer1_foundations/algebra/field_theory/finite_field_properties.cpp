/*
 * finite_field_properties.cpp
 *
 *  Created on: Apr 26, 2025
 *      Author: betten
 */





#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace field_theory {



finite_field_properties::finite_field_properties()
{
	Record_birth();
	F = NULL;

	f_related_fields_have_been_computed = false;
	Related_fields = NULL;

	f_equianharmonic = false;
	equianharmonic_a = 0;

}

finite_field_properties::~finite_field_properties()
{
	Record_death();
	int f_v = false;

	if (f_related_fields_have_been_computed) {
		if (f_v) {
			cout << "finite_field_properties::~finite_field_properties "
					"before FREE_OBJECT(Related_fields)" << endl;
		}
		FREE_OBJECT(Related_fields);
		if (f_v) {
			cout << "finite_field_properties::~finite_field_properties "
					"after FREE_OBJECT(Related_fields)" << endl;
		}
	}

}

void finite_field_properties::init(
		finite_field *F,
		int f_compute_related_fields,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_properties::init" << endl;
	}

	finite_field_properties::F = F;

	if (f_compute_related_fields) {
		if (f_v) {
			cout << "finite_field_properties::init "
					"before setup_related_fields" << endl;
		}
		setup_related_fields(f_compute_related_fields, verbose_level);
		if (f_v) {
			cout << "finite_field_properties::init "
					"after setup_related_fields" << endl;
		}
	}

	if (f_v) {
		cout << "finite_field_properties::init "
				"before compute_harmonics" << endl;
	}
	compute_harmonics(verbose_level);
	if (f_v) {
		cout << "finite_field_properties::init "
				"after compute_harmonics" << endl;
	}


	if (f_v) {
		cout << "finite_field_properties::init done" << endl;
	}
}

void finite_field_properties::compute_harmonics(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_properties::compute_harmonics" << endl;
	}

	int a;

	for (a = 1; a < F->q; a++) {
		if (F->add3(F->mult(a, a), F->negate(a), 1) == 0) {
			f_equianharmonic = true;
			equianharmonic_a = a;
			break;
		}
	}

	if (f_v) {
		cout << "finite_field_properties::compute_harmonics" << endl;

		if (f_equianharmonic) {
			cout << "finite_field_properties::compute_harmonics "
					"The field is equianharmonic, a = " << equianharmonic_a << endl;
		}
		else {
			cout << "finite_field_properties::compute_harmonics "
					"The field is not equianharmonic" << endl;
		}
	}

	if (f_v) {
		cout << "finite_field_properties::compute_harmonics done" << endl;
	}
}

void finite_field_properties::setup_related_fields(
		int f_compute_related_fields,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_properties::setup_related_fields q=" << F->q << endl;
		cout << "f_compute_related_fields = " << f_compute_related_fields << endl;
	}

	if (f_compute_related_fields) {

		if (f_related_fields_have_been_computed) {
			if (f_v) {
				cout << "finite_field_properties::setup_related_fields "
						"related fields have been computed already" << endl;
			}
		}
		else {
			Related_fields = NEW_OBJECT(related_fields);

			if (f_v) {
				cout << "finite_field_properties::setup_related_fields "
						"before Related_fields->init" << endl;
			}
			Related_fields->init(F, verbose_level);
			if (f_v) {
				cout << "finite_field_properties::setup_related_fields "
						"after Related_fields->init" << endl;
			}
			f_related_fields_have_been_computed = true;
		}
	}
	else {
		if (f_v) {
			cout << "finite_field_properties::setup_related_fields q=" << F->q
				<< " not computing related fields" << endl;
		}
		f_related_fields_have_been_computed = false;

	}

	if (f_v) {
		cout << "finite_field_properties::setup_related_fields done" << endl;
	}
}

void finite_field_properties::report_latex(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field_properties::report_latex" << endl;
	}

	if (f_equianharmonic) {
		ost << "The field is equianharmonic, $a = " << equianharmonic_a << "$\\\\" << endl;
	}
	else {
		ost << "The field is not equianharmonic.\\\\" << endl;

	}

	if (f_v) {
		cout << "finite_field_properties::report_latex done" << endl;
	}
}



}}}}

