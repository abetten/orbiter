/*
 * combinatorial_object.cpp
 *
 *  Created on: Dec 17, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combinatorial_object::combinatorial_object()
{
	//Data_input_stream_description = NULL;

	IS = NULL;

	Classification = NULL;
	Classification_CO = NULL;

}

combinatorial_object::~combinatorial_object()
{
	if (Classification) {
		FREE_OBJECT(Classification);
	}
	if (Classification_CO) {
		FREE_OBJECT(Classification_CO);
	}
}


void combinatorial_object::init(
		canonical_form_classification::data_input_stream_description
				*Data_input_stream_description,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object::init" << endl;
	}
	//combinatorial_object::Data_input_stream_description = Data_input_stream_description;


	IS = NEW_OBJECT(canonical_form_classification::data_input_stream);

	if (f_v) {
		cout << "combinatorial_object::init "
				"before IS->init" << endl;
	}

	IS->init(Data_input_stream_description, verbose_level);

	if (f_v) {
		cout << "combinatorial_object::init "
				"after IS->init" << endl;
	}


	if (f_v) {
		cout << "combinatorial_object::init done" << endl;
	}

}

}}}


