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
	//std::string fname_mask;
	nb_files = 0;


	f_fname_base_out = FALSE;
	//std::string fname_base_out;

	column_label_po.assign("PO");
	column_label_so.assign("orbit");
	column_label_eqn.assign("curve");
	column_label_pts.assign("pts_on_curve");
	column_label_bitangents.assign("bitangents");


	f_degree = FALSE;
	degree = 0;

	f_algorithm_nauty = FALSE;
	f_algorithm_substructure = FALSE;
	substructure_size = 0;

	PA = NULL;

	Canon_substructure = NULL;


}


canonical_form_classifier_description::~canonical_form_classifier_description()
{
}


}}}

