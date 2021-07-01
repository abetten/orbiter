/*
 * top_level_geometry_global.cpp
 *
 *  Created on: May 23, 2021
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


top_level_geometry_global::top_level_geometry_global()
{

}

top_level_geometry_global::~top_level_geometry_global()
{

}


void top_level_geometry_global::set_stabilizer_projective_space(
		projective_space_with_action *PA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space" << endl;
	}

	substructure_classifier *SubC;

	SubC = NEW_OBJECT(substructure_classifier);

	SubC->set_stabilizer_in_any_space(
			PA->A, PA->A, PA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			verbose_level);


	FREE_OBJECT(SubC);
	if (f_v) {
		cout << "top_level_geometry_global::set_stabilizer_projective_space done" << endl;
	}

}


}}



