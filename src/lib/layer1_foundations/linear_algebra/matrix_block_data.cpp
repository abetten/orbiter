/*
 * matrix_block_data.cpp
 *
 *  Created on: Feb 9, 2019
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace linear_algebra {





matrix_block_data::matrix_block_data()
{
	d = m = 0;
	poly_coeffs = NULL;
	b0 = b1 = 0;

	K = NULL;
	cnt = 0;
	dual_part = NULL;
	part = NULL;
	height = 0;
	part_idx = 0;
}


matrix_block_data::~matrix_block_data()
{
	if (K) {
		FREE_OBJECTS(K);
	}
	if (dual_part) {
		FREE_int(dual_part);
	}
	if (part) {
		FREE_int(part);
	}
}

void matrix_block_data::allocate(int k)
{
	K = NEW_OBJECTS(data_structures::int_matrix, k);
	dual_part = NEW_int(k);
	part = NEW_int(k);
}


}}}
