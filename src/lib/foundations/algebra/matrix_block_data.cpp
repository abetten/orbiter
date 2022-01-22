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
namespace algebra {





matrix_block_data::matrix_block_data()
{
	null();
}

matrix_block_data::~matrix_block_data()
{
	freeself();
}

void matrix_block_data::null()
{
	K = NULL;
	part = NULL;
	dual_part = NULL;
	height = 0;
}

void matrix_block_data::freeself()
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
	null();
}

void matrix_block_data::allocate(int k)
{
	K = NEW_OBJECTS(data_structures::int_matrix, k);
	dual_part = NEW_int(k);
	part = NEW_int(k);
}


}}}
