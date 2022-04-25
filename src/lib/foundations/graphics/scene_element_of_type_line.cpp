/*
 * scene_element_of_type_line.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */




#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace graphics {

scene_element_of_type_line::scene_element_of_type_line()
{
}

scene_element_of_type_line::~scene_element_of_type_line()
{
}

void scene_element_of_type_line::init(double *coord6)
{
	Line_coords[0] = coord6[0];
	Line_coords[1] = coord6[1];
	Line_coords[2] = coord6[2];
	Line_coords[3] = coord6[3];
	Line_coords[4] = coord6[4];
	Line_coords[5] = coord6[5];
}

void scene_element_of_type_line::print()
{
	numerics N;

	N.vec_print(Line_coords + 0, 3);
	cout << ", ";
	N.vec_print(Line_coords + 3, 3);
	cout << endl;
}


}}}

