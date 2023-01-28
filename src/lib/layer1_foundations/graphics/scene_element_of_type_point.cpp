/*
 * scene_element_of_type_point.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */





#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace graphics {

scene_element_of_type_point::scene_element_of_type_point()
{
}

scene_element_of_type_point::~scene_element_of_type_point()
{
}

void scene_element_of_type_point::init(double *coord3)
{
	Point_coords[0] = coord3[0];
	Point_coords[1] = coord3[1];
	Point_coords[2] = coord3[2];
}

void scene_element_of_type_point::print()
{
	orbiter_kernel_system::numerics N;

	N.vec_print(Point_coords, 3);
	cout << endl;
}


}}}


