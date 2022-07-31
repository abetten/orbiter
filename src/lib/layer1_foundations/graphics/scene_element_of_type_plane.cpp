/*
 * scene_element_of_type_plane.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */



#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace graphics {

scene_element_of_type_plane::scene_element_of_type_plane()
{
}

scene_element_of_type_plane::~scene_element_of_type_plane()
{
}

void scene_element_of_type_plane::init(double *coord4)
{
	Plane_coords[0] = coord4[0];
	Plane_coords[1] = coord4[1];
	Plane_coords[2] = coord4[2];
	Plane_coords[3] = coord4[3];
}

void scene_element_of_type_plane::print()
{
	numerics N;

	N.vec_print(Plane_coords, 4);
	cout << endl;
}


}}}


