/*
 * scene_element_of_type_surface.cpp
 *
 *  Created on: Apr 4, 2022
 *      Author: betten
 */






#include "foundations.h"

using namespace std;




namespace orbiter {
namespace layer1_foundations {
namespace graphics {

scene_element_of_type_surface::scene_element_of_type_surface()
{
	d = 0;
	nb_coeffs = 0;
	Eqn = NULL;
}

scene_element_of_type_surface::~scene_element_of_type_surface()
{
	if (Eqn) {
		delete [] Eqn;
	}
}

void scene_element_of_type_surface::init(int d, int nb_coeffs, double *coords)
{
	int i;

	scene_element_of_type_surface::d = d;
	scene_element_of_type_surface::nb_coeffs = nb_coeffs;
	Eqn = new double [nb_coeffs];
	for (i = 0; i < nb_coeffs; i++) {
		Eqn[i] = coords[i];
	}
}

void scene_element_of_type_surface::print()
{
	orbiter_kernel_system::numerics N;

	cout << "surface of degree " << d << " with " << nb_coeffs << " coefficients : ";
	N.vec_print(Eqn, nb_coeffs);
	cout << endl;
}


}}}

