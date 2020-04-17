/*
 * parametric_curve_point.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


parametric_curve_point::parametric_curve_point()
{
	t = 0;
	f_is_valid = FALSE;
}

parametric_curve_point::~parametric_curve_point()
{

}

void parametric_curve_point::init2(double t, int f_is_valid, double x0, double x1)
{
	parametric_curve_point::t = t;
	parametric_curve_point::f_is_valid = f_is_valid;
	coords.push_back(x0);
	coords.push_back(x1);
}


}}

