/*
 * override_double.cpp
 *
 *  Created on: Sep 16, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orbiter_kernel_system {





override_double::override_double(
		double *p, double value)
{
	override_double::p = p;
	original_value = *p;
	new_value = value;
	*p = value;
}

override_double::~override_double()
{
	if (p) {
		*p = original_value;
	}
}


}}}

