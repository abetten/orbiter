/*
 * parametric_curve_point.cpp
 *
 *  Created on: Apr 16, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {


parametric_curve_point::parametric_curve_point()
{
	t = 0;
	f_is_valid = FALSE;
}

parametric_curve_point::~parametric_curve_point()
{

}

void parametric_curve_point::init(double t, int f_is_valid,
		double *x, int nb_dimensions, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "parametric_curve_point::init" << endl;
	}
	parametric_curve_point::t = t;
	parametric_curve_point::f_is_valid = f_is_valid;
	for (i = 0; i < nb_dimensions; i++) {
		coords.push_back(x[i]);
	}
	if (f_v) {
		for (i = 0; i < (int) coords.size(); i++) {
			cout << coords[i];
			if (i < (int) coords.size() - 1) {
				cout << ",";
			}
		}
		cout << endl;
	}
	if (f_v) {
		cout << "parametric_curve_point::init done" << endl;
	}
}



}}

