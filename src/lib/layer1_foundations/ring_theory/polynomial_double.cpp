/*
 * polynomial_double.cpp
 *
 *  Created on: Feb 17, 2019
 *      Author: betten
 */


#include "foundations.h"


using namespace std;



#define EPSILON 0.01

namespace orbiter {
namespace layer1_foundations {
namespace ring_theory {



polynomial_double::polynomial_double()
{
	alloc_length = 0;
	degree = 0;
	coeff = NULL;
}

polynomial_double::~polynomial_double()
{
	if (coeff) {
		delete [] coeff;
	}
}

void polynomial_double::init(int alloc_length)
{
	int i;

	polynomial_double::alloc_length = alloc_length;
	polynomial_double::degree = 0;
	coeff = new double[alloc_length];
	for (i = 0; i < alloc_length; i++) {
		coeff[i] = 0;
	}
}

void polynomial_double::print(ostream &ost)
{
	int i;

	for (i = 0; i <= degree; i++) {
		if (i) {
			ost << " + ";
		}
		ost << coeff[i];
		if (i) {
			ost << "* t";
			if (i > 1) {
				ost << "^" << i;
			}
		}
	}
}

double polynomial_double::root_finder(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	double l, r, m;
	double vl, vr, vm;
	double eps = 0.0000001;
	numerics N;

	if (ABS(coeff[degree]) < eps) {
		cout << "polynomial_double::root_finder error, ABS(coeff[degree]) < eps" << endl;
		exit(1);
	}
	if (degree == 2) {
		double a, b, c, d;

		a = coeff[2];
		b = coeff[1];
		c = coeff[0];
		if (b * b - 4 * a * c < 0) {
			cout << "polynomial_double::root_finder error discriminant is negative" << endl;
			exit(1);
		}
		d = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		return d;
	}
	else if (degree == 1) {
		double a, b, d;
		a = coeff[1];
		b = coeff[0];
		d = - b / a;
		return d;
	}
	l = -1.;
	r = 1.;
	vl = evaluate_at(l);
	vr = evaluate_at(r);
	if (ABS(vl) < eps) {
		return l;
	}
	if (ABS(vr) < eps) {
		return r;
	}
	while (N.sign_of(evaluate_at(l)) == N.sign_of(evaluate_at(r))) {
		l *= 2.;
		r *= 2.;
	}
	vl = evaluate_at(l);
	vr = evaluate_at(r);
	if (f_v) {
		cout << "polynomial_double::root_finder l=" << l << " r=" << r << endl;
		cout << "value at l = " << vl << endl;
		cout << "value at r = " << vr << endl;
	}
	m = (l + r) *.5;
	vm = evaluate_at(m);
	while (ABS(vm) > eps) {
		if (f_v) {
			cout << "polynomial_double::root_finder l=" << l << " r=" << r << endl;
			cout << "value at l = " << vl << endl;
			cout << "value at r = " << vr << endl;
			cout << "r - l = " << r - l << endl;
			cout << "m = " << m << endl;
			cout << "vm = " << vm << endl;
		}
		if (f_v) {
			cout << "m=" << m << " value=" << vm << " sign=" << N.sign_of(vm) << endl;
		}

		if (N.sign_of(vl) == N.sign_of(vm)) {
			l = m;
			vl = vm;
		}
		else if (N.sign_of(vm) == N.sign_of(vr)) {
			r = m;
			vr = vm;
		}
		else {
			if (ABS(vm) < eps) {
				if (TRUE) {
					cout << "polynomial_double::root_finder hit on a root by chance" << endl;
				}
				return m;
			}
			else {
				cout << "polynomial_double::root_finder problem" << endl;
				cout << "polynomial_double::root_finder l=" << l << " m=" << m << " r=" << r << endl;
				cout << "value at l = " << vl << endl;
				cout << "value at r = " << vr << endl;
				cout << "value at m = " << vm << endl;
				exit(1);
			}
		}
		m = (l + r) *.5;
		vm = evaluate_at(m);
	}
	if (f_v) {
		cout << "polynomial_double::root_finder l=" << l << " r=" << r << endl;
		cout << "value at l = " << vl << endl;
		cout << "value at r = " << vr << endl;
	}
	return m;

}

double polynomial_double::evaluate_at(double t)
{
	int i;
	double a;

	a = coeff[degree];
	for (i = degree - 1; i >= 0; i--) {
		a = a * t + coeff[i];
	}
	return a;
}

}}}

