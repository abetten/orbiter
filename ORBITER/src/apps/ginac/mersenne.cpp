/*
 * mersenne.cpp
 *
 *  Created on: Dec 8, 2019
 *      Author: betten
 *
 *      adapted from Richard Kreckel's talk at ICMS in 2006
 */




#include <iostream>
#include <cln/cln.h>

using namespace std;
using namespace cln;

int main() {
	int p = 82589933; // found on Dec 7, 2018;
	cl_I x = ((cl_I(1) << p) - 1); cout << x << endl;
}
