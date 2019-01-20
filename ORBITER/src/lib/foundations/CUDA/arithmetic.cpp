/*
 * arithmetic.cpp
 *
 *  Created on: Nov 20, 2018
 *      Author: sajeeb
 */

#include "arithmetic.h"


namespace arithmetic {

	// The reason this function exists is because the modulo
	// operator in c++ is implementation dependent. This block
	// of code works around the implementation dependent modulo
	// operator
	int mod(int a, int p)
	{
		if (a < 0) {
			int v = a % p;
			if (v < 0)
				a = p + v;
			else
				return v;
		} else {
			a %= p;
		}
		return a;
	}

	void xgcd(long *result, long a, long b)
	{
		// This block of code has been adapted from:
		// https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
		long aa[2]={1,0}, bb[2]={0,1}, q;
		while(1) {
			q = a / b;
			a = a % b;
			aa[0] = aa[0] - q*aa[1];
			bb[0] = bb[0] - q*bb[1];
			if (a == 0) {
				result[0] = b;
				result[1] = aa[1];
				result[2] = bb[1];
				return;
			};
			q = b / a;
			b = b % a;
			aa[1] = aa[1] - q*aa[0];
			bb[1] = bb[1] - q*bb[0];
			if (b == 0) {
				result[0] = a;
				result[1] = aa[0];
				result[2] = bb[0];
				return;
			};
		};
	}

	int modinv(int a, int b)
	{
		long c[3];
		xgcd(c,a,b);
		long x = c[1];
		return mod(x, b);
	}

}
