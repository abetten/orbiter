// srg.C
// 
// Anton Betten
// Dec 4, 2015
//
//
//
//

#include "orbiter.h"


#define V  50


void list_parameters(INT v_max);

int main(int argc, const char **argv)
{
	list_parameters(V);
}


void list_parameters(INT v_max)
{
	INT v, v2, k, lambda, mu, cnt = 0;
	INT top, top2, bottom, b, tb;
	INT i, f, g, r, s;
	
	for (v = 2; v <= v_max; v++) {
		v2 = v >> 1;
		for (k = 1; k <= v2; k++) {
			for (lambda = 0; lambda <= k; lambda++) {
				for (mu = 1; mu <= k; mu++) {
					if (k * (k - lambda - 1) != mu * (v - k - 1)) {
						continue;
						}
					top = (v - 1) * (mu - lambda) - 2 * k;
					top2 = top * top;
					bottom = (mu - lambda) * (mu - lambda) + 4 * (k - mu);
					cnt++;
					cout << "cnt=" << cnt << " v=" << v << " k=" << k << " lambda=" << lambda << " mu=" << mu << " top=" << top << " bottom=" << bottom << endl;
					if (top2 % bottom) {
						cout << "is ruled out by integrality condition" << endl;
						continue;
						}

					INT nb;
					INT *primes, *exponents;
					nb = factor_INT(bottom, primes, exponents);
					for (i = 0; i < nb; i++) {
						if (ODD(exponents[i])) {
							break;
							}
						}
					if (i < nb) {
						cout << "bottom is not a square" << endl;
						continue;
						}
					for (i = 0; i < nb; i++) {
						exponents[i] >>= 1;
						}
					b = 1;
					for (i = 0; i < nb; i++) {
						b *= i_power_j(primes[i], exponents[i]);
						}
					cout << "b=" << b << endl;
					tb = top / b;
					cout << "tb=" << tb << endl;
					if (ODD(v - 1 + tb)) {
						cout << "is ruled out by integrality condition (2)" << endl;
						continue;
						}
					if (ODD(v - 1 - tb)) {
						cout << "is ruled out by integrality condition (3)" << endl;
						continue;
						}
					f = (v - 1 + tb) >> 1;
					g = (v - 1 - tb) >> 1;
					if (ODD(lambda - mu + b)) {
						cout << "r is not integral, skip" << endl;
						continue;
						}
					if (ODD(lambda - mu - b)) {
						cout << "r is not integral, skip" << endl;
						continue;
						}
					r = (lambda - mu + b) >> 1;
					s = (lambda - mu - b) >> 1;
					cout << "f=" << f << " g=" << g << " r=" << r << " s=" << s << endl;
					
					INT L1, R1, L2, R2;

					L1 = (r + 1) * (k + r + 2 * r * s);
					R1 = (k + r) * (s + 1) * (s + 1);
					L2 = (s + 1) * (k + s + 2 * r * s);
					R2 = (k + s) * (r + 1) * (r + 1);

					if (L1 > R1) {
						cout << "is ruled out by Krein condition (1)" << endl;
						continue;
						}
					if (L2 > R2) {
						cout << "is ruled out by Krein condition (2)" << endl;
						continue;
						}
					}
				}
			}
		}
}


