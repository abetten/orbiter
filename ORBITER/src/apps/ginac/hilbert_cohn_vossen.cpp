/*
 * hilbert_cohn_vossen.cpp
 *
 *  Created on: Dec 8, 2019
 *      Author: betten
 */



#include "orbiter.h"


using namespace std;
using namespace orbiter;
using namespace orbiter::top_level;



#include "ginac/ginac.h"

using namespace GiNaC;

#include <iostream>
using namespace std;


void surface();

void surface()
{
	symbol u("u"), mu("mu"), v("v"), mv("mv");
	matrix V(20,3);
	matrix P1(1,3);
	matrix P2(1,3);
	matrix P3(1,3);
	matrix P4(1,3);
	matrix P5(1,3);
	matrix P6(1,3);
	matrix P7(1,3);
	matrix P8(1,3);
	matrix P9(1,3);
	matrix P10(1,3);
	matrix P11(1,3);
	matrix P12(1,3);
	matrix P13(1,3);
	matrix P14(1,3);
	matrix P15(1,3);
	matrix P16(1,3);
	matrix P17(1,3);
	matrix P18(1,3);
	matrix P19(1,3);
	//ex v1, v2, v4, v4;
	//std::vector<ex> v1;
	//lst v1, v2, v3;
	V = matrix{
		{u, u, u},
		{u,u,mu},
		{u,mu,u},
		{u,mu,mu},
		{mu,u,u},
		{mu,u,mu},
		{mu,mu,u},
		{mu,mu,mu},
		{v,u,u},
		{u,v,u},
		{u,u,v},
		{mv,mu,u},
		{mu,mv,u},
		{mu,mu,v},
		{v,mu,mu},
		{u,mv,mu},
		{u,mu,mv},
		{mv,u,mu},
		{mu,v,mu},
		{mu,u,mv},
	};
	//v1.append(u).append(u).append(u);
	//v1.push_back(ex("u",lst{u});
	cout << "V=" << V << endl;

	int red_idx[] = {9, 18, 10, 13, 11, 17, 12, 15, 16, 19, 14, 20};
	int blue_idx[] = {14, 17, 15, 18, 13, 19, 11, 20, 9, 12, 10, 16};

	int i;
	for (i = 0; i < 12; i++) {
		red_idx[i]--;
	}
	for (i = 0; i < 12; i++) {
		blue_idx[i]--;
	}
	int idx, idx1, idx2;
	int a, b;

	// four points on a1:
	idx = red_idx[0 * 2 + 0];
	P1 = matrix{{V(idx, 0), V(idx, 1), V(idx, 2)}};
	cout << "P1=" << P1 << endl;
	idx = red_idx[0 * 2 + 1];
	P2 = matrix{{V(idx, 0), V(idx, 1), V(idx, 2)}};
	cout << "P2=" << P2 << endl;
	a = -1; b = 2;
	P3 = matrix {{
		a * P1(0,0) + b * P2(0,0),
		a * P1(0,1) + b * P2(0,1),
		a * P1(0,2) + b * P2(0,2)
	}};
	cout << "P3=" << P3 << endl;
	a = -2; b = 3;
	P4 = matrix {{
		a * P1(0,0) + b * P2(0,0),
		a * P1(0,1) + b * P2(0,1),
		a * P1(0,2) + b * P2(0,2)
	}};
	cout << "P4=" << P4 << endl;
	// three points on b2:
	idx1 = blue_idx[1 * 2 + 1];
	idx2 = blue_idx[1 * 2 + 0];
	a = 0; b = 1;
	P5 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P5=" << P5 << endl;
	a = -1; b = 2;
	P6 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P6=" << P6 << endl;
	a = -2; b = 3;
	P7 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P7=" << P7 << endl;
	// three points on b3:
	idx1 = blue_idx[2 * 2 + 0];
	idx2 = blue_idx[2 * 2 + 1];
	a = 1; b = 0;
	P8 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P8=" << P8 << endl;
	a = 0; b = 1;
	P9 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P9=" << P9 << endl;
	a = -2; b = 3;
	P10 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P10=" << P10 << endl;
	// three points on b4:
	idx1 = blue_idx[3 * 2 + 0];
	idx2 = blue_idx[3 * 2 + 1];
	a = 1; b = 0;
	P11 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P11=" << P11 << endl;
	a = 0; b = 1;
	P12 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P12=" << P12 << endl;
	a = -2; b = 3;
	P13 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P13=" << P13 << endl;
	// three points on b5:
	idx1 = blue_idx[4 * 2 + 0];
	idx2 = blue_idx[4 * 2 + 1];
	a = 1; b = 0;
	P14 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P14=" << P14 << endl;
	a = -1; b = 2;
	P15 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P15=" << P15 << endl;
	a = -2; b = 3;
	P16 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P16=" << P16 << endl;
	// three points on b6:
	idx1 = blue_idx[5 * 2 + 0];
	idx2 = blue_idx[5 * 2 + 1];
	a = -1; b = 2;
	P17 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P17=" << P17 << endl;
	a = -2; b = 3;
	P18 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P18=" << P18 << endl;
	a = -3; b = 4;
	P19 = matrix {{
		a * V(idx1,0) + b * V(idx2,0),
		a * V(idx1,1) + b * V(idx2,1),
		a * V(idx1,2) + b * V(idx2,2)
	}};
	cout << "P19=" << P19 << endl;
}




int main() {

	surface();

}
