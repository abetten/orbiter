/*
 * schlaefli_labels.cpp
 *
 *  Created on: May 22, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {
namespace algebraic_geometry {


schlaefli_labels::schlaefli_labels()
{
	Sets = NULL;
	M = NULL;
	Sets2 = NULL;


	Line_label = NULL;
	Line_label_tex = NULL;
}

schlaefli_labels::~schlaefli_labels()
{
	if (Sets) {
		FREE_lint(Sets);
	}
	if (M) {
		FREE_int(M);
	}
	if (Sets2) {
		FREE_lint(Sets2);
	}


	if (Line_label) {
		delete [] Line_label;
	}
	if (Line_label_tex) {
		delete [] Line_label_tex;
	}

}


void schlaefli_labels::init(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	latex_interface L;

	int i, j, h, h2;

	if (f_v) {
		cout << "schlaefli_labels::init" << endl;
	}

	Sets = NEW_lint(30 * 2);
	M = NEW_int(6 * 6);
	Orbiter->Int_vec->zero(M, 6 * 6);

	h = 0;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (j == i) {
				continue;
			}
			M[i * 6 + j] = h;
			Sets[h * 2 + 0] = i;
			Sets[h * 2 + 1] = 6 + j;
			h++;
		}
	}


	if (h != 30) {
		cout << "h != 30" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "schlaefli_labels::init Sets:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Sets, 30, 2, FALSE /* f_tex */);
	}


	Sets2 = NEW_lint(15 * 2);
	h2 = 0;
	for (i = 0; i < 6; i++) {
		for (j = i + 1; j < 6; j++) {
			Sets2[h2 * 2 + 0] = M[i * 6 + j];
			Sets2[h2 * 2 + 1] = M[j * 6 + i];
			h2++;
		}
	}
	if (h2 != 15) {
		cout << "h2 != 15" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "Sets2:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Sets2, 15, 2, FALSE /* f_tex */);
	}



	Line_label = new std::string[28];
	Line_label_tex = new std::string[28];
	char str[1000];
	int a, b, c;

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			snprintf(str, 1000, "a_%d", i + 1);
			}
		else if (i < 12) {
			snprintf(str, 1000, "b_%d", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			snprintf(str, 1000, "c_{%d%d}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str
				<< " for line " << i << endl;
			}
		Line_label[i].assign(str);
		}
	Line_label[27].assign("d");

	for (i = 0; i < 27; i++) {
		if (i < 6) {
			snprintf(str, 1000, "a_{%d}", i + 1);
			}
		else if (i < 12) {
			snprintf(str, 1000, "b_{%d}", i - 6 + 1);
			}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			snprintf(str, 1000, "c_{%d%d}", a, b);
			}
		if (f_v) {
			cout << "creating label " << str
				<< " for line " << i << endl;
			}
		Line_label_tex[i].assign(str);
		}
	Line_label_tex[27].assign("d");




	if (f_v) {
		cout << "schlaefli_labels::init done" << endl;
	}
}


}}}

