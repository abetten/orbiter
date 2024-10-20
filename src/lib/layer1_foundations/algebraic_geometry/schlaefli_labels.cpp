/*
 * schlaefli_labels.cpp
 *
 *  Created on: May 22, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebraic_geometry {


schlaefli_labels::schlaefli_labels()
{
	Sets = NULL;
	M = NULL;
	Sets2 = NULL;


	Line_label = NULL;
	Line_label_tex = NULL;

	Tritangent_plane_label = NULL;
	Tritangent_plane_label_tex = NULL;
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

	if (Tritangent_plane_label) {
		delete [] Tritangent_plane_label;
	}
	if (Tritangent_plane_label_tex) {
		delete [] Tritangent_plane_label_tex;
	}

}


void schlaefli_labels::init(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	l1_interfaces::latex_interface L;

	int i, j, h, h2;

	if (f_v) {
		cout << "schlaefli_labels::init" << endl;
	}

	Sets = NEW_lint(30 * 2);
	M = NEW_int(6 * 6);
	Int_vec_zero(M, 6 * 6);

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
		cout << "schlaefli_labels::init h != 30" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "schlaefli_labels::init Sets:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Sets, 30, 2, false /* f_tex */);
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
		cout << "schlaefli_labels::init h2 != 15" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "schlaefli_labels::init Sets2:" << endl;
		L.print_lint_matrix_with_standard_labels(cout,
			Sets2, 15, 2, false /* f_tex */);
	}



	Line_label = new std::string[28];
	Line_label_tex = new std::string[28];
	int a, b, c;

	for (i = 0; i < 27; i++) {

		string str;

		if (i < 6) {
			str = "a_" + std::to_string(i + 1);
		}
		else if (i < 12) {
			str = "b_" + std::to_string(i - 6 + 1);
		}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			str = "c_{" + std::to_string(a) + std::to_string(b) + "}";
		}
		if (f_v) {
			cout << "schlaefli_labels::init creating label " << str
				<< " for line " << i << endl;
		}
		Line_label[i].assign(str);
	}
	Line_label[27].assign("d");

	for (i = 0; i < 27; i++) {

		string str;

		if (i < 6) {
			str = "a_{" + std::to_string(i + 1) + "}";
		}
		else if (i < 12) {
			str = "b_{" + std::to_string(i - 6 + 1) + "}";
		}
		else {
			h = i - 12;
			c = Sets2[h * 2 + 0];
			a = Sets[c * 2 + 0] + 1;
			b = Sets[c * 2 + 1] - 6 + 1;
			str = "c_{" + std::to_string(a) + std::to_string(b) + "}";
		}
		if (f_v) {
			cout << "schlaefli_labels::init creating label " << str
				<< " for line " << i << endl;
		}
		Line_label_tex[i].assign(str);
	}
	Line_label_tex[27].assign("d");

	if (f_v) {
		cout << "schlaefli_labels::init before new std::string[45];" << endl;
	}

	Tritangent_plane_label = new std::string[45];
	Tritangent_plane_label_tex = new std::string[45];

	for (i = 0; i < 45; i++) {

		if (f_v) {
			cout << "schlaefli_labels::init creating label for tritangent plane " << i << endl;
		}
		eckardt_point Eckardt_point;

		if (f_v) {
			cout << "schlaefli_labels::init before Eckardt_point.init_by_rank" << endl;
		}
		Eckardt_point.init_by_rank(i);
		if (f_v) {
			cout << "schlaefli_labels::init after Eckardt_point.init_by_rank" << endl;
		}

		string str;

		if (f_v) {
			cout << "schlaefli_labels::init before Eckardt_point.make_label" << endl;
		}
		str = Eckardt_point.make_label();
		if (f_v) {
			cout << "schlaefli_labels::init after Eckardt_point.make_label" << endl;
		}

		Tritangent_plane_label[i] = "\\pi_{" + str + "}";
		Tritangent_plane_label_tex[i] = "\\pi_{" + str + "}";
		if (f_v) {
			cout << "schlaefli_labels::init Tritangent_plane_label[i]=" << Tritangent_plane_label[i] << endl;
			cout << "schlaefli_labels::init Tritangent_plane_label_tex[i]=" << Tritangent_plane_label_tex[i] << endl;
		}
	}


	if (f_v) {
		cout << "schlaefli_labels::init done" << endl;
	}
}


}}}

