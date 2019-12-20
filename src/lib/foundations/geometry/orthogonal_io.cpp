/*
 * orthogonal_io.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */


#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void orthogonal::list_points_by_type(int verbose_level)
{
	int t;

	for (t = 1; t <= nb_point_classes; t++) {
		list_points_of_given_type(t, verbose_level);
		}
}

void orthogonal::list_points_of_given_type(int t, int verbose_level)
{
	long int i, j, rk, u;

	cout << "points of type P" << t << ":" << endl;
	for (i = 0; i < P[t - 1]; i++) {
		rk = type_and_index_to_point_rk(t, i, verbose_level);
		cout << i << " : " << rk << " : ";
		unrank_point(v1, 1, rk, verbose_level - 1);
		int_vec_print(cout, v1, n);
		point_rk_to_type_and_index(rk, u, j, verbose_level);
		cout << " : " << u << " : " << j << endl;
		if (u != t) {
			cout << "type wrong" << endl;
			exit(1);
			}
		if (j != i) {
			cout << "index wrong" << endl;
			exit(1);
			}
		}
	cout << endl;
}

void orthogonal::list_all_points_vs_points(int verbose_level)
{
	int t1, t2;

	for (t1 = 1; t1 <= nb_point_classes; t1++) {
		for (t2 = 1; t2 <= nb_point_classes; t2++) {
			list_points_vs_points(t1, t2, verbose_level);
			}
		}
}

void orthogonal::list_points_vs_points(int t1, int t2, int verbose_level)
{
	int i, j, rk1, rk2, u, cnt;

	cout << "lines between points of type P" << t1
			<< " and points of type P" << t2 << endl;
	for (i = 0; i < P[t1 - 1]; i++) {
		rk1 = type_and_index_to_point_rk(t1, i, verbose_level);
		cout << i << " : " << rk1 << " : ";
		unrank_point(v1, 1, rk1, verbose_level - 1);
		int_vec_print(cout, v1, n);
		cout << endl;
		cout << "is incident with:" << endl;

		cnt = 0;

		for (j = 0; j < P[t2 - 1]; j++) {
			rk2 = type_and_index_to_point_rk(t2, j, verbose_level);
			unrank_point(v2, 1, rk2, verbose_level - 1);

			//cout << "testing: " << j << " : " << rk2 << " : ";
			//int_vec_print(cout, v2, n);
			//cout << endl;

			u = evaluate_bilinear_form(v1, v2, 1);
			if (u == 0 && rk2 != rk1) {
				//cout << "yes" << endl;
				if (test_if_minimal_on_line(v2, v1, v3)) {
					cout << cnt << " : " << j << " : " << rk2 << " : ";
					int_vec_print(cout, v2, n);
					cout << endl;
					cnt++;
					}
				}
			}
		cout << endl;
		}

}


void orthogonal::print_schemes()
{
	int i, j;


	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
		}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << A[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;
	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << setw(7) << L[j];
		}
	cout << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << setw(7) << B[i * nb_line_classes + j];
		}
		cout << endl;
	}
	cout << endl;

	cout << "\\begin{array}{r||*{" << nb_line_classes << "}{r}}" << endl;
	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << " & " << setw(7) << L[j];
		}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\hline" << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << " & " << setw(7) << A[i * nb_line_classes + j];
		}
		cout << "\\\\" << endl;
	}
	cout << "\\end{array}" << endl;
	cout << "\\begin{array}{r||*{" << nb_line_classes << "}{r}}" << endl;
	cout << "       ";
	for (j = 0; j < nb_line_classes; j++) {
		cout << " & " << setw(7) << L[j];
		}
	cout << "\\\\" << endl;
	cout << "\\hline" << endl;
	cout << "\\hline" << endl;
	for (i = 0; i < nb_point_classes; i++) {
		cout << setw(7) << P[i];
		for (j = 0; j < nb_line_classes; j++) {
			cout << " & " << setw(7) << B[i * nb_line_classes + j];
		}
		cout << "\\\\" << endl;
	}
	cout << "\\end{array}" << endl;
}



}}



