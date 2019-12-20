/*
 * orthogonal_hyperbolic.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */



#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {



// #############################################################################
// orthogonal_hyperbolic.cpp
// #############################################################################

//##############################################################################
// ranking / unranking points according to the partition:
//##############################################################################

long int orthogonal::hyperbolic_type_and_index_to_point_rk(
		long int type, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk;

	if (f_v) {
		cout << "orthogonal::hyperbolic_type_and_index_to_point_rk "
				"type" << type << "index=" << index << endl;
	}
	rk = 0;
	if (type == 4) {
		if (index >= p4) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p4" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	rk += p4;
	if (type == 6) {
		if (index >= p6) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p6" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	rk += p6;
	if (type == 3) {
		if (index >= p3) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p3" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	rk += p3;
	if (type == 5) {
		if (index >= p5) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p5" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	rk += p5;
	if (type == 2) {
		if (index >= p2) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p2" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	rk += p2;
	if (type == 1) {
		if (index >= p1) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p1" << endl;
			exit(1);
			}
		rk += index;
		goto done;
		}
	cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
			"unknown type" << endl;
	exit(1);
done:
	if (f_v) {
		cout << "orthogonal::hyperbolic_type_and_index_to_point_rk "
				"type" << type << "index=" << index << " rk=" << rk << endl;
	}
	return rk;
}

void orthogonal::hyperbolic_point_rk_to_type_and_index(
		long int rk, long int &type, long int &index)
{
	if (rk < p4) {
		type = 4;
		index = rk;
		return;
		}
	rk -= p4;
	if (rk == 0) {
		type = 6;
		index = 0;
		return;
		}
	rk--;
	if (rk < p3) {
		type = 3;
		index = rk;
		return;
		}
	rk -= p3;
	if (rk == 0) {
		type = 5;
		index = 0;
		return;
		}
	rk--;
	if (rk < p2) {
		type = 2;
		index = rk;
		return;
		}
	rk -= p2;
	if (rk < p1) {
		type = 1;
		index = rk;
		return;
		}
	cout << "error in orthogonal::hyperbolic_point_rk_to_type_and_index" << endl;
	exit(1);

}

//##############################################################################
// ranking / unranking neighbors of the favorite point:
//##############################################################################


//##############################################################################
// ranking / unranking lines:
//##############################################################################

void orthogonal::hyperbolic_unrank_line(
		long int &p1, long int &p2, long int rk, int verbose_level)
{
	if (m == 0) {
		cout << "orthogonal::hyperbolic_unrank_line "
				"Witt index zero, there is no line to unrank" << endl;
		exit(1);
		}
	if (rk < l1) {
		unrank_line_L1(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l1;
	if (rk < l2) {
		unrank_line_L2(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l2;
	if (rk < l3) {
		unrank_line_L3(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l3;
	if (rk < l4) {
		unrank_line_L4(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l4;
	if (rk < l5) {
		unrank_line_L5(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l5;
	if (rk < l6) {
		unrank_line_L6(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l6;
	if (rk < l7) {
		unrank_line_L7(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l7;
	cout << "error in orthogonal::hyperbolic_unrank_line, "
			"rk too big" << endl;
	exit(1);
}

long int orthogonal::hyperbolic_rank_line(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int pt1_type, pt2_type;
	long int pt1_index, pt2_index;
	long int line_type;
	long int rk = 0;
	long int cp1, cp2;

	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line" << endl;
	}
	if (m == 0) {
		cout << "orthogonal::hyperbolic_rank_line Witt index zero, "
				"there is no line to rank" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"p1=" << p1 << " p2=" << p2 << endl;
		}
	point_rk_to_type_and_index(p1, pt1_type, pt1_index, verbose_level);
	point_rk_to_type_and_index(p2, pt2_type, pt2_index, verbose_level);
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line pt1_type=" << pt1_type
				<< " pt2_type=" << pt2_type << endl;
		}
	line_type = line_type_given_point_types(p1, p2,
			pt1_type, pt2_type);
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"line_type=" << line_type << endl;
		}
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"before canonical_points_of_line" << endl;
		}
	canonical_points_of_line(line_type, p1, p2,
			cp1, cp2, verbose_level);
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line canonical points "
				"cp1=" << cp1 << " cp2=" << cp2 << endl;
		}
	if (line_type == 1) {
		rk += rank_line_L1(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l1;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"after adding l1=" << l1 << ", rk=" << rk << endl;
	}
	if (line_type == 2) {
		rk += rank_line_L2(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l2;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"after adding l2=" << l2 << ", rk=" << rk << endl;
	}
	if (line_type == 3) {
		rk += rank_line_L3(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l3;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"after adding l3=" << l3 << ", rk=" << rk << endl;
	}
	if (line_type == 4) {
		rk += rank_line_L4(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l4;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line after "
				"adding l4=" << l4 << ", rk=" << rk << endl;
	}
	if (line_type == 5) {
		rk += rank_line_L5(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l5;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"after adding l5=" << l5 << ", rk=" << rk << endl;
	}
	if (line_type == 6) {
		rk += rank_line_L6(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l6;
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line "
				"after adding l6=" << l6 << ", rk=" << rk << endl;
	}
	if (line_type == 7) {
		rk += rank_line_L7(cp1, cp2, verbose_level);
		goto done;
		}
	rk += l7;
	cout << "error in orthogonal::hyperbolic_rank_line "
			"illegal line_type" << endl;
	exit(1);
done:
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line done" << endl;
	}
	return rk;
}

void orthogonal::unrank_line_L1(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P4_index, P4_sub_index, P4_line_index;
	long int P4_field_element, root, i;

	if (f_v) {
		cout << "orthogonal::unrank_line_L1" << endl;
	}
	if (index >= l1) {
		cout << "error in orthogonal::unrank_line_L1 "
				"index too large" << endl;
		}
	P4_index = index / a41;
	P4_sub_index = index % a41;
	P4_line_index = P4_sub_index / (q - 1);
	P4_field_element = P4_sub_index % (q - 1);
	P4_field_element++;
	if (f_v) {
		cout << "orthogonal::unrank_line_L1 index=" << index << endl;
		}
	if (index >= l1) {
		cout << "error in orthogonal::unrank_line_L1 index too large" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "orthogonal::unrank_line_L1 P4_index=" << P4_index
				<< " P4_sub_index=" << P4_sub_index << endl;
		cout << "P4_line_index=" << P4_line_index
				<< " P4_field_element=" << P4_field_element << endl;
		}
	p1 = type_and_index_to_point_rk(4, P4_index, verbose_level);
	if (f_vv) {
		cout << "p1=" << p1 << endl;
		}
	v1[0] = 0;
	v1[1] = 0;
	unrank_N1(v1 + 2, 1, m - 2, P4_line_index);
	if (f_vvv) {
		cout << "orthogonal::unrank_line_L1 after unrank_N1" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}
	for (i = 1; i < m - 1; i++) {
		v1[2 * i] = F->mult(P4_field_element, v1[2 * i]);
		}
	if (f_vvv) {
		cout << "orthogonal::unrank_line_L1 after scaling" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}

	if (P4_index) {
		if (m > 2) {
			root = find_root_hyperbolic(P4_index, m - 1,
					verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P4_index, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->mult_matrix_matrix(v1, T1, v2, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++) {
			v2[i] = v1[i];
			}
		}
	v2[n - 2] = F->negate(P4_field_element);
	v2[n - 1] = 1;
	if (f_vv) {
		cout << "orthogonal::unrank_line_L1 before rank_Sbar" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	p2 = rank_Sbar(v2, 1, m);
	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "orthogonal::unrank_line_L1 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "orthogonal::unrank_line_L1 done" << endl;
	}
}

long int orthogonal::rank_line_L1(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P4_index, P4_sub_index, P4_line_index;
	long int P4_field_element, root, i;
	long int P4_field_element_inverse;
	long int index, a, b;

	if (f_v) {
		cout << "orthogonal::rank_line_L1" << endl;
	}
	if (f_v) {
		cout << "orthogonal::rank_line_L1 p1=" << p1 << " p2=" << p2 << endl;
		}
	P4_index = p1;
	unrank_Sbar(v2, 1, m, p2);
	if (f_vvv) {
		cout << "p2 = " << p2 << " v2=" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	if (v2[n - 1] != 1) {
		cout << "orthogonal::rank_line_L1 v2[n - 1] != 1" << endl;
		exit(1);
		}
	if (P4_index) {
		if (m > 2) {
			root = find_root_hyperbolic(
					P4_index, m - 1, verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P4_index, root, m - 1, verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->invert_matrix(T1, T2, n - 2);
		F->mult_matrix_matrix(v2, T2, v1, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++)
			v1[i] = v2[i];
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L1 mapped back to v1=" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	unrank_Sbar(v3, 1, m, 0);
	a = v1[0];
	if (a) {
		b = F->mult(a, F->negate(F->inverse(v3[0])));
		for (i = 0; i < n; i++) {
			v1[i] = F->add(F->mult(b, v3[i]), v1[i]);
			}
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L1 after Gauss reduction v1=" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	P4_field_element = F->negate(v2[n - 2]);
	if (P4_field_element == 0) {
		cout << "orthogonal::rank_line_L1: "
				"P4_field_element == 0" << endl;
		exit(1);
		}
	P4_field_element_inverse = F->inverse(P4_field_element);
	for (i = 1; i < m - 1; i++) {
		v1[2 * i] = F->mult(P4_field_element_inverse, v1[2 * i]);
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L1 after scaling" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}
	if (v1[0] != 0 || v1[1] != 0) {
		cout << "orthogonal::rank_line_L1: "
				"v1[0] != 0 || v1[1] != 0" << endl;
		exit(1);
		}
	P4_line_index = rank_N1(v1 + 2, 1, m - 2);
	if (f_vvv) {
		cout << "orthogonal::rank_line_L1 after rank_N1, "
				"P4_line_index=" << P4_line_index << endl;
		}
	P4_field_element--;
	P4_sub_index = P4_line_index * (q - 1) + P4_field_element;
	index = P4_index * a41 + P4_sub_index;
	if (f_v) {
		cout << "orthogonal::rank_line_L1 p1=" << p1
				<< " p2=" << p2 << " index=" << index << endl;
		}
	if (index >= l1) {
		cout << "error in rank_line_L1 index too large" << endl;
		cout << "index=" << index << endl;
		cout << "l1=" << l1 << endl;
		cout << "P4_index=" << P4_index << endl;
		cout << "a41=" << a41 << endl;
		cout << "P4_sub_index=" << P4_sub_index << endl;
		cout << "P4_line_index=" << P4_line_index << endl;
		cout << "P4_field_element=" << P4_field_element << endl;
		exit(1);
		}
	return index;
}

void orthogonal::unrank_line_L2(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P3_index, P3_sub_index, P3_point;
	long int root, a, b, c, d, e, i;
	long int P3_field_element;

	P3_index = index / a32;
	P3_sub_index = index % a32;
	if (f_v) {
		cout << "orthogonal::unrank_line_L2 index=" << index << endl;
		}
	if (index >= l2) {
		cout << "error in orthogonal::unrank_line_L2 index too large" << endl;
		}
	P3_point = P3_index / (q - 1);
	P3_field_element = P3_index % (q - 1);
	if (f_vv) {
		cout << "orthogonal::unrank_line_L2 P3_index=" << P3_index
				<< " P3_sub_index=" << P3_sub_index << endl;
		cout << "unrank_line_L2 P3_point=" << P3_point
				<< " P3_field_element=" << P3_field_element << endl;
		}
	unrank_Sbar(v3, 1, m - 1, P3_point);
	v3[n - 2] = 1 + P3_field_element;
	v3[n - 1] = 0;
	if (f_vv) {
		cout << "orthogonal::unrank_line_L2 before rank_Sbar  v3=" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	p1 = rank_Sbar(v3, 1, m);
	if (f_vv) {
		cout << "orthogonal::unrank_line_L2 p1=" << p1 << endl;
		}
	if (P3_sub_index == 0) {
		if (f_vv) {
			cout << "orthogonal::unrank_line_L2 case 1" << endl;
			}
		v1[0] = 0;
		v1[1] = F->negate(1);
		for (i = 2; i < n - 2; i++) {
			v1[i] = 0;
			}
		}
	else {
		P3_sub_index--;
		if (P3_sub_index < (q - 1) * T1_mm2) {
			v1[0] = 0;
			v1[1] = F->negate(1);
			a = P3_sub_index / (q - 1);
			b = P3_sub_index % (q - 1);
			if (f_vv) {
				cout << "orthogonal::unrank_line_L2 case 2, "
						"a=" << a << " b=" << b << endl;
				}
			unrank_Sbar(v1 + 2, 1, m - 2, a);
			for (i = 2; i < n - 2; i++)
				v1[i] = F->mult(v1[i], (1 + b));
			}
		else {
			P3_sub_index -= (q - 1) * T1_mm2;
			a = P3_sub_index / (q - 1);
			b = P3_sub_index % (q - 1);
			v1[0] = 1 + b;
			v1[1] = F->negate(1);
			c = F->mult(v1[0], v1[1]);
			d = F->negate(c);
			if (f_vv) {
				cout << "orthogonal::unrank_line_L2 case 3, "
						"a=" << a << " b=" << b << endl;
				}
			unrank_N1(v1 + 2, 1, m - 2, a);
			for (i = 1; i < m - 1; i++) {
				v1[2 * i] = F->mult(d, v1[2 * i]);
				}
			}
		}
	if (f_vvv) {
		cout << "orthogonal::unrank_line_L2 partner of 10...10 created:" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}
	if (P3_point) {
		if (m > 2) {
			root = find_root_hyperbolic(P3_point, m - 1,
					verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P3_point, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		if (f_vvv) {
			cout << "orthogonal::unrank_line_L2 the Siegel map is" << endl;
			print_integer_matrix(cout, T1, n - 2, n - 2);
			}
		F->mult_matrix_matrix(v1, T1, v2, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++) {
			v2[i] = v1[i];
			}
		}
	if (f_vvv) {
		cout << "orthogonal::unrank_line_L2 maps to v2=" << endl;
		int_vec_print(cout, v2, n - 2);
		cout << endl;
		}
	c = evaluate_hyperbolic_bilinear_form(v3, v2, 1, m - 1);
	if (f_vvv) {
		cout << "c=" << c << endl;
		}
	v2[n - 2] = 0;
	v2[n - 1] = F->mult(F->negate(c),F->inverse(v3[n - 2]));
	if (f_vv) {
		cout << "orthogonal::unrank_line_L2 before rank_Sbar v2=" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	e = evaluate_hyperbolic_bilinear_form(v3, v2, 1, m);
	if (e) {
		cout << "orthogonal::unrank_line_L2 error, not orthogonal" << endl;
		exit(1);
		}
	p2 = rank_Sbar(v2, 1, m);
	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "orthogonal::unrank_line_L2 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L2(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P3_index, P3_sub_index, P3_point;
	long int root, a, b, c, d, i, alpha;
	long int P3_field_element;
	long int index;

	if (f_v) {
		cout << "orthogonal::rank_line_L2 p1=" << p1 << " p2=" << p2 << endl;
		}
	unrank_Sbar(v2, 1, m, p2);
	unrank_Sbar(v3, 1, m, p1);
	if (f_vvv) {
		cout << "p1 = " << p1 << " : v3=:" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	if (v3[n - 1]) {
		cout << "orthogonal::rank_line_L2 v3[n - 1]" << endl;
		exit(1);
		}
	for (i = n - 3; i >= 0; i--) {
		if (v3[i]) {
			break;
			}
		}
	if (i < 0) {
		cout << "orthogonal::rank_line_L2 i < 0" << endl;
		exit(1);
		}
	a = v3[i];
	if (a != 1) {
		b = F->inverse(a);
		for (i = 0; i < n; i++) {
			v3[i] = F->mult(v3[i], b);
			}
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L2 after scaling, v3=:" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	P3_field_element = v3[n - 2] - 1;
	P3_point = rank_Sbar(v3, 1, m - 1);
	P3_index = P3_point * (q - 1) + P3_field_element;
	if (f_vvv) {
		cout << "orthogonal::rank_line_L2 P3_point=" << P3_point
				<< " P3_field_element=" << P3_field_element << endl;
		cout << "orthogonal::rank_line_L2 P3_index=" << P3_index << endl;
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L2 p2 = " << p2 << " : v2=:" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	c = evaluate_hyperbolic_bilinear_form(v3, v2, 1, m - 1);


	if (P3_point) {
		if (m > 2) {
			root = find_root_hyperbolic(P3_point, m - 1,
					verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P3_point, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->invert_matrix(T1, T2, n - 2);
		F->mult_matrix_matrix(v2, T2, v1, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++)
			v1[i] = v2[i];
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L2 maps back to v1=:" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}
	for (i = 2; i < n - 2; i++)
		if (v1[i])
			break;
	if (i == n - 2) {
		// case 1
		if (f_vvv) {
			cout << "orthogonal::rank_line_L2 case 1" << endl;
			}
		if (v1[0]) {
			cout << "orthogonal::rank_line_L2, case 1 v1[0]" << endl;
			exit(1);
			}
		c = v1[1];
		if (c == 0) {
			cout << "orthogonal::rank_line_L2, case 1 v1[1] == 0" << endl;
			exit(1);
			}
		if (c != F->negate(1)) {
			d = F->mult(F->inverse(c), F->negate(1));
			for (i = 0; i < n; i++) {
				v1[i] = F->mult(v1[i], d);
				}
			}
		if (f_vvv) {
			cout << "orthogonal::rank_line_L2 after scaling v1=:" << endl;
			int_vec_print(cout, v1, n);
			cout << endl;
			}
		P3_sub_index = 0;
		}
	else {
		alpha = evaluate_hyperbolic_quadratic_form(v1 + 2, 1, m - 2);
		if (alpha == 0) {
			// case 2
			if (f_vvv) {
				cout << "orthogonal::rank_line_L2 case 2" << endl;
				}
			if (v1[0]) {
				cout << "orthogonal::rank_line_L2, case 1 "
						"v1[0]" << endl;
				exit(1);
				}
			c = v1[1];
			if (c == 0) {
				cout << "orthogonal::rank_line_L2, case 1 "
						"v1[1] == 0" << endl;
				exit(1);
				}
			if (c != F->negate(1)) {
				d = F->mult(F->inverse(c), F->negate(1));
				for (i = 0; i < n; i++) {
					v1[i] = F->mult(v1[i], d);
					}
				}
			if (f_vvv) {
				cout << "orthogonal::rank_line_L2 after scaling v1=:" << endl;
				int_vec_print(cout, v1, n);
				cout << endl;
				}

			for (i = n - 3; i >= 2; i--) {
				if (v1[i])
					break;
				}
			if (i == 1) {
				cout << "orthogonal::rank_line_L2 case 2, "
						"i == 1" << endl;
				exit(1);
				}
			b = v1[i];
			c = F->inverse(b);
			for (i = 2; i < n - 2; i++)
				v1[i] = F->mult(v1[i], c);
			b--;
			if (f_vvv) {
				cout << "orthogonal::rank_line_L2 before rank_Sbar:" << endl;
				int_vec_print(cout, v1, n);
				cout << endl;
				}
			a = rank_Sbar(v1 + 2, 1, m - 2);
			if (f_vvv) {
				cout << "a=" << a << " b=" << b << endl;
				}

			P3_sub_index = 1 + a * (q - 1) + b;
			}
		else {
			if (f_vvv) {
				cout << "orthogonal::rank_line_L2 case 3" << endl;
				}
			P3_sub_index = 1 + (q - 1) * T1_mm2;
			c = v1[1];
			if (c == 0) {
				cout << "orthogonal::rank_line_L2, case 3 "
						"v1[1] == 0" << endl;
				exit(1);
				}
			if (c != F->negate(1)) {
				d = F->mult(F->inverse(c), F->negate(1));
				for (i = 0; i < n; i++) {
					v1[i] = F->mult(v1[i], d);
					}
				}
			if (f_vvv) {
				cout << "orthogonal::rank_line_L2 after scaling v1=:" << endl;
				int_vec_print(cout, v1, n);
				cout << endl;
				}
			if (v1[0] == 0) {
				cout << "orthogonal::rank_line_L2, case 3 "
						"v1[0] == 0" << endl;
				exit(1);
				}
			b = v1[0] - 1;
			d = F->inverse(v1[0]);
			for (i = 1; i < m - 1; i++) {
				v1[2 * i] = F->mult(d, v1[2 * i]);
				}
			a = rank_N1(v1 + 2, 1, m - 2);
			if (f_vvv) {
				cout << "a=" << a << " b=" << b << endl;
				}
			P3_sub_index += a * (q - 1) + b;
			}
		}
	if (f_v) {
		cout << "orthogonal::rank_line_L2 p1=" << p1 << " p2=" << p2
				<< " P3_sub_index=" << P3_sub_index << endl;
		}

	index = P3_index * a32 + P3_sub_index;

	if (f_v) {
		cout << "orthogonal::rank_line_L2 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l2) {
		cout << "error in orthogonal::rank_line_L2 index too large" << endl;
		}
	return index;
}

void orthogonal::unrank_line_L3(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P4_index, P4_sub_index, P4_line_index;
	long int P4_field_element, root, i, e;

	P4_index = index / a43;
	P4_sub_index = index % a43;
	P4_line_index = P4_sub_index / (q - 1);
	P4_field_element = P4_sub_index % (q - 1);
	P4_field_element++;
	if (f_v) {
		cout << "orthogonal::unrank_line_L3 index=" << index << endl;
		}
	if (index >= l3) {
		cout << "error in orthogonal::unrank_line_L3 index too large" << endl;
		}
	if (f_vv) {
		cout << "orthogonal::unrank_line_L3 P4_index=" << P4_index
				<< " P4_sub_index=" << P4_sub_index << endl;
		cout << "P4_line_index=" << P4_line_index
				<< " P4_field_element=" << P4_field_element << endl;
		}
	p1 = P4_index;
	unrank_Sbar(v3, 1, m, P4_index);
	if (f_vv) {
		cout << "p1=" << p1 << " v3=" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	v1[0] = 0;
	v1[1] = 0;
	unrank_Sbar(v1 + 2, 1, m - 2, P4_line_index);
	if (f_vvv) {
		cout << "orthogonal::unrank_line_L3 after unrank_Sbar" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}

	if (P4_index) {
		if (m > 2) {
			root = find_root_hyperbolic(
					P4_index, m - 1, verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P4_index, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->mult_matrix_matrix(v1, T1, v2, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++)
			v2[i] = v1[i];
		}
	v2[n - 2] = 0;
	v2[n - 1] = P4_field_element;
	if (f_vv) {
		cout << "orthogonal::unrank_line_L3 before rank_Sbar" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	e = evaluate_hyperbolic_bilinear_form(v3, v2, 1, m);
	if (e) {
		cout << "orthogonal::unrank_line_L3 error, not orthogonal" << endl;
		exit(1);
		}
	p2 = rank_Sbar(v2, 1, m);
	if (f_vv) {
		cout << "orthogonal::unrank_line_L3 p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "orthogonal::unrank_line_L3 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L3(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P4_index, P4_sub_index, P4_line_index;
	long int P4_field_element, root, i;
	long int index;
	long int a, b;

	if (f_v) {
		cout << "orthogonal::rank_line_L3 p1=" << p1 << " p2=" << p2 << endl;
		}
	unrank_Sbar(v3, 1, m, p1);
	unrank_Sbar(v2, 1, m, p2);
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 p1=" << p1 << " v3=" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 p2=" << p2 << " v2=" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	P4_index = p1;
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 P4_index=" << P4_index << endl;
		}
	if (P4_index) {
		if (m > 2) {
			root = find_root_hyperbolic(
					P4_index, m - 1, verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P4_index, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->invert_matrix(T1, T2, n - 2);
		F->mult_matrix_matrix(v2, T2, v1, 1, n - 2, n - 2,
				0 /* verbose_level */);
		v1[n - 2] = v2[n - 2];
		v1[n - 1] = v2[n - 1];
		}
	else {
		for (i = 0; i < n; i++)
			v1[i] = v2[i];
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 maps back to" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	v1[0] = 0;
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 after setting v1[0] = 0, v1=" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	if (v1[0] || v1[1]) {
		cout << "orthogonal::rank_line_L3 rank_line_L3 v1[0] || v1[1]" << endl;
		exit(1);
		}
	P4_line_index = rank_Sbar(v1 + 2, 1, m - 2);
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 P4_line_index=" << P4_line_index << endl;
		}
	for (i = n - 3; i >= 0; i--) {
		if (v1[i]) {
			break;
			}
		}
	if (i < 0) {
		cout << "orthogonal::rank_line_L3 i < 0" << endl;
		exit(1);
		}
	a = v1[i];
	if (a != 1) {
		b = F->inverse(a);
		for (i = 0; i < n; i++) {
			v1[i] = F->mult(v1[i], b);
			}
		}
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 after scaling, v1=:" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	if (v1[n - 2]) {
		cout << "orthogonal::rank_line_L3 v1[n - 2]" << endl;
		exit(1);
		}
	if (v1[n - 1] == 0) {
		cout << "orthogonal::rank_line_L3 v1[n - 1] == 0" << endl;
		exit(1);
		}
	P4_field_element = v1[n - 1] - 1;
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 P4_field_element=" << P4_field_element << endl;
		}
	P4_sub_index = P4_line_index * (q - 1) + P4_field_element;
	if (f_vvv) {
		cout << "orthogonal::rank_line_L3 P4_sub_index=" << P4_sub_index << endl;
		}
	index = (long int) P4_index * a43 + P4_sub_index;

	if (f_v) {
		cout << "orthogonal::rank_line_L3 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l3) {
		cout << "error in orthogonal::rank_line_L3 index too large" << endl;
		}
	return index;
}

void orthogonal::unrank_line_L4(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P4_index, P4_sub_index, P4_line_index;
	long int P4_field_element, root, i, e;

	P4_index = index / a44;
	P4_sub_index = index % a44;
	P4_line_index = P4_sub_index / (q - 1);
	P4_field_element = P4_sub_index % (q - 1);
	P4_field_element++;
	if (f_v) {
		cout << "unrank_line_L4 index=" << index << endl;
		}
	if (index >= l4) {
		cout << "error in unrank_line_L4 index too large" << endl;
		}
	if (f_vv) {
		cout << "unrank_line_L4 P4_index=" << P4_index
				<< " P4_sub_index=" << P4_sub_index << endl;
		cout << "P4_line_index=" << P4_line_index
				<< " P4_field_element=" << P4_field_element << endl;
		}
	p1 = P4_index;
	unrank_Sbar(v3, 1, m, P4_index);
	if (f_vv) {
		cout << "p1=" << p1 << endl;
		}
	v1[0] = 0;
	v1[1] = 0;
	unrank_Sbar(v1 + 2, 1, m - 2, P4_line_index);
	if (f_vvv) {
		cout << "after unrank_Sbar" << endl;
		int_vec_print(cout, v1, n - 2);
		cout << endl;
		}

	if (P4_index) {
		if (m > 2) {
			root = find_root_hyperbolic(
					P4_index, m - 1, verbose_level - 1);

			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P4_index, root, m - 1,
				verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->mult_matrix_matrix(v1, T1, v2, 1, n - 2, n - 2,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < n - 2; i++)
			v2[i] = v1[i];
		}
	v2[n - 2] = P4_field_element;
	v2[n - 1] = 0;
	if (f_vv) {
		cout << "before rank_Sbar" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	e = evaluate_hyperbolic_bilinear_form(v3, v2, 1, m);
	if (e) {
		cout << "error, not orthogonal" << endl;
		exit(1);
		}
	p2 = rank_Sbar(v2, 1, m);
	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "unrank_line_L4 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L4(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	long int P3_index, P3_sub_index, P3_line_index;
	long int P3_field_element, root, i;
	long int index;
	long int a, b;

	if (f_v) {
		cout << "rank_line_L4 p1=" << p1 << " p2=" << p2 << endl;
		}
	unrank_Sbar(v3, 1, m, p1);
	unrank_Sbar(v2, 1, m, p2);
	if (f_vvv) {
		cout << "p1=" << p1 << " v3=" << endl;
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	if (f_vvv) {
		cout << "p2=" << p2 << " v2=" << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	P3_index = p1;
	if (f_vvv) {
		cout << "P3_index=" << P3_index << endl;
		}
	if (P3_index) {
		if (m > 2) {
			root = find_root_hyperbolic(P3_index, m - 1, verbose_level - 1);
			Siegel_map_between_singular_points_hyperbolic(T1,
				0, P3_index, root, m - 1, verbose_level - 1);
			}
		else {
			T1[0] = T1[3] = 0;
			T1[1] = T1[2] = 1;
			}
		F->invert_matrix(T1, T2, n - 2);
		F->mult_matrix_matrix(v2, T2, v1, 1, n - 2, n - 2,
				0 /* verbose_level */);
		v1[n - 2] = v2[n - 2];
		v1[n - 1] = v2[n - 1];
		}
	else {
		for (i = 0; i < n; i++)
			v1[i] = v2[i];
		}
	if (f_vvv) {
		cout << "maps back to" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	v1[0] = 0;
	if (f_vvv) {
		cout << "after setting v1[0] = 0, v1=" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	if (v1[0] || v1[1]) {
		cout << "rank_line_L4 v1[0] || v1[1]" << endl;
		exit(1);
		}
	P3_line_index = rank_Sbar(v1 + 2, 1, m - 2);
	if (f_vvv) {
		cout << "P3_line_index=" << P3_line_index << endl;
		}
	for (i = n - 3; i >= 0; i--) {
		if (v1[i]) {
			break;
			}
		}
	if (i < 0) {
		cout << "orthogonal::rank_line_L4 i < 0" << endl;
		exit(1);
		}
	a = v1[i];
	if (a != 1) {
		b = F->inverse(a);
		for (i = 0; i < n; i++) {
			v1[i] = F->mult(v1[i], b);
			}
		}
	if (f_vvv) {
		cout << "after scaling, v1=:" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		}
	if (v1[n - 2] == 0) {
		cout << "orthogonal::rank_line_L4 v1[n - 2] == 0" << endl;
		exit(1);
		}
	if (v1[n - 1]) {
		cout << "orthogonal::rank_line_L4 v1[n - 1]" << endl;
		exit(1);
		}
	P3_field_element = v1[n - 2] - 1;
	if (f_vvv) {
		cout << "P3_field_element=" << P3_field_element << endl;
		}
	P3_sub_index = P3_line_index * (q - 1) + P3_field_element;
	if (f_vvv) {
		cout << "P3_sub_index=" << P3_sub_index << endl;
		}
	index = P3_index * a44 + P3_sub_index;

	if (f_v) {
		cout << "rank_line_L4 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l4) {
		cout << "error in rank_line_L4 index too large" << endl;
		}
	return index;
}

void orthogonal::unrank_line_L5(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "unrank_line_L5 index=" << index << endl;
		}
	if (index >= l5) {
		cout << "error in unrank_line_L5 index "
				"too large, l5=" << l5 << endl;
		}
	subspace->unrank_line(p1, p2, index, verbose_level);
	if (f_v) {
		cout << "unrank_line_L5 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L5(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int index;

	if (f_v) {
		cout << "rank_line_L5 p1=" << p1 << " p2=" << p2 << endl;
		}
	index = subspace->rank_line(p1, p2, verbose_level);
	if (f_v) {
		cout << "rank_line_L5 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l5) {
		cout << "error in rank_line_L5 index too large" << endl;
		}
	return index;
}

void orthogonal::unrank_line_L6(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "unrank_line_L6 index=" << index << endl;
		}
	if (index >= l6) {
		cout << "error in unrank_line_L6 index too large" << endl;
		}
	p1 = index;
	p2 = type_and_index_to_point_rk(5, 0, verbose_level);
	if (f_v) {
		cout << "unrank_line_L6 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L6(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int index;

	if (f_v) {
		cout << "rank_line_L6 p1=" << p1 << " p2=" << p2 << endl;
		}
	index = p1;
	if (f_v) {
		cout << "rank_line_L6 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l6) {
		cout << "error in rank_line_L6 index too large" << endl;
		}
	return index;
}

void orthogonal::unrank_line_L7(
		long int &p1, long int &p2, long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "unrank_line_L7 index=" << index << endl;
		}
	if (index >= l7) {
		cout << "error in unrank_line_L7 index too large" << endl;
		}
	p1 = index;
	p2 = type_and_index_to_point_rk(6, 0, verbose_level);
	if (f_v) {
		cout << "unrank_line_L7 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

long int orthogonal::rank_line_L7(long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int index;

	if (f_v) {
		cout << "rank_line_L7 p1=" << p1 << " p2=" << p2 << endl;
		}
	index = p1;
	if (f_v) {
		cout << "rank_line_L7 p1=" << p1 << " p2=" << p2
				<< " index=" << index << endl;
		}
	if (index >= l7) {
		cout << "error in rank_line_L7 index too large" << endl;
		}
	return index;
}


void orthogonal::hyperbolic_canonical_points_of_line(
	int line_type,
	long int pt1, long int pt2,
	long int &cpt1, long int &cpt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (line_type == 1) {
		canonical_points_L1(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 2) {
		canonical_points_L2(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 3) {
		canonical_points_L3(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 4) {
		canonical_points_L4(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 5) {
		canonical_points_L5(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 6) {
		canonical_points_L6(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 7) {
		canonical_points_L7(pt1, pt2, cpt1, cpt2);
		}
	if (f_v) {
		cout << "hyperbolic_canonical_points_of_line "
				"of type " << line_type << endl;
		cout << "pt1=" << pt1 << " pt2=" << pt2 << endl;
		cout << "cpt1=" << cpt1 << " cpt2=" << cpt2 << endl;
		}
}

void orthogonal::canonical_points_L1(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	int a, b, c, d, lambda1, lambda2, i;

	unrank_Sbar(v1, 1, m, pt1);
	unrank_Sbar(v2, 1, m, pt2);
	a = v1[n - 2];
	b = v1[n - 1];
	c = v2[n - 2];
	d = v2[n - 1];
	if (a == 0 && b == 0) {
		cpt1 = pt1;
		cpt2 = pt2;
		return;
		}
	if (c == 0 && d == 0) {
		cpt1 = pt2;
		cpt2 = pt1;
		return;
		}
	lambda1 = F->mult(c, F->negate(F->inverse(a)));
	lambda2 = F->mult(d, F->negate(F->inverse(b)));
	if (lambda1 != lambda2) {
		cout << "orthogonal::canonical_points_L1: "
				"lambda1 != lambda2" << endl;
		exit(1);
		}
	for (i = 0; i < n; i++) {
		v3[i] = F->add(F->mult(lambda1, v1[i]), v2[i]);
		}
	if (v3[n - 2] || v3[n - 1]) {
		cout << "orthogonal::canonical_points_L1: "
				"v3[n - 2] || v3[n - 1]" << endl;
		exit(1);
		}
	cpt1 = rank_Sbar(v3, 1, m);
	cpt2 = pt1;
}

void orthogonal::canonical_points_L2(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	int a, b, c, d, lambda, i;
	long int p1, p2;

	unrank_Sbar(v1, 1, m, pt1);
	unrank_Sbar(v2, 1, m, pt2);
	a = v1[n - 2];
	b = v1[n - 1];
	c = v2[n - 2];
	d = v2[n - 1];
	if (b == 0) {
		p1 = pt1;
		p2 = pt2;
		}
	else if (d == 0) {
		p1 = pt2;
		p2 = pt1;
		}
	else {
		lambda = F->mult(d, F->negate(F->inverse(b)));
		for (i = 0; i < n; i++) {
			v3[i] = F->add(F->mult(lambda, v1[i]), v2[i]);
			}
		if (v3[n - 1]) {
			cout << "orthogonal::canonical_points_L2: "
					"v3[n - 1]" << endl;
			exit(1);
			}
		p1 = rank_Sbar(v3, 1, m);
		p2 = pt1;
		}
	unrank_Sbar(v1, 1, m, p1);
	unrank_Sbar(v2, 1, m, p2);
	a = v1[n - 2];
	b = v1[n - 1];
	c = v2[n - 2];
	d = v2[n - 1];
	if (b) {
		cout << "orthogonal::canonical_points_L2: b" << endl;
		exit(1);
		}
	lambda = F->mult(c, F->negate(F->inverse(a)));
	for (i = 0; i < n; i++) {
		v3[i] = F->add(F->mult(lambda, v1[i]), v2[i]);
		}
	if (v3[n - 2]) {
		cout << "orthogonal::canonical_points_L2: "
				"v3[n - 2]" << endl;
		exit(1);
		}
	cpt1 = p1;
	cpt2 = rank_Sbar(v3, 1, m);
}

void orthogonal::canonical_points_L3(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	int a, b, c, d, lambda, i;

	unrank_Sbar(v1, 1, m, pt1);
	unrank_Sbar(v2, 1, m, pt2);
	a = v1[n - 2]; // always zero
	b = v1[n - 1];
	c = v2[n - 2]; // always zero
	d = v2[n - 1];
	if (a) {
		cout << "orthogonal::canonical_points_L3 a" << endl;
		exit(1);
		}
	if (c) {
		cout << "orthogonal::canonical_points_L3 c" << endl;
		exit(1);
		}
	if (b == 0) {
		cpt1 = pt1;
		cpt2 = pt2;
		return;
		}
	if (d == 0) {
		cpt1 = pt2;
		cpt2 = pt1;
		return;
		}
	// now b and d are nonzero

	lambda = F->mult(d, F->negate(F->inverse(b)));
	for (i = 0; i < n; i++) {
		v3[i] = F->add(F->mult(lambda, v1[i]), v2[i]);
		}
	if (v3[n - 2] || v3[n - 1]) {
		cout << "orthogonal::canonical_points_L3: "
				"v3[n - 2] || v3[n - 1]" << endl;
		exit(1);
		}
	cpt1 = rank_Sbar(v3, 1, m);
	cpt2 = pt1;
}

void orthogonal::canonical_points_L4(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	int a, b, c, d, lambda, i;

	unrank_Sbar(v1, 1, m, pt1);
	unrank_Sbar(v2, 1, m, pt2);
	a = v1[n - 2];
	b = v1[n - 1]; // always zero
	c = v2[n - 2];
	d = v2[n - 1]; // always zero
	if (b) {
		cout << "orthogonal::canonical_points_L4 b" << endl;
		exit(1);
		}
	if (d) {
		cout << "orthogonal::canonical_points_L3 d" << endl;
		exit(1);
		}
	if (a == 0) {
		cpt1 = pt1;
		cpt2 = pt2;
		return;
		}
	if (c == 0) {
		cpt1 = pt2;
		cpt2 = pt1;
		return;
		}
	// now a and c are nonzero

	lambda = F->mult(c, F->negate(F->inverse(a)));
	for (i = 0; i < n; i++) {
		v3[i] = F->add(F->mult(lambda, v1[i]), v2[i]);
		}
	if (v3[n - 2] || v3[n - 1]) {
		cout << "orthogonal::canonical_points_L4: "
				"v3[n - 2] || v3[n - 1]" << endl;
		exit(1);
		}
	cpt1 = rank_Sbar(v3, 1, m);
	cpt2 = pt1;
}

void orthogonal::canonical_points_L5(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	cpt1 = pt1;
	cpt2 = pt2;
}

void orthogonal::canonical_points_L6(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	canonical_points_L3(pt1, pt2, cpt1, cpt2);
}

void orthogonal::canonical_points_L7(
		long int pt1, long int pt2, long int &cpt1, long int &cpt2)
{
	canonical_points_L4(pt1, pt2, cpt1, cpt2);
}

int orthogonal::hyperbolic_line_type_given_point_types(
		long int pt1, long int pt2, int pt1_type, int pt2_type)
{
	if (pt1_type == 1) {
		if (pt2_type == 1) {
			return hyperbolic_decide_P1(pt1, pt2);
			}
		else if (pt2_type == 2) {
			return 2;
			}
		else if (pt2_type == 3) {
			return 2;
			}
		else if (pt2_type == 4) {
			return 1;
			}
		}
	else if (pt1_type == 2) {
		if (pt2_type == 1) {
			return 2;
			}
		else if (pt2_type == 2) {
			return hyperbolic_decide_P2(pt1, pt2);
			}
		else if (pt2_type == 3) {
			return 2;
			}
		else if (pt2_type == 4) {
			return hyperbolic_decide_P2(pt1, pt2);
			}
		else if (pt2_type == 5) {
			return 6;
			}
		}
	else if (pt1_type == 3) {
		if (pt2_type == 1)
			return 2;
		else if (pt2_type == 2) {
			return 2;
			}
		else if (pt2_type == 3) {
			return hyperbolic_decide_P3(pt1, pt2);
			}
		else if (pt2_type == 4) {
			return hyperbolic_decide_P3(pt1, pt2);
			}
		else if (pt2_type == 6) {
			return 7;
			}
		}
	else if (pt1_type == 4) {
		if (pt2_type == 1)
			return 1;
		else if (pt2_type == 2) {
			return hyperbolic_decide_P2(pt1, pt2);
			}
		else if (pt2_type == 3) {
			return hyperbolic_decide_P3(pt1, pt2);
			}
		else if (pt2_type == 4) {
			return 5;
			}
		else if (pt2_type == 5) {
			return 6;
			}
		else if (pt2_type == 6) {
			return 7;
			}
		}
	else if (pt1_type == 5) {
		if (pt2_type == 2) {
			return 6;
			}
		else if (pt2_type == 4) {
			return 6;
			}
		}
	else if (pt1_type == 6) {
		if (pt2_type == 3) {
			return 7;
			}
		else if (pt2_type == 4) {
			return 7;
			}
		}
	cout << "orthogonal::hyperbolic_line_type_given_point_types "
			"illegal combination" << endl;
	cout << "pt1_type = " << pt1_type << endl;
	cout << "pt2_type = " << pt2_type << endl;
	exit(1);
}

int orthogonal::hyperbolic_decide_P1(long int pt1, long int pt2)
{
	unrank_Sbar(v1, 1, m, pt1);
	unrank_Sbar(v2, 1, m, pt2);
	if (is_ending_dependent(v1, v2)) {
		return 1;
		}
	else {
		return 2;
		}
}

int orthogonal::hyperbolic_decide_P2(long int pt1, long int pt2)
{
	if (triple_is_collinear(pt1, pt2, pt_Q)) {
		return 6;
		}
	else {
		return 3;
		}
}

int orthogonal::hyperbolic_decide_P3(long int pt1, long int pt2)
{
	if (triple_is_collinear(pt1, pt2, pt_P)) {
		return 7;
		}
	else {
		return 4;
		}
}

int orthogonal::find_root_hyperbolic(
		long int rk2, int m, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int root, u, v;

	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic "
				"rk2=" << rk2 << " m=" << m << endl;
	}
	if (rk2 == 0) {
		cout << "orthogonal::find_root_hyperbolic: "
				"rk2 must not be 0" << endl;
		exit(1);
	}
	if (m == 1) {
		cout << "orthogonal::find_root_hyperbolic: "
				"m must not be 1" << endl;
		exit(1);
	}
	find_root_hyperbolic_xyz(rk2, m,
			find_root_x, find_root_y, find_root_z,
			verbose_level);
	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic root=" << endl;
		int_vec_print(cout, find_root_z, 2 * m);
		cout << endl;
	}

	u = evaluate_hyperbolic_bilinear_form(
			find_root_z, find_root_x, 1, m);
	if (u == 0) {
		cout << "orthogonal::find_root_hyperbolic u=" << u << endl;
		exit(1);
	}
	v = evaluate_hyperbolic_bilinear_form(
			find_root_z, find_root_y, 1, m);
	if (v == 0) {
		cout << "orthogonal::find_root_hyperbolic v=" << v << endl;
		exit(1);
	}
	root = rank_Sbar(find_root_z, 1, m);
	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic root=" << root << endl;
	}
	return root;
}

void orthogonal::find_root_hyperbolic_xyz(
		long int rk2, int m, int *x, int *y, int *z,
		int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = 2 * m;
	int i;
	int y2_minus_y3, minus_y1, y3_minus_y2, a, a2;

	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic_xyz "
				"rk2=" << rk2 << " m=" << m << endl;
	}
	for (i = 0; i < d; i++) {
		x[i] = 0;
		z[i] = 0;
	}
	x[0] = 1;

	unrank_Sbar(y, 1, m, rk2);
	if (f_vv) {
		cout << "orthogonal::find_root_hyperbolic_xyz y=" << endl;
		int_vec_print(cout, y, 2 * m);
		cout << endl;
	}
	if (y[0]) {
		if (f_vv) {
			cout << "detected y[0] is nonzero" << endl;
		}
		z[1] = 1;
		if (f_v) {
			cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
			int_vec_print(cout, z, 2 * m);
			cout << endl;
		}
		return;
	}
	if (f_vv) {
		cout << "detected y[0] is zero" << endl;
	}
	if (y[1] == 0) {
		if (f_vv) {
			cout << "detected y[1] is zero" << endl;
		}
		for (i = 2; i < d; i++) {
			if (y[i]) {
				if (f_vv) {
					cout << "detected y[" << i << "] is nonzero" << endl;
				}
				if (EVEN(i)) {
					z[1] = 1;
					z[i + 1] = 1;
					if (f_v) {
						cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
						int_vec_print(cout, z, 2 * m);
						cout << endl;
					}
					return;
				}
				else {
					z[1] = 1;
					z[i - 1] = 1;
					if (f_v) {
						cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
						int_vec_print(cout, z, 2 * m);
						cout << endl;
					}
					return;
				}
			}
		}
		cout << "orthogonal::find_root_hyperbolic_xyz error: y is zero vector" << endl;
	}
	if (f_vv) {
		cout << "detected y[1] is nonzero" << endl;
	}

	// now: y[0] = 0, y[1] <> 0

	// try to choose z[0] = z[1] = 1:
	y2_minus_y3 = F->add(y[2], F->negate(y[3]));
	minus_y1 = F->negate(y[1]);
	if (minus_y1 != y2_minus_y3) {
		if (f_vv) {
			cout << "detected -y[1] != y[2] - y[3]" << endl;
		}
		z[0] = 1;
		z[1] = 1;
		z[2] = F->negate(1);
		z[3] = 1;
		// z = (1,1,-1,1) is singular
		// <x,z> = 1
		// <y,z> = y[1] - y[3] + y[2] = 0
		// iff -y[1] = y[2] - y[3]
		// which is not the case
		if (f_v) {
			cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
			int_vec_print(cout, z, 2 * m);
			cout << endl;
		}
		return;
	}
	if (f_vv) {
		cout << "detected -y[1] = y[2] - y[3]" << endl;
	}
	y3_minus_y2 = F->add(y[3], F->negate(y[2]));
	if (minus_y1 != y3_minus_y2) {
		if (f_vv) {
			cout << "detected -y[1] != y[3] - y[2]" << endl;
		}
		z[0] = 1;
		z[1] = 1;
		z[2] = 1;
		z[3] = F->negate(1);
		// z = (1,1,1,-1) is singular
		// <x,z> = 1
		// <y,z> = y[1] + y[3] - y[2] = 0
		// iff -y[1] = y[3] - y[2]
		// which is not the case
		if (f_v) {
			cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
			int_vec_print(cout, z, 2 * m);
			cout << endl;
		}
		return;
	}
	if (f_vv) {
		cout << "detected -y[1] = y[2] - y[3] = y[3] - y[2]" << endl;
	}

	// now -y[1] = y[2] - y[3] = y[3] - y[2],
	// i.e., we are in characteristic 2
	// i.e., y[1] = y[2] + y[3]

	if (F->q == 2) {
		if (f_vv) {
			cout << "detected field of order 2" << endl;
		}
		// that is, y[1] = 1 and y[3] = 1 + y[2]
		if (y[2] == 0) {
			if (f_vv) {
				cout << "detected y[2] == 0" << endl;
			}
			// that is, y[3] = 1
			z[1] = 1;
			z[2] = 1;
			// z=(0,1,1,0) is singular
			// <x,z> = 1
			// <y,z> = y[0] + y[3] = 0 + 1 = 1
			if (f_v) {
				cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
				int_vec_print(cout, z, 2 * m);
				cout << endl;
			}
			return;
		}
		else if (y[3] == 0) {
			if (f_vv) {
				cout << "detected y[3] == 0" << endl;
			}
			// that is, y[2] = 1
			z[1] = 1;
			z[3] = 1;
			// z=(0,1,0,1) is singular
			// <x,z> = 1
			// <y,z> = y[0] + y[2] = 0 + 1 = 1
			if (f_v) {
				cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
				int_vec_print(cout, z, 2 * m);
				cout << endl;
			}
			return;
		}
		cout << "orthogonal::find_root_hyperbolic_xyz error "
				"neither y2 nor y3 is zero" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "detected field has at least 4 elements" << endl;
	}
	// now the field has at least 4 elements
	a = 3;
	a2 = F->mult(a, a);
	z[0] = a2;
	z[1] = 1;
	z[2] = a;
	z[3] = a;
	// z=(alpha^2,1,alpha,alpha) is singular
	// <x,z> = alpha^2
	// <y,z> = y[0] + alpha^2 y[1] + alpha (y[2] + y[3])
	// = alpha^2 y[1] + alpha (y[2] + y[3])
	// = alpha^2 y[1] + alpha y[1]
	// = (alpha^2 + alpha) y[1]
	// = alpha (alpha + 1) y[1]
	// which is nonzero
	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic_xyz z=" << endl;
		int_vec_print(cout, z, 2 * m);
		cout << endl;
	}
	if (f_v) {
		cout << "orthogonal::find_root_hyperbolic_xyz done" << endl;
	}
}

int orthogonal::evaluate_hyperbolic_quadratic_form(
		int *v, int stride, int m)
{
	int alpha = 0, beta, i;

	for (i = 0; i < m; i++) {
		beta = F->mult(v[2 * i * stride], v[(2 * i + 1) * stride]);
		alpha = F->add(alpha, beta);
		}
	return alpha;
}

int orthogonal::evaluate_hyperbolic_bilinear_form(
		int *u, int *v, int stride, int m)
{
	int alpha = 0, beta1, beta2, i;

	for (i = 0; i < m; i++) {
		beta1 = F->mult(u[2 * i * stride], v[(2 * i + 1) * stride]);
		beta2 = F->mult(u[(2 * i + 1) * stride], v[2 * i * stride]);
		alpha = F->add(alpha, beta1);
		alpha = F->add(alpha, beta2);
		}
	return alpha;
}



}}
