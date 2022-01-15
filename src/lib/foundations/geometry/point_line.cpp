// point_line.cpp
// Anton Betten
//
// started: 2001



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


int point_line::is_desarguesian_plane(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int line = 0;
	int *pts_on_line;
	int u, v, o, e = 0, loe, p0, a, i, y1, y2;
	int slope, b, x, f_slope, f_b, f_x;
	int f_found_quadrangle = FALSE;
	int aa;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "is_desarguesian_plane plane_order ="
				<< plane_order << endl;
		}
	pts_on_line = NEW_int(plane_order + 1);
	plane_get_points_on_line(line, pts_on_line);
	if (f_vv) {
		cout << "line " << line << " ";
		Orbiter->Int_vec.set_print(cout, pts_on_line, plane_order + 1);
		}
	u = pts_on_line[0];
	v = pts_on_line[1];
	if (f_vv) {
		cout << "choosing points u=" << u << " and v=" << v
				<< " on line " << line << endl;
		}
	for (o = 0; o < nb_pts; o++) {
		if (o == u)
			continue;
		if (o == v)
			continue;
		for (e = 0; e < nb_pts; e++) {
			if (e == u)
				continue;
			if (e == v)
				continue;
			if (e == o)
				continue;
			loe = plane_line_through_two_points(o, e);
			if (loe == line)
				continue;
			p0 = plane_line_intersection(loe, line);
			if (p0 == u)
				continue;
			if (p0 == v)
				continue;
			if (p0 == o)
				continue;
			if (p0 == e)
				continue;
			// now: (o,e,u,v) is a quadrangle with u and v both on line
			f_found_quadrangle = TRUE;
			break;
			}
		if (f_found_quadrangle)
			break;
		}
	if (!f_found_quadrangle) {
		cout << "did not find a quadrangle, something is wrong" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "found quadrangle (" << o << "," << e
				<< "," << u << "," << v << ")";
		}
	coordinatize_plane(o, e, u, v, MOLS, f_vv);
	if (f_vv) {
		cout << "plane coordinatized" << endl;
		}
	plane_prime = NT.smallest_primedivisor(plane_order);
	aa = plane_order;
	plane_exponent = 0;
	while (aa > 1) {
		plane_exponent++;
		aa /= plane_prime;
		}
	if (NT.i_power_j(plane_prime, plane_exponent) != plane_order) {
		cout << "plane order not a power of a prime." << endl;
		return FALSE;
		}
	if (plane_exponent > 1) {
		if (f_v) {
			cout << "plane order is not prime:" << endl;
			cout << plane_order << " = " << plane_prime << "^"
					<< plane_exponent << endl;
			}
		return identify_field_not_of_prime_order(verbose_level);
		}
	else {
		if (f_v) {
			cout << "plane order is prime" << endl;
			}
		field_element[0] = 0;
		field_element_inv[0] = 0;
		field_element[1] = 1;
		field_element_inv[1] = 1;
		a = 1;
		for (i = 2; i < plane_order; i++) {
			a = MOLSsxb(0, a, 1) % plane_order;
			field_element[i] = a;
			field_element_inv[a] = i;
			cout << "field element " << i << " = " << a << endl;
			}
		for (slope = 1; slope < plane_order; slope++) {
			f_slope = field_element[slope];
			
			for (b = 0; b < plane_order; b++) {
				f_b = field_element[b];
				
				for (x = 0; x < plane_order; x++) {
					f_x = field_element[x];
					
					// compute slope * x + b from the field, 
					// i.e. MOLS 0 (addition) and plane_order (multiplication)
					y1 = (slope * x + b) % plane_order;
						//MOLSaddition(MOLSmultiplication(slope, x), b);
					a = MOLSsxb(f_slope, f_x, f_b);
					y2 = field_element_inv[a];
					if (y1 != y2) {
						if (f_v) {
							cout << "the plane is not desarguesian:" << endl;
							cout << "slope = " << slope << endl;
							cout << "f_slope = " << f_slope << endl;
							cout << "b = " << b << endl;
							cout << "f_b = " << f_b << endl;
							cout << "x = " << x << endl;
							cout << "f_x = " << f_x << endl;
							cout << "y1 = " << y1 << endl;
							cout << "y2 = " << y2 << endl;
							cout << "a = " << a << endl;
							}
						return FALSE;
						}
					}
				}
			}
		}
	if (f_v) {
		cout << "the plane is desarguesian" << endl;
		}	
	return TRUE;
}

int point_line::identify_field_not_of_prime_order(int verbose_level)
{
	cout << "point_line::identify_field_not_of_prime_order "
			"not yet implemented" << endl;
	exit(1);
#if 0
	int m, n;  // the size of the incidence matrix
	int d, d2;
	int i, j, k, ii, jj, kk, a;
	int nb_inc = 0; // the number of incidences
	int mpn;
	int *M1;
	int *canonical_form1;
	int *canonical_form_inv1;
	permutation_group_generators Aut_gens1;
	//int *Aut_gens1;
	//int nb_Aut_gens1;
	longinteger_object ago1;
	int *M2;
	int *canonical_form2;
	int *canonical_form_inv2;
	permutation_group_generators Aut_gens2;
	//int *Aut_gens2;
	//int nb_Aut_gens2;
	longinteger_object ago2;
	int row_parts[5];
	int col_parts[3];
	int nb_row_parts = 3;
	int nb_col_parts = 3;
	finite_field F;
	int ret = TRUE;
	
	d = plane_order;
	d2 = d * d;
	m = 3 * d;
	n = d + 2 * d2;
	mpn = m + n;
	row_parts[0] = d;
	row_parts[1] = d;
	row_parts[2] = d;
	col_parts[0] = d;
	col_parts[1] = d2;
	col_parts[2] = d2;
	M1 = NEW_int(m * n);
	canonical_form1 = NEW_int(mpn);
	canonical_form_inv1 = NEW_int(mpn);
	//Aut_gens1 = NEW_int(mpn * mpn);
	M2 = NEW_int(m * n);
	canonical_form2 = NEW_int(mpn);
	canonical_form_inv2 = NEW_int(mpn);
	//Aut_gens2 = NEW_int(mpn * mpn);

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M1[i * n + j] = 0;
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < d; j++) {
			M1[(i * d + j) * n + j] = 1;
			nb_inc++;
			}
		}

	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			a = MOLSsxb(0, i, j); // attention, we need the digits from 0 to d-1 !!
			M1[i * n + d + i * d + j] = 1;
			nb_inc++;
			M1[(d + j) * n + d + i * d + j] = 1;
			nb_inc++;
			M1[(2 * d + a) * n + d + i * d + j] = 1;
			nb_inc++;
			}
		}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			a = MOLSsxb(plane_order, i, j); // attention, we need the digits from 0 to d-1 !!
			M1[i * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			M1[(d + j) * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			M1[(2 * d + a) * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			}
		}
	if (f_v) {
		cout << "computing canonical labelling of the planar ternary ring of order " << plane_order << endl;
		}
	
#if 0
	compute_canonical_labeling_of_01matrix(M1, m, n, 
		TRUE, row_parts, nb_row_parts, col_parts, nb_col_parts, 
		canonical_form1, canonical_form_inv1, 
		Aut_gens1, 
		FALSE, FALSE, FALSE, FALSE);
#endif

	Aut_gens1.compute_group_order(ago1);
	if (ago1.as_int() != plane_exponent) {
		cout << "the planar ternanry ring does not have the right number of automorphisms, not desarguesian" << endl;
		ret = FALSE;
		goto finish;
		}

	if (f_v) {
		cout << "setting up the field of order " << plane_order << endl;
		}
	F.init(plane_order, FALSE, FALSE);
	nb_inc = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			M2[i * n + j] = 0;
			}
		}
	for (i = 0; i < 3; i++) {
		for (j = 0; j < d; j++) {
			M2[(i * d + j) * n + j] = 1;
			nb_inc++;
			}
		}

	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			a = F.add(i, j); // attention, we need the digits from 0 to d-1 !!
			M2[i * n + d + i * d + j] = 1;
			nb_inc++;
			M2[(d + j) * n + d + i * d + j] = 1;
			nb_inc++;
			M2[(2 * d + a) * n + d + i * d + j] = 1;
			nb_inc++;
			}
		}
	for (i = 0; i < d; i++) {
		for (j = 0; j < d; j++) {
			a = F.mult(i, j); // attention, we need the digits from 0 to d-1 !!
			M2[i * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			M2[(d + j) * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			M2[(2 * d + a) * n + d + d2 + i * d + j] = 1;
			nb_inc++;
			}
		}
	if (f_v) {
		cout << "computing canonical labelling of the field of order " << plane_order << endl;
		}
#if 0
	compute_canonical_labeling_of_01matrix(M2, m, n, 
		TRUE, row_parts, nb_row_parts, col_parts, nb_col_parts, 
		canonical_form2, canonical_form_inv2, 
		Aut_gens2, 
		FALSE, FALSE, FALSE, FALSE);
#endif
	Aut_gens2.compute_group_order(ago2);
	if (ago2.as_int() != plane_exponent) {
		cout << "the field does not have the right number of automorphisms, something is wrong" << endl;
		exit(1);
		}
	for (i = 0; i < d; i++) {
		a = canonical_form1[canonical_form_inv2[i]];
		field_element[i] = a;
		field_element_inv[a] = i;
		}
	if (f_vv) {
		for (i = 0; i < plane_order; i++) {
			cout << "field element " << i << " = " << field_element[i] << endl;
			}
		}
	if (field_element[0] != 0) {
		cout << "field_element[0] != 0, something is wrong" << endl;
		exit(1);
		}
	if (field_element[1] != 1) {
		cout << "field_element[1] != 1, something is wrong" << endl;
		exit(1);
		}
		
	// we double check everything once more:
	for (i = 0; i < plane_order; i++) {
		ii = field_element[i];
		for (j = 0; j < plane_order; j++) {
			jj = field_element[j];
			k = F.add(i, j);
			kk = MOLSsxb(0, ii, jj);
			if (field_element[k] != kk) {
				cout << "i=" << i << " j=" << j << " i+j=" << k << endl;
				cout << "ii=" << ii << " jj=" << jj << " ii+jj=" << kk << endl;
				cout << "but field_element[k]=" << field_element[k] << endl;
				exit(1);
				}
			k = F.mult(i, j);
			kk = MOLSsxb(plane_order, ii, jj);
			if (field_element[k] != kk) {
				cout << "i=" << i << " j=" << j << " i*j=" << k << endl;
				cout << "ii=" << ii << " jj=" << jj << " ii*jj=" << kk << endl;
				cout << "but field_element[k]=" << field_element[k] << endl;
				exit(1);
				}
			}
		}
	// at this point, the fields are really isomorphic!!!
	
	ret = TRUE;
finish:
	FREE_int(M1);
	FREE_int(canonical_form1);
	FREE_int(canonical_form_inv1);
	//delete [] Aut_gens1;
	FREE_int(M2);
	FREE_int(canonical_form2);
	FREE_int(canonical_form_inv2);
	//delete [] Aut_gens2;
	
	return ret;
#endif
}

void point_line::init_projective_plane(int order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;
	
	f_plane_data_computed = FALSE;
	if (f_v) {
		cout << "computing data for a projective plane of order "
				<< order << " with " << m << " points" << endl;
		}
	plane_order = order;
	nb_pts = m;
	plane.points_on_lines = NEW_int(nb_pts * (plane_order + 1));
	plane.line_through_two_points = NEW_int(nb_pts * nb_pts);
	for (i = 0; i < nb_pts; i++) {
		plane_get_points_on_line(i,
				plane.points_on_lines + i * (plane_order + 1));
		}
	for (i = 0; i < nb_pts; i++) {
		plane.line_through_two_points[i * nb_pts + i] = -1;
		for (j = i + 1; j < nb_pts; j++) {
			h = plane_line_through_two_points(i, j);
			plane.line_through_two_points[i * nb_pts + j] = h;
			plane.line_through_two_points[j * nb_pts + i] = h;
			}
		}
	
	dual_plane.points_on_lines = NEW_int(nb_pts * (plane_order + 1));
	dual_plane.line_through_two_points = NEW_int(nb_pts * nb_pts);
	for (i = 0; i < nb_pts; i++) {
		plane_get_lines_through_point(i,
				dual_plane.points_on_lines + i * (plane_order + 1));
		}
	for (i = 0; i < nb_pts; i++) {
		dual_plane.line_through_two_points[i * nb_pts + i] = -1;
		for (j = i + 1; j < nb_pts; j++) {
			h = plane_line_intersection(i, j);
			dual_plane.line_through_two_points[i * nb_pts + j] = h;
			dual_plane.line_through_two_points[j * nb_pts + i] = h;
			}
		}
	
	if (f_v) {
		cout << "allocating data for the coordinatization" << endl;
		}
	
	pt_labels = NEW_int(m);
	points = NEW_int(m);
	// pt_labels and points are mutually inverse permutations of {0,1,...,m-1}
	
	pts_on_line_x_eq_y = NEW_int(plane_order + 1);
	pts_on_line_x_eq_y_labels = NEW_int(plane_order + 1);
	lines_through_X = NEW_int(plane_order + 1);
	lines_through_Y = NEW_int(plane_order + 1);
	pts_on_line = NEW_int(plane_order + 1);
	MOLS = NEW_int((plane_order + 1) * plane_order * plane_order);
	field_element = NEW_int(plane_order);
	field_element_inv = NEW_int(plane_order);
	if (f_v) {
		cout << "finished" << endl;
	}
	f_plane_data_computed = TRUE;
}

void point_line::free_projective_plane()
{
	if (f_plane_data_computed) {
		FREE_int(plane.points_on_lines);
		FREE_int(plane.line_through_two_points);
		FREE_int(dual_plane.points_on_lines);
		FREE_int(dual_plane.line_through_two_points);
		FREE_int(pt_labels);
		FREE_int(points);
		FREE_int(pts_on_line_x_eq_y);
		FREE_int(pts_on_line_x_eq_y_labels);
		FREE_int(lines_through_X);
		FREE_int(lines_through_Y);
		FREE_int(pts_on_line);
		FREE_int(MOLS);
		FREE_int(field_element);
		FREE_int(field_element_inv);
		f_plane_data_computed = FALSE;
	}
}

void point_line::plane_report(ostream &ost)
{
	int i, j, h;
	
	
	ost << "lines on points:" << endl;
	ost << "pt : lines through pt" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & ";
		for (j = 0; j <= plane_order; j++) {
			ost << dual_plane.points_on_lines[i * (plane_order + 1) + j];
			if (j < plane_order)
				ost << ", ";
			}
		ost << endl;
		}
	ost << "points on lines:" << endl;
	ost << "line : pts on line" << endl;
	for (i = 0; i < m; i++) {
		ost << i << " & ";
		for (j = 0; j <= plane_order; j++) {
			ost << plane.points_on_lines[i * (plane_order + 1) + j];
			if (j < plane_order)
				ost << ", ";
			}
		ost << endl;
		}
	ost << "lines through two points:" << endl;
	ost << "pt_1, pt_2 : line through pt_1 and pt_2" << endl;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			h = plane.line_through_two_points[i * nb_pts + j];
			ost << i << ", " << j << " & " << h << endl;
			}
		}
	ost << "intersections of lines:" << endl;
	ost << "line_1, line_2 : point of intersection" << endl;
	for (i = 0; i < m; i++) {
		for (j = i + 1; j < m; j++) {
			h = dual_plane.line_through_two_points[i * nb_pts + j];
			ost << i << ", " << j << " & " << h << endl;
			}
		}
	
}

int point_line::plane_line_through_two_points(int pt1, int pt2)
{
	if (pt1 == pt2) {
		cout << "point_line::plane_line_through_two_points "
				"pts are equal" << endl;
		exit(1);
		}
	if (f_plane_data_computed) {
		return plane.line_through_two_points[pt1 * nb_pts + pt2];
		}
	else {
		int j;
	
		for (j = 0; j < n; j++) {
			if (a[pt1 * n + j] && a[pt2 * n + j])
				return j;
			}
		cout << "point_line::plane_line_through_two_points "
				"there is no line through pt1=" << pt1
				<< " and pt2=" << pt2 << endl;
		exit(1);
		}
}

int point_line::plane_line_intersection(int line1, int line2)
{
	if (line1 == line2) {
		cout << "point_line::plane_line_intersection "
				"lines are equal" << endl;
		exit(1);
		}
	if (f_plane_data_computed) {
		return dual_plane.line_through_two_points[line1 * nb_pts + line2];
		}
	else {
		int i;
	
		for (i = 0; i < m; i++) {
			if (a[i * n + line1] && a[i * n + line2])
				return i;
			}
		cout << "point_line::plane_line_intersection "
				"there is no common point to line1=" << line1
				<< " and line2=" << line2 << endl;
		exit(1);
		}
}

void point_line::plane_get_points_on_line(int line, int *pts)
{
	if (f_plane_data_computed) {
		int j;
		
		for (j = 0; j <= plane_order; j++) {
			pts[j] = plane.points_on_lines[line * (plane_order + 1) + j];
			}
		}
	else {
		int i, l;
	
		l = 0;
		for (i = 0; i < m; i++) {
			if (a[i * n + line]) {
				pts[l++] = i;
				}
			}
		if (l != plane_order + 1) {
			cout << "point_line::plane_get_points_on_line "
					"l != plane_order + 1" << endl;
			exit(1);
			}
		}
}

void point_line::plane_get_lines_through_point(int pt, int *lines)
{
	if (f_plane_data_computed) {
		int j;
		
		for (j = 0; j <= plane_order; j++) {
			lines[j] = dual_plane.points_on_lines[pt * (plane_order + 1) + j];
			}
		}
	else {
		int j, l;
	
		l = 0;
		for (j = 0; j < m; j++) {
			if (a[pt * n + j]) {
				lines[l++] = j;
				}
			}
		if (l != plane_order + 1) {
			cout << "point_line::plane_get_lines_through_point "
					"l != plane_order + 1" << endl;
			exit(1);
			}
		}
}

int point_line::plane_points_collinear(int pt1, int pt2, int pt3)
{
	int line;
	line = plane_line_through_two_points(pt1, pt2);
	if (a[pt3 * n + line])
		return TRUE;
	else
		return FALSE;
}

int point_line::plane_lines_concurrent(int line1, int line2, int line3)
{
	int pt;
	pt = plane_line_intersection(line1, line2);
	if (a[pt * n + line3])
		return TRUE;
	else
		return FALSE;
}

int point_line::plane_first_quadrangle(int &pt1, int &pt2, int &pt3, int &pt4)
{
	// int v = m;
	int pts[4];
	
	pts[0] = pt1;
	pts[1] = pt2;
	pts[2] = pt3;
	pts[3] = pt4;
	if (!plane_quadrangle_first_i(pts, 0)) {
		cout << "point_line::plane_first_quadrangle "
				"no quadrangle" << endl;
		exit(1);
		}
	else {
		pt1 = pts[0];
		pt2 = pts[1];
		pt3 = pts[2];
		pt4 = pts[3];
		return TRUE;
		}
}

int point_line::plane_next_quadrangle(int &pt1, int &pt2, int &pt3, int &pt4)
{
	// int v = m;
	int pts[4];
	
	pts[0] = pt1;
	pts[1] = pt2;
	pts[2] = pt3;
	pts[3] = pt4;
	if (!plane_quadrangle_next_i(pts, 0)) {
		return FALSE;
		}
	else {
		pt1 = pts[0];
		pt2 = pts[1];
		pt3 = pts[2];
		pt4 = pts[3];
		return TRUE;
		}
}

int point_line::plane_quadrangle_first_i(int *pt, int i)
{
	int v = m, pt0;
	
	if (i > 0)
		pt0 = pt[i - 1] + 1;
	else
		pt0 = 0;
	for (pt[i] = pt0; pt[i] < v; pt[i]++) {
		if (i == 2) {
			if (plane_points_collinear(pt[0], pt[1], pt[2]))
				continue;
			}
		else if (i == 3) {
			if (plane_points_collinear(pt[0], pt[1], pt[3]))
				continue;
			if (plane_points_collinear(pt[0], pt[2], pt[3]))
				continue;
			if (plane_points_collinear(pt[1], pt[2], pt[3]))
				continue;
			}
			
		if (i == 3)
			return TRUE;
		else {
			if (!plane_quadrangle_first_i(pt, i + 1))
				continue;
			else
				return TRUE;
			}
		}
	return FALSE;
}

int point_line::plane_quadrangle_next_i(int *pt, int i)
{
	int v = m;
	
	if (i < 3) {
		if (plane_quadrangle_next_i(pt, i + 1))
			return TRUE;
		}
	while (pt[i] < v) {
		pt[i]++;
		if (i == 2) {
			if (plane_points_collinear(pt[0], pt[1], pt[2]))
				continue;
			}
		else if (i == 3) {
			if (plane_points_collinear(pt[0], pt[1], pt[3]))
				continue;
			if (plane_points_collinear(pt[0], pt[2], pt[3]))
				continue;
			if (plane_points_collinear(pt[1], pt[2], pt[3]))
				continue;
			}
		if (i == 3)
			return TRUE;
		else {
			if (!plane_quadrangle_first_i(pt, i + 1))
				continue;
			else
				return TRUE;
			}
		}
	return FALSE;
}

void point_line::coordinatize_plane(int O, int I, int X, int Y,
		int *MOLS, int verbose_level)
// needs pt_labels, points, pts_on_line_x_eq_y, pts_on_line_x_eq_y_labels, 
// lines_through_X, lines_through_Y, pts_on_line, MOLS to be allocated
{
	int f_v = (verbose_level >= 1);
	int i, j, ii, jj, x, y, l, pt, line, pt2, pt3, b, pt_label;
	int slope;
	

	quad_O = O;
	quad_I = I;
	quad_X = X;
	quad_Y = Y;
	line_x_eq_y = plane_line_through_two_points(O, I);
	line_infty = plane_line_through_two_points(X, Y);
	line_x_eq_0 = plane_line_through_two_points(O, Y);
	line_y_eq_0 = plane_line_through_two_points(O, X);
	
	if (f_v) {
		cout << "coordinatizing plane with O=" << O
				<< " I=" << I << " X=" << X << " Y=" << Y << endl;
		cout << "plane_order=" << plane_order << endl;
		cout << "m=" << m << endl;
		}
	
	// label the points on the line y = x:
	// first we label them as points in the coordinatized plane, 
	// second we label them as elements from 0 to plane_order
	// O = (0,0) = 0
	// I = (1,1) = 1
	// C = (1)   = plane_order
	// the remaining points are labeled (l,l)
	// in arbitrary order, 2 \le l < plane_order
	
	
	plane_get_points_on_line(line_x_eq_y, pts_on_line_x_eq_y);
	quad_C = plane_line_intersection(line_x_eq_y,
			line_infty); // the point at infinity (1)
	l = 2;
	for (i = 0; i <= plane_order; i++) {
		pt = pts_on_line_x_eq_y[i];
		if (pt == O) {
			pts_on_line_x_eq_y_labels[i] = 0;
			pt_labels[pt] = 0 * plane_order + 0;
			}
		else if (pt == I) {
			pts_on_line_x_eq_y_labels[i] = 1;
			pt_labels[pt] = 1 * plane_order + 1;
			}
		else if (pt == quad_C) {
			pts_on_line_x_eq_y_labels[i] = plane_order;
			pt_labels[pt] = plane_order * plane_order + 1;
			}
		else {
			pts_on_line_x_eq_y_labels[i] = l;
			pt_labels[pt] = l * plane_order + l;
			l++;
			}
		}
	if (f_v) {
		cout << "points on line y=x:" << endl;
		for (i = 0; i <= plane_order; i++) {
			cout << pts_on_line_x_eq_y[i] << " : "
					<< pts_on_line_x_eq_y_labels[i] << endl;
			}
		}
	
	// label the affine points:
	// (x,y) = x * plane_order + y
	plane_get_lines_through_point(X, lines_through_X);
	plane_get_lines_through_point(Y, lines_through_Y);
	
	for (i = 0; i <= plane_order; i++) {
		if (lines_through_X[i] == line_infty)
			continue;
		ii = plane_line_intersection(lines_through_X[i], line_x_eq_y);
		y = pt_labels[ii] % plane_order;
		// cout << "ii=" << ii << " y=" << y << endl;
		
		for (j = 0; j <= plane_order; j++) {
			if (lines_through_Y[j] == line_infty)
				continue;
			jj = plane_line_intersection(lines_through_Y[j], line_x_eq_y);
			x = pt_labels[jj] % plane_order;
			// cout << "jj=" << jj << " x=" << x << endl;
			
			pt = plane_line_intersection(
					lines_through_X[i], lines_through_Y[j]);
			
			points[x * plane_order + y] = pt;
			pt_labels[pt] = x * plane_order + y;
			}
		}
	if (f_v) {
		cout << "the affine points (x,y):" << endl;
		for (x = 0; x < plane_order; x++) {
			for (y = 0; y < plane_order; y++) {
				cout << "(" << x << ", " << y << ")="
						<< points[x * plane_order + y] << endl;
				}
			}
		}
	
	// label the points at infinity:
	if (f_v) {
		cout << "the points at infinity:" << endl;
		}
	for (y = 0; y < plane_order; y++) {
		pt = points[1 * plane_order + y];
		line = plane_line_through_two_points(O, pt);
		pt2 = plane_line_intersection(line, line_infty);
		pt_labels[pt2] = plane_order * plane_order + y;
		points[plane_order * plane_order + y] = pt2;
		if (f_v) {
			cout << "y=" << y << " pt " << pt2 << endl;
			}
		}
	pt_labels[Y] = plane_order * plane_order + plane_order; // (infty)
	
	if (f_v) {
		cout << "all point labels:" << endl;
		for (i = 0; i < m; i++) {
			cout << i << " : " << pt_labels[i] << endl;
			}
		}
	
	// get the mutually orthogonal latin squares:
	// first we fill MOLS {1,2,...,plane_order-1}
	
	for (slope = 1; slope < plane_order; slope++) {
		
		pt2 = points[plane_order * plane_order + slope];
		// the point (slope) at infinity
		
		for (b = 0; b < plane_order; b++) {
		
			// we consider the line: y = slope * x + b
			// we let the (x,b) entry in the MOL
			// corresponding to slope be y
			
			pt = points[0 * plane_order + b]; // y intercept
			
			line = plane_line_through_two_points(pt, pt2);
			// this is the line y = slope * x + b
			
			
			plane_get_points_on_line(line, pts_on_line);
			// we loop over all affine points on this line 
			// i.e., all points except for (slope)
			
			for (i = 0; i <= plane_order; i++) {
				pt3 = pts_on_line[i];
				if (pt3 == pt2)
					continue; // skip (slope)
				
				pt_label = pt_labels[pt3];
				x = pt_label / plane_order;
				y = pt_label % plane_order;
				MOLSsxb(slope, x, b) = y;
				//MOLS[slope * plane_order *
				//plane_order + x * plane_order + b] = y;
				}
			}
		}
	
	// we can recover addition from the slope-1 lines as y = x + b
	// this information will go into MOLS 0
	slope = 1;
	for (x = 0; x < plane_order; x++) {
		for (b = 0; b < plane_order; b++) {
			y = MOLSsxb(slope, x, b);
			//y = MOLS[slope * plane_order * plane_order
			//+ x * plane_order + b];
			MOLSsxb(0, x, b) = y;
			//MOLS[0 * plane_order * plane_order
			// + x * plane_order + b] = y;
			}
		}
	// we can recover multiplication from the lines
	// with y-intercept 0 as y = slope * x + 0
	// this information will go into MOLS plane_order
	b = 0;
	for (slope = 0; slope < plane_order; slope++) {
		for (x = 0; x < plane_order; x++) {
			if (slope == 0 || x == 0)
				y = 0;
			else {
				y = MOLSsxb(slope, x, b);
				//y = MOLS[slope * plane_order *
				// plane_order + x * plane_order + b];
				}
			MOLSsxb(plane_order, slope, x) = y;
			//MOLS[plane_order * plane_order * plane_order
			// + slope * plane_order + x] = y;
			}
		}
	
	if (f_v) {
		print_MOLS(cout);
		cout << "finished" << endl;
		}
}

int &point_line::MOLSsxb(int s, int x, int b)
{
	return MOLS[s * plane_order * plane_order + x * plane_order + b];
}

int &point_line::MOLSaddition(int a, int b)
{
	return MOLSsxb(0, a, b);
}

int &point_line::MOLSmultiplication(int a, int b)
{
	return MOLSsxb(plane_order, a, b);
}

int point_line::ternary_field_is_linear(int *MOLS, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x, m, b, y1, mx, y2;
	int *addition = MOLS;
	int *multiplication = MOLS + plane_order * plane_order * plane_order;
	
	for (m = 1; m < plane_order; m++) {
		for (x = 0; x < plane_order; x++) {
			mx = multiplication[m * plane_order + x];
			for (b = 0; b < plane_order; b++) {
				y1 = MOLS[m * plane_order * plane_order + x * plane_order + b];
				y2 = addition[mx * plane_order + b];
				if (y1 != y2) {
					if (f_v) {
						cout << "not linear:" << endl;
						cout << "m=" << m << endl;
						cout << "x=" << x << endl;
						cout << "b=" << b << endl;
						cout << "y1=" << y1 << endl;
						cout << "mx=" << mx << endl;
						cout << "y2=" << y2 << endl;
						}
					return FALSE;
					}
				}
			}
		}
	return TRUE;
}

void point_line::print_MOLS(ostream &ost)
{
	int *M, slope, i, j;
		
	ost << "all mutually orthogonal latin squares:" << endl;
	ost << "addition:" << endl;
	get_MOLm(MOLS, plane_order, 0, M);
	for (i = 0; i < plane_order; i++) {
		for (j = 0; j < plane_order; j++) {
			ost << setw(2) << M[i * plane_order + j] << " ";
			}
		ost << endl;
		}
	FREE_int(M);
	ost << "multiplication:" << endl;
	get_MOLm(MOLS, plane_order, plane_order, M);
	for (i = 0; i < plane_order; i++) {
		for (j = 0; j < plane_order; j++) {
			ost << setw(2) << M[i * plane_order + j] << " ";
			}
		ost << endl;
		}
	FREE_int(M);
	for (slope = 1; slope < plane_order; slope++) {
		ost << "for slope=" << slope << endl;
		get_MOLm(MOLS, plane_order, slope, M);
		for (i = 0; i < plane_order; i++) {
			for (j = 0; j < plane_order; j++) {
				ost << setw(2) << M[i * plane_order + j] << " ";
				}
			ost << endl;
			}
		FREE_int(M);
		}
}

int point_line::is_projective_plane(partitionstack &P, int &order, int verbose_level)
// if it is a projective plane, the order is returned.
// otherwise, 0 is returned.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v, n, r, k, lambda, mu;
	
	if (f_v) {
		cout << "point_line::is_projective_plane checking for projective plane:" << endl;
		}
	if (P.ht != 2) {
		if (f_vv) {
			cout << "not a projective plane: "
					"partition has more than two parts" << endl;
			}
		return FALSE;
		}
	if (P.cellSize[0] != P.cellSize[1]) {
		if (f_vv) {
			cout << "not a projective plane: "
					"partition classes have different sizes" << endl;
			}
		return FALSE;
		}
	v = P.cellSize[0];
	r = count_RC(P, 0, 1);
	if (r == -1) {
		if (f_vv) {
			cout << "not a projective plane: "
					"r not constant" << endl;
			}
		return FALSE;
		}
	k = count_CR(P, 1, 0);
	if (k == -1) {
		if (f_vv) {
			cout << "not a projective plane: "
					"k not constant" << endl;
			}
		return FALSE;
		}
	if (f_vv) {
		cout << "r = " << r << endl;
		cout << "k = " << k << endl;
		}
	if (r != k) {
		if (f_vv) {
			cout << "not a projective plane: r != k" << endl;
			}
		return FALSE;
		}
	if (r < 3) {
		if (f_vv) {
			cout << "not a projective plane: r < 3" << endl;
			}
		return FALSE;
		}
	n = r - 1;
	if (v != n * n + n + 1) {
		if (f_vv) {
			cout << "not a projective plane: "
					"v != n^2 + n + 1" << endl;
			}
		return FALSE;
		}
	lambda = count_pairs_RRC(P, 0, 0, 1);
	if (lambda != 1) {
		if (f_vv) {
			cout << "not a projective plane: "
					"rows are not joined correctly" << endl;
			}
		return FALSE;
		}
	mu = count_pairs_CCR(P, 1, 1, 0);
	if (mu != 1) {
		if (f_vv) {
			cout << "not a projective plane: "
					"cols are not joined correctly" << endl;
			}
		return FALSE;
		}
	if (f_vv) {
		cout << "lambda = " << lambda << endl;
		cout << "mu = " << mu << endl;
		}
	if (f_v) {
		cout << "detected a projective plane of order " << n << endl;
		}
	order = n;
	return TRUE;
}

int point_line::count_RC(partitionstack &P, int row_cell, int col_cell)
{
	int l1, i, nb = -1, nb1;
	
	l1 = P.cellSize[row_cell];
	for (i = 0; i < l1; i++) {
		nb1 = count_RC_representative(P, 
			row_cell, i, col_cell);
		if (nb1 == -1)
			return -1;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}

int point_line::count_CR(partitionstack &P, int col_cell, int row_cell)
{
	int l1, i, nb = -1, nb1;
	
	l1 = P.cellSize[col_cell];
	for (i = 0; i < l1; i++) {
		nb1 = count_CR_representative(P, col_cell, i, row_cell);
		if (nb1 == -1)
			return -1;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}

int point_line::count_RC_representative(partitionstack &P, 
	int row_cell, int row_cell_pt, int col_cell)
{
	int f1, f2, /*l1,*/ l2, e1, e2, j, s = 0;
	int first_column_element = P.startCell[1];
	
	f1 = P.startCell[row_cell];
	f2 = P.startCell[col_cell];
	//l1 = P.cellSize[row_cell];
	l2 = P.cellSize[col_cell];
	e1 = P.pointList[f1 + row_cell_pt];
	for (j = 0; j < l2; j++) {
		e2 = P.pointList[f2 + j] - first_column_element;
		// cout << "e1=" << e1 << " e2=" << e2 << endl;
		if (a[e1 * n + e2])
			s++;
		}
	return s;
}

int point_line::count_CR_representative(partitionstack &P, 
	int col_cell, int col_cell_pt, int row_cell)
{
	int f1, f2, l1, /*l2,*/ e1, e2, i, s = 0;
	int first_column_element = P.startCell[1];
	
	f1 = P.startCell[row_cell];
	f2 = P.startCell[col_cell];
	l1 = P.cellSize[row_cell];
	//l2 = P.cellSize[col_cell];
	e2 = P.pointList[f2 + col_cell_pt] - first_column_element;
	for (i = 0; i < l1; i++) {
		e1 = P.pointList[f1 + i];
		// cout << "e1=" << e1 << " e2=" << e2 << endl;
		if (a[e1 * n + e2])
			s++;
		}
	return s;
}

int point_line::count_pairs_RRC(partitionstack &P,
		int row_cell1, int row_cell2, int col_cell)
{
	int l1, i, nb = -1, nb1;
	
	l1 = P.cellSize[row_cell1];
	for (i = 0; i < l1; i++) {
		nb1 = count_pairs_RRC_representative(P,
				row_cell1, i, row_cell2, col_cell);
		if (nb1 == -1)
			return -1;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}

int point_line::count_pairs_CCR(partitionstack &P,
		int col_cell1, int col_cell2, int row_cell)
{
	int l1, i, nb = -1, nb1;
	
	l1 = P.cellSize[col_cell1];
	for (i = 0; i < l1; i++) {
		nb1 = count_pairs_CCR_representative(P,
				col_cell1, i, col_cell2, row_cell);
		if (nb1 == -1)
			return -1;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}

int point_line::count_pairs_RRC_representative(partitionstack &P,
		int row_cell1, int row_cell_pt, int row_cell2, int col_cell)
// returns the number of joinings from a point of
// row_cell1 to elements of row_cell2 within col_cell
// if that number exists, -1 otherwise
{
	int f1, f2, f3, /*l1,*/ l2, l3, e1, e2, e3, u, j, nb = -1, nb1;
	int first_column_element = P.startCell[1];
	
	f1 = P.startCell[row_cell1];
	f2 = P.startCell[row_cell2];
	f3 = P.startCell[col_cell];
	//l1 = P.cellSize[row_cell1];
	l2 = P.cellSize[row_cell2];
	l3 = P.cellSize[col_cell];
	e1 = P.pointList[f1 + row_cell_pt];
	for (u = 0; u < l2; u++) {
		e2 = P.pointList[f2 + u];
		if (e1 == e2)
			continue;
		nb1 = 0;
		for (j = 0; j < l3; j++) {
			e3 = P.pointList[f3 + j] - first_column_element;
			if (a[e1 * n + e3] && a[e2 * n + e3]) {
				nb1++;
				}
			}
		// cout << "e1=" << e1 << " e2=" << e2
		//<< " e3=" << e3 << " nb=" << nb1 << endl;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}


int point_line::count_pairs_CCR_representative(partitionstack &P,
		int col_cell1, int col_cell_pt, int col_cell2, int row_cell)
// returns the number of joinings from a point of
// col_cell1 to elements of col_cell2 within row_cell
// if that number exists, -1 otherwise
{
	int f1, f2, f3, /*l1,*/ l2, l3, e1, e2, e3, u, i, nb = -1, nb1;
	int first_column_element = P.startCell[1];
	
	f1 = P.startCell[col_cell1];
	f2 = P.startCell[col_cell2];
	f3 = P.startCell[row_cell];
	//l1 = P.cellSize[col_cell1];
	l2 = P.cellSize[col_cell2];
	l3 = P.cellSize[row_cell];
	e1 = P.pointList[f1 + col_cell_pt] - first_column_element;
	for (u = 0; u < l2; u++) {
		e2 = P.pointList[f2 + u] - first_column_element;
		if (e1 == e2)
			continue;
		nb1 = 0;
		for (i = 0; i < l3; i++) {
			e3 = P.pointList[f3 + i];
			if (a[e3 * n + e1] && a[e3 * n + e2]) {
				nb1++;
				}
			}
		// cout << "e1=" << e1 << " e2=" << e2 << " e3=" << e3
		//<< " nb=" << nb1 << endl;
		if (nb == -1) {
			nb = nb1;
			}
		else {
			if (nb1 != nb)
				return -1;
			}
		}
	return nb;
}


void point_line::get_MOLm(int *MOLS, int order, int m, int *&M)
{
	int x, b, y, *mol = MOLS + m * order * order;
	
	M = NEW_int(order * order);
	for (x = 0; x < order; x++) {
		for (b = 0; b < order; b++) {
			y = mol[x * order + b];
			M[x * order + b] = y;
			}
		}
}

}
}

