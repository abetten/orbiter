// orthogonal.cpp
// 
// Anton Betten
// 3/8/7: lines in hyperbolic spaces
//
// continued May 2007 with parabolic type
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace foundations {


void orthogonal::unrank_point(
		int *v, int stride, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::unrank_point rk=" << rk
				<< " epsilon=" << epsilon << " n=" << n << endl;
		}
	F->Q_epsilon_unrank(v, stride, epsilon, n - 1,
			form_c1, form_c2, form_c3, rk);
}

int orthogonal::rank_point(int *v, int stride, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal::rank_point" << endl;
		}
	// copy the vector since Q_epsilon_rank has side effects 
	// (namely, Q_epsilon_rank damages its input vector)
	
	for (i = 0; i < n; i++)
		rk_pt_v[i] = v[i * stride];
	
	return F->Q_epsilon_rank(rk_pt_v, 1, epsilon, n - 1,
			form_c1, form_c2, form_c3);
}


void orthogonal::unrank_line(int &p1, int &p2,
		int rk, int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_unrank_line(p1, p2, rk, verbose_level);
		return;
		}
	else if (epsilon == 0) {
		parabolic_unrank_line(p1, p2, rk, verbose_level);
		return;
		}
	else {
		cout << "orthogonal::unrank_line epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::rank_line(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "orthogonal::rank_line" << endl;
	}
	if (epsilon == 1) {
		ret = hyperbolic_rank_line(p1, p2, verbose_level);
		}
	else if (epsilon == 0) {
		ret = parabolic_rank_line(p1, p2, verbose_level);
		}
	else {
		cout << "orthogonal::rank_line epsilon = " << epsilon << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal::rank_line done" << endl;
	}
	return ret;
}

int orthogonal::line_type_given_point_types(
		int pt1, int pt2, int pt1_type, int pt2_type)
{
	if (epsilon == 1) {
		return hyperbolic_line_type_given_point_types(
				pt1, pt2, pt1_type, pt2_type);
		}
	else if (epsilon == 0) {
		return parabolic_line_type_given_point_types(
				pt1, pt2, pt1_type, pt2_type, FALSE);
		}
	else {
		cout << "type_and_index_to_point_rk "
				"epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	if (epsilon == 1) {
		return hyperbolic_type_and_index_to_point_rk(
				type, index);
		}
	else if (epsilon == 0) {
		return parabolic_type_and_index_to_point_rk(
				type, index, verbose_level);
		}
	else {
		cout << "type_and_index_to_point_rk "
				"epsilon = " << epsilon << endl;
		exit(1);
		}
}

void orthogonal::point_rk_to_type_and_index(
		int rk, int &type, int &index,
		int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_point_rk_to_type_and_index(
				rk, type, index);
		}
	else if (epsilon == 0) {
		parabolic_point_rk_to_type_and_index(
				rk, type, index, verbose_level);
		}
	else {
		cout << "type_and_index_to_point_rk epsilon = " << epsilon << endl;
		exit(1);
		}
}

void orthogonal::canonical_points_of_line(
	int line_type, int pt1, int pt2,
	int &cpt1, int &cpt2, int verbose_level)
{
	if (epsilon == 1) {
		hyperbolic_canonical_points_of_line(line_type,
				pt1, pt2, cpt1, cpt2, verbose_level);
		}
	else if (epsilon == 0) {
		parabolic_canonical_points_of_line(line_type,
				pt1, pt2, cpt1, cpt2, verbose_level);
		}
	else {
		cout << "canonical_points_of_line epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::evaluate_quadratic_form(int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_quadratic_form(v, stride, m);
		}
	else if (epsilon == 0) {
		int a, b, c;
		
		a = evaluate_hyperbolic_quadratic_form(v + stride, stride, m);
		//if (f_even)
			//return a;
		b = F->mult(v[0], v[0]);
		c = F->add(a, b);
		return c;
		}
	else if (epsilon == -1) {
		int a, x1, x2, b, c, d;
		
		a = evaluate_hyperbolic_quadratic_form(v, stride, m);
		x1 = v[2 * m * stride];
		x2 = v[(2 * m + 1) * stride];
		b = F->mult(x1, x1);
		b = F->mult(form_c1, b);
		c = F->mult(x1, x2);
		c = F->mult(form_c2, c);
		d = F->mult(x2, x2);
		d = F->mult(form_c3, d);
		a = F->add(a, b);
		c = F->add(a, c);
		c = F->add(d, c);
		return c;
		}
	else {
		cout << "evaluate_quadratic_form epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::evaluate_bilinear_form(int *u, int *v, int stride)
{
	if (epsilon == 1) {
		return evaluate_hyperbolic_bilinear_form(u, v, stride, m);
		}
	else if (epsilon == 0) {
		return evaluate_parabolic_bilinear_form(u, v, stride, m);
		}
	else if (epsilon == -1) {
		return F->evaluate_bilinear_form(
				u, v, n, Gram_matrix);
		}
	else {
		cout << "evaluate_bilinear_form epsilon = " << epsilon << endl;
		exit(1);
		}
}

int orthogonal::evaluate_bilinear_form_by_rank(int i, int j)
{
	unrank_point(v1, 1, i, 0);
	unrank_point(v2, 1, j, 0);
	return evaluate_bilinear_form(v1, v2, 1);
}

int orthogonal::find_root(int rk2, int verbose_level)
{
	if (epsilon == 1) {
		return find_root_hyperbolic(rk2, m, verbose_level);
		}
	else if (epsilon == 0) {
		return find_root_parabolic(rk2, verbose_level);
		}
	else {
		cout << "find_root epsilon = " << epsilon << endl;
		exit(1);
		}
}

void orthogonal::points_on_line_by_line_rank(
		int line_rk, int *line, int verbose_level)
{
	int p1, p2;
	
	unrank_line(p1, p2, line_rk, verbose_level);
	points_on_line(p1, p2, line, verbose_level);
}

void orthogonal::points_on_line(int pi, int pj,
		int *line, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v1, *v2, *v3;
	int coeff[2], t, i, a, b;
	
	v1 = determine_line_v1;
	v2 = determine_line_v2;
	v3 = determine_line_v3;
	unrank_point(v1, 1, pi, verbose_level - 1);
	unrank_point(v2, 1, pj, verbose_level - 1);
	if (f_v) {
		cout << "points_on_line" << endl;
		cout << "v1=";
		int_vec_print(cout, v1, n);
		cout << endl;
		cout << "v2=";
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	for (t = 0; t <= q; t++) {
		F->PG_element_unrank_modified(coeff, 1, 2, t);
		for (i = 0; i < n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
			}
		if (f_v) {
			cout << "t=" << t << " ";
			int_vec_print(cout, coeff, 2);
			cout << " v3=";
			int_vec_print(cout, v3, n);
			cout << endl;
			}
		normalize_point(v3, 1);
		if (f_v) {
			cout << "normalized:";
			int_vec_print(cout, v3, n);
			cout << endl;
			}
		line[t] = rank_point(v3, 1, verbose_level - 1);
		if (f_v) {
			cout << "rank=" << line[t] << endl;
			}
		}
}

void orthogonal::points_on_line_by_coordinates(
		int pi, int pj, int *pt_coords, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v1, *v2, *v3;
	int coeff[2], t, i, a, b;
	
	v1 = determine_line_v1;
	v2 = determine_line_v2;
	v3 = determine_line_v3;
	unrank_point(v1, 1, pi, verbose_level - 1);
	unrank_point(v2, 1, pj, verbose_level - 1);
	if (f_v) {
		cout << "points_on_line_by_coordinates" << endl;
		cout << "v1=";
		int_vec_print(cout, v1, n);
		cout << endl;
		cout << "v2=";
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	for (t = 0; t <= q; t++) {
		F->PG_element_unrank_modified(coeff, 1, 2, t);
		for (i = 0; i < n; i++) {
			a = F->mult(coeff[0], v1[i]);
			b = F->mult(coeff[1], v2[i]);
			v3[i] = F->add(a, b);
			}
		if (f_v) {
			cout << "v3=";
			int_vec_print(cout, v3, n);
			cout << endl;
			}
		normalize_point(v3, 1);
		for (i = 0; i < n; i++) {
			pt_coords[t * n + i] = v3[i];
			}
		}
}

void orthogonal::lines_on_point(int pt,
		int *line_pencil_point_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk, rk1, root1, root2;
	
	if (f_v) {
		cout << "lines_on_point" << endl;
		cout << "pt=" << pt << endl;
		}
	t = subspace_point_type;
	for (i = 0; i < alpha; i++) {
		rk = type_and_index_to_point_rk(t, i, 0);
		unrank_point(lines_on_point_coords1 + i * n,
				1, rk, verbose_level - 1);
		}
	if (pt != pt_P) {
		root1 = find_root(pt_P, verbose_level);
		rk1 = type_and_index_to_point_rk(t, 0, verbose_level);
		Siegel_Transformation(T1, pt_P, rk1, root1, verbose_level);
		if (pt != 0) {
			root2 = find_root(pt, verbose_level);
			Siegel_Transformation(T2, rk1, pt, root2, verbose_level);
			F->mult_matrix_matrix(T1, T2, T3, n, n, n,
					0 /* verbose_level */);
			}
		else {
			F->copy_matrix(T1, T3, n, n);
			}
		F->mult_matrix_matrix(lines_on_point_coords1, T3,
				lines_on_point_coords2, alpha, n, n,
				0 /* verbose_level */);
		}
	else {
		for (i = 0; i < alpha; i++) {
			for (j = 0; j < n; j++) {
				lines_on_point_coords2[i * n + j] =
						lines_on_point_coords1[i * n + j];
				}
			}
		}
	for (i = 0; i < alpha; i++) {
		line_pencil_point_ranks[i] = rank_point(
				lines_on_point_coords2 + i * n, 1, verbose_level - 1);
		}
	if (f_v) {
		cout << "line pencil (point ranks) on point " << pt << " : ";
		int_vec_print(cout, line_pencil_point_ranks, alpha);
		cout << endl;
		}
}

void orthogonal::lines_on_point_by_line_rank(int pt,
		int *line_pencil_line_ranks, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int t, i, j, rk, rk1, root1, root2, pt2;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank" << endl;
		cout << "pt=" << pt << endl;
		cout << "pt_P=" << pt_P << endl;
		}
	t = subspace_point_type;
	for (i = 0; i < alpha; i++) {
		rk = type_and_index_to_point_rk(t, i, 0);
		unrank_point(lines_on_point_coords1 + i * n,
				1, rk, verbose_level - 1);
		}
	if (pt != pt_P) {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank applying transformation" << endl;
		}
		rk1 = type_and_index_to_point_rk(t, 0, verbose_level);
		if (pt == rk1) {
			root1 = find_root(pt_P, verbose_level);
			Siegel_Transformation(T3, pt_P, rk1, root1, verbose_level);
			}
		else {
			root1 = find_root(pt_P, verbose_level);
			root2 = find_root(pt, verbose_level);
			Siegel_Transformation(T1, pt_P, rk1, root1, verbose_level);
			Siegel_Transformation(T2, rk1, pt, root2, verbose_level);
			F->mult_matrix_matrix(T1, T2, T3, n, n, n,
					0 /* verbose_level */);
			}
		F->mult_matrix_matrix(lines_on_point_coords1,
				T3, lines_on_point_coords2, alpha, n, n,
				0 /* verbose_level */);
		}
	else {
		if (f_v) {
			cout << "orthogonal::lines_on_point_by_line_rank pt == pt_P, "
					"no need to apply transformation" << endl;
		}
		for (i = 0; i < alpha; i++) {
			for (j = 0; j < n; j++) {
				lines_on_point_coords2[i * n + j] =
						lines_on_point_coords1[i * n + j];
				}
			}
		}
	for (i = 0; i < alpha; i++) {
		pt2 = rank_point(lines_on_point_coords2 + i * n, 1,
				verbose_level - 1);
		line_pencil_line_ranks[i] =
				rank_line(pt, pt2, verbose_level);
		}
	Sorting.int_vec_quicksort_increasingly(line_pencil_line_ranks, alpha);
	if (f_v) {
		cout << "line pencil on point " << pt << " by line rank : ";
		int_vec_print(cout, line_pencil_line_ranks, alpha);
		cout << endl;
		}
	if (f_v) {
		cout << "orthogonal::lines_on_point_by_line_rank done" << endl;
		}
}

void orthogonal::list_points_by_type(int verbose_level)
{
	int t;
	
	for (t = 1; t <= nb_point_classes; t++) {
		list_points_of_given_type(t, verbose_level);
		}
}

void orthogonal::list_points_of_given_type(int t, int verbose_level)
{
	int i, j, rk, u;
	
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

void orthogonal::test_Siegel(int index, int verbose_level)
{
	int rk1, rk2, rk1_subspace, rk2_subspace, root, j, rk3, cnt, u, t2;

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	cout << 0 << " : " << rk1 << " : ";
	unrank_point(v1, 1, rk1, verbose_level - 1);
	int_vec_print(cout, v1, n);
	cout << endl;

	rk2 = type_and_index_to_point_rk(5, index, verbose_level);
	cout << index << " : " << rk2 << " : ";
	unrank_point(v2, 1, rk2, verbose_level - 1);
	int_vec_print(cout, v2, n);
	cout << endl;
	
	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	cout << "rk1_subspace=" << rk1_subspace << endl;
	cout << "rk2_subspace=" << rk2_subspace << endl;
	
	root = subspace->find_root_parabolic(
			rk2_subspace, verbose_level);
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level);

	cout << "Siegel map takes 1st point to" << endl;
	F->mult_matrix_matrix(v1, T1, v3, 1, n - 2, n - 2,
			0 /* verbose_level */);
	int_vec_print(cout, v3, n - 2);
	cout << endl;

	cnt = 0;
	
	t2 = 1;
	for (j = 0; j < subspace->P[t2 - 1]; j++) {
		if (f_even) {
			cout << "f_even" << endl;
			exit(1);
			}
		parabolic_neighbor51_odd_unrank(j, v3, FALSE);
		//rk3 = type_and_index_to_point_rk(t2, j);
		//unrank_point(v3, 1, rk3);
		rk3 = rank_point(v3, 1, verbose_level - 1);
			
		u = evaluate_bilinear_form(v1, v3, 1);
		if (u) {
			cout << "error, u not zero" << endl;
			}
		
		//if (test_if_minimal_on_line(v3, v1, v_tmp)) {


		cout << "Siegel map takes 2nd point ";
		cout << cnt << " : " << j << " : " << rk3 << " : ";
		int_vec_print(cout, v3, n);
		cout << " to ";
		F->mult_matrix_matrix(v3, T1, v_tmp, 1, n - 2, n - 2,
				0 /* verbose_level */);
				
			
		v_tmp[n - 2] = v3[n - 2];
		v_tmp[n - 1] = v3[n - 1];
		int_vec_print(cout, v_tmp, n);


		//cout << "find_minimal_point_on_line " << endl;
		//find_minimal_point_on_line(v_tmp, v2, v4);
				
		//cout << " minrep: ";
		//int_vec_print(cout, v4, n);
				
		//normalize_point(v4, 1);
		//cout << " normalized: ";
		//int_vec_print(cout, v4, n);
				
		cout << endl;

		cnt++;
		//}
		}
	cout << endl;
}

void orthogonal::make_initial_partition(
		partitionstack &S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, l, a, f;
	
	if (f_v) {
		cout << "orthogonal::make_initial_partition" << endl;
		}
	S.allocate(nb_points + nb_lines, f_v);
	
	// split off the column class:
	S.subset_continguous(nb_points, nb_lines);
	S.split_cell(FALSE);
	
	for (i = nb_point_classes; i >= 2; i--) {
		l = P[i - 1];
		if (l == 0)
			continue;
		if (f_v) {
			cout << "splitting off point class " << i
					<< " of size " << l << endl;
			}
		for (j = 0; j < l; j++) {
			a = type_and_index_to_point_rk(i, j, verbose_level - 2);
			//if (f_v) {cout << "j=" << j << " a=" << a << endl;}
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
	for (i = nb_line_classes; i >= 2; i--) {
		f = nb_points;
		for (j = 1; j < i; j++)
			f += L[j - 1];
		l = L[i - 1];
		if (l == 0)
			continue;
		if (f_v) {
			cout << "splitting off line class " << i
					<< " of size " << l << endl;
			}
		for (j = 0; j < l; j++) {
			S.subset[j] = f + j;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		f += l;
		}
	if (f_v) {
		cout << "the initial partition of points and lines is:" << endl;
		cout << S << endl;
		}
}

void orthogonal::point_to_line_map(int size,
		int *point_ranks, int *&line_vector, int verbose_level)
{
	int i, j, h;
	int *neighbors;
	
	neighbors = NEW_int(alpha);
	
	line_vector = NEW_int(nb_lines);
	for (j = 0; j < nb_lines; j++)
		line_vector[j] = 0;
	
	for (i = 0; i < size; i++) {
		lines_on_point_by_line_rank(
				point_ranks[i], neighbors, verbose_level - 2);
		for (h = 0; h < alpha; h++) {
			j = neighbors[h];
			line_vector[j]++;
			}
		}
	FREE_int(neighbors);
}

void orthogonal::move_points_by_ranks_in_place(
	int pt_from, int pt_to,
	int nb, int *ranks, int verbose_level)
{
	int *input_coords, *output_coords, i;
	
	input_coords = NEW_int(nb * n);
	output_coords = NEW_int(nb * n);
	for (i = 0; i < nb; i++) {
		unrank_point(
				input_coords + i * n, 1, ranks[i],
				verbose_level - 1);
		}
	
	move_points(pt_from, pt_to, 
		nb, input_coords, output_coords, verbose_level);
	
	for (i = 0; i < nb; i++) {
		ranks[i] = rank_point(
				output_coords + i * n, 1, verbose_level - 1);
		}
	
	FREE_int(input_coords);
	FREE_int(output_coords);
}

void orthogonal::move_points_by_ranks(int pt_from, int pt_to, 
	int nb, int *input_ranks, int *output_ranks,
	int verbose_level)
{
	int *input_coords, *output_coords, i;
	
	input_coords = NEW_int(nb * n);
	output_coords = NEW_int(nb * n);
	for (i = 0; i < nb; i++) {
		unrank_point(input_coords + i * n, 1,
				input_ranks[i], verbose_level - 1);
		}
	
	move_points(pt_from, pt_to, 
		nb, input_coords, output_coords, verbose_level);
	
	for (i = 0; i < nb; i++) {
		output_ranks[i] = rank_point(
				output_coords + i * n, 1, verbose_level - 1);
		}
	
	FREE_int(input_coords);
	FREE_int(output_coords);
}

void orthogonal::move_points(int pt_from, int pt_to, 
	int nb, int *input_coords, int *output_coords,
	int verbose_level)
{
	int root, i;
	int *tmp_coords = NULL;
	int *input_coords2;
	int *T;
	
	if (pt_from == pt_to) {
		for (i = 0; i < nb * n; i++) {
			output_coords[i] = input_coords[i];
			}
		return;
		}
	
	T = NEW_int(n * n);
	if (pt_from != 0) {
		
		tmp_coords = NEW_int(n * nb);
		root = find_root(pt_from, verbose_level - 2);
		Siegel_Transformation(T,
				pt_from /* from */,
				0 /* to */,
				root /* root */,
				verbose_level - 2);
		F->mult_matrix_matrix(input_coords,
				T, tmp_coords, nb, n, n,
				0 /* verbose_level */);
		input_coords2 = tmp_coords;
		}
	else {
		input_coords2 = input_coords;
		}
		
	root = find_root(pt_to, verbose_level - 2);
	Siegel_Transformation(T,
			0 /* from */,
			pt_to /* to */,
			root /* root */,
			verbose_level - 2);
	F->mult_matrix_matrix(input_coords2, T, output_coords, nb, 5, 5,
			0 /* verbose_level */);

	if (tmp_coords) FREE_int(tmp_coords);
	
	FREE_int(T);
}

int orthogonal::BLT_test_full(int size, int *set, int verbose_level)
{
	if (!collinearity_test(size, set, 0/*verbose_level - 2*/)) {
		return FALSE;
		}
	if (!BLT_test(size, set, verbose_level)) {
		return FALSE;
		}
	return TRUE;
}

int orthogonal::BLT_test(int size, int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, x, y, z, a;
	int f_OK = TRUE;
	int fxy, fxz, fyz, l1, l2, l3;
	int two;
	int m1[5], m3[5];
	
	if (size <= 2)
		return TRUE;
	if (f_v) {
		cout << "BLT_test for" << endl;
		int_vec_print(cout, set, size);
		if (f_vv) {
			for (i = 0; i < size; i++) {
				unrank_point(v1, 1, set[i], verbose_level - 1);
				cout << i << " : " << set[i] << " : ";
				int_vec_print(cout, v1, n);
				cout << endl;
				}
			}
		}
	x = set[0];
	z = set[size - 1];
	two = F->add(1, 1);
	unrank_point(v1, 1, x, verbose_level - 1);
	unrank_point(v3, 1, z, verbose_level - 1);
	
	m1[0] = F->mult(two, v1[0]);
	m1[1] = v1[2];
	m1[2] = v1[1];
	m1[3] = v1[4];
	m1[4] = v1[3];

	//fxz = evaluate_bilinear_form(v1, v3, 1);
	// too slow !!!
	fxz = F->add5(
			F->mult(m1[0], v3[0]), 
			F->mult(m1[1], v3[1]), 
			F->mult(m1[2], v3[2]), 
			F->mult(m1[3], v3[3]), 
			F->mult(m1[4], v3[4]) 
		);

	m3[0] = F->mult(two, v3[0]);
	m3[1] = v3[2];
	m3[2] = v3[1];
	m3[3] = v3[4];
	m3[4] = v3[3];


	if (f_vv) {
		l1 = F->log_alpha(fxz);
		cout << "fxz=" << fxz << " (log " << l1 << ") ";
		if (EVEN(l1))
			cout << "+";
		else
			cout << "-";
		cout << endl;
		}
	
	for (i = 1; i < size - 1; i++) {
	
		y = set[i];

		unrank_point(v2, 1, y, verbose_level - 1);
		
		//fxy = evaluate_bilinear_form(v1, v2, 1);
		fxy = F->add5(
				F->mult(m1[0], v2[0]), 
				F->mult(m1[1], v2[1]), 
				F->mult(m1[2], v2[2]), 
				F->mult(m1[3], v2[3]), 
				F->mult(m1[4], v2[4]) 
			);
		
		//fyz = evaluate_bilinear_form(v2, v3, 1);
		fyz = F->add5(
				F->mult(m3[0], v2[0]), 
				F->mult(m3[1], v2[1]), 
				F->mult(m3[2], v2[2]), 
				F->mult(m3[3], v2[3]), 
				F->mult(m3[4], v2[4]) 
			);

		a = F->product3(fxy, fxz, fyz);
		if (f_vv) {
			l2 = F->log_alpha(fxy);
			l3 = F->log_alpha(fyz);
			cout << "i=" << i << " fxy=" << fxy << " (log=" << l2 
				<< ") fyz=" << fyz << " (log=" << l3
				<< ") a=" << a << endl;
			}
		
		
		if (f_is_minus_square[a]) {
			f_OK = FALSE;
			if (f_v) {
				l1 = F->log_alpha(fxz);
				l2 = F->log_alpha(fxy);
				l3 = F->log_alpha(fyz);
				cout << "not OK; i=" << i << endl;
				cout << "{x,y,z}={" << x << "," << y
						<< "," << z << "}" << endl;
				int_vec_print(cout, v1, n);
				cout << endl;
				int_vec_print(cout, v2, n);
				cout << endl;
				int_vec_print(cout, v3, n);
				cout << endl;
				cout << "fxz=" << fxz << " ";
				if (EVEN(l1))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l1 << ")" << endl;
				cout << "fxy=" << fxy << " ";
				if (EVEN(l2))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l2 << ")" << endl;
				cout << "fyz=" << fyz << " ";
				if (EVEN(l3))
					cout << "+";
				else
					cout << "-";
				cout << " (log=" << l3 << ")" << endl;
				cout << "a=" << a << "(log=" << F->log_alpha(a)
						<< ") is the negative of a square" << endl;
				print_minus_square_tables();
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "BLT_test fails" << endl;
			}
		else {
			cout << endl;
			}
		}
	return f_OK;
}

int orthogonal::collinearity_test(int size, int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, x, y;
	int f_OK = TRUE;
	int fxy;
	
	if (f_v) {
		cout << "collinearity test for" << endl;
		for (i = 0; i < size; i++) {
			unrank_point(v1, 1, set[i], verbose_level - 1);
			//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k, 
				//form_c1, form_c2, form_c3, line[i]);
			int_vec_print(cout, v1, 5);
			cout << endl;
			}
		}
	y = set[size - 1];
	//Q_epsilon_unrank(*M->GFq, v, 1, epsilon, k,
	//form_c1, form_c2, form_c3, y);
	unrank_point(v1, 1, y, verbose_level - 1);
	
	for (i = 0; i < size - 1; i++) {
		x = set[i];
		unrank_point(v2, 1, x, verbose_level - 1);
		//Q_epsilon_unrank(*M->GFq, u, 1, epsilon, k,
		//form_c1, form_c2, form_c3, x);
		
		//fxy = evaluate_bilinear_form(*M->GFq, u, v, d, Gram);
		fxy = evaluate_bilinear_form(v1, v2, 1);
		
		if (fxy == 0) {
			f_OK = FALSE;
			if (f_v) {
				cout << "not OK; ";
				cout << "{x,y}={" << x << "," << y
						<< "} are collinear" << endl;
				int_vec_print(cout, v1, 5);
				cout << endl;
				int_vec_print(cout, v2, 5);
				cout << endl;
				cout << "fxy=" << fxy << endl;
				}
			break;
			}
		}
	
	if (f_v) {
		if (!f_OK) {
			cout << "collinearity test fails" << endl;
			}
		}
	return f_OK;
}

// #############################################################################
// orthogonal_init.cpp
// #############################################################################

orthogonal::orthogonal()
{
	epsilon = n = m = q = 0;
	f_even = FALSE;
	form_c1 = form_c2 = form_c3 = 0;
	Gram_matrix = NULL;
	T1 = NULL;
	T2 = NULL;
	T3 = NULL;
	pt_P = pt_Q = 0;
	nb_points = 0;
	nb_lines = 0;

	T1_m = 0;
	T1_mm1 = 0;
	T1_mm2 = 0;
	T2_m = 0;
	T2_mm1 = 0;
	T2_mm2 = 0;
	N1_m = 0;
	N1_mm1 = 0;
	N1_mm2 = 0;
	S_m = 0;
	S_mm1 = 0;
	S_mm2 = 0;
	Sbar_m = 0;
	Sbar_mm1 = 0;
	Sbar_mm2 = 0;

	alpha = beta = gamma = 0;
	subspace_point_type = 0;
	subspace_line_type = 0;

	nb_point_classes = nb_line_classes = 0;
	A = B = P = L = NULL;

	p1 = p2 = p3 = p4 = p5 = p6 = 0;
	l1 = l2 = l3 = l4 = l5 = l6 = l7 = 0;
	a11 = a12 = a22 = a23 = a26 = a32 = a34 = a37 = 0;
	a41 = a43 = a44 = a45 = a46 = a47 = a56 = a67 = 0;
	b11 = b12 = b22 = b23 = b26 = b32 = b34 = b37 = 0;
	b41 = b43 = b44 = b45 = b46 = b47 = b56 = b67 = 0;

	p7 = l8 = 0;
	a21 = a36 = a57 = a22a = a33 = a22b = 0;
	a32b = a42b = a51 = a53 = a54 = a55 = a66 = a77 = 0;
	b21 = b36 = b57 = b22a = b33 = b22b = 0;
	b32b = b42b = b51 = b53 = b54 = b55 = b66 = b77 = 0;
	a12b = a52a = 0;
	b12b = b52a = 0;
	delta = omega = lambda = mu = nu = zeta = 0;

	minus_squares = NULL;
	minus_squares_without = NULL;
	minus_nonsquares = NULL;
	f_is_minus_square = FALSE;
	index_minus_square = NULL;
	index_minus_square_without = NULL;
	index_minus_nonsquare = NULL;

	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	v4 = NULL;
	v5 = NULL;
	v_tmp = NULL;
	v_tmp2 = NULL;
	v_neighbor5 = NULL;

	find_root_x = NULL;
	find_root_y = NULL;
	find_root_z = NULL;
	line1 = NULL;
	line2 = NULL;
	line3 = NULL;
	F = NULL;

	rk_pt_v = NULL;

	Sv1 = NULL;
	Sv2 = NULL;
	Sv3 = NULL;
	Sv4 = NULL;
	Gram2 = NULL;
	ST_N1 = NULL;
	ST_N2 = NULL;
	ST_w = NULL;
	STr_B = STr_Bv = STr_w = STr_z = STr_x = NULL;

	determine_line_v1 = NULL;
	determine_line_v2 = NULL;
	determine_line_v3 = NULL;

	lines_on_point_coords1 = NULL;
	lines_on_point_coords2 = NULL;

	subspace = NULL;

	line_pencil = NULL;
	Perp1 = NULL;


}

orthogonal::~orthogonal()
{
	//cout << "orthogonal::~orthogonal freeing v1" << endl;
	if (v1)
		FREE_int(v1);
	//cout << "orthogonal::~orthogonal freeing v2" << endl;
	if (v2)
		FREE_int(v2);
	//cout << "orthogonal::~orthogonal freeing v3" << endl;
	if (v3)
		FREE_int(v3);
	if (v4)
		FREE_int(v4);
	if (v5)
		FREE_int(v5);
	if (v_tmp)
		FREE_int(v_tmp);
	if (v_tmp2)
		FREE_int(v_tmp2);
	if (v_neighbor5)
		FREE_int(v_neighbor5);
	if (find_root_x)
		FREE_int(find_root_x);
	if (find_root_y)
		FREE_int(find_root_y);
	if (find_root_z)
		FREE_int(find_root_z);
	if (T1)
		FREE_int(T1);
	if (T2)
		FREE_int(T2);
	if (T3)
		FREE_int(T3);

#if 0
	//cout << "orthogonal::~orthogonal freeing F" << endl;
	if (F)
		delete F;
#endif

	//cout << "orthogonal::~orthogonal freeing A" << endl;
	if (A)
		FREE_int(A);
	//cout << "orthogonal::~orthogonal freeing B" << endl;
	if (B)
		FREE_int(B);
	//cout << "orthogonal::~orthogonal freeing P" << endl;
	if (P)
		FREE_int(P);
	//cout << "orthogonal::~orthogonal freeing L" << endl;
	if (L)
		FREE_int(L);
	if (Gram_matrix)
		FREE_int(Gram_matrix);
	if (subspace)
		delete subspace;
	if (line1)
		FREE_int(line1);
	if (line2)
		FREE_int(line2);
	if (line3)
		FREE_int(line3);
	if (minus_squares)
		FREE_int(minus_squares);
	if (minus_squares_without)
		FREE_int(minus_squares_without);
	if (minus_nonsquares)
		FREE_int(minus_nonsquares);
	if (f_is_minus_square)
		FREE_int(f_is_minus_square);
	if (index_minus_square)
		FREE_int(index_minus_square);
	if (index_minus_square_without)
		FREE_int(index_minus_square_without);
	if (index_minus_nonsquare)
		FREE_int(index_minus_nonsquare);
	if (rk_pt_v)
		FREE_int(rk_pt_v);
	if (Sv1)
		FREE_int(Sv1);
	if (Sv2)
		FREE_int(Sv2);
	if (Sv3)
		FREE_int(Sv3);
	if (Sv4)
		FREE_int(Sv4);
	if (Gram2)
		FREE_int(Gram2);
	if (ST_N1)
		FREE_int(ST_N1);
	if (ST_N2)
		FREE_int(ST_N2);
	if (ST_w)
		FREE_int(ST_w);
	if (STr_B)
		FREE_int(STr_B);
	if (STr_Bv)
		FREE_int(STr_Bv);
	if (STr_w)
		FREE_int(STr_w);
	if (STr_z)
		FREE_int(STr_z);
	if (STr_x)
		FREE_int(STr_x);
	if (determine_line_v1)
		FREE_int(determine_line_v1);
	if (determine_line_v2)
		FREE_int(determine_line_v2);
	if (determine_line_v3)
		FREE_int(determine_line_v3);
	if (lines_on_point_coords1)
		FREE_int(lines_on_point_coords1);
	if (lines_on_point_coords2)
		FREE_int(lines_on_point_coords2);
	
	if (line_pencil) {
		FREE_int(line_pencil);
		}
	if (Perp1) {
		FREE_int(Perp1);
		}
	//cout << "orthogonal::~orthogonal finished" << endl;
}

void orthogonal::init(int epsilon, int n,
		finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, j;
	geometry_global Gg;
	
	
	orthogonal::epsilon = epsilon;
	orthogonal::m = Gg.Witt_index(epsilon, n - 1);
	orthogonal::F = F;
	orthogonal::q = F->q;
	orthogonal::n = n;
	
	if (f_v) {
		cout << "orthogonal::init: epsilon=" << epsilon 
			<< " n=" << n << " (= vector space dimension)"
			<< " m=" << m << " (= Witt index)"
			<< " q=" << q 
			<< " verbose_level=" << verbose_level 
			<< endl;
		}

	if (EVEN(q)) {
		f_even = TRUE;
		}
	else {
		f_even = FALSE;
		}

	v1 = NEW_int(n);
	v2 = NEW_int(n);
	v3 = NEW_int(n);
	v4 = NEW_int(n);
	v5 = NEW_int(n);
	v_tmp = NEW_int(n);
	v_tmp2 = NEW_int(n);
	v_neighbor5 = NEW_int(n);
	find_root_x = NEW_int(n);
	find_root_y = NEW_int(n);
	find_root_z = NEW_int(n);
	T1 = NEW_int(n * n);
	T2 = NEW_int(n * n);
	T3 = NEW_int(n * n);
	line1 = NEW_int(q + 1);
	line2 = NEW_int(q + 1);
	line3 = NEW_int(q + 1);

	rk_pt_v = NEW_int(n);

	// for Siegel transformations:
	Sv1 = NEW_int(n);
	Sv2 = NEW_int(n);
	Sv3 = NEW_int(n);
	Sv4 = NEW_int(n);
	Gram2 = NEW_int(n * n);
	ST_N1 = NEW_int(n * n);
	ST_N2 = NEW_int(n * n);
	ST_w = NEW_int(n);
	STr_B = NEW_int(n * n);
	STr_Bv = NEW_int(n * n);
	STr_w = NEW_int(n);
	STr_z = NEW_int(n);
	STr_x = NEW_int(n);
	determine_line_v1 = NEW_int(n);
	determine_line_v2 = NEW_int(n);
	determine_line_v3 = NEW_int(n);
	
	form_c1 = 1;
	form_c2 = 0;
	form_c3 = 0;
	if (epsilon == -1) {
		F->choose_anisotropic_form(
				form_c1, form_c2, form_c3, verbose_level);
		}
	if (f_v) {
		cout << "orthogonal::init computing Gram matrix" << endl;
		}
	F->Gram_matrix(
			epsilon, n - 1,
			form_c1, form_c2, form_c3, Gram_matrix);
	if (f_v) {
		cout << "orthogonal::init "
				"computing Gram matrix done" << endl;
		}
	
	T1_m = Gg.count_T1(epsilon, m, q);
	if (f_vvv) {
		cout << "T1_m(" << epsilon << ","
				<< m << "," << q << ") = " << T1_m << endl;
		}
	T1_mm1 = Gg.count_T1(epsilon, m - 1, q);
	if (f_vvv) {
		cout << "T1_mm1(" << epsilon << ","
				<< m - 1 << "," << q << ") = " << T1_mm1 << endl;
		}
	if (m > 1) {
		T1_mm2 = Gg.count_T1(epsilon, m - 2, q);
		if (f_vvv) {
			cout << "T1_mm2(" << epsilon << ","
					<< m - 2 << "," << q << ") = " << T1_mm2 << endl;
			}
		}
	else {
		T1_mm2 = 0;
		}
	T2_m = Gg.count_T2(m, q);
	T2_mm1 = Gg.count_T2(m - 1, q);
	if (m > 1) {
		T2_mm2 = Gg.count_T2(m - 2, q);
		}
	else {
		T2_mm2 = 0;
		}
	N1_m = Gg.count_N1(m, q);
	N1_mm1 = Gg.count_N1(m - 1, q);
	if (m > 1) {
		N1_mm2 = Gg.count_N1(m - 2, q);
		}
	else {
		N1_mm2 = 0;
		}
	S_m = Gg.count_S(m, q);
	S_mm1 = Gg.count_S(m - 1, q);
	if (m > 1) {
		S_mm2 = Gg.count_S(m - 2, q);
		}
	else {
		S_mm2 = 0;
		}
	Sbar_m = Gg.count_Sbar(m, q);
	Sbar_mm1 = Gg.count_Sbar(m - 1, q);
	if (m > 1) {
		Sbar_mm2 = Gg.count_Sbar(m - 2, q);
		}
	else {
		Sbar_mm2 = 0;
		}
	
	if (f_vvv) {
		cout << "T1(" << m << "," << q << ") = " << T1_m << endl;
		if (m >= 1)
			cout << "T1(" << m - 1 << "," << q << ") = " << T1_mm1 << endl;
		if (m >= 2)
			cout << "T1(" << m - 2 << "," << q << ") = " << T1_mm2 << endl;
		cout << "T2(" << m << "," << q << ") = " << T2_m << endl;
		if (m >= 1)
			cout << "T2(" << m - 1 << "," << q << ") = " << T2_mm1 << endl;
		if (m >= 2)
			cout << "T2(" << m - 2 << "," << q << ") = " << T2_mm2 << endl;
		cout << "nb_pts_N1(" << m << "," << q << ") = " << N1_m << endl;
		if (m >= 1)
			cout << "nb_pts_N1(" << m - 1 << "," << q << ") = "
			<< N1_mm1 << endl;
		if (m >= 2)
			cout << "nb_pts_N1(" << m - 2 << "," << q << ") = "
			<< N1_mm2 << endl;
		cout << "S_m=" << S_m << endl;
		cout << "S_mm1=" << S_mm1 << endl;
		cout << "S_mm2=" << S_mm2 << endl;
		cout << "Sbar_m=" << Sbar_m << endl;
		cout << "Sbar_mm1=" << Sbar_mm1 << endl;
		cout << "Sbar_mm2=" << Sbar_mm2 << endl;
		cout << "N1_m=" << N1_m << endl;
		cout << "N1_mm1=" << N1_mm1 << endl;
		cout << "N1_mm2=" << N1_mm2 << endl;
		}
	

	if (epsilon == 1) {
#if 1
		int u;
		
		u = Gg.nb_pts_Qepsilon(epsilon, 2 * m - 1, q);
		if (T1_m != u) {
			cout << "T1_m != nb_pts_Qepsilon" << endl;
			cout << "T1_m=" << T1_m << endl;
			cout << "u=" << u << endl;
			exit(1);
			}
#endif
		init_hyperbolic(verbose_level - 3);
		if (f_v) {
			cout << "after init_hyperbolic" << endl;
			}
		}
	else if (epsilon == 0) {
		init_parabolic(verbose_level /*- 3*/);
		if (f_v) {
			cout << "after init_parabolic" << endl;
			}
		}
	else if (epsilon == -1) {
		nb_points = Gg.nb_pts_Qepsilon(epsilon, n - 1, q);
		nb_lines = 0;
		if (f_v) {
			cout << "nb_points=" << nb_points << endl;
			}
		//cout << "elliptic type not yet implemented" << endl;
		return;
		//exit(1);
		}
	else {
		cout << "epsilon = " << epsilon << " unknown" << endl;
		}
	
	nb_points = 0;
	for (i = 0; i < nb_point_classes; i++) {
		nb_points += P[i];
		}
	nb_lines = 0;
	for (i = 0; i < nb_line_classes; i++) {
		nb_lines += L[i];
		}
	lines_on_point_coords1 = NEW_int(alpha * n);
	lines_on_point_coords2 = NEW_int(alpha * n);

	if (m > 1) {
		subspace = NEW_OBJECT(orthogonal);
		if (f_v) {
			cout << "initializing subspace" << endl;
			}
		subspace->init(epsilon, n - 2, F, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "initializing subspace finished" << endl;
			cout << "subspace->epsilon=" << subspace->epsilon << endl;
			cout << "subspace->n=" << subspace->n << endl;
			cout << "subspace->m=" << subspace->m << endl;
			}
		}
	else {
		if (f_v) {
			cout << "no subspace" << endl;
			}
		subspace = NULL;
		}

	if (f_v) {
		cout << "O^" << epsilon << "(" << n << "," << q << ")" << endl;
		cout << "epsilon=" << epsilon
				<< " n=" << n << " m=" << m << " q=" << q << endl;
		cout << "pt_P = " << pt_P << endl;
		cout << "pt_Q=" << pt_Q << endl;
		cout << "nb_points = " << nb_points << endl;
		cout << "nb_lines = " << nb_lines << endl;
		cout << "alpha = " << alpha << endl;
		cout << "beta = " << beta << endl;
		cout << "gamma = " << gamma << endl;
		}


	if (f_v) {
		cout << "before allocating line_pencil of size " << alpha << endl;
		}
	line_pencil = NEW_int(alpha);
	if (f_v) {
		cout << "before allocating Perp1 of size "
				<< alpha * (q + 1) << endl;
		}
	Perp1 = NEW_int(alpha * (q + 1));
	if (f_v) {
		cout << "after allocating Perp1" << endl;
		}



	if (f_v) {
		print_schemes();
		cout << "Gram matrix:" << endl;
		print_integer_matrix_width(cout,
				Gram_matrix, n, n, n, F->log10_of_q + 1);
		}
	if (FALSE) {
		for (i = 0; i < T1_m; i++) {
			F->Q_epsilon_unrank(v1, 1, epsilon, n - 1,
					form_c1, form_c2, form_c3, i);
			cout << i << " : ";
			int_vec_print(cout, v1, n);
			j = F->Q_epsilon_rank(v1, 1, epsilon, n - 1,
					form_c1, form_c2, form_c3);
			cout << " : " << j << endl;
			}
		}
	if (FALSE) {
		if (nb_points < 300) {
			cout << "points of O^" << epsilon
					<< "(" << n << "," << q << ") by type:" << endl;
			list_points_by_type(verbose_level);
			}
		if (nb_points < 300 && nb_lines < 300) {
			cout << "points and lines of O^" << epsilon
					<< "(" << n << "," << q << ") by type:" << endl;
			list_all_points_vs_points(verbose_level);
			}
		}
	if (f_v) {
		cout << "orthogonal::init finished" << endl;
		if (subspace) {
			cout << "subspace->epsilon=" << subspace->epsilon << endl;
			cout << "subspace->n=" << subspace->n << endl;
			cout << "subspace->m=" << subspace->m << endl;
			}
		}
}

void orthogonal::init_parabolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	geometry_global Gg;

	//int a, b, c;
	
	if (f_v) {
		cout << "init_parabolic m=" << m << " q=" << q << endl;
		}
	
	nb_point_classes = 7;
	nb_line_classes = 8;
	subspace_point_type = 5;
	subspace_line_type = 6;
	
	A = NEW_int(nb_point_classes * nb_line_classes);
	B = NEW_int(nb_point_classes * nb_line_classes);
	P = NEW_int(nb_point_classes);
	L = NEW_int(nb_line_classes);

	for (i = 0; i < nb_point_classes * nb_line_classes; i++) {
		A[i] = B[i] = 0;
		}

	if (f_even) {
		init_parabolic_even(verbose_level);
		}
	else {
		init_parabolic_odd(verbose_level);
		}
	
	
	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	P[6] = p7;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	L[7] = l8;

	pt_P = Gg.count_T1(1, m - 1, q);
	pt_Q = pt_P + Gg.count_S(m - 1, q);

	for (j = 0; j < nb_line_classes; j++) {
		if (L[j] == 0) {
			for (i = 0; i < nb_point_classes; i++) {
				B[i * nb_line_classes + j] = 0;
				}
			}
		}
	if (f_v) {
		cout << "init_parabolic done" << endl;
		}
}

void orthogonal::init_parabolic_even(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	geometry_global Gg;
	
	if (f_v) {
		cout << "init_parabolic_even" << endl;
		}
	if (m >= 2)
		beta = Gg.count_T1(0, m - 2, q);
	else
		beta = 0;
	if (m >= 1) {
		alpha = Gg.count_T1(0, m - 1, q);
		gamma = alpha * beta / (q + 1);
		}
	else {
		alpha = 0;
		gamma = 0;
		}
	delta = alpha - 1 - q * beta;
	zeta = alpha - beta - 2 * (q - 1) * beta - q - 1;
	//cout << "alpha = " << alpha << endl;
	//cout << "beta = " << beta << endl;
	//cout << "gamma = " << gamma << endl;
	//cout << "delta = " << delta << endl;
	//cout << "zeta = " << zeta << endl;
	
	p1 = q - 1;
	p2 = alpha * (q - 1) * (q - 1);
	p3 = p4 = (q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;
		
	l1 = alpha * (q - 1);
	l2 = (q - 1) * (q - 1) * alpha * beta;
	l3 = (q - 1) * alpha * delta;
	l4 = l5 = alpha * beta * (q - 1);
	l6 = gamma;
	l7 = l8 = alpha;

	a11 = alpha;
	a21 = a36 = a47 = a56 = a57 = 1;
	a22a = a33 = a44 = q * beta;
	a22b = a32b = a42b = delta;
	a51 = q - 1;
	a52a = zeta;
	a53 = a54 = (q - 1) * beta;
	a55 = beta;
	a66 = a77 = alpha;
		
	b11 = b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b21 = b22b = b36 = b47 = q - 1;
	b22a = b33 = b44 = q;
	b55 = q + 1;


	fill(A, 1, 1, a11);
	fill(A, 2, 1, a21);
	fill(A, 5, 1, a51);
	
	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);
	
	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);
	
	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);
	
	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);
	
	fill(A, 5, 6, a55);
	
	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);
	
	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);
	
	fill(B, 1, 1, b11);
	fill(B, 2, 1, b21);
	fill(B, 5, 1, b51);
	
	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);
	
	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);
	
	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);
	
	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);
	
	fill(B, 5, 6, b55);
	
	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);
	
	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);
	if (f_v) {
		cout << "init_parabolic_even done" << endl;
		}
}

void orthogonal::init_parabolic_odd(int verbose_level)
{
	int a, b, c, i, j;
	int f_v = (verbose_level >= 1);
	geometry_global Gg;
	
	if (f_v) {
		cout << "init_parabolic_odd" << endl;
		cout << "count_N1(" << m - 1 << "," << q << ")=";
		cout << Gg.count_N1(m - 1, q) << endl;
		cout << "count_S(" << m - 1 << "," << q << ")=";
		cout << Gg.count_S(m - 1, q) << endl;
		}
	a = Gg.count_N1(m - 1, q) * (q - 1) / 2;
	b = Gg.count_S(m - 1, q) * (q - 1);
	c = (((q - 1) / 2) - 1) * (q - 1) * Gg.count_N1(m - 1, q);
	p1 = a + b + c;
	p2 = a + ((q - 1) / 2) * (q - 1) * Gg.count_N1(m - 1, q);
	if (f_v) {
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "c=" << c << endl;
		cout << "p1=" << p1 << endl;
		cout << "p2=" << p2 << endl;
		}
		
	if (m >= 2)
		beta = Gg.count_T1(0, m - 2, q);
	else
		beta = 0;
	if (m >= 1) {
		alpha = Gg.count_T1(0, m - 1, q);
		gamma = alpha * beta / (q + 1);
		}
	else {
		alpha = 0;
		gamma = 0;
		}
	if (f_v) {
		cout << "alpha=" << alpha << endl;
		cout << "beta=" << beta << endl;
		cout << "gamma=" << gamma << endl;
		}
	p3 = p4 = (q - 1) * alpha;
	p5 = alpha;
	p6 = p7 = 1;
	if (f_v) {
		cout << "p3=" << p3 << endl;
		cout << "p5=" << p5 << endl;
		cout << "p6=" << p6 << endl;
		}
		
	omega = (q - 1) * Gg.count_S(m - 2, q) +
		Gg.count_N1(m - 2, q) * (q - 1) / 2 +
		Gg.count_N1(m - 2, q) * ((q - 1) / 2 - 1) * (q - 1);
	if (f_v) {
		cout << "omega=" << omega << endl;
		}
	zeta = alpha - omega - 2 * (q - 1) * beta - beta - 2;
	if (f_v) {
		cout << "zeta=" << zeta << endl;
		}

		
	a66 = a77 = alpha;
	a56 = a57 = a36 = a47 = 1;
	a55 = beta;
	a53 = a54 = (q - 1) * beta;
	a33 = a44 = q * beta;
	a32b = a42b = alpha - 1 - q * beta;
	a51 = omega;
	a52a = zeta;
		
	l1 = p5 * omega;
	l2 = p5 * zeta;
	l3 = (q - 1) * alpha * (alpha - 1 - q * beta);
	l4 = l5 = (q - 1) * alpha * beta;
	l6 = gamma;
	l7 = l8 = alpha;

	if (f_v) {
		cout << "l1=" << l1 << endl;
		cout << "l2=" << l2 << endl;
		cout << "l3=" << l3 << endl;
		cout << "l4=" << l4 << endl;
		cout << "l5=" << l5 << endl;
		cout << "l6=" << l6 << endl;
		cout << "l7=" << l7 << endl;
		cout << "l8=" << l8 << endl;
		}
	
	if (p1) {
		lambda = l1 * q / p1;
		}
	else {
		lambda = 0;
		}
	if (p2) {
		delta = l2 * q / p2;
		}
	else {
		delta = 0;
		}
	a11 = lambda;
	a22a = delta;
	a12b = alpha - lambda;
	a22b = alpha - delta;
	mu = alpha - lambda;
	nu = alpha - delta;
	a12b = mu;
	a22b = nu;
		
	b51 = b52a = b32b = b42b = b53 = b54 = b56 = b57 = b66 = b77 = 1;
	b11 = b22a = b33 = b44 = q;
	b55 = q + 1;
	b36 = b47 = q - 1;
	if (l3) {
		b12b = p1 * mu / l3;
		b22b = p2 * nu / l3;
		}
	else {
		b12b = 0;
		b22b = 0;
		}
		

	fill(A, 1, 1, a11);
	fill(A, 5, 1, a51);
	
	fill(A, 2, 2, a22a);
	fill(A, 5, 2, a52a);
	
	fill(A, 1, 3, a12b);
	fill(A, 2, 3, a22b);
	fill(A, 3, 3, a32b);
	fill(A, 4, 3, a42b);
	
	fill(A, 3, 4, a33);
	fill(A, 5, 4, a53);
	
	fill(A, 4, 5, a44);
	fill(A, 5, 5, a54);
	
	fill(A, 5, 6, a55);
	
	fill(A, 3, 7, a36);
	fill(A, 5, 7, a56);
	fill(A, 6, 7, a66);
	
	fill(A, 4, 8, a47);
	fill(A, 5, 8, a57);
	fill(A, 7, 8, a77);
	
	fill(B, 1, 1, b11);
	fill(B, 5, 1, b51);
	
	fill(B, 2, 2, b22a);
	fill(B, 5, 2, b52a);
	
	fill(B, 1, 3, b12b);
	fill(B, 2, 3, b22b);
	fill(B, 3, 3, b32b);
	fill(B, 4, 3, b42b);
	
	fill(B, 3, 4, b33);
	fill(B, 5, 4, b53);
	
	fill(B, 4, 5, b44);
	fill(B, 5, 5, b54);
	
	fill(B, 5, 6, b55);
	
	fill(B, 3, 7, b36);
	fill(B, 5, 7, b56);
	fill(B, 6, 7, b66);
	
	fill(B, 4, 8, b47);
	fill(B, 5, 8, b57);
	fill(B, 7, 8, b77);
	
	minus_squares = NEW_int((q-1)/2);
	minus_squares_without = NEW_int((q-1)/2 - 1);
	minus_nonsquares = NEW_int((q-1)/2);
	f_is_minus_square = NEW_int(q);
	index_minus_square = NEW_int(q);
	index_minus_square_without = NEW_int(q);
	index_minus_nonsquare = NEW_int(q);
	a = b = c = 0;
	if (f_v) {
		cout << "computing minus_squares:" << endl;
		}
	for (i = 0; i < q; i++) {
		index_minus_square[i] = -1;
		index_minus_square_without[i] = -1;
		index_minus_nonsquare[i] = -1;
		f_is_minus_square[i]= FALSE;
		}
	for (i = 0; i < q - 1; i++) {
		j = F->alpha_power(i);
		if (is_minus_square(i)) {
			if (f_v) {
				cout << "i=" << i << " j=" << j
						<< " is minus a square" << endl;
				}
			f_is_minus_square[j]= TRUE;
			minus_squares[a] = j;
			index_minus_square[j] = a;
			if (j != F->negate(1)) {
				minus_squares_without[b] = j;
				index_minus_square_without[j] = b;
				b++;
				}
			a++;
			}
		else {
			minus_nonsquares[c] = j;
			index_minus_nonsquare[j] = c;
			c++;
			}
		}
	if (f_v) {
		cout << "minus_squares:" << endl;
		for (i = 0; i < a; i++) {
			cout << i << " : " << minus_squares[i] << endl;
			}
		cout << "minus_squares_without:" << endl;
		for (i = 0; i < b; i++) {
			cout << i << " : " << minus_squares_without[i] << endl;
			}
		cout << "minus_nonsquares:" << endl;
		for (i = 0; i < c; i++) {
			cout << i << " : " << minus_nonsquares[i] << endl;
			}
		print_minus_square_tables();
		}
	if (f_v) {
		cout << "init_parabolic_odd done" << endl;
		}
}

void orthogonal::print_minus_square_tables()
{
	int i;
	
	cout << "field element indices and f_minus_square:" << endl;
	for (i = 0; i < q; i++) {
			cout << i << " : " 
			<< setw(3) << index_minus_square[i] << "," 
			<< setw(3) << index_minus_square_without[i] << "," 
			<< setw(3) << index_minus_nonsquare[i] << " : " 
			<< setw(3) << f_is_minus_square[i] << endl;
		}
}

void orthogonal::init_hyperbolic(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory_domain NT;
	
	if (f_v) {
		cout << "init_hyperbolic" << endl;
		}

	nb_point_classes = 6;
	nb_line_classes = 7;
	subspace_point_type = 4;
	subspace_line_type = 5;
	
	p5 = p6 = 1;
	p4 = T1_mm1;
	p2 = p3 = (q - 1) * T1_mm1;
	p1 = NT.i_power_j(q, 2 * m - 2) - 1 - p2;
	l6 = l7 = T1_mm1;
	l5 = T2_mm1;
	l3 = l4 = (q - 1) * T1_mm2 * T1_mm1;
	
	alpha = T1_mm1;
	beta = T1_mm2;
	gamma = alpha * beta / (q + 1);
	
	a47 = a46 = a37 = a26 = 1;
	b67 = b56 = b47 = b46 = b44 = b43 = b41 = b32 = b22 = 1;
	b45 = q + 1;
	b37 = b26 = b12 = q - 1;
	b34 = b23 = b11 = q;
	a67 = a56 = T1_mm1;
	a45 = T1_mm2;
	a44 = a43 = T1_mm2 * (q - 1);
	
	a41 = (q - 1) * N1_mm2;
	
	a34 = q * T1_mm2;
	a23 = q * T1_mm2;
	a32 = a22 = T1_mm1 - 1 - a23;
	
	l2 = p2 * a22;
	if (p1 == 0) {
		//cout << "orthogonal::init_hyperbolic p1 == 0" << endl;
		a12 = 0;
		}
	else {
		a12 = l2 * (q - 1) / p1;
		}
	a11 = T1_mm1 - a12;
	l1 = a11 * p1 / q;
		
	//a41 = l1 / T1_mm1;
		
	
	A = NEW_int(6 * 7);
	B = NEW_int(6 * 7);
	P = NEW_int(6);
	L = NEW_int(7);

	for (i = 0; i < 6 * 7; i++)
		A[i] = B[i] = 0;
	P[0] = p1;
	P[1] = p2;
	P[2] = p3;
	P[3] = p4;
	P[4] = p5;
	P[5] = p6;
	L[0] = l1;
	L[1] = l2;
	L[2] = l3;
	L[3] = l4;
	L[4] = l5;
	L[5] = l6;
	L[6] = l7;
	fill(A, 1, 1, a11);
	fill(A, 1, 2, a12);
	fill(A, 2, 2, a22);
	fill(A, 2, 3, a23);
	fill(A, 2, 6, a26);
	fill(A, 3, 2, a32);
	fill(A, 3, 4, a34);
	fill(A, 3, 7, a37);
	fill(A, 4, 1, a41);
	fill(A, 4, 3, a43);
	fill(A, 4, 4, a44);
	fill(A, 4, 5, a45);
	fill(A, 4, 6, a46);
	fill(A, 4, 7, a47);
	fill(A, 5, 6, a56);
	fill(A, 6, 7, a67);
	
	fill(B, 1, 1, b11);
	fill(B, 1, 2, b12);
	fill(B, 2, 2, b22);
	fill(B, 2, 3, b23);
	fill(B, 2, 6, b26);
	fill(B, 3, 2, b32);
	fill(B, 3, 4, b34);
	fill(B, 3, 7, b37);
	fill(B, 4, 1, b41);
	fill(B, 4, 3, b43);
	fill(B, 4, 4, b44);
	fill(B, 4, 5, b45);
	fill(B, 4, 6, b46);
	fill(B, 4, 7, b47);
	fill(B, 5, 6, b56);
	fill(B, 6, 7, b67);

	pt_P = p4;
	pt_Q = p4 + 1 + p3;

	if (f_v) {
		cout << "init_hyperbolic done" << endl;
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

void orthogonal::fill(int *M, int i, int j, int a)
{
	M[(i - 1) * nb_line_classes + j - 1] = a;
}

// #############################################################################
// orthogonal_hyperbolic.cpp
// #############################################################################

//##############################################################################
// ranking / unranking points according to the partition:
//##############################################################################

int orthogonal::hyperbolic_type_and_index_to_point_rk(
		int type, int index)
{
	int rk;
	
	rk = 0;
	if (type == 4) {
		if (index >= p4) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p4" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	rk += p4;
	if (type == 6) {
		if (index >= p6) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p6" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	rk += p6;
	if (type == 3) {
		if (index >= p3) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p3" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	rk += p3;
	if (type == 5) {
		if (index >= p5) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p5" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	rk += p5;
	if (type == 2) {
		if (index >= p2) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p2" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	rk += p2;
	if (type == 1) {
		if (index >= p1) {
			cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
					"index >= p1" << endl;
			exit(1);
			}
		rk += index;
		return rk;
		}
	cout << "error in orthogonal::hyperbolic_type_and_index_to_point_rk, "
			"unknown type" << endl;
	exit(1);
}

void orthogonal::hyperbolic_point_rk_to_type_and_index(
		int rk, int &type, int &index)
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
		int &p1, int &p2, int rk, int verbose_level)
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

int orthogonal::hyperbolic_rank_line(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pt1_type, pt2_type;
	int pt1_index, pt2_index;
	int line_type, rk = 0;
	int cp1, cp2;
	
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line" << endl;
	}
	if (m == 0) {
		cout << "orthogonal::hyperbolic_rank_line Witt index zero, "
				"there is no line to rank" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line p1=" << p1 << " p2=" << p2 << endl;
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
		cout << "orthogonal::hyperbolic_rank_line line_type=" << line_type << endl;
		}
	canonical_points_of_line(line_type, p1, p2,
			cp1, cp2, verbose_level);
	if (f_v) {
		cout << "orthogonal::hyperbolic_rank_line canonical points "
				"cp1=" << cp1 << " cp2=" << cp2 << endl;
		}
	if (line_type == 1) {
		return rk + rank_line_L1(cp1, cp2, verbose_level);
		}
	rk += l1;
	if (line_type == 2) {
		return rk + rank_line_L2(cp1, cp2, verbose_level);
		}
	rk += l2;
	if (line_type == 3) {
		return rk + rank_line_L3(cp1, cp2, verbose_level);
		}
	rk += l3;
	if (line_type == 4) {
		return rk + rank_line_L4(cp1, cp2, verbose_level);
		}
	rk += l4;
	if (line_type == 5) {
		return rk + rank_line_L5(cp1, cp2, verbose_level);
		}
	rk += l5;
	if (line_type == 6) {
		return rk + rank_line_L6(cp1, cp2, verbose_level);
		}
	rk += l6;
	if (line_type == 7) {
		return rk + rank_line_L7(cp1, cp2, verbose_level);
		}
	rk += l7;
	cout << "error in orthogonal::hyperbolic_rank_line, illegal line_type" << endl;
	exit(1);
}

void orthogonal::unrank_line_L1(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P4_index, P4_sub_index, P4_line_index;
	int P4_field_element, root, i;
	
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
		cout << "orthogonal::unrank_line_L1 done" << index
	}
}

int orthogonal::rank_line_L1(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P4_index, P4_sub_index, P4_line_index;
	int P4_field_element, root, i;
	int P4_field_element_inverse;
	int index, a, b;
	
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
		cout << "orthogonal::rank_line_L1 after rank_N1, P4_line_index=" << P4_line_index << endl;
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
		exit(1);
		}
	return index;
}

void orthogonal::unrank_line_L2(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P3_index, P3_sub_index, root, a, b, c, d, e, i;
	int P3_point, P3_field_element;
	
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
				cout << "orthogonal::unrank_line_L2 case 2, a=" << a << " b=" << b << endl;
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
				cout << "orthogonal::unrank_line_L2 case 3, a=" << a << " b=" << b << endl;
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

int orthogonal::rank_line_L2(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P3_index, P3_sub_index, root, a, b, c, d, i, alpha;
	int P3_point, P3_field_element;
	int index;
	
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
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P4_index, P4_sub_index, P4_line_index;
	int P4_field_element, root, i, e;
	
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

int orthogonal::rank_line_L3(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P4_index, P4_sub_index, P4_line_index;
	int P4_field_element, root, i, index;
	int a, b;
	
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
	index = P4_index * a43 + P4_sub_index;
	
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
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P4_index, P4_sub_index, P4_line_index;
	int P4_field_element, root, i, e;
	
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

int orthogonal::rank_line_L4(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int P3_index, P3_sub_index, P3_line_index;
	int P3_field_element, root, i, index;
	int a, b;
	
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
		int &p1, int &p2, int index, int verbose_level)
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

int orthogonal::rank_line_L5(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int index;
	
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
		int &p1, int &p2, int index, int verbose_level)
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

int orthogonal::rank_line_L6(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int index;
	
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
		int &p1, int &p2, int index, int verbose_level)
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

int orthogonal::rank_line_L7(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int index;
	
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
	int line_type, int pt1, int pt2,
	int &cpt1, int &cpt2, int verbose_level)
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
		int pt1, int pt2, int &cpt1, int &cpt2)
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
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int a, b, c, d, lambda, i, p1, p2;
	
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
		int pt1, int pt2, int &cpt1, int &cpt2)
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
		int pt1, int pt2, int &cpt1, int &cpt2)
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
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	cpt1 = pt1;
	cpt2 = pt2;
}

void orthogonal::canonical_points_L6(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	canonical_points_L3(pt1, pt2, cpt1, cpt2);
}

void orthogonal::canonical_points_L7(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	canonical_points_L4(pt1, pt2, cpt1, cpt2);
}

int orthogonal::hyperbolic_line_type_given_point_types(
		int pt1, int pt2, int pt1_type, int pt2_type)
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

int orthogonal::hyperbolic_decide_P1(int pt1, int pt2)
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

int orthogonal::hyperbolic_decide_P2(int pt1, int pt2)
{
	if (triple_is_collinear(pt1, pt2, pt_Q)) {
		return 6;
		}
	else {
		return 3;
		}
}

int orthogonal::hyperbolic_decide_P3(int pt1, int pt2)
{
	if (triple_is_collinear(pt1, pt2, pt_P)) {
		return 7;
		}
	else {
		return 4;
		}
}

int orthogonal::find_root_hyperbolic(
		int rk2, int m, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int root, u, v;

	if (f_v) {
		cout << "find_root_hyperbolic "
				"rk2=" << rk2 << " m=" << m << endl;
		}
	if (rk2 == 0) {
		cout << "find_root_hyperbolic: "
				"rk2 must not be 0" << endl;
		exit(1);
		}
	if (m == 1) {
		cout << "find_root_hyperbolic: "
				"m must not be 1" << endl;
		exit(1);
		}
	find_root_hyperbolic_xyz(rk2, m,
			find_root_x, find_root_y, find_root_z,
			verbose_level);
	if (f_v) {
		cout << "find_root_hyperbolic root=" << endl;
		int_vec_print(cout, find_root_z, 2 * m);
		cout << endl;
		}
	
	u = evaluate_hyperbolic_bilinear_form(
			find_root_z, find_root_x, 1, m);
	if (u == 0) {
		cout << "find_root_hyperbolic u=" << u << endl;
		exit(1);
		}
	v = evaluate_hyperbolic_bilinear_form(
			find_root_z, find_root_y, 1, m);
	if (v == 0) {
		cout << "find_root_hyperbolic v=" << v << endl;
		exit(1);
		}
	root = rank_Sbar(find_root_z, 1, m);
	if (f_v) {
		cout << "find_root_hyperbolic root=" << root << endl;
		}
	return root;
}

void orthogonal::find_root_hyperbolic_xyz(
		int rk2, int m, int *x, int *y, int *z,
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
		cout << "find_root_hyperbolic_xyz y=" << endl;
		int_vec_print(cout, y, 2 * m);
		cout << endl;
		}
	if (y[0]) {
		if (f_vv) {
			cout << "detected y[0] is nonzero" << endl;
			}
		z[1] = 1;
		if (f_v) {
			cout << "find_root_hyperbolic_xyz z=" << endl;
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
						cout << "find_root_hyperbolic_xyz z=" << endl;
						int_vec_print(cout, z, 2 * m);
						cout << endl;
						}
					return;
					}
				else {
					z[1] = 1;
					z[i - 1] = 1;
					if (f_v) {
						cout << "find_root_hyperbolic_xyz z=" << endl;
						int_vec_print(cout, z, 2 * m);
						cout << endl;
						}
					return;
					}
				}
			}
		cout << "find_root_hyperbolic_xyz error: y is zero vector" << endl;
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
			cout << "find_root_hyperbolic_xyz z=" << endl;
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
			cout << "find_root_hyperbolic_xyz z=" << endl;
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
				cout << "find_root_hyperbolic_xyz z=" << endl;
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
				cout << "find_root_hyperbolic_xyz z=" << endl;
				int_vec_print(cout, z, 2 * m);
				cout << endl;
				}
			return;
			}
		cout << "find_root_hyperbolic_xyz error "
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
		cout << "find_root_hyperbolic_xyz z=" << endl;
		int_vec_print(cout, z, 2 * m);
		cout << endl;
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



// #############################################################################
// orthogonal_parabolic.cpp
// #############################################################################

//##############################################################################
// ranking / unranking points according to the partition:
//##############################################################################

int orthogonal::parabolic_type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;
	
	if (f_v) {
		cout << "parabolic_type_and_index_to_point_rk "
			"type=" << type << " index=" << index
			<< " epsilon=" << epsilon << " n=" << n << endl;
		}
	if (type == 3) {
		int field, sub_index, len;
		
		len = alpha;
		field = index / len;
		sub_index = index % len;
		field++;
		if (f_vv) {
			cout << "field=" << field
					<< " sub_index=" << sub_index << endl;
			}
		subspace->unrank_point(v_tmp2, 1, sub_index, verbose_level - 1);
		v_tmp2[n - 2] = 0;
		v_tmp2[n - 1] = field;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	else if (type == 4) {
		int field, sub_index, len;
		
		len = alpha;
		field = index / len;
		sub_index = index % len;
		field++;
		if (f_vv) {
			cout << "field=" << field << " sub_index=" << sub_index << endl;
			}
		subspace->unrank_point(v_tmp2, 1, sub_index, verbose_level - 1);
		v_tmp2[n - 2] = field;
		v_tmp2[n - 1] = 0;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	else if (type == 5) {
		if (f_v) {
			cout << "parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index << endl;
			cout << "parabolic_type_and_index_to_point_rk "
				"before subspace->unrank_point" << endl;
			}
		if (subspace == NULL) {
			cout << "parabolic_type_and_index_to_point_rk "
				"subspace == NULL" << endl;
			exit(1);
			}
		subspace->unrank_point(v_tmp2, 1, index, verbose_level /*- 1*/);
		if (f_v) {
			cout << "parabolic_type_and_index_to_point_rk "
					"after subspace->unrank_point" << endl;
			}
		v_tmp2[n - 2] = 0;
		v_tmp2[n - 1] = 0;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	else if (type == 6) {
		if (index < 1) {
			rk = pt_Q;
			if (f_v) {
				cout << "parabolic_type_and_index_to_point_rk "
					"type=" << type << " index=" << index
					<< " rk=" << rk << endl;
				}
			return rk;
			}
		else {
			cout << "error in parabolic_P3to7_type_and_index_to_point_rk, "
					"illegal index" << endl;
			exit(1);
			}
		}
	else if (type == 7) {
		if (index < 1) {
			rk = pt_P;
			if (f_v) {
				cout << "parabolic_type_and_index_to_point_rk "
					"type=" << type << " index=" << index
					<< " rk=" << rk << endl;
				}
			return rk;
			}
		else {
			cout << "error in parabolic_P3to7_type_and_index_to_point_rk, "
				"illegal index" << endl;
			exit(1);
			}
		}
	else {
		if (f_even) {
			return parabolic_even_type_and_index_to_point_rk(
					type, index, verbose_level);
			}
		else {
			return parabolic_odd_type_and_index_to_point_rk(
					type, index, verbose_level);
			}
		}
}

int orthogonal::parabolic_even_type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int rk;
	
	if (f_v) {
		cout << "parabolic_even_type_and_index_to_point_rk "
			"type=" << type << " index=" << index << endl;
		}	
	if (type == 1) {
		parabolic_even_type1_index_to_point(index, v_tmp2);
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_even_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	else if (type == 2) {
		parabolic_even_type2_index_to_point(index, v_tmp2);
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_even_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	cout << "error in parabolic_even_type_and_index_to_point_rk "
			"illegal type " << type << endl;
	exit(1);
}

void orthogonal::parabolic_even_type1_index_to_point(int index, int *v)
{
	int a, b;
	
	if (index >= p1) {
		cout << "error in parabolic_even_type1_index_to_point, "
				"index >= p1" << endl;
		exit(1);
		}
	zero_vector(v + 1, 1, 2 * (m - 1));
	a = 1 + index;
	b = F->inverse(a);
	v[0] = 1;
	v[1 + 2 * (m - 1) + 0] = a;
	v[1 + 2 * (m - 1) + 1] = b;
}

void orthogonal::parabolic_even_type2_index_to_point(
		int index, int *v)
{
	int a, b, c, d, l, ll, lll, field1, field2;
	int sub_index, sub_sub_index;
	
	l = (q - 1) * N1_mm1;
	if (index < l) {
		field1 = index / N1_mm1;
		sub_index = index % N1_mm1;
		v[0] = 0;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		a = 1 + field1;
		b = 1;
		c = a;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		change_form_value(v + 1, 1, m - 1, c);
		//int_vec_print(cout, v, n);
		return;
		}
	index -= l;
	ll = S_mm1 - 1;
	l = (q - 1) * ll;
	if (index < l) {
		field1 = index / ll;
		sub_index = index % ll;
		lll = Sbar_mm1;
		field2 = sub_index / lll;
		sub_sub_index = sub_index % lll;
		v[0] = 1;
		unrank_Sbar(v + 1, 1, m - 1, sub_sub_index);
		scalar_multiply_vector(v + 1, 1, n - 3, 1 + field2);
		a = 1 + field1;
		b = F->inverse(a);
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		return;
		}
	index -= l;
	l = (q - 2) * (q - 1) * N1_mm1;
	if (index < l) {
		ll = (q - 1) * N1_mm1;
		field1 = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		//cout << "field1=" << field1 << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		v[0] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		a = 2 + field1;
		b = 1 + field2;
		c = F->mult(a, F->inverse(b));
		v[1 + 2 * (m - 1) + 0] = b;
		v[1 + 2 * (m - 1) + 1] = c;
		d = F->add(1, a);
		change_form_value(v + 1, 1, m - 1, d);
		return;
		}
	else {
		cout << "error in parabolic_even_type2_index_to_point "
				"illegal index" << endl;
		exit(1);
		}		
}

int orthogonal::parabolic_odd_type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int rk;
	
	if (f_v) {
		cout << "parabolic_odd_type_and_index_to_point_rk "
			"type=" << type << " index=" << index << endl;
		}	
	if (type == 1) {
		parabolic_odd_type1_index_to_point(index, v_tmp2, verbose_level);
		if (f_v) {
			cout << "parabolic_odd_type_and_index_to_point_rk created ";
			int_vec_print(cout, v_tmp2, n);
			cout << endl;
			}
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_odd_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	else if (type == 2) {
		parabolic_odd_type2_index_to_point(index, v_tmp2, verbose_level);
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_odd_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
			}
		return rk;
		}
	cout << "error in parabolic_odd_type_and_index_to_point_rk "
			"illegal type " << type << endl;
	exit(1);
}

void orthogonal::parabolic_odd_type1_index_to_point(
		int index, int *v, int verbose_level)
{
	int a, b, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_odd_type1_index_to_point "
				"m = " << m << " index = " << index << endl;
		}
	if (index >= p1) {
		cout << "error in parabolic_odd_type1_index_to_point, "
				"index >= p1" << endl;
		exit(1);
		}
	l = (q - 1) / 2 * N1_mm1;
	if (index < l) {
		ms_idx = index / N1_mm1;
		sub_index = index % N1_mm1;
		field1 = minus_squares[ms_idx];
		if (f_v) {
			cout << "case a) ms_idx = " << ms_idx
				<< " sub_index=" << sub_index
				<< " field1 = " << field1 << endl;
			}
		v[0] = 0;
		v[1 + 2 * (m - 1) + 0] = field1;
		v[1 + 2 * (m - 1) + 1] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		c = F->negate(field1);
		change_form_value(v + 1, 1, m - 1, c);
		return;
		}
	index -= l;
	l = (q - 1) * S_mm1;
	if (index < l) {
		field1 = index / S_mm1;
		sub_index = index % S_mm1;
		if (f_v) {
			cout << "case b) sub_index=" << sub_index
					<< " field1 = " << field1 << endl;
			}
		if (sub_index == 0) {
			a = 1 + field1;
			b = F->mult(F->inverse(a), F->negate(1));
			v[0] = 1;
			v[1 + 2 * (m - 1) + 0] = a;
			v[1 + 2 * (m - 1) + 1] = b;
			zero_vector(v + 1, 1, n - 3);
			return;
			}
		else {
			sub_index--;
			field2 = sub_index / Sbar_mm1;
			sub_sub_index = sub_index % Sbar_mm1;
			//cout << "field1=" << field1 << " field2=" << field2
			//<< " sub_sub_index=" << sub_sub_index << endl;
			a = 1 + field1;
			b = F->mult(F->inverse(a), F->negate(1));
			v[0] = 1;
			v[1 + 2 * (m - 1) + 0] = a;
			v[1 + 2 * (m - 1) + 1] = b;
			unrank_Sbar(v + 1, 1, m - 1, sub_sub_index);
			scalar_multiply_vector(v + 1, 1, n - 3, 1 + field2);
			return;
			}
		}
	index -= l;
	l = ((q - 1) / 2 - 1) * (q - 1) * N1_mm1;
	ll = (q - 1) * N1_mm1;
	//cout << "index = " << index << " l=" << l << endl;
	if (index < l) {
		ms_idx = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		field1 = minus_squares_without[ms_idx];
		if (f_v) {
			cout << "case c) ms_idx = " << ms_idx 
				<< " sub_index=" << sub_index 
				<< " field2 = " << field2 
				<< " sub_sub_index=" << sub_sub_index 
				<< " field1 = " << field1 
				<< endl;
			}
		a = 1 + field2;
		b = F->mult(F->inverse(a), field1);
		v[0] = 1;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		c = F->negate(F->add(1, field1));
		change_form_value(v + 1, 1, m - 1, c);
		return;
		}
	else {
		cout << "error in parabolic_odd_type1_index_to_point "
				"illegal index" << endl;
		exit(1);
		}
}

void orthogonal::parabolic_odd_type2_index_to_point(
		int index, int *v, int verbose_level)
{
	int a, b, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "parabolic_odd_type2_index_to_point "
				"index = " << index << endl;
		}
	if (index >= p1) {
		cout << "error in parabolic_odd_type2_index_to_point, "
				"index >= p1" << endl;
		exit(1);
		}
	l = (q - 1) / 2 * N1_mm1;
	if (index < l) {
		ms_idx = index / N1_mm1;
		sub_index = index % N1_mm1;
		field1 = minus_nonsquares[ms_idx];
		if (f_v) {
			cout << "case 1 ms_idx=" << ms_idx
					<< " field1=" << field1
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 0;
		v[1 + 2 * (m - 1) + 0] = field1;
		v[1 + 2 * (m - 1) + 1] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		c = F->negate(field1);
		change_form_value(v + 1, 1, m - 1, c);
		return;
		}
	index -= l;
	l = (q - 1) / 2 * (q - 1) * N1_mm1;
	ll = (q - 1) * N1_mm1;
	if (index < l) {
		ms_idx = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		field1 = minus_nonsquares[ms_idx];
		if (f_v) {
			cout << "case 2 ms_idx=" << ms_idx
				<< " field1=" << field1 << " field2=" << field2
				<< " sub_sub_index=" << sub_sub_index << endl;
			}
		//cout << "ms_idx=" << ms_idx << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		a = 1 + field2;
		b = F->mult(F->inverse(a), field1);
		v[0] = 1;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		c = F->negate(F->add(1, field1));
		change_form_value(v + 1, 1, m - 1, c);
		return;
		}
	cout << "error in parabolic_odd_type2_index_to_point "
			"illegal index" << endl;
	exit(1);
}

void orthogonal::parabolic_point_rk_to_type_and_index(
		int rk, int &type, int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "parabolic_point_rk_to_type_and_index "
				"rk = " << rk << endl;
		}
	if (rk == pt_Q) {
		type = 6;
		index = 0;
		if (f_v) {
			cout << "parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
			}
		return;
		}
	if (rk == pt_P) {
		type = 7;
		index = 0;
		if (f_v) {
			cout << "parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
			}
		return;
		}
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	if (f_v) {
		cout << "parabolic_point_rk_to_type_and_index created vector ";
		int_vec_print(cout, v_tmp2, n);
		cout << endl;
		}
	if (v_tmp2[n - 2] == 0 && v_tmp2[n - 1]) {
		int field, sub_index, len;
		type = 3;
		len = alpha;
		parabolic_normalize_point_wrt_subspace(v_tmp2, 1);
		field = v_tmp2[n - 1];
		sub_index = subspace->rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "field=" << field << " sub_index=" << sub_index << endl;
			}
		index = (field - 1) * len + sub_index;
		if (f_v) {
			cout << "parabolic_point_rk_to_type_and_index rk = " << rk
					<< " type = " << type
					<< " index = " << index << endl;
			}
		return;
		}
	else if (v_tmp2[n - 2] && v_tmp2[n - 1] == 0) {
		int field, sub_index, len;
		type = 4;
		len = alpha;
		parabolic_normalize_point_wrt_subspace(v_tmp2, 1);
		field = v_tmp2[n - 2];
		sub_index = subspace->rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "field=" << field << " sub_index=" << sub_index << endl;
			}
		index = (field - 1) * len + sub_index;
		if (f_v) {
			cout << "parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
			}
		return;
		}
	else if (v_tmp2[n - 2] == 0 && v_tmp2[n - 1] == 0) {
		type = 5;
		index = subspace->rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
			}
		return;
		}
	if (f_even) {
		parabolic_even_point_rk_to_type_and_index(rk,
				type, index, verbose_level);
		}
	else {
		parabolic_odd_point_rk_to_type_and_index(rk,
				type, index, verbose_level);
		}
}

void orthogonal::parabolic_even_point_rk_to_type_and_index(
		int rk, int &type, int &index, int verbose_level)
{
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	parabolic_even_point_to_type_and_index(v_tmp2,
			type, index, verbose_level);
}

void orthogonal::parabolic_even_point_to_type_and_index(
		int *v, int &type, int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_start_with_one, value_middle, value_end, f_middle_is_zero;
	int a, b, /*c,*/ l, ll, lll, field1, field2, sub_index, sub_sub_index;
	
	if (f_v) {
		cout << "parabolic_even_point_to_type_and_index:";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[0] != 0 && v[0] != 1) {
		cout << "parabolic_even_point_to_type_and_index: "
				"error in unrank_point" << endl;
		exit(1);
		}
	parabolic_point_properties(v, 1, n, 
		f_start_with_one, value_middle, value_end, verbose_level);
	if (value_middle == 0) {
		f_middle_is_zero = is_zero_vector(v + 1, 1, n - 3);
		}
	else
		f_middle_is_zero = FALSE;
	if (f_v) {
		cout << "parabolic_even_point_to_type_and_index: "
				"f_start_with_one=" << f_start_with_one
				<< " value_middle=" << value_middle
				<< " f_middle_is_zero=" << f_middle_is_zero
				<< " value_end=" << value_end << endl;
		}
	if (f_start_with_one &&
			value_middle == 0 &&
			f_middle_is_zero &&
			value_end == 1) {
		type = 1;
		a = v[1 + 2 * (m - 1) + 0];
		b = v[1 + 2 * (m - 1) + 1];
		index = a - 1;
		if (f_v) {
			cout << "parabolic_even_point_to_type_and_index "
				"type = " << type << " index = " << index << endl;
			}
		return;
		}
	else if (value_end) {
		type = 2;
		index = 0;
		if (!f_start_with_one) {
			change_form_value(v + 1, 1, m - 1, F->inverse(value_middle));
			sub_index = rank_N1(v + 1, 1, m - 1);
			a = v[1 + 2 * (m - 1) + 0];
			b = v[1 + 2 * (m - 1) + 1];
			field1 = a - 1;
			index += field1 * N1_mm1 + sub_index;
			if (f_v) {
				cout << "parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
				}
			return;
			}
		index += (q - 1) * N1_mm1;
		ll = S_mm1 - 1;
		l = (q - 1) * ll;
		if (value_middle == 0) {
			a = v[1 + 2 * (m - 1) + 0];
			b = v[1 + 2 * (m - 1) + 1];
			field2 = last_non_zero_entry(v + 1, 1, n - 3);
			scalar_multiply_vector(v + 1, 1, n - 3, F->inverse(field2));
			sub_sub_index = rank_Sbar(v + 1, 1, m - 1);
			field2--;
			lll = Sbar_mm1;
			sub_index = field2 * lll + sub_sub_index;
			field1 = a - 1;
			index += field1 * ll + sub_index;
			if (f_v) {
				cout << "parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
				}
			return;
			}
		index += l;
		l = (q - 2) * (q - 1) * N1_mm1;
		change_form_value(v + 1, 1, m - 1, F->inverse(value_middle));
		sub_sub_index = rank_N1(v + 1, 1, m - 1);
		a = F->add(1, value_middle);
		b = v[1 + 2 * (m - 1) + 0];
		//c = v[1 + 2 * (m - 1) + 1];
		if (a == 0 || a == 1) {
			cout << "error in parabolic_even_point_to_type_and_index "
					"a == 0 || a == 1" << endl;
			exit(1);
			}
		if (b == 0) {
			cout << "error in parabolic_even_point_to_type_and_index "
					"b == 0" << endl;
			exit(1);
			}
		field2 = b - 1;
		field1 = a - 2;
		//cout << "field1=" << field1 << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		ll = (q - 1) * N1_mm1;
		index += field1 * ll + sub_index;
		if (f_v) {
			cout << "parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
		return;
		}
	else {
		cout << "error in parabolic_even_point_to_type_and_index, "
				"unknown type, type = " << type << endl;
		exit(1);
		}
}

void orthogonal::parabolic_odd_point_rk_to_type_and_index(
		int rk, int &type, int &index, int verbose_level)
{
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	parabolic_odd_point_to_type_and_index(v_tmp2,
			type, index, verbose_level);
}

void orthogonal::parabolic_odd_point_to_type_and_index(
		int *v, int &type, int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_start_with_one, value_middle, value_end;
	int f_middle_is_zero, f_end_value_is_minus_square;
	int a, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;
	
	if (f_v) {
		cout << "parabolic_odd_point_to_type_and_index:";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[0] != 0 && v[0] != 1) {
		cout << "parabolic_odd_point_to_type_and_index: "
				"error in unrank_point" << endl;
		exit(1);
		}
	parabolic_point_properties(v, 1, n, 
		f_start_with_one, value_middle, value_end, verbose_level);
	if (f_v) {
		cout << "f_start_with_one=" << f_start_with_one
				<< " value_middle=" << value_middle
				<< " value_end=" << value_end << endl;
		}
	if (value_middle == 0) {
		f_middle_is_zero = is_zero_vector(v + 1, 1, n - 3);
		}
	else {
		f_middle_is_zero = FALSE;
		}
	if (f_v) {
		cout << "f_middle_is_zero=" << f_middle_is_zero << endl;
		}
	f_end_value_is_minus_square = f_is_minus_square[value_end];
	if (f_v) {
		cout << "f_end_value_is_minus_square="
				<< f_end_value_is_minus_square << endl;
		}

	if (f_end_value_is_minus_square) {
		type = 1;
		index = 0;
		l = (q - 1) / 2 * N1_mm1;
		if (!f_start_with_one) {
			ms_idx = index_minus_square[value_end];
			if (ms_idx == -1) {
				cout << "parabolic_odd_point_to_type_and_index: "
						"ms_idx == -1" << endl;
				}
			c = F->negate(value_end);
			change_form_value(v + 1, 1, m - 1, F->inverse(c));
			sub_index = rank_N1(v + 1, 1, m - 1);
			index += ms_idx * N1_mm1 + sub_index;
			if (f_v) {
				cout << "parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
				}
			return;
			}
		index += l;
		l = (q - 1) * S_mm1;
		if (value_middle == 0) {
			if (f_middle_is_zero) {
				a = v[1 + 2 * (m - 1) + 0];
				field1 = a - 1;
				sub_index = 0;
				index += field1 * S_mm1 + sub_index;
				if (f_v) {
					cout << "parabolic_odd_point_to_type_and_index "
						"type = " << type << " index = " << index << endl;
					}
				return;
				}
			else {
				a = v[1 + 2 * (m - 1) + 0];
				//b = v[1 + 2 * (m - 1) + 1];
				field1 = a - 1;
				field2 = last_non_zero_entry(v + 1, 1, n - 3);
				scalar_multiply_vector(v + 1, 1, n - 3, F->inverse(field2));
				sub_sub_index = rank_Sbar(v + 1, 1, m - 1);
				field2--;
				//cout << "field1=" << field1 << " field2=" << field2
				//<< " sub_sub_index=" << sub_sub_index << endl;
				sub_index = field2 * Sbar_mm1 + sub_sub_index + 1;
				index += field1 * S_mm1 + sub_index;
				if (f_v) {
					cout << "parabolic_odd_point_to_type_and_index "
							"type = " << type << " index = " << index << endl;
					}
				return;
				}
			}
		index += l;
		l = ((q - 1) / 2 - 1) * (q - 1) * N1_mm1;
		ll = (q - 1) * N1_mm1;
		ms_idx = index_minus_square_without[value_end];
		if (ms_idx == -1) {
			cout << "parabolic_odd_point_to_type_and_index: "
					"ms_idx == -1" << endl;
			}
		field1 = minus_squares_without[ms_idx];
		c = F->negate(F->add(1, field1));
		change_form_value(v + 1, 1, m - 1, F->inverse(c));
		sub_sub_index = rank_N1(v + 1, 1, m - 1);
		a = v[1 + 2 * (m - 1) + 0];
		field2 = a - 1;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		index += ms_idx * ll + sub_index;
		if (f_v) {
			cout << "parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
		return;
		}
	else if (value_end) {
		type = 2;
		l = (q - 1) / 2 * N1_mm1;
		index = 0;
		if (!f_start_with_one) {
			ms_idx = index_minus_nonsquare[value_end];
			if (ms_idx == -1) {
				cout << "parabolic_odd_point_to_type_and_index: "
						"ms_idx == -1" << endl;
				}
			c = F->negate(value_end);
			change_form_value(v + 1, 1, m - 1, F->inverse(c));
			sub_index = rank_N1(v + 1, 1, m - 1);
			index += ms_idx * N1_mm1 + sub_index;
			if (f_v) {
				cout << "parabolic_odd_point_to_type_and_index "
						"type = " << type << " index = " << index << endl;
				}
			return;
			}
		index += l;
		l = (q - 1) / 2 * (q - 1) * N1_mm1;
		ll = (q - 1) * N1_mm1;
		ms_idx = index_minus_nonsquare[value_end];
		if (ms_idx == -1) {
			cout << "parabolic_odd_point_to_type_and_index: "
					"ms_idx == -1" << endl;
			}
		//field1 = minus_nonsquares[ms_idx];
		//c = F->negate(F->add(1, field1));
		change_form_value(v + 1, 1, m - 1, F->inverse(value_middle));
		sub_sub_index = rank_N1(v + 1, 1, m - 1);
		a = v[1 + 2 * (m - 1) + 0];
		field2 = a - 1;
		//cout << "ms_idx=" << ms_idx << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		index += ms_idx * ll + sub_index;
		if (f_v) {
			cout << "parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
		return;
		}
	cout << "error in parabolic_odd_point_to_type_and_index, "
			"unknown type, type = " << type << endl;
	exit(1);
}

//##############################################################################
// ranking / unranking neighbors of the favorite point:
//##############################################################################

void orthogonal::parabolic_neighbor51_odd_unrank(
		int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor51_odd_unrank "
				"index=" << index << endl;
		}
	subspace->parabolic_odd_type1_index_to_point(
			index, subspace->v_tmp2, verbose_level);
	v[0] = subspace->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < subspace->n; i++) {
		v[2 + i] = subspace->v_tmp2[i];
		}
	if (f_v) {
		cout << "parabolic_neighbor51_odd_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor51_odd_rank(
		int *v, int verbose_level)
{
	int i, type, index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor51_odd_rank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[2]) {
		cout << "parabolic_neighbor51_odd_rank v[2]" << endl;
		exit(1);
		}
	subspace->v_tmp2[0] = v[0];
	for (i = 1; i < subspace->n; i++) {
		subspace->v_tmp2[i] = v[2 + i];
		}
	subspace->normalize_point(subspace->v_tmp2, 1);
	if (f_v) {
		cout << "normalized and in subspace: ";
		int_vec_print(cout, subspace->v_tmp2, subspace->n);
		cout << endl;
		}
	subspace->parabolic_odd_point_to_type_and_index(
			subspace->v_tmp2, type, index, verbose_level);
	if (type != 1) {
		cout << "parabolic_neighbor51_odd_rank type != 1" << endl;
		exit(1);
		}
	return index;
}


void orthogonal::parabolic_neighbor52_odd_unrank(
		int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor52_odd_unrank index=" << index << endl;
		}
	subspace->parabolic_odd_type2_index_to_point(
			index, subspace->v_tmp2, verbose_level);
	v[0] = subspace->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < subspace->n; i++) {
		v[2 + i] = subspace->v_tmp2[i];
		}
	if (f_v) {
		cout << "parabolic_neighbor52_odd_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor52_odd_rank(int *v, int verbose_level)
{
	int i, type, index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor52_odd_rank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[2]) {
		cout << "parabolic_neighbor52_odd_rank v[2]" << endl;
		exit(1);
		}
	subspace->v_tmp2[0] = v[0];
	for (i = 1; i < subspace->n; i++) {
		subspace->v_tmp2[i] = v[2 + i];
		}
	subspace->normalize_point(subspace->v_tmp2, 1);
	subspace->parabolic_odd_point_to_type_and_index(
			subspace->v_tmp2, type, index, verbose_level);
	if (type != 2) {
		cout << "parabolic_neighbor52_odd_rank type != 2" << endl;
		exit(1);
		}
	return index;
}

void orthogonal::parabolic_neighbor52_even_unrank(
		int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor52_even_unrank index=" << index << endl;
		}
	subspace->parabolic_even_type2_index_to_point(index, subspace->v_tmp2);
	v[0] = subspace->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < subspace->n; i++) {
		v[2 + i] = subspace->v_tmp2[i];
		}
	if (f_v) {
		cout << "parabolic_neighbor52_even_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor52_even_rank(int *v, int verbose_level)
{
	int i, type, index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor52_even_rank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[2]) {
		cout << "parabolic_neighbor52_even_rank v[2]" << endl;
		exit(1);
		}
	subspace->v_tmp2[0] = v[0];
	for (i = 1; i < subspace->n; i++) {
		subspace->v_tmp2[i] = v[2 + i];
		}
	subspace->normalize_point(subspace->v_tmp2, 1);
	subspace->parabolic_even_point_to_type_and_index(
			subspace->v_tmp2, type, index, verbose_level);
	if (type != 2) {
		cout << "parabolic_neighbor52_even_rank type != 1" << endl;
		exit(1);
		}
	return index;
}

void orthogonal::parabolic_neighbor34_unrank(
		int index, int *v, int verbose_level)
{
	int len, sub_len, a, av, b, sub_index;
	int sub_sub_index, multiplyer;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor34_unrank "
				"index=" << index << endl;
		}
	len = S_mm2;
	if (index < len) {
		// case 1:
		if (f_v) {
			cout << "case 1 index=" << index << endl;
			}
		v[0] = 0;
		v[n - 2] = 1;
		v[n - 1] = 0;
		v[1] = 0;
		v[2] = F->negate(1);
		unrank_S(v + 3, 1, m - 2, index);
		goto finish;
		}
	index -= len;
	len = (q - 1) * N1_mm2;
	if (index < len) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 0;
		v[n - 2] = 1;
		v[n - 1] = 0;
		v[1] = a;
		v[2] = F->negate(1);
		unrank_N1(v + 3, 1, m - 2, sub_index);
		change_form_value(v + 3, 1, m - 2, a);
		goto finish;
		}
	index -= len;
	len = (q - 1) * N1_mm2;
	if (index < len) {
		// case 3:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 3 a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 1;
		v[1] = 0;
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		change_form_value(v + 3, 1, m - 2, F->negate(1));
		goto finish;
		}
	index -= len;
	len = (q - 1) * S_mm2;
	if (index < len) {
		// case 4:
		a = index / S_mm2;
		sub_index = index % S_mm2;
		a++;
		if (f_v) {
			cout << "case 4 a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 1;
		v[1] = F->inverse(a);
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_S(v + 3, 1, m - 2, sub_index);
		goto finish;
		}
	index -= len;
	len = (q - 1) * (q - 2) * N1_mm2;
	if (index < len) {
		// case 5:
		sub_len = (q - 2) * N1_mm2;
		a = index / sub_len;
		sub_index = index % sub_len;
		b = sub_index / N1_mm2;
		sub_sub_index = sub_index % N1_mm2;
		a++;
		av = F->inverse(a);
		b++;
		if (b >= av) {
			b++;
			}
		if (f_v) {
			cout << "case 5 a=" << a << " b=" << b
					<< " sub_sub_index=" << sub_sub_index << endl;
			}
		v[0] = 1;
		v[1] = b;
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_sub_index);
		multiplyer = F->add(F->negate(1), F->mult(a, b));
		if (f_v) {
			cout << "case 5 multiplyer=" << multiplyer << endl;
			}
		change_form_value(v + 3, 1, m - 2, multiplyer);
		goto finish;
		}
	cout << "parabolic_neighbor34_unrank index illegal" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor34_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor34_rank(int *v, int verbose_level)
{
	int len1, len2, len3, len4, /*len5,*/ av;
	int index, sub_len, a, b, sub_index, sub_sub_index, multiplyer;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor34_rank " << endl;
		int_vec_print(cout, v, n);
		cout << endl;
		}
	normalize_point(v, 1);
	if (v[n - 1]) {
		cout << "parabolic_neighbor34_rank v[n - 1]" << endl;
		exit(1);
		}
	if (v[n - 2] == 0) {
		cout << "parabolic_neighbor34_rank v[n - 2] == 0" << endl;
		exit(1);
		}

	len1 = S_mm2;
	len2 = (q - 1) * N1_mm2;
	len3 = (q - 1) * N1_mm2;
	len4 = (q - 1) * S_mm2;
	//len5 = (q - 1) * (q - 2) * N1_mm2;

	if (v[0] == 0) {
		if (v[2] != F->negate(1)) {
			cout << "parabolic_neighbor34_rank "
					"v[2] != F->negate(1)" << endl;
			exit(1);
			}
		a = v[1];
		if (a == 0) {
			// case 1:
			index = rank_S(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 1 index=" << index << endl;
				}
			goto finish;
			}
		else {
			// case 2:
			change_form_value(v + 3, 1, m - 2, F->inverse(a));
			sub_index = rank_N1(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 2 a=" << a
						<< " sub_index=" << sub_index << endl;
				}
			index = (a - 1) * N1_mm2 + sub_index;
			index += len1;
			goto finish;
			}
		}
	else {
		if (v[0] != 1) {
			cout << "parabolic_neighbor34_rank v[1] != 1" << endl;
			exit(1);
			}
		a = v[n - 2];
		if (v[2] != F->negate(a)) {
			cout << "parabolic_neighbor34_rank "
					"v[2] != F->negate(a)" << endl;
			exit(1);
			}
		if (v[1] == 0) {
			// case 3:
			change_form_value(v + 3, 1, m - 2, F->negate(1));
			sub_index = rank_N1(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 3 a=" << a
						<< " sub_index=" << sub_index << endl;
				}
			index = (a - 1) * N1_mm2 + sub_index;
			index += len1;
			index += len2;
			goto finish;
			}
		else {
			av = F->inverse(a);
			if (v[1] == av) {
				// case 4:
				sub_index = rank_S(v + 3, 1, m - 2);
				if (f_v) {
					cout << "case 4 a=" << a
							<< " sub_index=" << sub_index << endl;
					}
				index = (a - 1) * S_mm2 + sub_index;
				index += len1;
				index += len2;
				index += len3;
				goto finish;
				}
			else {
				// case 5:
				sub_len = (q - 2) * N1_mm2;
				b = v[1];
				if (b == av) {
					cout << "parabolic_neighbor34_rank b = av" << endl;
					exit(1);
					}
				multiplyer = F->add(F->negate(1), F->mult(a, b));
				if (f_v) {
					cout << "case 5 multiplyer=" << multiplyer << endl;
					}
				change_form_value(v + 3, 1, m - 2, F->inverse(multiplyer));
				sub_sub_index = rank_N1(v + 3, 1, m - 2);
				if (f_v) {
					cout << "case 5 a=" << a << " b=" << b
							<< " sub_sub_index=" << sub_sub_index << endl;
					}
				if (b >= av)
					b--;
				b--;
				sub_index = b * N1_mm2 + sub_sub_index;
				index = (a - 1) * sub_len + sub_index;
				index += len1;
				index += len2;
				index += len3;
				index += len4;
				goto finish;
				}
			}
		}
	cout << "parabolic_neighbor34_rank illegal point" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor34_rank index = " << index << endl;
		}
	return index;
}


void orthogonal::parabolic_neighbor53_unrank(
		int index, int *v, int verbose_level)
{
	int a, sub_index;
	int f_v = (verbose_level >= 1);
	int len1, len2;
	
	if (f_v) {
		cout << "parabolic_neighbor53_unrank index=" << index << endl;
		}
	len1 = (q - 1) * Sbar_mm2;
	len2 = (q - 1) * N1_mm2;
	if (index < len1) {
		// case 1:
		a = index / Sbar_mm2;
		sub_index = index % Sbar_mm2;
		a++;
		if (f_v) {
			cout << "case 1 index=" << index << " a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		
		v[0] = 0;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = 0;
		v[n - 1] = a;
		unrank_Sbar(v + 3, 1, m - 2, sub_index);
		goto finish;
		}
	index -= len1;
	if (index < len2) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 index=" << index << " a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = 0;
		v[n - 1] = a;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		change_form_value(v + 3, 1, m - 2, F->negate(1));
		goto finish;
		}
	cout << "parabolic_neighbor53_unrank index illegal" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor53_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor53_rank(int *v, int verbose_level)
{
	int len1; //, len2;
	int index, a, sub_index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor53_rank " << endl;
		int_vec_print(cout, v, n);
		cout << endl;
		}
	parabolic_normalize_point_wrt_subspace(v, 1);
	if (v[n - 2]) {
		cout << "parabolic_neighbor53_rank v[n - 2]" << endl;
		exit(1);
		}
	if (v[n - 1] == 0) {
		cout << "parabolic_neighbor53_rank v[n - 1] == 0" << endl;
		exit(1);
		}
	a = v[n - 1];

	len1 = (q - 1) * Sbar_mm2;
	//len2 = (q - 1) * N1_mm2;

	if (v[0] == 0) {
		// case 1
		sub_index = rank_Sbar(v + 3, 1, m - 2);
		index = (a - 1) * Sbar_mm2 + sub_index;
		goto finish;
		}
	else {
		// case 2
		change_form_value(v + 3, 1, m - 2, F->negate(1));
		sub_index = rank_N1(v + 3, 1, m - 2);
		index = len1 + (a - 1) * N1_mm2 + sub_index;
		goto finish;
		}

	cout << "parabolic_neighbor53_rank illegal point" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor53_rank index = " << index << endl;
		}
	return index;
}

void orthogonal::parabolic_neighbor54_unrank(
		int index, int *v, int verbose_level)
{
	int a, sub_index;
	int f_v = (verbose_level >= 1);
	int len1, len2;
	
	if (f_v) {
		cout << "parabolic_neighbor54_unrank index=" << index << endl;
		}
	len1 = (q - 1) * Sbar_mm2;
	len2 = (q - 1) * N1_mm2;
	if (index < len1) {
		// case 1:
		a = index / Sbar_mm2;
		sub_index = index % Sbar_mm2;
		a++;
		if (f_v) {
			cout << "case 1 index=" << index
					<< " a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		
		v[0] = 0;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_Sbar(v + 3, 1, m - 2, sub_index);
		goto finish;
		}
	index -= len1;
	if (index < len2) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 index=" << index
					<< " a=" << a
					<< " sub_index=" << sub_index << endl;
			}
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		change_form_value(v + 3, 1, m - 2, F->negate(1));
		goto finish;
		}
	cout << "parabolic_neighbor54_unrank index illegal" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor54_unrank ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

int orthogonal::parabolic_neighbor54_rank(int *v, int verbose_level)
{
	int len1; //, len2;
	int index, a, sub_index;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_neighbor54_rank " << endl;
		int_vec_print(cout, v, n);
		cout << endl;
		}
	parabolic_normalize_point_wrt_subspace(v, 1);
	if (f_v) {
		cout << "normalized wrt subspace " << endl;
		int_vec_print(cout, v, n);
		cout << endl;
		}
	if (v[n - 1]) {
		cout << "parabolic_neighbor54_rank v[n - 2]" << endl;
		exit(1);
		}
	if (v[n - 2] == 0) {
		cout << "parabolic_neighbor54_rank v[n - 1] == 0" << endl;
		exit(1);
		}
	a = v[n - 2];

	len1 = (q - 1) * Sbar_mm2;
	//len2 = (q - 1) * N1_mm2;

	if (v[0] == 0) {
		// case 1
		sub_index = rank_Sbar(v + 3, 1, m - 2);
		index = (a - 1) * Sbar_mm2 + sub_index;
		goto finish;
		}
	else {
		// case 2
		change_form_value(v + 3, 1, m - 2, F->negate(1));
		sub_index = rank_N1(v + 3, 1, m - 2);
		index = len1 + (a - 1) * N1_mm2 + sub_index;
		goto finish;
		}

	cout << "parabolic_neighbor54_rank illegal point" << endl;
	exit(1);
	
finish:
	if (f_v) {
		cout << "parabolic_neighbor54_rank index = " << index << endl;
		}
	return index;
}


//##############################################################################
// ranking / unranking lines:
//##############################################################################

void orthogonal::parabolic_unrank_line(
		int &p1, int &p2, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_unrank_line rk=" << rk << endl;
		}
	if (m == 0) {
		cout << "orthogonal::parabolic_unrank_line "
				"Witt index zero, there is no line to unrank" << endl;
		exit(1);
		}
	if (rk < l1) {
		if (f_even)
			parabolic_unrank_line_L1_even(p1, p2, rk, verbose_level);
		else
			parabolic_unrank_line_L1_odd(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l1;
	if (f_v) {
		cout << "reducing rk to " << rk << " l2=" << l2 << endl;
		}
	if (rk < l2) {
		if (f_even)
			parabolic_unrank_line_L2_even(p1, p2, rk, verbose_level);
		else
			parabolic_unrank_line_L2_odd(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l2;
	if (f_v) {
		cout << "reducing rk to " << rk << " l3=" << l3 << endl;
		}
	if (rk < l3) {
		parabolic_unrank_line_L3(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l3;
	if (rk < l4) {
		parabolic_unrank_line_L4(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l4;
	if (rk < l5) {
		parabolic_unrank_line_L5(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l5;
	if (rk < l6) {
		parabolic_unrank_line_L6(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l6;
	if (rk < l7) {
		parabolic_unrank_line_L7(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l7;
	if (rk < l8) {
		parabolic_unrank_line_L8(p1, p2, rk, verbose_level);
		return;
		}
	rk -= l8;
	cout << "error in orthogonal::parabolic_unrank_line, "
			"rk too big" << endl;
	exit(1);
}

int orthogonal::parabolic_rank_line(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int p1_type, p2_type, p1_index, p2_index, type, cp1, cp2;
	
	if (f_v) {
		cout << "parabolic_rank_line "
				"p1=" << p1 << " p2=" << p2 << endl;
		}
	point_rk_to_type_and_index(p1,
			p1_type, p1_index, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line "
				"p1_type=" << p1_type
				<< " p1_index=" << p1_index << endl;
		}
	point_rk_to_type_and_index(p2,
			p2_type, p2_index, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line "
				"p2_type=" << p2_type
				<< " p2_index=" << p2_index << endl;
		}
	type = parabolic_line_type_given_point_types(
			p1, p2, p1_type, p2_type, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line "
				"line type = " << type << endl;
		}
	parabolic_canonical_points_of_line(type,
			p1, p2, cp1, cp2, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line "
				"cp1=" << cp1 << " cp2=" << cp2 << endl;
		}

	if (type == 1) {
		if (f_even)
			return parabolic_rank_line_L1_even(cp1, cp2, verbose_level);
		else
			return parabolic_rank_line_L1_odd(cp1, cp2, verbose_level);
		}
	else if (type == 2) {
		if (f_even)
			return l1 +
					parabolic_rank_line_L2_even(cp1, cp2, verbose_level);
		else
			return l1 +
					parabolic_rank_line_L2_odd(cp1, cp2, verbose_level);
		}
	else if (type == 3) {
		return l1 + l2 +
				parabolic_rank_line_L3(cp1, cp2, verbose_level);
		}
	else if (type == 4) {
		return l1 + l2 + l3 +
				parabolic_rank_line_L4(cp1, cp2, verbose_level);
		}
	else if (type == 5) {
		return l1 + l2 + l3 + l4 +
				parabolic_rank_line_L5(cp1, cp2, verbose_level);
		}
	else if (type == 6) {
		return l1 + l2 + l3 + l4 + l5 +
				parabolic_rank_line_L6(cp1, cp2, verbose_level);
		}
	else if (type == 7) {
		return l1 + l2 + l3 + l4 + l5 + l6 +
				parabolic_rank_line_L7(cp1, cp2, verbose_level);
		}
	else if (type == 8) {
		return l1 + l2 + l3 + l4 + l5 + l6 + l7 +
				parabolic_rank_line_L8(cp1, cp2, verbose_level);
		}
	else {
		cout << "parabolic_rank_line type nyi" << endl;
		exit(1);
		}
}

void orthogonal::parabolic_unrank_line_L1_even(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, sub_idx;
	
	if (index >= l1) {
		cout << "error in parabolic_unrank_line_L1_even "
				"index too large" << endl;
		}
	idx = index / (q - 1);
	sub_idx = index % (q - 1);
	if (f_v) {
		cout << "parabolic_unrank_line_L1_even "
				"index=" << index << " idx=" << idx
				<< " sub_idx=" << sub_idx << endl;
		}
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	if (f_vv) {
		cout << "p1=" << p1 << endl;
		}
	p2 = type_and_index_to_point_rk(1, sub_idx, verbose_level);
	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "parabolic_unrank_line_L1_even "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L1_even(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, type, idx, sub_idx;

	if (f_v) {
		cout << "parabolic_unrank_line_L1_even "
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "parabolic_rank_line_L1_even p1 must be in P5" << endl;
		exit(1);
		}
	point_rk_to_type_and_index(p2, type, sub_idx, verbose_level);
	if (type != 1) {
		cout << "parabolic_rank_line_L1_even p2 must be in P1" << endl;
		exit(1);
		}
	index = idx * (q - 1) + sub_idx;
	return index;
}

void orthogonal::parabolic_unrank_line_L1_odd(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L1_odd "
				"index=" << index << " l1=" << l1
				<< " a51=" << a51 << endl;
		}
	if (index >= l1) {
		cout << "error in parabolic_unrank_line_L1_odd "
				"index too large" << endl;
		exit(1);
		}
	idx = index / a51;
	index2 = index % a51;

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	if (f_v) {
		cout << "rk1=" << rk1 << endl;
		}
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	if (f_v) {
		cout << "p1=" << p1 << endl;
		}

	parabolic_neighbor51_odd_unrank(index2, v3, verbose_level);
	
	Siegel_move_forward_by_index(rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);

	if (f_v) {
		cout << "parabolic_unrank_line_L1_odd "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L1_odd(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, type, idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_rank_line_L1_odd "
				"p1=" << p1 << " p2=" << p2 << endl;
		}

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line_L1_odd "
				"type=" << type << " idx=" << idx << endl;
		}
	if (type != 5) {
		cout << "parabolic_rank_line_L1_odd "
				"point 1 must be of type 5" << endl;
		exit(1);
		}
	unrank_point(v4, 1, p2, verbose_level - 1);
	Siegel_move_backward_by_index(rk1, p1, v4, v3, verbose_level);
	
	index2 = parabolic_neighbor51_odd_rank(v3, verbose_level);	
		
	if (f_v) {
		cout << "parabolic_rank_line_L1_odd "
				"idx=" << idx << " index2=" << index2 << endl;
		}
	
	index = idx * a51 + index2;

	if (f_v) {
		cout << "parabolic_unrank_line_L1_odd index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L2_even(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L2_even index=" << index << endl;
		}
	if (index >= l2) {
		cout << "error in parabolic_unrank_line_L2_even "
				"index too large" << endl;
		exit(1);
		}
	idx = index / a52a;
	index2 = index % a52a;

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	parabolic_neighbor52_even_unrank(index2, v3, FALSE);
	
	Siegel_move_forward_by_index(rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);
	

	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "parabolic_unrank_line_L2_even "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
		}
}

void orthogonal::parabolic_unrank_line_L2_odd(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L2_odd "
				"index=" << index << endl;
		}
	if (index >= l2) {
		cout << "error in parabolic_unrank_line_L2_odd "
				"index too large" << endl;
		exit(1);
		}
	idx = index / a52a;
	index2 = index % a52a;
	if (f_v) {
		cout << "parabolic_unrank_line_L2_odd "
				"idx=" << idx << " index2=" << index2 << endl;
		}

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	parabolic_neighbor52_odd_unrank(index2, v3, FALSE);
	
	Siegel_move_forward_by_index(rk1, p1, v3, v4, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_forward_by_index";
		int_vec_print(cout, v4, n);
		cout << endl;
		}
	p2 = rank_point(v4, 1, verbose_level - 1);
	

	if (f_vv) {
		cout << "p2=" << p2 << endl;
		}
	if (f_v) {
		cout << "parabolic_unrank_line_L2_odd "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L2_even(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, type, idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_rank_line_L2_even "
				"p1=" << p1 << " p2=" << p2 << endl;
		}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line_L2_even "
				"type=" << type << " idx=" << idx << endl;
		}
	if (type != 5) {
		cout << "parabolic_rank_line_L2_even "
				"point 1 must be of type 5" << endl;
		exit(1);
		}
	unrank_point(v4, 1, p2, verbose_level - 1);
	Siegel_move_backward_by_index(rk1, p1, v4, v3, verbose_level);
	
	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	index2 = parabolic_neighbor52_even_rank(v3, verbose_level);	
		
	if (f_v) {
		cout << "parabolic_rank_line_L2_even idx=" << idx
				<< " index2=" << index2 << endl;
		}
	
	index = idx * a52a + index2;

	if (f_v) {
		cout << "parabolic_rank_line_L2_even index=" << index << endl;
		}
	return index;
}

int orthogonal::parabolic_rank_line_L2_odd(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, type, idx, index2, rk1;
	
	if (f_v) {
		cout << "parabolic_rank_line_L2_odd "
				"p1=" << p1 << " p2=" << p2 << endl;
		}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (f_v) {
		cout << "parabolic_rank_line_L2_odd type=" << type
				<< " idx=" << idx << endl;
		}
	if (type != 5) {
		cout << "parabolic_rank_line_L2_odd "
				"point 1 must be of type 5" << endl;
		exit(1);
		}
	unrank_point(v4, 1, p2, verbose_level - 1);
	Siegel_move_backward_by_index(rk1, p1, v4, v3, verbose_level);
	
	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	index2 = parabolic_neighbor52_odd_rank(v3, verbose_level);	
		
	if (f_v) {
		cout << "parabolic_rank_line_L2_odd idx=" << idx
				<< " index2=" << index2 << endl;
		}
	
	index = idx * a52a + index2;

	if (f_v) {
		cout << "parabolic_unrank_line_L2_odd index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L3(
		int &p1, int &p2, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, index2, idx2, field, rk1, rk2, a, b, c, multiplyer, i;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L3 index=" << index << endl;
		}
	if (index >= l3) {
		cout << "error in parabolic_unrank_line_L3 "
				"index too large" << endl;
		exit(1);
		}
	idx = index / a32b;
	index2 = index % a32b;
	idx2 = idx / (q - 1);
	field = idx % (q - 1);
	field++;
	if (f_v) {
		cout << "parabolic_unrank_line_L3 idx=" << idx
				<< " index2=" << index2 << " idx2=" << idx2
				<< " field=" << field << endl;
		}

	rk1 = type_and_index_to_point_rk(3, 0, verbose_level);
	rk2 = type_and_index_to_point_rk(5, idx2, verbose_level);
	if (f_v) {
		cout << "parabolic_unrank_line_L3 rk1=" << rk1
				<< " rk2=" << rk2 << " idx2=" << idx2
				<< " field=" << field << endl;
		}
	unrank_point(v1, 1, rk1, verbose_level - 1);
	unrank_point(v2, 1, rk2, verbose_level - 1);
	v2[n - 1] = 1;
	
	
	if (f_v) {
		int_vec_print(cout, v1, n); cout << endl;
		int_vec_print(cout, v2, n); cout << endl;
		}

	parabolic_neighbor34_unrank(index2, v3, verbose_level);
	
	Siegel_move_forward(v1, v2, v3, v4, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_forward" << endl;
		int_vec_print(cout, v3, n); cout << endl;
		int_vec_print(cout, v4, n); cout << endl;
		}
	a = subspace->evaluate_bilinear_form(v1, v3, 1);
	b = subspace->evaluate_bilinear_form(v2, v4, 1);
	if (f_v) {
		cout << "a=" << a << " b=" << b << endl;
		}
	if (a != b) {
		if (a == 0) {
			cout << "a != b but a = 0" << endl;
			exit(1);
			}
		if (b == 0) {
			cout << "a != b but b = 0" << endl;
			exit(1);
			}
		multiplyer = F->mult(a, F->inverse(b));
		if (f_v) {
			cout << "multiplyer=" << multiplyer << endl;
			}
		for (i = 0; i < n - 2; i++) {
			v4[i] = F->mult(v4[i], multiplyer);
			}
		if (f_v) {
			cout << "after scaling" << endl;
			int_vec_print(cout, v4, n); cout << endl;
			}
		c = subspace->evaluate_bilinear_form(v2, v4, 1);
		if (f_v) {
			cout << "c=" << c << endl;
			}
		if (c != a) {
			cout << "c != a" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "now changing the last components:" << endl;
		}
	
	v2[n - 2] = 0;
	v2[n - 1] = field;
	normalize_point(v2, 1);
	p1 = rank_point(v2, 1, verbose_level - 1);
	v4[n - 2] = F->mult(v4[n - 2], F->inverse(field));
	p2 = rank_point(v4, 1, verbose_level - 1);
	
	

	if (f_v) {
		cout << "parabolic_unrank_line_L3 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L3(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, idx, index2, idx2, field;
	int rk1, rk2, type, a, b, c, i, multiplyer;
	
	if (f_v) {
		cout << "parabolic_rank_line_L3 "
				"p1=" << p1 << " p2=" << p2 << endl;
		}


	rk1 = type_and_index_to_point_rk(3, 0, verbose_level);

	unrank_point(v1, 1, rk1, verbose_level - 1);
	unrank_point(v2, 1, p1, verbose_level - 1);
	if (f_v) {
		int_vec_print(cout, v1, n); cout << endl;
		int_vec_print(cout, v2, n); cout << endl;
		}
		
	parabolic_normalize_point_wrt_subspace(v2, 1);
	if (f_v) {
		cout << "after parabolic_normalize_point_wrt_subspace ";
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	field = v2[n - 1];
	if (f_v) {
		cout << "field=" << field << endl;
		}
	v2[n - 1] = 0;
	rk2 = rank_point(v2, 1, verbose_level - 1);
	parabolic_point_rk_to_type_and_index(rk2,
			type, idx2, verbose_level);
	if (f_v) {
		cout << "parabolic_unrank_line_L3 "
				"rk1=" << rk1 << " rk2=" << rk2
				<< " idx2=" << idx2 << " field=" << field << endl;
		}
	if (type != 5) {
		cout << "parabolic_rank_line_L3  type != 5" << endl;
		exit(1);
		}
	v2[n - 1] = 1;


	unrank_point(v4, 1, p2, verbose_level - 1);
	v4[n - 2] = F->mult(v4[n - 2], field);

	idx = idx2 * (q - 1) + (field - 1);
	
	Siegel_move_backward(v1, v2, v4, v3, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_backward" << endl;
		int_vec_print(cout, v3, n); cout << endl;
		int_vec_print(cout, v4, n); cout << endl;
		}
	a = subspace->evaluate_bilinear_form(v1, v3, 1);
	b = subspace->evaluate_bilinear_form(v2, v4, 1);
	if (f_v) {
		cout << "a=" << a << " b=" << b << endl;
		}
	if (a != b) {
		if (a == 0) {
			cout << "a != b but a = 0" << endl;
			exit(1);
			}
		if (b == 0) {
			cout << "a != b but b = 0" << endl;
			exit(1);
			}
		multiplyer = F->mult(b, F->inverse(a));
		if (f_v) {
			cout << "multiplyer=" << multiplyer << endl;
			}
		for (i = 0; i < n - 2; i++) {
			v3[i] = F->mult(v3[i], multiplyer);
			}
		if (f_v) {
			cout << "after scaling" << endl;
			int_vec_print(cout, v3, n); cout << endl;
			}
		c = subspace->evaluate_bilinear_form(v1, v3, 1);
		if (f_v) {
			cout << "c=" << c << endl;
			}
		if (c != b) {
			cout << "c != a" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "after scaling" << endl;
		int_vec_print(cout, v3, n); cout << endl;
		int_vec_print(cout, v4, n); cout << endl;
		}

	index2 = parabolic_neighbor34_rank(v3, verbose_level);	
		
	if (f_v) {
		cout << "parabolic_rank_line_L3 idx=" << idx
				<< " index2=" << index2 << " idx2=" << idx2
				<< " field=" << field << endl;
		}
	
	index = idx * a32b + index2;

	if (f_v) {
		cout << "parabolic_unrank_line_L3 index=" << index << endl;
		}

	return index;
}

void orthogonal::parabolic_unrank_line_L4(
		int &p1, int &p2, int index, int verbose_level)
// from P5 to P3
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, neighbor_idx, rk1;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L4 index=" << index << endl;
		}
	if (index >= l4) {
		cout << "error in parabolic_unrank_line_L4 index too large" << endl;
		exit(1);
		}
	idx = index / a53;
	neighbor_idx = index % a53;
	if (f_v) {
		cout << "parabolic_unrank_line_L4 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
		}

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	parabolic_neighbor53_unrank(neighbor_idx, v3, verbose_level);
	
	Siegel_move_forward_by_index(rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);
	
	if (f_v) {
		unrank_point(v5, 1, p1, verbose_level - 1);
		cout << "p1=" << p1 << " ";
		int_vec_print(cout, v5, n);
		cout << endl;
		cout << "p2=" << p2 << " ";
		int_vec_print(cout, v4, n);
		cout << endl;
		}
	if (f_v) {
		cout << "parabolic_unrank_line_L4 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L4(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, idx, neighbor_idx, rk1, type;
	
	if (f_v) {
		cout << "parabolic_rank_line_L4 "
				"p1=" << p1 << " p2=" << p2 << endl;
		}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "parabolic_rank_line_L4 type != 5" << endl;
		exit(1);
		}
	
	unrank_point(v4, 1, p2, verbose_level - 1);
	Siegel_move_backward_by_index(rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	neighbor_idx = parabolic_neighbor53_rank(v3, verbose_level);
		
	if (f_v) {
		cout << "parabolic_rank_line_L4 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
		}

	index = idx * a53 + neighbor_idx;
	
	if (f_v) {
		cout << "parabolic_rank_line_L4 index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L5(
	int &p1, int &p2, int index, int verbose_level)
// from P5 to P4
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int idx, neighbor_idx, rk1;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L5 index=" << index << endl;
		}
	if (index >= l5) {
		cout << "error in parabolic_unrank_line_L5 index too large" << endl;
		exit(1);
		}
	idx = index / a54;
	neighbor_idx = index % a54;
	if (f_v) {
		cout << "parabolic_unrank_line_L5 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
		}

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	parabolic_neighbor54_unrank(neighbor_idx, v3, verbose_level);
	
	Siegel_move_forward_by_index(rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);
	
	if (f_v) {
		unrank_point(v5, 1, p1, verbose_level - 1);
		cout << "p1=" << p1 << " ";
		int_vec_print(cout, v5, n);
		cout << endl;
		cout << "p2=" << p2 << " ";
		int_vec_print(cout, v4, n);
		cout << endl;
		}
	if (f_v) {
		cout << "parabolic_unrank_line_L5 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L5(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int index, idx, neighbor_idx, rk1, type;
	
	if (f_v) {
		cout << "parabolic_rank_line_L5 p1=" << p1 << " p2=" << p2 << endl;
		}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	
	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "parabolic_rank_line_L5 type != 5" << endl;
		exit(1);
		}
	
	unrank_point(v4, 1, p2, verbose_level - 1);
	Siegel_move_backward_by_index(rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		int_vec_print(cout, v3, n);
		cout << endl;
		}
	neighbor_idx = parabolic_neighbor54_rank(v3, verbose_level);
		
	if (f_v) {
		cout << "parabolic_rank_line_L5 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
		}

	index = idx * a54 + neighbor_idx;
	
	if (f_v) {
		cout << "parabolic_rank_line_L5 index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L6(
		int &p1, int &p2, int index, int verbose_level)
// within P5
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int pt1, pt2;
	
	if (f_v) {
		cout << "parabolic_unrank_line_L6 index=" << index << endl;
		}
	if (index >= l6) {
		cout << "error in parabolic_unrank_line_L6 "
				"index too large" << endl;
		exit(1);
		}
	subspace->parabolic_unrank_line(pt1, pt2, index, verbose_level);
	subspace->unrank_point(v1, 1, pt1, verbose_level - 1);
	subspace->unrank_point(v2, 1, pt2, verbose_level - 1);
	v1[n - 2] = 0;
	v1[n - 1] = 0;
	v2[n - 2] = 0;
	v2[n - 1] = 0;
	p1 = rank_point(v1, 1, verbose_level - 1);
	p2 = rank_point(v2, 1, verbose_level - 1);

	if (f_v) {
		cout << "parabolic_unrank_line_L6 "
				"index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L6(
		int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int pt1, pt2;
	int index;
	
	if (f_v) {
		cout << "parabolic_rank_line_L6 p1=" << p1 << " p2=" << p2 << endl;
		}
	unrank_point(v1, 1, p1, verbose_level - 1);
	unrank_point(v2, 1, p2, verbose_level - 1);
	if (v1[n - 2] || v1[n - 1] || v2[n - 2] || v2[n - 1]) {
		cout << "parabolic_rank_line_L6 points not in subspace" << endl;
		exit(1);
		}
	pt1 = subspace->rank_point(v1, 1, verbose_level - 1);
	pt2 = subspace->rank_point(v2, 1, verbose_level - 1);
	index = subspace->parabolic_rank_line(pt1, pt2, verbose_level);
	
	if (f_v) {
		cout << "parabolic_rank_line_L6 index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L7(
		int &p1, int &p2, int index, int verbose_level)
// from P6 = {Q}  to P5 via P3
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "parabolic_unrank_line_L7 index=" << index << endl;
		}
	if (index >= l7) {
		cout << "error in parabolic_unrank_line_L7 "
				"index too large" << endl;
		exit(1);
		}
	p1 = pt_Q;
	p2 = type_and_index_to_point_rk(5, index, verbose_level);

	if (f_v) {
		cout << "parabolic_unrank_line_L7 "
				"index=" << index << " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L7(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int type, index;
	
	if (f_v) {
		cout << "parabolic_rank_line_L7 p1=" << p1 << " p2=" << p2 << endl;
		}
	if (p1 != pt_Q) {
		cout << "parabolic_rank_line_L7 p1 != pt_Q" << endl;
		exit(1);
		}
	point_rk_to_type_and_index(p2, type, index, verbose_level);
	if (type != 5) {
		cout << "parabolic_rank_line_L7 type != 5" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "parabolic_rank_line_L7 index=" << index << endl;
		}
	return index;
}

void orthogonal::parabolic_unrank_line_L8(
		int &p1, int &p2, int index, int verbose_level)
// from P7 = {P}  to P5 via P4
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "parabolic_unrank_line_L8 index=" << index << endl;
		}
	if (index >= l8) {
		cout << "error in parabolic_unrank_line_L8 "
				"index too large" << endl;
		exit(1);
		}
	p1 = pt_P;
	p2 = type_and_index_to_point_rk(5, index, verbose_level);

	if (f_v) {
		cout << "parabolic_unrank_line_L8 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
		}
}

int orthogonal::parabolic_rank_line_L8(int p1, int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int type, index;
	
	if (f_v) {
		cout << "parabolic_rank_line_L8 p1=" << p1 << " p2=" << p2 << endl;
		}
	if (p1 != pt_P) {
		cout << "parabolic_rank_line_L8 p1 != pt_P" << endl;
		exit(1);
		}
	point_rk_to_type_and_index(p2, type, index, verbose_level);
	if (type != 5) {
		cout << "parabolic_rank_line_L8 type != 5" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "parabolic_rank_line_L8 index=" << index << endl;
		}
	return index;
}

int orthogonal::parabolic_line_type_given_point_types(int pt1, int pt2, 
	int pt1_type, int pt2_type, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	
	if (f_v) {
		cout << "parabolic_line_type_given_point_types "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
		}
	if (pt1_type > pt2_type) {
		return parabolic_line_type_given_point_types(
				pt2, pt1, pt2_type, pt1_type, verbose_level);
		}
	
	// from now on, we assume pt1_type <= pt2_type
	
	if (pt1_type == 1) {
		if (f_even) {
			return 1;
			}
		else {
			if (pt2_type == 1) {
				return parabolic_decide_P11_odd(pt1, pt2);
				}
			else if (pt2_type == 2) {
				return 3;
				}
			else if (pt2_type == 3) {
				return 3;
				}
			else if (pt2_type == 4) {
				return 3;
				}
			else if (pt2_type == 5) {
				return 1;
				}
			}
		}
	else if (pt1_type == 2) {
		if (f_even) {
			if (pt2_type == 2) {
				return parabolic_decide_P22_even(pt1, pt2);
				}
			else if (pt2_type == 3) {
				return 3;
				}
			else if (pt2_type == 4) {
				return 3;
				}
			else if (pt2_type == 5) {
				return parabolic_decide_P22_even(pt1, pt2);
				}
			}
		else {
			if (pt2_type == 2) {
				return parabolic_decide_P22_odd(pt1, pt2);
				}
			else if (pt2_type == 3) {
				return 3;
				}
			else if (pt2_type == 4) {
				return 3;
				}
			else if (pt2_type == 5) {
				return 2;
				}
			}
		}
	else if (pt1_type == 3) {
		if (f_even) {
			if (pt2_type == 3) {
				return parabolic_decide_P33(pt1, pt2);
				}
			else if (pt2_type == 4) {
				return 3;
				}
			else if (pt2_type == 5) {
				return parabolic_decide_P35(pt1, pt2);
				}
			else if (pt2_type == 6) {
				return 7;
				}
			}
		else {
			if (pt2_type == 3) {
				return parabolic_decide_P33(pt1, pt2);
				}
			else if (pt2_type == 4) {
				return 3;
				}
			else if (pt2_type == 5) {
				return parabolic_decide_P35(pt1, pt2);
				}
			else if (pt2_type == 6) {
				return 7;
				}
			}
		}
	else if (pt1_type == 4) {
		if (f_even) {
			if (pt2_type == 4) {
				return parabolic_decide_P44(pt1, pt2);
				}
			else if (pt2_type == 5) {
				return parabolic_decide_P45(pt1, pt2);
				}
			else if (pt2_type == 7) {
				return 8;
				}
			}
		else {
			if (pt2_type == 4) {
				return parabolic_decide_P44(pt1, pt2);
				}
			else if (pt2_type == 5) {
				return parabolic_decide_P45(pt1, pt2);
				}
			else if (pt2_type == 7) {
				return 8;
				}
			}
		}
	else if (pt1_type == 5) {
		if (pt2_type == 5) {
			return 6;
			}
		else if (pt2_type == 6) {
			return 7;
			}
		else if (pt2_type == 7) {
			return 8;
			}
		}
	cout << "orthogonal::parabolic_line_type_given_point_types "
			"illegal combination" << endl;
	cout << "pt1_type = " << pt1_type << endl;
	cout << "pt2_type = " << pt2_type << endl;
	exit(1);
}

int orthogonal::parabolic_decide_P11_odd(int pt1, int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P11_odd" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;
	
	if (is_ending_dependent(v1, v2)) {
		return 1;
		}
	else {
		return 3;
		}
}

int orthogonal::parabolic_decide_P22_even(int pt1, int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P22_even" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;
	
	
	if (is_ending_dependent(v1, v2)) {
		//cout << "ending is dependent, i.e. 1 or 2" << endl;
		if (parabolic_is_middle_dependent(v1, v2)) {
			return 1;
			}
		else {
			return 2;
			}
		}
	else {
		//cout << "ending is not dependent, hence 3" << endl;
		return 3;
		}
}

int orthogonal::parabolic_decide_P22_odd(int pt1, int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	if (is_ending_dependent(v1, v2)) {
		return 2;
		}
	else {
		return 3;
		}
}

int orthogonal::parabolic_decide_P33(int pt1, int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P33" << endl;
	if (is_ending_dependent(v1, v2)) {
		//cout << "ending is dependent" << endl;
		if (triple_is_collinear(pt1, pt2, pt_Q)) {
			return 7;
			}
		else {
			return 4;
			}
		}
	else {
		cout << "parabolic_decide_P33 ending is not dependent" << endl;
		exit(1);
		}
}

int orthogonal::parabolic_decide_P35(int pt1, int pt2)
{
	//cout << "parabolic_decide_P35 pt1 = " << pt1
	//<< " pt2=" << pt2 << endl;
	//unrank_point(v1, 1, pt1, verbose_level - 1);
	//unrank_point(v2, 1, pt2, verbose_level - 1);
	if (triple_is_collinear(pt1, pt2, pt_Q)) {
		return 7;
		}
	else {
		return 4;
		}
}

int orthogonal::parabolic_decide_P45(int pt1, int pt2)
{
	//unrank_point(v1, 1, pt1, verbose_level - 1);
	//unrank_point(v2, 1, pt2, verbose_level - 1);
	if (triple_is_collinear(pt1, pt2, pt_P)) {
		return 8;
		}
	else {
		return 5;
		}
}

int orthogonal::parabolic_decide_P44(int pt1, int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	if (is_ending_dependent(v1, v2)) {
		if (triple_is_collinear(pt1, pt2, pt_P)) {
			return 8;
			}
		else {
			return 5;
			}
		}
	else {
		cout << "parabolic_decide_P44 ending is not dependent" << endl;
		exit(1);
		}
}

void orthogonal::find_root_parabolic_xyz(
	int rk2, int *x, int *y, int *z, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int i;

	for (i = 0; i < n; i++) {
		x[i] = 0;
		z[i] = 0;
		}
	x[1] = 1;
	
	if (f_v) {
		cout << "find_root_parabolic_xyz rk2=" << rk2 << endl;
		}
	unrank_point(y, 1, rk2, verbose_level - 1);
	if (f_v) {
		int_vec_print(cout, y, n);
		cout << endl;
		}
	if (y[1]) {
		z[2] = 1;
		return;
		}
	if (y[2] && y[0] == 0) {
		z[0] = 1;
		z[1] = 1;
		z[2] = F->negate(1);
		return;
		}
	if (n == 3) {
		cout << "find_root_parabolic_xyz n == 3, "
				"we should not be in this case" << endl;
		exit(1);
		}
	// now y[2] = 0 or y = (*0*..) and
	// m > 1 and y_i \neq 0 for some i \ge 3
	for (i = 3; i < n; i++) {
		if (y[i]) {
			if (EVEN(i)) {
				z[2] = 1;
				z[i - 1] = 1;
				return;
				}
			else {
				z[2] = 1;
				z[i + 1] = 1;
				return;
				}
			}
		}
	cout << "error in find_root_parabolic_xyz" << endl;
	exit(1);
}

int orthogonal::find_root_parabolic(int rk2, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int root, u, v;

	if (f_v) {
		cout << "find_root_parabolic rk2=" << rk2 << endl;
		}
	if (rk2 == 0) {
		cout << "find_root_parabolic: rk2 must not be 0" << endl;
		exit(1);
		}
#if 0
	if (m == 1) {
		cout << "find_root_parabolic: m must not be 1" << endl;
		exit(1);
		}
#endif
	find_root_parabolic_xyz(rk2,
			find_root_x, find_root_y, find_root_z, verbose_level);
	if (f_v) {
		cout << "found root: ";
		int_vec_print(cout, find_root_x, n);
		int_vec_print(cout, find_root_y, n);
		int_vec_print(cout, find_root_z, n);
		cout << endl;
		}
	u = evaluate_parabolic_bilinear_form(find_root_z, find_root_x, 1, m);
	if (u == 0) {
		cout << "find_root_parabolic u=" << u << endl;
		exit(1);
		}
	v = evaluate_parabolic_bilinear_form(find_root_z, find_root_y, 1, m);
	if (v == 0) {
		cout << "find_root_parabolic v=" << v << endl;
		exit(1);
		}
	root = rank_point(find_root_z, 1, verbose_level - 1);
	if (f_v) {
		cout << "find_root_parabolic root=" << root << endl;
		}
	return root;
}

void orthogonal::Siegel_move_forward_by_index(
		int rk1, int rk2, int *v, int *w, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "Siegel_move_forward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
		}
	if (rk1 == rk2) {
		for (i = 0; i < n; i++)
			w[i] = v[i];
		return;
		}
	unrank_point(Sv1, 1, rk1, verbose_level - 1);
	unrank_point(Sv2, 1, rk2, verbose_level - 1);
	if (f_v) {
		cout << "Siegel_move_forward_by_index" << endl;
		cout << rk1 << " : ";
		int_vec_print(cout, Sv1, n);
		cout << endl;
		cout << rk2 << " : ";
		int_vec_print(cout, Sv2, n);
		cout << endl;
		}
	Siegel_move_forward(Sv1, Sv2, v, w, verbose_level);
	if (f_v) {
		cout << "moving forward: ";
		int_vec_print(cout, v, n);
		cout << endl;
		cout << "            to: ";
		int_vec_print(cout, w, n);
		cout << endl;
		}
}

void orthogonal::Siegel_move_backward_by_index(
		int rk1, int rk2, int *w, int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "Siegel_move_backward_by_index "
				"rk1=" << rk1 << " rk2=" << rk2 << endl;
		}
	if (rk1 == rk2) {
		for (i = 0; i < n; i++)
			v[i] = w[i];
		return;
		}
	unrank_point(Sv1, 1, rk1, verbose_level - 1);
	unrank_point(Sv2, 1, rk2, verbose_level - 1);
	if (f_v) {
		cout << "Siegel_move_backward_by_index" << endl;
		cout << rk1 << " : ";
		int_vec_print(cout, Sv1, n);
		cout << endl;
		cout << rk2 << " : ";
		int_vec_print(cout, Sv2, n);
		cout << endl;
		}
	Siegel_move_backward(Sv1, Sv2, w, v, verbose_level);
	if (f_v) {
		cout << "moving backward: ";
		int_vec_print(cout, w, n);
		cout << endl;
		cout << "              to ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
}

void orthogonal::Siegel_move_forward(
		int *v1, int *v2, int *v3, int *v4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk1_subspace, rk2_subspace, root, i;
	
	if (f_v) {
		cout << "Siegel_move_forward" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	if (f_v) {
		cout << "rk1_subspace=" << rk1_subspace << endl;
		cout << "rk2_subspace=" << rk2_subspace << endl;
		}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < n; i++)
			v4[i] = v3[i];
		return;
		}
	
	root = subspace->find_root_parabolic(rk2_subspace, verbose_level - 2);
	if (f_vv) {
		cout << "root=" << root << endl;
		}
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level - 2);
	F->mult_matrix_matrix(v3, T1, v4, 1, n - 2, n - 2,
			0 /* verbose_level */);
	v4[n - 2] = v3[n - 2];
	v4[n - 1] = v3[n - 1];
	if (f_v) {
		cout << "moving: ";
		int_vec_print(cout, v3, n);
		cout << endl;
		cout << "     to ";
		int_vec_print(cout, v4, n);
		cout << endl;
		}
}

void orthogonal::Siegel_move_backward(
		int *v1, int *v2, int *v3, int *v4, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk1_subspace, rk2_subspace, root, i;
	
	if (f_v) {
		cout << "Siegel_move_backward" << endl;
		int_vec_print(cout, v1, n);
		cout << endl;
		int_vec_print(cout, v2, n);
		cout << endl;
		}
	rk1_subspace = subspace->rank_point(v1, 1, verbose_level - 1);
	rk2_subspace = subspace->rank_point(v2, 1, verbose_level - 1);
	if (f_v) {
		cout << "rk1_subspace=" << rk1_subspace << endl;
		cout << "rk2_subspace=" << rk2_subspace << endl;
		}
	if (rk1_subspace == rk2_subspace) {
		for (i = 0; i < n; i++)
			v4[i] = v3[i];
		return;
		}
	
	root = subspace->find_root_parabolic(
			rk2_subspace, verbose_level - 2);
	if (f_vv) {
		cout << "root=" << root << endl;
		cout << "image, to be moved back: " << endl;
		int_vec_print(cout, v4, n);
		cout << endl;
		}
	subspace->Siegel_Transformation(T1,
			rk1_subspace, rk2_subspace, root, verbose_level - 2);
	F->invert_matrix(T1, T2, n - 2);
	F->mult_matrix_matrix(v3, T2, v4, 1, n - 2, n - 2,
			0 /* verbose_level */);
	v4[n - 2] = v3[n - 2];
	v4[n - 1] = v3[n - 1];
	if (f_v) {
		cout << "moving: ";
		int_vec_print(cout, v3, n);
		cout << endl;
		cout << "     to ";
		int_vec_print(cout, v4, n);
		cout << endl;
		}
}

void orthogonal::parabolic_canonical_points_of_line(
	int line_type, int pt1, int pt2,
	int &cpt1, int &cpt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "parabolic_canonical_points_of_line "
				"line_type=" << line_type
				<< " pt1=" << pt1 << " pt2=" << pt2 << endl;
		}
	if (line_type == 1) {
		if (f_even) {
			parabolic_canonical_points_L1_even(pt1, pt2, cpt1, cpt2);
			}
		else {
			parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
			}
		}
	else if (line_type == 2) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 3) {
		parabolic_canonical_points_L3(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 4) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 5) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 6) {
		cpt1 = pt1;
		cpt2 = pt2;
		}
	else if (line_type == 7) {
		parabolic_canonical_points_L7(pt1, pt2, cpt1, cpt2);
		}
	else if (line_type == 8) {
		parabolic_canonical_points_L8(pt1, pt2, cpt1, cpt2);
		}
	if (f_v) {
		cout << "parabolic_canonical_points_of_line "
				"of type " << line_type << endl;
		cout << "pt1=" << pt1 << " pt2=" << pt2 << endl;
		cout << "cpt1=" << cpt1 << " cpt2=" << cpt2 << endl;
		}
}

void orthogonal::parabolic_canonical_points_L1_even(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int verbose_level = 0;
	int i;
	
	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	
	//cout << "parabolic_canonical_points_L1_even" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;

	Gauss_step(v2, v1, n, n - 1);


	//cout << "after Gauss_step n - 1" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;

	if (!is_zero_vector(v1 + n - 2, 1, 2)) {
		cout << "parabolic_canonical_points_L1_even ending "
				"of v1 is not zero" << endl;
		exit(1);
		}
	for (i = 1; i < n - 2; i++) {
		if (v2[i]) {
			Gauss_step(v1, v2, n, i);
			//cout << "after Gauss_step " << i << endl;
			//int_vec_print(cout, v1, n); cout << endl;
			//int_vec_print(cout, v2, n); cout << endl;

			if (!is_zero_vector(v2 + 1, 1, n - 3)) {
				cout << "parabolic_canonical_points_L1_even "
						"not zero" << endl;
				exit(1);
				}
			break;
			}
		}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void orthogonal::parabolic_canonical_points_separate_P5(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int verbose_level = 0;
	int i;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
#if 0
	cout << "parabolic_canonical_points_separate_P5" << endl;
	cout << "v1=";
	int_vec_print(cout, v1, n);
	cout << "v2=";
	int_vec_print(cout, v2, n);
	cout << endl;
#endif
	for (i = n - 2; i < n; i++)
		if (v1[i])
			break;
	if (i < n)
		Gauss_step(v2, v1, n, i);
#if 0
	cout << "after Gauss_step" << endl;
	cout << "v1=";
	int_vec_print(cout, v1, n);
	cout << "v2=";
	int_vec_print(cout, v2, n);
	cout << endl;
#endif
	if (!is_zero_vector(v1 + n - 2, 1, 2)) {
		cout << "parabolic_canonical_points_separate_P5 "
				"ending of v1 is not zero" << endl;
		cout << "v1=";
		int_vec_print(cout, v1, n);
		cout << endl;
		exit(1);
		}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void orthogonal::parabolic_canonical_points_L3(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	Gauss_step(v2, v1, n, n - 2);
	if (v1[n - 2]) {
		cout << "parabolic_canonical_points_L3 v1[n - 2]" << endl;
		exit(1);
		}
	Gauss_step(v1, v2, n, n - 1);
	if (v2[n - 1]) {
		cout << "parabolic_canonical_points_L3 v2[n - 1]" << endl;
		exit(1);
		}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void orthogonal::parabolic_canonical_points_L7(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int verbose_level = 0;
	int i;
	
	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	Gauss_step(v1, v2, n, n - 1);
	if (!is_zero_vector(v2 + n - 2, 1, 2)) {
		cout << "parabolic_canonical_points_L7 "
				"ending of v2 is not zero" << endl;
		exit(1);
		}
	// now v2 is a point in P5
	
	for (i = 0; i < n - 2; i++) {
		if (v1[i]) {
			Gauss_step(v2, v1, n, i);
			if (!is_zero_vector(v1, 1, n - 2)) {
				cout << "parabolic_canonical_points_L7 "
						"not zero" << endl;
				exit(1);
				}
			break;
			}
		}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	if (cpt1 != pt_Q) {
		cout << "parabolic_canonical_points_L7 "
				"cpt1 != pt_Q" << endl;
		exit(1);
		}
	return;
}

void orthogonal::parabolic_canonical_points_L8(
		int pt1, int pt2, int &cpt1, int &cpt2)
{
	int verbose_level = 0;
	int i;
	
	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	Gauss_step(v1, v2, n, n - 2);
	if (!is_zero_vector(v2 + n - 2, 1, 2)) {
		cout << "parabolic_canonical_points_L8 "
				"ending of v2 is not zero" << endl;
		exit(1);
		}
	// now v2 is a point in P5
	
	for (i = 0; i < n - 2; i++) {
		if (v1[i]) {
			Gauss_step(v2, v1, n, i);
			if (!is_zero_vector(v1, 1, n - 2)) {
				cout << "parabolic_canonical_points_L8 "
						"not zero" << endl;
				exit(1);
				}
			break;
			}
		}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	if (cpt1 != pt_P) {
		cout << "parabolic_canonical_points_L8 "
				"cpt1 != pt_P" << endl;
		exit(1);
		}
	return;
}

int orthogonal::evaluate_parabolic_bilinear_form(
		int *u, int *v, int stride, int m)
{
	int a, b, c;
	
	a = evaluate_hyperbolic_bilinear_form(
			u + stride, v + stride, stride, m);
	if (f_even) {
		return a;
		}
	b = F->mult(2, u[0]);
	b = F->mult(b, v[0]);
	c = F->add(a, b);
	return c;
}


void orthogonal::parabolic_point_normalize(
		int *v, int stride, int n)
{
	if (v[0]) {
		if (v[0] != 1) {
			F->PG_element_normalize_from_front(v, stride, n);
			}
		}
	else {
		F->PG_element_normalize(v, stride, n);
		}
}

void orthogonal::parabolic_normalize_point_wrt_subspace(
		int *v, int stride)
{
	int i, a, av;
	
	if (v[0]) {
		F->PG_element_normalize_from_front(v, stride, n);
		return;
		}
	for (i = n - 3; i >= 0; i--) {
		if (v[i * stride])
			break;
		}
	if (i < 0) {
		cout <<  "parabolic_normalize_point_wrt_subspace i < 0" << endl;
		exit(1);
		}
	a = v[i * stride];
	//cout << "parabolic_normalize_point_wrt_subspace "
	// "a=" << a << " in position " << i << endl;
	av = F->inverse(a);
	for (i = 0; i < n; i++) {
		v[i * stride] = F->mult(av, v[i * stride]);
		}
}

void orthogonal::parabolic_point_properties(int *v, int stride, int n, 
	int &f_start_with_one, int &middle_value, int &end_value, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m;
	
	if (f_v) {
		cout << "orthogonal::parabolic_point_properties ";
		int_vec_print(cout, v, n);
		cout << endl;
		}
	m = (n - 1) / 2;
	if (v[0]) {
		if (v[0] != 1) {
			cout << "error in parabolic_point_properties: "
					"v[0] != 1" << endl;
			exit(1);
			}
		f_start_with_one = TRUE;
		}
	else {
		f_start_with_one = FALSE;
		F->PG_element_normalize(v + 1, stride, n - 1);
		if (f_v) {
			cout << "orthogonal::parabolic_point_properties "
					"after normalization: ";
			int_vec_print(cout, v, n);
			cout << endl;
			}
		}
	middle_value = evaluate_hyperbolic_quadratic_form(
			v + 1 * stride, stride, m - 1);
	end_value = evaluate_hyperbolic_quadratic_form(
			v + (1 + 2 * (m - 1)) * stride, stride, 1);
}

int orthogonal::parabolic_is_middle_dependent(int *vec1, int *vec2)
{
	int i, j, *V1, *V2, a, b;
	
	V1 = NULL;
	V2 = NULL;
	for (i = 1; i < n - 2; i++) {
		if (vec1[i] == 0 && vec2[i] == 0)
			continue;
		if (vec1[i] == 0) {
			V1 = vec2;
			V2 = vec1;
			}
		else {
			V1 = vec1;
			V2 = vec2;
			}
		a = F->mult(V2[i], F->inverse(V1[i]));
		for (j = i; j < n - 2; j++) {
			b = F->add(F->mult(a, V1[j]), V2[j]);
			V2[j] = b;
			}
		break;
		}
	return is_zero_vector(V2 + 1, 1, n - 3);
}



// #############################################################################
// orthogonal_util.cpp
// #############################################################################


int orthogonal::test_if_minimal_on_line(int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = rank_point(v1, 1, verbose_level - 1);
	for (t = 1; t < q; t++) {
		for (i = 0; i < n; i++) {
			//cout << "i=" << i << ":" << v1[i] << " + "
			//<< t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
			}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = rank_point(v3, 1, verbose_level - 1);
		if (rk < rk0) {
			return FALSE;
			}
		}
	return TRUE;
}

void orthogonal::find_minimal_point_on_line(int *v1, int *v2, int *v3)
{
	int verbose_level = 0;
	int i, t, rk, rk0, t0;
	
	//cout << "testing point : ";
	//int_vec_print(cout, v1, n);
	//cout << " : ";
	//int_vec_print(cout, v2, n);
	//cout << endl;
	rk0 = rank_point(v1, 1, verbose_level - 1);
	t0 = 0;
	for (t = 1; t < q; t++) {
		for (i = 0; i < n; i++) {
			//cout << "i=" << i << ":" << v1[i]
			//<< " + " << t << " * " << v2[i] << "=";
			v3[i] = F->add(v1[i], F->mult(t, v2[i]));
			//cout << v3[i] << endl;
			}
		//cout << "t=" << t << " : ";
		//int_vec_print(cout, v3, n);
		//cout << endl;
		
		rk = rank_point(v3, 1, verbose_level - 1);
		if (rk < rk0) {
			t0 = t;
			}
		}
	for (i = 0; i < n; i++) {
		//cout << "i=" << i << ":" << v1[i] << " + "
		//<< t << " * " << v2[i] << "=";
		v3[i] = F->add(v1[i], F->mult(t0, v2[i]));
		//cout << v3[i] << endl;
		}
}

void orthogonal::zero_vector(int *u, int stride, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		u[stride * i] = 0;
		}
}

int orthogonal::is_zero_vector(int *u, int stride, int len)
{
	int i;
	
	for (i = 0; i < len; i++) {
		if (u[stride * i]) {
			return FALSE;
			}
		}
	return TRUE;
}

void orthogonal::change_form_value(int *u,
		int stride, int m, int multiplyer)
{
	int i;
	
	for (i = 0; i < m; i++) {
		u[stride * 2 * i] = F->mult(multiplyer, u[stride * 2 * i]);
		}
}

void orthogonal::scalar_multiply_vector(int *u,
		int stride, int len, int multiplyer)
{
	int i;
	
	for (i = 0; i < len; i++) {
		u[stride * i] = F->mult(multiplyer, u[stride * i]);
		}
}

int orthogonal::last_non_zero_entry(int *u, int stride, int len)
{
	int i;
	
	for (i = len - 1; i >= 0; i--) {
		if (u[stride * i]) {
			return u[stride * i];
			}
		}
	cout << "error in last_non_zero_entry: the vector "
			"is the zero vector" << endl;
	exit(1);
}

void orthogonal::Siegel_map_between_singular_points(int *T, 
	int rk_from, int rk_to, int root, int verbose_level)
{
	F->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root, 
		epsilon, n,
		form_c1, form_c2, form_c3, Gram_matrix, 
		verbose_level);
}

void orthogonal::Siegel_map_between_singular_points_hyperbolic(int *T, 
	int rk_from, int rk_to, int root, int m, int verbose_level)
{
	int *Gram;
	
	F->Gram_matrix(
			1, 2 * m - 1, 0,0,0, Gram);
	F->Siegel_map_between_singular_points(T,
		rk_from, rk_to, root, 
		epsilon, 2 * m,
		0, 0, 0, Gram, 
		verbose_level);
	delete [] Gram;
}

void orthogonal::Siegel_Transformation(int *T,
	int rk_from, int rk_to, int root,
	int verbose_level)
// root is not perp to from and to.
{
	int f_v = (verbose_level >= 1);
	if (f_v) {
		cout << "Siegel_Transformation rk_from=" << rk_from
				<< " rk_to=" << rk_to << " root=" << root << endl;
		}
	Siegel_Transformation2(T, 
		rk_from, rk_to, root, 
		STr_B, STr_Bv, STr_w, STr_z, STr_x,
		verbose_level);

}

void orthogonal::Siegel_Transformation2(int *T, 
	int rk_from, int rk_to, int root, 
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *From, *To, *Root;
	
	if (f_v) {
		cout << "Siegel_Transformation2" << endl;
		}
	From = NEW_int(n);
	To = NEW_int(n);
	Root = NEW_int(n);
	unrank_point(Root, 1, root, verbose_level - 1);
	unrank_point(From, 1, rk_from, verbose_level - 1);
	unrank_point(To, 1, rk_to, verbose_level - 1);
	if (f_v) {
		cout << "root: ";
		int_vec_print(cout, Root, n);
		cout << endl;
		cout << "rk_from: ";
		int_vec_print(cout, From, n);
		cout << endl;
		cout << "rk_to: ";
		int_vec_print(cout, To, n);
		cout << endl;
		}
	
	Siegel_Transformation3(T, 
		From, To, Root, 
		B, Bv, w, z, x,
		verbose_level - 1);
	FREE_int(From);
	FREE_int(To);
	FREE_int(Root);
	if (f_v) {
		cout << "the Siegel transformation is:" << endl;
		print_integer_matrix(cout, T, n, n);
		}
}

void orthogonal::Siegel_Transformation3(int *T,
	int *from, int *to, int *root,
	int *B, int *Bv, int *w, int *z, int *x,
	int verbose_level)
{
	int i, j, a, b, av, bv, minus_one;
	//int k;
	int *Gram;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics_domain Combi;
	
	if (f_v) {
		cout << "Siegel_Transformation3" << endl;
		}
	//k = n - 1;
	Gram = Gram_matrix;
	if (f_v) {
		cout << "n=" << n << endl;
		cout << "Gram matrix:" << endl;
		Combi.print_int_matrix(cout, Gram, n, n);
		}
	
	//Q_epsilon_unrank(*F, B, 1, epsilon, k,
	//form_c1, form_c2, form_c3, root);
	//Q_epsilon_unrank(*F, B + d, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_from);
	//Q_epsilon_unrank(*F, w, 1, epsilon, k,
	//form_c1, form_c2, form_c3, rk_to);
	
	for (i = 0; i < n; i++) {
		B[i] = root[i];
		B[n + i] = from[i];
		w[i] = to[i];
		}
	if (f_v) {
		cout << "root: ";
		int_vec_print(cout, B, n);
		cout << endl;
		cout << "from: ";
		int_vec_print(cout, B + n, n);
		cout << endl;
		cout << "to: ";
		int_vec_print(cout, w, n);
		cout << endl;
		}
	
	a = F->evaluate_bilinear_form(B, B + n, n, Gram);
	b = F->evaluate_bilinear_form(B, w, n, Gram);
	av = F->inverse(a);
	bv = F->inverse(b);
	for (i = 0; i < n; i++) {
		B[n + i] = F->mult(B[n + i], av);
		w[i] = F->mult(w[i], bv);
		}
	for (i = 2; i < n; i++) {
		for (j = 0; j < n; j++) {
			B[i * n + j] = 0;
			}
		}
	
	if (f_vv) {
		cout << "before perp, the matrix B is:" << endl;
		print_integer_matrix(cout, B, n, n);
		}
	F->perp(n, 2, B, Gram);
	if (f_vv) {
		cout << "the matrix B is:" << endl;
		print_integer_matrix(cout, B, n, n);
		}
	F->invert_matrix(B, Bv, n);
	if (f_vv) {
		cout << "the matrix Bv is:" << endl;
		print_integer_matrix(cout, B, n, n);
		}
	F->mult_matrix_matrix(w, Bv, z, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		print_integer_matrix(cout, z, 1, n);
		}
	z[0] = 0;
	z[1] = 0;
	if (f_vv) {
		cout << "the coefficient vector z is:" << endl;
		print_integer_matrix(cout, z, 1, n);
		}
	F->mult_matrix_matrix(z, B, x, 1, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "the vector x is:" << endl;
		print_integer_matrix(cout, x, 1, n);
		}
	minus_one = F->negate(1);
	for (i = 0; i < n; i++) {
		x[i] = F->mult(x[i], minus_one);
		}
	if (f_vv) {
		cout << "the vector -x is:" << endl;
		print_integer_matrix(cout, x, 1, n);
		}
	make_Siegel_Transformation(T, x, B, n, Gram, FALSE);
	if (f_v) {
		cout << "the Siegel transformation is:" << endl;
		print_integer_matrix(cout, T, n, n);
		}
}

void orthogonal::random_generator_for_orthogonal_group(
	int f_action_is_semilinear, 
	int f_siegel, 
	int f_reflection, 
	int f_similarity,
	int f_semisimilarity, 
	int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::random_generator_for_orthogonal_group" << endl;
		cout << "f_action_is_semilinear=" << f_action_is_semilinear << endl;
		cout << "f_siegel=" << f_siegel << endl;
		cout << "f_reflection=" << f_reflection << endl;
		cout << "f_similarity=" << f_similarity << endl;
		cout << "f_semisimilarity=" << f_semisimilarity << endl;
		}


	while (TRUE) {
		r = Os.random_integer(4);
		if (r == 0 && f_siegel) {
			break;
			}
		else if (r == 1 && f_reflection) {
			break;
			}
		else if (r == 2 && f_similarity) {
			break;
			}
		else if (r == 3 && f_semisimilarity) {
			if (!f_action_is_semilinear) {
				continue;
				}
			break;
			}
		}
		
	if (r == 0) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing Siegel_transformation" << endl;
			}
		create_random_Siegel_transformation(Mtx, verbose_level /*- 2 */);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
			}
		}
	else if (r == 1) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing orthogonal reflection" << endl;
			}

		create_random_orthogonal_reflection(Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
			}
		}
	else if (r == 2) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing similarity" << endl;
			}
		create_random_similarity(Mtx, verbose_level - 2);
		if (f_action_is_semilinear) {
			Mtx[n * n] = 0;
			}
		}
	else if (r == 3) {
		if (f_vv) {
			cout << "orthogonal::random_generator_for_orthogonal_group "
					"choosing random similarity" << endl;
			}
		create_random_semisimilarity(Mtx, verbose_level - 2);
		}
	if (f_v) {
		cout << "orthogonal::random_generator_for_orthogonal_group "
				"done" << endl;
		}
}


void orthogonal::create_random_Siegel_transformation(
		int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk_u, alpha, i;
	int nb_pts; //, nb_pts_affine;
	int k = m; // the Witt index, previously orthogonal_k;
	int d = n;
	int *u, *v;
	os_interface Os;
	
	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation" << endl;
		}
	
	u = NEW_int(d);
	v = NEW_int(d);

	nb_pts = nb_points; //nb_pts_Qepsilon(epsilon, d - 1, q);
	//nb_pts_affine = i_power_j(q, d);

	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"q=" << q << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"d=" << d << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"Witt index k=" << k << endl;
		cout << "orthogonal::create_random_Siegel_transformation "
				"nb_pts=" << nb_pts << endl;
		//cout << "orthogonal::create_random_Siegel_transformation "
		//		"nb_pts_affine=" << nb_pts_affine << endl;
		}

	rk_u = Os.random_integer(nb_pts);
	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"rk_u=" << rk_u << endl;
		}
	unrank_point(u, 1, rk_u, 0 /* verbose_level*/);
	//Q_epsilon_unrank(*F, u, 1 /*stride*/, epsilon, d - 1,
	// form_c1, form_c2, form_c3, rk_u);
			
	while (TRUE) {

#if 0
		rk_v = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal::create_random_Siegel_transformation "
					"trying rk_v=" << rk_v << endl;
			}
		AG_element_unrank(q, v, 1 /* stride */, d, rk_v);
#else
		for (i = 0; i < d; i++) {
			v[i] = Os.random_integer(q);
		}

#endif

		alpha = F->evaluate_bilinear_form(
				u, v, d, Gram_matrix);
		if (alpha == 0) {
			if (f_v) {
				cout << "orthogonal::create_random_Siegel_transformation "
						"it works" << endl;
				}
			break;
			}
		if (f_v) {
			cout << "orthogonal::create_random_Siegel_transformation "
					"fail, try again" << endl;
			}
		}
	if (f_vv) {
		cout << "rk_u = " << rk_u << " : ";
		int_vec_print(cout, u, d);
		cout << endl;
		//cout << "rk_v = " << rk_v << " : ";
		cout << "v=";
		int_vec_print(cout, v, d);
		cout << endl;
		}
		
	F->Siegel_Transformation(
			epsilon, d - 1,
			form_c1, form_c2, form_c3,
			Mtx, v, u, verbose_level - 1);

	if (f_vv) {
		cout << "form_c1=" << form_c1 << endl;
		cout << "form_c2=" << form_c2 << endl;
		cout << "form_c3=" << form_c3 << endl;
		cout << "\\rho_{";
		int_vec_print(cout, u, d);
		cout << ",";
		int_vec_print(cout, v, d);
		cout << "}=" << endl;
		int_matrix_print(Mtx, d, d);
	}
	FREE_int(u);
	FREE_int(v);
	if (f_v) {
		cout << "orthogonal::create_random_Siegel_transformation "
				"done" << endl;
		}
}


void orthogonal::create_random_semisimilarity(int *Mtx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int d = n;
	int i, a, b, c, k;
	os_interface Os;

	if (f_v) {
		cout << "orthogonal::create_random_semisimilarity" << endl;
		}
	for (i = 0; i < d * d; i++) {
		Mtx[i] = 0;
		}
	for (i = 0; i < d; i++) {
		Mtx[i * d + i] = 1;
		}
	
#if 0
	if (!f_semilinear) {
		return;
		}
#endif

	if (epsilon == 1) {
		Mtx[d * d] = Os.random_integer(F->e);
		}
	else if (epsilon == 0) {
		Mtx[d * d] = Os.random_integer(F->e);
		}
	else if (epsilon == -1) {
		if (q == 4) {
			int u, v, w, x;
			
			Mtx[d * d] = 1;
			for (i = 0; i < d - 2; i++) {
				if (EVEN(i)) {
					Mtx[i * d + i] = 3;
					Mtx[(i + 1) * d + i + 1] = 2;
					}
				}
			u = 1;
			v = 0;
			w = 3;
			x = 1;
			Mtx[(d - 2) * d + d - 2] = u;
			Mtx[(d - 2) * d + d - 1] = v;
			Mtx[(d - 1) * d + d - 2] = w;
			Mtx[(d - 1) * d + d - 1] = x;
			}
		else if (EVEN(q)) {
			cout << "orthogonal::create_random_semisimilarity "
					"semisimilarity for even characteristic and "
					"q != 4 not yet implemented" << endl;
			exit(1);
			}
		else {
			k = (F->p - 1) >> 1;
			a = F->primitive_element();
			b = F->power(a, k);
			c = F->frobenius_power(b, F->e - 1);
			Mtx[d * d - 1] = c;
			Mtx[d * d] = 1;
			cout << "orthogonal::create_random_semisimilarity "
					"k=(p-1)/2=" << k << " a=prim elt=" << a
					<< " b=a^k=" << b << " c=b^{p^{h-1}}=" << c << endl;

			}
		}

	if (f_v) {
		cout << "orthogonal::create_random_semisimilarity done" << endl;
		}
}


void orthogonal::create_random_similarity(int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int d = n;
	int i, r, r2;
	os_interface Os;
	
	if (f_v) {
		cout << "orthogonal::create_random_similarity" << endl;
		}
	for (i = 0; i < d * d; i++) {
		Mtx[i] = 0;
		}
#if 0
	if (f_semilinear) {
		Mtx[d * d] = 0;
		}
#endif
	for (i = 0; i < d; i++) {
		Mtx[i * d + i] = 1;
		}
	r = Os.random_integer(q - 1) + 1;
	if (f_vv) {
		cout << "orthogonal::create_random_similarity "
				"r=" << r << endl;
		}
	if (epsilon == 1) {
		for (i = 0; i < d; i++) {
			if (EVEN(i)) {
				Mtx[i * d + i] = r;
				}
			}
		}
	else if (epsilon == 0) {
		r2 = F->mult(r, r);
		if (f_vv) {
			cout << "orthogonal::create_random_similarity "
					"r2=" << r2 << endl;
			}
		Mtx[0 * d + 0] = r;
		for (i = 1; i < d; i++) {
			if (EVEN(i - 1)) {
				Mtx[i * d + i] = r2;
				}
			}
		}
	else if (epsilon == -1) {
		r2 = F->mult(r, r);
		for (i = 0; i < d - 2; i++) {
			if (EVEN(i)) {
				Mtx[i * d + i] = r2;
				}
			}
		i = d - 2; Mtx[i * d + i] = r;
		i = d - 1; Mtx[i * d + i] = r;
		}
	if (f_v) {
		cout << "orthogonal::create_random_similarity done" << endl;
		}
}

void orthogonal::create_random_orthogonal_reflection(
		int *Mtx, int verbose_level)
// Only makes a n x n matrix. Does not put a semilinear component.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int alpha;
	int i;
	//int rk_z;
	//int nb_pts_affine;
	int d = n;
	int cnt;
	int *z;
	os_interface Os;
	
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	
	z = NEW_int(d);

#if 0
	nb_pts_affine = i_power_j(q, d);
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection" << endl;
		cout << "nb_pts_affine=" << nb_pts_affine << endl;
		}
#endif

	cnt = 0;
	while (TRUE) {
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"iteration = " << cnt << endl;
			}

#if 0
		rk_z = random_integer(nb_pts_affine);
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"iteration = " << cnt
					<< " trying rk_z=" << rk_z << endl;
			}

		AG_element_unrank(q, z, 1 /* stride */, d, rk_z);
#else
		for (i = 0; i < d; i++) {
			z[i] = Os.random_integer(q);
		}
#endif

		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"trying ";
			int_vec_print(cout, z, d);
			cout << endl;
		}

		alpha = evaluate_quadratic_form(z, 1 /* stride */);
		if (f_v) {
			cout << "orthogonal::create_random_orthogonal_reflection "
					"value of the quadratic form is " << alpha << endl;
		}
		if (alpha) {
			break;
			}
		cnt++;
		}
	if (f_vv) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"cnt=" << cnt
				//"rk_z = " << rk_z
				<< " alpha = " << alpha << " : ";
		int_vec_print(cout, z, d);
		cout << endl;
		}
	
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"before make_orthogonal_reflection" << endl;
		}

	make_orthogonal_reflection(Mtx, z, verbose_level - 1);

	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"after make_orthogonal_reflection" << endl;
		}



	{
	int *new_Gram;
	new_Gram = NEW_int(d * d);
	
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"before transform_form_matrix" << endl;
		}

	F->transform_form_matrix(Mtx, Gram_matrix, new_Gram, d);

	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"after transform_form_matrix" << endl;
		}

	if (int_vec_compare(Gram_matrix, new_Gram, d * d) != 0) {
		cout << "create_random_orthogonal_reflection "
				"The Gram matrix is not preserved" << endl;
		cout << "Gram matrix:" << endl;
		print_integer_matrix_width(cout, Gram_matrix,
				d, d, d, F->log10_of_q);
		cout << "transformed Gram matrix:" << endl;
		print_integer_matrix_width(cout, new_Gram,
				d, d, d, F->log10_of_q);
		exit(1);
		}
	FREE_int(new_Gram);
	}
	
	FREE_int(z);
	if (f_v) {
		cout << "orthogonal::create_random_orthogonal_reflection "
				"done" << endl;
		}
	
}


void orthogonal::make_orthogonal_reflection(
		int *M, int *z, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int Qz, Qzv, i, j;
	
	if (f_v) {
		cout << "orthogonal::make_orthogonal_reflection" << endl;
		}
	Qz = evaluate_quadratic_form(z, 1);
	Qzv = F->inverse(Qz);
	Qzv = F->negate(Qzv);

	F->mult_vector_from_the_right(Gram_matrix, z, ST_w, n, n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			M[i * n + j] = F->mult(Qzv, F->mult(ST_w[i], z[j]));
			if (i == j) {
				M[i * n + j] = F->add(1, M[i * n + j]);
				}
			}
		}
	
	if (f_vv) {
		cout << "orthogonal::make_orthogonal_reflection created:" << endl;
		print_integer_matrix(cout, M, n, n);
		}
	if (f_v) {
		cout << "orthogonal::make_orthogonal_reflection done" << endl;
		}
}

void orthogonal::make_Siegel_Transformation(int *M, int *v, int *u, 
	int n, int *Gram, int verbose_level)
// if u is singular and v \in \la u \ra^\perp, then
// \pho_{u,v}(x) := x + \beta(x,v) u - \beta(x,u) v - Q(v) \beta(x,u) u
// is called the Siegel transform (see Taylor p. 148)
// Here Q is the quadratic form and \beta is
// the corresponding bilinear form
{
	int f_v = (verbose_level >= 1);
	int i, j, Qv, e;
	
	Qv = F->evaluate_quadratic_form(
			v, 1 /*stride*/,
			epsilon, n - 1,
			form_c1, form_c2, form_c3);
	F->identity_matrix(M, n);


	// compute w^T := Gram * v^T

	F->mult_vector_from_the_right(Gram, v, ST_w, n, n);


	// M := M + w^T * u
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], u[j]);
			M[i * n + j] = F->add(M[i * n + j], e);
			}
		}

	// compute w^T := Gram * u^T
	F->mult_vector_from_the_right(Gram, u, ST_w, n, n);



	// M := M - w^T * v
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], v[j]);
			M[i * n + j] = F->add(M[i * n + j], F->negate(e));
			}
		}

	// M := M - Q(v) * w^T * u

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			e = F->mult(ST_w[i], u[j]);
			M[i * n + j] = F->add(M[i * n + j],
					F->mult(F->negate(e), Qv));
			}
		}
	if (f_v) {
		cout << "Siegel matrix:" << endl;
		print_integer_matrix_width(cout, M, n, n, n, 2);
		F->transform_form_matrix(M, Gram, Gram2, n);
		cout << "transformed Gram matrix:" << endl;
		print_integer_matrix_width(cout, Gram2, n, n, n, 2);
		cout << endl;
		}
	
}

void orthogonal::unrank_S(int *v, int stride, int m, int rk)
// m = Witt index
{
	if (m == 0) {
		return;
		}
	F->S_unrank(v, stride, m, rk);
}

int orthogonal::rank_S(int *v, int stride, int m)
// m = Witt index
{
	int rk;
	
	if (m == 0) {
		return 0;
		}
	F->S_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_N(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->N_unrank(v, stride, m, rk);
}

int orthogonal::rank_N(int *v, int stride, int m)
// m = Witt index
{
	int rk;
	
	F->N_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_N1(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->N1_unrank(v, stride, m, rk);
}

int orthogonal::rank_N1(int *v, int stride, int m)
// m = Witt index
{
	int rk;
	
	F->N1_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::unrank_Sbar(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->Sbar_unrank(v, stride, m, rk);
}

int orthogonal::rank_Sbar(int *v, int stride, int m)
// m = Witt index
{
	int rk, i;
	
	for (i = 0; i < 2 * m; i++) {
		v_tmp[i] = v[i * stride];
		}
	F->Sbar_rank(v_tmp, 1, m, rk);
	return rk;
}

void orthogonal::unrank_Nbar(int *v, int stride, int m, int rk)
// m = Witt index
{
	F->Nbar_unrank(v, stride, m, rk);
}

int orthogonal::rank_Nbar(int *v, int stride, int m)
// m = Witt index
{
	int rk;
	
	F->Nbar_rank(v, stride, m, rk);
	return rk;
}

void orthogonal::normalize_point(int *v, int stride)
{
	if (epsilon == 1) {
		F->PG_element_normalize(v, stride, n);
		}
	else if (epsilon == 0) {
		parabolic_point_normalize(v, stride, n);
		}
}

int orthogonal::triple_is_collinear(int pt1, int pt2, int pt3)
{
	int verbose_level = 0;
	int rk;
	int *base_cols;
	
	base_cols = NEW_int(n);
	unrank_point(T1, 1, pt1, verbose_level - 1);
	unrank_point(T1 + n, 1, pt2, verbose_level - 1);
	unrank_point(T1 + 2 * n, 1, pt3, verbose_level - 1);
	rk = F->Gauss_int(T1,
			FALSE /* f_special */,
			FALSE /* f_complete */,
			base_cols,
			FALSE /* f_P */, NULL, 3, n, 0,
			0 /* verbose_level */);
	FREE_int(base_cols);
	if (rk < 2) {
		cout << "orthogonal::triple_is_collinear rk < 2" << endl;
		exit(1);
		}
	if (rk == 2) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

int orthogonal::is_minus_square(int i)
{
	if (DOUBLYEVEN(q - 1)) {
		if (EVEN(i)) {
			return TRUE;
			}
		else {
			return FALSE;
			}
		}
	else {
		if (EVEN(i)) {
			return FALSE;
			}
		else {
			return TRUE;
			}
		}
}

int orthogonal::is_ending_dependent(int *vec1, int *vec2)
{
	int i;
	
	for (i = n - 2; i < n; i++) {
		if (vec2[i]) {
			Gauss_step(vec1, vec2, n, i);
			if (vec2[n - 2] == 0 && vec2[n - 1] == 0) {
				return TRUE;
				}
			else {
				return FALSE;
				}
			}
		}
	//now vec2 is zero;
	return TRUE;
}

void orthogonal::Gauss_step(int *v1, int *v2, int len, int idx)
// afterwards: v2[idx] = 0 and v1,v2 span the same space as before
{
	int i, a;
	
	if (v2[idx] == 0) {
		return;
		}
	if (v1[idx] == 0) {
		for (i = 0; i < len; i++) {
			a = v2[i];
			v2[i] = v1[i];
			v1[i] = a;
			}
		return;
		}
	a = F->negate(F->mult(F->inverse(v1[idx]), v2[idx]));
	//cout << "Gauss_step a=" << a << endl;
	for (i = 0; i < len; i++) {
		v2[i] = F->add(F->mult(v1[i], a), v2[i]);
		}
}

void orthogonal::perp(int pt,
		int *Perp_without_pt, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i, j;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp pt=" << pt << endl;
		}
	
	if (f_v) {
		cout << "orthogonal::perp before lines_on_point_by_line_rank" << endl;
		}
	lines_on_point_by_line_rank(pt, line_pencil,
			verbose_level - 2);
	if (f_v) {
		cout << "orthogonal::perp line_pencil=";
		int_vec_print(cout, line_pencil, alpha);
		cout << endl;
		}

	if (f_v) {
		cout << "orthogonal::perp before points_on_line_by_line_rank" << endl;
		}
	for (i = 0; i < alpha; i++) {
		points_on_line_by_line_rank(line_pencil[i],
				Perp1 + i * (q + 1), 0 /* verbose_level */);
		}

	if (f_v) {
		cout << "orthogonal::perp points collinear "
				"with pt " << pt << ":" << endl;
		int_matrix_print(Perp1, alpha, q + 1);
		}

	Sorting.int_vec_heapsort(Perp1, alpha * (q + 1));
	if (f_v) {
		cout << "orthogonal::perp after sorting:" << endl;
		int_vec_print(cout, Perp1, alpha * (q + 1));
		cout << endl;
		}

	j = 0;
	for (i = 0; i < alpha * (q + 1); i++) {
		if (Perp1[i] != pt) {
			Perp1[j++] = Perp1[i];
			}
		}
	sz = j;
	Sorting.int_vec_heapsort(Perp1, sz);
	if (f_v) {
		cout << "orthogonal::perp after removing "
				"pt and sorting:" << endl;
		int_vec_print(cout, Perp1, sz);
		cout << endl;
		cout << "sz=" << sz << endl;
		}
	int_vec_copy(Perp1, Perp_without_pt, sz);

	if (f_v) {
		cout << "orthogonal::perp done" << endl;
		} 
}

void orthogonal::perp_of_two_points(int pt1, int pt2,
		int *Perp, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Perp1;
	int *Perp2;
	int *Perp3;
	int sz1, sz2;
	sorting Sorting;
	
	if (f_v) {
		cout << "orthogonal::perp_of_two_points "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
		}

	Perp1 = NEW_int(alpha * (q + 1));
	Perp2 = NEW_int(alpha * (q + 1));
	perp(pt1, Perp1, sz1, 0 /*verbose_level*/);
	perp(pt2, Perp2, sz2, 0 /*verbose_level*/);
	Sorting.int_vec_intersect(Perp1, sz1, Perp2, sz2, Perp3, sz);
	int_vec_copy(Perp3, Perp, sz);
	FREE_int(Perp1);
	FREE_int(Perp2);
	FREE_int(Perp3);

	if (f_v) {
		cout << "orthogonal::perp_of_two_points done" << endl;
		} 
}

void orthogonal::perp_of_k_points(int *pts, int nb_pts,
		int *&Perp, int &sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "orthogonal::perp_of_k_points nb_pts=" << nb_pts << endl;
		cout << "pts=";
		int_vec_print(cout, pts, nb_pts);
		cout << endl;
		for (i = 0; i < nb_pts; i++) {
			unrank_point(v1, 1, pts[i], 0 /* verbose_level*/);
			cout << i << " : " << pts[i] << " : ";
			int_vec_print(cout, v1, n);
			cout << endl;
			}
		}
	if (nb_pts < 2) {
		cout << "orthogonal::perp_of_k_points nb_pts < 2" << endl;
		exit(1);
		}

	int **Perp_without_pt;
	int *Intersection1 = NULL;
	int sz1;
	int *Intersection2 = NULL;
	int sz2;
	int sz0, perp_sz = 0;
	sorting Sorting;

	sz0 = alpha * q;
	Perp_without_pt = NEW_pint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		if (f_v) {
			cout << "orthogonal::perp_of_k_points "
					"computing perp of point " << i
					<< " / " << nb_pts << ":" << endl;
			}
		Perp_without_pt[i] = NEW_int(sz0);
		perp(pts[i], Perp_without_pt[i], perp_sz,
				0 /* verbose_level */);
		if (f_v) {
			cout << "orthogonal::perp_of_k_points perp of pt "
					<< i << " / " << nb_pts << " has size "
					<< perp_sz << " and is equal to ";
			int_vec_print_fully(cout, Perp_without_pt[i], perp_sz);
			cout << endl;
			}
		if (perp_sz != sz0) {
			cout << "orthogonal::perp_of_k_points perp_sz != sz0" << endl;
			exit(1);
			}
		}
	Sorting.int_vec_intersect(Perp_without_pt[0], perp_sz,
			Perp_without_pt[1], perp_sz, Intersection1, sz1);
	if (f_v) {
		cout << "orthogonal::perp_of_k_points intersection of "
				"P[0] and P[1] has size " << sz1 << " : ";
		int_vec_print_fully(cout, Intersection1, sz1);
		cout << endl;
		}
	for (i = 2; i < nb_pts; i++) {
		if (f_v) {
			cout << "intersecting with perp[" << i << "]" << endl;
			}
		Sorting.int_vec_intersect(Intersection1, sz1,
				Perp_without_pt[i], sz0, Intersection2, sz2);

		if (f_v) {
			cout << "orthogonal::perp_of_k_points intersection "
					"with P[" << i << "] has size " << sz2 << " : ";
			int_vec_print_fully(cout, Intersection2, sz2);
			cout << endl;
			}


		FREE_int(Intersection1);
		Intersection1 = Intersection2;
		Intersection2 = NULL;
		sz1 = sz2;
		}

	Perp = NEW_int(sz1);
	int_vec_copy(Intersection1, Perp, sz1);
	sz = sz1;


	FREE_int(Intersection1);
	for (i = 0; i < nb_pts; i++) {
		FREE_int(Perp_without_pt[i]);
	}
	FREE_pint(Perp_without_pt);
	//free_pint_all(Perp_without_pt, nb_pts);


	if (f_v) {
		cout << "orthogonal::perp_of_k_points done" << endl;
		} 
}


void orthogonal::create_FTWKB_BLT_set(int *set, int verbose_level)
// for q congruent 2 mod 3
// a(t)= t, b(t) = 3*t^2, c(t) = 3*t^3, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int r, i, a, b, c;

	int q = F->q;

	if (q <= 5) {
		cout << "orthogonal::create_FTWKB_BLT_set q <= 5" << endl;
		exit(1);
		}
	r = q % 3;
	if (r != 2) {
		cout << "orthogonal::create_FTWKB_BLT_set q mod 3 must be 2" << endl;
		exit(1);
		}
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(3, F->power(i, 2));
		c = F->mult(3, F->power(i, 3));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
			}
		F->create_BLT_point(v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	int_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		int_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "orthogonal::create_FTWKB_BLT_set the BLT set FTWKB is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_K1_BLT_set(int *set, int verbose_level)
// for a nonsquare m, and q=p^e
// a(t)= t, b(t) = 0, c(t) = -m*t^p, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int i, m, minus_one, exponent, a, b, c;
	int q;

	q = F->q;
	m = F->p; // the primitive element is a nonsquare
	exponent = F->p;
	minus_one = F->negate(1);
	if (f_v) {
		cout << "m=" << m << endl;
		cout << "exponent=" << exponent << endl;
		cout << "minus_one=" << minus_one << endl;
		}
	for (i = 0; i < q; i++) {
		a = i;
		b = 0;
		c = F->mult(minus_one, F->mult(m, F->power(i, exponent)));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
			}
		F->create_BLT_point(v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	int_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		int_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "orthogonal::create_K1_BLT_set the BLT set K1 is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_K2_BLT_set(int *set, int verbose_level)
// for q congruent 2 or 3 mod 5
// a(t)= t, b(t) = 5*t^3, c(t) = 5*t^5, all t \in GF(q)
// together with the point (0, 0, 0, 1, 0)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5];
	int five, r, i, a, b, c;
	int q;

	q = F->q;
	if (q <= 5) {
		cout << "orthogonal::create_K2_BLT_set q <= 5" << endl;
		return;
		}
	r = q % 5;
	if (r != 2 && r != 3) {
		cout << "orthogonal::create_K2_BLT_set "
				"q mod 5 must be 2 or 3" << endl;
		return;
		}
	five = 5 % F->p;
	for (i = 0; i < q; i++) {
		a = i;
		b = F->mult(five, F->power(i, 3));
		c = F->mult(five, F->power(i, 5));
		if (f_vv) {
			cout << "i=" << i << " a=" << a
					<< " b=" << b << " c=" << c << endl;
			}
		F->create_BLT_point(v, a, b, c, verbose_level - 2);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	int_vec_init5(v, 0, 0, 0, 1, 0);
	if (f_vv) {
		cout << "point : ";
		int_vec_print(cout, v, 5);
		cout << endl;
		}
	set[q] = rank_point(v, 1, 0);
	if (f_vv) {
		cout << "rank " << set[q] << endl;
		}
	if (f_v) {
		cout << "orthogonal::create_K2_BLT_set "
				"the BLT set K2 is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_72_BLT_set(
		int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_72_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		int_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_72_BLT_set "
				"the BLT set LP_37_72 is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_4a_BLT_set(int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,9,16,8,5,
		1,13,20,26,2,
		1,4,12,14,22,
		1,19,23,5,5,
		1,24,17,19,32,
		1,18,18,10,14,
		1,2,4,36,23,
		1,7,5,24,29,
		1,36,20,22,29,
		1,14,10,13,14,
		1,28,22,7,23,
		1,32,28,20,19,
		1,30,27,23,24,
		1,3,30,28,15,
		1,1,20,31,13,
		1,11,36,33,6,
		1,29,22,30,15,
		1,20,10,4,5,
		1,8,14,32,29,
		1,25,15,9,31,
		1,26,13,18,29,
		1,23,19,6,19,
		1,35,11,15,20,
		1,22,11,25,32,
		1,10,16,2,20,
		1,17,18,27,31,
		1,15,29,16,29,
		1,31,18,1,15,
		1,12,34,35,15,
		1,33,23,17,20,
		1,27,23,21,14,
		1,34,22,3,6,
		1,21,11,11,18,
		1,5,33,12,35,
		1,6,22,34,15,
		1,16,31,29,18
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_4a_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		int_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_4a_BLT_set "
				"the BLT set LP_37_4a is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_LP_37_4b_BLT_set(int *set, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
		0,0,0,0,1,
		1,0,0,0,0,
		1,3,7,25,24,
		1,35,30,32,15,
		1,4,10,30,2,
		1,14,8,17,31,
		1,30,18,2,23,
		1,19,0,10,32,
		1,8,18,12,24,
		1,34,2,20,19,
		1,28,34,15,15,
		1,2,21,23,31,
		1,13,29,36,23,
		1,23,13,8,17,
		1,25,12,35,17,
		1,1,14,4,22,
		1,17,2,19,6,
		1,12,17,1,32,
		1,27,23,3,19,
		1,20,2,21,20,
		1,33,30,22,2,
		1,11,16,31,32,
		1,29,6,13,31,
		1,16,17,7,6,
		1,6,25,14,31,
		1,32,27,29,8,
		1,15,8,9,23,
		1,5,17,24,35,
		1,18,13,33,14,
		1,7,36,26,2,
		1,21,34,28,32,
		1,10,22,16,22,
		1,26,34,27,29,
		1,31,13,34,35,
		1,9,13,18,2,
		1,22,28,5,31,
		1,24,3,11,23,
		1,36,27,6,17
		};
	int q;

	q = F->q;
	if (q != 37) {
		cout << "orthogonal::create_LP_37_4b_BLT_set q = 37" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		int_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_LP_37_4b_BLT_set "
				"the BLT set LP_37_4b is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}

void orthogonal::create_Law_71_BLT_set(
		int *set, int verbose_level)
// This example can be found in Maska Law's thesis on page 115.
// Maska Law: Flocks, generalised quadrangles
// and translatrion planes from BLT-sets,
// The University of Western Australia, 2003.
// Note the coordinates here are different (for an unknown reason).
// Law suggests to construct an infinite family
// starting form the subgroup A_4 of
// the stabilizer of the Fisher/Thas/Walker/Kantor examples.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int v[5], v0, v1, v2, v3, v4;
	int i;
	int coordinates[] = {
#if 1
		0,0,0,0,1,
		1,0,0,0,0,
		1,20,1,33,5,
		1,6,23,19,23,
		1,32,11,35,17,
		1,33,12,14,23,
		1,25,8,12,6,
		1,16,6,1,22,
		1,23,8,5,6,
		1,8,6,13,8,
		1,22,19,20,13,
		1,21,23,16,23,
		1,28,6,9,8,
		1,2,26,7,13,
		1,5,9,36,35,
		1,12,23,10,17,
		1,14,16,25,23,
		1,9,8,26,35,
		1,1,11,8,19,
		1,19,12,11,17,
		1,18,27,22,22,
		1,24,36,17,35,
		1,26,27,23,5,
		1,27,25,24,22,
		1,36,21,32,35,
		1,7,16,31,8,
		1,35,5,15,5,
		1,10,36,6,13,
		1,30,4,3,5,
		1,4,3,30,19,
		1,17,13,2,19,
		1,11,28,18,17,
		1,13,16,27,22,
		1,29,12,28,6,
		1,15,10,34,19,
		1,3,30,4,13,
		1,31,9,21,8,
		1,34,9,29,6
#endif
		};
	int q;

	q = F->q;
	if (q != 71) {
		cout << "orthogonal::create_Law_71_BLT_set q = 71" << endl;
		return;
		}
	for (i = 0; i <= q; i++) {
		v0 = coordinates[i * 5 + 2];
		v1 = coordinates[i * 5 + 0];
		v2 = coordinates[i * 5 + 4];
		v3 = coordinates[i * 5 + 1];
		v4 = coordinates[i * 5 + 3];
		int_vec_init5(v, v0, v1, v2, v3, v4);
		if (f_vv) {
			cout << "point " << i << " : ";
			int_vec_print(cout, v, 5);
			cout << endl;
			}
		set[i] = rank_point(v, 1, 0);
		if (f_vv) {
			cout << "rank " << set[i] << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal::create_Law_71_BLT_set "
				"the BLT set LP_71 is ";
		int_vec_print(cout, set, q + 1);
		cout << endl;
		}
}


// formerly DISCRETA/extras.cpp
//
// Anton Betten
// Sept 17, 2010

// plane_invariant started 2/23/09


void orthogonal::plane_invariant(unusual_model *U,
	int size, int *set,
	int &nb_planes, int *&intersection_matrix,
	int &Block_size, int *&Blocks,
	int verbose_level)
// using hash values
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *Mtx;
	int *Hash;
	int rk, H, log2_of_q, n_choose_k;
	int f_special = FALSE;
	int f_complete = TRUE;
	int base_col[1000];
	int subset[1000];
	int level = 3;
	int n = 5;
	int cnt;
	int i;
	int q;
	number_theory_domain NT;
	combinatorics_domain Combi;
	sorting Sorting;


	q = F->q;
	n_choose_k = Combi.int_n_choose_k(size, level);
	log2_of_q = NT.int_log2(q);

	Mtx = NEW_int(level * n);
	Hash = NEW_int(n_choose_k);

	Combi.first_k_subset(subset, size, level);
	cnt = -1;

	if (f_v) {
		cout << "computing planes spanned by 3-subsets" << endl;
		cout << "n_choose_k=" << n_choose_k << endl;
		cout << "log2_of_q=" << log2_of_q << endl;
		}
	while (TRUE) {
		cnt++;

		for (i = 0; i < level; i++) {
			F->Q_unrank(Mtx + i * n, 1, n - 1, set[subset[i]]);
			}
		if (f_vvv) {
			cout << "subset " << setw(5) << cnt << " : ";
			int_vec_print(cout, subset, level);
			cout << " : "; // << endl;
			}
		//print_integer_matrix_width(cout, Mtx, level, n, n, 3);
		rk = F->Gauss_int(Mtx, f_special, f_complete,
				base_col, FALSE, NULL, level, n, n, 0);
		if (f_vvv) {
			cout << "after Gauss, rank = " << rk << endl;
			print_integer_matrix_width(cout, Mtx, level, n, n, 3);
			}
		H = 0;
		for (i = 0; i < level * n; i++) {
			H = hashing_fixed_width(H, Mtx[i], log2_of_q);
			}
		if (f_vvv) {
			cout << "hash =" << setw(10) << H << endl;
			}
		Hash[cnt] = H;
		if (!Combi.next_k_subset(subset, size, level)) {
			break;
			}
		}
	int *Hash_sorted, *sorting_perm, *sorting_perm_inv,
		nb_types, *type_first, *type_len;

	Sorting.int_vec_classify(n_choose_k, Hash, Hash_sorted,
		sorting_perm, sorting_perm_inv,
		nb_types, type_first, type_len);


	if (f_v) {
		cout << nb_types << " types of planes" << endl;
		}
	if (f_vvv) {
		for (i = 0; i < nb_types; i++) {
			cout << setw(3) << i << " : "
				<< setw(4) << type_first[i] << " : "
				<< setw(4) << type_len[i] << " : "
				<< setw(10) << Hash_sorted[type_first[i]] << endl;
			}
		}
	int *type_len_sorted, *sorting_perm2, *sorting_perm_inv2,
		nb_types2, *type_first2, *type_len2;

	Sorting.int_vec_classify(nb_types, type_len, type_len_sorted,
		sorting_perm2, sorting_perm_inv2,
		nb_types2, type_first2, type_len2);

	if (f_v) {
		cout << "multiplicities:" << endl;
		for (i = 0; i < nb_types2; i++) {
			//cout << setw(3) << i << " : "
			//<< setw(4) << type_first2[i] << " : "
			cout << setw(4) << type_len2[i] << " x "
				<< setw(10) << type_len_sorted[type_first2[i]] << endl;
			}
		}
	int f, ff, ll, j, u, ii, jj, idx;

	f = type_first2[nb_types2 - 1];
	nb_planes = type_len2[nb_types2 - 1];
	if (f_v) {
		if (nb_planes == 1) {
			cout << "there is a unique plane that appears "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
			}
		else {
			cout << "there are " << nb_planes
					<< " planes that each appear "
					<< type_len_sorted[f]
					<< " times among the 3-sets of points" << endl;
			for (i = 0; i < nb_planes; i++) {
				j = sorting_perm_inv2[f + i];
				cout << "The " << i << "-th plane, which is " << j
						<< ", appears " << type_len_sorted[f + i]
						<< " times" << endl;
				}
			}
		}
	if (f_vvv) {
		cout << "these planes are:" << endl;
		for (i = 0; i < nb_planes; i++) {
			cout << "plane " << i << endl;
			j = sorting_perm_inv2[f + i];
			ff = type_first[j];
			ll = type_len[j];
			for (u = 0; u < ll; u++) {
				cnt = sorting_perm_inv[ff + u];
				Combi.unrank_k_subset(cnt, subset, size, level);
				cout << "subset " << setw(5) << cnt << " : ";
				int_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			}
		}

	//return;

	//int *Blocks;
	int *Block;
	//int Block_size;


	Block = NEW_int(size);
	Blocks = NEW_int(nb_planes * size);

	for (i = 0; i < nb_planes; i++) {
		j = sorting_perm_inv2[f + i];
		ff = type_first[j];
		ll = type_len[j];
		if (f_vv) {
			cout << setw(3) << i << " : " << setw(3) << " : "
				<< setw(4) << ff << " : "
				<< setw(4) << ll << " : "
				<< setw(10) << Hash_sorted[type_first[j]] << endl;
			}
		Block_size = 0;
		for (u = 0; u < ll; u++) {
			cnt = sorting_perm_inv[ff + u];
			Combi.unrank_k_subset(cnt, subset, size, level);
			if (f_vvv) {
				cout << "subset " << setw(5) << cnt << " : ";
				int_vec_print(cout, subset, level);
				cout << " : " << endl;
				}
			for (ii = 0; ii < level; ii++) {
				F->Q_unrank(Mtx + ii * n, 1, n - 1, set[subset[ii]]);
				}
			for (ii = 0; ii < level; ii++) {
				if (!Sorting.int_vec_search(Block, Block_size, subset[ii], idx)) {
					for (jj = Block_size; jj > idx; jj--) {
						Block[jj] = Block[jj - 1];
						}
					Block[idx] = subset[ii];
					Block_size++;
					}
				}
			rk = F->Gauss_int(Mtx, f_special,
					f_complete, base_col, FALSE, NULL, level, n, n, 0);
			if (f_vvv)  {
				cout << "after Gauss, rank = " << rk << endl;
				print_integer_matrix_width(cout, Mtx, level, n, n, 3);
				}

			H = 0;
			for (ii = 0; ii < level * n; ii++) {
				H = hashing_fixed_width(H, Mtx[ii], log2_of_q);
				}
			if (f_vvv) {
				cout << "hash =" << setw(10) << H << endl;
				}
			}
		if (f_vv) {
			cout << "found Block ";
			int_vec_print(cout, Block, Block_size);
			cout << endl;
			}
		for (u = 0; u < Block_size; u++) {
			Blocks[i * Block_size + u] = Block[u];
			}
		}
	if (f_vv) {
		cout << "Incidence structure between points "
				"and high frequency planes:" << endl;
		if (nb_planes < 30) {
			print_integer_matrix_width(cout, Blocks,
					nb_planes, Block_size, Block_size, 3);
			}
		}

	int *Incma, *Incma_t, *IIt, *ItI;
	int a;

	Incma = NEW_int(size * nb_planes);
	Incma_t = NEW_int(nb_planes * size);
	IIt = NEW_int(size * size);
	ItI = NEW_int(nb_planes * nb_planes);


	for (i = 0; i < size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < Block_size; j++) {
			a = Blocks[i * Block_size + j];
			Incma[a * nb_planes + i] = 1;
			}
		}
	if (f_vv) {
		cout << "Incidence matrix:" << endl;
		print_integer_matrix_width(cout, Incma,
				size, nb_planes, nb_planes, 1);
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < nb_planes; j++) {
			Incma_t[j * size + i] = Incma[i * nb_planes + j];
			}
		}
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] * Incma_t[u * size + j];
				}
			IIt[i * size + j] = a;
			}
		}
	if (f_vv) {
		cout << "I * I^\\top = " << endl;
		print_integer_matrix_width(cout, IIt, size, size, size, 2);
		}
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < size; u++) {
				a += Incma[u * nb_planes + i] * Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}
	if (f_v) {
		cout << "I^\\top * I = " << endl;
		print_integer_matrix_width(cout, ItI,
				nb_planes, nb_planes, nb_planes, 3);
		}

	intersection_matrix = NEW_int(nb_planes * nb_planes);
	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			intersection_matrix[i * nb_planes + j] = ItI[i * nb_planes + j];
			}
		}

#if 0
	{
		char fname[1000];

		sprintf(fname, "plane_invariant_%d_%d.txt", q, k);

		ofstream fp(fname);
		fp << nb_planes << endl;
		for (i = 0; i < nb_planes; i++) {
			for (j = 0; j < nb_planes; j++) {
				fp << ItI[i * nb_planes + j] << " ";
				}
			fp << endl;
			}
		fp << -1 << endl;
		fp << "# Incidence structure between points "
				"and high frequency planes:" << endl;
		fp << l << " " << Block_size << endl;
		print_integer_matrix_width(fp,
				Blocks, nb_planes, Block_size, Block_size, 3);
		fp << -1 << endl;

	}
#endif

	FREE_int(Mtx);
	FREE_int(Hash);
	FREE_int(Block);
	//FREE_int(Blocks);
	FREE_int(Incma);
	FREE_int(Incma_t);
	FREE_int(IIt);
	FREE_int(ItI);


	FREE_int(Hash_sorted);
	FREE_int(sorting_perm);
	FREE_int(sorting_perm_inv);
	FREE_int(type_first);
	FREE_int(type_len);



	FREE_int(type_len_sorted);
	FREE_int(sorting_perm2);
	FREE_int(sorting_perm_inv2);
	FREE_int(type_first2);
	FREE_int(type_len2);



}



}}

