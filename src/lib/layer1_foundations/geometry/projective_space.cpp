// projective_space.cpp
//
// Anton Betten
// Jan 17, 2010

#include "foundations.h"


using namespace std;


#define MAX_NUMBER_OF_LINES_FOR_INCIDENCE_MATRIX 100000
#define MAX_NUMBER_OF_LINES_FOR_LINE_TABLE 1000000
#define MAX_NUMBER_OF_POINTS_FOR_POINT_TABLE 1000000
#define MAX_NB_POINTS_FOR_LINE_THROUGH_TWO_POINTS_TABLE 10000
#define MAX_NB_POINTS_FOR_LINE_INTERSECTION_TABLE 10000


namespace orbiter {
namespace layer1_foundations {
namespace geometry {


projective_space::projective_space()
{
	orbiter_kernel_system::Orbiter->nb_times_projective_space_created++;

	Subspaces = NULL;


	Plane = NULL;

	Solid = NULL;


	Arc_in_projective_space = NULL;

	Reporting = NULL;



}



projective_space::~projective_space()
{
	int f_v = false;

	if (f_v) {
		cout << "projective_space::~projective_space" << endl;
	}

	if (Subspaces) {
		FREE_OBJECT(Subspaces);
	}

	if (Plane) {
		FREE_OBJECT(Plane);
	}
	if (Solid) {
		FREE_OBJECT(Solid);
	}

	if (Arc_in_projective_space) {
		FREE_OBJECT(Arc_in_projective_space);
	}
	if (Reporting) {
		FREE_OBJECT(Reporting);
	}



	if (f_v) {
		cout << "projective_space::~projective_space done" << endl;
	}
}

void projective_space::projective_space_init(
		int n,
		field_theory::finite_field *F,
	int f_init_incidence_structure, 
	int verbose_level)
// n is projective dimension
{
	int f_v = (verbose_level >= 1);
	//int i;
	combinatorics::combinatorics_domain C;
	ring_theory::longinteger_object a;

	if (f_v) {
		cout << "projective_space::projective_space_init "
				"PG(" << n << "," << F->q << ")" << endl;
		cout << "f_init_incidence_structure="
				<< f_init_incidence_structure << endl;
	}
	char str[1000];

	snprintf(str, sizeof(str), "PG_%d_%d", n, F->q);
	label_txt.assign(str);
	snprintf(str, sizeof(str), "PG(%d,%d)", n, F->q);
	label_tex.assign(str);



	Subspaces = NEW_OBJECT(projective_space_subspaces);

	Subspaces->init(
		this,
		n,
		F,
		f_init_incidence_structure,
		verbose_level);


	if (n == 2) {

		Plane = NEW_OBJECT(projective_space_plane);

		if (f_v) {
			cout << "projective_space::projective_space_init "
					"before Plane->init" << endl;
		}
		Plane->init(this, verbose_level);
		if (f_v) {
			cout << "projective_space::projective_space_init "
					"after Plane->init" << endl;
		}
	}

	else if (n == 3) {

		Solid = NEW_OBJECT(projective_space_of_dimension_three);

		if (f_v) {
			cout << "projective_space::projective_space_init "
					"before Solid->init" << endl;
		}
		Solid->init(this, verbose_level);
		if (f_v) {
			cout << "projective_space::projective_space_init "
					"after Solid->init" << endl;
		}
	}


	Arc_in_projective_space = NEW_OBJECT(arc_in_projective_space);
	Arc_in_projective_space->init(this, 0 /* verbose_level */);

	Reporting = NEW_OBJECT(projective_space_reporting);
	if (f_v) {
		cout << "projective_space::projective_space_init "
				"before Reporting->init" << endl;
	}
	Reporting->init(this, verbose_level);
	if (f_v) {
		cout << "projective_space::projective_space_init "
				"after Reporting->init" << endl;
	}

	if (f_v) {
		
		cout << "projective_space::projective_space_init n=" << n
				<< " q=" << F->q << " done" << endl;
	}
}




long int projective_space::rank_point(int *v)
{
	long int b;

	b = Subspaces->rank_point(v);
	return b;
}

void projective_space::unrank_point(int *v, long int rk)
{	
	Subspaces->unrank_point(v, rk);
}

void projective_space::unrank_points(int *v, long int *Rk, int sz)
{
	Subspaces->unrank_points(v, Rk, sz);
}

long int projective_space::rank_line(int *basis)
{
	long int b;
	
	b = Subspaces->rank_line(basis);
	return b;
}

void projective_space::unrank_line(int *basis, long int rk)
{	
	Subspaces->unrank_line(basis, rk);
}

void projective_space::unrank_lines(int *v, long int *Rk, int nb)
{
	Subspaces->unrank_lines(v, Rk, nb);
}

long int projective_space::rank_plane(int *basis)
{
	long int b;

	b = Subspaces->rank_plane(basis);
	return b;
}

void projective_space::unrank_plane(int *basis, long int rk)
{	
	Subspaces->unrank_plane(basis, rk);
}



long int projective_space::line_through_two_points(
		long int p1, long int p2)
{
	long int b;

	b = Subspaces->line_through_two_points(p1, p2);
	return b;
}

int projective_space::intersection_of_two_lines(
		long int l1, long int l2)
{
	int b;

	b = Subspaces->intersection_of_two_lines(l1, l2);
	return b;
}












void projective_space::Baer_subline(
		long int *pts3,
	long int *&pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int *M;
	int *Basis;
	int *N; // local coordinates w.r.t. basis
	int *base_cols;
	int *z;
	int rk;
	int len;
	int i, j;
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "projective_space::Baer_subline" << endl;
	}
	if (ODD(Subspaces->F->e)) {
		cout << "projective_space::Baer_subline the field degree "
				"must be even (because we need a "
				"quadratic subfield)" << endl;
		exit(1);
	}
	len = Subspaces->n + 1;
	M = NEW_int(3 * len);
	base_cols = NEW_int(len);
	z = NEW_int(len);
	for (j = 0; j < 3; j++) {
		unrank_point(M + j * len, pts3[j]);
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline" << endl;
		cout << "M=" << endl;
		Int_vec_print_integer_matrix_width(cout,
				M, 3, len, len, Subspaces->F->log10_of_q);
	}
	rk = Subspaces->F->Linear_algebra->Gauss_simple(
			M,
			3, len, base_cols, verbose_level - 3);
	if (f_vv) {
		cout << "projective_space::Baer_subline" << endl;
		cout << "has rank " << rk << endl;
		cout << "base_cols=";
		Int_vec_print(cout, base_cols, rk);
		cout << endl;
		cout << "basis:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				M, rk, len, len, Subspaces->F->log10_of_q);
	}

	if (rk != 2) {
		cout << "projective_space::Baer_subline: rk should "
				"be 2 (points are not collinear)" << endl;
		exit(1);
	}
	
	Basis = NEW_int(rk * len);
	for (j = 0; j < rk * len; j++) {
		Basis[j] = M[j];
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline basis:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Basis, rk, len, len, Subspaces->F->log10_of_q);
	}
		
	N = NEW_int(3 * rk);
	for (j = 0; j < 3; j++) {
		unrank_point(M + j * len, pts3[j]);
		//cout << "M + j * len:";
		//int_vec_print(cout, M + j * len, len);
		//cout << endl;
		//cout << "basis:" << endl;
		//print_integer_matrix_width(cout,
		//Basis, rk, 5, 5, P4->F->log10_of_q);
		
		Subspaces->F->Linear_algebra->reduce_mod_subspace_and_get_coefficient_vector(
			rk, len, Basis, base_cols, 
			M + j * len, N + j * rk, verbose_level - 3);
	}
	//cout << "after reduce_mod_subspace_and_get_
	//coefficient_vector: M=" << endl;
	//print_integer_matrix_width(cout, M, 3, len, len, F->log10_of_q);
	//cout << "(should be all zeros)" << endl;
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"local coordinates in the subspace are N=" << endl;
		Int_vec_print_integer_matrix_width(cout,
				N, 3, rk, rk, Subspaces->F->log10_of_q);
	}
	int *Frame;
	int *base_cols2;
	int rk2, a;

	Frame = NEW_int(2 * 3);
	base_cols2 = NEW_int(3);
	for (j = 0; j < 3; j++) {
		for (i = 0; i < 2; i++) {
			Frame[i * 3 + j] = N[j * 2 + i];
		}
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"Frame=" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Frame, 2, 3, 3, Subspaces->F->log10_of_q);
	}
	rk2 = Subspaces->F->Linear_algebra->Gauss_simple(Frame,
			2, 3, base_cols2, verbose_level - 3);
	if (rk2 != 2) {
		cout << "projective_space::Baer_subline: "
				"rk2 should be 2" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"after Gauss Frame=" << endl;
		Int_vec_print_integer_matrix_width(cout,
				Frame, 2, 3, 3, Subspaces->F->log10_of_q);
		cout << "projective_space::Baer_subline "
				"base_cols2=";
		Int_vec_print(cout, base_cols2, rk2);
		cout << endl;
	}
	for (i = 0; i < 2; i++) {
		a = Frame[i * 3 + 2];
		for (j = 0; j < 2; j++) {
			N[i * 2 + j] = Subspaces->F->mult(a, N[i * 2 + j]);
		}
	}
	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"local coordinates in the subspace are N=" << endl;
		Int_vec_print_integer_matrix_width(
				cout, N, 3, rk, rk, Subspaces->F->log10_of_q);
	}

#if 0
	int *Local_pts;
	int *Local_pts_sorted;
	int w[2];


	Local_pts = NEW_int(nb_pts);
	Local_pts_sorted = NEW_int(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < 2; j++) {
			w[j] = N[i * 2 + j];
		}
		PG_element_rank_modified(*F, w, 1, 2, a);
		Local_pts[i] = a;
		Local_pts_sorted[i] = a;
	}
	int_vec_heapsort(Local_pts_sorted, nb_pts);
	if (f_vv) {
		cout << "Local_pts=" << endl;
		int_vec_print(cout, Local_pts, nb_pts);
		cout << endl;
		cout << "Local_pts_sorted=" << endl;
		int_vec_print(cout, Local_pts_sorted, nb_pts);
		cout << endl;
	}
#endif


	int q0, index, t;


	q0 = NT.i_power_j(Subspaces->F->p, Subspaces->F->e >> 1);
	index = (Subspaces->F->q - 1) / (q0 - 1);
	
	nb_pts = q0 + 1;
	pts = NEW_lint(nb_pts);


	if (f_vv) {
		cout << "projective_space::Baer_subline q0=" << q0 << endl;
		cout << "projective_space::Baer_subline index=" << index << endl;
		cout << "projective_space::Baer_subline nb_pts=" << nb_pts << endl;
	}

#if 0
	for (i = 0; i < 3; i++) {
		for (j = 0; j < len; j++) {
			if (i < 2) {
				z[j] = Basis[i * len + j];
			}
			else {
				z[j] = F->add(Basis[0 * len + j], Basis[1 * len + j]);
			}
		}
		pts[i] = rank_point(z);
	}
#endif
	for (t = 0; t < 3; t++) {
		if (f_vvv) {
			cout << "t=" << t << endl;
		}
		Subspaces->F->Linear_algebra->mult_vector_from_the_left(
				N + t * 2, Basis, z, 2, len);
		if (f_vvv) {
			cout << "z=w*Basis";
			Int_vec_print(cout, z, len);
			cout << endl;
		}
		a = rank_point(z);
		pts[t] = a;
	}
	for (t = 2; t < q0; t++) {
		a = Subspaces->F->alpha_power((t - 1) * index);
		if (f_vvv) {
			cout << "t=" << t << " a=" << a << endl;
		}
		for (j = 0; j < 2; j++) {
			Subspaces->w[j] = Subspaces->F->add(N[0 * 2 + j], Subspaces->F->mult(a, N[1 * 2 + j]));
		}
		if (f_vvv) {
			cout << "w=";
			Int_vec_print(cout, Subspaces->w, 2);
			cout << endl;
		}
		Subspaces->F->Linear_algebra->mult_vector_from_the_left(
				Subspaces->w, Basis, z, 2, len);
		if (f_vvv) {
			cout << "z=w*Basis";
			Int_vec_print(cout, z, len);
			cout << endl;
		}
		a = rank_point(z);
		pts[t + 1] = a;
		if (f_vvv) {
			cout << "rank=" << a << endl;
		}
#if 0
		PG_element_rank_modified(*F, w, 1, 2, a);
		pts[t] = a;
		if (!int_vec_search(Local_pts_sorted, nb_pts, a, idx)) {
			ret = false;
			if (f_vv) {
				cout << "did not find this point in the list of "
						"points, hence not contained in Baer subline" << endl;
			}
			goto done;
		}
#endif
		
	}

	if (f_vv) {
		cout << "projective_space::Baer_subline "
				"The Baer subline is";
		Lint_vec_print(cout, pts, nb_pts);
		cout << endl;
		Reporting->print_set(pts, nb_pts);
	}
	



	FREE_int(N);
	FREE_int(M);
	FREE_int(base_cols);
	FREE_int(Basis);
	FREE_int(Frame);
	FREE_int(base_cols2);
	FREE_int(z);
}




}}}




