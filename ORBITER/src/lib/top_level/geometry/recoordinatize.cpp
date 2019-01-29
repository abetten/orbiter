// recoordinatize.C
// 
// Anton Betten
// November 17, 2009
//
// moved out of translation_plane.C: 4/16/2013
// moved to TOP_LEVEL: 11/2/2013
// 
//
//

#include "orbiter.h"


namespace orbiter {
namespace top_level {

recoordinatize::recoordinatize()
{
	null();
}

recoordinatize::~recoordinatize()
{
	freeself();
}

void recoordinatize::null()
{
	A0 = NULL;
	gens2 = NULL;

	live_points = NULL;

	f_data_is_allocated = FALSE;
}

void recoordinatize::freeself()
{
	if (f_data_is_allocated) {
		FREE_int(M);
		FREE_int(M1);
		FREE_int(AA);
		FREE_int(AAv);
		FREE_int(TT);
		FREE_int(TTv);
		FREE_int(B);
		FREE_int(C);
		FREE_int(N);
		FREE_int(Elt);
		}
	if (A0) {
		FREE_OBJECT(A0);
		}
	if (gens2) {
		FREE_OBJECT(gens2);
		}
	
	if (live_points) {
		FREE_int(live_points);
		}
	null();
}

void recoordinatize::init(int n, int k,
	finite_field *F, grassmann *Grass, action *A, action *A2,
	int f_projective, int f_semilinear, 
	int (*check_function_incremental)(int len,
			int *S, void *data, int verbose_level),
	void *check_function_incremental_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "recoordinatize::init" << endl;
		}

	recoordinatize::A = A;
	recoordinatize::A2 = A2;
	recoordinatize::Grass = Grass;
	recoordinatize::F = F;
	recoordinatize::q = F->q;
	recoordinatize::k = k;
	recoordinatize::n = n;
	recoordinatize::f_projective = f_projective;
	recoordinatize::f_semilinear = f_semilinear;
	recoordinatize::check_function_incremental
		= check_function_incremental;
	recoordinatize::check_function_incremental_data
		= check_function_incremental_data;
	nCkq = generalized_binomial(n, k, q);
	

	M = NEW_int((3 * k) * n);
	M1 = NEW_int((3 * k) * n);
	AA = NEW_int(n * n);
	AAv = NEW_int(n * n);
	TT = NEW_int(k * k);
	TTv = NEW_int(k * k);
	B = NEW_int(n * n);
	C = NEW_int(n * n);
	N = NEW_int((3 * k) * n);
	Elt = NEW_int(A->elt_size_in_int);
	f_data_is_allocated = TRUE;
	if (f_v) {
		cout << "recoordinatize::init done" << endl;
		}
}

void recoordinatize::do_recoordinatize(
		int i1, int i2, int i3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 3);
	int i, j;
	int j1, j2, j3;

	if (f_v) {
		cout << "translation_plane::recoordinatize "
				<< i1 << "," << i2 << "," << i3 << endl;
		}
	Grass->unrank_int_here(M, i1, 0 /*verbose_level - 4*/);
	Grass->unrank_int_here(M + k * n, i2, 0 /*verbose_level - 4*/);
	Grass->unrank_int_here(M + 2 * k * n, i3, 0 /*verbose_level - 4*/);
	if (f_vv) {
		cout << "M:" << endl;
		print_integer_matrix_width(cout,
				M, 3 * k, n, n, F->log10_of_q + 1);
		}
	int_vec_copy(M, AA, n * n);
	F->matrix_inverse(AA, AAv, n, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "AAv:" << endl;
		print_integer_matrix_width(cout,
				AAv, n, n, n, F->log10_of_q + 1);
		}
	F->mult_matrix_matrix(M, AAv, N, 3 * k, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "N:" << endl;
		print_integer_matrix_width(cout,
				N, 3 * k, n, n, F->log10_of_q + 1);
		}

	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			TT[i * k + j] = N[2 * k * n + i * n + j];
			}
		}
	if (f_vv) {
		cout << "TT:" << endl;
		print_integer_matrix_width(cout,
				TT, k, k, k, F->log10_of_q + 1);
		}
	F->matrix_inverse(TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "TTv:" << endl;
		print_integer_matrix_width(cout,
				TTv, k, k, k, F->log10_of_q + 1);
		}

	int_vec_zero(B, n * n);
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			B[i * n + j] = TTv[i * k + j];
			}
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			TT[i * k + j] = N[2 * k * n + i * n + k + j];
			}
		}
	if (f_vv) {
		cout << "TT:" << endl;
		print_integer_matrix_width(cout,
				TT, k, k, k, F->log10_of_q + 1);
		}
	F->matrix_inverse(TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "TTv:" << endl;
		print_integer_matrix_width(cout,
				TTv, k, k, k, F->log10_of_q + 1);
		}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			B[(k + i) * n + k + j] = TTv[i * k + j];
			}
		}
	if (f_vv) {
		cout << "B:" << endl;
		print_integer_matrix_width(cout,
				B, n, n, n, F->log10_of_q + 1);
		}

	
	F->mult_matrix_matrix(AAv, B, C, n, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "C:" << endl;
		print_integer_matrix_width(cout,
				C, n, n, n, F->log10_of_q + 1);
		}
	
	F->mult_matrix_matrix(M, C, M1, 3 * k, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "M1:" << endl;
		print_integer_matrix_width(cout,
				M1, 3 * k, n, n, F->log10_of_q + 1);
		}
	j1 = Grass->rank_int_here(M1, 0 /*verbose_level - 4*/);
	j2 = Grass->rank_int_here(M1 + k * n, 0 /*verbose_level - 4*/);
	j3 = Grass->rank_int_here(M1 + 2 * k * n, 0 /*verbose_level - 4*/);
	if (f_v) {
		cout << "j1=" << j1 << " j2=" << j2 << " j3=" << j3 << endl;
		}
	
	A->make_element(Elt, C, 0);
	if (f_vv) {
		cout << "translation_plane::recoordinatize "
				"transporter:" << endl;
		A->element_print(Elt, cout);
		}
	if (f_v) {
		cout << "translation_plane::recoordinatize done" << endl;
		}
}

void recoordinatize::compute_starter(int *&S, int &size, 
	strong_generators *&Strong_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 3);
	//int f_vvv = (verbose_level >= 3);
	
	
	if (f_v) {
		cout << "recoordinatize::compute_starter" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}

	

	make_first_three(starter_j1, starter_j2, starter_j3,
			verbose_level - 1);

	// initialize S with the vector (j1,j2,j3):
	size = 3;
	S = NEW_int(size);

	S[0] = starter_j1;
	S[1] = starter_j2;
	S[2] = starter_j3;


	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"before stabilizer_of_first_three" << endl;
		}
	stabilizer_of_first_three(Strong_gens, verbose_level - 1);
	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"after stabilizer_of_first_three" << endl;
		}




	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"before compute_live_points" << endl;
		}
	compute_live_points(verbose_level - 1);
	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"after compute_live_points" << endl;
		}


	if (f_v) {
		cout << "recoordinatize::compute_starter finished" << endl;
		cout << "we found " << nb_live_points << " live points" << endl;
		}
	
}

void recoordinatize::stabilizer_of_first_three(
	strong_generators *&Strong_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;

	longinteger_object target_go, six, target_go2, go, go_linear;
	longinteger_object go_small;

	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three" << endl;
		}

	A0 = NEW_OBJECT(action);
	A0_linear = NEW_OBJECT(action);
	gens2 = NEW_OBJECT(vector_ge);


	
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before  A0->init_matrix_group" << endl;
		}
	A0->init_projective_group(k, F, 
		f_semilinear, 
		TRUE /* f_basis */, 0 /* verbose_level */);
		
	A0->group_order(target_go);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"target_go=" << target_go
			<< " = order of PGGL(" << k << "," << q << ")" << endl;
		cout << "action A0 created: ";
		A0->print_info();
		}

	A0_linear->init_projective_group(k, F, 
		FALSE /*f_semilinear*/, 
		TRUE /*f_basis*/, 0/*verbose_level - 2*/);
		
	A0_linear->group_order(go_linear);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"order of PGL(" << k << "," << q << ") is "
			<< go_linear << endl;
		cout << "action A0_linear created: ";
		A0_linear->print_info();
		}



	
	six.create(6);
	D.mult(target_go, six, target_go2);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"target_go2=" << target_go2
			<< " = target_go times 6" << endl;
		}
	


	gens2->init(A);


	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before make_generators_stabilizer_of_"
				"three_components" << endl;
		}

	make_generators_stabilizer_of_three_components(
		A /* A_PGL_n_q */,
		A0 /* A_PGL_k_q */,
		k, gens2, verbose_level - 1);
		// in ACTION/action_global.C

	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before generators_to_strong_generators" << endl;
		}


	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go2, 
		gens2, Strong_gens, verbose_level - 1);
		// in ACTION/action_global.C


	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"done" << endl;
		}
}


void recoordinatize::compute_live_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "recoordinatize::compute_live_points" << endl;
		}


	char fname[1000];


#if 0
	int f_path_select = TRUE;
	int select_value;
	int len;


	len = i_power_j(Fq->q, Mtx->n) - 1;
	for (select_value = 1; select_value < len;  select_value++) {
		if (f_path_select) {
			sprintf(fname, "live_points_%d.txt", select_value);
			}
		else {
			sprintf(fname, "live_points.txt");
			}	
		if (file_size(fname) > 1) {
			cout << "reading live points from file "
					<< fname << endl;
			read_set_from_file(fname,
					live_points, nb_live_points, verbose_level);
			cout << "reading live points from file "
					<< fname << " done" << endl;
			return;
			}



	
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"checking all " << gos * Fq->q
					<< " elements in GL(" << k << "," << q << ")" << endl;
			cout << "order of PGL(" << k << "," << q << ")=" << gos << endl;
			}

#endif


	// we wish to run through the elements of GL(k,q).
	// instead, we run through PGL(k,q) and multiply by nonzero scalars:

#if 0

		Mtx->matrices_without_eigenvector_one(
			A0_linear->Sims, live_points, nb_live_points,
			f_path_select, select_value, verbose_level);
		// ToDo: need to change the ranks from matrices to subspaces !!!
#else


	sprintf(fname, "live_points.txt");

	if (file_size(fname) > 1) {
		cout << "recoordinatize::compute_live_points "
				"reading live points from file " << fname << endl;
		read_set_from_file(fname,
				live_points, nb_live_points, verbose_level);
		cout << "recoordinatize::compute_live_points "
				"we found " << nb_live_points << " live points" << endl;
		return;
		}
	else {
		cout << "recoordinatize::compute_live_points "
				"befor we compute_live_points_low_level" << endl;
		compute_live_points_low_level(
				live_points, nb_live_points, verbose_level - 1);

		write_set_to_file(fname,
				live_points, nb_live_points, verbose_level);
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"written file " << fname << endl;
			}
		}

#if 0

	sprintf(fname, "live_points.txt");
	if (file_size(fname) > 1) {
		cout << "reading live points from file " << fname << endl;
		read_set_from_file(fname, live_points, nb_live_points, verbose_level);
		cout << "reading live points from file " << fname << " done" << endl;
		return;
		}
	else {
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"before Mtx->matrices_without_eigenvector_one" << endl;
			}
		Mtx->matrices_without_eigenvector_one(
			A0_linear->Sims, live_points, nb_live_points,
			FALSE /* f_path_select */, 0 /*select_value*/, verbose_level);

		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"after Mtx->matrices_without_eigenvector_one" << endl;
			}

		int_vec_heapsort(live_points, nb_live_points);

		write_set_to_file(fname,
				live_points, nb_live_points, verbose_level);
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"written file " << fname << endl;
			}
		}

#endif


#if 0
	if (FALSE) {
		for (h = 0; h < nb_live_points; h++) {
			cout << "live point " << h << " is point " << live_points[h] << ":" << endl;
			Grass->unrank_int(live_points[h], 0);
			print_integer_matrix_width(cout, Grass->M, k, n, n, F->log10_of_q + 1);
			cout << endl;
			for (i = 0; i < k; i++) {
				for (j = 0; j < k; j++) {
					Elt1[i * k + j] = Grass->M[i * n + k + j];
					}
				}
			a = A0_linear->Sims->element_rank_int(Elt1);
			cout << "rank in A0 is " << a << endl;
			//A0->element_print(Elt1, cout);
			}
		}

		cout << "they are:" << endl;
		for (i = 0; i < nb_live_points; i++) {
			cout << setw(5) << i << " : " << setw(10) << live_points[i] << endl;
			}
#endif

	if (f_v) {
		cout << "recoordinatize::compute_live_points done" << endl;
		}
}

void recoordinatize::compute_live_points_low_level(
		int *&live_points, int &nb_live_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	matrix_group *Mtx;
	finite_field *Fq;
	int SS[4];
	int *Elt1;

	int cnt, z;
	int cnt_mod = 1000;
	longinteger_object go_linear;
	int gos;
	int i, j, h, a;


	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level" << endl;
		}


	Mtx = A0->G.matrix_grp;
	Fq = Mtx->GFq;

	SS[0] = starter_j1;
	SS[1] = starter_j2;
	SS[2] = starter_j3;
	
	Elt1 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level "
				"nCkq = " << nCkq << endl;
		}
	live_points = NEW_int(nCkq);
	nb_live_points = 0;

	A0_linear->group_order(go_linear);
	gos = go_linear.as_int();

	cnt = 0;
	for (h = 0; h < gos; h++) {
		for (z = 1; z < q; z++, cnt++) {
			if (f_vv) {
				if ((cnt % cnt_mod) == 0 && cnt) {
					cout << "recoordinatize::compute_live_points_low_level"
							<< cnt << " iterations, h=" << h << " found "
							<< nb_live_points << " points so far" << endl;
					}
				}
			A0_linear->Sims->element_unrank_int(h, Elt1);
			Fq->PG_element_normalize(Elt1, 1, k * k);
			if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
				cout << "recoordinatize::compute_live_points_low_level "
						"element " << cnt << " = " << h
						<< ", normalized:" << endl;
				A0->element_print(Elt1, cout);
				}
			for (i = 0; i < k * k; i++) {
				Elt1[i] = Fq->mult(Elt1[i], z);
				}
			if (f_v && (cnt % cnt_mod) == 0 && cnt) {
				cout << "recoordinatize::compute_live_points_low_level "
						"element " << cnt << " = " << h
						<< ", multiplied by z=" << z << ":" << endl;
				print_integer_matrix_width(cout,
						Elt1, k, k, k, F->log10_of_q + 1);
				}
	
			// make the k x n matrix ( I_k | Elt1 )
			int_vec_zero(Grass->M, k * n);
			for (i = 0; i < k; i++) {
				Grass->M[i * n + i] = 1;
				}
			for (i = 0; i < k; i++) {
				for (j = 0; j < k; j++) {
					Grass->M[i * n + k + j] = Elt1[i * k + j];
					}
				}
			if (f_vv && (cnt % cnt_mod) == 0) {
				cout << "recoordinatize::compute_live_points_low_level "
						"element " << h << ":" << endl;
				print_integer_matrix_width(cout, Grass->M, k, n, n, 2);
				}
			if (FALSE || ((h & ((1 << 15) - 1)) == 0 && z == 1)) {
				cout << h << " / " << gos
						<< " nb_live_points=" << nb_live_points << endl;
				print_integer_matrix_width(cout, Grass->M, k, n, n, 2);
				}
			a = Grass->rank_int(0);
			SS[3] = a;
			if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
				cout << "has rank " << a << endl;
				}
			if ((*check_function_incremental)(4, SS,
					check_function_incremental_data,
					0/*verbose_level - 4*/)) {
				if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
					cout << "recoordinatize::compute_live_points_"
							"low_level element " << cnt << " = " << h << ", "
							<< z << " subspace rank " << a
							<< " is accepted as live point no "
							<< nb_live_points << endl;
					}
				live_points[nb_live_points++] = a;
				}
			else {
				if (f_vv && (cnt % cnt_mod) == 0) {
					cout << "recoordinatize::compute_live_points_"
							"low_level element " << cnt << " = " << h << ", "
							<< z << " subspace rank " << a
							<< " is not accepted" << endl;
					}
				}
			}
		}
#endif

	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level "
				"we found " << nb_live_points << " live points" << endl;
		}
	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level "
				"sorting" << endl;
		}

	int_vec_heapsort(live_points, nb_live_points);
	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level done" << endl;
		}
}

void recoordinatize::make_first_three(
		int &j1, int &j2, int &j3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int *M;
	int i;
	
	if (f_v) {
		cout << "recoordinatize::make_first_three" << endl;
		}
	M = NEW_int(k * n);

	// make the element (I_k | 0). Let j1 be its rank
	int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		M[i * n + i] = 1;
		}
	if (f_v3) {
		cout << "recoordinatize::compute_starter M1:" << endl;
		print_integer_matrix_width(cout, M, k, n, n, F->log10_of_q + 1);
		}
	j1 = Grass->rank_int_here(M, 0/*verbose_level - 4*/);
	if (f_v3) {
		cout << "recoordinatize::compute_starter j1=" << j1 << endl;
		}

	// make the element (0 | I_k). Let j2 be its rank
	int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		M[i * n + k + i] = 1;
		}
	if (f_v3) {
		cout << "recoordinatize::compute_starter M2:" << endl;
		print_integer_matrix_width(cout, M, k, n, n, F->log10_of_q + 1);
		}
	j2 = Grass->rank_int_here(M, 0/*verbose_level - 4*/);
	if (f_v3) {
		cout << "recoordinatize::compute_starter j2=" << j2 << endl;
		}

	// make the element (I_k | I_k). Let j3 be its rank
	int_vec_zero(M, k * n);
	for (i = 0; i < k; i++) {
		M[i * n + i] = 1;
		M[i * n + k + i] = 1;
		}
	if (f_v3) {
		cout << "recoordinatize::compute_starter M3:" << endl;
		print_integer_matrix_width(cout, M, k, n, n, F->log10_of_q + 1);
		}
	j3 = Grass->rank_int_here(M, 0/*verbose_level - 4*/);
	if (f_v3) {
		cout << "recoordinatize::compute_starter j3=" << j3 << endl;
		}

	FREE_int(M);
	if (f_vv) {
		cout << "recoordinatize::make_first_three j1=" << j1
				<< ",j2=" << j2 << ",j3=" << j3 << endl;
		}
	if (f_v) {
		cout << "recoordinatize::make_first_three done" << endl;
		}
}

}}


