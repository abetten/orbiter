// recoordinatize.cpp
// 
// Anton Betten
// November 17, 2009
//
// moved out of translation_plane.cpp: 4/16/2013
// moved to TOP_LEVEL: 11/2/2013
// 
//
//

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace spreads {


recoordinatize::recoordinatize()
{

	SD = NULL;

	n = k = q = 0;

	Grass = NULL;
	F = NULL;
	A = NULL;
	A2 = NULL;
	f_projective = FALSE;
	f_semilinear = FALSE;
	nCkq = 0;
	check_function_incremental = NULL;
	check_function_incremental_data = NULL;

	//fname_live_points

	f_data_is_allocated = FALSE;
	M = M1 = AA = AAv = TT = TTv = B = C = N = Elt = NULL;

	starter_j1 = starter_j2 = starter_j3 = 0;
	A0 = NULL;
	A0_linear = NULL;
	gens2 = NULL;

	live_points = NULL;
	nb_live_points = 0;

}


recoordinatize::~recoordinatize()
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
		FREE_lint(live_points);
	}
}

void recoordinatize::init(
		geometry::spread_domain *SD,
		actions::action *A, actions::action *A2,
	int f_projective, int f_semilinear, 
	int (*check_function_incremental)(int len,
			long int *S, void *data, int verbose_level),
	void *check_function_incremental_data, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "recoordinatize::init" << endl;
	}

	recoordinatize::SD = SD;

	recoordinatize::Grass = SD->Grass;
	recoordinatize::F = SD->F;
	recoordinatize::q = F->q;
	recoordinatize::k = SD->k;
	recoordinatize::n = SD->n;


	recoordinatize::A = A;
	recoordinatize::A2 = A2;

	recoordinatize::f_projective = f_projective;
	recoordinatize::f_semilinear = f_semilinear;
	recoordinatize::check_function_incremental
		= check_function_incremental;
	recoordinatize::check_function_incremental_data
		= check_function_incremental_data;
	//recoordinatize::fname_live_points.assign(fname_live_points);
	nCkq = Combi.generalized_binomial(n, k, q);

#if 0
	if (f_v) {
		cout << "recoordinatize::init n=" << n << " k=" << k << " fname_live_points=" << fname_live_points << endl;
	}
#endif
	

	M = NEW_int((3 * k) * n);
	M1 = NEW_int((3 * k) * n);
	AA = NEW_int(n * n);
	AAv = NEW_int(n * n);
	TT = NEW_int(k * k);
	TTv = NEW_int(k * k);
	B = NEW_int(n * n);
	C = NEW_int(n * n + 1);
	N = NEW_int((3 * k) * n);
	Elt = NEW_int(A->elt_size_in_int);
	f_data_is_allocated = TRUE;
	if (f_v) {
		cout << "recoordinatize::init done" << endl;
	}
}



void recoordinatize::do_recoordinatize(
		long int i1, long int i2, long int i3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	long int j1, j2, j3;

	if (f_v) {
		cout << "recoordinatize::do_recoordinatize "
				<< i1 << "," << i2 << "," << i3 << endl;
	}
	Grass->unrank_lint_here(M, i1, 0 /*verbose_level - 4*/);
	Grass->unrank_lint_here(M + k * n, i2, 0 /*verbose_level - 4*/);
	Grass->unrank_lint_here(M + 2 * k * n, i3, 0 /*verbose_level - 4*/);
	if (f_vv) {
		cout << "M:" << endl;
		Int_vec_print_integer_matrix_width(cout, M, 3 * k, n, n, F->log10_of_q + 1);
	}
	Int_vec_copy(M, AA, n * n);
	F->Linear_algebra->matrix_inverse(AA, AAv, n, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "AAv:" << endl;
		Int_vec_print_integer_matrix_width(cout, AAv, n, n, n, F->log10_of_q + 1);
	}
	F->Linear_algebra->mult_matrix_matrix(M, AAv, N, 3 * k, n, n,
			0 /* verbose_level */);
	if (f_vv) {
		cout << "N:" << endl;
		Int_vec_print_integer_matrix_width(cout, N, 3 * k, n, n, F->log10_of_q + 1);
	}

	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			TT[i * k + j] = N[2 * k * n + i * n + j];
		}
	}
	if (f_vv) {
		cout << "TT:" << endl;
		Int_vec_print_integer_matrix_width(cout, TT, k, k, k, F->log10_of_q + 1);
	}
	F->Linear_algebra->matrix_inverse(TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "TTv:" << endl;
		Int_vec_print_integer_matrix_width(cout, TTv, k, k, k, F->log10_of_q + 1);
	}

	Int_vec_zero(B, n * n);
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
		Int_vec_print_integer_matrix_width(cout, TT, k, k, k, F->log10_of_q + 1);
	}
	F->Linear_algebra->matrix_inverse(TT, TTv, k, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "TTv:" << endl;
		Int_vec_print_integer_matrix_width(cout, TTv, k, k, k, F->log10_of_q + 1);
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < k; j++) {
			B[(k + i) * n + k + j] = TTv[i * k + j];
		}
	}
	if (f_vv) {
		cout << "B:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				B, n, n, n, F->log10_of_q + 1);
	}

	
	F->Linear_algebra->mult_matrix_matrix(AAv, B, C, n, n, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "C:" << endl;
		Int_vec_print_integer_matrix_width(cout, C, n, n, n, F->log10_of_q + 1);
	}
	
	F->Linear_algebra->mult_matrix_matrix(M, C, M1, 3 * k, n, n, 0 /* verbose_level */);
	if (f_vv) {
		cout << "M1:" << endl;
		Int_vec_print_integer_matrix_width(cout,
				M1, 3 * k, n, n, F->log10_of_q + 1);
	}
	j1 = Grass->rank_lint_here(M1, 0 /*verbose_level - 4*/);
	j2 = Grass->rank_lint_here(M1 + k * n, 0 /*verbose_level - 4*/);
	j3 = Grass->rank_lint_here(M1 + 2 * k * n, 0 /*verbose_level - 4*/);
	if (f_v) {
		cout << "j1=" << j1 << " j2=" << j2 << " j3=" << j3 << endl;
	}
	
	// put a zero, just in case we are in a semilinear group:

	C[n * n] = 0;

	A->make_element(Elt, C, 0);
	if (f_vv) {
		cout << "recoordinatize::do_recoordinatize "
				"transporter:" << endl;
		A->element_print(Elt, cout);
	}
	if (f_v) {
		cout << "recoordinatize::do_recoordinatize done" << endl;
	}
}

void recoordinatize::compute_starter(long int *&S, int &size,
		groups::strong_generators *&Strong_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	
	if (f_v) {
		cout << "recoordinatize::compute_starter" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}

	

	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"before make_first_three" << endl;
	}
	make_first_three(starter_j1, starter_j2, starter_j3, verbose_level - 1);
	if (f_v) {
		cout << "recoordinatize::compute_starter "
				"after make_first_three" << endl;
	}

	// initialize S with the vector (j1,j2,j3):
	size = 3;
	S = NEW_lint(size);

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
	compute_live_points(verbose_level - 10);
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
		groups::strong_generators *&Strong_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_domain D;

	ring_theory::longinteger_object target_go, six, target_go2, go, go_linear;
	ring_theory::longinteger_object go_small;

	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three" << endl;
	}

	A0 = NEW_OBJECT(actions::action);
	A0_linear = NEW_OBJECT(actions::action);
	gens2 = NEW_OBJECT(data_structures_groups::vector_ge);

	data_structures_groups::vector_ge *nice_gens;

	
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before A0->init_projective_group" << endl;
		cout << "recoordinatize::stabilizer_of_first_three "
				"k=" << k << endl;
		cout << "recoordinatize::stabilizer_of_first_three "
				"f_semilinear=" << f_semilinear << endl;
	}
	A0->init_projective_group(k, F, 
		f_semilinear, 
		TRUE /* f_basis */,
		TRUE /* f_init_sims */,
		nice_gens,
		0/*verbose_level - 2*/);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"after A0->init_projective_group" << endl;
	}
	FREE_OBJECT(nice_gens);

	A0->group_order(target_go);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"target_go=" << target_go
			<< " = order of PGGL(" << k << "," << q << ")" << endl;
		cout << "action A0 created: ";
		A0->print_info();
	}

	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before A0_linear->init_projective_group" << endl;
	}
	A0_linear->init_projective_group(k, F, 
		FALSE /*f_semilinear*/, 
		TRUE /*f_basis*/, TRUE /* f_init_sims */,
		nice_gens,
		0/*verbose_level - 2*/);
	FREE_OBJECT(nice_gens);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"after A0_linear->init_projective_group" << endl;
	}

	A0_linear->group_order(go_linear);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"order of PGL(" << k << "," << q << ") is "
			<< go_linear << endl;
		cout << "action A0_linear created: ";
		A0_linear->print_info();
	}



	
	six.create(6, __FILE__, __LINE__);
	D.mult(target_go, six, target_go2);
	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"target_go2=" << target_go2
			<< " = target_go times 6" << endl;
	}
	


	gens2->init(A, verbose_level - 2);


	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before make_generators_stabilizer_of_three_components" << endl;
	}

	actions::action_global AG;

	AG.make_generators_stabilizer_of_three_components(
		A /* A_PGL_n_q */,
		A0 /* A_PGL_k_q */,
		k, gens2, verbose_level - 1);

	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"after make_generators_stabilizer_of_three_components" << endl;
	}


	if (f_v) {
		cout << "recoordinatize::stabilizer_of_first_three "
				"before generators_to_strong_generators" << endl;
	}


	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go2, 
		gens2, Strong_gens, verbose_level - 1);


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


	//string fname;


#if 0
	int f_path_select = TRUE;
	int select_value;
	int len;


	len = i_power_j(Fq->q, Mtx->n) - 1;
	for (select_value = 1; select_value < len;  select_value++) {
		if (f_path_select) {
			snprintf(fname, sizeof(fname), "live_points_%d.txt", select_value);
			}
		else {
			snprintf(fname, sizeof(fname), "live_points.txt");
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
#endif


#if 0
	file_io Fio;

	if (Fio.file_size(fname_live_points) > 1) {
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"reading live points from file " << fname_live_points << endl;
		}
		Fio.read_set_from_file(fname_live_points,
				live_points, nb_live_points, verbose_level);
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"we found " << nb_live_points << " live points" << endl;
		}
		return;
	}
	else {
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"before compute_live_points_low_level" << endl;
		}
		compute_live_points_low_level(
				live_points, nb_live_points, verbose_level - 1);
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"after compute_live_points_low_level" << endl;
		}

		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"before Fio.write_set_to_file" << endl;
		}
		Fio.write_set_to_file(fname_live_points,
				live_points, nb_live_points, verbose_level);
		if (f_v) {
			cout << "recoordinatize::compute_live_points "
					"written file " << fname_live_points << endl;
		}
	}
#else
	if (f_v) {
		cout << "recoordinatize::compute_live_points "
				"before compute_live_points_low_level" << endl;
	}
	compute_live_points_low_level(
			live_points, nb_live_points, verbose_level - 1);
	if (f_v) {
		cout << "recoordinatize::compute_live_points "
				"after compute_live_points_low_level" << endl;
	}
#endif

#if 0

	snprintf(fname, sizeof(fname), "live_points.txt");
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
		long int *&live_points, int &nb_live_points, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	groups::matrix_group *Mtx;
	field_theory::finite_field *Fq;
	long int set[4];
	int *Elt1;
	int *Elt2;
	int f_good;

	int cnt, z;
	int cnt_mod = 1000;
	ring_theory::longinteger_object go_linear;
	long int gos;
	long int i, h, a;
	data_structures::sorting Sorting;


	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level" << endl;
	}


	Mtx = A0->G.matrix_grp;
	Fq = Mtx->GFq;

	set[0] = starter_j1;
	set[1] = starter_j2;
	set[2] = starter_j3;
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);


	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level "
				"nCkq = " << nCkq << endl;
	}

	{
		vector<long int> L;

		nb_live_points = 0;

		A0_linear->group_order(go_linear);
		gos = go_linear.as_lint();

		cnt = 0;
		for (h = 0; h < gos; h++) {
			A0_linear->Sims->element_unrank_lint(h, Elt1);
			if (f_vv) {
				if ((cnt % cnt_mod) == 0 && cnt) {
					cout << "recoordinatize::compute_live_points_low_level"
							<< cnt << " iterations, h=" << h << " found "
							<< nb_live_points << " points so far" << endl;
				}
			}
			Fq->PG_element_normalize(Elt1, 1, k * k);
			if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
				cout << "recoordinatize::compute_live_points_low_level "
						"element " << cnt << " = " << h
						<< ", normalized:" << endl;
				A0->element_print(Elt1, cout);
			}

			for (z = 1; z < q; z++, cnt++) {

				// multiply Elt1 by non-zero scalar z to get Elt2:

				for (i = 0; i < k * k; i++) {
					Elt2[i] = Fq->mult(Elt1[i], z);
				}

				if (f_v && (cnt % cnt_mod) == 0 && cnt) {
					cout << "recoordinatize::compute_live_points_low_level "
							"element " << cnt << " = " << h
							<< ", multiplied by z=" << z << ":" << endl;
					Int_vec_print_integer_matrix_width(cout,
							Elt2, k, k, k, F->log10_of_q + 1);
				}


				// make spread element from Elt2:

				Grass->make_spread_element(Grass->M, Elt2, 0/* verbose_level*/);
					// make the k x n matrix M = ( I_k | Elt2 )


				if (f_vv && (cnt % cnt_mod) == 0) {
					cout << "recoordinatize::compute_live_points_low_level "
							"element " << h << ":" << endl;
					Int_vec_print_integer_matrix_width(cout, Grass->M, k, n, n, 2);
				}
				if (FALSE || ((h & ((1 << 15) - 1)) == 0 && z == 1)) {
					cout << h << " / " << gos
							<< " nb_live_points=" << nb_live_points << endl;
					Int_vec_print_integer_matrix_width(cout, Grass->M, k, n, n, 2);
				}

				a = Grass->rank_lint(0);
				set[3] = a;

				if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
					cout << "has rank " << a << endl;
				}

				// perform test:

				f_good = apply_test(set /* set */, 4 /* sz */, 0 /* verbose_level */);


				if (f_good) {
					if (f_vv && (cnt % cnt_mod) == 0 && cnt) {
						cout << "recoordinatize::compute_live_points_low_level "
								"element " << cnt << " = " << h << ", "
								<< z << " subspace rank " << a
								<< " is accepted as live point no "
								<< nb_live_points << endl;
					}
					L.push_back(a);
					nb_live_points++;
					//live_points[nb_live_points++] = a;
				}
				else {
					if (f_vv && (cnt % cnt_mod) == 0) {
						cout << "recoordinatize::compute_live_points_low_level "
								"element " << cnt << " = " << h << ", "
								<< z << " subspace rank " << a
								<< " is not accepted" << endl;
					}
				}
			}
		}

		if (f_v) {
			cout << "recoordinatize::compute_live_points_low_level "
					"we found " << nb_live_points << " live points" << endl;
		}

		if (f_v) {
			cout << "recoordinatize::compute_live_points_low_level "
					"allocating and copying data" << endl;
		}

		live_points = NEW_lint(nb_live_points);
		for (i = 0; i < nb_live_points; i++) {
			live_points[i] = L[i];
		}
	} // end of L


	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level "
				"sorting" << endl;
	}

	Sorting.lint_vec_heapsort(live_points, nb_live_points);

	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "recoordinatize::compute_live_points_low_level done" << endl;
	}
}

int recoordinatize::apply_test(long int *set, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "recoordinatize::apply_test" << endl;
	}
	int ret;

	ret = (*check_function_incremental)(sz, set,
			check_function_incremental_data,
			0/*verbose_level - 4*/);
	if (f_v) {
		cout << "recoordinatize::apply_test done" << endl;
	}
	return ret;
}

void recoordinatize::make_first_three(
		long int &j1, long int &j2, long int &j3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "recoordinatize::make_first_three" << endl;
	}


	if (f_v) {
		cout << "recoordinatize::make_first_three before Grass->make_special_element_zero" << endl;
	}
	j1 = Grass->make_special_element_zero(0 /* verbose_level */);
	if (f_v) {
		cout << "recoordinatize::make_first_three after Grass->make_special_element_zero" << endl;
	}
	j2 = Grass->make_special_element_infinity(0 /* verbose_level */);
	j3 = Grass->make_special_element_one(0 /* verbose_level */);

	if (f_v) {
		cout << "recoordinatize::make_first_three j1=" << j1 << " j2=" << j2 << " j3=" << j3 << endl;
	}

	if (f_v) {
		cout << "recoordinatize::make_first_three done" << endl;
	}
}

}}}

