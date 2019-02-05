// projective_space.C
// 
// Anton Betten
//
// started March 14, 2012
//
//
// Hill cap and Buekenhout Metz unitals
//
//

#include "orbiter.h"

namespace orbiter {
namespace top_level {


void Hill_cap56(int argc, const char **argv, 
	char *fname, int &nb_Pts, int *&Pts, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int epsilon, n, q, w, i;
	polar *P;
	action *A;
	action *An;
	finite_field *F;

	if (f_v) {
		cout << "Hill_cap" << endl;
		}
	epsilon = -1;
	n = 6;
	q = 3;
	w = Witt_index(epsilon, n - 1);
	
	P = NEW_OBJECT(polar);
	A = NEW_OBJECT(action);
	An = NEW_OBJECT(action);
	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	if (f_v) {
		cout << "Hill_cap before init_orthogonal" << endl;
		}

	int f_semilinear;
	
	if (is_prime(F->q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

	if (f_v) {
		cout << "f_semilinear=" << f_semilinear << endl;
		}

	A->init_orthogonal_group(epsilon, 
		n, F, 
		TRUE /* f_on_points */, FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, TRUE /* f_basis */, 
		0/*verbose_level*/);




	if (f_v) {
		cout << "Hill_cap created action:" << endl;
		A->print_info();
		}


	action_on_orthogonal *AO = A->G.AO;
	orthogonal *O;

	O = AO->O;
	
	if (f_v) {
		cout << "after init_orthogonal" << endl;
		}
	vector_ge *nice_gens;

	An->init_projective_group(n, F, TRUE /* f_semilinear */, 
		TRUE /* f_basis */,
		nice_gens,
		verbose_level - 2);

	FREE_OBJECT(nice_gens);

	if (f_v) {
		cout << "after init_projective_group" << endl;
		}
	
	if (f_v) {
		cout << "Hill_cap before P.init" << endl;
		}
	P->init(argc, argv, A, O, epsilon, n, w, F, w, verbose_level - 2);
	if (f_v) {
		cout << "Hill_cap before P.init2" << endl;
		}
	P->init2(verbose_level - 2);	
	if (f_v) {
		cout << "Hill_cap before P.compute_orbits" << endl;
		}
	int t0 = os_ticks();
	P->compute_orbits(t0, verbose_level - 2);
	
	if (f_v) {
		cout << "we found " << P->nb_orbits
				<< " orbits at depth " << w << endl;
		}
	
	//P.compute_cosets(w, 0, verbose_level);

#if 1

	longinteger_object *Rank_lines;
	int nb_lines;
		
	if (f_v) {
		cout << "Hill_cap before P.dual_polar_graph" << endl;
		}
	P->dual_polar_graph(w, 0, Rank_lines, nb_lines, verbose_level - 2);


	cout << "there are " << nb_lines << " lines" << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << setw(5) << i << " : " << Rank_lines[i] << endl;
		}
	grassmann Grass;
	
	if (f_v) {
		cout << "Hill_cap before Grass.init" << endl;
		}
	Grass.init(n, w, F, 0 /*verbose_level*/);

	cout << "there are " << nb_lines
			<< " lines, generator matrices are:" << endl;
	for (i = 0; i < nb_lines; i++) {
		Grass.unrank_longinteger(Rank_lines[i], 0/*verbose_level - 3*/);
		cout << setw(5) << i << " : " << Rank_lines[i] << ":" << endl;
		print_integer_matrix_width(cout, Grass.M, w, n, n, 2);
		}

#endif



	sims *S;
	longinteger_object go;
	//int goi;
	int *Elt;

	Elt = NEW_int(P->A->elt_size_in_int);
	S = P->A->Sims;
	S->group_order(go);	
	cout << "found a group of order " << go << endl;
	//goi = go.as_int();

	if (f_v) {
		cout << "Hill_cap finding an element of order 7" << endl;
		}
	S->random_element_of_order(Elt, 7 /* order */, verbose_level);
	cout << "an element of order 7 is:" << endl;
	P->A->element_print_quick(Elt, cout);



	schreier *Orb;
	int N;

	if (f_v) {
		cout << "Hill_cap computing orbits on points" << endl;
		}
	Orb = NEW_OBJECT(schreier);
	Orb->init(P->A);
	Orb->init_single_generator(Elt);
	Orb->compute_all_point_orbits(verbose_level - 2);
	if (f_vv) {
		cout << "Hill_cap the orbits on points are:" << endl;
		Orb->print_and_list_orbits(cout);
		}


	





	int *pt_coords;
	//int *Good_orbits;
	int *set;
	int a, nb_pts, j;

	N = Orb->nb_orbits;	
	nb_pts = P->A->degree;
	pt_coords = NEW_int(nb_pts * n);
	set = NEW_int(nb_pts);
	//Good_orbits = NEW_int(N);

	for (i = 0; i < nb_pts; i++) {
		O->unrank_point(pt_coords + i * n, 1, i, 0);
		}
	cout << "point coordinates:" << endl;
	print_integer_matrix_width(cout, pt_coords, nb_pts, n, n, 2);
	
	cout << "evaluating quadratic form:" << endl;
	for (i = 0; i < nb_pts; i++) {
		a = O->evaluate_quadratic_form(pt_coords + i * n, 1);
		cout << setw(3) << i << " : " << a << endl;
		}
	int sz[9];
	int i1, i2, i3, i4, i5, i6, i7, i8, ii;
	int nb_sol;

	int *Sets; // [max_sol * 56]
	int max_sol = 100;
	
	Sets = NEW_int(max_sol * 56);
	
	sz[0] = 0;
	nb_sol = 0;
	for (i1 = 0; i1 < N; i1++) {
		sz[1] = sz[0];
		append_orbit_and_adjust_size(Orb, i1, set, sz[1]);
		//cout << "after append_orbit_and_adjust_size :";
		//int_vec_print(cout, set, sz[1]);
		//cout << endl;
		if (!test_if_arc(F, pt_coords, set, sz[1], n, verbose_level)) {
			continue;
			}
		for (i2 = i1 + 1; i2 < N; i2++) {
			sz[2] = sz[1];
			append_orbit_and_adjust_size(Orb, i2, set, sz[2]);
			if (!test_if_arc(F, pt_coords, set, sz[2], n, verbose_level)) {
				continue;
				}
			for (i3 = i2 + 1; i3 < N; i3++) {
				sz[3] = sz[2];
				append_orbit_and_adjust_size(Orb, i3, set, sz[3]);
				if (!test_if_arc(F, pt_coords, set, sz[3], n, verbose_level)) {
					continue;
					}
				for (i4 = i3 + 1; i4 < N; i4++) {
					sz[4] = sz[3];
					append_orbit_and_adjust_size(Orb, i4, set, sz[4]);
					if (!test_if_arc(F, pt_coords, set, sz[4], n, verbose_level)) {
						continue;
						}
					for (i5 = i4 + 1; i5 < N; i5++) {
						sz[5] = sz[4];
						append_orbit_and_adjust_size(Orb, i5, set, sz[5]);
						if (!test_if_arc(F, pt_coords, set, sz[5], n, verbose_level)) {
							continue;
							}
						for (i6 = i5 + 1; i6 < N; i6++) {
							sz[6] = sz[5];
							append_orbit_and_adjust_size(Orb, i6, set, sz[6]);
							if (!test_if_arc(F, pt_coords, set, sz[6], n, verbose_level)) {
								continue;
								}
							for (i7 = i6 + 1; i7 < N; i7++) {
								sz[7] = sz[6];
								append_orbit_and_adjust_size(Orb, i7, set, sz[7]);
								if (!test_if_arc(F, pt_coords, set, sz[7], n, verbose_level)) {
									continue;
									}
								for (i8 = i7 + 1; i8 < N; i8++) {
									sz[8] = sz[7];
									append_orbit_and_adjust_size(Orb, i8, set, sz[8]);
									if (!test_if_arc(F, pt_coords, set, sz[8], n, verbose_level)) {
										continue;
										}

									if (sz[8] != 56) {
										cout << "error, the size of the arc is not 56" << endl;
										exit(1);
										}
									for (ii = 0; ii < sz[8]; ii++) {
										int rk;
										O->F->PG_element_rank_modified(pt_coords + set[ii] * n, 1, n, rk);
										Sets[nb_sol * 56 + ii] = rk;
										}

									nb_sol++;
									cout << "solution " << nb_sol << ", a set of size " << sz[8] << " : ";
									cout << i1 << "," << i2 << "," << i3 << "," << i4 << "," << i5 << "," << i6 << "," << i7 << "," << i8 << endl;
									int_vec_print(cout, set, sz[8]);
									cout << endl;


#if 0
									solution(w, n, A, O, pt_coords, 
										set, sz[8], Rank_lines, nb_lines, verbose_level);
									cout << endl;
#endif

									} // next i8
								} // next i7
							} // next i6
						} // next i5
					} // next i4
				} // next i3
			} // next i2
		} // next i1
	cout << "there are " << nb_sol << " solutions" << endl;
	cout << "out of " << int_n_choose_k(N, 8) << " possibilities" << endl;


	for (i = 0; i < nb_sol; i++) {
		cout << "Solution " << i << ":" << endl;
		for (j = 0; j < 56; j++) {
			cout << Sets[i * 56 + j] << " ";
			}
		cout << endl;
		}

	if (nb_sol == 0) {
		cout << "error, no solution" << endl;
		exit(1);
		}

	nb_Pts = 56;
	Pts = NEW_int(56);
	for (j = 0; j < 56; j++) {
		Pts[j] = Sets[0 * 56 + j];
		}
	sprintf(fname, "Hill_cap_56.txt");

	FREE_int(Sets);

	delete P;
	delete A;
	delete An;
	delete F;
	
}

void append_orbit_and_adjust_size(schreier *Orb,
		int idx, int *set, int &sz)
// Used by Hill_cap56()
{
	int f, i, len;

	f = Orb->orbit_first[idx];
	len = Orb->orbit_len[idx];
	for (i = 0; i < len; i++) {
		set[sz++] = Orb->orbit[f + i];
		}
}



int test_if_arc(finite_field *Fq, int *pt_coords,
		int *set, int set_sz, int k, int verbose_level)
// Used by Hill_cap56()
{
	int f_v = FALSE; //(verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int subset[3];
	int subset1[3];
	int *Mtx;
	int ret = FALSE;
	int i, j, a, rk;


	if (f_v) {
		cout << "test_if_arc testing set" << endl;
		int_vec_print(cout, set, set_sz);
		cout << endl;
		}
	Mtx = NEW_int(3 * k);
	
	first_k_subset(subset, set_sz, 3);
	while (TRUE) {
		for (i = 0; i < 3; i++) {
			subset1[i] = set[subset[i]];
			}
		int_vec_sort(3, subset1);
		if (f_vv) {
			cout << "testing subset ";
			int_vec_print(cout, subset1, 3);
			cout << endl;
			}
				
		for (i = 0; i < 3; i++) {
			a = subset1[i];
			for (j = 0; j < k; j++) {
				Mtx[i * k + j] = pt_coords[a * k + j];
				}
			}
		if (f_vv) {
			cout << "matrix:" << endl;
			print_integer_matrix_width(cout, Mtx, 3, k, k, 1);
			}
		rk = Fq->Gauss_easy(Mtx, 3, k);
		if (rk < 3) {
			if (f_v) {
				cout << "not an arc" << endl;
				}
			goto done;
			}
		if (!next_k_subset(subset, set_sz, 3)) {
			break;
			}
		}
	if (f_v) {
		cout << "passes the arc test" << endl;
		}
	ret = TRUE;
done:
	
	FREE_int(Mtx);
	return ret;
}

void create_Buekenhout_Metz(
	finite_field *Fq, finite_field *FQ, 
	int f_classical, int f_Uab, int parameter_a, int parameter_b, 
	char *fname, int &nb_pts, int *&Pts, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, rk, d = 3;
	int v[3];
	buekenhout_metz *BM;
		// in TOP_LEVEL/buekenhout_metz.C

	if (f_v) {
		cout << "create_Buekenhout_Metz" << endl;
		}

	
	BM = NEW_OBJECT(buekenhout_metz);

	BM->init(Fq, FQ, 
		f_Uab, parameter_a, parameter_b, f_classical, verbose_level);
	

	if (BM->f_Uab) {
		BM->init_ovoid_Uab_even(
				BM->parameter_a, BM->parameter_b,
				verbose_level);
		}
	else {
		BM->init_ovoid(verbose_level);
		}

	BM->create_unital(verbose_level);

	//BM->write_unital_to_file();

	nb_pts = BM->sz;
	Pts = NEW_int(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = BM->U[i];
		}


	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];
		BM->P2->unrank_point(v, rk);
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
			}
		}



	strcpy(fname, "unital_");
	BM->get_name(fname + strlen(fname));
	strcat(fname, ".txt");

	FREE_OBJECT(BM);

}

}}




