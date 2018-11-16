// hill.C
// 
// Anton Betten
// Oct 14, 2010
//
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);
void Hill_cap(int argc, const char **argv, int verbose_level);
void init_orthogonal(action *A, int epsilon, int n, finite_field *F, int verbose_level);
int get_orbit(schreier *Orb, int idx, int *set);
void append_orbit_and_adjust_size(schreier *Orb, int idx, int *set, int &sz);
int test_if_arc(finite_field *Fq, int *pt_coords, int *set, int set_sz, int k, int verbose_level);
void solution(int w, int n, action *A, orthogonal *O, int *coords, int *set, int sz, 
	longinteger_object *Rank_lines, int nb_lines, int verbose_level);



int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}

	Hill_cap(argc, argv, verbose_level);
	
	the_end(t0);
}

void Hill_cap(int argc, const char **argv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int epsilon, n, q, w, i;
	polar P;
	finite_field *F;
	action *A;
	action *An;

	if (f_v) {
		cout << "Hill_cap" << endl;
		}
	epsilon = -1;
	n = 6;
	q = 3;
	w = Witt_index(epsilon, n - 1);
	
	A = NEW_OBJECT(action);
	An = NEW_OBJECT(action);
	
	if (f_v) {
		cout << "Hill_cap before init_orthogonal" << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	init_orthogonal(A, epsilon, n, F, verbose_level - 2);
	action_on_orthogonal *AO;
	orthogonal *O;

	AO = A->G.AO;
	O = AO->O;
	
	if (f_v) {
		cout << "after init_orthogonal" << endl;
		}

	An->init_projective_group(n, F, 
		TRUE /* f_semilinear */, 
		TRUE /* f_basis */, verbose_level - 2);

	if (f_v) {
		cout << "after init_matrix_group" << endl;
		}
	
	if (f_v) {
		cout << "Hill_cap before P.init" << endl;
		}
	P.init(argc, argv, A, O, epsilon, n, w, F, w, verbose_level - 2);
	if (f_v) {
		cout << "Hill_cap before P.init2" << endl;
		}
	P.init2(verbose_level - 2);	
	if (f_v) {
		cout << "Hill_cap before P.compute_orbits" << endl;
		}
	P.compute_orbits(t0, verbose_level - 2);
	
	if (f_v) {
		cout << "we found " << P.nb_orbits << " orbits at depth " << w << endl;
		}
	
	//P.compute_cosets(w, 0, verbose_level);

#if 1

	longinteger_object *Rank_lines;
	int nb_lines;
		
	if (f_v) {
		cout << "Hill_cap before P.dual_polar_graph" << endl;
		}
	P.dual_polar_graph(w, 0, Rank_lines, nb_lines, verbose_level - 2);


	cout << "there are " << nb_lines << " lines" << endl;
	for (i = 0; i < nb_lines; i++) {
		cout << setw(5) << i << " : " << Rank_lines[i] << endl;
		}
	grassmann Grass;
	
	if (f_v) {
		cout << "Hill_cap before Grass.init" << endl;
		}
	Grass.init(n, w, F, 0 /*verbose_level*/);

	cout << "there are " << nb_lines << " lines, generator matrices are:" << endl;
	for (i = 0; i < nb_lines; i++) {
		Grass.unrank_longinteger(Rank_lines[i], 0/*verbose_level - 3*/);
		cout << setw(5) << i << " : " << Rank_lines[i] << ":" << endl;
		print_integer_matrix_width(cout, Grass.M, w, n, n, 2);
		}

#endif



	sims *S;
	longinteger_object go;
	int goi;
	int *Elt;

	Elt = NEW_int(P.A->elt_size_in_int);
	S = P.A->Sims;
	S->group_order(go);	
	cout << "found a group of order " << go << endl;
	goi = go.as_int();

	if (f_v) {
		cout << "Hill_cap finding an element of order 7" << endl;
		}
	S->random_element_of_order(Elt, 7 /* order */, verbose_level);
	cout << "an element of order 7 is:" << endl;
	P.A->element_print_quick(Elt, cout);



	schreier *Orb;
	int N;

	if (f_v) {
		cout << "Hill_cap computing orbits on points" << endl;
		}
	Orb = NEW_OBJECT(schreier);
	Orb->init(P.A);
	Orb->init_single_generator(Elt);
	Orb->compute_all_point_orbits(verbose_level - 2);
	if (f_vv) {
		cout << "Hill_cap the orbits on points are:" << endl;
		Orb->print_and_list_orbits(cout);
		}


	





	int *pt_coords;
	int *Good_orbits;
	int *set;
	int a, nb_pts, j;

	N = Orb->nb_orbits;	
	nb_pts = P.A->degree;
	pt_coords = NEW_int(nb_pts * n);
	set = NEW_int(nb_pts);
	Good_orbits = NEW_int(N);

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


#if 1
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


	FREE_int(Sets);


#if 0

	int *Adj;
	int nb_good_orbits;


	int size;

	nb_good_orbits = 0;
	for (i = 0; i < N; i++) {
		size = get_orbit(Orb, i, set);
		if (test_if_arc(F, pt_coords, set, size, n, verbose_level)) {
			Good_orbits[nb_good_orbits++] = i;
			}
		}
	
	cout << "we found " << nb_good_orbits << " good orbits" << endl;
	int_vec_print(cout, Good_orbits, nb_good_orbits);
	
	Adj = NEW_int(nb_good_orbits * nb_good_orbits);
	for (i = 0; i < nb_good_orbits * nb_good_orbits; i++) {
		Adj[i] = 0;
		}
	for (i = 0; i < nb_good_orbits; i++) {
		for (j = 0; j < nb_good_orbits; j++) {
			size = get_orbit(Orb, Good_orbits[i], set);
			size = get_orbit(Orb, Good_orbits[j], set + size);
			if (test_if_arc(F, pt_coords, set, size, n, verbose_level)) {
				a = 1;
				}
			else {
				a = 0;
				}
			Adj[i * nb_good_orbits + j] = a;
			Adj[j * nb_good_orbits + i] = a;
			}
		}
	cout << "Adj:" << endl;
	print_integer_matrix_width(cout, Adj, nb_good_orbits, nb_good_orbits, nb_good_orbits, 1);


	FREE_int(Elt);
	delete [] Rank_lines;		

#endif


	
	FREE_OBJECT(A);
	
}

void init_orthogonal(action *A, int epsilon, int n, finite_field *F, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	const char *override_poly;
	int p, hh, f_semilinear;
	int f_basis = TRUE;
	int q = F->q;

	if (f_v) {
		cout << "init_orthogonal epsilon=" << epsilon << " n=" << n << " q=" << q << endl;
		}

	is_prime_power(q, p, hh);
	if (hh > 1) {
		f_semilinear = TRUE;
		}
	else {
		f_semilinear = FALSE;
		}

	override_poly = override_polynomial_subfield(q);
	if (f_v && override_poly) {
		cout << "override_poly=" << override_poly << endl;
		}
	if (f_v) {
		cout << "f_semilinear=" << f_semilinear << endl;
		}

	A->init_orthogonal_group(epsilon, 
		n, F, 
		TRUE /* f_on_points */, FALSE /* f_on_lines */, FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, 
		0/*verbose_level*/);

#if 0
	matrix_group *M;
	orthogonal *O;

	M = A->subaction->G.matrix_grp;
	O = M->O;
#endif

	if (f_vv) {
		A->print_base();
		}
	
	
	if (f_v) {
		cout << "init_orthogonal finished, created action:" << endl;
		A->print_info();
		}
}

int get_orbit(schreier *Orb, int idx, int *set)
{
	int f, i, len;

	f = Orb->orbit_first[idx];
	len = Orb->orbit_len[idx];
	for (i = 0; i < len; i++) {
		set[i] = Orb->orbit[f + i];
		}
	return len;
}

void append_orbit_and_adjust_size(schreier *Orb, int idx, int *set, int &sz)
{
	int f, i, len;

	f = Orb->orbit_first[idx];
	len = Orb->orbit_len[idx];
	for (i = 0; i < len; i++) {
		set[sz++] = Orb->orbit[f + i];
		}
}

int test_if_arc(finite_field *Fq, int *pt_coords, int *set, int set_sz, int k, int verbose_level)
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

void solution(int w, int n, action *A, orthogonal *O, int *coords, int *set, int sz, 
	longinteger_object *Rank_lines, int nb_lines, int verbose_level)
{
	int *M;
	int *Weights;
	int *ranks;
	int *PG_ranks;
	int i, j;
	finite_field *F;
	int q;
	int nb_points;

	F = O->F;
	q = F->q;
	nb_points = nb_PG_elements(n - 1, q);
	cout << "nb_points = " << nb_points << endl;
	ranks = NEW_int(sz);
	PG_ranks = NEW_int(sz);

	for (i = 0; i < sz; i++) {
		ranks[i] = O->rank_point(coords + set[i] * n, 1, 0);
		O->F->PG_element_rank_modified(coords + set[i] * n, 1, n, PG_ranks[i]);
		}
	cout << "point ranks: " << endl;
	cout << "i : orthogonal rank : projective rank" << endl;
	for (i = 0; i < sz; i++) {
		cout << setw(3) << i << " : " << setw(6) << ranks[i] << " : " << setw(6) << PG_ranks[i] << endl;
		}
	


	Weights = NEW_int(n + 1);
	M = NEW_int(n * sz);
	for (j = 0; j < sz; j++) {
		for (i = 0; i < n; i++) {
			M[i * sz + j] = coords[set[j] * n + i];
			}
		}
	cout << "generator matrix:" << endl;
	print_integer_matrix_width(cout, M, n, sz, sz, F->log10_of_q);

	cout << "computing the weight enumerator:" << endl;
	F->code_weight_enumerator_fast(sz, n, M, Weights, verbose_level);



	grassmann Grass;
	
	Grass.init(n, w, F, 0 /*verbose_level*/);


	int *line_type;
	int *mtx;
	int h, rk;
	line_type = NEW_int(nb_lines);
	mtx = NEW_int(3 * n);
	
	for (h = 0; h < nb_lines; h++) {
		line_type[h] = 0;
		}
	
	for (h = 0; h < nb_lines; h++) {
		Grass.unrank_longinteger(Rank_lines[h], 0/*verbose_level - 3*/);

		if (FALSE) {
			cout << setw(5) << h << " : " << Rank_lines[h] << ":" << endl;
			print_integer_matrix_width(cout, Grass.M, w, n, n, 2);
			}

		for (j = 0; j < sz; j++) {
			for (i = 0; i < 2 * n; i++) {
				mtx[i] = Grass.M[i];
				}
			for (i = 0; i < n; i++) {
				mtx[2 * n + i] = coords[set[j] * n + i];
				}
			rk = F->Gauss_easy(mtx, 3, n);
			if (rk < 3) {
				line_type[h]++;
				}
			}
		}

	classify C;

	cout << "the types of the " << nb_lines << " lines are:" << endl;
	C.init(line_type, nb_lines, FALSE, verbose_level);
	C.print(FALSE /*f_backwards*/);


	cout << "computing tactical decomposition scheme:" << endl;
	
	int nb_subsets = 2;
	int Sz[2];
	int *Subsets[2];
	//int f_semilinear = TRUE;
	//char *override_poly = NULL;
	//int f_basis = FALSE;
	
	Sz[0] = nb_lines;
	Subsets[0] = NEW_int(nb_lines);
	
	for (i = 0; i < nb_lines; i++) {
		Subsets[0][i] = nb_points + Rank_lines[i].as_int();
		}


	Sz[1] = sz;
	Subsets[1] = PG_ranks;
	
	decomposition_projective_space(n - 1, F, nb_subsets, Sz, Subsets, 
		//f_semilinear, f_basis, 
		verbose_level - 2);

		// in TOP_LEVEL/decomposition.C


	FREE_int(line_type);
	FREE_int(mtx);
	FREE_int(M);
	FREE_int(Weights);
	FREE_int(ranks);
	FREE_int(PG_ranks);
	//FREE_int(tl);
	//delete gens;
	//delete Stab;


	cout << "solution, we are done" << endl;

	exit(1);
}



