// geometric_operations.C
//
// Anton Betten
// November 18, 2014
//
//
// started from stuff that was in TOP_LEVEL/projective_space.C



#include "foundations.h"

void do_Klein_correspondence(INT n, finite_field *F, 
	INT *set_in, INT set_size,
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	projective_space *P5;

	if (f_v) {
		cout << "do_Klein_correspondence" << endl;
		}
	if (n != 3) {
		cout << "do_Klein_correspondence n != 3" << endl;
		exit(1);
		}
	
	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	P5 = new projective_space;
	
	P5->init(5, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	the_set_out = NEW_INT(set_size);
	set_size_out = set_size;
	
	P->klein_correspondence(P5, 
		set_in, set_size, the_set_out, verbose_level);


	delete P;
	delete P5;
}

void do_m_subspace_type(INT n, finite_field *F, INT m, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	projective_space *P;
	INT j, a, N;
	INT d = n + 1;
	INT *v;
	INT *intersection_numbers;
	
	if (f_v) {
		cout << "do_m_subspace_type" << endl;
		cout << "We will now compute the m_subspace type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_m_subspace_type before P->init" << endl;
		}


	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_m_subspace_type after P->init" << endl;
		}

	
	v = NEW_INT(d);

	N = P->nb_rk_k_subspaces_as_INT(m + 1);
	if (f_v) {
		cout << "do_m_subspace_type N = " << N << endl;
		}

	intersection_numbers = NEW_INT(N);
	if (f_v) {
		cout << "after allocating intersection_numbers" << endl;
		}
	if (m == 1) {
		P->line_intersection_type_basic(set, set_size, intersection_numbers, verbose_level - 1);
		}
	else if (m == 2) {
		P->plane_intersection_type_basic(set, set_size, intersection_numbers, verbose_level - 1);
		}
	else if (m == n - 1) {
		P->hyperplane_intersection_type_basic(set, set_size, intersection_numbers, verbose_level - 1);
		}
	else {
		cout << "do_m_subspace_type m=" << m << " not implemented" << endl;
		exit(1);
		}
	
	classify C;
	INT f_second = FALSE;

	C.init(intersection_numbers, N, f_second, 0);
	if (f_v) {
		cout << "do_m_subspace_type: " << m << "-subspace intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}
	
	if (f_show) {
		INT h, f, l, b;
		INT *S;
		//INT *basis;
		grassmann *G;
		
		G = new grassmann;

		G->init(d, m + 1, F, 0 /* verbose_level */);

		//basis = NEW_INT((m + 1) * d);
		S = NEW_INT(N);
		for (h = 0; h < C.nb_types; h++) {
			f = C.type_first[h];
			l = C.type_len[h];
			a = C.data_sorted[f];
			if (f_v) {
				cout << a << "-spaces: ";
				}
			for (j = 0; j < l; j++) {
				b = C.sorting_perm_inv[f + j];
				S[j] = b;
				}
			INT_vec_quicksort_increasingly(S, l);
			if (f_v) {
				INT_vec_print(cout, S, l);
				cout << endl;
				}


			for (j = 0; j < l; j++) {

				INT *intersection_set;
				INT intersection_set_size;
	
				b = S[j];
				G->unrank_INT(b, 0);
				
				cout << "subspace " << j << " / " << l << " which is " << b << " has a basis:" << endl;
				print_integer_matrix_width(cout, G->M, m + 1, d, d, P->F->log10_of_q);


				P->intersection_of_subspace_with_point_set(
					G, b, set, set_size, 
					intersection_set, intersection_set_size, verbose_level);
				cout << "intersection set of size " << intersection_set_size << ":" << endl;
				P->print_set(intersection_set, intersection_set_size);

				FREE_INT(intersection_set);
				}
			}
		FREE_INT(S);
		//FREE_INT(basis);
		delete G;
#if 0
		cout << "i : intersection number of plane i" << endl;
		for (i = 0; i < N_planes; i++) {
			cout << setw(4) << i << " : " << setw(3) << intersection_numbers[i] << endl;
			}
#endif
		}

	FREE_INT(v);
	FREE_INT(intersection_numbers);
	delete P;
}

void do_m_subspace_type_fast(INT n, finite_field *F, INT m, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	projective_space *P;
	grassmann *G;
	INT i;
	INT N;
	INT d = n + 1;
	INT *v;
	longinteger_object *R;
	INT **Pts_on_plane;
	INT *nb_pts_on_plane;
	INT len;
	
	if (f_v) {
		cout << "do_m_subspace_type_fast" << endl;
		cout << "We will now compute the m_subspace type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_m_subspace_type_fast before P->init" << endl;
		}

	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_m_subspace_type_fast after P->init" << endl;
		}

	
	v = NEW_INT(d);

	N = P->nb_rk_k_subspaces_as_INT(m + 1);
	if (f_v) {
		cout << "do_m_subspace_type_fast N = " << N << endl;
		}

	G = new grassmann;

	G->init(n + 1, m + 1, F, 0 /* verbose_level */);

	if (m == 2) {
		P->plane_intersection_type_fast(G, set, set_size, 
			R, Pts_on_plane, nb_pts_on_plane, len, 
			verbose_level - 1);
		}
	else {
		cout << "do_m_subspace_type m=" << m << " not implemented" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "do_m_subspace_type_fast: We found " << len << " planes." << endl;
#if 1
		for (i = 0; i < len; i++) {
			cout << setw(3) << i << " : " << R[i] 
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			INT_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl; 
			}
#endif
		}
	
	classify C;
	INT f_second = FALSE;

	C.init(nb_pts_on_plane, len, f_second, 0);
	if (f_v) {
		cout << "do_m_subspace_type_fast: " << m << "-subspace intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}
	

	// we will now look at the subspaces that intersect in the largest number of points:


	INT *Blocks;
	INT f, a, b, j, nb_planes, intersection_size, u;
	INT *S;
	//INT *basis;
	//grassmann *G;
		
	//G = new grassmann;

	//G->init(d, m + 1, q, F, 0 /* verbose_level */);

	//basis = NEW_INT((m + 1) * d);
	S = NEW_INT(N);
	intersection_size = C.nb_types - 1;

	f = C.type_first[intersection_size];
	nb_planes = C.type_len[intersection_size];
	intersection_size = C.data_sorted[f];
	if (f_v) {
		cout << intersection_size << "-spaces: ";
		}
	for (j = 0; j < nb_planes; j++) {
		b = C.sorting_perm_inv[f + j];
		S[j] = b;
		}
	INT_vec_quicksort_increasingly(S, nb_planes);
	if (f_v) {
		INT_vec_print(cout, S, nb_planes);
		cout << endl;
		}



	Blocks = NEW_INT(nb_planes * intersection_size);


	for (i = 0; i < nb_planes; i++) {

		INT *intersection_set;
	
		b = S[i];
		G->unrank_longinteger(R[b], 0);
				
		cout << "subspace " << i << " / " << nb_planes << " which is " << R[b] << " has a basis:" << endl;
		print_integer_matrix_width(cout, G->M, m + 1, d, d, P->F->log10_of_q);


		P->intersection_of_subspace_with_point_set_rank_is_longinteger(
			G, R[b], set, set_size, 
			intersection_set, u, verbose_level);

		if (u != intersection_size) {
			cout << "u != intersection_size" << endl;
			cout << "u=" << u << endl;
			cout << "intersection_size=" << intersection_size << endl;
			exit(1);
			}
		cout << "intersection set of size " << intersection_size << ":" << endl;
		P->print_set(intersection_set, intersection_size);

		for (j = 0; j < intersection_size; j++) {
			a = intersection_set[j];
			if (!INT_vec_search_linear(set, set_size, a, b)) {
				cout << "did not find point" << endl;
				exit(1);
				}
			Blocks[i * intersection_size + j] = b;
			}
		
		FREE_INT(intersection_set);
		} // next i

	cout << "Blocks:" << endl;
	INT_matrix_print(Blocks, nb_planes, intersection_size);

	INT *Incma;
	INT *ItI;
	INT *IIt;
	INT g;
			
	if (f_v) {
		cout << "Computing plane invariant for " << nb_planes << " planes:" << endl;
		}
	Incma = NEW_INT(set_size * nb_planes);
	ItI = NEW_INT(nb_planes * nb_planes);
	IIt = NEW_INT(set_size * set_size);
	for (i = 0; i < set_size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (u = 0; u < nb_planes; u++) {
		for (g = 0; g < intersection_size; g++) {
			i = Blocks[u * intersection_size + g];
			Incma[i * nb_planes + u] = 1;
			}
		}

	cout << "Incma:" << endl;
	INT_matrix_print(Incma, set_size, nb_planes);

	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < set_size; u++) {
				a += Incma[u * nb_planes + i] * Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}

	cout << "I^t*I:" << endl;
	INT_matrix_print(ItI, nb_planes, nb_planes);

	for (i = 0; i < set_size; i++) {
		for (j = 0; j < set_size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] * Incma[j * nb_planes + u];
				}
			IIt[i * set_size + j] = a;
			}
		}

	cout << "I*I^t:" << endl;
	INT_matrix_print(IIt, set_size, set_size);

	
	FREE_INT(Incma);
	FREE_INT(ItI);
	FREE_INT(IIt);
	FREE_INT(Blocks);
	FREE_INT(S);
	//FREE_INT(basis);
	delete G;


	FREE_INT(v);
	delete P;
}

void do_line_type(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT f_show, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	projective_space *P;
	INT j, a;
	INT *v;
	INT *intersection_numbers;
	
	if (f_v) {
		cout << "do_line_type" << endl;
		cout << "We will now compute the line type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_line_type before P->init" << endl;
		}
	
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_line_type after P->init" << endl;
		}

	
	v = NEW_INT(n + 1);


	intersection_numbers = NEW_INT(P->N_lines);
	if (f_v) {
		cout << "after allocating intersection_numbers" << endl;
		}
	P->line_intersection_type(set, set_size, intersection_numbers, 0 /* verbose_level */);
	
#if 0
	for (i = 0; i < P->N_lines; i++) {
		intersection_numbers[i] = 0;
		}
	for (i = 0; i < set_size; i++) {
		a = set[i];
		for (h = 0; h < P->r; h++) {
			j = P->Lines_on_point[a * P->r + h];
			//if (j == 17) {
			//	cout << "set point " << i << " which is " << a << " lies on line 17" << endl;
			//	}
			intersection_numbers[j]++;
			}
		}
#endif

	classify C;
	INT f_second = FALSE;

	C.init(intersection_numbers, P->N_lines, f_second, 0);
	if (TRUE) {
		cout << "do_line_type: line intersection type: ";
		C.print(TRUE /*f_backwards*/);
		}
	
	if (f_vv) {
		INT h, f, l, b;
		INT *S;
		INT *basis;

		basis = NEW_INT(2 * (P->n + 1));
		S = NEW_INT(P->N_lines);
		for (h = 0; h < C.nb_types; h++) {
			f = C.type_first[h];
			l = C.type_len[h];
			a = C.data_sorted[f];
			if (f_v) {
				cout << a << "-lines: ";
				}
			for (j = 0; j < l; j++) {
				b = C.sorting_perm_inv[f + j];
				S[j] = b;
				}
			INT_vec_quicksort_increasingly(S, l);
			if (f_v) {
				INT_vec_print(cout, S, l);
				cout << endl;
				}
			for (j = 0; j < l; j++) {
				b = S[j];
				P->unrank_line(basis, b);
				if (f_show) {
					cout << "line " << b << " has a basis:" << endl;
					print_integer_matrix_width(cout, basis, 2, P->n + 1, P->n + 1, P->F->log10_of_q);
					}
				INT *L;
				INT *I;
				INT sz;

				if (P->Lines == NULL) {
					continue;
					}
				L = P->Lines + b * P->k;
				INT_vec_intersect(L, P->k, set, set_size, I, sz);

				if (f_show) {
					cout << "intersects in " << sz << " points : ";
					INT_vec_print(cout, I, sz);
					cout << endl;
					cout << "they are:" << endl;
					P->print_set(I, sz);
					}

				FREE_INT(I);
				}
			}
		FREE_INT(S);
		FREE_INT(basis);
#if 0
		cout << "i : intersection number of line i" << endl;
		for (i = 0; i < P->N_lines; i++) {
			cout << setw(4) << i << " : " << setw(3) << intersection_numbers[i] << endl;
			}
#endif
		}

	FREE_INT(v);
	FREE_INT(intersection_numbers);
	delete P;
}

void do_plane_type(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT *&intersection_type, INT &highest_intersection_number, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	grassmann *G;
	projective_space *P;
	
	if (f_v) {
		cout << "do_plane_type" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_plane_type before P->init" << endl;
		}

	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_plane_type after P->init" << endl;
		}

	G = new grassmann;

	G->init(n + 1, 3, F, 0 /*verbose_level - 2*/);

	P->plane_intersection_type(G, 
		set, set_size, 
		intersection_type, highest_intersection_number, 
		verbose_level - 2);

	//FREE_INT(intersection_type);
	delete G;
	delete P;
}

void do_plane_type_failsafe(INT n, finite_field *F, 
	INT *set, INT set_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT N_planes;
	INT *type;
	
	if (f_v) {
		cout << "do_plane_type_failsafe" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_plane_type_failsafe before P->init" << endl;
		}

	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_plane_type_failsafe after P->init" << endl;
		}


	N_planes = P->nb_rk_k_subspaces_as_INT(3);
	type = NEW_INT(N_planes);

	P->plane_intersection_type_basic(set, set_size, 
		type, 
		verbose_level - 2);


	classify C;

	C.init(type, N_planes, FALSE, 0);
	cout << "The plane type is:" << endl;
	C.print(FALSE /*f_backwards*/);

	FREE_INT(type);
	delete P;
	if (f_v) {
		cout << "do_plane_type_failsafe done" << endl;
		}
}

void do_conic_type(INT n, finite_field *F, INT f_randomized, INT nb_times, 
	INT *set, INT set_size, 
	INT *&intersection_type, INT &highest_intersection_number, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT f_save_largest_sets = FALSE;
	set_of_sets *largest_sets = NULL;
	
	if (f_v) {
		cout << "do_conic_type" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_conic_type before P->init" << endl;
		}

	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_conic_type after P->init" << endl;
		}


	P->conic_intersection_type(f_randomized, nb_times, 
		set, set_size, 
		intersection_type, highest_intersection_number, 
		f_save_largest_sets, largest_sets, 
		verbose_level - 2);

	delete P;
}

void do_test_diagonal_line(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	BYTE *fname_orbits_on_quadrangles, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT h;

	if (f_v) {
		cout << "do_test_diagonal_line" << endl;
		cout << "fname_orbits_on_quadrangles=" << fname_orbits_on_quadrangles << endl;
		}
	if (n != 2) {
		cout << "do_test_diagonal_line we need n = 2" << endl;
		exit(1);
		}
	if (ODD(F->q)) {
		cout << "do_test_diagonal_line we need q even" << endl;
		exit(1);
		}
	if (set_size != F->q + 2) {
		cout << "do_test_diagonal_line we need set_size == q + 2" << endl;
		exit(1);
		}
	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);


	INT f_casenumbers = FALSE;
	INT *Casenumbers;
	INT nb_cases;
	//BYTE **data;
	INT **sets;
	INT *set_sizes;
	BYTE **Ago_ascii;
	BYTE **Aut_ascii;

	INT *Nb;

	
#if 0
	read_and_parse_data_file(fname_orbits_on_quadrangles, 
		nb_cases, data, sets, set_sizes);
#endif

	read_and_parse_data_file_fancy(fname_orbits_on_quadrangles, 
		f_casenumbers, 
		nb_cases, 
		set_sizes, sets, Ago_ascii, Aut_ascii, 
		Casenumbers, 
		verbose_level);
		// GALOIS/util.C

	if (f_v) {
		cout << "read " << nb_cases << " orbits on qudrangles" << endl;
		}

	Nb = NEW_INT(nb_cases);

	for (h = 0; h < nb_cases; h++) {


		INT pt[4];
		INT p_idx[4];
		INT line[6];
		INT diag_pts[3];
		INT diag_line;
		INT nb;
		INT i, j, a;
		INT basis[6];


		cout << "orbit " << h << " : ";
		INT_vec_print(cout, sets[h], set_sizes[h]);
		cout << endl;

		if (set_sizes[h] != 4) {
			cout << "size != 4" << endl;
			exit(1);
			}

		for (i = 0; i < 4; i++) {
			a = sets[h][i];
			pt[i] = a;
			if (!INT_vec_search_linear(set_in, set_size, a, j)) {
				cout << "the point " << a << " is not contained in the hyperoval" << endl;
				exit(1);
				}
			p_idx[i] = j;
			}

		cout << "p_idx[4]: ";
		INT_vec_print(cout, p_idx, 4);
		cout << endl;
		
		line[0] = P->line_through_two_points(pt[0], pt[1]);
		line[1] = P->line_through_two_points(pt[0], pt[2]);
		line[2] = P->line_through_two_points(pt[0], pt[3]);
		line[3] = P->line_through_two_points(pt[1], pt[2]);
		line[4] = P->line_through_two_points(pt[1], pt[3]);
		line[5] = P->line_through_two_points(pt[2], pt[3]);

		cout << "line[6]: ";
		INT_vec_print(cout, line, 6);
		cout << endl;
		

		diag_pts[0] = P->line_intersection(line[0], line[5]);
		diag_pts[1] = P->line_intersection(line[1], line[4]);
		diag_pts[2] = P->line_intersection(line[2], line[3]);
	
		cout << "diag_pts[3]: ";
		INT_vec_print(cout, diag_pts, 3);
		cout << endl;
		

		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);	
		cout << "The diagonal line is " << diag_line << endl;

		P->unrank_line(basis, diag_line);
		INT_matrix_print(basis, 2, 3);
		
		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		nb = 0;
		for (i = 0; i < set_size; i++) {
			a = set_in[i];
			if (P->is_incident(a, diag_line)) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		Nb[h] = nb;
		
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			}
		else if (nb == 2) {
			cout << "the diagonal line is secant" << endl;
			}
		else {
			cout << "something else" << endl;
			}


		} // next h
	
	cout << "h : Nb[h]" << endl;
	for (h = 0; h < nb_cases; h++) {
		cout << setw(3) << h << " : " << setw(3) << Nb[h] << endl;
		}
	INT l0, l2;
	INT *V0, *V2;
	INT i, a;

	l0 = 0;
	l2 = 0;
	V0 = NEW_INT(nb_cases);
	V2 = NEW_INT(nb_cases);
	for (i = 0; i < nb_cases; i++) {
		if (Nb[i] == 0) {
			V0[l0++] = i;
			}
		else {
			V2[l2++] = i;
			}
		}
	cout << "external orbits:" << endl;
	for (i = 0; i < l0; i++) {
		a = V0[i];
		cout << i << " : " << a << " : " << Ago_ascii[a] << endl;
		}
	cout << "secant orbits:" << endl;
	for (i = 0; i < l2; i++) {
		a = V2[i];
		cout << i << " : " << a << " : " << Ago_ascii[a] << endl;
		}
	cout << "So, there are " << l0 << " external diagonal orbits  and " << l2 << " secant diagonal orbits" << endl;

	delete P;
}

void do_andre(finite_field *FQ, finite_field *Fq, 
	INT *the_set_in, INT set_size_in, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	projective_space *P2, *P4;
	INT a, a0, a1;
	INT b, b0, b1;
	INT i, h, k, alpha; //, d;
	INT *v, *w1, *w2, *w3, *v2;
	INT *components;
	INT *embedding;
	INT *pair_embedding;

	if (f_v) {
		cout << "do_andre for a set of size " << set_size_in << endl;
		}
	P2 = new projective_space;
	P4 = new projective_space;

	
	P2->init(2, FQ, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P4->init(4, Fq, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	//d = 5;


	if (f_v) {
		cout << "before subfield_embedding_2dimensional" << endl;
		}

	FQ->subfield_embedding_2dimensional(*Fq, 
		components, embedding, pair_embedding, verbose_level);

		// we think of FQ as two dimensional vector space 
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have 
		// pair_embedding[i * q + j] = x;
		// also, 
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element 
		// in FQ that corresponds to i 
		
		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_v) {
		cout << "after  subfield_embedding_2dimensional" << endl;
		}
	if (f_vv) {
		FQ->print_embedding(*Fq, 
			components, embedding, pair_embedding);
		}
	alpha = FQ->p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}

	
	v = NEW_INT(3);
	w1 = NEW_INT(5);
	w2 = NEW_INT(5);
	w3 = NEW_INT(5);
	v2 = NEW_INT(2);


	the_set_out = NEW_INT(P4->N_points);
	set_size_out = 0;
	
	for (i = 0; i < set_size_in; i++) {
		if (f_vv) {
			cout << "input point " << i << " is " << the_set_in[i] << " : ";
			}
		P2->unrank_point(v, the_set_in[i]);
		PG_element_normalize(*FQ, v, 1, 3);
		if (f_vv) {
			INT_vec_print(cout, v, 3);
			cout << " becomes ";
			}

		if (v[2] == 0) {

			// we are dealing with a point on the line at infinity.
			// Such a point corresponds to a line of the spread. 
			// We create the line and then create all q + 1 points on that line.
			
			if (f_vv) {
				cout << endl;
				}
			// w1[4] is the GF(q)-vector corresponding to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2). 
			// That way, w1[4] and w2[4] are a GF(q)-basis for the 
			// 2-dimensional subspace v[2] (when viewed over GF(q)), 
			// which is an element of the regular spread.
						
			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				b = FQ->mult(a, alpha);
				b0 = components[b * 2 + 0];
				b1 = components[b * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				w2[2 * h + 0] = b0;
				w2[2 * h + 1] = b1;
				}
			if (FALSE) {
				cout << "w1=";
				INT_vec_print(cout, w1, 4);
				cout << "w2=";
				INT_vec_print(cout, w2, 4);
				cout << endl;
				}
			
			// now we create all points on the line spanned by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors have a zero in the last spot.
			
			for (h = 0; h < Fq->q + 1; h++) {
				PG_element_unrank_modified(*Fq, v2, 1, 2, h);
				if (FALSE) {
					cout << "v2=";
					INT_vec_print(cout, v2, 2);
					cout << " : ";
					}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]), Fq->mult(v2[1], w2[k]));
					}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					INT_vec_print(cout, w3, 5);
					}
				a = P4->rank_point(w3);
				if (f_vv) {
					cout << " rank " << a << endl;
					}
				the_set_out[set_size_out++] = a;
				}
			}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector has a one in the last spot.


			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				}
			w1[4] = 1;
			if (f_vv) {
				//cout << "w1=";
				INT_vec_print(cout, w1, 5);
				}
			a = P4->rank_point(w1);
			if (f_vv) {
				cout << " rank " << a << endl;
				}
			the_set_out[set_size_out++] = a;
			}
		}

	if (f_v) {
		for (i = 0; i < set_size_out; i++) {
			a = the_set_out[i];
			P4->unrank_point(w1, a);
			cout << setw(3) << i << " : " << setw(5) << a << " : ";
			INT_vec_print(cout, w1, 5);
			cout << endl;
			}
		}

	delete P2;
	delete P4;
	FREE_INT(v);
	FREE_INT(w1);
	FREE_INT(w2);
	FREE_INT(w3);
	FREE_INT(v2);
	FREE_INT(components);
	FREE_INT(embedding);
	FREE_INT(pair_embedding);
}

void do_print_lines_in_PG(INT n, finite_field *F, 
	INT *set_in, INT set_size)
{
	projective_space *P;
	INT d = n + 1;
	INT h, a;
	INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print = "\\alpha";

	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P->Grass_lines->unrank_INT(a, 0 /* verbose_level */);
		cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
		F->latex_matrix(cout, f_elements_exponential, 
			symbol_for_print, P->Grass_lines->M, 2, d);
		cout << endl;
		}
	delete P;
}

void do_print_points_in_PG(INT n, finite_field *F, 
	INT *set_in, INT set_size)
{
	projective_space *P;
	INT d = n + 1;
	INT h, a;
	//INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print = "\\alpha";
	INT *v;

	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	v = NEW_INT(d);
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P->unrank_point(v, a);
		cout << setw(5) << h << " : " << setw(5) << a << " : ";
		INT_vec_print(cout, v, d);
		cout << " : ";
		F->INT_vec_print_elements_exponential(cout, v, d, symbol_for_print);
		cout << endl;
		}
	FREE_INT(v);
	delete P;
}

void do_print_points_in_orthogonal_space(INT epsilon, INT n, finite_field *F, 
	INT *set_in, INT set_size, INT verbose_level)
{
	INT d = n + 1;
	INT h, a;
	//INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print = "\\alpha";
	INT *v;
	orthogonal *O;

	O = new orthogonal;
	
	O->init(epsilon, d, F, verbose_level - 1);
	F = O->F;

	v = NEW_INT(d);
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		O->unrank_point(v, 1, a, 0);
		//cout << setw(5) << h << " : ";
		cout << setw(5) << a << " & ";
		if (F->e > 1) {
			F->INT_vec_print_elements_exponential(cout, v, d, symbol_for_print);
			}
		else {
			INT_vec_print(cout, v, d);
			}
		cout << "\\\\" << endl;
		}
	FREE_INT(v);
	delete O;
}

void do_print_points_on_grassmannian(INT n, INT k, finite_field *F, 
	INT *set_in, INT set_size)
{
	grassmann *Grass;
	projective_space *P;
	INT d = n + 1;
	INT h, a;
	INT f_elements_exponential = TRUE;
	const BYTE *symbol_for_print = "\\alpha";

	P = new projective_space;
	Grass = new grassmann;
	
	//N = generalized_binomial(n + 1, k + 1, q);
	//r = generalized_binomial(k + 1, 1, q);
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);
	Grass->init(n + 1, k + 1, F, 0);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		//cout << "unrank " << a << endl;
		Grass->unrank_INT(a, 0 /* verbose_level */);
		cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
		F->latex_matrix(cout, f_elements_exponential, 
			symbol_for_print, Grass->M, k + 1, d);
		cout << endl;
		}
	delete P;
	delete Grass;
}

void do_embed_orthogonal(INT epsilon, INT n, finite_field *F, 
	INT *set_in, INT *&set_out, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	INT *v;
	INT d = n + 1;
	INT h, a, b;
	INT c1 = 0, c2 = 0, c3 = 0;

	if (f_v) {
		cout << "do_embed_orthogonal" << endl;
		}
	P = new projective_space;
	
	P->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	if (epsilon == -1) {
		choose_anisotropic_form(*F, c1, c2, c3, verbose_level);
		}

	v = NEW_INT(d);
	set_out = NEW_INT(set_size);
	
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		Q_epsilon_unrank(*F, v, 1, epsilon, n, c1, c2, c3, a);
		b = P->rank_point(v);
		set_out[h] = b;
		}

	FREE_INT(v);
	delete P;

}

void do_embed_points(INT n, finite_field *F, 
	INT *set_in, INT *&set_out, INT set_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	INT *v;
	INT d = n + 2;
	INT h, a, b;

	if (f_v) {
		cout << "do_embed_points" << endl;
		}
	P1 = new projective_space;
	P2 = new projective_space;
	
	P1->init(n, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(n + 1, F, 
		FALSE /* f_init_incidence_structure */, 
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_INT(d);
	set_out = NEW_INT(set_size);
	
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P1->unrank_point(v, a);
		v[d - 1] = 0;
		b = P2->rank_point(v);
		set_out[h] = b;
		}

	FREE_INT(v);
	delete P1;
	delete P2;

}

void do_draw_points_in_plane(finite_field *F, 
	INT *set, INT set_size, 
	const BYTE *fname_base, INT f_point_labels, INT f_embedded, INT f_sideways, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	projective_space *P;
	INT n = 2;
	INT rad = 17000;
	
	if (f_v) {
		cout << "do_draw_points_in_plane" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_draw_points_in_plane before P->init" << endl;
		}
	
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_draw_points_in_plane after P->init" << endl;
		}

	P->draw_point_set_in_plane(fname_base, set, set_size, TRUE /*f_with_points*/, f_point_labels, f_embedded, f_sideways, rad, 0 /* verbose_level */);
	delete P;

	if (f_v) {
		cout << "do_draw_points_in_plane done" << endl;
		}

}
	
void do_ideal(INT n, finite_field *F, 
	INT *set_in, INT set_size, INT degree, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	homogeneous_polynomial_domain *HPD;
	INT *Kernel;
	INT *w1;
	INT *w2;
	INT *Pts;
	INT nb_pts;
	INT r, h, ns, N;

	if (f_v) {
		cout << "do_ideal" << endl;
		}

	HPD = new homogeneous_polynomial_domain;

	HPD->init(F, n + 1, degree, FALSE /* f_init_incidence_structure */, verbose_level);

	Kernel = NEW_INT(HPD->nb_monomials * HPD->nb_monomials);
	w1 = NEW_INT(HPD->nb_monomials);
	w2 = NEW_INT(HPD->nb_monomials);
	
	HPD->vanishing_ideal(set_in, set_size, r, Kernel, 0 /*verbose_level */);

	ns = HPD->nb_monomials - r; // dimension of null space
	cout << "The system has rank " << r << endl;
	cout << "The ideal has dimension " << ns << endl;
	cout << "and is generated by:" << endl;
	INT_matrix_print(Kernel, ns, HPD->nb_monomials);
	cout << "corresponding to the following basis of polynomials:" << endl;
	for (h = 0; h < ns; h++) {
		HPD->print_equation(cout, Kernel + h * HPD->nb_monomials);
		cout << endl;
		}
	
	Pts = NEW_INT(HPD->P->N_points);
	cout << "looping over all generators of the ideal:" << endl;
	for (h = 0; h < ns; h++) {
		cout << "generator " << h << " / " << ns << ":" << endl;
		HPD->enumerate_points(Kernel + h * HPD->nb_monomials, Pts, nb_pts, verbose_level);
		cout << "We found " << nb_pts << " points on the curve" << endl;
		cout << "They are : ";
		INT_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(Pts, nb_pts);
		}
	cout << "looping over all elements of the ideal:" << endl;
	N = nb_PG_elements(ns - 1, F->q);
	for (h = 0; h < N; h++) {
		PG_element_unrank_modified(*F, w1, 1, ns, h);
		cout << "element " << h << " / " << N << " w1=";
		INT_vec_print(cout, w1, ns);
		F->mult_vector_from_the_left(w1, Kernel, w2, ns, HPD->nb_monomials);
		cout << " w2=";
		INT_vec_print(cout, w2, HPD->nb_monomials);
		HPD->enumerate_points(w2, Pts, nb_pts, verbose_level);
		cout << " We found " << nb_pts << " points on this curve" << endl;
		}

	delete HPD;
	FREE_INT(Kernel);
	FREE_INT(Pts);
	FREE_INT(w1);
	FREE_INT(w2);
}



#if 0
void do_move_line_in_PG(INT n, finite_field *F, 
	INT from_line, INT to_line, 
	INT *the_set_in, INT set_size_in, 
	INT *&the_set_out, INT &set_size_out, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	projective_space *P;
	action *A;
	action *A_pts;
	INT i, a, b;

	if (f_v) {
		cout << "do_move_line_in_PG" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_move_line_in_PG before P->init" << endl;
		}

	P->init(n, F, TRUE /* f_init_group */, 
		TRUE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		TRUE /* f_semilinear */, 
		TRUE /* f_basis */,
		0 /* verbose_level - 2 */);
	A = P->A2;
	A_pts = P->A;

	if (f_v) {
		cout << "do_move_line_in_PG after P->init" << endl;
		}

	schreier *Sch;
	INT *Elt1;
	INT len, pos;

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Sch = new schreier;
	Sch->init(A);
	Sch->init_generators(*P->A2->Strong_gens->gens);
	Sch->initialize_tables();
	Sch->compute_point_orbit(from_line, 0 /*verbose_level*/);
	len = Sch->orbit_len[0];
	if (f_v) {
		cout << "from_line " << from_line << " lies in an orbit of length " << len << endl;
		}
	pos = Sch->orbit_inv[to_line];
	if (pos >= len) {
		cout << "to_line " << to_line << " does not lie in the same orbit as from_line " << from_line << endl;
		exit(1);
		}
	Sch->coset_rep(pos);
	A->element_move(Sch->cosetrep, Elt1, 0);
	cout << "coset rep is:" << endl;
	A->element_print_quick(Elt1, cout);
	cout << endl;
	A->element_print_as_permutation(Elt1, cout);
	cout << endl;
	a = A->element_image_of(from_line, Elt1, 0);
	cout << "image of " << from_line << " is " << a << endl;
	if (a != to_line) {
		cout << "image of from_line is not to_line" << endl;
		exit(1);
		}
	set_size_out = set_size_in;
	the_set_out = NEW_INT(set_size_out);
	for (i = 0; i < set_size_in; i++) {
		a = the_set_in[i];
		b = A_pts->element_image_of(a, Elt1, 0);
		the_set_out[i] = b;
		}
	delete Sch;
	FREE_INT(Elt1);
	delete P;
}

void do_group_in_PG(INT n, finite_field *F, 
	INT *the_set_in, INT set_size_in, INT f_list_group_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	projective_space *P;
	//INT f_elements_exponential = TRUE;
	//const BYTE *symbol_for_print = "\\alpha";
	INT f_semilinear = TRUE;

	if (f_v) {
		cout << "do_group_in_PG computing group" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_group_in_PG before P->init" << endl;
		}
	if (is_prime(F->q)) {
		f_semilinear = FALSE;
		}
	P->init(n, F, TRUE /* f_init_group */, 
		FALSE /* f_line_action */, 
		FALSE /* f_init_incidence_structure */, 
		f_semilinear, 
		TRUE /* f_basis */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_group_in_PG after P->init" << endl;
		}


#if 0
	v = NEW_INT(d);
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P->unrank_point(v, a);
		cout << setw(5) << h << " : " << setw(5) << a << " : ";
		F->INT_vec_print_elements_exponential(cout, v, d, symbol_for_print);
		cout << endl;
		}
	FREE_INT(v);
#endif
	sims *S;
	longinteger_object ago;
		
	if (f_v) {
		cout << "do_group_in_PG before projective_space_set_stabilizer" << endl;
		}
	S = P->set_stabilizer(the_set_in, set_size_in, verbose_level - 1);


	if (f_v) {
		cout << "do_group_in_PG after projective_space_set_stabilizer" << endl;
		}

	S->group_order(ago);
	cout << "Found a stabilizer of order " << ago << endl;
	if (f_v) {
		cout << "strong generators are (in tex):" << endl;
		S->print_generators_tex(cout);
		cout << "strong generators are:" << endl;
		S->print_generators();
		cout << "Found a stabilizer of order " << ago << endl;
		}


	if (f_list_group_elements) {
		cout << "listing all group elements" << endl;
		S->print_all_group_elements();
		}

	delete S;
	delete P;
}

#endif


#if 0
void do_find_Eckardt_points_from_arc(INT n, finite_field *F, 
	INT *set_in, INT set_size, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "do_find_Eckardt_points_from_arc" << endl;
		}


#if 0
	{
	INT rk;
	
	rk = unordered_triple_pair_rank(2, 6, 3, 4, 1, 5);

	cout << "rk=" << rk << endl;
	exit(1);
	}
#endif


	projective_space *P;
	if (n != 2) {
		cout << "do_find_Eckardt_points_from_arc need n = 2" << endl;
		exit(1);
		}
	if (set_size != 6) {
		cout << "do_find_Eckardt_points_from_arc need set_size = 6" << endl;
		exit(1);
		}
	
	if (f_v) {
		cout << "do_find_Eckardt_points_from_arc" << endl;
		}

	P = new projective_space;
	
	if (f_v) {
		cout << "do_find_Eckardt_points_from_arc before P->init" << endl;
		}
	
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "do_find_Eckardt_points_from_arc after P->init" << endl;
		}

	surface *S;


	S = new surface;
	S->init(F, verbose_level);


	eckardt_point *E;
	INT *E_idx;
	INT nb_E;
	INT s, i;
	
	P->find_Eckardt_points_from_arc_not_on_conic(set_in, E, nb_E, verbose_level);
	for (s = 0; s < nb_E; s++) {
		cout << s << " / " << nb_E << " : ";
		E[s].print();
		cout << endl;
		}

	cout << "We found " << nb_E << " Eckardt points" << endl;

	E_idx = NEW_INT(nb_E);
	for (s = 0; s < nb_E; s++) {
		E_idx[s] = E[s].rank();
		}
	cout << "by rank: ";
	INT_vec_print(cout, E_idx, nb_E);
	cout << endl;

	
	

	INT *T_idx;
	INT nb_T;
	
	S->find_trihedral_pairs_from_collinear_triples_of_Eckardt_points(E_idx, nb_E, 
		T_idx, nb_T, verbose_level);
	
	INT t_idx;

	if (nb_T == 0) {
		cout << "nb_T == 0" << endl;	
		exit(1);
		}



	INT *Planes;
	INT *The_equation;
	INT *Lambda;
	INT *Transformation;
	INT *Nb_maps;
	INT nb_maps;
	
	Planes = NEW_INT(nb_T * 6 * 4);
	The_equation = NEW_INT(nb_T * 20);
	Lambda = NEW_INT(nb_T);
	Nb_maps = NEW_INT(nb_T);
	Transformation = NEW_INT(4 * 4);
	
	for (i = 0; i < nb_T; i++) {
		t_idx = T_idx[i];
		cout << i << " / " << nb_T << " T_{" << S->Trihedral_pair_labels[t_idx] << "}" << endl;
		
		S->create_surface_and_planes_from_trihedral_pair_and_arc(set_in /* arc6 */, t_idx, 
			Planes + i * 6 * 4, The_equation + i * 20, Lambda[i], 0 /*verbose_level*/);

		cout << "planes:" << endl;
		INT_matrix_print(Planes + i * 6 * 4, 6, 4);
		cout << "lambda = " << Lambda[i] << endl;
		cout << "equation:";
		INT_vec_print(cout, The_equation + i * 20, 20);
		cout << endl;
		}



	for (i = 0; i < nb_T; i++) {

		cout << "testing trihedral pair " << i << " / " << nb_T << ":" << endl;

		F->map_points_to_points_projectively(4 /* d */, 1 /* k */, Planes, Planes + i * 6 * 4, Transformation, nb_maps, verbose_level);
		Nb_maps[i] = nb_maps;
		cout << "testing trihedral pair " << i << " / " << nb_T << " done, we found " << nb_maps << " mappings" << endl;
		}


	delete [] E;
	FREE_INT(E_idx);
	FREE_INT(T_idx);
	FREE_INT(Planes);
	FREE_INT(The_equation);
	FREE_INT(Lambda);
	FREE_INT(Nb_maps);
	FREE_INT(Transformation);

	



	delete S;
	delete P;
	if (f_v) {
		cout << "do_find_Eckardt_points_from_arc done" << endl;
		}

}
#endif




