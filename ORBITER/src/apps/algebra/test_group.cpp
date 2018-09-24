// test_group.C
// 
// Anton Betten
// 1/2/2008
//
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

void test_group(int argc, const char **argv, int f_full, int verbose_level);
void test1(int verbose_level);
void test2(int verbose_level);
void test3(int verbose_level);
void test4(int verbose_level);
void test5(int verbose_level);
void test6(int verbose_level);
void O4_grid_action(action *A_PGL4, int *Elt, int verbose_level);
void test7(int verbose_level);
void test8(int verbose_level);
void test9(int verbose_level);
void test10(int verbose_level);
void test11(int verbose_level);
void test12(int verbose_level);
void test13(int verbose_level);
void test14(int verbose_level);
void test15(int verbose_level);
void test16(int verbose_level);
void test17(int verbose_level);
void print_fancy(action *A, finite_field *Fq, int *Elt, int n);
void test18(int verbose_level);
void test19(int verbose_level);
void test20(int verbose_level);
void instant_insanity(int verbose_level);
void apply6(action *A, int *Elt, int *src, int *image);
int test_sides(int *v1, int *v2);
void test_bricks(int verbose_level);
void test_null_polarity_generator(int verbose_level);
void test_linear_group_generator(int verbose_level);
void test_homogeneous_polynomials(int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_group = FALSE;
	int group_argc;
	char **group_argv;
	int f_full = FALSE;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-group") == 0) {
			f_group = TRUE;
			group_argv = argv + (i + 1);
			group_argc = argc - i - 1;
			cout << "-group " << endl;
			}
		else if (strcmp(argv[i], "-full") == 0) {
			f_full = TRUE;
			cout << "-full " << endl;
			}
		}
	
	if (f_group) {
		test_group(group_argc, (const char **)group_argv, f_full, verbose_level);
		}
	//test_vector(3, 2, FALSE, 2);
	test1(verbose_level);
	//test2(verbose_level);
	//test3(verbose_level);
	//test4(verbose_level);
	//test5(verbose_level);
	//test6(verbose_level);
	//test7(verbose_level);
	//test8(verbose_level);
	//test9(verbose_level);
	//test10(verbose_level);
	//test11(verbose_level);
	//test12(verbose_level);
	//test13(verbose_level);
	//test14(verbose_level);
	//test15(verbose_level);
	//test16(verbose_level);
	//test17(verbose_level);
	//test18(verbose_level);
	//test19(verbose_level);
	//test20(verbose_level);
	//instant_insanity(verbose_level);
	//test_bricks(verbose_level);
	//test_null_polarity_generator(verbose_level);
	//test_linear_group_generator(verbose_level);
	//test_homogeneous_polynomials(verbose_level);

	the_end(t0);
}

void test_group(int argc, const char **argv, int f_full, int verbose_level)
{
	//const char *argv[] = { "-PSL", "2", "3" };
	//int argc = 3;
	linear_group_description *Descr;
	linear_group *L;
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int h, i, j, q;
	int *Elt;


	cout << "arguments:" << endl;
	for (i = 0; i < argc; i++) {
		cout << argv[i] << endl;
		}
	Descr = new linear_group_description;
	Descr->read_arguments(argc, argv, verbose_level);

	cout << "Descr->input_q = " << Descr->input_q << endl;
	F = new finite_field;
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;
	


	L = new linear_group;
	cout << "before LG->init, creating the group" << endl;
	L->init(Descr, verbose_level);
	
	cout << "after LG->init" << endl;

	A = L->A2;
	sims *S;

	//S = L->initial_strong_gens->create_sims(verbose_level);

	S = L->Strong_gens->create_sims(verbose_level);
	S->group_order(Go);
	Elt = NEW_int(A->elt_size_in_int);
	go = Go.as_int();
	cout << "Group of order " << go << endl;
	
	if (go > 100) {
		cout << "group order too large to list elements" << endl;
		}
	else {
		for (h = 0; h < go; h++) {
			S->element_unrank_int(h, Elt);
			cout << "Element " << h << " / " << go << " : path=";
			int_vec_print(cout, S->path, S->A->base_len);
			cout << endl;
			for (i = 0; i < S->A->base_len; i++) {
				j = S->path[i];
				S->coset_rep(i, j, 0 /* verbose_level*/);
				// coset rep now in cosetrep
				if (i) {
					cout << "*" << endl;
					}
				A->element_print_quick(S->cosetrep, cout);
				}
			cout << "=" << endl;
			A->element_print_quick(Elt, cout);
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
			}
		}
	char fname[1000];
	
	sprintf(fname, "%s_group_tree.tree", L->prefix);
	cout << "writing file " << fname << endl;
	S->create_group_tree(fname, f_full, verbose_level);
}

void test1(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go; //, i, N = 0;
	int d = 4;
	int q = 4;
	int f_semilinear = FALSE;
	int *Elt1;
	int *Elt2;
	int i;
	
	F = new finite_field;
	F->init(q, 0);
	A = new action;
	A->init_projective_group(d /* n */, F, 
		f_semilinear, TRUE /* f_basis */, verbose_level);
	A->print_base();
	A->group_order(Go);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	go = Go.as_int();
	cout << "Group of order " << go << endl;
	
#if 0
	for (i = 0; i < go; i++) {
		A->Sims->element_unrank_int(i, Elt);
		cout << "Element " << i << " / " << go << ":" << endl;
		A->element_print_quick(Elt, cout);
		}
#endif

	if (d == 4 && q == 4) {
		int ord;

		int data[] = {
			1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,1,1, 
			1,0,0,0, 1,1,0,0, 0,0,1,0, 0,0,0,1, 
			1,0,0,0, 1,1,0,0, 0,1,1,0, 0,0,1,1, 
			2,0,0,0, 1,2,0,0, 0,0,1,0, 0,0,0,1, 
			1,0,0,0, 1,1,0,0, 0,1,1,0, 0,0,0,1, 
			2,0,0,0, 1,2,0,0, 0,0,1,0, 0,0,0,1, 
			3,0,0,0, 1,3,0,0, 0,0,3,0, 0,0,1,3, 
			2,0,0,0, 1,2,0,0, 0,0,2,0, 0,0,1,2};

		for (i = 0; i < 8; i++) {
			cout << i << " / " << 8 << ":" << endl;
			A->make_element(Elt1, data + i * 16, verbose_level);
			cout << "Elt1:" << endl;
			A->element_print_quick(Elt1, cout);
			ord = A->element_order(Elt1);
			cout << "Elt1 has order " << ord << endl;

			if (EVEN(ord)) {
				A->element_mult(Elt1, Elt1, Elt2, 0);
				ord = A->element_order(Elt2);
				cout << "Elt2 = Elt1^2 has order " << ord << endl;
				cout << "Elt2=" << endl;
				A->element_print_quick(Elt2, cout);
				}
			}

		}
	
#if 0
	N = 0;
	for (i = 0; i < go; i++) {
		A->Sims->element_unrank_int(i, A->Elt1);
		if (A->element_order(A->Elt1) == 4) {
			cout << "Element " << i << " / " << go << ":" << endl;
			A->element_print_quick(A->Elt1, cout);
			cout << "is the " << N << "th element of order 4" << endl;
			N++;
			}
		}
	cout << "found " << N << " elements of order 4" << endl;
#endif
	FREE_int(Elt1);
	FREE_int(Elt2);
	delete A;
}

void test2(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	//int go, i;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int epsilon = 1;
	int n = 6;
	int q = 13;

	action_on_orthogonal *AO;
	orthogonal *O;
	
	int N;
		
	F = new finite_field;
	A = new action;
	
	F->init(q, 0);
	A->init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);
	
	AO = A->G.AO;
	O = AO->O;

	N = O->nb_points + O->nb_lines;
	
	//A->degree = O->nb_points;
		


	A->print_base();
	A->group_order(Go);


	strong_generators *gens;
	longinteger_object go;

	gens = new strong_generators;

	cout << "before gens->init" << endl;
	gens->init(A, verbose_level);

	cout << "before gens->even_subgroup" << endl;
	gens->even_subgroup(verbose_level);

	gens->group_order(go);
	cout << "Created generators for the even "
			"subgroup of order " << go << endl;
	
#if 0
	go = Go.as_int();
	
	for (i = 0; i < go; i++) {
		A->element_unrank_int(i, A->Elt1, 0);
		if (A->element_order_if_divisor_of(A->Elt1, 13) == 13) {
			N++;
			}
		}
	cout << "found " << N << " elements of order 13" << endl;
#endif

	delete A;
}

void test3(int verbose_level)
{
	action *A;
	//longinteger_object Go;
	int i;
	int degree = 12;
	int nb_gens = 1;
	int gens[] = {
		1, 2, 3, 4, 5, 6, 7, 0, 9, 8, 11, 10
		};
	vector_ge Gens;
		
	A = new action;
	
	A->init_permutation_group(degree, verbose_level);
	Gens.init(A);
	Gens.allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		Gens.copy_in(i, gens + i * degree);
		}
	Gens.print(cout);
	
	schreier S;
	
	S.init(A);
	S.init_generators(Gens);
	S.compute_all_point_orbits(verbose_level);
	S.print_orbit_lengths(cout);
	S.print_orbit_length_distribution(cout);
	
	delete A;
}

void test4(int verbose_level)
{
	int degree = 12;
	int perm[] = {
		1, 2, 3, 4, 5, 6, 7, 0, 9, 8, 11, 10
		};
	perm_print_cycles_sorted_by_length(cout, degree, perm, verbose_level);
}

void test5(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	
		
	F = new finite_field;
	A = new action;
	F->init(8, 0);
	A->init_projective_group(3 /* n */, F, 
		TRUE /* f_semilinear */, TRUE /* f_basis */, verbose_level);
	A->print_base();
	A->group_order(Go);
	
#if 0
	int N = 0;
	int i, go;

	go = Go.as_int();
	for (i = 0; i < go; i++) {
		A->element_unrank_int(i, A->Elt1, 0);
		if (A->element_order_if_divisor_of(A->Elt1, 13) == 13) {
			N++;
			}
		}
	cout << "found " << N << " elements of order 13" << endl;
#endif

	delete A;
}

void test6(int verbose_level)
{
	finite_field *F;
	action *A;
	action *A4;
	longinteger_object Go;
	int go, i, j, rk, ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
	int *ELT1, *ELT2;
	int *Elt_At, *Elt_As, *Elt_Bt, *Elt_Bs, *ELT_A, *ELT_B;
		
	F = new finite_field;
	A = new action;
	F->init(67, 0);
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);




	A4 = new action;
	A4->init_projective_group(4 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);
	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];

	Elt_At = new int[A->elt_size_in_int];
	Elt_As = new int[A->elt_size_in_int];
	Elt_Bt = new int[A->elt_size_in_int];
	Elt_Bs = new int[A->elt_size_in_int];

	ELT1 = new int[A4->elt_size_in_int];
	ELT2 = new int[A4->elt_size_in_int];

	ELT_A = new int[A4->elt_size_in_int];
	ELT_B = new int[A4->elt_size_in_int];

	int dataAt[] = {61, 45, 66, 1};
	int dataAs[] = {40, 8, 28, 1};
	int mtxA[16];
	int dataBt[] = {2, 50, 35, 1};
	int dataBs[] = {2, 22, 18, 1};
	int mtxB[16];



	cout << "making elements At,As,Bt,Bs:" << endl;
	A->make_element(Elt_At, dataAt, FALSE);
	A->make_element(Elt_As, dataAs, FALSE);
	A->make_element(Elt_Bt, dataBt, FALSE);
	A->make_element(Elt_Bs, dataBs, FALSE);

	cout << "making elements A,B:" << endl;
	O4_isomorphism_2to4(F, dataAt, dataAs, TRUE, mtxA);
	A4->make_element(ELT_A, mtxA, verbose_level);
	cout << "A:" << endl;
	A4->element_print_quick(ELT_A, cout);
	ord = A4->element_order(ELT_A);
	cout << "A has order " << ord << endl;


	O4_isomorphism_2to4(F, dataBt, dataBs, TRUE, mtxB);
	A4->make_element(ELT_B, mtxB, verbose_level);
	cout << "B:" << endl;
	A4->element_print_quick(ELT_B, cout);
	ord = A4->element_order(ELT_B);
	cout << "B has order " << ord << endl;



	A->make_element(Elt1, dataBt, FALSE);
	A->make_element(Elt2, dataAs, FALSE);
	A->make_element(Elt6, dataAt, FALSE);
	A->element_invert(Elt1, Elt3, FALSE);
	A->element_mult(Elt3, Elt2, Elt4, FALSE);
	A->element_mult(Elt4, Elt1, Elt5, FALSE);
	cout << "B_t^{-1} * A_s * B_t:" << endl;
	A->element_print_quick(Elt5, cout);
	A->element_print_as_permutation(Elt5, cout);


	cout << "A_t^4:" << endl;
	A->make_element(Elt6, dataAt, FALSE);
	A->element_power_int_in_place(Elt6, 4, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);



	cout << "################" << endl;

	cout << "A_t^17:" << endl;
	A->make_element(Elt6, dataAt, FALSE);
	A->element_power_int_in_place(Elt6, 17, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);

	cout << "A_s^17:" << endl;
	A->make_element(Elt6, dataAs, FALSE);
	A->element_power_int_in_place(Elt6, 17, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);

	cout << "B_t^66:" << endl;
	A->make_element(Elt6, dataBt, FALSE);
	A->element_power_int_in_place(Elt6, 66, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);

	cout << "B_s^68:" << endl;
	A->make_element(Elt6, dataBs, FALSE);
	A->element_power_int_in_place(Elt6, 68, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);

	cout << "################" << endl;



	A->make_element(Elt1, dataBs, FALSE);
	A->make_element(Elt2, dataAt, FALSE);
	A->make_element(Elt6, dataAs, FALSE);
	A->element_invert(Elt1, Elt3, FALSE);
	A->element_mult(Elt3, Elt2, Elt4, FALSE);
	A->element_mult(Elt4, Elt1, Elt5, FALSE);
	cout << "B_s^{-1} * A_t * B_s:" << endl;
	A->element_print_quick(Elt5, cout);
	A->element_print_as_permutation(Elt5, cout);

	cout << "A_s^4:" << endl;
	A->element_power_int_in_place(Elt6, 4, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);




	A->make_element(Elt7, dataBt, FALSE);
	ord = A->element_order(Elt7);
	cout << "Bt has order " << ord << endl;

	A->make_element(Elt7, dataBs, FALSE);
	ord = A->element_order(Elt7);
	cout << "Bs has order " << ord << endl;

	A->make_element(Elt7, dataAt, FALSE);
	ord = A->element_order(Elt7);
	cout << "At has order " << ord << endl;

	A->make_element(Elt7, dataAs, FALSE);
	ord = A->element_order(Elt7);
	cout << "As has order " << ord << endl;

	
	int dataD[4];


	{
	finite_field GFp;
	GFp.init(F->p, 0);

	unipoly_domain FX(&GFp);
	unipoly_object m; //, n;

	const char *polynomial;
	int e = 2;
	
	polynomial = get_primitive_polynomial(F->p, e, verbose_level);
	FX.create_object_by_rank_string(m, polynomial, 0);

	dataD[0] = 0;
	dataD[2] = 1;
	dataD[1] = GFp.negate(FX.s_i(m, 0));
	dataD[3] = GFp.negate(FX.s_i(m, 1));
	}

	A->make_element(Elt7, dataD, FALSE);
	ord = A->element_order(Elt7);
	A->element_print_quick(Elt7, cout);
	cout << "D has order " << ord << endl;
	
	cout << "################" << endl;

	cout << "dataBs: ";
	int_vec_print(cout, dataBs, 4);
	cout << endl;
	A->make_element(Elt7, dataBs, FALSE);
	ord = A->element_order(Elt7);
	cout << "Bs has order " << ord << endl;
	A->element_print_quick(Elt7, cout);
	A->element_print_as_permutation(Elt7, cout);


	cout << "dataBt: ";
	int_vec_print(cout, dataBt, 4);
	cout << endl;
	A->make_element(Elt6, dataBt, FALSE);
	ord = A->element_order(Elt6);
	cout << "Bt has order " << ord << endl;
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);




	O4_isomorphism_2to4(F, dataBt, dataBs, TRUE, mtxB);
	A4->make_element(ELT1, mtxB, verbose_level);
	cout << "B:" << endl;
	A4->element_print_quick(ELT1, cout);
	ord = A4->element_order(ELT1);
	cout << "B has order " << ord << endl;

	O4_grid_action(A4, ELT1, verbose_level);

#if 0
	O4_isomorphism_2to4(F, dataAt, dataAs, FALSE, mtxA);
	A4->make_element(ELT2, mtxA, verbose_level);
	cout << "A:" << endl;
	A4->element_print_quick(ELT2, cout);
	ord = A4->element_order(ELT2);
	cout << "A has order " << ord << endl;

	O4_grid_action(A4, ELT2, verbose_level);
#endif




	int R1[] = {65,65,45,52};
	int R_start[4], R_cur[4], R_next[4];
	int **tangent_planes;

	cout << "computing tangent planes first orbit:" << endl;
	tangent_planes = new pint[17];
	R_start[0] = R_cur[0] = R1[0];
	R_start[1] = R_cur[1] = R1[1];
	R_start[2] = R_cur[2] = R1[2];
	R_start[3] = R_cur[3] = R1[3];
	for (i = 0; i < 17; i++) {
		tangent_planes[i] = new int[3 * 4];
		cout << "i=" << i << endl;
		O4_find_tangent_plane(*F, R_cur[0], R_cur[1], R_cur[2], R_cur[3],
			tangent_planes[i], verbose_level);
		F->mult_vector_from_the_left(R_cur, mtxA, R_next, 4, 4);
		R_cur[0] = R_next[0];
		R_cur[1] = R_next[1];
		R_cur[2] = R_next[2];
		R_cur[3] = R_next[3];
		}
	//cout << "O4_find_tangent_plane" << endl;
	//O4_find_tangent_plane(*F, 1, 1, 0, 0, verbose_level);

	for (i = 0; i < 17; i++) {
		cout << "i=" << i << " basis:" << endl;
		print_integer_matrix_width(cout, tangent_planes[i],
			3, 4, 4, F->log10_of_q);
		}
	
	int *intersection;

	intersection = new int[4 * 4];
	
	for (i = 0; i < 17; i++) {
		for (j = i + 1; j < 17; j++) {
			cout << "intersection T_" << i << " \\cap T_" << j << ":" << endl;
			 
			int a;
			
			rk = F->intersect_subspaces(4, 3,
				tangent_planes[i], 3, tangent_planes[j],
				a, intersection, 0);
			print_integer_matrix_width(cout, intersection,
				rk, 4, 4, F->log10_of_q);
			}
		}
	
	delete A;
}

void O4_grid_action(action *A_PGL4, int *Elt, int verbose_level)
{
	int q;
	int x1, x2, x3, x4;
	int v[4], w[4];
	int x, y, z, xx, yy, zz, z0;
	finite_field *F;
	F = A_PGL4->G.matrix_grp->GFq;
	q = F->q;
	int size = q + 1;
	int *perm;

	perm = new int[size * size];
	z = z0 = 0; // 69;
	while (TRUE) {
		x = z / size;
		y = z % size;
		cout << "(" << x << "," << y << ") -> ";
		O4_grid_coordinates_unrank(*F, x1, x2, x3, x4,
			x, y, verbose_level);
		v[0] = x1;
		v[1] = x2;
		v[2] = x3;
		v[3] = x4;
		cout << "v=";
		int_vec_print(cout, v, 4);
		cout << endl;
		F->mult_vector_from_the_left(v, Elt, w, 4, 4);
		cout << "w=";
		int_vec_print(cout, w, 4);
		cout << endl;
		O4_grid_coordinates_rank(*F, w[0], w[1], w[2], w[3],
				xx, yy, verbose_level);
		cout << "(" << xx << "," << yy << ")" << endl;
		zz = xx * size + yy;
		if (zz == z0)
			break;
		z = zz;
		}
	
#if 0
	for (x = 0; x < size; x++) {
		for (y = 0; y < size; y++) {
			z = x * size + y;
			cout << "(" << x << "," << y << ") -> ";
			O4_grid_coordinates_unrank(*F, x1, x2, x3, x4,
				x, y, verbose_level);
			v[0] = x1;
			v[1] = x2;
			v[2] = x3;
			v[3] = x4;
			//cout << "v=";
			//int_vec_print(cout, v, 4);
			//cout << endl;
			F->mult_vector_from_the_left(v, Elt, w, 4, 4);
			//cout << "w=";
			//int_vec_print(cout, w, 4);
			//cout << endl;
			O4_grid_coordinates_rank(*F, w[0], w[1], w[2], w[3],
				xx, yy, verbose_level);
			cout << "(" << xx << "," << yy << ")" << endl;
			zz = xx * size + yy;
			perm[z] = zz;
			}
		}
	cout << "element:" << endl;
	A_PGL4->element_print(Elt, cout);
	cout << "action on grid:" << endl;
	perm_print(cout, perm, size * size);
	cout << endl;
#endif

}


void test7(int verbose_level)
{
	finite_field *F;
	action *A;
	action *A4;
	longinteger_object Go;
	int go, ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
	int *ELT1, *ELT2;
	int *Elt_At, *Elt_As, *Elt_Bt, *Elt_Bs, *ELT_A, *ELT_B;
	
	F = new finite_field;
	F->init(67, 0);
	A = new action;
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A4 = new action;
	A4->init_projective_group(4 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);
	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];

	Elt_At = new int[A->elt_size_in_int];
	Elt_As = new int[A->elt_size_in_int];
	Elt_Bt = new int[A->elt_size_in_int];
	Elt_Bs = new int[A->elt_size_in_int];

	ELT1 = new int[A4->elt_size_in_int];
	ELT2 = new int[A4->elt_size_in_int];

	ELT_A = new int[A4->elt_size_in_int];
	ELT_B = new int[A4->elt_size_in_int];

	int dataAt[] = {62,3,32,1};
	int dataAs[] = {18,25,21,1};
	int f_switchA = FALSE;
	int mtxA[16];
	int dataBt[] = {38,15,26,1};
	int dataBs[] = {43,31,1,24};
	int f_switchB = TRUE;
	int mtxB[16];



	cout << "making elements At,As,Bt,Bs:" << endl;
	A->make_element(Elt_At, dataAt, FALSE);
	A->make_element(Elt_As, dataAs, FALSE);
	A->make_element(Elt_Bt, dataBt, FALSE);
	A->make_element(Elt_Bs, dataBs, FALSE);

	cout << "making elements A,B (including fine_tune):" << endl;
	O4_isomorphism_2to4(F, dataAt, dataAs, f_switchA, mtxA);
	fine_tune(F, mtxA, verbose_level);
	A4->make_element(ELT_A, mtxA, verbose_level);
	cout << "A:" << endl;
	A4->element_print_quick(ELT_A, cout);
	ord = A4->element_order(ELT_A);
	cout << "A has order " << ord << endl;


	O4_isomorphism_2to4(F, dataBt, dataBs, f_switchB, mtxB);
	A4->make_element(ELT_B, mtxB, verbose_level);
	fine_tune(F, mtxB, verbose_level);
	cout << "B:" << endl;
	A4->element_print_quick(ELT_B, cout);
	ord = A4->element_order(ELT_B);
	cout << "B has order " << ord << endl;



	A->make_element(Elt1, dataBt, FALSE);
	A->make_element(Elt2, dataAs, FALSE);
	A->make_element(Elt6, dataAt, FALSE);
	A->element_invert(Elt1, Elt3, FALSE);
	A->element_mult(Elt3, Elt2, Elt4, FALSE);
	A->element_mult(Elt4, Elt1, Elt5, FALSE);
	cout << "B_t^{-1} * A_s * B_t:" << endl;
	A->element_print_quick(Elt5, cout);
	A->element_print_as_permutation(Elt5, cout);


	cout << "A_t^4:" << endl;
	A->make_element(Elt6, dataAt, FALSE);
	A->element_power_int_in_place(Elt6, 4, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);




	A->make_element(Elt1, dataBs, FALSE);
	A->make_element(Elt2, dataAt, FALSE);
	A->make_element(Elt6, dataAs, FALSE);
	A->element_invert(Elt1, Elt3, FALSE);
	A->element_mult(Elt3, Elt2, Elt4, FALSE);
	A->element_mult(Elt4, Elt1, Elt5, FALSE);
	cout << "B_s^{-1} * A_t * B_s:" << endl;
	A->element_print_quick(Elt5, cout);
	A->element_print_as_permutation(Elt5, cout);

	cout << "A_s^4:" << endl;
	A->element_power_int_in_place(Elt6, 4, FALSE);
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);


}

void test8(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go, i;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	A = new action;
	F->init(7, 0);
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data[] = {0,1,4,6};



	cout << "making element:" << endl;
	A->make_element(Elt1, data, FALSE);
	A->element_move(Elt1, Elt2, 0);
	for (i = 0; i < 8; i++) {
		A->element_print_quick(Elt2, cout);
		A->element_print_as_permutation(Elt2, cout);
		A->element_mult(Elt1, Elt2, Elt3, 0);
		A->element_move(Elt3, Elt2, 0);
		}

}

void test9(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	F->init(8, 0);
	A = new action;
	A->init_projective_group(3 /* n */, F, 
		TRUE /* f_semilinear */, TRUE /* f_basis */, verbose_level);




	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data1[] = {0,1,0,1,0,0,0,0,1,0};
	int data2[] = {2,0,2,0,2,2,0,0,4,2};



	cout << "making element:" << endl;
	A->make_element(Elt1, data1, FALSE);
	A->make_element(Elt2, data2, FALSE);
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);
	A->element_print_quick(Elt2, cout);
	A->element_print_as_permutation(Elt2, cout);
	
	A->element_mult(Elt1, Elt2, Elt3, 0);
	A->element_print_quick(Elt3, cout);
	A->element_print_as_permutation(Elt3, cout);

}

void test10(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	F->init(8, 0);
	A = new action;
	A->init_projective_group(3 /* n */, F, 
		TRUE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data1[] = {1,0,0,0,6,0,0,0,1,0};
	int data2[] = {3,0,2,0,5,0,0,0,2,1};



	cout << "making element:" << endl;
	A->make_element(Elt1, data1, FALSE);
	A->make_element(Elt2, data2, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);
	cout << "Elt2:" << endl;
	A->element_print_quick(Elt2, cout);
	A->element_print_as_permutation(Elt2, cout);
	
	cout << "Elt3:" << endl;
	A->element_mult(Elt2, Elt2, Elt3, 0);
	A->element_print_quick(Elt3, cout);
	A->element_print_as_permutation(Elt3, cout);

}

void test11(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	F->init_override_polynomial(16, "19", 0);
	A = new action;
	A->init_projective_group(3 /* n */, F, 
		TRUE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data1[] = {1,6,6,6,1,6,1,1,7,2};
	int data2[] = {1,6,6,6,1,6,1,1,7,2};



	cout << "making element:" << endl;
	A->make_element(Elt1, data1, FALSE);
	A->make_element(Elt2, data2, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);
	cout << "Elt2:" << endl;
	A->element_print_quick(Elt2, cout);
	A->element_print_as_permutation(Elt2, cout);
	
	cout << "Elt3:" << endl;
	A->element_mult(Elt2, Elt2, Elt3, 0);
	A->element_print_quick(Elt3, cout);
	A->element_print_as_permutation(Elt3, cout);

}

void test12(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	F->init_override_polynomial(16, "19", 0);
	A = new action;
	A->init_projective_group(3 /* n */, F, 
		TRUE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	//int data2[] = {5,0,0,0,5,0,0,0,5,2};
	int data1[] = {1,0,0,0,7,0,0,0,6,2};
	int data3[] = {1,0,0,0,6,0,0,0,7,0};



	cout << "making element:" << endl;
	A->make_element(Elt1, data3, FALSE);
	A->make_element(Elt2, data1, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);
	cout << "Elt2:" << endl;
	A->element_print_quick(Elt2, cout);
	A->element_print_as_permutation(Elt2, cout);
	
	cout << "Elt3:" << endl;
	A->element_mult(Elt2, Elt2, Elt3, 0);
	A->element_print_quick(Elt3, cout);
	A->element_print_as_permutation(Elt3, cout);

}

void test13(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	F = new finite_field;
	A = new action;
	F->init(4, 0);
	A->init_projective_group(3 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data1[] = {0,1,0,0,0,1,2,2,3};

	display_all_PG_elements(2, *A->G.matrix_grp->GFq);

	cout << "making element:" << endl;
	A->make_element(Elt1, data1, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);
	

}

void test14(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
	int q = 19;
	int f_basis = TRUE;
	int f_init_hash_table = FALSE;
	//int verbose_level = 2;
	int BLT_set_size = 20;
	int BLT_set[] = {
		7162, 
		2629, 
		2060, 
		1758, 
		4223, 
		5692, 
		417, 
		6386, 
		832, 
		5534, 
		7103, 
		6654, 
		6629, 
		5629, 
		6343, 
		6395, 
		465, 
		3280, 
		2228, 
		4442, 
		};

	F = new finite_field;
	F->init(q, 0);
	A = new action;
	A->init_BLT(F, f_basis, f_init_hash_table, verbose_level);
	//A->init_matrix_group(TRUE /* f_projective */,
	//FALSE /* f_no_translations */, 3 /* n */, 4,
	//	NULL, FALSE /* f_semilinear */, TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];


	int data1[] = {
		18,0,0,0,0,
		0,16,6,16,13,
		0,1,18,8,12,
		0,8,18,8,1,
		0,17,13,3,15,
		};
	int data2[] = {
		1,0,0,0,0,
		0,4,2,12,12,
		0,5,4,6,3,
		0,13,14,8,1,
		0,16,13,10,2,
		};

	display_all_PG_elements(2, *A->G.matrix_grp->GFq);

	cout << "making element:" << endl;
	A->make_element(Elt1, data1, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);

	A->make_element(Elt2, data2, FALSE);
	cout << "Elt2:" << endl;
	A->element_print_quick(Elt2, cout);
	A->element_print_as_permutation(Elt2, cout);
	
	A->element_invert(Elt1, Elt4, 0);
	A->element_mult(Elt4, Elt2, Elt5, 0);
	A->element_mult(Elt5, Elt1, Elt6, 0);
	A->element_mult(Elt1, Elt1, Elt7, 0);

	cout << "Elt6 = Elt1^-1 * Elt2 * Elt1:" << endl;
	A->element_print_quick(Elt6, cout);
	A->element_print_as_permutation(Elt6, cout);

	cout << "Elt7 = Elt1^2:" << endl;
	A->element_print_quick(Elt7, cout);
	A->element_print_as_permutation(Elt7, cout);

	action *AR;
	int f_induce_action = FALSE;

	cout << "computing restricted action:" << endl;
	AR = new action;
	AR->induced_action_by_restriction(*A, f_induce_action, A->Sims, 
		BLT_set_size, BLT_set, verbose_level);


	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	AR->element_print_as_permutation(Elt1, cout);

	cout << "Elt2:" << endl;
	A->element_print_quick(Elt2, cout);
	AR->element_print_as_permutation(Elt2, cout);

	cout << "Elt6:" << endl;
	A->element_print_quick(Elt6, cout);
	AR->element_print_as_permutation(Elt6, cout);
	
	cout << "Elt7 = Elt1^2:" << endl;
	A->element_print_quick(Elt7, cout);
	AR->element_print_as_permutation(Elt7, cout);
}

void test15(int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int f_semilinear = TRUE;
	int f_basis = TRUE;
	int epsilon = 0;
	int n = 5;
	int q = 67;

	action_on_orthogonal *AO;
	orthogonal *O;
	
	int N;
		
	F = new finite_field;

	F->init(q, 0);
	A = new action;
	
	A->init_orthogonal_group(epsilon, n, F, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);
	
	A->lex_least_base_in_place(verbose_level);
	
	AO = A->G.AO;
	O = AO->O;

	N = O->nb_points + O->nb_lines;
	
	//A->degree = O->nb_points;
		


	A->print_base();
	A->group_order(Go);

#if 0
	go = Go.as_int();
	
	for (i = 0; i < go; i++) {
		A->element_unrank_int(i, A->Elt1, 0);
		if (A->element_order_if_divisor_of(A->Elt1, 13) == 13) {
			N++;
			}
		}
	cout << "found " << N << " elements of order 13" << endl;
#endif

	delete A;
}

void test16(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A;
	action *A2;
	longinteger_object Go;
	int f_semilinear = TRUE;
	int f_basis = TRUE;
	int epsilon = 0;
	int n = 5;
	int q = 67;
	int i, j, h, u, v, r;
	int pts[] = {
0, 1, 0, 0, 0 ,
0, 0, 1, 0, 0 ,
0, 1, 33, 66, 33 ,
0, 1, 25, 33, 50 ,
0, 1, 15, 25, 53 ,
1, 65, 17, 40, 36 ,
1, 3, 61, 21, 4 ,
1, 54, 49, 17, 65 ,
1, 2, 3, 48, 18 ,
 1, 33, 14, 13, 52 ,
 1, 22, 4, 3, 15 ,
 1, 34, 28, 30, 33 ,
 1, 6, 64, 1, 17 ,
 1, 39, 47, 53, 64 ,
 1, 12, 50, 10, 27 ,
 1, 36, 54, 40, 10 ,
 1, 24, 4, 24, 49 ,
 1, 47, 6, 59, 27 ,
 1, 20, 43, 35, 29 ,
 1, 16, 15, 24, 43 ,
 1, 35, 23, 15, 49 ,
 1, 42, 11, 34, 12 ,
 1, 53, 2, 38, 6 ,
 1, 24, 15, 45, 56 ,
 1, 59, 39, 25, 58 ,
 1, 38, 50, 59, 45 ,
 1, 45, 45, 6, 42 ,
 1, 59, 47, 12, 48 ,
 1, 32, 48, 12, 45 ,
 1, 43, 30, 42, 57 ,
 1, 55, 14, 58, 41 ,
 1, 43, 46, 35, 43 ,
 1, 56, 17, 62, 3 ,
 1, 7, 34, 56, 40 ,
 1, 52, 18, 23, 35 ,
 1, 32, 34, 17, 66 ,
 1, 16, 19, 33, 7 ,
 1, 16, 56, 35, 5 ,
 1, 62, 64, 6, 42 ,
 1, 18, 57, 45, 1 ,
 1, 41, 20, 62, 57 ,
 1, 37, 35, 40, 48 ,
 1, 39, 62, 7, 66 ,
 1, 23, 37, 30, 52 ,
 1, 52, 42, 6, 49 ,
 1, 10, 19, 45, 30 ,
 1, 34, 11, 49, 32 ,
 1, 39, 4, 57, 9 ,
 1, 5, 3, 47, 41 ,
 1, 62, 10, 34, 31 ,
 1, 32, 63, 15, 4 ,
 1, 25, 19, 35, 40 ,
 1, 37, 62, 6, 53 ,
 1, 37, 36, 38, 9 ,
 1, 29, 10, 12, 26 ,
 1, 63, 42, 58, 41 ,
 1, 64, 56, 57, 57 ,
 1, 28, 66, 8, 62 ,
 1, 60, 47, 46, 45 ,
 1, 14, 6, 25, 10 ,
 1, 17, 29, 18, 47 ,
 1, 27, 43, 28, 59 ,
 0, 1, 64, 43, 25 ,
 1, 42, 11, 58, 44 ,
 1, 50, 12, 30, 9 ,
 1, 56, 14, 14, 54 ,
 1, 14, 49, 16, 45 ,
 1, 50, 3, 33, 34 
 };

	int nb_stab_gens = 2;
	int stab_gens[] = {
//19  ,  0  ,  6  , 41  , 19 ,
// 3  ,  1  ,  1  , 18  , 59 ,
// 0  , 65  ,  1  ,  2  ,  1 ,
//43  ,  1  , 59  , 57  , 30 ,
//54  ,  2  , 18  , 54  , 57 ,

// element of order 17:
9, 54, 0, 32, 31, 
3, 29, 10, 36, 1, 
0, 47, 0, 0, 0, 
59, 60, 0, 28, 36, 
36, 50, 0, 10, 58, 


// element of order 4:
49  ,  7  ,  2  , 37  ,  9 ,
33  , 14  , 10  ,  7  , 35 ,
30  , 17  , 29  , 54  , 35 ,
19  ,  4  , 26  ,  2  ,  2 ,
 8  , 17  , 12  ,  0  , 39 ,
};
	int nb_moves = 1;
	int transporter_data[] = {
	//55,9,21,51,36,
	//20,18,26,63,16,
	//39,41,34,35,22,
	//7,17,32,35,29,
	//59,49,4,6,46,

	14,21,9,63,11,
	57,8,14,27,12,
	62,6,56,36,3,
	2,1,4,4,65,
	35,26,46,24,8,
		};


	orthogonal *O;
	finite_field *Fq;
	
	int *Pts;
	int N = q + 1;
	int *elt1;
		
	A = new action;
	
	Fq = new finite_field;
	Fq->init(q, 0);
	A->init_orthogonal_group(epsilon, n, Fq, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level - 1);
	
	A->lex_least_base_in_place(verbose_level - 1);
	
	O = A->G.AO->O;

	//N = O->nb_points + O->nb_lines;
	
	//A->degree = O->nb_points;
	
		


	A->print_base();
	A->group_order(Go);


	A2 = new action;
	A2->init_projective_group(2, Fq, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, verbose_level - 1);
	A2->lex_least_base_in_place(verbose_level - 1);
	elt1 = NEW_int(A2->elt_size_in_int);
		
	Pts = NEW_int(N);
	for (i = 0; i < N; i++) {
		Pts[i] = O->rank_point(pts + i * 5, 1, 0);
		}
	cout << "Pts: " << endl;
	for (i = 0; i < N; i++) {
		cout << setw(10) << Pts[i];
		if (((i + 1) % 5) == 0)
			cout << endl;
		}
	//int_vec_print(cout, Pts, N);
	cout << endl;

	vector_ge *gens;
	vector_ge *transporter;
	int *Elt1;

	gens = new vector_ge;
	gens->init(A);
	gens->allocate(nb_stab_gens);
	transporter = new vector_ge;
	transporter->init(A);
	transporter->allocate(nb_moves);
	
	Elt1 = NEW_int(A->elt_size_in_int);
	for (i = 0; i < nb_stab_gens; i++) {
		A->make_element(Elt1, stab_gens + i * 25, verbose_level);

		gens->copy_in(i, Elt1);
		
		cout << "stabilizer generator " << i << ":" << endl;
		A->element_print_quick(Elt1, cout);
		
		}
	for (i = 0; i < nb_moves; i++) {
		A->make_element(Elt1, transporter_data + i * 25, verbose_level);

		transporter->copy_in(i, Elt1);
		
		cout << "transporter " << i << ":" << endl;
		A->element_print_quick(Elt1, cout);
		
		}

	
	sims *Stab;
	action *A_restr;

	Stab = new sims;
	
	longinteger_object target_go;
	int f_has_target_group_order = TRUE;

	target_go.create(N);
	
	schreier_sims *ss;

	ss = new schreier_sims;
	
	ss->init(A, verbose_level - 1);

	ss->interested_in_kernel(A->subaction, verbose_level - 1);
	
	if (f_has_target_group_order) {
		ss->init_target_group_order(target_go, verbose_level - 1);
		}
	
	ss->init_generators(gens, verbose_level);
#if 0
	ss->init_random_process(
		callback_choose_random_generator_orthogonal, 
		ss, 
		verbose_level - 1);
#endif
	
	ss->create_group(verbose_level - 1);
	


	int f_induce_action = TRUE;
	
	A_restr = new action;
	A_restr->induced_action_by_restriction(*A, f_induce_action, ss->G, 
		N, Pts, verbose_level - 1);
	if (f_v) {
		cout << "after A_restr->induced_action_by_restriction" << endl;
		}

	int order;
	int w[5];
	
	for (i = 0; i < N; i++) {
		A_restr->Sims->element_unrank_int(i, Elt1);
		cout << setw(3) << i << " : ";
		order = A_restr->element_order(Elt1);
		cout << setw(3) << order << " : ";
		A_restr->element_print_as_permutation(Elt1, cout);
		cout << endl;
		A_restr->element_print_quick(Elt1, cout);
		cout << endl;
		}

	for (i = 0; i < nb_moves; i++) {
		gens->conjugate_sasv(transporter->ith(i));

		cout << "after conjugation:" << endl;
		gens->print(cout);
		
		A->element_invert(transporter->ith(i), Elt1, 0);
		//A->element_move(transporter->ith(i), Elt1, 0);
		
		for (j = 0; j < N; j++) {
			Pts[j] = A->element_image_of(Pts[j], Elt1, 0);
			}
		cout << "new point set:" << endl;
		for (j = 0; j < N; j++) {
			O->unrank_point(w, 1, Pts[j], 0);
			cout << setw(3) << j << " : " << setw(10) << Pts[j] << " : ";
			int_vec_print(cout, w, 5);
			cout << endl;
			}
		{

		schreier_sims *ss3;

		ss3 = new schreier_sims;
	
		ss3->init(A, verbose_level - 1);

		ss3->interested_in_kernel(A->subaction, verbose_level - 1);
	
		if (f_has_target_group_order) {
			ss3->init_target_group_order(target_go, verbose_level - 1);
			}
	
		ss3->init_generators(gens, verbose_level);
	
		ss3->create_group(verbose_level - 1);


		action *A3;
		A3 = new action;
		A3->induced_action_by_restriction(*A, f_induce_action, ss3->G, 
			N, Pts, verbose_level - 1);
		for (j = 0; j < nb_stab_gens; j++) {
			cout << "generator " << setw(3) << j << " : ";
			order = A3->element_order(gens->ith(j));
			cout << setw(3) << order << " : ";
			A3->element_print_as_permutation(gens->ith(j), cout);
			cout << endl;
			A3->element_print_quick(gens->ith(j), cout);
			cout << endl;
			}
		delete ss3;
		delete A3;
		for (j = 0; j < nb_stab_gens; j++) {
			int Data[25];
			int data[16];

			cout << "generator " << j << " :" << endl;
			for (h = 0; h < 25; h++) {
				Data[h] = gens->ith(j)[h];
				}
			PG_element_normalize_from_front(*Fq, Data, 1, 25);
			print_integer_matrix_width(cout, Data, 5, 5, 5, 3);

			for (u = 0; u < 4; u++) {
				for (v = 0; v < 4; v++) {
					data[u * 4 + v] = Data[(u + 1) * 5 + v + 1];
					}
				}
			print_integer_matrix_width(cout, data, 4, 4, 4, 3);
			
			int small[8], f_switch;
			
			O4_isomorphism_4to2(Fq, small, small + 4, f_switch, data, verbose_level);
			cout << "after isomorphism:" << endl;
			cout << "f_switch=" << f_switch << endl;
			for (r = 0; r < 2; r++) {
				cout << "component " << r << ":" << endl;
				PG_element_normalize_from_front(*Fq, small + r * 4, 1, 4);
				print_integer_matrix_width(cout, small + r * 4, 2, 2, 2, 3);
				A2->make_element(elt1, small + r * 4, verbose_level);
				order = A2->element_order(elt1);
				cout << "has order " << order << endl;
				A2->element_print_as_permutation(elt1, cout);
				cout << endl;
				A2->element_print_quick(elt1, cout);
				cout << endl;
				
				}
			}
		}
		}

#if 0

	int T[25];
	int From[5] = {45,50,5,60,66};
	int To[5] = {0,2,2,0,0};
	int Root[5] = {0,1,0,1,0};
	int B[25], Bv[25];
	int w[5], z[5], x[5];
	int Tmp1[5];
	
	PG_element_normalize_from_front(*Fq, From, 1, 5);
	cout << "From = ";
	int_vec_print(cout, From, 5);
	cout << endl;
	cout << "To = ";
	int_vec_print(cout, To, 5);
	cout << endl;
	
	O->Siegel_Transformation3(T, 
		From, To, Root, 
		B, Bv, w, z, x,
		verbose_level);

	cout << "Siegel transformation T:" << endl;
	A->make_element(Elt1, T, 0);
	A->element_print_quick(Elt1, cout);
	cout << endl;

	Fq->mult_vector_from_the_left(From, T, Tmp1, 5, 5);
	PG_element_normalize_from_front(*Fq, Tmp1, 1, 5);
	cout << "From * T = ";
	int_vec_print(cout, Tmp1, 5);
	cout << endl;

	Fq->mult_vector_from_the_left(To, T, Tmp1, 5, 5);
	PG_element_normalize_from_front(*Fq, Tmp1, 1, 5);
	cout << "To * T = ";
	int_vec_print(cout, Tmp1, 5);
	cout << endl;

	gens->conjugate_svas(Elt1);

	cout << "after conjugation:" << endl;
	gens->print(cout);
#endif

	delete ss;
	delete gens;
	delete transporter;
	delete A;
	delete A2;
	delete A_restr;
	FREE_int(Pts);
	FREE_int(Elt1);
	FREE_int(elt1);
}

void test17(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	action *A2;
	action *A4;
	int q = 67;
	int *T1, *S1;
	int *T2, *S2;
	int *T3, *S3;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	int *Elt5;
	int *Elt6;
	int *elt1;
	int *elt2;
	int *elt3;
	int *elt4;
	int *elt5;
	int f_semilinear = FALSE;
	orthogonal *O4;
	finite_field *Fq;

	Fq = new finite_field;
	Fq->init(q, 0);
	A2 = new action;
	A2->init_projective_group(
		2, Fq, 
		f_semilinear, 
		TRUE /* f_basis */, 
		verbose_level - 1);
	A2->lex_least_base_in_place(verbose_level - 1);
	
	elt1 = NEW_int(A2->elt_size_in_int);
	elt2 = NEW_int(A2->elt_size_in_int);
	elt3 = NEW_int(A2->elt_size_in_int);
	elt4 = NEW_int(A2->elt_size_in_int);
	elt5 = NEW_int(A2->elt_size_in_int);
	T1 = NEW_int(A2->elt_size_in_int);
	S1 = NEW_int(A2->elt_size_in_int);
	T2 = NEW_int(A2->elt_size_in_int);
	S2 = NEW_int(A2->elt_size_in_int);
	T3 = NEW_int(A2->elt_size_in_int);
	S3 = NEW_int(A2->elt_size_in_int);

	A4 = new action;
	
	A4->init_orthogonal_group(1 /*epsilon*/, 4, Fq, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, FALSE /*f_basis*/, verbose_level - 1);
	
	A4->lex_least_base_in_place(verbose_level - 1);
	
	O4 = A4->G.AO->O;
	
	Elt1 = NEW_int(A4->elt_size_in_int);
	Elt2 = NEW_int(A4->elt_size_in_int);
	Elt3 = NEW_int(A4->elt_size_in_int);
	Elt4 = NEW_int(A4->elt_size_in_int);
	Elt5 = NEW_int(A4->elt_size_in_int);
	Elt6 = NEW_int(A4->elt_size_in_int);
	
	int f_switch1, f_switch2, f_switch3;
	
#if 0
	int data16_1[16];
	int data16_2[16];
	int data16_3[16];
	int r;
	
	for (r = 0; r < 100; r++) {
		A2->Sims->random_element(T1);
		A2->Sims->random_element(S1);
		A2->Sims->random_element(T2);
		A2->Sims->random_element(S2);
		f_switch1 = random_integer(2);
		f_switch2 = random_integer(2);

		if (f_switch2) {
			A2->element_mult(S1, T2, T3, 0);
			A2->element_mult(T1, S2, S3, 0);
			}
		else {
			A2->element_mult(T1, T2, T3, 0);
			A2->element_mult(S1, S2, S3, 0);
			}
		f_switch3 = (f_switch1 + f_switch2) % 2;

		O4_isomorphism_2to4(Fq, T1, S1, f_switch1, data16_1);
		O4_isomorphism_2to4(Fq, T2, S2, f_switch2, data16_2);
		O4_isomorphism_2to4(Fq, T3, S3, f_switch3, data16_3);

		A4->make_element(Elt1, data16_1, verbose_level - 1);
		A4->make_element(Elt2, data16_2, verbose_level - 1);
		A4->make_element(Elt3, data16_3, verbose_level - 1);
		A4->element_mult(Elt1, Elt2, Elt4, 0);
		A4->element_invert(Elt4, Elt5, 0);
		A4->element_mult(Elt4, Elt5, Elt6, 0);

		if (!A4->is_one(Elt6)) {
			cout << "something is wrong" << endl;
			}
		}
	cout << "works fine" << endl;
#else
	int data4_1[]={1,43,23,54};
	int data4_2[]={1,23,19,16};
	f_switch1 = TRUE;
	A2->make_element(T1, data4_1, verbose_level - 1);
	A2->make_element(S1, data4_2, verbose_level - 1);
	f_switch2 = TRUE;
	A2->make_element(T2, data4_1, verbose_level - 1);
	A2->make_element(S2, data4_2, verbose_level - 1);
	if (f_switch2) {
		A2->element_mult(S1, T2, T3, 0);
		A2->element_mult(T1, S2, S3, 0);
		}
	else {
		A2->element_mult(T1, T2, T3, 0);
		A2->element_mult(S1, S2, S3, 0);
		}
	f_switch3 = (f_switch1 + f_switch2) % 2;
	cout << "T3:" << endl;
	print_fancy(A2, Fq, T3, 2);
	cout << "S3:" << endl;
	print_fancy(A2, Fq, S3, 2);

	
	
	A2->element_move(S2, elt1, 0);
	int i;
	
	cout << "and now the powers of S2^4" << endl;
	A2->element_power_int_in_place(elt1, 4, 0);
	A2->element_move(elt1, elt2, 0);
	//A2->element_print_quick(elt1, cout);
	print_fancy(A2, Fq, elt1, 2);
	for (i = 2; i < 16; i++) {
		A2->element_move(elt2, elt1, 0);
		A2->element_power_int_in_place(elt1, i, 0);
		cout << "^" << i << ":" << endl;
		print_fancy(A2, Fq, elt1, 2);
		
		}

	int data_T2[] = {1, 43, 23, 54};
	int data_B[] = {37, 66, 25, 66};
	A2->make_element(elt1, data_T2, verbose_level - 1);
	A2->make_element(elt2, data_B, verbose_level - 1);
	A2->element_invert(elt2, elt3, 0);
	A2->element_mult(elt2, elt1, elt4, 0);
	A2->element_mult(elt4, elt3, elt5, 0);
	cout << "B T_2 B^{-1}=:" << endl;
	print_fancy(A2, Fq, elt5, 2);
	cout << "B^{-1}=:" << endl;
	print_fancy(A2, Fq, elt3, 2);
#endif

}

void print_fancy(action *A, finite_field *Fq, int *Elt, int n)
{
	int order;
	
	//A->element_print_quick(Elt, cout);
	order = A->element_order(Elt);
	PG_element_normalize_from_front(*Fq, Elt, 1, n * n);
	print_integer_matrix_width(cout, Elt, n, n, n, 3);
	cout << "order = " << order << endl;
}

void test18(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	action *A;
	orthogonal *O;
	finite_field *Fq;
	int i;
	
	int epsilon = 0;
	int n = 5;
	int q = 67;
	int f_semilinear = TRUE;
	int f_basis = FALSE;
		
	Fq = new finite_field;
	Fq->init(q, 0);
	A = new action;
	
	A->init_orthogonal_group(epsilon, n, Fq, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level - 1);
	
	A->lex_least_base_in_place(verbose_level - 1);
	
	O = A->G.AO->O;

	int nb_stab_gens = 2;
	int stab_gens[] = {
//19  ,  0  ,  6  , 41  , 19 ,
// 3  ,  1  ,  1  , 18  , 59 ,
// 0  , 65  ,  1  ,  2  ,  1 ,
//43  ,  1  , 59  , 57  , 30 ,
//54  ,  2  , 18  , 54  , 57 ,

// element of order 17:
9, 54, 0, 32, 31, 
3, 29, 10, 36, 1, 
0, 47, 0, 0, 0, 
59, 60, 0, 28, 36, 
36, 50, 0, 10, 58, 


// element of order 4:
49  ,  7  ,  2  , 37  ,  9 ,
33  , 14  , 10  ,  7  , 35 ,
30  , 17  , 29  , 54  , 35 ,
19  ,  4  , 26  ,  2  ,  2 ,
 8  , 17  , 12  ,  0  , 39 ,
};

	vector_ge *gens;
	int *Elt1;

	gens = new vector_ge;
	gens->init(A);
	gens->allocate(nb_stab_gens);
	
	Elt1 = NEW_int(A->elt_size_in_int);
	for (i = 0; i < nb_stab_gens; i++) {
		A->make_element(Elt1, stab_gens + i * 25, verbose_level);

		gens->copy_in(i, Elt1);
		
		cout << "stabilizer generator " << i << ":" << endl;
		A->element_print_quick(Elt1, cout);
		}

	sims *G;
	longinteger_object go;

	G = create_sims_from_generators_with_target_group_order_int(
			A, gens, 68, verbose_level);

	cout << "sims created" << endl;
	G->group_order(go);
	cout << "group order " << go << endl;
	
	cout << "test18: before delete G" << endl;
	delete G;
	cout << "test18: after delete G" << endl;

}

void test19(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	action *A;
	orthogonal *O;
	finite_field *Fq;
	int i, j;
	
	int epsilon = 0;
	int n = 5;
	int q = 67;
	int f_semilinear = TRUE;
	int f_basis = FALSE;

	Fq = new finite_field;

	Fq->init(q, 0);	
	A = new action;
	
	A->init_orthogonal_group(epsilon, n, Fq, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level - 1);
	
	A->lex_least_base_in_place(verbose_level - 1);
	
	O = A->G.AO->O;

	int nb_gens = 2;
	int gens_data[] = {
		37, 50, 18, 55, 23,
		7, 28, 26, 4, 0, 
		4, 53, 17, 49, 52, 
		59, 63, 4, 52, 7, 
		53, 47, 58, 15, 30, 


//p1:=GL(5,67)![ xi^(37), xi^(50), xi^(18), xi^(55), xi^(23) ,
//      xi^7, xi^(28), xi^(26), xi^4, xi^0 ,
//       xi^4, xi^(53), xi^(17), xi^(49), xi^(52) ,
//       xi^(59), xi^(63), xi^4, xi^(52), xi^7 ,
//       xi^(53), xi^(47), xi^(58), xi^(15), xi^(30)  ];

		40, 7, 62, 0, 42, 
		62, 34, 30, 8, 51, 
		9, 10, 14, 65, 20, 
		9, 38, 4, 34, 5, 
		0, 6, 23, 58, 25, 

//p2:=GL(5,67)![  xi^(40), xi^7, xi^(62), xi^0, xi^(42) ,
//       xi^(62), xi^(34), xi^(30), xi^8, xi^(51) ,
//       xi^9, xi^(10), xi^(14), xi^(65), xi^(20) ,
//       xi^9, xi^(38), xi^4, xi^(34), xi^5 ,
//       xi^0, xi^6, xi^(23), xi^(58), xi^(25)  ];
};

	vector_ge *gens;
	int *Elt1;
	int M5[25];
	gens = new vector_ge;
	gens->init(A);
	gens->allocate(nb_gens);
	
	Elt1 = NEW_int(A->elt_size_in_int);
	for (i = 0; i < nb_gens; i++) {

		cout << "stabilizer generator " << i << ":" << endl;
		for (j = 0; j < 25; j++) {
			M5[j] = Fq->alpha_power(gens_data[i * 25 + j]);
			}
		print_integer_matrix_width(cout, M5, 5, 5, 5, Fq->log10_of_q);
		A->make_element(Elt1, M5, verbose_level);

		gens->copy_in(i, Elt1);
		
		A->element_print_quick(Elt1, cout);
		}

	sims *G;
	longinteger_object go;

	G = create_sims_from_generators_without_target_group_order(
		A, gens, verbose_level);

	cout << "sims created" << endl;
	G->group_order(go);
	cout << "group order " << go << endl;
	
	// gives a group of order 1822431646435973760


	cout << "test19: before delete G" << endl;
	delete G;
	cout << "test19: after delete G" << endl;

}

void test20(int verbose_level)
// March 19, 2012
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3;
	int i;

	F = new finite_field;
	F->init(71 * 71, 0);
	A = new action;
	A->init_projective_group(
		2 /* n */, F, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	int data1[4] = {4426,3033,1143,1766};
	int data2[4];


	for (i = 0; i < 4; i++) {
		data2[i] = F->alpha_power(data1[i]);
		}

	cout << "making element:" << endl;
	A->make_element(Elt1, data2, FALSE);
	cout << "Elt1:" << endl;
	A->element_print_quick(Elt1, cout);
	A->element_print_as_permutation(Elt1, cout);

	A->element_mult(Elt1, Elt1, Elt2, 0);
	A->element_mult(Elt2, Elt1, Elt3, 0);

	cout << "Elt1^3:" << endl;
	A->element_print_quick(Elt3, cout);
	A->element_print_as_permutation(Elt3, cout);
	

}

void instant_insanity(int verbose_level)
{
	action *A;
	//longinteger_object Go;
	int i;
	int degree = 6;
	int nb_gens = 2;
	int gens[] = {
		0,4,3,1,2,5,
		1,4,3,5,0,2
		};
	vector_ge Gens;
	sims *G;
	longinteger_object go;
		
	A = new action;
	
	A->init_permutation_group(degree, verbose_level);
	Gens.init(A);
	Gens.allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		Gens.copy_in(i, gens + i * degree);
		}
	Gens.print(cout);
	
	G = create_sims_from_generators_without_target_group_order(A, 
		&Gens, verbose_level);
	G->group_order(go);
	cout << "Group order is " << go << endl;

	int cubes[] = {
		1,2,0,1,3,1,
		3,1,3,0,2,0,
		3,2,1,0,1,0,
		1,3,3,2,0,2
		};
	int image[] = {
		1,2,0,1,3,1,
		3,1,3,0,2,0,
		3,2,1,0,1,0,
		1,3,3,2,0,2
		};
	int cnt[4];
	int *Elt[4];

	for (i = 0; i < 4; i++) {
		Elt[i] = NEW_int(A->elt_size_in_int);
		}

	int nb_sol = 0;

	for (cnt[0] = 0; cnt[0] < 24; cnt[0]++) {
		G->element_unrank_int(cnt[0], Elt[0]);
		apply6(A, Elt[0], cubes + 0 * 6, image + 0 * 6);
		cout << "level 0: " << cnt[0] << " = ";
		A->element_print(Elt[0], cout); 
		cout << "image=" << endl;
		int_vec_print(cout, image + 0 * 6, 6);
		cout << endl;
		for (cnt[1] = 0; cnt[1] < 24; cnt[1]++) {
			G->element_unrank_int(cnt[1], Elt[1]);
			apply6(A, Elt[1], cubes + 1 * 6, image + 1 * 6);
			cout << "level 1: " << cnt[1] << " = ";
			A->element_print(Elt[1], cout); 
			cout << "image=" << endl;
			int_vec_print(cout, image + 0 * 6, 6);
			cout << endl;
			int_vec_print(cout, image + 1 * 6, 6);
			cout << endl;
			if (!test_sides(image + 0 * 6, image + 1 * 6)) {
				continue;
				}
			cout << "go deeper" << endl;
			for (cnt[2] = 0; cnt[2] < 24; cnt[2]++) {
				G->element_unrank_int(cnt[2], Elt[2]);
				apply6(A, Elt[2], cubes + 2 * 6, image + 2 * 6);
				cout << "level 2: " << cnt[2] << " = ";
				A->element_print(Elt[2], cout); 
				cout << "image=" << endl;
				int_vec_print(cout, image + 0 * 6, 6);
				cout << endl;
				int_vec_print(cout, image + 1 * 6, 6);
				cout << endl;
				int_vec_print(cout, image + 2 * 6, 6);
				cout << endl;
				if (!test_sides(image + 0 * 6, image + 2 * 6)) {
					continue;
					}
				if (!test_sides(image + 1 * 6, image + 2 * 6)) {
					continue;
					}
				cout << "go deeper" << endl;
				for (cnt[3] = 0; cnt[3] < 24; cnt[3]++) {
					G->element_unrank_int(cnt[3], Elt[3]);
					apply6(A, Elt[3], cubes + 3 * 6, image + 3 * 6);
					cout << "level 3: " << cnt[3] << " = ";
					A->element_print(Elt[3], cout); 
					cout << "image=" << endl;
					int_vec_print(cout, image + 0 * 6, 6);
					cout << endl;
					int_vec_print(cout, image + 1 * 6, 6);
					cout << endl;
					int_vec_print(cout, image + 2 * 6, 6);
					cout << endl;
					int_vec_print(cout, image + 3 * 6, 6);
					cout << endl;
					if (!test_sides(image + 0 * 6, image + 3 * 6)) {
						continue;
						}
					if (!test_sides(image + 1 * 6, image + 3 * 6)) {
						continue;
						}
					if (!test_sides(image + 2 * 6, image + 3 * 6)) {
						continue;
						}
					cout << "solution " << nb_sol << endl;
					nb_sol++;
					}
				}
			}
		}
	cout << "Number of solutions=" << nb_sol << endl;
#if 0
	schreier S;
	
	S.init(A);
	S.init_generators(Gens);
	S.compute_all_point_orbits(verbose_level);
	S.print_orbit_lengths(cout);
	S.print_orbit_length_distribution(cout);
#endif

	delete G;
	delete A;
}

int test_sides(int *v1, int *v2)
{
	if (v1[2] == v2[2]) {
		return FALSE;
		}
	if (v1[3] == v2[3]) {
		return FALSE;
		}
	if (v1[1] == v2[1]) {
		return FALSE;
		}
	if (v1[4] == v2[4]) {
		return FALSE;
		}
	return TRUE;
}

void apply6(action *A, int *Elt, int *src, int *image)
{
	int i, j;

	for (i = 0; i < 6; i++) {
		j = A->element_image_of(i, Elt, 0);
		image[j] = src[i];
		}
}

void test_bricks(int verbose_level)
// January 10, 2013
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int go;
	int *Elt1, *Elt2, *Elt3;
	int i;
	int q = 5;

	F = new finite_field;
	F->init(q, 0);
	A = new action;
	A->init_projective_group(3 /* n */, F, 
		FALSE /* f_semilinear */, 
		TRUE /* f_basis */, verbose_level);



	A->print_base();
	A->group_order(Go);
	go = Go.as_int();
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	int m_one;

	m_one = F->negate(1);
	int data1[9] = {1,0,0, 0,1,0, 0,0,1};
	int data2[9];
	vector_ge gens;
	int nb_gens = 5;
	int h;

	gens.init(A);
	gens.allocate(nb_gens);

	for (h = 0; h < nb_gens; h++) {
		for (i = 0; i < 9; i++) {
			data2[i] = data1[i];
			}
		if (h == 0) {
			data2[0 * 3 + 0] = m_one;
			}
		else if (h == 1) {
			data2[1 * 3 + 1] = m_one;
			}
		else if (h == 2) {
			data2[0 * 3 + 0] = 0;
			data2[0 * 3 + 1] = 1;
			data2[1 * 3 + 0] = m_one;
			data2[1 * 3 + 1] = 0;
			}
		else if (h == 3) {
			data2[0 * 3 + 2] = 1;
			}
		else if (h == 4) {
			data2[1 * 3 + 2] = 1;
			}
	

		cout << "making element:" << endl;
		A->make_element(Elt1, data2, FALSE);
		cout << "Elt1:" << endl;
		A->element_print_quick(Elt1, cout);
		A->element_print_as_permutation(Elt1, cout);

		A->element_move(Elt1, gens.ith(h), 0);

		} // next h
	cout << "generators:" << endl;
	for (h = 0; h < nb_gens; h++) {
		A->element_print_quick(gens.ith(h), cout);
		}
	
	sims *S;

	S = create_sims_from_generators_without_target_group_order(A, 
		&gens, verbose_level);

	S->group_order(Go);
	cout << "Created group of order " << Go << endl;
	
	brick_domain B;
	action *AB;
	int f_linear_action = TRUE;

	B.init(F, verbose_level);
	AB = new action;

	AB->induced_action_on_bricks(*A, &B,
			f_linear_action, verbose_level);

	cout << "created AB" << endl;
	AB->print_info();
}

void test_null_polarity_generator(int verbose_level)
// December 11, 2015
{
	finite_field *F;
	null_polarity_generator *N;

	F = new finite_field;

	F->init(2, 0);

	N = new null_polarity_generator;
	N->init(F, 6, verbose_level);

	delete N;
	delete F;
}

void test_linear_group_generator(int verbose_level)
// December 25, 2015
{
	finite_field *F;
	linear_group_description *Descr;
	linear_group *LG;
	const char *fname = "codes_18_9_4_8_stab_18_0.bin";
	const char *label = "ttp_9_4";

	cout << "test_linear_group_generator" << endl;
	
	F = new finite_field;
	F->init(4, 0);
	Descr = new linear_group_description;
	Descr->n = 9;
	Descr->F = F;
	Descr->f_semilinear = TRUE;
	Descr->f_subgroup_from_file = TRUE;
	Descr->subgroup_fname = fname;
	Descr->subgroup_label = label;

	LG = new linear_group;

	cout << "before LG->init" << endl;

	LG->init(Descr, verbose_level);
	
	cout << "after LG->init" << endl;
	
	delete LG;
	delete Descr;
	delete F;
}

void test_homogeneous_polynomials(int verbose_level)
// September 9, 2016
{
	finite_field *F;
	homogeneous_polynomial_domain *HPD;
	int n = 4;
	int d = 3;

	cout << "test_homogeneous_polynomials" << endl;
	
	F = new finite_field;
	F->init(4, 0);

	HPD = new homogeneous_polynomial_domain;

	HPD->init(F, n, d,
		FALSE /* f_init_incidence_structure */, verbose_level);

	delete HPD;
	delete F;
}



