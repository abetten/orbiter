// reflection.C
// 
// Anton Betten
// started:     03/10/2009
// last change: 03/12/2009
//
// 
//
// moved here from ACTION: Aug 5, 2014

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

#define MAX_M 100000
#define MAX_NB_ROOTS 1000
#define MAX_DIMENSION 8
#define TYPE_F4 1
#define TYPE_H4 2
#define TYPE_E6 3


int dimension = 0;
int nb_roots = 0;
int roots[MAX_NB_ROOTS * MAX_DIMENSION];
int nb_M = 0;
int *Elts;



void usage(int argc, char **argv);
int main(int argc, char **argv);
void choose_random_generator(sims *G, int *Elt, int verbose_level);
void do_it(int q, int type, int f_analyze, int f_words, int verbose_level);
void words(action *A, finite_field *F, sims *S, 
	int nb_simple_roots, int *simple_roots_new_labels, 
	const char *label, 
	int verbose_level);
void create_roots(finite_field *F, int type);
void create_roots_F4(finite_field *F);
void create_roots_H4(finite_field *F);
void create_roots_E6(finite_field *F);
int dot_product(finite_field *F, int *v1, int *v2);
void create_reflection(action *A, finite_field *F, int *Elt, 
	int root_idx, int nb_roots, int *v);
void print_matrix(int *M);
void print_matrix_extended(int *M);
void print_all_matrices(action *A);
int try_new_reflection(action *A, finite_field *F);
void create_new_reflection(action *A, finite_field *F);
int find(action *A, int *Elt);
void create_matrices(action *A, finite_field *F);
void create_group(action *A, int target_go, finite_field *F);
int create_reduced_expressions(action *A, finite_field *F, sims *S, 
	int nb_gens, int *gens, 
	int *&Nb_elements_by_length, int **&Elements_by_length, 
	int *&reduced_word_length, int **&reduced_word, 
	int verbose_level);
	
#if 0
// analyze_group.C
void analyze(action *A, sims *S, vector_ge *SG, vector_ge *gens2, int verbose_level);
void compute_regular_representation(action *A, sims *S, vector_ge *SG, int *&perm, int verbose_level);

void presentation(action *A, sims *S, vector_ge *gens, int *primes, int verbose_level);

#endif


void usage(int argc, char **argv)
{
	cout << "usage: " << argv[0] << " [options] q" << endl;
	cout << "where options can be:" << endl;
	cout << "-v <n>                   : verbose level n" << endl;
}

int main(int argc, char **argv)
{
	int i;
	int q;
	int verbose_level = 0;
	int type = 0;
	int f_analyze = FALSE;
	int f_words = FALSE;

	t0 = os_ticks();
	
	if (argc <= 1) {
		usage(argc, argv);
		exit(1);
		}
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		if (strcmp(argv[i], "-F4") == 0) {
			type = TYPE_F4;
			cout << "-F4 " << endl;
			}
		if (strcmp(argv[i], "-H4") == 0) {
			type = TYPE_H4;
			cout << "-H4 " << endl;
			}
		if (strcmp(argv[i], "-E6") == 0) {
			type = TYPE_E6;
			cout << "-E6 " << endl;
			}
		if (strcmp(argv[i], "-analyze") == 0) {
			f_analyze = TRUE;
			cout << "-analyze " << endl;
			}
		if (strcmp(argv[i], "-words") == 0) {
			f_words = TRUE;
			cout << "-words " << endl;
			}
		}
	q = atoi(argv[argc - 1]);
	
	do_it(q, type, f_analyze, f_words, verbose_level);
}

void choose_random_generator(sims *G, int *Elt, int verbose_level)
{
	int r;
	
	r = random_integer(nb_M);
	cout << "choosing random generator " << r << ":" << endl;
	G->A->element_move(Elts + r * G->A->elt_size_in_int, Elt, 0);
	G->A->element_print_quick(Elt, cout);
}

void do_it(int q, int type, int f_analyze, int f_words, int verbose_level)
{
	finite_field *F;
	action *A;
	matrix_group *M;
	int p, h;
	int n;
	int f_semilinear = FALSE;
	int f_basis = TRUE;
	int target_go;
	
	F = new finite_field;
	F->init(q, 0);
	A = new action;
	is_prime_power(q, p, h);
	if (h > 1)
		f_semilinear = TRUE;
	else
		f_semilinear = FALSE;
	
	if (type == TYPE_F4) {
		target_go = 1152;
		dimension = 4;
		}
	else if (type == TYPE_H4) {
		target_go = 14400;
		dimension = 4;
		}
	else if (type == TYPE_E6) {
		target_go = 51840;
		dimension = 8;
		}
	n = dimension + 1;
	A->init_projective_group(n,  F, 
		f_semilinear, f_basis, verbose_level);
	
	cout << "A->f_has_sims=" << A->f_has_sims << endl;

	M = A->G.matrix_grp;
	
	create_roots(M->GFq, type);
	
	cout << "created " << nb_roots << " roots" << endl;
	
	nb_M = 0;
	Elts = new int[MAX_M * A->elt_size_in_int];
	
	create_matrices(A, M->GFq);

	longinteger_object go;
	sims S;

	S.init(A);
	S.init_trivial_group(verbose_level - 2);
	S.build_up_subgroup_random_process(&S,  
		choose_random_generator, 
		verbose_level);
	S.group_order(go);
	cout << "group order " << go << endl;


	int i;

	int simple_roots_F4[] = { 11, 15, 3, 23 }; // maybe incorrect
	int simple_roots_H4[] = { 36, 48, 50, 30 }; // maybe incorrect
	int simple_roots_E6[] = { 70, 0, 2, 18, 30, 38 };
	const char *label;
	int *simple_roots = NULL;
	int *simple_roots_rank = NULL;
	int nb_simple_roots = 0;

	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

	if (type == TYPE_F4) {
		nb_simple_roots = sizeof(simple_roots_F4) / sizeof(int);
		simple_roots = simple_roots_F4;
		label = "F4";
		}
	else if (type == TYPE_H4) {
		nb_simple_roots = sizeof(simple_roots_H4) / sizeof(int);
		simple_roots = simple_roots_H4;
		label = "H4";
		}
	else if (type == TYPE_E6) {
		nb_simple_roots = sizeof(simple_roots_E6) / sizeof(int);
		simple_roots = simple_roots_E6;
		label = "E6";
		}
	simple_roots_rank = NEW_int(nb_simple_roots);

	
	for (i = 0; i < nb_simple_roots; i++) {

		create_reflection(A, M->GFq, Elt, simple_roots[i], nb_roots, roots + simple_roots[i] * dimension);
		simple_roots_rank[i] = S.element_rank_int(Elt);		
		}

	if (f_words) {
		words(A, M->GFq, &S, 
			nb_simple_roots, simple_roots_rank, 
			label, verbose_level);
		}
	
#if 0
	//exit(1);
	create_group(A, target_go, M->GFq);
	
	int i;
	int *new_labels;
	sims S;
	int *Elt2;
	longinteger_object go;
	vector_ge SG;
	
	Elt2 = new int[A->elt_size_in_int];
	S.init(A);
	S.init_trivial_group(verbose_level - 2);
	for (i = 0; i < nb_M; i++) {
		S.strip_and_add(Elts + i * A->elt_size_in_int, Elt2, 0);
		}
	S.group_order(go);
	cout << "group order " << go << endl;
	S.print_transversal_lengths();
	int *tl = new int [A->base_len];
	S.extract_strong_generators_in_order(SG, tl, 0);
	cout << "strong generators:" << endl;
	for (i = 0; i < SG.len; i++) {
		cout << "generator " << i << endl;
		A->print(cout, SG.ith(i));
		}
	
	for (i = 0; i < nb_M; i++) {
		S.element_unrank_int(i, Elt2);
		cout << "element " << i << ":" << endl;
		A->print(cout, Elt2);
		}
		
	new_labels = new int[nb_M];
	for (i = 0; i < nb_M; i++) {
		new_labels[i] = S.element_rank_int(Elts + i * A->elt_size_in_int);
		}
	
	cout << "old rank : new rank" << endl;
	for (i = 0; i < nb_M; i++) {
		cout << setw(5) << i << " : " << setw(5) << new_labels[i] << endl;
		}
	
	//exit(1);
	
	int nb_simple_roots;
	int simple_roots_F4_all[24];
	int simple_roots_F4[4] = { 11, 15, 3, 23 };
	int simple_roots_H4[4] = { 36, 48, 50, 30 };
	int *simple_roots_new_labels;
	const char *label = NULL;
	int *simple_roots;
	
	if (type == TYPE_F4) {
		if (TRUE) {
			simple_roots = simple_roots_F4;
			nb_simple_roots = 4;
			}
		else {
			for (i = 0; i < 24; i++) {
				simple_roots_F4_all[i] = i;
				}
			simple_roots = simple_roots_F4_all;
			nb_simple_roots = 24;
			}
		label = "F4";
		}
	else if (type == TYPE_H4) {
		simple_roots = simple_roots_H4;
		nb_simple_roots = 4;
		label = "H4";
		}
	
	simple_roots_new_labels = new int[nb_simple_roots];
	
	for (i = 0; i < nb_simple_roots; i++) {
		simple_roots_new_labels[i] = new_labels[simple_roots[i]];
		}
	cout << "simple_roots_new_labels: ";
	int_vec_print(cout, simple_roots_new_labels, 4);
	cout << endl;

	for (i = 0; i < nb_simple_roots; i++) {
		S.element_unrank_int(simple_roots_new_labels[i], Elt2);
		cout << "generator " << i << ":" << endl;
		A->print(cout, Elt2);
		}
	
	if (f_analyze) {
		vector_ge gens2;
		int primes[] = {2, 2,2,2,2, 3,3, 2,2};
		
		analyze(A, &S, &SG, &gens2, verbose_level);
		presentation(A, /*M->GFq,*/ &S, &gens2, primes, verbose_level);
		}
	if (f_words) {
		words(A, M->GFq, &S, &SG, 
			nb_simple_roots, simple_roots_new_labels, 
			label, verbose_level);
		}

#endif
}


void words(action *A, finite_field *F, sims *S, 
	int nb_simple_roots, int *simple_roots_new_labels, 
	const char *label, 
	int verbose_level)
{
	int *Elt1;
	int i, j;
	int *Nb_elements_by_length;
	int **Elements_by_length; 
	int *reduced_word_length;
	int **reduced_word;
	int max_length, l, a, N;
	
	Elt1 = new int[A->elt_size_in_int];
	max_length = create_reduced_expressions(A, F, S, nb_simple_roots, simple_roots_new_labels, 
		Nb_elements_by_length, Elements_by_length, 
		reduced_word_length, reduced_word, 
		0/*verbose_level*/);
#if 0
	for (i = 0; i < nb_M; i++) {
		S.element_unrank_int(i, Elt1);
		cout << "element " << i << ":" << endl;
		A->print(cout, Elt1);
		cout << "reduced expression: ";
		int_vec_print(cout, reduced_word[i], reduced_word_length[i]);
		cout << endl;
		}
#endif
	N = 0;
	for (l = 0; l <= max_length; l++) {
		cout << "There are " << Nb_elements_by_length[l] << " words of length " << l << endl;
		N += Nb_elements_by_length[l];
		}
	cout << "the sum of the number of reduced words is " << N << endl;
	
	for (l = 0; l <= max_length; l++) {
		cout << "There are " << Nb_elements_by_length[l] << " words of length " << l << ":" << endl;
		cout << "length : i : element : reduced expression" << endl;		
		for (i = 0; i < Nb_elements_by_length[l]; i++) {
			a = Elements_by_length[l][i];
			S->element_unrank_int(a, Elt1);
			cout << l << " : " << i << " : " << a << ":" << endl;
			//A->print(cout, Elt1);
			//cout << "reduced expression: ";
			int_vec_print(cout, reduced_word[a], reduced_word_length[a]);
			cout << endl;
			}
		}

	char fname[1000];
	
	sprintf(fname, "reduced_words_%s.txt", label);
	{
	ofstream f(fname);
	
	for (l = 0; l <= max_length; l++) {
		cout << "There are " << Nb_elements_by_length[l] << " words of length " << l << ":" << endl;
		for (i = 0; i < Nb_elements_by_length[l]; i++) {
			a = Elements_by_length[l][i];
			f << setw(3) << l << " ";
			for (j = 0; j < l; j++) {
				f << setw(3) << reduced_word[a][j];
				}
			f << endl;
			}
		}
	f << -1 << endl;
	}
	cout << "written file " << fname << " of size " << file_size(fname) << endl;





	delete [] Elt1;
}

void create_roots(finite_field *F, int type)
{
	int i, j;

	if (type == TYPE_F4) {
		create_roots_F4(F);
		}
	else if (type == TYPE_H4) {
		create_roots_H4(F);
		}
	else if (type == TYPE_E6) {
		create_roots_E6(F);
		}

	cout << "the " << nb_roots << " roots are:" << endl;
	for (i = 0; i < nb_roots; i++) {
		cout << setw(3) << i << " : ";
		for (j = 0; j < dimension; j++) {
			cout << setw(3) << roots[i * dimension + j];
			}
		cout << endl;
		}
	//exit(1);
}

void create_roots_F4(finite_field *F)
{
	int i, j, k, i1, i2, j1, j2, j3, j4, n;
	int v[4];
	int one, m_one;
	
	one = 1;
	m_one = F->negate(one);
	n = 0;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (k = 0; k < 4; k++)
				v[k] = 0;
			if (j == 0)
				v[i] = one;
			else
				v[i] = m_one;
			for (k = 0; k < 4; k++)
				roots[n * 4 + k] = v[k];
			n++;
			}
		} 
	for (i1 = 0; i1 < 4; i1++) {
		for (i2 = i1 + 1; i2 < 4; i2++) {
			for (j1 = 0; j1 < 2; j1++) {
				for (j2 = 0; j2 < 2; j2++) { 	
					for (k = 0; k < 4; k++)
						v[k] = 0;
					if (j1 == 0)
						v[i1] = one;
					else
						v[i1] = m_one;
					if (j2 == 0)
						v[i2] = one;
					else
						v[i2] = m_one;
					for (k = 0; k < 4; k++)
						roots[n * 4 + k] = v[k];
					n++;
					}
				}
			}
		} 
	for (j1 = 0; j1 < 2; j1++) {
		for (j2 = 0; j2 < 2; j2++) { 	
			for (j3 = 0; j3 < 2; j3++) {
				for (j4 = 0; j4 < 2; j4++) {
					for (k = 0; k < 4; k++)
						v[k] = 0;
					if (j1 == 0)
						v[0] = one;
					else
						v[0] = m_one;
					if (j2 == 0)
						v[1] = one;
					else
						v[1] = m_one;
					if (j3 == 0)
						v[2] = one;
					else
						v[2] = m_one;
					if (j4 == 0)
						v[3] = one;
					else
						v[3] = m_one;
					for (k = 0; k < 4; k++)
						roots[n * 4 + k] = v[k];
					n++;
					}
				}
			}
		}
	nb_roots = n;
}

void create_roots_H4(finite_field *F)
{
	int i, j, k, j1, j2, j3, j4, n;
	int v[4];
	int L[4], P[4], sgn;
	int one, m_one, half, quarter, c, c2, tau, tau_inv, a, b, m_a, m_b, m_half;
	
	one = 1;
	m_one = F->negate(one);
	half = F->inverse(2);
	quarter = F->inverse(4);
	n = 0;
	for (c = 1; c < F->q; c++) {
		c2 = F->mult(c, c);
		if (c2 == 5)
			break;
		}
	if (c == F->q) {
		cout << "create_roots_H4: the field of order " << F->q << " does not contain a square root of 5" << endl;
		exit(1);
		}
	tau = F->mult(F->add(1, c), half);
	tau_inv = F->inverse(tau);
	a = F->mult(F->add(1, c), quarter);
	b = F->mult(F->add(m_one, c), quarter);
	m_a = F->negate(a);
	m_b = F->negate(b);
	m_half = F->negate(half);
	cout << "a=" << a << endl;
	cout << "b=" << b << endl;
	cout << "c=" << c << endl;
	cout << "tau=" << tau << endl;
	cout << "tau_inv=" << tau_inv << endl;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 2; j++) {
			for (k = 0; k < 4; k++)
				v[k] = 0;
			if (j == 0)
				v[i] = one;
			else
				v[i] = m_one;
			for (k = 0; k < 4; k++)
				roots[n * 4 + k] = v[k];
			n++;
			}
		} 
	for (j1 = 0; j1 < 2; j1++) {
		for (j2 = 0; j2 < 2; j2++) { 	
			for (j3 = 0; j3 < 2; j3++) {
				for (j4 = 0; j4 < 2; j4++) {
					for (k = 0; k < 4; k++)
						v[k] = 0;
					if (j1 == 0)
						v[0] = one;
					else
						v[0] = m_one;
					if (j2 == 0)
						v[1] = one;
					else
						v[1] = m_one;
					if (j3 == 0)
						v[2] = one;
					else
						v[2] = m_one;
					if (j4 == 0)
						v[3] = one;
					else
						v[3] = m_one;
					for (k = 0; k < 4; k++)
						roots[n * 4 + k] = F->mult(half, v[k]);
					n++;
					}
				}
			}
		}
	for (j1 = 0; j1 < 2; j1++) {
		for (j2 = 0; j2 < 2; j2++) { 	
			for (j3 = 0; j3 < 2; j3++) {
				for (k = 0; k < 4; k++)
					v[k] = 0;
				if (j1 == 0)
					v[0] = a;
				else
					v[0] = m_a;
				if (j2 == 0)
					v[1] = half;
				else
					v[1] = m_half;
				if (j3 == 0)
					v[2] = b;
				else
					v[2] = m_b;
				first_lehmercode(4, L);
				while (TRUE) {
					lehmercode_to_permutation(4, L, P);
					sgn = perm_signum(P, 4);
					if (sgn == 1) {
						for (k = 0; k < 4; k++)
							roots[n * 4 + k] = v[P[k]];
						n++;
						}
					if (!next_lehmercode(4, L))
						break;
					}
				}
			}
		}
	nb_roots = n;
}

void create_roots_E6(finite_field *F)
{
	int i, j, s1, s2, k, n, h;
	int v[8];
	int one, m_one, half;
	
	one = 1;
	m_one = F->negate(one);
	half = F->inverse(2);
	n = 0;
	for (i = 0; i < 5; i++) {
		for (j = i + 1; j < 5; j++) {
			for (s1 = 0; s1 < 2; s1++) {
				for (s2 = 0; s2 < 2; s2++) {
					for (k = 0; k < 8; k++)
						v[k] = 0;
					if (s1) {
						v[i] = m_one;
						}
					else {
						v[i] = one;
						}
					if (s2) {
						v[j] = m_one;
						}
					else {
						v[j] = one;
						}

					for (k = 0; k < 8; k++)
						roots[n * 8 + k] = v[k];
					n++;

					}
				}
			
			}
		}
	int w[5];
	int N;
	int cnt;
	
	N = nb_AG_elements(5, 2);

	for (i = 0; i < N; i++) {
		AG_element_unrank(2, w, 1, 5, i);
		cnt = 0;
		for (j = 0; j < 5; j++) {
			cnt += w[j];
			}
		if (ODD(cnt))
			continue;

		v[7] = 1;
		v[6] = m_one;
		v[5] = m_one;
		for (j = 0; j < 5; j++) {
			if (w[j]) {
				v[j] = m_one;
				}
			else {
				v[j] = one;
				}
			}
		for (j = 0; j < 8; j++) {
			v[j] = F->mult(v[j], half);
			}
		
		for (h = 0; h < 2; h++) {
			if (h) {
				for (j = 0; j < 8; j++) {
					v[j] = F->negate(v[j]);
					}
				}
			for (k = 0; k < 8; k++)
				roots[n * 8 + k] = v[k];
			n++;

			}

		}
	
	nb_roots = n;
}



int dot_product(finite_field *F, int *v1, int *v2)
{
	int a, i;
	
	a = 0;
	a = F->mult(v1[0], v2[0]);
	for (i = 1; i < dimension; i++) {
		a = F->add(a, F->mult(v1[i], v2[i]));
		}
	return a;
}

void create_reflection(action *A, finite_field *F, int *Elt, 
	int root_idx, int nb_roots, int *v)
{
	int i, j, k;
	int *N, *M, *w, d1, d1v, d2, d3;
	
	N = NEW_int(dimension * dimension);
	M = NEW_int((dimension + 1) * (dimension + 1));
	w = NEW_int(dimension);
	
	//cout << "create reflection, root = " << root_idx << " = ";
	//int_vec_print(cout, v, dimension);
	//cout << endl;
	for (i = 0; i < dimension * dimension; i++) {
		N[i] = 0;
		}
	for (i = 0; i < (dimension + 1) * (dimension + 1); i++) {
		M[i] = 0;
		}
	d1 = dot_product(F, v, v);
	d1v = F->inverse(d1);
	//cout << "d1=" << d1 << endl;
	for (j = 0; j < dimension; j++) {
		//cout << "j=" << j << endl;
		for (k = 0; k < dimension; k++) {
			w[k] = 0;
			}
		w[j] = 1;
		d2 = dot_product(F, w, v);
		d3 = F->mult(F->mult(2, d2), d1v);
		//cout << "d2=" << d2 << endl;
		//cout << "d3=" << d3 << endl;
		for (k = 0; k < dimension; k++) {
			w[k] = F->add(w[k], F->negate(F->mult(d3, v[k])));
			}
		for (i = 0; i < dimension; i++) {
			N[i * dimension + j] = w[i];
			}
		}
	cout << "root " << root_idx << " / " << nb_roots << " = ";
	int_vec_print(cout, v, dimension);
	cout << " corresponds to the matrix" << endl;
	print_matrix(N);
	for (i = 0; i < dimension; i++) {
		for (j = 0; j < dimension; j++) {
			M[i * (dimension + 1) + j] = N[i * dimension + j];
			}
		}
	M[dimension * (dimension + 1) + dimension] = 1;
	A->make_element(Elt, M, 0);
	A->element_print(Elt, cout);
	cout << endl;
	
#if 0
	if (!test_if_orthogonal_matrix(M)) {
		cout << "matrix is not orthogonal" << endl;
		exit(1);
		}
#endif
	FREE_int(N);
	FREE_int(M);
	FREE_int(w);
}

void print_matrix(int *M)
{
	int i, j;
	
	for (i = 0; i < dimension; i++) {
		for (j = 0; j < dimension; j++) {
			cout << setw(4) << M[i * dimension + j];
			}
		cout << endl;
		}
}

void print_matrix_extended(int *M)
{
	int i, j;
	
	for (i = 0; i < dimension + 1; i++) {
		for (j = 0; j < dimension + 1; j++) {
			cout << setw(4) << M[i * (dimension + 1) + j];
			}
		cout << endl;
		}
}

void print_all_matrices(action *A)
{
	int i;
	
	for (i = 0; i < nb_M; i++) {
		cout << "matrix " << i << ":" << endl;
		print_matrix_extended(Elts + i * A->elt_size_in_int);
		}
}


int try_new_reflection(action *A, finite_field *F)
{
	int r1, r2, j;
	int *M1, *M2;
	int *Elt;
	
	Elt = new int[A->elt_size_in_int];
	
	r1 = random_integer(nb_M);
	while (TRUE) {
		r2 = random_integer(nb_M);
		if (r2 != r1)
			break;
		}
	//cout << "try_new_reflection r1=" << r1 << " r2=" << r2 << endl;
	M1 = Elts + r1 * A->elt_size_in_int;
	M2 = Elts + r2 * A->elt_size_in_int;
	A->mult(M1, M2, Elt);

#if 0
	//cout << "new matrix:" << endl;
	//print_matrix(M3);
	if (!test_if_orthogonal_matrix(M3)) {
		cout << "matrix is not orthogonal" << endl;
		print_matrix(M3);
		cout << "try_new_reflection r1=" << r1 << " r2=" << r2 << endl;
		exit(1);
		}
#endif
	j = find(A, Elt);
	//cout << "find returns " << j << endl;
	if (j == -1) {
		A->move(Elt, Elts + nb_M * A->elt_size_in_int);
		nb_M++;
		delete [] Elt;
		return TRUE;
		}
	delete [] Elt;
	return FALSE;
}

void create_new_reflection(action *A, finite_field *F)
{
	while (!try_new_reflection(A, F)) {
		}
}

int find(action *A, int *Elt)
{
	int i, j;
	int *Elt_i;
	
	for (i = 0; i < nb_M; i++) {
		Elt_i = Elts + i * A->elt_size_in_int;
		for (j = 0; j < A->elt_size_in_int; j++) {
			if (Elt_i[j] != Elt[j])
				break;
			}
		if (j == A->elt_size_in_int) {
			return i;
			}
		}
	return -1;
}

void create_matrices(action *A, finite_field *F)
{
	int k, j;
	int *R;
	
	R = new int[A->elt_size_in_int];
	nb_M = 0;
	for (k = 0; k < nb_roots; k++) {
		create_reflection(A, F, R, k, nb_roots, roots + k * dimension);
		j = find(A, R);
		//cout << "root " << k << " find returns " << j << endl;
		if (j == -1) {
			cout << "root " << k << " yields matrix " << nb_M << endl;
			A->move(R, Elts + nb_M * A->elt_size_in_int);
			nb_M++;
			}
		else {
			cout << "root " << k << " yields matrix " << j << endl;
			}
		}
	print_all_matrices(A);
	delete [] R;
}

void create_group(action *A, int target_go, finite_field *F)
{
	int i, cnt = 0;
	
	while (nb_M < target_go) {
		//cout << "cnt=" << cnt << endl;
		cnt++;
		create_new_reflection(A, F);
		cout << "new group order = " << nb_M << endl;
		if (nb_M == target_go) {
			cout << "reached target group order " << target_go << endl;
			print_all_matrices(A);
			}
		if (nb_M > target_go) {
			cout << "too many matrices" << endl;
			cout << "new matrix" << endl;
			print_matrix_extended(Elts + (nb_M - 1) * A->elt_size_in_int);
			exit(1);
			}
		}
	for (i = 0; i < 10000; i++) {
		//cout << "cnt=" << cnt << endl;
		cnt++;
		try_new_reflection(A, F);
		//cout << "nb_M = " << nb_M << endl;
#if 0
		if (nb_M > target_go) {
			cout << "too many matrices" << endl;
			cout << "new matrix" << endl;
			print_matrix_5x5(Elts + (nb_M - 1) * A->elt_size_in_int);
			exit(1);
			}
#endif
		}
}


int create_reduced_expressions(action *A, finite_field *F, sims *S, 
	int nb_gens, int *gens, 
	int *&Nb_elements_by_length, int **&Elements_by_length, 
	int *&reduced_word_length, int **&reduced_word, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = 0;
	int **W;
	longinteger_object go;
	int goi;
	int length;
	int nb_words;
	int word[1000];
	int word2[1000];
	int i, j, ii, a, b, c;
	int *Elt1;
	int max_length = 100;
	
	Elt1 = new int[A->elt_size_in_int];
	
	S->group_order(go);
	goi = go.as_int();
	reduced_word_length = new int[goi];
	reduced_word = new pint[goi];
	
	for (i = 0; i < goi; i++) {
		reduced_word_length[i] = -1;
		}
	
	W = new pint[max_length + 1];
	
	Nb_elements_by_length = new int[max_length];
	Elements_by_length = new pint[max_length];
	
	for (length = 0; length < max_length; length++) {
		nb_words = i_power_j(nb_gens, length);
		cout << endl << "computing words whose reduced length is " << length << endl;
		//cout << "nb_words=" << nb_words << endl;


		Nb_elements_by_length[length] = 0;
		Elements_by_length[length] = new int[goi];
		
		if (length == 0) {
			W[length] = new int[nb_words];
			for (i = 0; i < nb_words; i++) {
				if (length) {
					AG_element_unrank(nb_gens, word, 1, length, i);
					}
				for (j = 0; j < length; j++) {
					a = word[j];
					b = gens[a];
					word2[j] = b;
					}
				S->evaluate_word_int(length, word2, Elt1, 0);
				a = S->element_rank_int(Elt1);
				W[length][i] = a;
				if (reduced_word_length[a] == -1) {
					reduced_word_length[a] = length;
					reduced_word[a] = new int[length];
					for (ii = 0; ii < length; ii++) {
						reduced_word[a][ii] = word[ii];
						}
					//reduced_word_idx[a] = i;
					
					Elements_by_length[length][Nb_elements_by_length[length]] = a;
					Nb_elements_by_length[length]++;
					}
				}
			}
		else {
			for (i = 0; i < Nb_elements_by_length[length - 1]; i++) {
				a = Elements_by_length[length - 1][i];
				for (ii = 0; ii < length - 1; ii++) {
					word[ii] = reduced_word[a][ii];
					}
				//b = reduced_word_idx[a];
				//if (length - 1) {
					//AG_element_unrank(nb_gens, word, 1, length - 1, b);
					//}
				if (f_v) {
					cout << "extending word ";
					int_vec_print(cout, word, length - 1);
					cout << endl;
					}
				for (c = 0; c < nb_gens; c++) {
					word[length - 1] = c;
					if (f_v) {
						cout << "trial :";
						int_vec_print(cout, word, length);
						cout << endl;
						}
					for (j = 0; j < length; j++) {
						word2[j] = gens[word[j]];
						}
					S->evaluate_word_int(length, word2, Elt1, 0);
					a = S->element_rank_int(Elt1);
					if (reduced_word_length[a] == -1) {
						if (f_v) {
							cout << "yields new group element " << a << endl;
							}
						reduced_word_length[a] = length;

						reduced_word[a] = new int[length];
						for (ii = 0; ii < length; ii++) {
							reduced_word[a][ii] = word[ii];
						}
						
						//AG_element_rank(nb_gens, word, 1, length, b);
						//reduced_word_idx[a] = b;
						//cout << "new group element" << endl;
					
						Elements_by_length[length][Nb_elements_by_length[length]] = a;
						Nb_elements_by_length[length]++;
						}
					}
				}
			}
		
		cout << "found " << Nb_elements_by_length[length] << " elements of length " << length << endl;
		for (i = 0; i < Nb_elements_by_length[length]; i++) {
			a = Elements_by_length[length][i];
			if (reduced_word_length[a] != length) {
				cout << "reduced_word_length[a] != length" << endl;
				exit(1);
				}
			cout << setw(5) << a << " = ";
			int_vec_print(cout, reduced_word[a], length);
			cout << endl;
			}
		N += Nb_elements_by_length[length];
		if (N == goi)
			break;
		}
	
	delete [] Elt1;
	return length;
}


void presentation(action *A, sims *S, vector_ge *gens, int *primes, int verbose_level)
{
	int *Elt1, *Elt2, *Elt3, *Elt4;
	int i, j, k, l, a, b;
	int word[100];
	int word_list[10000];
	int inverse_word_list[10000];
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	
	l = gens->len;
	
	cout << "presentation of length " << l << endl;
	cout << "primes: ";
	int_vec_print(cout, primes, l);
	cout << endl;
	
	
	// replace g5 by  g5 * g3:
	A->mult(gens->ith(5), gens->ith(3), Elt1);
	A->move(Elt1, gens->ith(5));
	
	
	// replace g7 by  g7 * g4:
	A->mult(gens->ith(7), gens->ith(4), Elt1);
	A->move(Elt1, gens->ith(7));
	
	
	
	for (i = 0; i < l; i++) {
		cout << "generator " << i << ":" << endl;
		A->print(cout, gens->ith(i));
		cout << endl;
		}
	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->element_power_int_in_place(Elt1, primes[i], 0);
		cout << "generator " << i << " to the power " << primes[i] << ":" << endl;
		A->print(cout, Elt1);
		cout << endl;
		}
	for (i = 0; i < 1152; i++) {
		inverse_word_list[i] = -1;
		}
	for (i = 0; i < 1152; i++) {
		A->one(Elt1);
		j = i;
		for (k = 0; k < l; k++) {
			b = j % primes[k];
			word[k] = b;
			j = j - b;
			j = j / primes[k];
			}
		for (k = 0; k < l; k++) {
			b = word[k];
			while (b) {
				A->mult(Elt1, gens->ith(k), Elt2);
				A->move(Elt2, Elt1);
				b--;
				}
			}
		A->move(Elt1, Elt2);
		a = S->element_rank_int(Elt2);
		word_list[i] = a;
		inverse_word_list[a] = i;
		cout << "word " << i << " = ";
		int_vec_print(cout, word, 9);
		cout << " gives " << endl;
		A->print(cout, Elt1);
		cout << "which is element " << word_list[i] << endl;
		cout << endl;
		}
	cout << "i : word_list[i] : inverse_word_list[i]" << endl;
	for (i = 0; i < 1152; i++) {
		cout << setw(5) << i << " : " << setw(5) << word_list[i] << " : " << setw(5) << inverse_word_list[i] << endl;
		}
	for (i = 0; i < l; i++) {
		A->move(gens->ith(i), Elt1);
		A->invert(Elt1, Elt2);
		for (j = 0; j < i; j++) {
			A->mult(Elt2, gens->ith(j), Elt3);
			A->mult(Elt3, Elt1, Elt4);
			cout << "g_" << j << "^{g_" << i << "} =" << endl;
			A->print(cout, Elt4);
			a = S->element_rank_int(Elt4);
			cout << "which is element " << a << " which is word " << inverse_word_list[a] << endl;
			cout << endl;
			}
		cout << endl;
		}

	delete [] Elt1;
	delete [] Elt2;
	delete [] Elt3;
	delete [] Elt4;
}


