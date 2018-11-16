// code_cosets.C
// 
// Anton Betten
// 
//
// June 9, 2016
//
// based on ALGEBRA/projective_group.C 
//
//

#include "orbiter.h"

void make_generator_matrix(finite_field *F,
		int *G, int k, int n, int *code, int verbose_level);
void projective_group(int n, int q, int f_semilinear, int verbose_level);
void grassmannian(int n, int k, finite_field *F, int verbose_level);
void read_group(const char *fname, int verbose_level);


	finite_field *F = NULL;
	grassmann *Gr = NULL;
	action *A = NULL;
	action *A_on_cols = NULL;
	action *A_perm = NULL;
	longinteger_object Go1, Go2, Go3, Go4, Nb_cosets;
	strong_generators *SG = NULL;
	sims *S = NULL;
	sims *S_perm = NULL;
	int *PElt = NULL;
	int *PElt1 = NULL;
	int *PElt2 = NULL;
	sims *S_big = NULL;

	


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i, j;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q = 0;
	int f_file = FALSE;
	const char *fname = NULL;
	int f_semilinear = FALSE;
	int f_code;
	int code[1000];
	int code2[1000];
	int code_sz = 0;
	int f_subspace_classify = FALSE;
		
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-semilinear") == 0) {
			f_semilinear = TRUE;
			cout << "-semilinear" << endl;
			}
		else if (strcmp(argv[i], "-code") == 0) {
			f_code = TRUE;
			while (TRUE) {
				code[code_sz] = atoi(argv[++i]);
				if (code[code_sz] == -1) {
					break;
					}
				code_sz++;
				}
			cout << "-code ";
			int_vec_print(cout, code, code_sz);
			cout << endl;
			}
		else if (strcmp(argv[i], "-subspace_classify") == 0) {
			f_subspace_classify = TRUE;
			cout << "-subspace_classify" << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}

	int f_v3 = (verbose_level >= 3);


	cout << "before projective_group" << endl;
	projective_group(n, q, f_semilinear, verbose_level);

	cout << "before grassmannian" << endl;
	grassmannian(code_sz, code_sz - n, F, verbose_level);
	
	cout << "before read_group" << endl;
	read_group(fname, verbose_level);
	cout << "after read_group" << endl;

	if (f_code) {
		int *Perms;
		int *Elt;
		int d;
		int *G; // [code_sz * code_sz]

		G = NEW_int(code_sz * code_sz);
		make_generator_matrix(F, G, n, code_sz, code, 0 /* verbose_level */);

		cout << "The original check matrix is:" << endl;
		int_matrix_print(G, n, code_sz);


		F->perp_standard(code_sz, n, G, 0 /* verbose_level */);

		cout << "the generator matrix of the original code is:" << endl;
		int_matrix_print(G + n * code_sz, code_sz, code_sz - n);


		d = F->code_minimum_distance(code_sz,
				code_sz - n, G + n * code_sz, 0 /* verbose_level */);
		cout << "the minimum distance of the code is " << d << endl;

		Perms = NEW_int(SG->gens->len * code_sz);
		A_on_cols = new action;

		cout << "creating action on the columns:" << endl;
		A_on_cols->induced_action_by_restriction(*A, 
			FALSE /* f_induce_action */, NULL /* old_G */, 
			code_sz, code, verbose_level);
		cout << "action on the columns has been created" << endl;

		
		for (i = 0; i < SG->gens->len; i++) {

			cout << "generator " << i << " / " << SG->gens->len << ":" << endl;
			Elt = SG->gens->ith(i);
			for (j = 0; j < code_sz; j++) {
				Perms[i * code_sz + j] =
						A_on_cols->element_image_of(j,
								Elt, 0 /*verbose_level */);
				}
			perm_print_list(cout, Perms + i * code_sz, code_sz);
			cout << endl;
			}

		
		A_perm = new action;
		A_perm->init_symmetric_group(code_sz, 0 /* verbose_level */);

		S_big = A_perm->Sims;
		S_big->group_order(Go4);
		
		PElt = NEW_int(A_perm->elt_size_in_int);
		PElt1 = NEW_int(A_perm->elt_size_in_int);
		PElt2 = NEW_int(A_perm->elt_size_in_int);
		vector_ge *gens;
		gens = new vector_ge;
		gens->init(A_perm);
		gens->allocate(SG->gens->len);
		for (i = 0; i < SG->gens->len; i++) {
			cout << "creating element " << i << " / "
					<< SG->gens->len << ":" << endl;
			A_perm->make_element(PElt,
					Perms + i * code_sz, 0 /*verbose_level*/);
			cout << "as permutation:" << endl;
			A_perm->element_print_quick(PElt, cout);
			cout << endl;
			A_perm->element_move(PElt, gens->ith(i), 0 /* verbose_level */);
			}

		cout << "creating group S_perm of order " << Go2 << endl;
		S_perm = create_sims_from_generators_with_target_group_order(A_perm, 
			gens, Go2, 0 /* verbose_level */);
		S_perm->group_order(Go3);
		cout << "created group S_perm of order " << Go3 << endl;

		longinteger_domain D;

		D.integral_division_exact(Go4, Go3, Nb_cosets);

		int nb_cosets, a, h, rk;
		//int *G;
		int *Ranks;


		nb_cosets = Nb_cosets.as_int();

		Ranks = NEW_int(nb_cosets);
		
		for (i = 0; i < nb_cosets; i++) {
			cout << "coset " << i << " / " << nb_cosets << " is ";
			A_perm->coset_unrank(S_big, S_perm, i, PElt1,
					0 /* verbose_level */);
			A_perm->element_print(PElt1, cout);
			cout << endl;


			A_perm->element_invert(PElt1, PElt2, 0);
			for (j = 0; j < code_sz; j++) {
				a = code[j];
				h = PElt2[j];
				code2[h] = a;
				}
			cout << "image code: ";
			int_vec_print(cout, code2, code_sz);
			cout << endl;
			
			
			make_generator_matrix(F, G, n, code_sz, code2,
					0 /* verbose_level */);
			cout << "The image code is:" << endl;
			int_matrix_print(G, n, code_sz);


			F->perp_standard(code_sz, n, G, 0 /* verbose_level */);

			cout << "the generator matrix of the dual is:" << endl;
			int_matrix_print(G + n * code_sz, code_sz - n, code_sz);
					
			F->Gauss_easy(G + n * code_sz, code_sz - n, code_sz);


			cout << "RREF:" << endl;
			int_matrix_print(G + n * code_sz, code_sz - n, code_sz);


			rk = Gr->rank_int_here(G + n * code_sz, 2 /*verbose_level*/);
			
			Ranks[i] = rk;
			
			cout << "rank of subspace = " << rk << endl;
			}


#if 1
		cout << "The ranks of the code images are:" << endl; 
		for (i = 0; i < nb_cosets; i++) {
			cout << i << " : " << Ranks[i] << endl;
			}
#endif

		int_vec_heapsort(Ranks, nb_cosets);
		cout << "The sorted ranks of the code images are:" << endl; 
		for (i = 0; i < nb_cosets; i++) {
			cout << i << " : " << Ranks[i] << endl;
			}
		
		if (f_subspace_classify) {
			int N, d1;
			int *Codes_d;
			int nb_codes_d;
			int *Mindist;

			N = Gr->nCkq.as_int();
			Mindist = NEW_int(N);
			Codes_d = NEW_int(N);
			nb_codes_d = 0;
		

			cout << "N=" << N << endl;
			for (i = 0; i < N; i++) {
				Gr->unrank_int_here(G, i, 0 /*verbose_level*/);

				if (f_v3) {
					cout << "subspace " << 	i << " / " << N
							<< " is generated by:" << endl;
					int_matrix_print(G, code_sz - n, code_sz);
					}
				d1 = F->code_minimum_distance(code_sz,
						code_sz - n, G, 0 /* verbose_level */);
				if (f_v3) {
					cout << "subspace " << 	i << " / "
							<< N << " mindist = " << d1 << endl;
					}
				Mindist[i] = d1;
				if (d1 == d) {
					Codes_d[nb_codes_d++] = i;
					}
				}


			cout << "There are " << nb_codes_d
					<< " codes with minimum distance "
					<< d << ". They are:" << endl;
			for (i = 0; i < nb_codes_d; i++) {
				cout << i << " : " << Codes_d[i] << endl;
				}

			classify C;

			C.init(Mindist, N, FALSE, 0);
			cout << "Classification of subspaces by minimum distance: ";
			C.print_naked(TRUE);
			cout << endl;
			}


		FREE_int(G);
		}

	delete S;
	delete SG;
	delete A;
	delete F;
}

void make_generator_matrix(finite_field *F,
		int *G, int k, int n, int *code, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "make_generator_matrix" << endl;
		}
	for (i = 0; i < n; i++) {
		a = code[i];
		F->PG_element_unrank_modified(G + i, n, k, a);
		}
}

void projective_group(int n, int q, int f_semilinear, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	

	if (f_v) {
		cout << "projective_group n=" << n << " q=" << q
				<< " f_semilinear=" << f_semilinear << endl;
		}
	F = new finite_field;
	F->init(q, 0);
	A = new action;
	A->init_projective_group(n, F, 
		f_semilinear, TRUE /* f_basis */, verbose_level);
	A->print_base();
	A->group_order(Go1);
	cout << "Group of order " << Go1 << endl;
	
	
}

void grassmannian(int n, int k, finite_field *F, int verbose_level)
{
	Gr = new grassmann;

	Gr->init(n, k, F, 0 /* verbose_level */);
}

void read_group(const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	

	if (f_v) {
		cout << "read_group file = " << fname << endl;
		}

	SG = new strong_generators;
	cout << "reading generators from file " << fname << endl;

	//SG->init(A, 0);
	SG->read_file(A, fname, verbose_level);

	cout << "read generators from file" << endl;
	
	cout << "generators:" << endl;
	SG->print_generators();
	
	cout << "creating the group:" << endl;
	S = SG->create_sims(verbose_level);
	cout << "created the group" << endl;

	S->group_order(Go2);
	cout << "the group has order " << Go2 << endl;
	


}
