// test_borel.C
// 
// Anton Betten
// Apr 22, 2016
//
//
// 
//
//

#include "orbiter.h"

using namespace orbiter;

// global data:

int t0; // the system time when the program started

void test_borel(int n, int q, int verbose_level);
void top(ostream &fp);
void bottom(ostream &fp);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q = 0;
	
 	t0 = os_ticks();
	
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
		}
	if (!f_n) {
		cout << "please use option -n <n>" << endl;
		exit(1);
	}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
	}
	test_borel(n, q, verbose_level);
}


void test_borel(int n, int q, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	finite_field *F;
	action *A;
	action *AP;
	sims *S;
	sims *SP;
	int *Elt1;
	int *Perm1;
	int *M;
	int *M1;
	int *M2;
	int *B1;
	int *B2;
	int *pivots;
	longinteger_object Go, goP;
	int goi, goPi, rk, rk1, rkP;
	
	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	A = NEW_OBJECT(action);

	cout << "before create_linear_group" << endl;
	create_linear_group(S, A, 
		F, n, 
		FALSE /* f_projective */,
		TRUE /* f_general */,
		FALSE /* f_affine */,
		FALSE /* f_semilinear */,
		TRUE /* f_special */,
		verbose_level);
	cout << "after create_linear_group" << endl;

	AP = NEW_OBJECT(action);
	AP->init_symmetric_group(n, 0 /* verbose_level */);
	AP->group_order(goP);
	cout << "created symmetric group Sym_" << n
			<< " of order " << goP << endl;
	goPi = goP.as_int();
	SP = AP->Sims;
	
	Elt1 = NEW_int(A->elt_size_in_int);
	Perm1 = NEW_int(AP->elt_size_in_int);
	M = NEW_int(n * n);
	M1 = NEW_int(n * n);
	M2 = NEW_int(n * n);
	B1 = NEW_int(n * n);
	B2 = NEW_int(n * n);
	pivots = NEW_int(n);
	
	strong_generators *SG;

	SG = NEW_OBJECT(strong_generators);

	SG->init_from_sims(S, 0 /* verbose_level */);

	A->print_base();
	S->group_order(Go);
	goi = Go.as_int();
	cout << "Group of order " << Go << endl;


	cout << "generators for the group are:" << endl;
	SG->print_generators();

	int *count;
	
	count = NEW_int(goPi);
	int_vec_zero(count, goPi);
	

	char fname[1000];

	sprintf(fname, "borel_SL_%d_%d.tex", n, q);
	{
	ofstream fp(fname);

	latex_head_easy(fp);

	//fp << "\\setlength{\\extrarowheight}{20pt}" << endl;

	top(fp);
	for (rk = 0; rk < goi; rk++) {

		if (rk && (rk % 15) == 0) {
			bottom(fp);
			top(fp);
			}
		S->element_unrank_int(rk, Elt1);
		cout << "element " << rk << " / " << goi << " is:" << endl;

		fp << rk << " & " << endl;
		fp << "\\left[" << endl;
		int_matrix_print_tex(fp, Elt1, n, n);
		fp << "\\right]" << endl;
		fp << " & " << endl;

		cout << "Elt1:" << endl;
		int_matrix_print(Elt1, n, n);
		cout << "using element_print_quick:" << endl;
		A->element_print_quick(Elt1, cout);

		int_vec_copy(Elt1, M, n * n);
		cout << "M:" << endl;
		int_matrix_print(M, n, n);
		F->identity_matrix(B1, n);
		F->identity_matrix(B2, n);

		F->Borel_decomposition(n, M, B1, B2, pivots, verbose_level);

		cout << "output:" << endl;
		int_matrix_print(M, n, n);
		cout << "pivots:" << endl;
		int_vec_print(cout, pivots, n);
		cout << endl;

		AP->make_element(Perm1, pivots, verbose_level);
		cout << "created permutations:" ;
		AP->element_print(Perm1, cout);
		cout << endl;

		rkP = SP->element_rank_int(Perm1);
		cout << "permutation group rank = " << rkP << endl;


		fp << "\\left[" << endl;
		int_matrix_print_tex(fp, B1, n, n);
		fp << "\\right]" << endl;
		fp << " & " << endl;
		fp << "\\left[" << endl;
		int_matrix_print_tex(fp, M, n, n);
		fp << "\\right]" << endl;
		fp << " & " << endl;
		fp << "\\left[" << endl;
		int_matrix_print_tex(fp, B2, n, n);
		fp << "\\right]" << endl;
		fp << " & \\ " << endl;
		AP->element_print_as_permutation_with_offset(Perm1, fp, 
			1 /* offset */,
			TRUE /* f_do_it_anyway_even_for_big_degree */,
			FALSE /* f_print_cycles_of_length_one */,
			0 /* verbose_level */);
		fp << " \\\\" << endl;
		fp << "\\hline" << endl;
		
		F->mult_matrix_matrix(B1, M, M1, n, n, n,
				0 /* verbose_level */);
		F->mult_matrix_matrix(M1, B2, M2, n, n, n,
				0 /* verbose_level */);
		cout << "N:" << endl;
		int_matrix_print(M, n, n);
		cout << "B1:" << endl;
		int_matrix_print(B1, n, n);
		cout << "B2:" << endl;
		int_matrix_print(B2, n, n);
		cout << "B1*N*B2:" << endl;
		int_matrix_print(M2, n, n);
		A->make_element(Elt1, M2, verbose_level);
		rk1 = S->element_rank_int(Elt1);
		if (rk1 != rk) {
			cout << "rk1 != rk" << endl;
			exit(1);
			}
		
		count[rkP]++;

		}
	bottom(fp);
	latex_foot(fp);
	}
	
	int i;
	
	cout << "i : permutation : count" << endl;
	for (i = 0; i < goPi; i++) {
		SP->element_unrank_int(i, Perm1);
		cout << i << " : ";
		AP->element_print_quick(Perm1, cout);
		cout << " : ";
		cout << count[i] << endl;
		}

	FREE_int(count);
	FREE_int(Elt1);
	FREE_int(Perm1);
	FREE_int(M);
	FREE_int(M1);
	FREE_int(M2);
	FREE_int(B1);
	FREE_int(B2);
	FREE_int(pivots);
	FREE_OBJECT(SG);
}

void top(ostream &fp)
{
	fp << "$$" << endl;
	fp << "\\begin{array}{|c|c|c|c|c|c|}" << endl;
	fp << "\\hline" << endl;
	fp << "\\mbox{rk} & A & B_1 & N & B_2 & \\mbox{perm} \\\\" << endl;
	fp << "\\hline" << endl;
	fp << "\\hline" << endl;
}

void bottom(ostream &fp)
{
	fp << "\\end{array}" << endl;
	fp << "$$" << endl;
}


