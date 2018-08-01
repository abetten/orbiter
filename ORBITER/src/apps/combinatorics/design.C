// design.C
//
// Anton Betten
// March 15, 2013
//

#include "orbiter.h"


INT t0;

void design_from_PG(INT PG_n, INT PG_q, INT verbose_level);
void design_from_PG_with_field(INT PG_n, finite_field *F, INT verbose_level);
void do_it(const BYTE *fname, INT verbose_level);
void design_properties(incidence_structure *Incidence, INT verbose_level);
INT number_of_blocks_through_set_of_points(incidence_structure *Incidence, INT *set_of_points, INT set_size);


int main(int argc, const char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_file = FALSE;
	const BYTE *fname;
	INT f_PG = FALSE;
	INT PG_n = 0;
	INT PG_q = 0;
	
	t0 = os_ticks();

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-PG") == 0) {
			f_PG = TRUE;
			PG_n = atoi(argv[++i]);
			PG_q = atoi(argv[++i]);
			cout << "-PG " << PG_n << " " << PG_q << endl;
			}
		}
	


	if (!f_file && !f_PG) {
		cout << "please specify either -file <fname> or -PG <n> <q>" << endl;
		exit(1);
		}

	if (f_file) {
		do_it(fname, verbose_level);
		}
	else if (f_PG) {
		design_from_PG(PG_n, PG_q, verbose_level);
		}	
}

void design_from_PG(INT PG_n, INT PG_q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	
	if (f_v) {
		cout << "design_from_PG" << endl;
		}
	F = new finite_field;
	F->init(PG_q, verbose_level);

	design_from_PG_with_field(PG_n, F, verbose_level);
	
	delete F;
}

void design_from_PG_with_field(INT PG_n, finite_field *F, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	projective_space *P;
	
	if (f_v) {
		cout << "design_from_PG_with_field" << endl;
		}


	P = new projective_space;

	P->init(PG_n, F, 
		TRUE /* f_init_incidence_structure */, 
		FALSE /* verbose_level */);
	
	incidence_structure *Incidence;
	partitionstack *Stack;

	P->make_incidence_structure_and_partition(Incidence, Stack, verbose_level);



	design_properties(Incidence, verbose_level);

	
	delete Stack;
	delete Incidence;
	
	if (f_v) {
		cout << "design_from_PG_with_field done" << endl;
		}
}

void do_it(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	INT *M;
	INT *Inc;
	INT i, j, h, m, n, k;

	if (f_v) {
		cout << "Reading matrix from file " << fname << endl;
		}
	INT_matrix_read_csv(fname, M, m, k, verbose_level - 1);

	cout << "Read matrix of size " << m << " x " << k << endl;
	INT_matrix_print(M, m, k);
	n = 0;
	for (i = 0; i < m * k; i++) {
		n = MAXIMUM(M[i], n);
		}
	cout << "largest entry is " << n << endl;
	n++;
	Inc = NEW_INT(n * m);
	for (i = 0; i < n * m; i++) {
		Inc[i] = 0;
		}
	for (j = 0; j < m; j++) {
		for (h = 0; h < k; h++) {
			i = M[j * k + h];
			Inc[i * m + j] = 1;
			}
		}
	cout << "Incidence matrix computed" << endl;


	incidence_structure *Incidence;
	INT set_size = n;
	INT nb_blocks = m;



	Incidence = new incidence_structure;
	Incidence->init_by_matrix(set_size, nb_blocks, Inc, 0 /* verbose_level */);


	design_properties(Incidence, verbose_level);


	FREE_INT(Inc);
	delete Incidence;
}




void design_properties(incidence_structure *Incidence, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	partitionstack *Stack;

	INT set_size = Incidence->nb_rows;
	INT nb_blocks = Incidence->nb_cols;

		
	Stack = new partitionstack;
	Stack->allocate(set_size + nb_blocks, 0 /* verbose_level */);
	Stack->subset_continguous(set_size, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();


	Incidence->refine_row_partition_safe(*Stack, 0/*verbose_level - 3*/);
	if (f_v) {
		cout << "Row-scheme:" << endl;
		Incidence->get_and_print_row_tactical_decomposition_scheme_tex(
			cout, FALSE /* f_enter_math */, TRUE /* f_print_subscripts */, *Stack);
		}

	INT *set;
	INT Nt, a, t, h;
	
	set = NEW_INT(set_size);
	for (t = 2; t <= 4; t++) {
		INT *Freq;
		
		cout << "Checking strength t=" << t << endl;
		Nt = INT_n_choose_k(set_size, t);
		cout << "There are " << Nt << " " << t << "-subsets of an " << set_size << "-set" << endl;

		Freq = NEW_INT(Nt);
		for (h = 0; h < Nt; h++) {
			if ((h % 100000) == 0) {
				cout << "checking subset " << h << " / " << Nt << endl;
				}
			unrank_k_subset(h, set, set_size, t);
			a = number_of_blocks_through_set_of_points(Incidence, set, t);
			//a = count(Inc, set_size, nb_blocks, set, t);
			Freq[h] = a;
			}
		classify C;

		C.init(Freq, Nt, FALSE, 0);
		cout << "Frequencies of t-subsets: ";
		C.print(TRUE /* f_backwards */);
		}

	FREE_INT(set);
	delete Stack;
}

INT number_of_blocks_through_set_of_points(incidence_structure *Incidence, INT *set_of_points, INT set_size)
{
	INT i, j;
	INT nb, h;
	INT m;

	m = Incidence->nb_rows;
	
	nb = 0;
	for (j = 0; j < m; j++) {
		for (h = 0; h < set_size; h++) {
			i = set_of_points[h];
			if (Incidence->get_ij(i, j) == 0 /*Inc[i * m + j] == 0 */) {
				break;
				}
			}
		if (h == set_size) {
			nb++;
			}
		}
	return nb;
}


