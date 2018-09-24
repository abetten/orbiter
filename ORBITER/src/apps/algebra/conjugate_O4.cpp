// conjugate.C
// 
// Anton Betten
// 12/23/2009
//
//
// 
//
//

#include "orbiter.h"

#include <fstream>

// global data:

int t0; // the system time when the program started
const char *version = "conjugate.C version 12/23/2009";

void conjugate(int q, int *from_elt, int *to_elt, int verbose_level);


int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i, j, q;
	int from_elt[16];
	int to_elt[16];
	
	cout << version << endl;
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-from") == 0) {
			cout << "-from "<< endl;
			for (j = 0; j < 16; j++) {
				from_elt[j] = atoi(argv[++i]);
				}
			int_vec_print(cout, from_elt, 16);
			cout << endl;
			}
		else if (strcmp(argv[i], "-to") == 0) {
			cout << "-to "<< endl;
			for (j = 0; j < 16; j++) {
				to_elt[j] = atoi(argv[++i]);
				}
			int_vec_print(cout, to_elt, 16);
			cout << endl;
			}
		}
	q = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;

	cout << "from_elt:" << endl;
	print_integer_matrix_width(cout, from_elt, 4, 4, 4, 3);
	cout << "to_elt:" << endl;
	print_integer_matrix_width(cout, to_elt, 4, 4, 4, 3);
	
	
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	conjugate(q, from_elt, to_elt, verbose_level);

	
	the_end_quietly(t0);
}

void conjugate(int q, int *from_elt, int *to_elt, int verbose_level)
{
	finite_field *F;
	action *A4;
	action *A_O4;
	longinteger_object Go;
	int go, ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
		


	int f_semilinear = TRUE;
	int f_basis = TRUE;

	F = new finite_field;
	F->init(q, 0);
	A_O4 = new action;
	A_O4->init_orthogonal_group(1 /*epsilon*/, 4/*n*/, F, 
		TRUE /* f_on_points */, FALSE /* f_on_lines */, FALSE /* f_on_points_and_lines */, 
		f_semilinear, f_basis, verbose_level);

	
	A4 = A_O4->subaction;


	A_O4->print_base();
	A_O4->group_order(Go);
	go = Go.as_int();
	
	Elt1 = new int[A4->elt_size_in_int];
	Elt2 = new int[A4->elt_size_in_int];
	Elt3 = new int[A4->elt_size_in_int];
	Elt4 = new int[A4->elt_size_in_int];
	Elt5 = new int[A4->elt_size_in_int];
	Elt6 = new int[A4->elt_size_in_int];
	Elt7 = new int[A4->elt_size_in_int];


	A4->make_element(Elt1, from_elt, verbose_level);
	A4->make_element(Elt2, to_elt, verbose_level);

	cout << "from elt:" << endl;
	A4->element_print_quick(Elt1, cout);
	ord = A4->element_order(Elt1);
	cout << "element has order " << ord << endl;

	cout << "to elt:" << endl;
	A4->element_print_quick(Elt2, cout);
	ord = A4->element_order(Elt1);
	cout << "element has order " << ord << endl;


	if (!A4->f_has_sims) {
		cout << "!A4->f_has_sims" << endl;
		exit(1);
		}

	sims *S = A4->Sims;


	longinteger_object rk_from, rk_to;
	
	S->element_rank(rk_from, Elt1);
	cout << "from element has rank " << rk_from << endl;
	S->element_rank(rk_to, Elt2);
	cout << "to element has rank " << rk_to << endl;
	


	int f_ownership = FALSE;
	
	action *A_conj;

	A_conj = new action;
	A_conj->induced_action_by_conjugation(S, 
		S, f_ownership, FALSE /*f_basis*/, verbose_level);
	
	schreier Sch;
	vector_ge gens;
	int *tl = new int[A4->base_len];
	//int j, rep;
	int i;

	cout << "extracting strong generators" << endl;
	S->extract_strong_generators_in_order(gens, tl, verbose_level);
	for (i = 0; i < gens.len; i++) {
		cout << "generator " << i << ":" << endl;
		A4->element_print_quick(gens.ith(i), cout);
		}

#if 0
	Sch.init(A_conj);
	Sch.init_generators(gens);
	cout << "computing point orbits" << endl;
#endif


#if 0
	Sch.compute_all_point_orbits(verbose_level);
	Sch.print_and_list_orbits(cout);
	for (i = 0; i < Sch.nb_orbits; i++) {
		j = Sch.orbit[Sch.orbit_first[i]];
		cout << "orbit " << i << " representative is " << j << " : ";
		int w[5];
		PG_element_unrank_modified(Fq, w, 1, 5, j);
		int_vec_print(cout, w, 5);
		cout << endl;
		}
	rep = Sch.orbit_representative(rk);
	cout << "rep=" << rep << endl;
	coset = Sch.orbit_inv[rk];
	cout << "coset=" << coset << endl;
	Sch.coset_rep(coset);
	cout << "coset representative:" << endl;
	A0->element_move(Sch.cosetrep, Elt1, FALSE);
	A0->element_print_quick(Elt1, cout);
	int a, b;
	a = A0->element_image_of(rep, Elt1, FALSE);
	b = A0->element_image_of(rk, Elt1, FALSE);
	cout << "image of " << rep << " is " << a << endl;
	cout << "image of " << rk << " is " << b << endl;

#endif

}

