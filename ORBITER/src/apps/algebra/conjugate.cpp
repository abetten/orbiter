// conjugate.C
// 
// Anton Betten
// 12/23/2009
//
//
// conjugate in PGL(2,q)
//
//

#include "orbiter.h"

#include <fstream>

using namespace orbiter;

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
	int from_elt[4];
	int to_elt[4];
	
	cout << version << endl;
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-from") == 0) {
			cout << "-from "<< endl;
			for (j = 0; j < 4; j++) {
				from_elt[j] = atoi(argv[++i]);
				}
			int_vec_print(cout, from_elt, 16);
			cout << endl;
			}
		else if (strcmp(argv[i], "-to") == 0) {
			cout << "-to "<< endl;
			for (j = 0; j < 4; j++) {
				to_elt[j] = atoi(argv[++i]);
				}
			int_vec_print(cout, to_elt, 16);
			cout << endl;
			}
		}
	q = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;

	cout << "from_elt:" << endl;
	print_integer_matrix_width(cout, from_elt, 2, 2, 2, 3);
	cout << "to_elt:" << endl;
	print_integer_matrix_width(cout, to_elt, 2, 2, 2, 3);
	
	
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	conjugate(q, from_elt, to_elt, verbose_level);

	
	the_end_quietly(t0);
}

void conjugate(int q, int *from_elt, int *to_elt, int verbose_level)
{
	finite_field *F;
	action *A2;
	longinteger_object Go;
	int ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6; //, *Elt7;
		


	int f_semilinear = TRUE;
	int f_basis = TRUE;
	vector_ge *nice_gens;
	F = new finite_field;
	F->init(q, 0);
	A2 = new action;
	A2->init_projective_group(2/*n*/, F, 
		f_semilinear, f_basis,
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	


	A2->print_base();
	A2->group_order(Go);
	//int go;
	//go = Go.as_int();
	
	Elt1 = new int[A2->elt_size_in_int];
	Elt2 = new int[A2->elt_size_in_int];
	Elt3 = new int[A2->elt_size_in_int];
	Elt4 = new int[A2->elt_size_in_int];
	Elt5 = new int[A2->elt_size_in_int];
	Elt6 = new int[A2->elt_size_in_int];
	//Elt7 = new int[A2->elt_size_in_int];


	A2->make_element(Elt1, from_elt, verbose_level);
	A2->make_element(Elt2, to_elt, verbose_level);

	cout << "from elt:" << endl;
	A2->element_print_quick(Elt1, cout);
	ord = A2->element_order(Elt1);
	cout << "element has order " << ord << endl;

	cout << "to elt:" << endl;
	A2->element_print_quick(Elt2, cout);
	ord = A2->element_order(Elt1);
	cout << "element has order " << ord << endl;

	//A2->init_matrix_group_strong_generators_builtin_projective(verbose_level);

	sims *S;
	longinteger_object Go1;
	
	S = new sims;

	S->init(A2);
	S->init_generators(*A2->Strong_gens->gens, verbose_level);
	S->compute_base_orbits_known_length(A2->Strong_gens->tl, verbose_level);
	S->group_order(Go1);
	cout << "found  group of order " << Go1 << endl;
#if 0
	if (!A2->f_has_sims) {
		cout << "!A2->f_has_sims" << endl;
		exit(1);
		}

	sims *S = A2->Sims;
#endif

	longinteger_object rk_from, rk_to;
	int rk_from_int, rk_to_int;
	
	
	S->element_rank(rk_from, Elt1);
	rk_from_int = rk_from.as_int();
	cout << "from element has rank " << rk_from << " = "
			<< rk_from_int << endl;
	S->element_rank(rk_to, Elt2);
	rk_to_int = rk_to.as_int();
	cout << "to element has rank " << rk_to << " = "
			<< rk_to_int << endl;
	


	int f_ownership = FALSE;
	
	action *A_conj;

	A_conj = new action;
	A_conj->induced_action_by_conjugation(S, 
		S, f_ownership, FALSE /*f_basis*/, verbose_level);
	
	schreier Sch;
	vector_ge gens;
	int *tl = new int[A2->base_len];
	int j, rep1, rep2, coset1, coset2;
	int i;
	int a;

	cout << "extracting strong generators" << endl;
	S->extract_strong_generators_in_order(gens, tl, verbose_level);
	for (i = 0; i < gens.len; i++) {
		cout << "generator " << i << ":" << endl;
		A2->element_print_quick(gens.ith(i), cout);
		}


	Sch.init(A_conj);
	Sch.init_generators(gens);
	cout << "computing point orbits" << endl;
	Sch.compute_all_point_orbits(verbose_level);
	Sch.print_and_list_orbits(cout);
	for (i = 0; i < Sch.nb_orbits; i++) {

		unipoly charpoly;
		int ord;

		j = Sch.orbit[Sch.orbit_first[i]];
		cout << "orbit " << i << " representative is " << j << " : " << endl;
		S->element_unrank_int(j, Elt3);
		A2->element_print_quick(Elt3, cout);
		cout << endl;
		ord = A2->element_order(Elt3);
		cout << "of order " << ord << endl;
				
		charpoly.charpoly(q, 2, Elt3, verbose_level - 2);
		cout << "characteristic polynomial: "
				<< charpoly << endl << endl;
		}

	cout << "#####################" << endl;
	for (i = 0; i < Sch.nb_orbits; i++) {

		unipoly charpoly;
		int ord;

		j = Sch.orbit[Sch.orbit_first[i]];
		S->element_unrank_int(j, Elt3);
		ord = A2->element_order(Elt3);
		if (ord != 17)
			continue;
		
		A2->element_print_quick(Elt3, cout);
		cout << "orbit " << i << " representative is " << j << " : " << endl;
		cout << "of order " << ord << endl;
				
		charpoly.charpoly(q, 2, Elt3, verbose_level - 2);
		cout << "characteristic polynomial: " << charpoly << endl << endl;
		}


	
	rep1 = Sch.orbit_representative(rk_from_int);
	cout << "rep of from matrix=" << rep1 << endl;
	S->element_unrank_int(rep1, Elt3);
	A2->element_print_quick(Elt3, cout);
	coset1 = Sch.orbit_inv[rk_from_int];
	cout << "coset1=" << coset1 << endl;
	Sch.coset_rep(coset1);
	cout << "coset representative:" << endl;
	A2->element_move(Sch.cosetrep, Elt3, FALSE);
	A2->element_print_quick(Elt3, cout);
	a = A_conj->element_image_of(rep1, Elt3, FALSE);
	cout << "image of " << rep1 << " under coset rep is " << a << endl;

	rep2 = Sch.orbit_representative(rk_to_int);
	cout << "rep of to matrix=" << rep2 << endl;
	S->element_unrank_int(rep2, Elt4);
	A2->element_print_quick(Elt4, cout);
	coset2 = Sch.orbit_inv[rk_to_int];
	cout << "coset2=" << coset2 << endl;
	Sch.coset_rep(coset2);
	cout << "coset representative:" << endl;
	A2->element_move(Sch.cosetrep, Elt4, FALSE);
	A2->element_print_quick(Elt4, cout);
	a = A_conj->element_image_of(rep2, Elt4, FALSE);
	cout << "image of " << rep2 << " under coset rep is " << a << endl;

	A2->element_invert(Elt3, Elt5, FALSE);
	a = A_conj->element_image_of(rk_from_int, Elt5, FALSE);
	cout << "image of " << rk_from_int << " under Elt5 is " << a << endl;

	A2->element_mult(Elt5, Elt4, Elt6, FALSE);
	cout << "conjugating element:" << endl;
	A2->element_print_quick(Elt6, cout);

	a = A_conj->element_image_of(rk_from_int, Elt6, FALSE);
	cout << "image of " << rk_from_int << " is " << a << endl;
	if (a != rk_to_int) {
		cout << "rk_to_int != a" << endl;
		exit(1);
		}



}

