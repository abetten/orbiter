// move_point_orthogonal.C
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


using namespace orbiter;

// global data:

int t0; // the system time when the program started
const char *version = "move_point_orthogonal.C version 12/23/2009";


int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i, q;
	
	cout << version << endl;
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	q = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;
	
	
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);


	finite_field *F;
	action *A;
	//orthogonal *O;
	int f_semilinear = TRUE;
	int f_basis = TRUE;
	int *Elt1;


	F = NEW_OBJECT(finite_field);
	F->init(q, 0);		
	A = NEW_OBJECT(action);
	
	A->init_orthogonal_group(0 /*epsilon*/, 5/*n*/, F, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);
	
	//O = A->G.AO->O;
	
	Elt1 = NEW_int(A->elt_size_in_int);


	{
	action *A0;
	
	A0 = A->subaction;
	cout << "subaction " << A0->label << endl;
	int vec[5] = {0,1,1,0,0};
	finite_field Fq;
	int rk, coset;

	if (q == 67) {
#if 0
		vec[0] = 1;
		vec[1] = 35;
		vec[2] = 15;
		vec[3] = 38;
		vec[4] = 63;
#else
		vec[0] = 45;
		vec[1] = 50;
		vec[2] = 5;
		vec[3] = 60;
		vec[4] = 66;
#endif
		}
	if (q == 19) {
		vec[0] = 1;
		vec[1] = 13;
		vec[2] = 7;
		vec[3] = 0;
		vec[4] = 0;
		}
	Fq.init(q, verbose_level - 1);
	Fq.PG_element_rank_modified(vec, 1, 5, rk);
	cout << "vector ";
	int_vec_print(cout, vec, 5);
	cout << " has rank " << rk << endl;
	
	if (!A->f_has_sims) {
		cout << "!A->f_has_sims" << endl;
		exit(1);
		}

	sims *S = A->Sims;
	schreier Sch;
	vector_ge gens;
	int *tl = NEW_int(A->base_len);
	int j, rep;

	cout << "extracting strong generators" << endl;
	S->extract_strong_generators_in_order(gens, tl, verbose_level);	
	Sch.init(A0);
	Sch.init_generators(gens);
	cout << "computing point orbits" << endl;
	Sch.compute_all_point_orbits(verbose_level);
	Sch.print_and_list_orbits(cout);
	for (i = 0; i < Sch.nb_orbits; i++) {
		j = Sch.orbit[Sch.orbit_first[i]];
		cout << "orbit " << i << " representative is " << j << " : ";
		int w[5];
		Fq.PG_element_unrank_modified(w, 1, 5, j);
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
#if 0
	if (A0->base[0] != 0) {
		cout << "A0->base[0] != 0" << endl;
		exit(1);
		}
	coset = S->orbit_inv[0][rk];
	cout << "coset = " << coset << endl;
	cout << "S->orbit_len[0]=" << S->orbit_len[0] << endl;

	if (coset >= S->orbit_len[0]) {
		cout << "point " << rk << " is not contained in orbit 0" << endl;
		exit(1);
		}
	for (i = 0; i < A0->base_len; i++) {
		S->path[i] = 0;
		}
	S->path[0] = coset;
	cout << "path: ";
	int_vec_print(cout, S->path, A0->base_len);
	cout << endl;

	S->element_from_path(Elt1, verbose_level);
	cout << "coset representative:" << endl;
	A0->element_print_quick(Elt1, cout);
	
	if (map_and_test(A, BLT, data, size,
			Elt1, cnt, q, no, verbose_level)) {
		cout << "it worked!" << endl;
		exit(0);
		}
#endif

	}

	the_end_quietly(t0);
}

