// find_element.C
// 
// Anton Betten
// 12/25/2009
//
//
// 
//
//

#include "orbiter.h"

#include <fstream>

// global data:

int t0; // the system time when the program started
const char *version = "find_element.C version 12/25/2009";

void find_element(int q, int *mtx, int verbose_level);


int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i, q, mtx[4];
	
	cout << version << endl;
	for (i = 1; i < argc - 5; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	q = atoi(argv[argc - 5]);
	mtx[0] = atoi(argv[argc - 4]);
	mtx[1] = atoi(argv[argc - 3]);
	mtx[2] = atoi(argv[argc - 2]);
	mtx[3] = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;
	
	
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	find_element(q, mtx, verbose_level);

	
	the_end_quietly(t0);
}

void find_element(int q, int *mtx, int verbose_level)
{
	finite_field *F;
	action *A;
	longinteger_object Go;
	int ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;

	int dataE[4];

	cout << "find_element searching for matrix:" << endl;
	print_integer_matrix_width(cout, mtx, 2, 2, 2, 3);

	F = new finite_field;
	F->init(67, 0);
	A = new action;
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */,
		TRUE /* f_basis */,
		verbose_level);



	A->print_base();
	A->group_order(Go);
	//int go;
	//go = Go.as_int();
	
	Elt1 = new int[A->elt_size_in_int];
	Elt2 = new int[A->elt_size_in_int];
	Elt3 = new int[A->elt_size_in_int];
	Elt4 = new int[A->elt_size_in_int];
	Elt5 = new int[A->elt_size_in_int];
	Elt6 = new int[A->elt_size_in_int];
	Elt7 = new int[A->elt_size_in_int];

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

	A->make_element(Elt6, dataD, FALSE);
	ord = A->element_order(Elt6);
	A->element_print_quick(Elt6, cout);
	cout << "D has projective order " << ord << endl;

	int k;
	int p0, p1, q0, q1;

	unipoly charpoly0;
	charpoly0.charpoly(67, 2, mtx, 0);
	p0 = charpoly0.s_ii(0);
	p1 = charpoly0.s_ii(1);
	cout << "charpoly " << charpoly0 << endl;
	
	for (k = 0; k < q * q - 1; k++) {

		//cout << "D1 = D^" << k << endl;
	
		A->make_element(Elt6, dataD, FALSE);
		A->element_power_int_in_place(Elt6, k, 0);
		//ord = A->element_order(Elt6);
		//A->element_print_quick(Elt6, cout);
		//cout << "D1 has projective order " << ord << endl;

		dataE[0] = Elt6[0];
		dataE[1] = Elt6[1];
		dataE[2] = Elt6[2];
		dataE[3] = Elt6[3];

		if (Elt6[0] == mtx[0] && 
			Elt6[1] == mtx[1] && 
			Elt6[2] == mtx[2] && 
			Elt6[3] == mtx[3]) {
			cout << "D^" << k << "=" << endl;
			A->element_print_quick(Elt6, cout);
			}
		unipoly charpoly1;

		charpoly1.charpoly(67, 2, dataE, 0);
		q0 = charpoly1.s_ii(0);
		q1 = charpoly1.s_ii(1);
		if (q0 == p0 && q1 == p1) {
			cout << "charpoly of D^" << k << "=" << endl;
			A->element_print_quick(Elt6, cout);
			cout << "matches: " << charpoly1 << endl;
			ord = A->element_order(Elt6);
			cout << "D1 has projective order " << ord << endl;
			}
		}

}

