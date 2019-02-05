// create_element_of_order.C
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
const char *version = "create_element_of_order.C version 12/23/2009";

void create_element(int q, int k, int verbose_level);


int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i, q, k;
	
	cout << version << endl;
	for (i = 1; i < argc - 2; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}
	q = atoi(argv[argc - 2]);
	k = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;
	cout << "k=" << k << endl;
	
	
	if ((q * q - 1) % k) {
		cout << "k must divide q^2 - 1" << endl;
		exit(1);
		}
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	create_element(q, k, verbose_level);

	
	the_end_quietly(t0);
}

void create_element(int q, int k, int verbose_level)
{
	finite_field *F;
	action *A;
	action *A4;
	action *A_O4;
	longinteger_object Go;
	int ord;
	int *Elt7;
	int *ELT1;
	vector_ge *nice_gens;
	
	F = new finite_field;
	F->init(q, 0);	
	A = new action;
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */, TRUE /* f_basis */,
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);

	int f_semilinear = TRUE;
	int f_basis = FALSE;

	A_O4 = new action;
	A_O4->init_orthogonal_group(1 /*epsilon*/, 4/*n*/, F, 
		TRUE /* f_on_points */,
		FALSE /* f_on_lines */,
		FALSE /* f_on_points_and_lines */,
		f_semilinear, f_basis, verbose_level);
	

	A4 = new action;
	A4->init_projective_group(4 /* n */, F, 
		FALSE /* f_semilinear */,
		TRUE /* f_basis */,
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);

	A->print_base();
	A->group_order(Go);
	//int go;
	//go = Go.as_int();
	
	Elt7 = new int[A->elt_size_in_int];

	ELT1 = new int[A4->elt_size_in_int];

	int dataD[4];
	int mtxD[16];


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
	cout << "D has projective order " << ord << endl;

	int h;
	h = (q * q - 1) / k;
	cout << "taking to the power " << h << endl;
	
	A->element_power_int_in_place(Elt7, h, 0);
	ord = A->element_order(Elt7);
	A->element_print_quick(Elt7, cout);
	cout << "D has projective order " << ord << endl;
	if (ord != k) {
		cout << "does not match " << k
			<<  " but this may be alright" << endl;
		//exit(1);
		}

	// do not switch a and d:
	dataD[0] = Elt7[0];
	dataD[1] = Elt7[1];
	dataD[2] = Elt7[2];
	dataD[3] = Elt7[3];
	
	cout << "dataD:" << endl;
	int_vec_print(cout, dataD, 4);
	cout << endl;

	int f_switch = FALSE;

	// remember that O4_isomorphism_2to4 switches a and d
	O4_isomorphism_2to4(F, dataD, dataD, f_switch, mtxD);

	cout << "mtxD:" << endl;
	print_integer_matrix_width(cout, mtxD,
			4, 4, 4, F->log10_of_q);

	A4->make_element(ELT1, mtxD, verbose_level);
	cout << "diagonally embedded:" << endl;
	A4->element_print_quick(ELT1, cout);
	ord = A4->element_order(ELT1);
	cout << "B has projective order " << ord << endl;


	if (ord != k) {
		cout << "does not match " << k
			<< " but this may be alright" << endl;
		//exit(1);
		}


}

