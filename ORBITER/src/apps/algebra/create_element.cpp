// create_element.C
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
const char *version = "create_element.C version 12/25/2009";

void create_element(int q, int k1, int k2, int verbose_level);
void create_element_O4_isomorphism(int q,
		int f_switch, int *data8, int verbose_level);


int main(int argc, char **argv)
{
	t0 = os_ticks();
	discreta_init();
	int verbose_level = 0;
	int i, j, q;
	int f_power = FALSE;
	int k1, k2;
	int f_O4 = FALSE;
	int f_switch = FALSE;
	int data8[8];
	
	cout << version << endl;
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-power") == 0) {
			f_power = TRUE;
			k1 = atoi(argv[++i]);
			k2 = atoi(argv[++i]);
			cout << "-power " << k1 << " " << k2 << endl;
			}
		else if (strcmp(argv[i], "-O4") == 0) {
			f_O4 = TRUE;
			f_switch = atoi(argv[++i]);
			for (j = 0; j < 8; j++) {
				data8[j] = atoi(argv[++i]);
				}
			cout << "-O4 " << f_switch << " ";
			int_vec_print(cout, data8, 8);
			cout << endl;
			}
		}
	q = atoi(argv[argc - 1]);
	
	cout << "q=" << q << endl;
	
	
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	if (f_power) {
		create_element(q, k1, k2, verbose_level);
		}
	if (f_O4) {
		create_element_O4_isomorphism(q,
				f_switch, data8, verbose_level);
		}
	
	the_end_quietly(t0);
}

void create_element(int q, int k1, int k2, int verbose_level)
{
	finite_field *F;
	action *A;
	action *A4;
	action *A_O4;
	longinteger_object Go;
	int go, ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
	int *ELT1, *ELT2;
	int *Elt_At, *Elt_As, *Elt_Bt, *Elt_Bs, *ELT_A, *ELT_B;
	
	F = new finite_field;
	F->init(q, 0);	
	A = new action;
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */,
		TRUE /* f_basis */,
		verbose_level);


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
		verbose_level);

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

	int dataD[4];
	int dataD1[4];
	int dataD2[4];
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

	A->make_element(Elt6, dataD, FALSE);
	ord = A->element_order(Elt6);
	A->element_print_quick(Elt6, cout);
	cout << "D has projective order " << ord << endl;

	cout << "D1 = D^" << k1 << endl;
	
	A->make_element(Elt6, dataD, FALSE);
	A->element_power_int_in_place(Elt6, k1, 0);
	ord = A->element_order(Elt6);
	A->element_print_quick(Elt6, cout);
	cout << "D1 has projective order " << ord << endl;

	cout << "D2 = D^" << k2 << endl;
	
	A->make_element(Elt7, dataD, FALSE);
	A->element_power_int_in_place(Elt7, k2, 0);
	ord = A->element_order(Elt7);
	A->element_print_quick(Elt7, cout);
	cout << "D2 has projective order " << ord << endl;

	// do not switch a and d:
	dataD1[0] = Elt6[0];
	dataD1[1] = Elt6[1];
	dataD1[2] = Elt6[2];
	dataD1[3] = Elt6[3];
	dataD2[0] = Elt7[0];
	dataD2[1] = Elt7[1];
	dataD2[2] = Elt7[2];
	dataD2[3] = Elt7[3];
	
	cout << "dataD1:" << endl;
	int_vec_print(cout, dataD1, 4);
	cout << endl;
	cout << "dataD2:" << endl;
	int_vec_print(cout, dataD2, 4);
	cout << endl;

	int f_switch = FALSE;

	// remember that O4_isomorphism_2to4 switches a and d
	O4_isomorphism_2to4(F, dataD1, dataD2, f_switch, mtxD);

	cout << "mtxD:" << endl;
	print_integer_matrix_width(cout, mtxD, 4, 4, 4, F->log10_of_q);

	A4->make_element(ELT1, mtxD, verbose_level);
	cout << "diagonally embedded:" << endl;
	A4->element_print_quick(ELT1, cout);
	ord = A4->element_order(ELT1);
	cout << "B has projective order " << ord << endl;


	fine_tune(F, mtxD, verbose_level);

}


void create_element_O4_isomorphism(int q,
		int f_switch, int *data8, int verbose_level)
{
	finite_field *F;
	action *A;
	action *A4;
	action *A_O4;
	longinteger_object Go;
	int go, ord;
	int *Elt1, *Elt2, *Elt3, *Elt4, *Elt5, *Elt6, *Elt7;
	int *ELT1, *ELT2;
	int *Elt_At, *Elt_As, *Elt_Bt, *Elt_Bs, *ELT_A, *ELT_B;
	//int i;
	
	F = new finite_field;
	F->init(q, 0);
	A = new action;
	A->init_projective_group(2 /* n */, F, 
		FALSE /* f_semilinear */,
		TRUE /* f_basis */,
		verbose_level);


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
		verbose_level);

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

	int mtxD[16];




	// remember that O4_isomorphism_2to4 switches a and d
	O4_isomorphism_2to4(F, data8, data8 + 4, f_switch, mtxD);

	cout << "mtxD:" << endl;
	print_integer_matrix_width(cout, mtxD, 4, 4, 4, F->log10_of_q);

	A4->make_element(ELT1, mtxD, verbose_level);
	cout << "group element D:" << endl;
	A4->element_print_quick(ELT1, cout);
	ord = A4->element_order(ELT1);
	cout << "has projective order " << ord << endl;

	fine_tune(F, mtxD, verbose_level);

	
}




