// interface_perm_group.C
//
// Anton Betten
//
// started:  November 13, 2007
// last change:  November 9, 2010
// moved here from interface.C:  January 30, 2014




#include "foundations/foundations.h"
#include "group_actions.h"

namespace orbiter {
namespace group_actions {

// #############################################################################
// interface functions: permutation group
// #############################################################################


int perm_group_element_image_of(action &A,
		int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;
	int b;
	
	if (f_v) {
		cout << "perm_group_element_image_of "
				"image of " << a;
		}
	if (G.f_product_action) {
		if (a < G.offset) {
			b = Elt[a];
			}
		else {
			int /*x,*/ y, xx, yy;
			
			a -= G.offset;
			//x = a / G.n;
			y = a % G.n;
			xx = Elt[a];
			yy = Elt[G.m + y] - G.m;
			b = xx * G.n + yy + G.offset;
			}
		}
	else {
		b = Elt[a];
		}
	if (f_v) {
		cout << " is " << b << endl;
		}
	return b;
}

void perm_group_element_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;
	
	if (f_v) {
		cout << "perm_group_element_one ";
		}
	G.one(Elt);
}

int perm_group_element_is_one(action &A, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;
	int f_is_one;
	
	if (f_v) {
		cout << "perm_group_element_is_one ";
		}
	f_is_one = G.is_one(Elt);
	if (f_v) {
		if (f_is_one)
			cout << " YES" << endl;
		else
			cout << " NO" << endl;
		}
	return f_is_one;
}

void perm_group_element_unpack(action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;
	
	if (f_v) {
		cout << "perm_group_element_unpack" << endl;
		}
	G.unpack(elt1, Elt1);
}

void perm_group_element_pack(action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;
	
	if (f_v) {
		cout << "perm_group_element_pack" << endl;
		}
	G.pack(Elt1, elt1);
}

void perm_group_element_retrieve(action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;
	uchar *p_elt;
	
	if (f_v) {
		cout << "perm_group_element_retrieve hdl = " << hdl << endl;
		}

#if 0
	if (FALSE /* A.f_group_order_is_small */) {
			//cout << "Eltrk1=" << G.Eltrk1 << endl;
			//cout << "Eltrk2=" << G.Eltrk2 << endl;
			//cout << "Eltrk3=" << G.Eltrk3 << endl;
		int i, j, ii, l, q, r;
		int a;
	
		a = hdl;
		for (ii = A.base_len - 1; ii >= 0; ii--) {
			l = A.transversal_length[ii];

			r = a % l;
			q = a / l;
			a = q;
		
			A.path[ii] = r;
			//cout << r << " ";
			}
		//cout << endl;
		A.element_one(G.Eltrk1, 0);
		for (i = 0; i < A.base_len; i++) {
			j = A.path[i];
		
		
			// pre multiply the coset representative:
			A.element_mult(A.transversal_reps[i] +
					j * A.elt_size_in_int, G.Eltrk1, G.Eltrk2, 0);
			A.element_move(G.Eltrk2, G.Eltrk1, 0);
			}
		A.element_move(G.Eltrk1, Elt, 0);
		
		}
	else {
#endif
		p_elt = G.Elts->s_i(hdl);
		G.unpack(p_elt, Elt);
		//}
	if (f_v) {
		G.print(Elt, cout);
		}
}

int perm_group_element_store(action &A, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;
	int hdl;
	
	if (f_v) {
		cout << "perm_group_element_store()" << endl;
		}
#if 0
	if (FALSE /* A.f_group_order_is_small */) {
		int i, j, bi, jj, l;
		int a;
	
		//cout << "perm_group_element_store" << endl;
		A.element_move(elt, G.Eltrk1, 0);
		a = 0;
		for (i = 0; i < A.base_len; i++) {
			bi = A.base[i];
			l = A.transversal_length[i];
			//cout << "i=" << i << " bi=" << bi
			//<< " l=" << l << " a=" << a << endl;
			
			if (i > 0) {
				a *= l;
				}
			
			jj = A.element_image_of(bi, G.Eltrk1, 0);
			j = A.orbit_inv[i][jj];
			//cout << "at level " << i << ", maps bi = "
			//<< bi << " to " << jj << " which is coset " << j << endl;
			if (j >= l) {
				cout << "perm_group_element_store() j >= l" << endl;
				exit(1);
				}
			a += j;
			
			//A.element_print(A.transversal_reps[i] +
			// j * A.elt_size_in_int, cout);
			//perm_print_list(cout, A.transversal_reps[i] +
			// j * A.elt_size_in_int, G.degree);
			
			G.invert(A.transversal_reps[i] +
					j * A.elt_size_in_int, G.Eltrk2);
			
			//cout << "after invert ";
			//perm_print_list(cout, G.Eltrk2, G.degree);
			//A.element_print(G.Eltrk2, cout);
			
			//cout << "Eltrk1=" << G.Eltrk1 << endl;
			//cout << "Eltrk2=" << G.Eltrk2 << endl;
			//cout << "Eltrk3=" << G.Eltrk3 << endl;
			A.element_mult(G.Eltrk1, G.Eltrk2, G.Eltrk3, 0);
			//cout << "after mult, stripped to ";
			//perm_print_list(cout, G.Eltrk3, G.degree);
			//A.element_print(G.Eltrk3, cout);
			
			
			A.element_move(G.Eltrk3, G.Eltrk1, 0);
			//cout << "stripped to ";
			//A.element_print(G.Eltrk1, cout);
			
			}
		//cout << endl;
		hdl = a;
		}
	else {
#endif
		G.pack(Elt, G.elt1);
		hdl = G.Elts->store(G.elt1);
		//}
	if (f_v) {
		cout << "hdl = " << hdl << endl;
		}
	return hdl;
}

void perm_group_element_mult(action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "perm_group_element_mult degree=" << G.degree << endl;
		cout << "A=" << endl;
		perm_print_list(cout, AA, G.degree);
		G.print(AA, cout);
		cout << "B=" << endl;
		perm_print_list(cout, BB, G.degree);
		G.print(BB, cout);
		}
	G.mult(AA, BB, AB);
	if (f_v) {
		cout << "degree=" << G.degree << endl;
		cout << "AB=" << endl;
		perm_print_list(cout, AB, G.degree);
		G.print(AB, cout);
		}
}

void perm_group_element_invert(action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "perm_group_element_invert" << endl;
		cout << "A=" << endl;
		G.print(AA, cout);
		}
	G.invert(AA, AAv);
	if (f_v) {
		cout << "Av=" << endl;
		G.print(AAv, cout);
		}
}

void perm_group_element_move(action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "perm_group_element_move" << endl;
		}
	G.copy(AA, BB);
}

void perm_group_element_dispose(action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	perm_group &G = *A.G.perm_grp;

	if (f_v) {
		cout << "perm_group_element_dispose hdl = " << hdl << endl;
		}
	if (FALSE /* A.f_group_order_is_small */) {
		// do nothing
		}
	else {
		G.Elts->dispose(hdl);
		}
}

void perm_group_element_print(action &A,
		void *elt, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	//cout << "perm_group_element_print" << endl;
	G.print(Elt, ost);
	//cout << "perm_group_element_print done" << endl;
	//G.print_with_action(&A, Elt, ost);
}

void perm_group_element_print_latex(action &A,
		void *elt, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	G.print(Elt, ost);
	//G.print_with_action(&A, Elt, ost);
}

void perm_group_element_print_verbose(action &A,
		void *elt, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	G.print(Elt, ost);
}

void perm_group_element_code_for_make_element(action &A,
		void *elt, int *data)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	G.code_for_make_element(Elt, data);
}

void perm_group_element_print_for_make_element(action &A,
		void *elt, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	G.print_for_make_element(Elt, ost);
}

void perm_group_element_print_for_make_element_no_commas(action &A,
		void *elt, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	int *Elt = (int *) elt;

	G.print_for_make_element_no_commas(Elt, ost);
}


void perm_group_print_point(action &A, int a, ostream &ost)
{
	perm_group &G = *A.G.perm_grp;
	
	if (G.f_product_action) {
		if (a < G.offset) {
			cout << "r_{" << a << "}";
			}
		else {
			int x, y;
			
			a -= G.offset;
			x = a / G.n;
			y = a % G.n;
			cout << "(" << x << "," << y << ")";
			}
		}
	else {
		ost << a;
		}
}

}}



