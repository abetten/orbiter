// perm_group.C
//
// Anton Betten
//
// started: May 25, 2006




#include "foundations/foundations.h"
#include "group_actions.h"


namespace orbiter {

perm_group::perm_group()
{
	null();
}

perm_group::~perm_group()
{
	free();
}

void perm_group::null()
{
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	elt1 = NULL;
	elt2 = NULL;
	elt3 = NULL;
	Elts = NULL;
	Eltrk1 = NULL;
	Eltrk2 = NULL;
	Eltrk3 = NULL;
}

void perm_group::free()
{
	//cout << "perm_group::free" << endl;
	if (Elt1)
		FREE_int(Elt1);
	if (Elt2)
		FREE_int(Elt2);
	if (Elt3)
		FREE_int(Elt3);
	if (Elt4)
		FREE_int(Elt4);
	//cout << "perm_group::free before elt1" << endl;
	if (elt1)
		FREE_uchar(elt1);
	if (elt2)
		FREE_uchar(elt2);
	if (elt3)
		FREE_uchar(elt3);
	//cout << "perm_group::free before Elts" << endl;
	if (Elts) {
		FREE_OBJECT(Elts);
	}
	if (Eltrk1)
		FREE_int(Eltrk1);
	if (Eltrk2)
		FREE_int(Eltrk2);
	if (Eltrk3)
		FREE_int(Eltrk3);
	null();
	//cout << "perm_group::free finished" << endl;
}

void perm_group::allocate()
{
	Elt1 = NEW_int(elt_size_int);
	Elt2 = NEW_int(elt_size_int);
	Elt3 = NEW_int(elt_size_int);
	Elt4 = NEW_int(elt_size_int);
	elt1 = NEW_uchar(char_per_elt);
	elt2 = NEW_uchar(char_per_elt);
	elt3 = NEW_uchar(char_per_elt);
	Eltrk1 = NEW_int(elt_size_int);
	Eltrk2 = NEW_int(elt_size_int);
	Eltrk3 = NEW_int(elt_size_int);

	Elts = NEW_OBJECT(page_storage);
}

void perm_group::init_product_action(int m, int n,
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "perm_group::init_product_action() "
				"m=" << m << " n=" << n << endl;
		}
	f_product_action = TRUE;
	perm_group::m = m;
	perm_group::n = n;
	mn = m * n;
	offset = m + n;
	
	degree = m + n + m * n;
	elt_size_int = m + n;
	char_per_elt = elt_size_int;
	
	init_data(page_length_log, verbose_level);
}

	
void perm_group::init(int degree,
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "perm_group::init()" << endl;
		}
	perm_group::degree = degree;
	f_product_action = FALSE;
	
	elt_size_int = degree;
	char_per_elt = elt_size_int * sizeof(int);
	
	init_data(page_length_log, verbose_level);
}

void perm_group::init_data(int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int hdl;

	if (f_v) {
		cout << "perm_group::init_data" << endl;
		cout << "degree=" << degree << endl;
		cout << "elt_size_int=" << elt_size_int << endl;
		cout << "page_length_log=" << page_length_log << endl;
		//cout << "base_len=" << A.base_len << endl;
		}

	allocate();

	int *tmp1 = NEW_int(elt_size_int);
	int *tmp2 = NEW_int(elt_size_int);
	int *tmp3 = NEW_int(elt_size_int);
	

	
	if (f_vv) {
		cout << "perm_group::init_data "
				"calling Elts->init()" << endl;
		}
	Elts->init(char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(perm_group_elt_print, (void *) this);


	if (f_vv) {
		cout << "perm_group::init_data "
				"calling one()" << endl;
		}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "perm_group::init_data "
				"calling Elts->store()" << endl;
		}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
		}
	

	if (f_vv) {
		cout << "perm_group::init_data "
				"finished" << endl;
		}
	
	FREE_int(tmp1);
	FREE_int(tmp2);
	FREE_int(tmp3);
}

void perm_group::init_with_base(int degree, 
	int base_length, int *base, int page_length_log, 
	action &A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, hdl;
	
	perm_group::degree = degree;
	f_product_action = FALSE;
	
	A.base_len = base_length;
	A.degree = degree;
	elt_size_int = degree;
	char_per_elt = elt_size_int;
	

	allocate();

	int *tmp1 = NEW_int(elt_size_int);
	int *tmp2 = NEW_int(elt_size_int);
	int *tmp3 = NEW_int(elt_size_int);
	

	
	if (f_v) {
		cout << "perm_group::init" << endl;
		cout << "degree=" << A.degree << endl;
		cout << "base_len=" << A.base_len << endl;
		}
	if (f_vv) {
		cout << "perm_group::init "
				"calling Elts->init" << endl;
		}
	Elts->init(char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(
	//perm_group_elt_print, (void *) this);


	if (f_vv) {
		cout << "perm_group::init "
				"calling one()" << endl;
		}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "perm_group::init "
				"calling Elts->store" << endl;
		}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
		}
	
	if (f_vv) {
		cout << "perm_group::init "
				"initializing base, and transversal_length" << endl;
		}
	A.type_G = perm_group_t;
	A.G.perm_grp = this;
	
	A.allocate_base_data(A.base_len);

	// init base:
	for (i = 0; i < A.base_len; i++)
		A.base[i] = base[i];
	

	if (f_v) {
		cout << "base: ";
		print_set(cout, A.base_len, A.base);
		cout << endl;
		//cout << "transversal_length: ";
		//print_set(cout, A.base_len, A.transversal_length);
		//cout << endl;
		}

	A.init_function_pointers_permutation_group();
	
	A.elt_size_in_int = elt_size_int;
	A.coded_elt_size_in_char = char_per_elt;
	
	A.allocate_element_data();

	sprintf(A.group_prefix, "Sym%d", degree);

	if (f_vv) {
		cout << "perm_group::init finished" << endl;
		}
	
	FREE_int(tmp1);
	FREE_int(tmp2);
	FREE_int(tmp3);
}

void perm_group::transversal_rep(int i, int j,
		int *Elt, int verbose_level)
{
	int j1, j2;
	
	one(Elt);
	j1 = i;
	j2 = i + j;
	Elt[j1] = j2;
	Elt[j2] = j1;
}

void perm_group::one(int *Elt)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		Elt[i] = i;
		}
}

int perm_group::is_one(int *Elt)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		if (Elt[i] != i) {
			return FALSE;
			}
		}
	return TRUE;
}

void perm_group::mult(int *A, int *B, int *AB)
{
	//cout << "in perm_group::mult()" << endl;
	perm_mult(A, B, AB, degree);
	//cout << "in perm_group::mult()
	// finished with perm_mult" << endl;
}

void perm_group::copy(int *A, int *B)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		B[i] = A[i];
		}
}

void perm_group::invert(int *A, int *Ainv)
{
	perm_inverse(A, Ainv, degree);
}

void perm_group::unpack(uchar *elt, int *Elt)
{
	int i, j;
	
	for (i = 0; i < degree; i++) {
		uchar *p;

		p = (uchar *)(Elt + i);
		for (j = 0; j < (int) sizeof(int); j++) {
			*p++ = *elt++;
			}
		}
}

void perm_group::pack(int *Elt, uchar *elt)
{
	int i, j;
	
	for (i = 0; i < degree; i++) {
		uchar *p;

		p = (uchar *)(Elt + i);
		for (j = 0; j < (int) sizeof(int); j++) {
			*elt++ = *p++;
			}
		}
}

void perm_group::print(int *Elt, ostream &ost)
{
	//cout << "perm_group::print before perm_print" << endl;
	perm_print(ost, Elt, degree);
	//ost << endl;
	//cout << "perm_group::print done" << endl;
}

void perm_group::code_for_make_element(int *Elt, int *data)
{
	int_vec_copy(Elt, data, degree);
}

void perm_group::print_for_make_element(int *Elt, ostream &ost)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << ", ";
		}
}

void perm_group::print_for_make_element_no_commas(
		int *Elt, ostream &ost)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << " ";
		}
}

void perm_group::print_with_action(action *A, int *Elt, ostream &ost)
{
	//perm_print(ost, Elt, degree);
	//ost << endl;
	int i, bi, a;
	int x1, y1, x2, y2; // if in product action
	
	if (A->base_len < A->degree) {
		for (i = 0; i < A->base_len; i++) {
			bi = A->base[i];
			a = Elt[bi];
			if (f_product_action) {
				cout << "bi=" << bi << "a=" << a << endl;
				if (bi < m) {
					ost << "(x=" << bi << ") -> (x=" << a << ")" << endl;
					}
				else if (bi < m + n) {
					ost << "(y=" << bi - m
							<< ") -> (y=" << a - m << ")" << endl;
					}
				else {
					bi -= m + n;
					a -= m + n;
					x1 = bi / n;
					y1 = bi % n;
					x2 = a / n;
					y2 = a % n;
					ost << bi << "=(" << x1 << "," << y1 << ")" 
						<< " -> " 
						<< a << "=(" << x2 << "," << y2 << ")";
					}
				}
			else {
				ost << bi << " -> " << a;
				}
			if (i < A->base_len - 1)
				ost << ", ";
			}
		}
	//perm_print(ost, Elt, degree);
	ost << " : ";
	perm_print_offset(ost, Elt, degree, 0 /* offset */,
			FALSE /* f_cycle_length */, FALSE, 0,
			FALSE /* f_orbit_structure */);
	ost << " : ";
	perm_print_list_offset(ost, Elt, degree, 1);
	ost << endl;
}

void perm_group::make_element(int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	
	if (f_v) {
		cout << "perm_group::make_element" << endl;
		}
	if (f_vv) {
		cout << "data: ";
		int_vec_print(cout, data, elt_size_int);
		cout << endl;
		}
	for (i = 0; i < elt_size_int; i++) {
		a = data[i];
		Elt[i] = a;
		}
	if (f_v) {
		cout << "perm_group::make_element done" << endl;
		}
}

#if 0

//#############################################################################
// global functions:
//#############################################################################


void perm_group_find_strong_generators_at_level(
	int level, int degree,
	int given_base_length, int *given_base,
	int nb_gens, int *gens, 
	int &nb_generators_found, int *idx_generators_found)
{
	int i, j, bj, bj_image;
	
	nb_generators_found = 0;
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < level; j++) {
			bj = given_base[j];
			bj_image = gens[i * degree + bj];
			if (bj_image != bj)
				break;
			}
		if (j == level) {
			idx_generators_found[nb_generators_found++] = i;
			}
		}
}

void perm_group_generators_direct_product(
	int degree1, int degree2, int &degree3,
	int nb_gens1, int nb_gens2, int &nb_gens3, 
	int *gens1, int *gens2, int *&gens3, 
	int base_len1, int base_len2, int &base_len3, 
	int *base1, int *base2, int *&base3)
{
	int u, i, j, ii, jj, k, offset;
	
	offset = degree1 + degree2;
	degree3 = offset + degree1 * degree2;
	nb_gens3 = nb_gens1 + nb_gens2;
	base_len3 = base_len1 + base_len2;
	gens3 = NEW_int(nb_gens3 * degree3);
	base3 = NEW_int(base_len3);
	for (u = 0; u < base_len1; u++) {
		base3[u] = base1[u];
		}
	for (u = 0; u < base_len2; u++) {
		base3[base_len1 + u] = degree1 + base2[u];
		}
	k = 0;
	for (u = 0; u < nb_gens1; u++, k++) {
		for (i = 0; i < degree1; i++) {
			ii = gens1[u * degree1 + i];
			gens3[k * degree3 + i] = ii;
			for (j = 0; j < degree2; j++) {
				gens3[k * degree3 + offset + i * degree2 + j] = 
					offset + ii * degree2 + j;
				}
			}
		for (j = 0; j < degree2; j++) {
			gens3[k * degree3 + degree1 + j] = degree1 + j;
			}
		}
	for (u = 0; u < nb_gens2; u++, k++) {
		for (i = 0; i < degree1; i++) {
			gens3[k * degree3 + i] = i;
			}
		for (j = 0; j < degree2; j++) {
			jj = gens2[u * degree2 + j];
			gens3[k * degree3 + degree1 + j] = degree1 + jj;
			for (i = 0; i < degree1; i++) {
				gens3[k * degree3 + offset + i * degree2 + j] = 
					offset + i * degree2 + jj;
				}
			}
		}
}

void perm_group_generators_direct_product(
	int nb_diagonal_elements,
	int degree1, int degree2, int &degree3, 
	int nb_gens1, int nb_gens2, int &nb_gens3, 
	int *gens1, int *gens2, int *&gens3, 
	int base_len1, int base_len2, int &base_len3, 
	int *base1, int *base2, int *&base3)
{
	int u, i, j, ii, jj, k, offset;
	
	offset = degree1 + degree2;
	degree3 = offset + degree1 * degree2;
	nb_gens3 = (nb_gens1 - nb_diagonal_elements) +
			(nb_gens2 - nb_diagonal_elements)
		+ nb_diagonal_elements;
	base_len3 = base_len1 + base_len2;
	gens3 = NEW_int(nb_gens3 * degree3);
	base3 = NEW_int(base_len3);
	for (u = 0; u < base_len1; u++) {
		base3[u] = base1[u];
		}
	for (u = 0; u < base_len2; u++) {
		base3[base_len1 + u] = degree1 + base2[u];
		}
	k = 0;
	for (u = 0; u < nb_gens1 - nb_diagonal_elements; u++, k++) {
		for (i = 0; i < degree1; i++) {
			ii = gens1[u * degree1 + i];
			gens3[k * degree3 + i] = ii;
			for (j = 0; j < degree2; j++) {
				gens3[k * degree3 + offset + i * degree2 + j] = 
					offset + ii * degree2 + j;
				}
			}
		for (j = 0; j < degree2; j++) {
			gens3[k * degree3 + degree1 + j] = degree1 + j;
			}
		}
	for (u = 0; u < nb_gens2 - nb_diagonal_elements; u++, k++) {
		for (i = 0; i < degree1; i++) {
			gens3[k * degree3 + i] = i;
			}
		for (j = 0; j < degree2; j++) {
			jj = gens2[u * degree2 + j];
			gens3[k * degree3 + degree1 + j] = degree1 + jj;
			for (i = 0; i < degree1; i++) {
				gens3[k * degree3 + offset + i * degree2 + j] = 
					offset + i * degree2 + jj;
				}
			}
		}
	for (u = 0; u < nb_diagonal_elements; u++, k++) {
		for (i = 0; i < degree1; i++) {
			ii = gens1[(nb_gens1 - nb_diagonal_elements + u)
					   * degree1 + i];
			gens3[k * degree3 + i] = ii;
			}
		for (j = 0; j < degree2; j++) {
			jj = gens2[(nb_gens2 - nb_diagonal_elements + u)
					   * degree2 + j];
			gens3[k * degree3 + degree1 + j] = degree1 + jj;
			}
		for (i = 0; i < degree1; i++) {
			ii = gens1[(nb_gens1 - nb_diagonal_elements + u)
					   * degree1 + i];
			for (j = 0; j < degree2; j++) {
				jj = gens2[(nb_gens2 - nb_diagonal_elements + u)
						   * degree2 + j];
				gens3[k * degree3 + offset + i * degree2 + j] = 
					offset + ii * degree2 + jj;
				}
			}
		}
}
#endif

}

