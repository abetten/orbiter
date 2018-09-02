// perm_group.C
//
// Anton Betten
//
// started: May 25, 2006




#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

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
		FREE_INT(Elt1);
	if (Elt2)
		FREE_INT(Elt2);
	if (Elt3)
		FREE_INT(Elt3);
	if (Elt4)
		FREE_INT(Elt4);
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
		FREE_INT(Eltrk1);
	if (Eltrk2)
		FREE_INT(Eltrk2);
	if (Eltrk3)
		FREE_INT(Eltrk3);
	null();
	//cout << "perm_group::free finished" << endl;
}

void perm_group::allocate()
{
	Elt1 = NEW_INT(elt_size_INT);
	Elt2 = NEW_INT(elt_size_INT);
	Elt3 = NEW_INT(elt_size_INT);
	Elt4 = NEW_INT(elt_size_INT);
	elt1 = NEW_uchar(char_per_elt);
	elt2 = NEW_uchar(char_per_elt);
	elt3 = NEW_uchar(char_per_elt);
	Eltrk1 = NEW_INT(elt_size_INT);
	Eltrk2 = NEW_INT(elt_size_INT);
	Eltrk3 = NEW_INT(elt_size_INT);

	Elts = NEW_OBJECT(page_storage);
}

void perm_group::init_product_action(INT m, INT n,
		INT page_length_log, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
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
	elt_size_INT = m + n;
	char_per_elt = elt_size_INT;
	
	init_data(page_length_log, verbose_level);
}

	
void perm_group::init(INT degree,
		INT page_length_log, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "perm_group::init()" << endl;
		}
	perm_group::degree = degree;
	f_product_action = FALSE;
	
	elt_size_INT = degree;
	char_per_elt = elt_size_INT * sizeof(INT);
	
	init_data(page_length_log, verbose_level);
}

void perm_group::init_data(INT page_length_log, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT hdl;

	if (f_v) {
		cout << "perm_group::init_data()" << endl;
		cout << "degree=" << degree << endl;
		cout << "elt_size_INT=" << elt_size_INT << endl;
		cout << "page_length_log=" << page_length_log << endl;
		//cout << "base_len=" << A.base_len << endl;
		}

	allocate();

	INT *tmp1 = NEW_INT(elt_size_INT);
	INT *tmp2 = NEW_INT(elt_size_INT);
	INT *tmp3 = NEW_INT(elt_size_INT);
	

	
	if (f_vv) {
		cout << "perm_group::init_data() "
				"calling Elts->init()" << endl;
		}
	Elts->init(char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(perm_group_elt_print, (void *) this);


	if (f_vv) {
		cout << "perm_group::init_data() "
				"calling one()" << endl;
		}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "perm_group::init_data() "
				"calling Elts->store()" << endl;
		}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
		}
	

	if (f_vv) {
		cout << "perm_group::init_data() "
				"finished" << endl;
		}
	
	FREE_INT(tmp1);
	FREE_INT(tmp2);
	FREE_INT(tmp3);
}

void perm_group::init_with_base(INT degree, 
	INT base_length, INT *base, INT page_length_log, 
	action &A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, hdl;
	
	perm_group::degree = degree;
	f_product_action = FALSE;
	
	A.base_len = base_length;
	A.degree = degree;
	elt_size_INT = degree;
	char_per_elt = elt_size_INT;
	

	allocate();

	INT *tmp1 = NEW_INT(elt_size_INT);
	INT *tmp2 = NEW_INT(elt_size_INT);
	INT *tmp3 = NEW_INT(elt_size_INT);
	

	
	if (f_v) {
		cout << "perm_group::init()" << endl;
		cout << "degree=" << A.degree << endl;
		cout << "base_len=" << A.base_len << endl;
		}
	if (f_vv) {
		cout << "perm_group::init() "
				"calling Elts->init()" << endl;
		}
	Elts->init(char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(
	//perm_group_elt_print, (void *) this);


	if (f_vv) {
		cout << "perm_group::init() "
				"calling one()" << endl;
		}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "perm_group::init() "
				"calling Elts->store()" << endl;
		}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
		}
	
	if (f_vv) {
		cout << "perm_group::init() "
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
	
	A.elt_size_in_INT = elt_size_INT;
	A.coded_elt_size_in_char = char_per_elt;
	
	A.allocate_element_data();

	sprintf(A.group_prefix, "Sym%ld", degree);

	if (f_vv) {
		cout << "perm_group::init() finished" << endl;
		}
	
	FREE_INT(tmp1);
	FREE_INT(tmp2);
	FREE_INT(tmp3);
}

void perm_group::transversal_rep(INT i, INT j,
		INT *Elt, INT verbose_level)
{
	INT j1, j2;
	
	one(Elt);
	j1 = i;
	j2 = i + j;
	Elt[j1] = j2;
	Elt[j2] = j1;
}

void perm_group::one(INT *Elt)
{
	INT i;
	
	for (i = 0; i < degree; i++) {
		Elt[i] = i;
		}
}

INT perm_group::is_one(INT *Elt)
{
	INT i;
	
	for (i = 0; i < degree; i++) {
		if (Elt[i] != i) {
			return FALSE;
			}
		}
	return TRUE;
}

void perm_group::mult(INT *A, INT *B, INT *AB)
{
	//cout << "in perm_group::mult()" << endl;
	perm_mult(A, B, AB, degree);
	//cout << "in perm_group::mult()
	// finished with perm_mult" << endl;
}

void perm_group::copy(INT *A, INT *B)
{
	INT i;
	
	for (i = 0; i < degree; i++) {
		B[i] = A[i];
		}
}

void perm_group::invert(INT *A, INT *Ainv)
{
	perm_inverse(A, Ainv, degree);
}

void perm_group::unpack(uchar *elt, INT *Elt)
{
	INT i, j;
	
	for (i = 0; i < degree; i++) {
		uchar *p;

		p = (uchar *)(Elt + i);
		for (j = 0; j < (INT) sizeof(INT); j++) {
			*p++ = *elt++;
			}
		}
}

void perm_group::pack(INT *Elt, uchar *elt)
{
	INT i, j;
	
	for (i = 0; i < degree; i++) {
		uchar *p;

		p = (uchar *)(Elt + i);
		for (j = 0; j < (INT) sizeof(INT); j++) {
			*elt++ = *p++;
			}
		}
}

void perm_group::print(INT *Elt, ostream &ost)
{
	//cout << "perm_group::print before perm_print" << endl;
	perm_print(ost, Elt, degree);
	//ost << endl;
	//cout << "perm_group::print done" << endl;
}

void perm_group::code_for_make_element(INT *Elt, INT *data)
{
	INT_vec_copy(Elt, data, degree);
}

void perm_group::print_for_make_element(INT *Elt, ostream &ost)
{
	INT i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << ", ";
		}
}

void perm_group::print_for_make_element_no_commas(
		INT *Elt, ostream &ost)
{
	INT i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << " ";
		}
}

void perm_group::print_with_action(action *A, INT *Elt, ostream &ost)
{
	//perm_print(ost, Elt, degree);
	//ost << endl;
	INT i, bi, a;
	INT x1, y1, x2, y2; // if in product action
	
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

void perm_group::make_element(INT *Elt, INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a;
	
	if (f_v) {
		cout << "perm_group::make_element" << endl;
		}
	if (f_vv) {
		cout << "data: ";
		INT_vec_print(cout, data, elt_size_INT);
		cout << endl;
		}
	for (i = 0; i < elt_size_INT; i++) {
		a = data[i];
		Elt[i] = a;
		}
	if (f_v) {
		cout << "perm_group::make_element done" << endl;
		}
}



//#############################################################################
// global functions:
//#############################################################################


void perm_group_find_strong_generators_at_level(
	INT level, INT degree,
	INT given_base_length, INT *given_base,
	INT nb_gens, INT *gens, 
	INT &nb_generators_found, INT *idx_generators_found)
{
	INT i, j, bj, bj_image;
	
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
	INT degree1, INT degree2, INT &degree3,
	INT nb_gens1, INT nb_gens2, INT &nb_gens3, 
	INT *gens1, INT *gens2, INT *&gens3, 
	INT base_len1, INT base_len2, INT &base_len3, 
	INT *base1, INT *base2, INT *&base3)
{
	INT u, i, j, ii, jj, k, offset;
	
	offset = degree1 + degree2;
	degree3 = offset + degree1 * degree2;
	nb_gens3 = nb_gens1 + nb_gens2;
	base_len3 = base_len1 + base_len2;
	gens3 = NEW_INT(nb_gens3 * degree3);
	base3 = NEW_INT(base_len3);
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
	INT nb_diagonal_elements,
	INT degree1, INT degree2, INT &degree3, 
	INT nb_gens1, INT nb_gens2, INT &nb_gens3, 
	INT *gens1, INT *gens2, INT *&gens3, 
	INT base_len1, INT base_len2, INT &base_len3, 
	INT *base1, INT *base2, INT *&base3)
{
	INT u, i, j, ii, jj, k, offset;
	
	offset = degree1 + degree2;
	degree3 = offset + degree1 * degree2;
	nb_gens3 = (nb_gens1 - nb_diagonal_elements) +
			(nb_gens2 - nb_diagonal_elements)
		+ nb_diagonal_elements;
	base_len3 = base_len1 + base_len2;
	gens3 = NEW_INT(nb_gens3 * degree3);
	base3 = NEW_INT(base_len3);
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

