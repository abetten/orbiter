// permutation_representation_domain.cpp
//
// Anton Betten
//
// started: May 25, 2006




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace groups {


permutation_representation_domain::permutation_representation_domain()
{
	degree = 0;
	f_induced_action = false;
	f_product_action = false;
	m = n = mn = offset = 0;
	char_per_elt = 0;
	elt_size_int = 0;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;

	elt1 = NULL;
	elt2 = NULL;
	elt3 = NULL;

	Eltrk1 = NULL;
	Eltrk2 = NULL;
	Eltrk3 = NULL;

	Elts = NULL;
}


permutation_representation_domain::~permutation_representation_domain()
{
	if (Elt1)
		FREE_int(Elt1);
	if (Elt2)
		FREE_int(Elt2);
	if (Elt3)
		FREE_int(Elt3);
	if (Elt4)
		FREE_int(Elt4);
	//cout << "permutation_representation_domain::free before elt1" << endl;
	if (elt1)
		FREE_uchar(elt1);
	if (elt2)
		FREE_uchar(elt2);
	if (elt3)
		FREE_uchar(elt3);
	//cout << "permutation_representation_domain::free before Elts" << endl;
	if (Elts) {
		FREE_OBJECT(Elts);
	}
	if (Eltrk1)
		FREE_int(Eltrk1);
	if (Eltrk2)
		FREE_int(Eltrk2);
	if (Eltrk3)
		FREE_int(Eltrk3);
}

void permutation_representation_domain::allocate()
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

	Elts = NEW_OBJECT(data_structures::page_storage);
}

void permutation_representation_domain::init_product_action(int m, int n,
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "permutation_representation_domain::init_product_action "
				"m=" << m << " n=" << n << endl;
	}
	f_product_action = true;
	permutation_representation_domain::m = m;
	permutation_representation_domain::n = n;
	mn = m * n;
	offset = m + n;
	
	degree = m + n + m * n;
	elt_size_int = m + n;
	char_per_elt = elt_size_int;
	
	init_data(page_length_log, verbose_level);
	if (f_v) {
		cout << "permutation_representation_domain::init_product_action done" << endl;
	}
}

	
void permutation_representation_domain::init(int degree,
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "permutation_representation_domain::init" << endl;
	}
	permutation_representation_domain::degree = degree;
	f_product_action = false;
	
	elt_size_int = degree;
	char_per_elt = elt_size_int * sizeof(int);
	
	init_data(page_length_log, verbose_level);
	if (f_v) {
		cout << "permutation_representation_domain::init done" << endl;
	}
}

void permutation_representation_domain::init_data(int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int hdl;

	if (f_v) {
		cout << "permutation_representation_domain::init_data" << endl;
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
		cout << "permutation_representation_domain::init_data "
				"calling Elts->init()" << endl;
	}
	Elts->init(char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(perm_group_elt_print, (void *) this);


	if (f_vv) {
		cout << "permutation_representation_domain::init_data "
				"calling one()" << endl;
	}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "permutation_representation_domain::init_data "
				"calling Elts->store()" << endl;
	}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
	}
	

	if (f_vv) {
		cout << "permutation_representation_domain::init_data "
				"finished" << endl;
	}
	
	FREE_int(tmp1);
	FREE_int(tmp2);
	FREE_int(tmp3);
}

void permutation_representation_domain::init_with_base(int degree,
	int base_length, int *base, int page_length_log, 
	actions::action &A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, hdl;
	
	permutation_representation_domain::degree = degree;
	f_product_action = false;
	
	//A.base_len = base_length;
	A.degree = degree;
	elt_size_int = degree;
	char_per_elt = elt_size_int;
	

	allocate();

	int *tmp1 = NEW_int(elt_size_int);
	int *tmp2 = NEW_int(elt_size_int);
	int *tmp3 = NEW_int(elt_size_int);
	

	
	if (f_v) {
		cout << "permutation_representation_domain::init" << endl;
		cout << "degree=" << A.degree << endl;
		cout << "base_len=" << base_length << endl;
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
		cout << "permutation_representation_domain::init "
				"calling one()" << endl;
	}
	one(tmp1);
	//print(tmp1, cout);
	pack(tmp1, elt1);
	if (f_vv) {
		cout << "permutation_representation_domain::init "
				"calling Elts->store" << endl;
	}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, "
				"hdl = " << hdl << endl;
	}
	
	if (f_vv) {
		cout << "permutation_representation_domain::init "
				"initializing base, and transversal_length" << endl;
	}
	A.type_G = perm_group_t;
	A.G.perm_grp = this;
	
	A.Stabilizer_chain = NEW_OBJECT(actions::stabilizer_chain_base_data);
	A.Stabilizer_chain->allocate_base_data(&A, base_length, verbose_level);
	//A.Stabilizer_chain->base_len = base_length;
	//A.allocate_base_data(A.base_len);

	// init base:
	for (i = 0; i < A.base_len(); i++) {
		A.base_i(i) = base[i];
	}
	

	if (f_v) {
		cout << "base: ";
		Lint_vec_print(cout, A.get_base(), A.base_len());
		cout << endl;
		//cout << "transversal_length: ";
		//print_set(cout, A.base_len, A.transversal_length);
		//cout << endl;
	}

	A.ptr = NEW_OBJECT(actions::action_pointer_table);
	A.ptr->init_function_pointers_permutation_group();
	
	A.elt_size_in_int = elt_size_int;
	A.coded_elt_size_in_char = char_per_elt;
	
	A.allocate_element_data();

	A.label = "Sym" + std::to_string(degree);
	A.label_tex = "{\\rm Sym}_{" + std::to_string(degree) + "}";

	if (f_vv) {
		cout << "permutation_representation_domain::init finished" << endl;
	}
	
	FREE_int(tmp1);
	FREE_int(tmp2);
	FREE_int(tmp3);
}

void permutation_representation_domain::transversal_rep(int i, int j,
		int *Elt, int verbose_level)
{
	int j1, j2;
	
	one(Elt);
	j1 = i;
	j2 = i + j;
	Elt[j1] = j2;
	Elt[j2] = j1;
}

void permutation_representation_domain::one(int *Elt)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		Elt[i] = i;
	}
}

int permutation_representation_domain::is_one(int *Elt)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		if (Elt[i] != i) {
			return false;
		}
	}
	return true;
}

void permutation_representation_domain::mult(int *A, int *B, int *AB)
{
	combinatorics::combinatorics_domain Combi;

	//cout << "in perm_group::mult()" << endl;
	Combi.perm_mult(A, B, AB, degree);
	//cout << "in perm_group::mult()
	// finished with perm_mult" << endl;
}

void permutation_representation_domain::copy(int *A, int *B)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		B[i] = A[i];
	}
}

void permutation_representation_domain::invert(int *A, int *Ainv)
{
	combinatorics::combinatorics_domain Combi;

	Combi.perm_inverse(A, Ainv, degree);
}

void permutation_representation_domain::unpack(uchar *elt, int *Elt)
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

void permutation_representation_domain::pack(int *Elt, uchar *elt)
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

void permutation_representation_domain::print(int *Elt, std::ostream &ost)
{
	combinatorics::combinatorics_domain Combi;

	//cout << "perm_group::print before perm_print" << endl;
	Combi.perm_print(ost, Elt, degree);
	//ost << endl;
	//cout << "perm_group::print done" << endl;
}

void permutation_representation_domain::print_with_print_point_function(int *Elt,
		ostream &ost,
		void (*point_label)(
				std::stringstream &sstr, long int pt, void *data),
		void *point_label_data)
{
	combinatorics::combinatorics_domain Combi;

	//cout << "perm_group::print before perm_print" << endl;
	Combi.perm_print_with_print_point_function(ost, Elt, degree, point_label, point_label_data);
	//ost << endl;
	//cout << "perm_group::print done" << endl;
}

void permutation_representation_domain::code_for_make_element(int *Elt, int *data)
{
	Int_vec_copy(Elt, data, degree);
}

void permutation_representation_domain::print_for_make_element(int *Elt, std::ostream &ost)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << ", ";
	}
}

void permutation_representation_domain::print_for_make_element_no_commas(
		int *Elt, std::ostream &ost)
{
	int i;
	
	for (i = 0; i < degree; i++) {
		ost << Elt[i] << " ";
	}
}

void permutation_representation_domain::print_with_action(actions::action *A, int *Elt, std::ostream &ost)
{
	//perm_print(ost, Elt, degree);
	//ost << endl;
	int i, bi, a;
	int x1, y1, x2, y2; // if in product action
	combinatorics::combinatorics_domain Combi;
	
	if (A->base_len() < A->degree) {
		for (i = 0; i < A->base_len(); i++) {
			bi = A->base_i(i);
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
			if (i < A->base_len() - 1) {
				ost << ", ";
			}
		}
	}
	//perm_print(ost, Elt, degree);
	ost << " : ";
	Combi.perm_print_offset(ost, Elt, degree, 0 /* offset */,
			false /* f_print_cycles_of_length_one */,
			false /* f_cycle_length */, false, 0,
			false /* f_orbit_structure */,
			NULL, NULL);
	ost << " : ";
	Combi.perm_print_list_offset(ost, Elt, degree, 1);
	ost << endl;
}

void permutation_representation_domain::make_element(int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a;
	int *my_data;
	combinatorics::combinatorics_domain Combi;
	

	if (f_v) {
		cout << "permutation_representation_domain::make_element" << endl;
	}
	if (f_vv) {
		cout << "data: ";
		Int_vec_print(cout, data, elt_size_int);
		cout << endl;
	}

	my_data = NEW_int(elt_size_int);

	for (i = 0; i < elt_size_int; i++) {
		a = data[i];
		my_data[i] = a;
		Elt[i] = a;
	}

	if (!Combi.is_permutation(my_data, elt_size_int)) {
		cout << "permutation_representation_domain::make_element "
				"The input is not a permutation" << endl;
		exit(1);
	}

	FREE_int(my_data);

	if (f_v) {
		cout << "permutation_representation_domain::make_element done" << endl;
	}
}


}}}

