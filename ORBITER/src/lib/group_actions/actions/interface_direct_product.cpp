// interface_direct_product_group.cpp
//
// Anton Betten
//
// started:  August 12, 2018




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

// #############################################################################
// interface functions: direct product group
// #############################################################################



static long int direct_product_group_element_image_of(action &A, long int a,
	void *elt, int verbose_level);
static void direct_product_group_element_image_of_low_level(action &A,
	int *input, int *output, void *elt, int verbose_level);
static int direct_product_group_element_linear_entry_ij(action &A,
	void *elt, int i, int j, int verbose_level);
static int direct_product_group_element_linear_entry_frobenius(action &A,
	void *elt, int verbose_level);
static void direct_product_group_element_one(action &A,
	void *elt, int verbose_level);
static int direct_product_group_element_is_one(action &A,
	void *elt, int verbose_level);
static void direct_product_group_element_unpack(action &A,
	void *elt, void *Elt, int verbose_level);
static void direct_product_group_element_pack(action &A,
	void *Elt, void *elt, int verbose_level);
static void direct_product_group_element_retrieve(action &A,
	int hdl, void *elt, int verbose_level);
static int direct_product_group_element_store(action &A,
	void *elt, int verbose_level);
static void direct_product_group_element_mult(action &A,
	void *a, void *b, void *ab, int verbose_level);
static void direct_product_group_element_invert(action &A,
	void *a, void *av, int verbose_level);
static void direct_product_group_element_transpose(action &A,
	void *a, void *at, int verbose_level);
static void direct_product_group_element_move(action &A,
	void *a, void *b, int verbose_level);
static void direct_product_group_element_dispose(action &A,
	int hdl, int verbose_level);
static void direct_product_group_element_print(action &A,
	void *elt, std::ostream &ost);
static void direct_product_group_element_code_for_make_element(
	action &A, void *elt, int *data);
static void direct_product_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
static void direct_product_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
static void direct_product_group_element_print_quick(action &A,
	void *elt, std::ostream &ost);
static void direct_product_group_element_print_latex(action &A,
	void *elt, std::ostream &ost);
static void direct_product_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data);
//static void direct_product_group_element_print_as_permutation(
//	action &A, void *elt, std::ostream &ost);
static void direct_product_group_element_print_verbose(action &A,
	void *elt, std::ostream &ost);
static void direct_product_group_print_point(action &A,
	long int a, std::ostream &ost);




void action_pointer_table::init_function_pointers_direct_product_group()
{
	ptr_element_image_of = direct_product_group_element_image_of;
	ptr_element_image_of_low_level =
			direct_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij =
			direct_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			direct_product_group_element_linear_entry_frobenius;
	ptr_element_one = direct_product_group_element_one;
	ptr_element_is_one = direct_product_group_element_is_one;
	ptr_element_unpack = direct_product_group_element_unpack;
	ptr_element_pack = direct_product_group_element_pack;
	ptr_element_retrieve = direct_product_group_element_retrieve;
	ptr_element_store = direct_product_group_element_store;
	ptr_element_mult = direct_product_group_element_mult;
	ptr_element_invert = direct_product_group_element_invert;
	ptr_element_transpose = direct_product_group_element_transpose;
	ptr_element_move = direct_product_group_element_move;
	ptr_element_dispose = direct_product_group_element_dispose;
	ptr_element_print = direct_product_group_element_print;
	ptr_element_print_quick = direct_product_group_element_print_quick;
	ptr_element_print_latex = direct_product_group_element_print_latex;
	ptr_element_print_latex_with_print_point_function =
			direct_product_group_element_print_latex_with_print_point_function;
	ptr_element_print_verbose = direct_product_group_element_print_verbose;
	ptr_element_code_for_make_element =
			direct_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			direct_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			direct_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = direct_product_group_print_point;
}



static long int direct_product_group_element_image_of(action &A,
		long int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;
	long int b;

	if (f_v) {
		cout << "direct_product_group_element_image_of "
				"computing image of " << a << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	b = P.element_image_of(Elt, a, verbose_level - 1);

	if (f_v) {
		cout << "direct_product_group_element_image_of "
				"image of " << a << " is " << b << endl;
		}
	return b;
}

static void direct_product_group_element_image_of_low_level(action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	cout << "direct_product_group_element_image_of_low_level "
			"N/A";
	exit(1);
}

static int direct_product_group_element_linear_entry_ij(action &A,
		void *elt, int i, int j, int verbose_level)
{
	cout << "direct_product_group_element_linear_entry_ij "
			"N/A" << endl;
	exit(1);
}

static int direct_product_group_element_linear_entry_frobenius(action &A,
		void *elt, int verbose_level)
{
	cout << "direct_product_group_element_linear_entry_frobenius "
			"N/A" << endl;
	exit(1);
}

static void direct_product_group_element_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "direct_product_group_element_one "
				"calling element_one" << endl;
		}
	P.element_one(Elt);
}

static int direct_product_group_element_is_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;
	int ret;

	if (f_v) {
		cout << "direct_product_group_element_one calling "
				"element_is_one" << endl;
		}
	ret = P.element_is_one(Elt);
	if (f_v) {
		if (ret) {
			cout << "direct_product_group_element_is_one "
					"returns YES" << endl;
			}
		else {
			cout << "direct_product_group_element_is_one "
					"returns NO" << endl;
			}
		}
	return ret;
}

static void direct_product_group_element_unpack(action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "direct_product_group_element_unpack" << endl;
		}
	P.element_unpack(elt1, Elt1);
}

static void direct_product_group_element_pack(action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "direct_product_group_element_pack" << endl;
		}
	P.element_pack(Elt1, elt1);
}

static void direct_product_group_element_retrieve(action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;
	uchar *p_elt;

	if (f_v) {
		cout << "direct_product_group_element_"
				"retrieve hdl = " << hdl << endl;
		}
	p_elt = P.Elts->s_i(hdl);
	//if (f_v) {
	//	element_print_packed(G, p_elt, cout);
	//	}
	P.element_unpack(p_elt, Elt);
	if (f_v) {
		P.element_print_easy(Elt, cout);
		}
}

static int direct_product_group_element_store(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;
	int hdl;

	if (f_v) {
		cout << "direct_product_group_element_store" << endl;
		}
	P.element_pack(Elt, P.elt1);
	hdl = P.Elts->store(P.elt1);
	if (f_v) {
		cout << "direct_product_group_element_store "
				"hdl = " << hdl << endl;
		}
	return hdl;
}

static void direct_product_group_element_mult(action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	direct_product &P = *A.G.direct_product_group;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "direct_product_group_element_mult" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		cout << "B=" << endl;
		P.element_print_easy(BB, cout);
		}
	P.element_mult(AA, BB, AB, verbose_level - 2);
	if (f_v) {
		cout << "direct_product_group_element_mult done" << endl;
		}
	if (f_vv) {
		cout << "AB=" << endl;
		P.element_print_easy(AB, cout);
		}
}

static void direct_product_group_element_invert(action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	direct_product &P = *A.G.direct_product_group;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "direct_product_group_element_invert" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		}
	P.element_invert(AA, AAv, verbose_level - 1);
	if (f_v) {
		cout << "direct_product_group_element_invert done" << endl;
		}
	if (f_vv) {
		cout << "Av=" << endl;
		P.element_print_easy(AAv, cout);
		}
}

static void direct_product_group_element_transpose(action &A,
		void *a, void *at, int verbose_level)
{
	cout << "direct_product_group_element_transpose "
			"not yet implemented" << endl;
	exit(1);
}

static void direct_product_group_element_move(action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "direct_product_group_element_move" << endl;
		}
	P.element_move(AA, BB, 0 /* verbose_level */);
}

static void direct_product_group_element_dispose(action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	direct_product &P = *A.G.direct_product_group;

	if (f_v) {
		cout << "direct_product_group_element_dispose "
				"hdl = " << hdl << endl;
		}
	P.Elts->dispose(hdl);
}

static void direct_product_group_element_print(action &A,
		void *elt, ostream &ost)
{
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;


	P.element_print_easy(Elt, ost);
	ost << endl;
}

static void direct_product_group_element_code_for_make_element(action &A,
		void *elt, int *data)
{
	cout << "direct_product_group_element_code_for_make_element "
			"not yet implemented" << endl;
	exit(1);
}

static void direct_product_group_element_print_for_make_element(action &A,
		void *elt, ostream &ost)
{
	cout << "direct_product_group_element_print_for_make_element "
			"not yet implemented" << endl;
	exit(1);
}

static void direct_product_group_element_print_for_make_element_no_commas(
		action &A, void *elt, ostream &ost)
{
	cout << "direct_product_group_element_print_for_make_element_no_commas "
			"not yet implemented" << endl;
	exit(1);
}

static void direct_product_group_element_print_quick(
		action &A, void *elt, ostream &ost)
{
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;

	P.element_print_easy(Elt, ost);

}

static void direct_product_group_element_print_latex(
		action &A, void *elt, ostream &ost)
{
	cout << "direct_product_group_element_print_latex "
			"not yet implemented" << endl;
	exit(1);
}


static void direct_product_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data)
{
	cout << "direct_product_group_element_print_latex_with_print_point_function "
			"not yet implemented" << endl;
	exit(1);

}

#if 0
static void direct_product_group_element_print_as_permutation(
		action &A, void *elt, ostream &ost)
{
	//direct_product &P = *A.G.direct_product_group;
	combinatorics_domain Combi;
	int f_v = FALSE;
	int *Elt = (int *) elt;
	int i, j;

	if (f_v) {
		cout << "direct_product_group_element_print_as_permutation "
				"degree = " << A.degree << endl;
		}
	int *p = NEW_int(A.degree);
	for (i = 0; i < A.degree; i++) {
		//cout << "matrix_group_element_print_as_permutation "
		//"computing image of i=" << i << endl;
		//if (i == 3)
			//f_v = TRUE;
		//else
			//f_v = FALSE;
		j = A.element_image_of(i, Elt, 0 /* verbose_level */);
		p[i] = j;
		}
	Combi.perm_print(ost, p, A.degree);
	FREE_int(p);
}
#endif

static void direct_product_group_element_print_verbose(
		action &A, void *elt, ostream &ost)
{
	direct_product &P = *A.G.direct_product_group;
	int *Elt = (int *) elt;

	P.element_print_easy(Elt, ost);
}

static void direct_product_group_print_point(action &A, long int a, ostream &ost)
{
	cout << "direct_product_group_print_point "
			"not yet implemented" << endl;
	exit(1);
}

}}



