/*
 * interface_permutation_representation.cpp
 *
 *  Created on: Aug 23, 2019
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


// #############################################################################
// interface functions: permutation_representation group
// #############################################################################


static long int permutation_representation_group_element_image_of(action &A, long int a,
	void *elt, int verbose_level);
static void permutation_representation_group_element_image_of_low_level(action &A,
	int *input, int *output, void *elt, int verbose_level);
static int permutation_representation_group_element_linear_entry_ij(action &A,
	void *elt, int i, int j, int verbose_level);
static int permutation_representation_group_element_linear_entry_frobenius(action &A,
	void *elt, int verbose_level);
static void permutation_representation_group_element_one(action &A,
	void *elt, int verbose_level);
static int permutation_representation_group_element_is_one(action &A,
	void *elt, int verbose_level);
static void permutation_representation_group_element_unpack(action &A,
	void *elt, void *Elt, int verbose_level);
static void permutation_representation_group_element_pack(action &A,
	void *Elt, void *elt, int verbose_level);
static void permutation_representation_group_element_retrieve(action &A,
	int hdl, void *elt, int verbose_level);
static int permutation_representation_group_element_store(action &A,
	void *elt, int verbose_level);
static void permutation_representation_group_element_mult(action &A,
	void *a, void *b, void *ab, int verbose_level);
static void permutation_representation_group_element_invert(action &A,
	void *a, void *av, int verbose_level);
static void permutation_representation_group_element_transpose(action &A,
	void *a, void *at, int verbose_level);
static void permutation_representation_group_element_move(action &A,
	void *a, void *b, int verbose_level);
static void permutation_representation_group_element_dispose(action &A,
	int hdl, int verbose_level);
static void permutation_representation_group_element_print(action &A,
	void *elt, std::ostream &ost);
static void permutation_representation_group_element_code_for_make_element(
	action &A, void *elt, int *data);
static void permutation_representation_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
static void permutation_representation_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
static void permutation_representation_group_element_print_quick(action &A,
	void *elt, std::ostream &ost);
static void permutation_representation_group_element_print_latex(action &A,
	void *elt, std::ostream &ost);
static void permutation_representation_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data);
static void permutation_representation_group_element_print_verbose(action &A,
	void *elt, std::ostream &ost);
static void permutation_representation_group_print_point(action &A,
	long int a, std::ostream &ost);


void action_pointer_table::init_function_pointers_permutation_representation_group()
{
	label.assign("function_pointers_permutation_representation_group");
	ptr_element_image_of = permutation_representation_group_element_image_of;
	ptr_element_image_of_low_level =
			permutation_representation_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = permutation_representation_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			permutation_representation_group_element_linear_entry_frobenius;
	ptr_element_one = permutation_representation_group_element_one;
	ptr_element_is_one = permutation_representation_group_element_is_one;
	ptr_element_unpack = permutation_representation_group_element_unpack;
	ptr_element_pack = permutation_representation_group_element_pack;
	ptr_element_retrieve = permutation_representation_group_element_retrieve;
	ptr_element_store = permutation_representation_group_element_store;
	ptr_element_mult = permutation_representation_group_element_mult;
	ptr_element_invert = permutation_representation_group_element_invert;
	ptr_element_transpose = permutation_representation_group_element_transpose;
	ptr_element_move = permutation_representation_group_element_move;
	ptr_element_dispose = permutation_representation_group_element_dispose;
	ptr_element_print = permutation_representation_group_element_print;
	ptr_element_print_quick = permutation_representation_group_element_print_quick;
	ptr_element_print_latex = permutation_representation_group_element_print_latex;
	ptr_element_print_latex_with_print_point_function =
			permutation_representation_group_element_print_latex_with_print_point_function;
	ptr_element_print_verbose = permutation_representation_group_element_print_verbose;
	ptr_element_code_for_make_element =
			permutation_representation_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			permutation_representation_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			permutation_representation_group_element_print_for_make_element_no_commas;
	ptr_print_point = permutation_representation_group_print_point;
}



static long int permutation_representation_group_element_image_of(action &A,
		long int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;
	long int b;

	if (f_v) {
		cout << "permutation_representation_group_element_image_of "
				"computing image of " << a << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	b = P.element_image_of(Elt, a, verbose_level - 1);

	if (f_v) {
		cout << "permutation_representation_group_element_image_of "
				"image of " << a << " is " << b << endl;
		}
	return b;
}

static void permutation_representation_group_element_image_of_low_level(action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *Elt = (int *) elt;

	cout << "permutation_representation_group_element_image_of_low_level "
			"nyi " << endl;
}

static int permutation_representation_group_element_linear_entry_ij(action &A,
		void *elt, int i, int j, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *Elt = (int *) elt;
	//int w;

	cout << "permutation_representation_group_element_linear_entry_ij "
			"does not exist" << endl;
	exit(1);
}

static int permutation_representation_group_element_linear_entry_frobenius(action &A,
		void *elt, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *Elt = (int *) elt;
	//int w;

	cout << "permutation_representation_group_element_linear_entry_frobenius "
			"does not exist" << endl;
	exit(1);
}

static void permutation_representation_group_element_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "permutation_representation_group_element_one "
				"calling element_one" << endl;
		}
	P.element_one(Elt);
}

static int permutation_representation_group_element_is_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;
	int ret;

	if (f_v) {
		cout << "permutation_representation_group_element_one calling "
				"element_is_one" << endl;
		}
	ret = P.element_is_one(Elt);
	if (f_v) {
		if (ret) {
			cout << "permutation_representation_group_element_is_one "
					"returns YES" << endl;
			}
		else {
			cout << "permutation_representation_group_element_is_one "
					"returns NO" << endl;
			}
		}
	return ret;
}

static void permutation_representation_group_element_unpack(action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "permutation_representation_group_element_unpack" << endl;
		}
	P.element_unpack(elt1, Elt1);
}

static void permutation_representation_group_element_pack(action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "permutation_representation_group_element_pack" << endl;
		}
	P.element_pack(Elt1, elt1);
}

static void permutation_representation_group_element_retrieve(action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;
	uchar *p_elt;

	if (f_v) {
		cout << "permutation_representation_group_element_"
				"retrieve hdl = " << hdl << endl;
		}
	p_elt = P.PS->s_i(hdl);
	P.element_unpack(p_elt, Elt);
	if (f_v) {
		P.element_print_easy(Elt, cout);
		}
}

static int permutation_representation_group_element_store(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;
	int hdl;

	if (f_v) {
		cout << "permutation_representation_group_element_store" << endl;
		}
	P.element_pack(Elt, P.elt1);
	hdl = P.PS->store(P.elt1);
	if (f_v) {
		cout << "permutation_representation_group_element_store "
				"hdl = " << hdl << endl;
		}
	return hdl;
}

static void permutation_representation_group_element_mult(action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "permutation_representation_group_element_mult" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		cout << "B=" << endl;
		P.element_print_easy(BB, cout);
		}
	P.element_mult(AA, BB, AB, verbose_level - 2);
	if (f_v) {
		cout << "permutation_representation_group_element_mult done" << endl;
		}
	if (f_vv) {
		cout << "AB=" << endl;
		P.element_print_easy(AB, cout);
		}
}

static void permutation_representation_group_element_invert(action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "permutation_representation_group_element_invert" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		}
	P.element_invert(AA, AAv, verbose_level - 1);
	if (f_v) {
		cout << "permutation_representation_group_element_invert done" << endl;
		}
	if (f_vv) {
		cout << "Av=" << endl;
		P.element_print_easy(AAv, cout);
		}
}

static void permutation_representation_group_element_transpose(action &A,
		void *a, void *at, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *AA = (int *) a;
	//int *Atv = (int *) at;

	cout << "permutation_representation_group_element_transpose "
			"not yet implemented" << endl;
	exit(1);
}

static void permutation_representation_group_element_move(action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "permutation_representation_group_element_move" << endl;
		}
	P.element_move(AA, BB, 0 /* verbose_level */);
}

static void permutation_representation_group_element_dispose(action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::permutation_representation &P = *A.G.Permutation_representation;

	if (f_v) {
		cout << "permutation_representation_group_element_dispose "
				"hdl = " << hdl << endl;
		}
	P.PS->dispose(hdl);
}

static void permutation_representation_group_element_print(action &A,
		void *elt, ostream &ost)
{
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;


	P.element_print_easy(Elt, ost);
	ost << endl;
}

static void permutation_representation_group_element_code_for_make_element(action &A,
		void *elt, int *data)
{
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *Elt = (int *) elt;

	cout << "permutation_representation_group_element_code_for_make_element "
			"not yet implemented" << endl;
	exit(1);
}

static void permutation_representation_group_element_print_for_make_element(action &A,
		void *elt, ostream &ost)
{
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;


	P.element_print_for_make_element(Elt, ost);
}

static void permutation_representation_group_element_print_for_make_element_no_commas(
		action &A, void *elt, ostream &ost)
{
	//permutation_representation &P = *A.G.Permutation_representation;
	//int *Elt = (int *) elt;

	cout << "permutation_representation_group_element_print_for_make_element_no_commas "
			"not yet implemented" << endl;
	exit(1);
}

static void permutation_representation_group_element_print_quick(
		action &A, void *elt, ostream &ost)
{
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;


	P.element_print_easy(Elt, ost);
}

static void permutation_representation_group_element_print_latex(
		action &A, void *elt, ostream &ost)
{
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;


	P.element_print_latex(Elt, ost);
}

static void permutation_representation_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data)
{
	cout << "permutation_representation_group_element_print_latex_with_print_point_function "
			"not yet implemented" << endl;
	exit(1);
}

static void permutation_representation_group_element_print_verbose(
		action &A, void *elt, ostream &ost)
{
	groups::permutation_representation &P = *A.G.Permutation_representation;
	int *Elt = (int *) elt;

	P.element_print_easy(Elt, ost);

}

static void permutation_representation_group_print_point(action &A, long int a, ostream &ost)
{
	//permutation_representation &P = *A.G.Permutation_representation;

	cout << "permutation_representation_group_print_point "
			"not yet implemented" << endl;
	exit(1);
}

}}}


