/*
 * interface_polarity_extension.cpp
 *
 *  Created on: Jan 24, 2024
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {

// #############################################################################
// interface functions: polarity extension
// #############################################################################



static long int polarity_extension_element_image_of(
		action &A, long int a,
	void *elt, int verbose_level);
static void polarity_extension_element_image_of_low_level(
		action &A,
	int *input, int *output, void *elt, int verbose_level);
static int polarity_extension_element_linear_entry_ij(
		action &A,
	void *elt, int i, int j, int verbose_level);
static int polarity_extension_element_linear_entry_frobenius(
		action &A,
	void *elt, int verbose_level);
static void polarity_extension_element_one(
		action &A,
	void *elt, int verbose_level);
static int polarity_extension_element_is_one(
		action &A,
	void *elt, int verbose_level);
static void polarity_extension_element_unpack(
		action &A,
	void *elt, void *Elt, int verbose_level);
static void polarity_extension_element_pack(
		action &A,
	void *Elt, void *elt, int verbose_level);
static void polarity_extension_element_retrieve(
		action &A,
	int hdl, void *elt, int verbose_level);
static int polarity_extension_element_store(
		action &A,
	void *elt, int verbose_level);
static void polarity_extension_element_mult(
		action &A,
	void *a, void *b, void *ab, int verbose_level);
static void polarity_extension_element_invert(
		action &A,
	void *a, void *av, int verbose_level);
static void polarity_extension_element_transpose(
		action &A,
	void *a, void *at, int verbose_level);
static void polarity_extension_element_move(
		action &A,
	void *a, void *b, int verbose_level);
static void polarity_extension_element_dispose(
		action &A,
	int hdl, int verbose_level);
static void polarity_extension_element_print(
		action &A,
	void *elt, std::ostream &ost);
static void polarity_extension_element_code_for_make_element(
	action &A, void *elt, int *data);
#if 0
static void polarity_extension_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
static void polarity_extension_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
#endif
static void polarity_extension_element_print_quick(
		action &A,
	void *elt, std::ostream &ost);
static void polarity_extension_element_print_latex(
		action &A,
	void *elt, std::ostream &ost);
static std::string polarity_extension_element_stringify(
		action &A, void *elt, std::string &options);
static void polarity_extension_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data);
//static void polarity_extension_element_print_as_permutation(
//	action &A, void *elt, std::ostream &ost);
static void polarity_extension_element_print_verbose(
		action &A,
	void *elt, std::ostream &ost);
static void polarity_extension_print_point(
		action &A,
	long int a, std::ostream &ost, int verbose_level);
static void polarity_extension_unrank_point(
		action &A, long int rk, int *v, int verbose_level);
static long int polarity_extension_rank_point(
		action &A, int *v, int verbose_level);
static std::string polarity_extension_stringify_point(
		action &A, long int rk, int verbose_level);




void action_pointer_table::init_function_pointers_polarity_extension()
{
	label.assign("function_pointers_polarity_extension");

	// the first 10:
	ptr_element_image_of = polarity_extension_element_image_of;
	ptr_element_image_of_low_level =
			polarity_extension_element_image_of_low_level;
	ptr_element_linear_entry_ij =
			polarity_extension_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			polarity_extension_element_linear_entry_frobenius;
	ptr_element_one = polarity_extension_element_one;
	ptr_element_is_one = polarity_extension_element_is_one;
	ptr_element_unpack = polarity_extension_element_unpack;
	ptr_element_pack = polarity_extension_element_pack;
	ptr_element_retrieve = polarity_extension_element_retrieve;
	ptr_element_store = polarity_extension_element_store;


	// the next 10:
	ptr_element_mult = polarity_extension_element_mult;
	ptr_element_invert = polarity_extension_element_invert;
	ptr_element_transpose = polarity_extension_element_transpose;
	ptr_element_move = polarity_extension_element_move;
	ptr_element_dispose = polarity_extension_element_dispose;
	ptr_element_print = polarity_extension_element_print;
	ptr_element_print_quick = polarity_extension_element_print_quick;
	ptr_element_print_latex = polarity_extension_element_print_latex;
	ptr_element_stringify = polarity_extension_element_stringify;
	ptr_element_print_latex_with_point_labels =
			polarity_extension_element_print_latex_with_point_labels;


	// the next 6:
	ptr_element_print_verbose = polarity_extension_element_print_verbose;
	ptr_element_code_for_make_element =
			polarity_extension_element_code_for_make_element;
#if 0
	ptr_element_print_for_make_element =
			polarity_extension_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			polarity_extension_element_print_for_make_element_no_commas;
#endif
	ptr_print_point = polarity_extension_print_point;
	ptr_unrank_point = polarity_extension_unrank_point;
	ptr_rank_point = polarity_extension_rank_point;
	ptr_stringify_point = polarity_extension_stringify_point;
}



static long int polarity_extension_element_image_of(
		action &A,
		long int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;
	long int b;

	if (f_v) {
		cout << "polarity_extension_element_image_of "
				"computing image of " << a << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	b = P.element_image_of(Elt, a, verbose_level - 1);

	if (f_v) {
		cout << "polarity_extension_element_image_of "
				"image of " << a << " is " << b << endl;
		}
	return b;
}

static void polarity_extension_element_image_of_low_level(
		action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	cout << "polarity_extension_element_image_of_low_level "
			"N/A";
	exit(1);
}

static int polarity_extension_element_linear_entry_ij(
		action &A,
		void *elt, int i, int j, int verbose_level)
{
	cout << "polarity_extension_element_linear_entry_ij "
			"N/A" << endl;
	exit(1);
}

static int polarity_extension_element_linear_entry_frobenius(
		action &A,
		void *elt, int verbose_level)
{
	cout << "polarity_extension_element_linear_entry_frobenius "
			"N/A" << endl;
	exit(1);
}

static void polarity_extension_element_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "polarity_extension_element_one "
				"calling element_one" << endl;
		}
	P.element_one(Elt);
}

static int polarity_extension_element_is_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;
	int ret;

	if (f_v) {
		cout << "polarity_extension_element_one calling "
				"element_is_one" << endl;
		}
	ret = P.element_is_one(Elt);
	if (f_v) {
		if (ret) {
			cout << "polarity_extension_element_is_one "
					"returns YES" << endl;
			}
		else {
			cout << "polarity_extension_element_is_one "
					"returns NO" << endl;
			}
		}
	return ret;
}

static void polarity_extension_element_unpack(
		action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "polarity_extension_element_unpack" << endl;
		}
	P.element_unpack(elt1, Elt1);
}

static void polarity_extension_element_pack(
		action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "polarity_extension_element_pack" << endl;
		}
	P.element_pack(Elt1, elt1);
}

static void polarity_extension_element_retrieve(
		action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;
	uchar *p_elt;

	if (f_v) {
		cout << "polarity_extension_element_"
				"retrieve hdl = " << hdl << endl;
		}
	p_elt = P.Page_storage->s_i(hdl);
	//if (f_v) {
	//	element_print_packed(G, p_elt, cout);
	//	}
	P.element_unpack(p_elt, Elt);
	if (f_v) {
		P.element_print_easy(Elt, cout);
		}
}

static int polarity_extension_element_store(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;
	int hdl;

	if (f_v) {
		cout << "polarity_extension_element_store" << endl;
		}
	P.element_pack(Elt, P.elt1);
	hdl = P.Page_storage->store(P.elt1);
	if (f_v) {
		cout << "polarity_extension_element_store "
				"hdl = " << hdl << endl;
		}
	return hdl;
}

static void polarity_extension_element_mult(
		action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "polarity_extension_element_mult" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		cout << "B=" << endl;
		P.element_print_easy(BB, cout);
		}
	P.element_mult(AA, BB, AB, verbose_level - 2);
	if (f_v) {
		cout << "polarity_extension_element_mult done" << endl;
		}
	if (f_vv) {
		cout << "AB=" << endl;
		P.element_print_easy(AB, cout);
		}
}

static void polarity_extension_element_invert(
		action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "polarity_extension_element_invert" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		P.element_print_easy(AA, cout);
		}
	P.element_invert(AA, AAv, verbose_level - 1);
	if (f_v) {
		cout << "polarity_extension_element_invert done" << endl;
		}
	if (f_vv) {
		cout << "Av=" << endl;
		P.element_print_easy(AAv, cout);
		}
}

static void polarity_extension_element_transpose(
		action &A,
		void *a, void *at, int verbose_level)
{
	cout << "polarity_extension_element_transpose "
			"not yet implemented" << endl;
	exit(1);
}

static void polarity_extension_element_move(
		action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "polarity_extension_element_move" << endl;
		}
	P.element_move(AA, BB, 0 /* verbose_level */);
}

static void polarity_extension_element_dispose(
		action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;

	if (f_v) {
		cout << "polarity_extension_element_dispose "
				"hdl = " << hdl << endl;
		}
	P.Page_storage->dispose(hdl);
}

static void polarity_extension_element_print(
		action &A,
		void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;


	P.element_print_easy(Elt, ost);
	ost << endl;
}

static void polarity_extension_element_code_for_make_element(
		action &A,
		void *elt, int *data)
{
	group_constructions::polarity_extension *P = A.G.Polarity_extension;
	int *Elt = (int *) elt;


	P->element_code_for_make_element(
			Elt, data);
}

#if 0
static void polarity_extension_element_print_for_make_element(
		action &A,
		void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension *P = A.G.Polarity_extension;
	int *Elt = (int *) elt;


	P->element_print_for_make_element(
			Elt, ost);
}

static void polarity_extension_element_print_for_make_element_no_commas(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension *P = A.G.Polarity_extension;
	int *Elt = (int *) elt;

	P->element_print_for_make_element_no_commas(
			Elt, ost);
}
#endif

static void polarity_extension_element_print_quick(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;

	P.element_print_easy(Elt, ost);

}

static void polarity_extension_element_print_latex(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;

	P.element_print_easy_latex(Elt, ost);
}

static std::string polarity_extension_element_stringify(
		action &A, void *elt, std::string &options)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;
	string s;

	s = P.element_stringify(Elt, options);
	return s;
}


static void polarity_extension_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data)
{
	cout << "polarity_extension_element_print_latex_with_point_labels "
			"not yet implemented" << endl;
	exit(1);

}

#if 0
static void polarity_extension_element_print_as_permutation(
		action &A, void *elt, std::ostream &ost)
{
	//polarity_extension &P = *A.G.polarity_extension;
	combinatorics_domain Combi;
	int f_v = false;
	int *Elt = (int *) elt;
	int i, j;

	if (f_v) {
		cout << "polarity_extension_element_print_as_permutation "
				"degree = " << A.degree << endl;
		}
	int *p = NEW_int(A.degree);
	for (i = 0; i < A.degree; i++) {
		//cout << "matrix_group_element_print_as_permutation "
		//"computing image of i=" << i << endl;
		//if (i == 3)
			//f_v = true;
		//else
			//f_v = false;
		j = A.element_image_of(i, Elt, 0 /* verbose_level */);
		p[i] = j;
		}
	Combi.perm_print(ost, p, A.degree);
	FREE_int(p);
}
#endif

static void polarity_extension_element_print_verbose(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	int *Elt = (int *) elt;

	P.element_print_easy(Elt, ost);
}

static void polarity_extension_print_point(
		action &A, long int a, std::ostream &ost, int verbose_level)
{
	cout << "polarity_extension_print_point "
			"not yet implemented" << endl;
	exit(1);
}

static void polarity_extension_unrank_point(
		action &A, long int rk, int *v, int verbose_level)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;

	P.unrank_point(rk, v, verbose_level);

}

static long int polarity_extension_rank_point(
		action &A, int *v, int verbose_level)
{
	group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	long int rk;


	rk = P.rank_point(v, verbose_level);

	return rk;
}

static std::string polarity_extension_stringify_point(
		action &A, long int rk, int verbose_level)
{
	//group_constructions::polarity_extension &P = *A.G.Polarity_extension;
	string s;


	//rk = P.rank_point(v, verbose_level);

	return s;
}


}}}




