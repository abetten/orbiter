// interface_wreath_product_group.cpp
//
// Anton Betten
//
// started:  August 4, 2018




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {


// #############################################################################
// interface functions: wreath product group
// #############################################################################


static long int wreath_product_group_element_image_of(
		action &A, long int a,
	void *elt, int verbose_level);
static void wreath_product_group_element_image_of_low_level(
		action &A,
	int *input, int *output, void *elt, int verbose_level);
static int wreath_product_group_element_linear_entry_ij(
		action &A,
	void *elt, int i, int j, int verbose_level);
static int wreath_product_group_element_linear_entry_frobenius(
		action &A,
	void *elt, int verbose_level);
static void wreath_product_group_element_one(
		action &A,
	void *elt, int verbose_level);
static int wreath_product_group_element_is_one(
		action &A,
	void *elt, int verbose_level);
static void wreath_product_group_element_unpack(
		action &A,
	void *elt, void *Elt, int verbose_level);
static void wreath_product_group_element_pack(
		action &A,
	void *Elt, void *elt, int verbose_level);
static void wreath_product_group_element_retrieve(
		action &A,
	int hdl, void *elt, int verbose_level);
static int wreath_product_group_element_store(
		action &A,
	void *elt, int verbose_level);
static void wreath_product_group_element_mult(
		action &A,
	void *a, void *b, void *ab, int verbose_level);
static void wreath_product_group_element_invert(
		action &A,
	void *a, void *av, int verbose_level);
static void wreath_product_group_element_transpose(
		action &A,
	void *a, void *at, int verbose_level);
static void wreath_product_group_element_move(
		action &A,
	void *a, void *b, int verbose_level);
static void wreath_product_group_element_dispose(
		action &A,
	int hdl, int verbose_level);
static void wreath_product_group_element_print(
		action &A,
	void *elt, std::ostream &ost);
static void wreath_product_group_element_code_for_make_element(
	action &A, void *elt, int *data);
static void wreath_product_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
static void wreath_product_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
static void wreath_product_group_element_print_quick(
		action &A,
	void *elt, std::ostream &ost);
static void wreath_product_group_element_print_latex(
		action &A,
	void *elt, std::ostream &ost);
static void wreath_product_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data);
static void wreath_product_group_element_print_verbose(
		action &A,
	void *elt, std::ostream &ost);
static void wreath_product_group_print_point(
		action &A,
	long int a, std::ostream &ost, int verbose_level);
static void wreath_product_group_unrank_point(
		action &A, long int rk, int *v, int verbose_level);
static long int wreath_product_group_rank_point(
		action &A, int *v, int verbose_level);


void action_pointer_table::init_function_pointers_wreath_product_group()
{
	label.assign("function_pointers_wreath_product_group");
	ptr_element_image_of = wreath_product_group_element_image_of;
	ptr_element_image_of_low_level =
			wreath_product_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = wreath_product_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius =
			wreath_product_group_element_linear_entry_frobenius;
	ptr_element_one = wreath_product_group_element_one;
	ptr_element_is_one = wreath_product_group_element_is_one;
	ptr_element_unpack = wreath_product_group_element_unpack;
	ptr_element_pack = wreath_product_group_element_pack;
	ptr_element_retrieve = wreath_product_group_element_retrieve;
	ptr_element_store = wreath_product_group_element_store;
	ptr_element_mult = wreath_product_group_element_mult;
	ptr_element_invert = wreath_product_group_element_invert;
	ptr_element_transpose = wreath_product_group_element_transpose;
	ptr_element_move = wreath_product_group_element_move;
	ptr_element_dispose = wreath_product_group_element_dispose;
	ptr_element_print = wreath_product_group_element_print;
	ptr_element_print_quick = wreath_product_group_element_print_quick;
	ptr_element_print_latex = wreath_product_group_element_print_latex;
	ptr_element_print_latex_with_print_point_function =
			wreath_product_group_element_print_latex_with_print_point_function;
	ptr_element_print_verbose = wreath_product_group_element_print_verbose;
	ptr_element_code_for_make_element =
			wreath_product_group_element_code_for_make_element;
	ptr_element_print_for_make_element =
			wreath_product_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			wreath_product_group_element_print_for_make_element_no_commas;
	ptr_print_point = wreath_product_group_print_point;
	ptr_unrank_point = wreath_product_group_unrank_point;
	ptr_rank_point = wreath_product_group_rank_point;
}



static long int wreath_product_group_element_image_of(
		action &A,
		long int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	long int b;

	if (f_v) {
		cout << "wreath_product_group_element_image_of "
				"computing image of " << a << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	b = W.element_image_of(Elt, a, verbose_level - 1);

	if (f_v) {
		cout << "wreath_product_group_element_image_of "
				"image of " << a << " is " << b << endl;
		}
	return b;
}

static void wreath_product_group_element_image_of_low_level(
		action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;


	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"computing image of ";
		Int_vec_print(cout, input, W.dimension_of_tensor_action);
		cout << endl;
		}
	W.element_image_of_low_level(Elt, input, output, verbose_level - 1);

	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"image of is ";
		Int_vec_print(cout, output, W.dimension_of_tensor_action);
		cout << endl;
		}
}

static int wreath_product_group_element_linear_entry_ij(
		action &A,
		void *elt, int i, int j, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;
	//int w;

	cout << "wreath_product_group_element_linear_entry_ij "
			"not yet implemented" << endl;
	exit(1);
}

static int wreath_product_group_element_linear_entry_frobenius(
		action &A,
		void *elt, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;
	//int w;

	cout << "wreath_product_group_element_linear_entry_frobenius "
			"not yet implemented" << endl;
	exit(1);
}

static void wreath_product_group_element_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "wreath_product_group_element_one "
				"calling element_one" << endl;
		}
	W.element_one(Elt);
}

static int wreath_product_group_element_is_one(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	int ret;

	if (f_v) {
		cout << "wreath_product_group_element_one calling "
				"element_is_one" << endl;
		}
	ret = W.element_is_one(Elt);
	if (f_v) {
		if (ret) {
			cout << "wreath_product_group_element_is_one "
					"returns YES" << endl;
			}
		else {
			cout << "wreath_product_group_element_is_one "
					"returns NO" << endl;
			}
		}
	return ret;
}

static void wreath_product_group_element_unpack(
		action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_unpack" << endl;
		}
	W.element_unpack(elt1, Elt1);
}

static void wreath_product_group_element_pack(
		action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_pack" << endl;
		}
	W.element_pack(Elt1, elt1);
}

static void wreath_product_group_element_retrieve(
		action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	uchar *p_elt;

	if (f_v) {
		cout << "wreath_product_group_element_"
				"retrieve hdl = " << hdl << endl;
		}
	p_elt = W.Elts->s_i(hdl);
	//if (f_v) {
	//	element_print_packed(G, p_elt, cout);
	//	}
	W.element_unpack(p_elt, Elt);
	if (f_v) {
		W.element_print_easy(Elt, cout);
		}
}

static int wreath_product_group_element_store(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	int hdl;

	if (f_v) {
		cout << "wreath_product_group_element_store" << endl;
		}
	W.element_pack(Elt, W.elt1);
	hdl = W.Elts->store(W.elt1);
	if (f_v) {
		cout << "wreath_product_group_element_store "
				"hdl = " << hdl << endl;
		}
	return hdl;
}

static void wreath_product_group_element_mult(
		action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "wreath_product_group_element_mult" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		W.element_print_easy(AA, cout);
		cout << "B=" << endl;
		W.element_print_easy(BB, cout);
		}
	W.element_mult(AA, BB, AB, verbose_level - 2);
	if (f_v) {
		cout << "wreath_product_group_element_mult done" << endl;
		}
	if (f_vv) {
		cout << "AB=" << endl;
		W.element_print_easy(AB, cout);
		}
}

static void wreath_product_group_element_invert(
		action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "wreath_product_group_element_invert" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		W.element_print_easy(AA, cout);
		}
	W.element_invert(AA, AAv, verbose_level - 1);
	if (f_v) {
		cout << "wreath_product_group_element_invert done" << endl;
		}
	if (f_vv) {
		cout << "Av=" << endl;
		W.element_print_easy(AAv, cout);
		}
}

static void wreath_product_group_element_transpose(
		action &A,
		void *a, void *at, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//wreath_product &W = *A.G.wreath_product_group;
	//int *AA = (int *) a;
	//int *Atv = (int *) at;

	cout << "wreath_product_group_element_transpose "
			"not yet implemented" << endl;
	exit(1);
}

static void wreath_product_group_element_move(
		action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "wreath_product_group_element_move" << endl;
		}
	W.element_move(AA, BB, 0 /* verbose_level */);
}

static void wreath_product_group_element_dispose(
		action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	group_constructions::wreath_product &W = *A.G.wreath_product_group;

	if (f_v) {
		cout << "wreath_product_group_element_dispose "
				"hdl = " << hdl << endl;
		}
	W.Elts->dispose(hdl);
}

static void wreath_product_group_element_print(
		action &A,
		void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;


	W.element_print_easy(Elt, ost);
	ost << endl;
}

static void wreath_product_group_element_code_for_make_element(
		action &A,
		void *elt, int *data)
{
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;

	cout << "wreath_product_group_element_code_for_make_element "
			"not yet implemented" << endl;
	exit(1);
#if 0
	//cout << "wreath_product_group_element_code_for_make_element "
	//"calling GL_print_for_make_element" << endl;
	W.element_code_for_make_element(Elt, data);
	//cout << "wreath_product_group_element_code_for_make_element "
	//"after GL_print_for_make_element" << endl;
#endif
}

static void wreath_product_group_element_print_for_make_element(
		action &A,
		void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	W.element_print_for_make_element(Elt, ost);
}

static void wreath_product_group_element_print_for_make_element_no_commas(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	W.element_print_for_make_element(Elt, ost);
}

static void wreath_product_group_element_print_quick(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;


	W.element_print_easy(Elt, ost);
}

static void wreath_product_group_element_print_latex(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	W.element_print_latex(Elt, ost);
}

static void wreath_product_group_element_print_latex_with_print_point_function(
	action &A,
	void *elt, std::ostream &ost,
	void (*point_label)(std::stringstream &sstr, long int pt, void *data),
	void *point_label_data)
{
	cout << "wreath_product_group_element_print_latex_with_print_point_function "
			"not yet implemented" << endl;
	exit(1);
}

static void wreath_product_group_element_print_verbose(
		action &A, void *elt, std::ostream &ost)
{
	group_constructions::wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	W.element_print_easy(Elt, ost);

}

static void wreath_product_group_print_point(
		action &A, long int a, std::ostream &ost, int verbose_level)
{
	//wreath_product &W = *A.G.wreath_product_group;

	cout << "wreath_product_group_print_point "
			"not yet implemented" << endl;
	exit(1);
}


static void wreath_product_group_unrank_point(
		action &A, long int rk, int *v, int verbose_level)
{
	action_global AG;
	//cout << "wreath_product_group_unrank_point" << endl;

	if (A.type_G == wreath_product_t) {
		group_constructions::wreath_product *W;
		W = A.G.wreath_product_group;

		W->unrank_point(rk, v, 0 /* verbose_level*/);


		}
	else {
		cout << "wreath_product_group_unrank_point type_G unknown:: type_G = ";
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
		}

}

static long int wreath_product_group_rank_point(
		action &A, int *v, int verbose_level)
{
	action_global AG;
	//cout << "wreath_product_group_rank_point" << endl;
	long int rk = -1;

	if (A.type_G == wreath_product_t) {
		group_constructions::wreath_product *W;
		W = A.G.wreath_product_group;

		rk = W->rank_point(v, 0 /* verbose_level */);
	}
	else {
		cout << "wreath_product_group_rank_point type_G unknown:: type_G = ";
		AG.action_print_symmetry_group_type(cout, A.type_G);
		cout << endl;
		exit(1);
		}

	return rk;
}



}}}


