// interface_wreath_product_group.C
//
// Anton Betten
//
// started:  August 4, 2018




#include "foundations/foundations.h"
#include "group_actions.h"


namespace orbiter {
namespace group_actions {

// #############################################################################
// interface functions: wreath product group
// #############################################################################




int wreath_product_group_element_image_of(action &A,
		int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	int b;

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

void wreath_product_group_element_image_of_low_level(action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;


	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"computing image of ";
		int_vec_print(cout, input, W.dimension_of_tensor_action);
		cout << endl;
		}
	W.element_image_of_low_level(Elt, input, output, verbose_level - 1);

	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"image of is ";
		int_vec_print(cout, output, W.dimension_of_tensor_action);
		cout << endl;
		}
}

int wreath_product_group_element_linear_entry_ij(action &A,
		void *elt, int i, int j, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;
	//int w;

	cout << "wreath_product_group_element_linear_entry_ij "
			"not yet implemented" << endl;
	exit(1);
#if 0
	if (f_v) {
		cout << "wreath_product_group_element_linear_entry_ij "
				"i=" << i << " j=" << j << endl;
		}
	w = W.element_entry_ij(Elt, i, j);
	return w;
#endif
}

int wreath_product_group_element_linear_entry_frobenius(action &A,
		void *elt, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;
	//int w;

	cout << "wreath_product_group_element_linear_entry_frobenius "
			"not yet implemented" << endl;
	exit(1);
#if 0
	if (f_v) {
		cout << "wreath_product_group_element_linear_entry_frobenius" << endl;
		}
	w = W.element_entry_frobenius(Elt);
	return w;
#endif
}

void wreath_product_group_element_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "wreath_product_group_element_one "
				"calling element_one" << endl;
		}
	W.element_one(Elt);
}

int wreath_product_group_element_is_one(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
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

void wreath_product_group_element_unpack(action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_unpack" << endl;
		}
	W.element_unpack(elt1, Elt1);
}

void wreath_product_group_element_pack(action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_pack" << endl;
		}
	W.element_pack(Elt1, elt1);
}

void wreath_product_group_element_retrieve(action &A,
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
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

int wreath_product_group_element_store(action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
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

void wreath_product_group_element_mult(action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	wreath_product &W = *A.G.wreath_product_group;
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

void wreath_product_group_element_invert(action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	wreath_product &W = *A.G.wreath_product_group;
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

void wreath_product_group_element_transpose(action &A,
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

void wreath_product_group_element_move(action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "wreath_product_group_element_move" << endl;
		}
	W.element_move(AA, BB, 0 /* verbose_level */);
}

void wreath_product_group_element_dispose(action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;

	if (f_v) {
		cout << "wreath_product_group_element_dispose "
				"hdl = " << hdl << endl;
		}
	W.Elts->dispose(hdl);
}

void wreath_product_group_element_print(action &A,
		void *elt, ostream &ost)
{
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;


	W.element_print_easy(Elt, ost);
	ost << endl;
#if 0
	if (G.GFq->q > 2) {
		ost << "=" << endl;
		W.element_print_easy_normalized(Elt, ost);
		ost << endl;
		}
#endif

#if 0
	int *fp, n;

	fp = NEW_int(A.degree);
	n = A.find_fixed_points(elt, fp, 0);
	cout << "with " << n << " fixed points ";
	A.element_print_base_images(Elt, ost);
	cout << endl;
	FREE_int(fp);
#endif

#if 0
	if (A.degree < 0 /*1000*/) {
		//cout << "matrix_group_element_print: "
		//"printing element as permutation" << endl;
		matrix_group_element_print_as_permutation(A, elt, ost);
		ost << endl;
		}
#endif
}

void wreath_product_group_element_code_for_make_element(action &A,
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

void wreath_product_group_element_print_for_make_element(action &A,
		void *elt, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;

	cout << "wreath_product_group_element_print_for_make_element "
			"not yet implemented" << endl;
	exit(1);
#if 0
	//cout << "wreath_product_group_element_print_for_make_element "
	//"calling GL_print_for_make_element" << endl;
	W.element_print_for_make_element(Elt, ost);
	//cout << "wreath_product_group_element_print_for_make_element "
	//"after GL_print_for_make_element" << endl;
#endif
}

void wreath_product_group_element_print_for_make_element_no_commas(
		action &A, void *elt, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;

	cout << "wreath_product_group_element_print_for_make_element_no_commas "
			"not yet implemented" << endl;
	exit(1);
#if 0
	//cout << "wreath_product_group_element_print_for_make_element_no_commas "
	//"calling GL_print_for_make_element_no_commas" << endl;
	W.element_print_for_make_element_no_commas(Elt, ost);
	//cout << "wreath_product_group_element_print_for_make_element_no_commas "
	//"after GL_print_for_make_element_no_commas" << endl;
#endif
}

void wreath_product_group_element_print_quick(
		action &A, void *elt, ostream &ost)
{
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;
	//int *fp; //, n;


	W.element_print_easy(Elt, ost);


#if 0
	ost << endl;
	ost << "=" << endl;
	G.GL_print_easy_normalized(Elt, ost);
	ost << endl;
#endif

#if 0
	A.element_print_base_images_verbose(Elt, ost, 0);
	ost << endl;
#endif

#if 0
	//fp = NEW_int(A.degree);
	//n = A.find_fixed_points(elt, fp, 0);
	//cout << "with " << n << " fixed points" << endl;
	//FREE_int(fp);
	if (FALSE /*A.degree < 0*/ /*1000*/) {
		//cout << "matrix_group_element_print: "
		//"printing element as permutation" << endl;
		matrix_group_element_print_as_permutation(A, elt, ost);
		ost << endl;
		}
#endif
}

void wreath_product_group_element_print_latex(
		action &A, void *elt, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;
	//int *Elt = (int *) elt;

	cout << "wreath_product_group_element_print_latex "
			"not yet implemented" << endl;
	exit(1);
#if 0
	W.element_print_latex(Elt, ost);
	//W.element_print_easy_latex(Elt, ost);
#endif
}

void wreath_product_group_element_print_as_permutation(
		action &A, void *elt, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;
	int f_v = FALSE;
	int *Elt = (int *) elt;
	int i, j;

	if (f_v) {
		cout << "wreath_product_group_element_print_as_permutation "
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
	perm_print(ost, p, A.degree);
	FREE_int(p);
}

void wreath_product_group_element_print_verbose(
		action &A, void *elt, ostream &ost)
{
	wreath_product &W = *A.G.wreath_product_group;
	int *Elt = (int *) elt;

	W.element_print_easy(Elt, ost);
#if 0
	ost << "\n";
	int i, j;

	if (A.degree < 1000) {
		int *p = NEW_int(A.degree);
		for (i = 0; i < A.degree; i++) {
			j = A.element_image_of(i, Elt, FALSE);
			p[i] = j;
			}
		perm_print(ost, p, A.degree);
		FREE_int(p);
		}
	else {
#if 0
		cout << "i : image" << endl;
		for (i = 0; i < MINIMUM(40, G.degree); i++) {
			j = A.element_image_of(i, Elt, FALSE);
			cout << i << " : " << j << endl;
			}
#endif
		}
#endif

}

void wreath_product_group_print_point(action &A, int a, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;

	cout << "wreath_product_group_print_point "
			"not yet implemented" << endl;
	exit(1);
}

}}


