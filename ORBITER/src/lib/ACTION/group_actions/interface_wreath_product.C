// interface_wreath_product_group.C
//
// Anton Betten
//
// started:  August 4, 2018




#include "GALOIS/galois.h"
#include "action.h"

// #############################################################################
// interface functions: wreath product group
// #############################################################################




INT wreath_product_group_element_image_of(action &A,
		INT a, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;
	INT b;

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
		INT *input, INT *output, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;


	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"computing image of ";
		INT_vec_print(cout, input, W.dimension_of_tensor_action);
		cout << endl;
		}
	W.element_image_of_low_level(Elt, input, output, verbose_level - 1);

	if (f_v) {
		cout << "wreath_product_group_element_image_of_low_level "
				"image of is ";
		INT_vec_print(cout, output, W.dimension_of_tensor_action);
		cout << endl;
		}
}

INT wreath_product_group_element_linear_entry_ij(action &A,
		void *elt, INT i, INT j, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//INT *Elt = (INT *) elt;
	//INT w;

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

INT wreath_product_group_element_linear_entry_frobenius(action &A,
		void *elt, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//wreath_product &W = *A.G.wreath_product_group;
	//INT *Elt = (INT *) elt;
	//INT w;

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
		void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;

	if (f_v) {
		cout << "wreath_product_group_element_one "
				"calling element_one" << endl;
		}
	W.element_one(Elt);
}

INT wreath_product_group_element_is_one(action &A,
		void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;
	INT ret;

	if (f_v) {
		cout << "wreath_product_group_element_one calling "
				"element_is_one" << endl;
		}
	ret = W.element_is_one(Elt);
	if (f_v) {
		if (ret) {
			cout << "wreath_product_group_element_is_one returns YES" << endl;
			}
		else {
			cout << "wreath_product_group_element_is_one returns NO" << endl;
			}
		}
	return ret;
#if 0
	if (f_v) {
		cout << "wreath_product_group_element_is_one" << endl;
		}
	if (G.f_kernel_is_diagonal_matrices) {
		f_is_one = G.GL_is_one(Elt);
		}
	else if (!G.f_projective) {
		f_is_one = G.GL_is_one(Elt);
		}
	else {
		/* if (A.ptr_element_image_of == element_image_of_line_through_vertex ||
		A.ptr_element_image_of == element_image_of_plane_not_through_vertex_in_contragredient_action ||
		A.ptr_element_image_of == element_image_under_wedge_action_from_the_right)*/
		cout << "matrix_group_element_is_one: warning: using slow identity element test" << endl;
		f_is_one = TRUE;
		for (i = 0; i < A.degree; i++) {
			j = A.element_image_of(i, elt, FALSE);
			if (j != i) {
				f_is_one = FALSE;
				break;
				}
			}
		}
	/*else {
		cout << "element_is_one() unrecognized ptr_element_image_of" << endl;
		exit(1);
		}*/
#endif
}

void wreath_product_group_element_unpack(action &A,
		void *elt, void *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt1 = (INT *) Elt;
	UBYTE *elt1 = (UBYTE *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_unpack" << endl;
		}
	W.element_unpack(elt1, Elt1);
}

void wreath_product_group_element_pack(action &A,
		void *Elt, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt1 = (INT *) Elt;
	UBYTE *elt1 = (UBYTE *)elt;

	if (f_v) {
		cout << "wreath_product_group_element_pack" << endl;
		}
	W.element_pack(Elt1, elt1);
}

void wreath_product_group_element_retrieve(action &A,
		INT hdl, void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;
	UBYTE *p_elt;

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

INT wreath_product_group_element_store(action &A,
		void *elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;
	INT hdl;

	if (f_v) {
		cout << "wreath_product_group_element_store" << endl;
		}
	W.element_pack(Elt, W.elt1);
	hdl = W.Elts->store(W.elt1);
	if (f_v) {
		cout << "wreath_product_group_element_store hdl = " << hdl << endl;
		}
	return hdl;
}

void wreath_product_group_element_mult(action &A,
		void *a, void *b, void *ab, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	wreath_product &W = *A.G.wreath_product_group;
	INT *AA = (INT *) a;
	INT *BB = (INT *) b;
	INT *AB = (INT *) ab;

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
		void *a, void *av, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	wreath_product &W = *A.G.wreath_product_group;
	INT *AA = (INT *) a;
	INT *AAv = (INT *) av;

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
		void *a, void *at, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//wreath_product &W = *A.G.wreath_product_group;
	//INT *AA = (INT *) a;
	//INT *Atv = (INT *) at;

	cout << "wreath_product_group_element_transpose "
			"not yet implemented" << endl;
	exit(1);
#if 0
	if (f_v) {
		cout << "wreath_product_group_element_transpose" << endl;
		}
	if (f_vv) {
		cout << "A=" << endl;
		G.GL_print_easy(AA, cout);
		}
	G.GL_transpose(AA, Atv, verbose_level);
	if (f_v) {
		cout << "wreath_product_group_element_transpose done" << endl;
		}
	if (f_vv) {
		cout << "At=" << endl;
		W.element_print_easy(Atv, cout);
		}
#endif
}

void wreath_product_group_element_move(action &A,
		void *a, void *b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;
	INT *AA = (INT *) a;
	INT *BB = (INT *) b;

	if (f_v) {
		cout << "wreath_product_group_element_move" << endl;
		}
	W.element_move(AA, BB, 0 /* verbose_level */);
}

void wreath_product_group_element_dispose(action &A,
		INT hdl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	wreath_product &W = *A.G.wreath_product_group;

	if (f_v) {
		cout << "wreath_product_group_element_dispose() hdl = " << hdl << endl;
		}
	W.Elts->dispose(hdl);
}

void wreath_product_group_element_print(action &A,
		void *elt, ostream &ost)
{
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;


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
	INT *fp, n;

	fp = NEW_INT(A.degree);
	n = A.find_fixed_points(elt, fp, 0);
	cout << "with " << n << " fixed points ";
	A.element_print_base_images(Elt, ost);
	cout << endl;
	FREE_INT(fp);
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
		void *elt, INT *data)
{
	//wreath_product &W = *A.G.wreath_product_group;
	//INT *Elt = (INT *) elt;

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
	//INT *Elt = (INT *) elt;

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
	//INT *Elt = (INT *) elt;

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
	INT *Elt = (INT *) elt;
	//INT *fp; //, n;


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
	//fp = NEW_INT(A.degree);
	//n = A.find_fixed_points(elt, fp, 0);
	//cout << "with " << n << " fixed points" << endl;
	//FREE_INT(fp);
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
	//INT *Elt = (INT *) elt;

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
	INT f_v = FALSE;
	INT *Elt = (INT *) elt;
	INT i, j;

	if (f_v) {
		cout << "wreath_product_group_element_print_as_permutation "
				"degree = " << A.degree << endl;
		}
	INT *p = NEW_INT(A.degree);
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
	FREE_INT(p);
}

void wreath_product_group_element_print_verbose(
		action &A, void *elt, ostream &ost)
{
	wreath_product &W = *A.G.wreath_product_group;
	INT *Elt = (INT *) elt;

	W.element_print_easy(Elt, ost);
#if 0
	ost << "\n";
	INT i, j;

	if (A.degree < 1000) {
		INT *p = NEW_INT(A.degree);
		for (i = 0; i < A.degree; i++) {
			j = A.element_image_of(i, Elt, FALSE);
			p[i] = j;
			}
		perm_print(ost, p, A.degree);
		FREE_INT(p);
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

void wreath_product_group_print_point(action &A, INT a, ostream &ost)
{
	//wreath_product &W = *A.G.wreath_product_group;

	cout << "wreath_product_group_print_point "
			"not yet implemented" << endl;
	exit(1);
}


