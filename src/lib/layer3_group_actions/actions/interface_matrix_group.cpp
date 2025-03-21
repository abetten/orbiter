// interface_matrix_group.cpp
//
// Anton Betten
//
// started:  November 13, 2007
// last change:  November 9, 2010
// moved here from interface.cpp:  January 25, 2014




#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;



namespace orbiter {
namespace layer3_group_actions {
namespace actions {


// #############################################################################
// interface functions: matrix group
// #############################################################################


static long int matrix_group_element_image_of(
		action &A, long int a,
	void *elt, int verbose_level);
static void matrix_group_element_image_of_low_level(
		action &A,
	int *input, int *output, void *elt, int verbose_level);
static int matrix_group_element_linear_entry_ij(
		action &A,
	void *elt, int i, int j, int verbose_level);
static int matrix_group_element_linear_entry_frobenius(
		action &A,
	void *elt, int verbose_level);
static void matrix_group_element_one(
		action &A,
	void *elt, int verbose_level);
static int matrix_group_element_is_one(
		action &A,
	void *elt, int verbose_level);
static void matrix_group_element_unpack(
		action &A,
	void *elt, void *Elt, int verbose_level);
static void matrix_group_element_pack(
		action &A,
	void *Elt, void *elt, int verbose_level);
static void matrix_group_element_retrieve(
		action &A,
	int hdl, void *elt, int verbose_level);
static int matrix_group_element_store(
		action &A,
	void *elt, int verbose_level);
static void matrix_group_element_mult(
		action &A,
	void *a, void *b, void *ab, int verbose_level);
static void matrix_group_element_invert(
		action &A,
	void *a, void *av, int verbose_level);
static void matrix_group_element_transpose(
		action &A,
	void *a, void *at, int verbose_level);
static void matrix_group_element_move(
		action &A,
	void *a, void *b, int verbose_level);
static void matrix_group_element_dispose(
		action &A,
	int hdl, int verbose_level);
static void matrix_group_element_print(
		action &A,
	void *elt, std::ostream &ost);
static void matrix_group_element_code_for_make_element(
	action &A, void *elt, int *data);
#if 0
static void matrix_group_element_print_for_make_element(
	action &A, void *elt, std::ostream &ost);
static void matrix_group_element_print_for_make_element_no_commas(
	action &A, void *elt, std::ostream &ost);
#endif
static void matrix_group_element_print_quick(
		action &A,
	void *elt, std::ostream &ost);
static void matrix_group_element_print_latex(
		action &A,
	void *elt, std::ostream &ost);
static std::string matrix_group_element_stringify(
		action &A,
		void *elt, std::string &options);
static void matrix_group_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data);
static void matrix_group_element_print_as_permutation(
	action &A, void *elt, std::ostream &ost);
static void matrix_group_element_print_verbose(
		action &A,
	void *elt, std::ostream &ost);
static void matrix_group_print_point(
		action &A,
	long int a, std::ostream &ost, int verbose_level);
static void matrix_group_unrank_point(
		action &A, long int rk, int *v, int verbose_level);
static long int matrix_group_rank_point(
		action &A, int *v, int verbose_level);
static std::string matrix_group_stringify_point(
		action &A, long int rk, int verbose_level);


void action_pointer_table::init_function_pointers_matrix_group()
{
	label.assign("function_pointers_matrix_group");

	// the first 10:
	ptr_element_image_of = matrix_group_element_image_of;
	ptr_element_image_of_low_level = matrix_group_element_image_of_low_level;
	ptr_element_linear_entry_ij = matrix_group_element_linear_entry_ij;
	ptr_element_linear_entry_frobenius = matrix_group_element_linear_entry_frobenius;
	ptr_element_one = matrix_group_element_one;
	ptr_element_is_one = matrix_group_element_is_one;
	ptr_element_unpack = matrix_group_element_unpack;
	ptr_element_pack = matrix_group_element_pack;
	ptr_element_retrieve = matrix_group_element_retrieve;
	ptr_element_store = matrix_group_element_store;


	// the next 10:
	ptr_element_mult = matrix_group_element_mult;
	ptr_element_invert = matrix_group_element_invert;
	ptr_element_transpose = matrix_group_element_transpose;
	ptr_element_move = matrix_group_element_move;
	ptr_element_dispose = matrix_group_element_dispose;
	ptr_element_print = matrix_group_element_print;
	ptr_element_print_quick = matrix_group_element_print_quick;
	ptr_element_print_latex = matrix_group_element_print_latex;
	ptr_element_stringify = matrix_group_element_stringify;
	ptr_element_print_latex_with_point_labels =
			matrix_group_element_print_latex_with_point_labels;


	// the next 6:
	ptr_element_print_verbose = matrix_group_element_print_verbose;
	ptr_element_code_for_make_element =
			matrix_group_element_code_for_make_element;
#if 0
	ptr_element_print_for_make_element =
			matrix_group_element_print_for_make_element;
	ptr_element_print_for_make_element_no_commas =
			matrix_group_element_print_for_make_element_no_commas;
#endif
	ptr_print_point = matrix_group_print_point;
	ptr_unrank_point = matrix_group_unrank_point;
	ptr_rank_point = matrix_group_rank_point;
	ptr_stringify_point = matrix_group_stringify_point;
}



static long int matrix_group_element_image_of(
		action &A,
		long int a, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	long int b;
	
	if (f_v) {
		cout << "matrix_group_element_image_of "
				"computing image of " << a << endl;
		}
	b = G.Element->image_of_element(Elt, a, verbose_level - 1);

	if (f_v) {
		cout << "matrix_group_element_image_of "
				"image of " << a << " is " << b << endl;
		}
	return b;
}

static void matrix_group_element_image_of_low_level(
		action &A,
		int *input, int *output, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	if (f_v) {
		cout << "matrix_group_element_image_of_low_level "
				"computing image of ";
		Int_vec_print(cout, input, A.low_level_point_size);
		cout << " in action " << A.label << endl;
		}
	G.Element->action_from_the_right_all_types(input,
			Elt, output, verbose_level - 1);


	if (f_v) {
		cout << "matrix_group_element_image_of_low_level ";
		Int_vec_print(cout, input, A.low_level_point_size);
		cout << " -> ";
		Int_vec_print(cout, output, A.low_level_point_size);
		cout << endl;
		}
}

static int matrix_group_element_linear_entry_ij(
		action &A,
		void *elt, int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	int w;

	if (f_v) {
		cout << "matrix_group_element_linear_entry_ij "
				"i=" << i << " j=" << j << endl;
		}
	w = G.Element->GL_element_entry_ij(Elt, i, j);
	return w;
}

static int matrix_group_element_linear_entry_frobenius(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	int w;

	if (f_v) {
		cout << "matrix_group_element_linear_entry_frobenius" << endl;
		}
	w = G.Element->GL_element_entry_frobenius(Elt);
	return w;
}

static void matrix_group_element_one(
		action &A, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	
	if (f_v) {
		cout << "matrix_group_element_one calling GL_one" << endl;
		}
	G.Element->GL_one(Elt);
}

static int matrix_group_element_is_one(
		action &A, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	int f_is_one, i, j;
	
	if (f_v) {
		cout << "matrix_group_element_is_one" << endl;
	}
	if (G.f_kernel_is_diagonal_matrices) {
		f_is_one = G.Element->GL_is_one(Elt);
	}
	else if (!G.f_projective) {
		f_is_one = G.Element->GL_is_one(Elt);
	}
	else {
		cout << "matrix_group_element_is_one: warning: "
				"using slow identity element test" << endl;
		f_is_one = true;
		for (i = 0; i < A.degree; i++) {
			j = A.Group_element->element_image_of(i, elt, false);
			if (j != i) {
				f_is_one = false;
				break;
			}
		}
	}
	if (f_v) {
		if (f_is_one) {
			cout << "matrix_group_element_is_one "
					"returns YES" << endl;
		}
		else {
			cout << "matrix_group_element_is_one "
					"returns NO" << endl;
		}
	}
	return f_is_one;
}

static void matrix_group_element_unpack(
		action &A,
		void *elt, void *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;
	
	if (f_v) {
		cout << "matrix_group_element_unpack" << endl;
	}
	G.Element->GL_unpack(elt1, Elt1, verbose_level - 1);
}

static void matrix_group_element_pack(
		action &A,
		void *Elt, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt1 = (int *) Elt;
	uchar *elt1 = (uchar *)elt;
	
	if (f_v) {
		cout << "matrix_group_element_pack" << endl;
	}
	if (f_v) {
		cout << "matrix_group_element_pack before G.GL_pack" << endl;
	}
	G.Element->GL_pack(Elt1, elt1, verbose_level);
	if (f_v) {
		cout << "matrix_group_element_pack after G.GL_pack" << endl;
	}
}

static void matrix_group_element_retrieve(
		action &A,
		int hdl, void *elt, int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	//int *Elt = (int *) elt;
	//uchar *p_elt;


	G.Element->retrieve(
			hdl, elt, verbose_level);
	
}

static int matrix_group_element_store(
		action &A,
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	//int *Elt = (int *) elt;
	int hdl;
	
	if (f_v) {
		cout << "matrix_group_element_store" << endl;
	}
	hdl = G.Element->store(
			elt, verbose_level);
	return hdl;
}

static void matrix_group_element_mult(
		action &A,
		void *a, void *b, void *ab, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *AA = (int *) a;
	int *BB = (int *) b;
	int *AB = (int *) ab;

	if (f_v) {
		cout << "matrix_group_element_mult" << endl;
	}
	if (f_vv) {
		cout << "matrix_group_element_mult A=" << endl;
		G.Element->GL_print_easy(AA, cout);
		cout << "matrix_group_element_mult B=" << endl;
		G.Element->GL_print_easy(BB, cout);
	}
	if (f_v) {
		cout << "matrix_group_element_mult "
				"before G.Element->GL_mult" << endl;
	}
	G.Element->GL_mult(AA, BB, AB, verbose_level - 2);
	if (f_v) {
		cout << "matrix_group_element_mult "
				"after G.Element->GL_mult" << endl;
	}
	if (f_v) {
		cout << "matrix_group_element_mult done" << endl;
	}
	if (f_vv) {
		cout << "matrix_group_element_mult AB=" << endl;
		G.Element->GL_print_easy(AB, cout);
	}
}

static void matrix_group_element_invert(
		action &A,
		void *a, void *av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *AA = (int *) a;
	int *AAv = (int *) av;

	if (f_v) {
		cout << "matrix_group_element_invert" << endl;
	}
	if (f_vv) {
		cout << "A=" << endl;
		G.Element->GL_print_easy(AA, cout);
	}
	G.Element->GL_invert(AA, AAv);
	if (f_v) {
		cout << "matrix_group_element_invert done" << endl;
	}
	if (f_vv) {
		cout << "Av=" << endl;
		G.Element->GL_print_easy(AAv, cout);
	}
}

static void matrix_group_element_transpose(
		action &A,
		void *a, void *at, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *AA = (int *) a;
	int *Atv = (int *) at;

	if (f_v) {
		cout << "matrix_group_element_transpose" << endl;
	}
	if (f_vv) {
		cout << "A=" << endl;
		G.Element->GL_print_easy(AA, cout);
	}
	G.Element->GL_transpose(AA, Atv, verbose_level);
	if (f_v) {
		cout << "matrix_group_element_transpose done" << endl;
	}
	if (f_vv) {
		cout << "At=" << endl;
		G.Element->GL_print_easy(Atv, cout);
	}
}

static void matrix_group_element_move(
		action &A,
		void *a, void *b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *AA = (int *) a;
	int *BB = (int *) b;

	if (f_v) {
		cout << "matrix_group_element_move" << endl;
	}
	G.Element->GL_copy(AA, BB);
}

static void matrix_group_element_dispose(
		action &A,
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;

	if (f_v) {
		cout << "matrix_group_element_dispose "
				"hdl = " << hdl << endl;
	}
	G.Element->dispose(
			hdl, verbose_level);
}

static void matrix_group_element_print(
		action &A,
		void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	

	G.Element->GL_print_easy(Elt, ost);
	ost << endl;
	if (G.GFq->q > 2) {
		ost << "=" << endl;
		G.Element->GL_print_easy_normalized(Elt, ost);
		ost << endl;
	}

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
		//cout << "matrix_group_element_print:
		//printing element as permutation" << endl;
		matrix_group_element_print_as_permutation(A, elt, ost);
		ost << endl;
		}
#endif
}

static void matrix_group_element_code_for_make_element(
		action &A,
		void *elt, int *data)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	//cout << "matrix_group_element_code_for_make_element
	//calling GL_print_for_make_element" << endl;
	G.Element->GL_code_for_make_element(Elt, data);
	//cout << "matrix_group_element_code_for_make_element
	//after GL_print_for_make_element" << endl;
}

#if 0
static void matrix_group_element_print_for_make_element(
		action &A,
		void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	//cout << "matrix_group_element_print_for_make_element
	//calling GL_print_for_make_element" << endl;
	G.Element->GL_print_for_make_element(Elt, ost);
	//cout << "matrix_group_element_print_for_make_element
	//after GL_print_for_make_element" << endl;
}

static void matrix_group_element_print_for_make_element_no_commas(
		action &A, void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	//cout << "matrix_group_element_print_for_make_element_
	//no_commas calling GL_print_for_make_element_no_commas" << endl;
	G.Element->GL_print_for_make_element_no_commas(Elt, ost);
	//cout << "matrix_group_element_print_for_make_element_
	//no_commas after GL_print_for_make_element_no_commas" << endl;
}
#endif

static void matrix_group_element_print_quick(
		action &A,
		void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	//int *fp; //, n;
	

	G.Element->GL_print_easy(Elt, ost);


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
	//fp = NEW_int(A.degree);
	//n = A.find_fixed_points(elt, fp, 0);
	//cout << "with " << n << " fixed points" << endl;
	//FREE_int(fp);
	if (false /*A.degree < 0*/ /*1000*/) {
		//cout << "matrix_group_element_print:
		//printing element as permutation" << endl;
		matrix_group_element_print_as_permutation(A, elt, ost);
		ost << endl;
	}
}

static void matrix_group_element_print_latex(
		action &A,
		void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	G.Element->GL_print_latex(Elt, ost);
#if 0
	ost << "=" << endl;
	//G.GL_print_easy_normalized(Elt, ost);
	G.GL_print_easy_latex_with_option_numerical(Elt, true, ost);
#endif
}

static std::string matrix_group_element_stringify(
		action &A,
		void *elt, std::string &options)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	std::string s;

	s = G.Element->GL_stringify(Elt, options);
	return s;
}

static void matrix_group_element_print_latex_with_point_labels(
	action &A,
	void *elt, std::ostream &ost,
	std::string *Point_labels, void *data)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;

	G.Element->GL_print_latex(Elt, ost);
	//G.GL_print_easy_latex(Elt, ost);
}

static void matrix_group_element_print_as_permutation(
		action &A,
		void *elt, std::ostream &ost)
{
	//matrix_group &G = *A.G.matrix_grp;
	int f_v = false;
	int *Elt = (int *) elt;
	//int i, j;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "matrix_group_element_print_as_permutation "
				"degree = " << A.degree << endl;
	}
	int *images = NEW_int(A.degree);

	A.Group_element->make_list_of_images(
			images, Elt);
#if 0
	for (i = 0; i < A.degree; i++) {
		//cout << "matrix_group_element_print_as_permutation
		//computing image of i=" << i << endl;
		//if (i == 3)
			//f_v = true;
		//else
			//f_v = false;
		j = A.Group_element->element_image_of(i, Elt, 0 /* verbose_level */);
		images[i] = j;
	}
#endif

	Combi.Permutations->perm_print(ost, images, A.degree);
	FREE_int(images);
}

static void matrix_group_element_print_verbose(
		action &A,
		void *elt, std::ostream &ost)
{
	algebra::basic_algebra::matrix_group &G = *A.G.matrix_grp;
	int *Elt = (int *) elt;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	G.Element->GL_print_easy(Elt, ost);
	ost << "\n";
	//int i, j;
	
	if (A.degree < 100) {

		int *images = NEW_int(A.degree);


		A.Group_element->make_list_of_images(
				images, Elt);

#if 0
		for (i = 0; i < A.degree; i++) {
			j = A.Group_element->element_image_of(i, Elt, false);
			images[i] = j;
		}
#endif

		Combi.Permutations->perm_print(ost, images, A.degree);
		FREE_int(images);
	}
	else {
#if 0
		cout << "i : image" << endl;
		for (i = 0; i < MINIMUM(40, G.degree); i++) {
			j = A.element_image_of(i, Elt, false);
			cout << i << " : " << j << endl;
		}
#endif
	}

}

static void matrix_group_print_point(
		action &A, long int a, std::ostream &ost, int verbose_level)
{
	algebra::basic_algebra::matrix_group *G = A.G.matrix_grp;


	G->Element->print_point(
			a, ost, verbose_level);
}

static void matrix_group_unrank_point(
		action &A, long int rk, int *v, int verbose_level)
{
	algebra::basic_algebra::matrix_group *G = A.G.matrix_grp;

	G->Element->unrank_point(rk, v, verbose_level);

}

static long int matrix_group_rank_point(
		action &A, int *v, int verbose_level)
{
	algebra::basic_algebra::matrix_group *G = A.G.matrix_grp;
	long int rk;


	rk = G->Element->rank_point(v, verbose_level);

	return rk;
}

static std::string matrix_group_stringify_point(
		action &A, long int rk, int verbose_level)
{
	algebra::basic_algebra::matrix_group *G = A.G.matrix_grp;
	string s;


	s = G->Element->stringify_point(rk, verbose_level);

	return s;
}


}}}
