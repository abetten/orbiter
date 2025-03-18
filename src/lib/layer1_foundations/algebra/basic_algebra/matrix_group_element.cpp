/*
 * matrix_group_element.cpp
 *
 *  Created on: Jan 20, 2024
 *      Author: betten
 */






#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {


matrix_group_element::matrix_group_element()
{
	Record_birth();
	Matrix_group = NULL;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	Elt5 = NULL;
	tmp_M = NULL;
	base_cols = NULL;
	v1 = NULL;
	v2 = NULL;
	v3 = NULL;
	elt1 = NULL;
	elt2 = NULL;
	elt3 = NULL;

	Page_storage = NULL;
}


matrix_group_element::~matrix_group_element()
{
	Record_death();
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::~matrix_group_element "
				"calling free_data" << endl;
	}
	free_data(verbose_level);
	if (f_v) {
		cout << "matrix_group_element::~matrix_group_element "
				"destroying Elts" << endl;
	}
	if (Page_storage) {
		FREE_OBJECT(Page_storage);
	}
}

void matrix_group_element::init(
		matrix_group *Matrix_group,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::init" << endl;
	}
	matrix_group_element::Matrix_group = Matrix_group;

	int page_length_log = PAGE_LENGTH_LOG;

	if (f_v) {
		cout << "matrix_group_element::init "
				"before allocate_data" << endl;
	}
	allocate_data(0/*verbose_level*/);
	if (f_v) {
		cout << "matrix_group_element::init "
				"after allocate_data" << endl;
	}

	if (f_v) {
		cout << "matrix_group_element::init "
				"before setup_page_storage" << endl;
	}
	setup_page_storage(page_length_log, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "matrix_group_element::init "
				"after setup_page_storage" << endl;
	}
	if (f_v) {
		cout << "matrix_group_element::init done" << endl;
	}

}

void matrix_group_element::allocate_data(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::allocate_data" << endl;
	}
	if (Matrix_group->elt_size_int == 0) {
		cout << "matrix_group_element::allocate_data "
				"elt_size_int == 0" << endl;
		exit(1);
	}

	Elt1 = NEW_int(Matrix_group->elt_size_int);
	Elt2 = NEW_int(Matrix_group->elt_size_int);
	Elt3 = NEW_int(Matrix_group->elt_size_int);
	Elt4 = NEW_int(Matrix_group->elt_size_int);
	Elt5 = NEW_int(Matrix_group->elt_size_int);
	tmp_M = NEW_int(Matrix_group->n * Matrix_group->n);
	v1 = NEW_int(2 * Matrix_group->n);
	v2 = NEW_int(2 * Matrix_group->n);
	v3 = NEW_int(2 * Matrix_group->n);
	elt1 = NEW_uchar(Matrix_group->char_per_elt);
	elt2 = NEW_uchar(Matrix_group->char_per_elt);
	elt3 = NEW_uchar(Matrix_group->char_per_elt);
	base_cols = NEW_int(Matrix_group->n);

	if (f_v) {
		cout << "matrix_group_element::allocate_data done" << endl;
	}
}

void matrix_group_element::free_data(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::free_data" << endl;
	}
	if (Elt1) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing Elt1" << endl;
		}
		FREE_int(Elt1);
	}
	if (Elt2) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing Elt2" << endl;
		}
		FREE_int(Elt2);
	}
	if (Elt3) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing Elt3" << endl;
		}
		FREE_int(Elt3);
	}
	if (Elt4) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing Elt4" << endl;
		}
		FREE_int(Elt4);
	}
	if (Elt5) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing Elt5" << endl;
		}
		FREE_int(Elt5);
	}
	if (tmp_M) {
		if (f_v) {
			cout << "matrix_group_element::free_data freeing tmp_M" << endl;
		}
		FREE_int(tmp_M);
	}
	if (f_v) {
		cout << "matrix_group_element::free_data destroying v1-3" << endl;
	}
	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
	if (v3) {
		FREE_int(v3);
	}
	if (f_v) {
		cout << "matrix_group_element::free_data "
				"destroying elt1-3" << endl;
	}
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (elt2) {
		FREE_uchar(elt2);
	}
	if (elt3) {
		FREE_uchar(elt3);
	}
	if (f_v) {
		cout << "matrix_group_element::free_data "
				"destroying base_cols" << endl;
	}
	if (base_cols) {
		FREE_int(base_cols);
	}

	if (f_v) {
		cout << "matrix_group_element::free_data done" << endl;
	}
}

void matrix_group_element::setup_page_storage(
		int page_length_log, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int hdl;

	if (f_v) {
		cout << "matrix_group_element::setup_page_storage" << endl;
	}
	if (Page_storage) {
		cout << "matrix_group_element::setup_page_storage "
				"Warning: Page_storage != NULL" << endl;
		FREE_OBJECT(Page_storage);
	}
	Page_storage = NEW_OBJECT(other::data_structures::page_storage);

	if (f_vv) {
		cout << "matrix_group_element::setup_page_storage "
				"calling Page_storage->init" << endl;
	}
	Page_storage->init(
			Matrix_group->char_per_elt /* entry_size */,
			page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(elt_print, (void *) this);


	if (f_vv) {
		cout << "matrix_group_element::setup_page_storage "
				"before GL_one" << endl;
	}
	GL_one(Elt1);
	GL_pack(Elt1, elt1, verbose_level);
	if (f_vv) {
		cout << "matrix_group_element::setup_page_storage "
				"before Page_storage->store" << endl;
	}
	hdl = Page_storage->store(elt1);
	if (f_vv) {
		cout << "matrix_group_element::setup_page_storage "
				"stored identity element, hdl = " << hdl << endl;
	}
	if (f_v) {
		cout << "matrix_group_element::setup_page_storage done" << endl;
	}
}

int matrix_group_element::GL_element_entry_ij(
		int *Elt, int i, int j)
{
	return Elt[i * Matrix_group->n + j];
}

int matrix_group_element::GL_element_entry_frobenius(
		int *Elt)
{
	if (!Matrix_group->f_semilinear) {
		cout << "matrix_group_element::GL_element_entry_frobenius "
				"fatal: !Matrix_group->f_semilinear" << endl;
		exit(1);
	}
	return Elt[Matrix_group->offset_frobenius];
#if 0
	if (Matrix_group->f_projective) {
		return Elt[Matrix_group->n * Matrix_group->n];
	}
	else if (Matrix_group->f_affine) {
		return Elt[Matrix_group->n * Matrix_group->n + Matrix_group->n];
	}
	else if (Matrix_group->f_general_linear) {
		return Elt[Matrix_group->n * Matrix_group->n];
	}
	else {
		cout << "matrix_group_element::GL_element_entry_frobenius "
				"unknown group type" << endl;
		exit(1);
	}
#endif
}

long int matrix_group_element::image_of_element(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b;

	if (f_v) {
		cout << "matrix_group_element::image_of_element" << endl;
	}
	if (Matrix_group->f_projective) {
		b = GL_image_of_PG_element(
				Elt, a, verbose_level - 1);
	}
	else if (Matrix_group->f_affine) {
		b = GL_image_of_AG_element(
				Elt, a, verbose_level - 1);
	}
	else if (Matrix_group->f_general_linear) {
		b = GL_image_of_AG_element(
				Elt, a, verbose_level - 1);
	}
	else {
		cout << "matrix_group_element::image_of_element "
				"unknown group type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group_element::image_of_element " << a
				<< " maps to " << b << endl;
	}
	return b;
}


long int matrix_group_element::GL_image_of_PG_element(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b;

	if (f_v) {
		cout << "matrix_group_element::GL_image_of_PG_element" << endl;
	}
	Matrix_group->GFq->Projective_space_basic->PG_element_unrank_modified_lint(
			v1, 1, Matrix_group->n, a);

	action_from_the_right_all_types(
			v1, Elt, v2, verbose_level - 1);

	Matrix_group->GFq->Projective_space_basic->PG_element_rank_modified_lint(
			v2, 1, Matrix_group->n, b);

	if (f_v) {
		cout << "matrix_group_element::GL_image_of_PG_element done" << endl;
	}
	return b;
}

long int matrix_group_element::GL_image_of_AG_element(
		int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int b;
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "matrix_group_element::GL_image_of_AG_element" << endl;
	}

	Gg.AG_element_unrank(
			Matrix_group->GFq->q, v1, 1, Matrix_group->n, a);

	action_from_the_right_all_types(
			v1, Elt, v2, verbose_level - 1);

	b = Gg.AG_element_rank(
			Matrix_group->GFq->q, v2, 1, Matrix_group->n);

	if (f_v) {
		cout << "matrix_group_element::GL_image_of_AG_element done" << endl;
	}
	return b;
}

void matrix_group_element::action_from_the_right_all_types(
		int *v, int *A, int *vA, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::action_from_the_right_all_types" << endl;
	}
	if (Matrix_group->f_projective) {
		projective_action_from_the_right(
				v, A, vA,
				verbose_level - 1);
	}
	else if (Matrix_group->f_affine) {
		Matrix_group->GFq->Linear_algebra->affine_action_from_the_right(
				Matrix_group->f_semilinear,
				v, A, vA, Matrix_group->n);
			// vA = (v * A)^{p^f} + b
			// where b = A + n * n
			// and f = A[n * n + n] if f_semilinear is true
	}
	else if (Matrix_group->f_general_linear) {
		general_linear_action_from_the_right(
				v, A, vA,
				verbose_level - 1);
	}
	else {
		cout << "matrix_group_element::action_from_the_right_all_types "
				"unknown group type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group_element::action_from_the_right_all_types done" << endl;
	}
}

void matrix_group_element::projective_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level)
// vA = (v * A)^{p^f} if f_semilinear,
// vA = v * A otherwise
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::projective_action_from_the_right"  << endl;
	}
	Matrix_group->GFq->Linear_algebra->projective_action_from_the_right(
			Matrix_group->f_semilinear,
			v, A, vA, Matrix_group->n,
			verbose_level - 1);
	// vA = (v * A)^{p^f}  if f_semilinear
	// (where f = A[n * n]),
	// vA = v * A otherwise
	if (f_v) {
		cout << "matrix_group_element::projective_action_from_the_right done"  << endl;
	}
}

void matrix_group_element::general_linear_action_from_the_right(
		int *v, int *A, int *vA, int verbose_level)
// vA = (v * A)^{p^f} if f_semilinear,
// vA = v * A otherwise
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::general_linear_action_from_the_right"  << endl;
	}
	Matrix_group->GFq->Linear_algebra->general_linear_action_from_the_right(
			Matrix_group->f_semilinear,
			v, A, vA, Matrix_group->n,
			verbose_level - 1);
	if (f_v) {
		cout << "matrix_group_element::general_linear_action_from_the_right done"  << endl;
	}
}

void matrix_group_element::substitute_surface_equation(
		int *Elt,
		int *coeff_in, int *coeff_out,
		geometry::algebraic_geometry::surface_domain *Surf,
		int verbose_level)
// used in arc_lifting.cpp, surface_classify_wedge.cpp,
// surface_create.cpp, create_surface_main.cpp, intersection.cpp
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::substitute_surface_equation" << endl;
	}
	if (Matrix_group->f_semilinear) {
		number_theory::number_theory_domain NT;
		int me;

		me = NT.int_negate(Elt[Matrix_group->offset_frobenius], Matrix_group->GFq->e);
		// (GFq->e - Elt[n * n]) % GFq->e
		Surf->substitute_semilinear(
					coeff_in,
					coeff_out,
					true /* f_semilinear */,
					me,
					Elt,
					0 /*verbose_level*/);

		Matrix_group->GFq->Projective_space_basic->PG_element_normalize(
				coeff_out, 1, 20);
	}
	else {
		Surf->substitute_semilinear(
					coeff_in,
					coeff_out,
					false /* f_semilinear */,
					0,
					Elt,
					0 /*verbose_level*/);

		Matrix_group->GFq->Projective_space_basic->PG_element_normalize(
				coeff_out, 1, 20);

	}
	if (f_v) {
		cout << "matrix_group_element::substitute_surface_equation done" << endl;
	}
}

void matrix_group_element::GL_one(
		int *Elt)
{
	GL_one_internal(Elt);
	GL_one_internal(Elt + Matrix_group->elt_size_int_half);
}

void matrix_group_element::GL_one_internal(
		int *Elt)
{

	if (Matrix_group->f_projective) {

		Matrix_group->GFq->Linear_algebra->identity_matrix(
				Elt, Matrix_group->n);

	}
	else if (Matrix_group->f_affine) {

		Matrix_group->GFq->Linear_algebra->identity_matrix(
				Elt, Matrix_group->n);

		Int_vec_zero(Elt + Matrix_group->offset_affine_vector, Matrix_group->n);

	}
	else {

		Matrix_group->GFq->Linear_algebra->identity_matrix(
				Elt, Matrix_group->n);

	}

	if (Matrix_group->f_semilinear) {
		Elt[Matrix_group->offset_frobenius] = 0;
	}

}

void matrix_group_element::GL_zero(
		int *Elt)
{
	if (Matrix_group->f_projective) {
		if (Matrix_group->f_semilinear) {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n + 1);
		}
		else {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n);
		}
	}
	else if (Matrix_group->f_affine) {
		if (Matrix_group->f_semilinear) {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n + Matrix_group->n + 1);
		}
		else {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n + Matrix_group->n);
		}
	}
	if (Matrix_group->f_general_linear) {
		if (Matrix_group->f_semilinear) {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n + 1);
		}
		else {
			Int_vec_zero(Elt, Matrix_group->n * Matrix_group->n);
		}
	}
	else {
		cout << "matrix_group_element::GL_zero unknown group type" << endl;
		exit(1);
	}
	GL_copy_internal(Elt, Elt + Matrix_group->elt_size_int_half);
}

int matrix_group_element::GL_is_one(
		int *Elt)
{
	int c;

	//cout << "matrix_group_element::GL_is_one" << endl;
	if (Matrix_group->f_projective) {
		if (!Matrix_group->GFq->Linear_algebra->is_scalar_multiple_of_identity_matrix(
				Elt, Matrix_group->n, c)) {
			return false;
		}
		if (Matrix_group->f_semilinear) {
			if (Elt[Matrix_group->offset_frobenius] != 0) {
				return false;
			}
		}
	}
	else if (Matrix_group->f_affine) {
		//cout << "matrix_group_element::GL_is_one f_affine" << endl;
		if (!Matrix_group->GFq->Linear_algebra->is_identity_matrix(
				Elt, Matrix_group->n)) {
			//cout << "matrix_group_element::GL_is_one
			// not the identity matrix" << endl;
			//print_integer_matrix(cout, Elt, n, n);
			return false;
		}
		if (!Matrix_group->GFq->Linear_algebra->is_zero_vector(
				Elt + Matrix_group->offset_affine_vector, Matrix_group->n)) {
			//cout << "matrix_group_element::GL_is_one
			// not the zero vector" << endl;
			return false;
		}
		if (Matrix_group->f_semilinear) {
			if (Elt[Matrix_group->offset_frobenius] != 0) {
				return false;
			}
		}
	}
	else if (Matrix_group->f_general_linear) {
		//cout << "matrix_group_element::GL_is_one f_general_linear" << endl;
		if (!Matrix_group->GFq->Linear_algebra->is_identity_matrix(
				Elt, Matrix_group->n)) {
			//cout << "matrix_group_element::GL_is_one
			// not the identity matrix" << endl;
			//print_integer_matrix(cout, Elt, n, n);
			return false;
		}
		if (Matrix_group->f_semilinear) {
			if (Elt[Matrix_group->offset_frobenius] != 0) {
				return false;
			}
		}
	}
	else {
		cout << "matrix_group_element::GL_is_one unknown group type" << endl;
		exit(1);
	}
	return true;
}

void matrix_group_element::GL_mult(
		int *A, int *B, int *AB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_mult" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "matrix_group_element::GL_mult A=" << endl;
		GL_print_easy(A, cout);
		cout << "matrix_group_element::GL_mult A+=" << endl;
		GL_print_easy(A + Matrix_group->elt_size_int_half, cout);
		cout << "matrix_group_element::GL_mult B=" << endl;
		GL_print_easy(B, cout);
		cout << "matrix_group_element::GL_mult B+=" << endl;
		GL_print_easy(B + Matrix_group->elt_size_int_half, cout);
	}

	if (f_v) {
		cout << "matrix_group_element::GL_mult_verbose "
				"before GL_mult_internal (1)" << endl;
	}

	GL_mult_internal(A, B, AB, verbose_level - 1);

	if (f_v) {
		cout << "matrix_group_element::GL_mult_verbose "
				"after GL_mult_internal (1)" << endl;
	}

	if (f_v) {
		cout << "matrix_group_element::GL_mult_verbose "
				"before GL_mult_internal (2)" << endl;
	}

	GL_mult_internal(
			B + Matrix_group->elt_size_int_half,
			A + Matrix_group->elt_size_int_half,
			AB + Matrix_group->elt_size_int_half,
			verbose_level - 1);

	if (f_v) {
		cout << "matrix_group_element::GL_mult_verbose "
				"after GL_mult_internal (2)" << endl;
	}

	if (f_v) {
		cout << "matrix_group_element::GL_mult_verbose AB=" << endl;
		GL_print_easy(AB, cout);
		cout << "matrix_group_element::GL_mult done" << endl;
	}
}

void matrix_group_element::GL_mult_internal(
		int *A, int *B, int *AB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_mult_internal" << endl;
		cout << "f_projective=" << Matrix_group->f_projective << endl;
		cout << "f_affine=" << Matrix_group->f_affine << endl;
		cout << "f_general_linear=" << Matrix_group->f_general_linear << endl;
		cout << "f_semilinear=" << Matrix_group->f_semilinear << endl;
	}

	if (Matrix_group->f_projective) {
		if (Matrix_group->f_semilinear) {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->semilinear_matrix_mult" << endl;
			}
			//GFq->semilinear_matrix_mult(A, B, AB, n);
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_mult_memory_given(
					A, B, AB, tmp_M, Matrix_group->n,
					verbose_level - 1);
		}
		else {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->mult_matrix_matrix" << endl;
			}
			Matrix_group->GFq->Linear_algebra->mult_matrix_matrix(
					A, B, AB, Matrix_group->n, Matrix_group->n, Matrix_group->n,
					0 /* verbose_level */);
		}
	}
	else if (Matrix_group->f_affine) {
		if (Matrix_group->f_semilinear) {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->semilinear_matrix_mult_affine" << endl;
			}
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_mult_affine(
					A, B, AB, Matrix_group->n);
		}
		else {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->matrix_mult_affine" << endl;
			}
			Matrix_group->GFq->Linear_algebra->matrix_mult_affine(
					A, B, AB, Matrix_group->n,
					verbose_level - 1);
		}
	}
	else if (Matrix_group->f_general_linear) {
		if (Matrix_group->f_semilinear) {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->semilinear_matrix_mult" << endl;
			}
			//GFq->semilinear_matrix_mult(A, B, AB, n);
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_mult_memory_given(
					A, B, AB, tmp_M, Matrix_group->n,
					verbose_level - 1);
		}
		else {
			if (f_v) {
				cout << "matrix_group_element::GL_mult_internal "
						"before GFq->mult_matrix_matrix" << endl;
			}
			Matrix_group->GFq->Linear_algebra->mult_matrix_matrix(
					A, B, AB, Matrix_group->n, Matrix_group->n, Matrix_group->n,
					0 /* verbose_level */);
		}
	}
	else {
		cout << "matrix_group_element::GL_mult_internal unknown group type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group_element::GL_mult_internal done" << endl;
	}
}

void matrix_group_element::GL_copy(
		int *A, int *B)
{
	Int_vec_copy(A, B, Matrix_group->elt_size_int);
}

void matrix_group_element::GL_copy_internal(
		int *A, int *B)
{
	Int_vec_copy(A, B, Matrix_group->elt_size_int_half);
}

void matrix_group_element::GL_transpose(
		int *A, int *At, int verbose_level)
// inverse and transpose
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_transpose" << endl;
	}

	GL_transpose_internal(A, At, verbose_level);

	//GL_transpose_internal(A + elt_size_int_half,
	//		At + elt_size_int_half, verbose_level);

	GL_invert_internal(At, At + Matrix_group->elt_size_int_half, verbose_level - 2);

	if (f_v) {
		cout << "matrix_group_element::GL_transpose done" << endl;
	}
}

void matrix_group_element::GL_transpose_only(
		int *A, int *At, int verbose_level)
// transpose only. no invert
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_transpose_only" << endl;
	}

	Matrix_group->GFq->Linear_algebra->transpose_square_matrix(
			A,
			At,
			Matrix_group->n);

	Matrix_group->GFq->Linear_algebra->transpose_square_matrix(
			A + Matrix_group->elt_size_int_half,
			At + Matrix_group->elt_size_int_half,
			Matrix_group->n);

#if 0
	GL_copy_internal(A, At);
	Matrix_group->GFq->Linear_algebra->transpose_matrix_in_place(
			At, Matrix_group->n);

	GL_copy_internal(A + Matrix_group->elt_size_int_half, At + Matrix_group->elt_size_int_half);
	Matrix_group->GFq->Linear_algebra->transpose_matrix_in_place(
			At + Matrix_group->elt_size_int_half, Matrix_group->n);
#endif

	if (f_v) {
		cout << "matrix_group_element::GL_transpose_only done" << endl;
	}
}


void matrix_group_element::GL_transpose_internal(
		int *A, int *At, int verbose_level)
// inverse and transpose
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_transpose_internal" << endl;
	}
	if (Matrix_group->f_affine) {
		cout << "matrix_group_element::GL_transpose_internal "
				"not yet implemented for affine groups" << endl;
		exit(1);
	}

	Matrix_group->GFq->Linear_algebra->matrix_invert(
			A, Elt4,
			base_cols, At, Matrix_group->n, verbose_level - 2);

	Matrix_group->GFq->Linear_algebra->transpose_matrix_in_place(
			At, Matrix_group->n);

	if (Matrix_group->f_semilinear) {
		At[Matrix_group->offset_frobenius] = A[Matrix_group->offset_frobenius];
	}
	if (f_v) {
		cout << "matrix_group_element::GL_transpose_internal done" << endl;
	}
}

void matrix_group_element::GL_invert(
		int *A, int *Ainv)
{
	GL_copy_internal(A, Ainv + Matrix_group->elt_size_int_half);
	GL_copy_internal(A + Matrix_group->elt_size_int_half, Ainv);
}

void matrix_group_element::GL_invert_transpose(
		int *A, int *Ainv)
{
	GL_copy_internal(A, Ainv + Matrix_group->elt_size_int_half);
	Matrix_group->GFq->Linear_algebra->transpose_matrix_in_place(
			Ainv + Matrix_group->elt_size_int_half, Matrix_group->n);

	GL_copy_internal(A + Matrix_group->elt_size_int_half, Ainv);
	Matrix_group->GFq->Linear_algebra->transpose_matrix_in_place(
			Ainv, Matrix_group->n);
}

void matrix_group_element::GL_invert_internal(
		int *A, int *Ainv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "matrix_group_element::GL_invert_internal" << endl;
	}
	if (Matrix_group->f_projective) {
		if (Matrix_group->f_semilinear) {
			if (f_vv) {
				cout << "matrix_group_element::GL_invert_internal "
						"calling GFq->semilinear_matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_invert(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
		else {
			if (f_vv) {
				cout << "matrix_group_element::GL_invert_internal "
						"calling GFq->matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->matrix_invert(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
	}
	else if (Matrix_group->f_affine) {
		if (Matrix_group->f_semilinear) {
			if (f_vv) {
				cout << "matrix_group_element::semilinear_matrix_invert_affine "
						"calling GFq->semilinear_matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_invert_affine(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
		else {
			if (f_vv) {
				cout << "matrix_group_element::matrix_invert_affine "
						"calling GFq->semilinear_matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->matrix_invert_affine(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
	}
	else if (Matrix_group->f_general_linear) {
		if (Matrix_group->f_semilinear) {
			if (f_vv) {
				cout << "matrix_group_element::GL_invert_internal "
						"calling GFq->semilinear_matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->semilinear_matrix_invert(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
		else {
			if (f_vv) {
				cout << "matrix_group_element::GL_invert_internal "
						"calling GFq->matrix_invert" << endl;
			}
			Matrix_group->GFq->Linear_algebra->matrix_invert(
					A, Elt4,
					base_cols, Ainv, Matrix_group->n, verbose_level - 2);
		}
	}
	else {
		cout << "matrix_group_element::GL_invert_internal "
				"unknown group type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group_element::GL_invert_internal done" << endl;
	}
}

void matrix_group_element::GL_unpack(
		uchar *elt, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "matrix_group_element::GL_unpack" << endl;
		cout << "matrix_group_element::GL_unpack f_projective=" << Matrix_group->f_projective << endl;
		cout << "matrix_group_element::GL_unpack f_affine=" << Matrix_group->f_affine << endl;
		cout << "matrix_group_element::GL_unpack f_general_linear=" << Matrix_group->f_general_linear << endl;
		cout << "matrix_group_element::GL_unpack f_semilinear=" << Matrix_group->f_semilinear << endl;
		cout << "matrix_group_element::GL_unpack n=" << Matrix_group->n << endl;
		cout << "matrix_group_element::GL_unpack bits_per_digit=" << Matrix_group->bits_per_digit << endl;
		cout << "matrix_group_element::GL_unpack bits_per_elt=" << Matrix_group->bits_per_elt << endl;
		cout << "matrix_group_element::GL_unpack bits_extension_degree=" << Matrix_group->bits_extension_degree << endl;
		cout << "matrix_group_element::GL_unpack char_per_elt=" << Matrix_group->char_per_elt << endl;
	}
	if (elt == NULL) {
		cout << "matrix_group_element::GL_unpack elt == NULL" << endl;
		exit(1);
	}
	if (Elt == NULL) {
		cout << "matrix_group_element::GL_unpack Elt == NULL" << endl;
		exit(1);
	}


	if (Matrix_group->f_projective) {
		decode_matrix(
				Elt, Matrix_group->n, elt);
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = decode_frobenius(elt);
		}
	}
	else if (Matrix_group->f_affine) {
		decode_matrix(
				Elt, Matrix_group->n, elt);
		for (i = 0; i < Matrix_group->n; i++) {
			Elt[Matrix_group->offset_affine_vector + i] = get_digit(elt, Matrix_group->n, i);
		}
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = decode_frobenius(elt);
		}
	}
	else if (Matrix_group->f_general_linear) {
		decode_matrix(
				Elt, Matrix_group->n, elt);
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = decode_frobenius(elt);
		}
	}
	else {
		cout << "matrix_group_element::GL_unpack unknown group type" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "GL_unpack read:" << endl;
		GL_print_easy(Elt, cout);
		cout << "GL_unpack calling GL_invert_internal" << endl;
	}
	GL_invert_internal(Elt, Elt + Matrix_group->elt_size_int_half, verbose_level - 2);
	if (f_v) {
		cout << "matrix_group_element::GL_unpack done" << endl;
	}
}

void matrix_group_element::GL_pack(
		int *Elt, uchar *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::GL_pack" << endl;
	}

	int i;

	if (Matrix_group->f_projective) {
		if (f_v) {
			cout << "matrix_group_element::GL_pack f_projective" << endl;
		}
		if (f_v) {
			cout << "matrix_group_element::GL_pack before encode_matrix" << endl;
		}
		encode_matrix(
				Elt, Matrix_group->n, elt, verbose_level);
		if (f_v) {
			cout << "matrix_group_element::GL_pack after encode_matrix" << endl;
		}
		if (Matrix_group->f_semilinear) {
			encode_frobenius(
					elt, Elt[Matrix_group->offset_frobenius]);
		}
	}
	else if (Matrix_group->f_affine) {
		if (f_v) {
			cout << "matrix_group_element::GL_pack f_affine" << endl;
		}
		if (f_v) {
			cout << "matrix_group_element::GL_pack before encode_matrix" << endl;
		}
		encode_matrix(
				Elt, Matrix_group->n, elt, verbose_level);
		if (f_v) {
			cout << "matrix_group_element::GL_pack after encode_matrix" << endl;
		}
		for (i = 0; i < Matrix_group->n; i++) {
			put_digit(
					elt, Matrix_group->n, i, Elt[Matrix_group->offset_affine_vector + i]);
		}
		if (Matrix_group->f_semilinear) {
			encode_frobenius(
					elt, Elt[Matrix_group->offset_frobenius]);
		}
	}
	else if (Matrix_group->f_general_linear) {
		if (f_v) {
			cout << "matrix_group_element::GL_pack f_general_linear" << endl;
		}
		if (f_v) {
			cout << "matrix_group_element::GL_pack before encode_matrix" << endl;
		}
		encode_matrix(
				Elt, Matrix_group->n, elt, verbose_level);
		if (f_v) {
			cout << "matrix_group_element::GL_pack after encode_matrix" << endl;
		}
		if (Matrix_group->f_semilinear) {
			encode_frobenius(
					elt, Elt[Matrix_group->offset_frobenius]);
		}
	}
	else {
		cout << "matrix_group_element::GL_pack unknown group type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "matrix_group_element::GL_pack done" << endl;
	}
}

void matrix_group_element::GL_print_easy(
		int *Elt, std::ostream &ost)
{
    //int i, j, a;
    int w;

	w = (int) Matrix_group->GFq->log10_of_q;

	Int_matrix_print_width(ost, Elt, Matrix_group->n, Matrix_group->n, w);
#if 0
	for (i = 0; i < Matrix_group->n; i++) {
		for (j = 0; j < Matrix_group->n; j++) {
			a = Elt[i * Matrix_group->n + j];
			ost << setw(w) << a << " ";
		}
		ost << endl;
	}
#endif
	if (Matrix_group->f_affine) {
		Int_vec_print(ost, Elt + Matrix_group->offset_affine_vector, Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			ost << ", " << Elt[Matrix_group->offset_frobenius] << endl;
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			ost << ", " << Elt[Matrix_group->offset_frobenius] << endl;
		}
	}
}

void matrix_group_element::GL_code_for_make_element(
		int *Elt, int *data)
{
	Int_vec_copy(Elt, data, Matrix_group->n * Matrix_group->n);
	if (Matrix_group->f_affine) {
		Int_vec_copy(
				Elt + Matrix_group->offset_affine_vector,
				data + Matrix_group->offset_affine_vector,
				Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			data[Matrix_group->offset_frobenius] = Elt[Matrix_group->offset_frobenius];
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			data[Matrix_group->offset_frobenius] = Elt[Matrix_group->offset_frobenius];
		}
	}
}

void matrix_group_element::GL_print_for_make_element(
		int *Elt, std::ostream &ost)
{
	//int i, j, a;
	//int w;

	//w = (int) GFq->log10_of_q;

	int *Data;
	Data = NEW_int(Matrix_group->make_element_size);

	GL_code_for_make_element(
			Elt, Data);

	Int_vec_print_bare_fully(ost, Data, Matrix_group->make_element_size);

#if 0
	Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);

	if (Matrix_group->f_projective) {
		Matrix_group->GFq->Projective_space_basic->PG_element_normalize_from_front(
				D, 1, Matrix_group->n * Matrix_group->n);
	}

	for (i = 0; i < Matrix_group->n; i++) {
		for (j = 0; j < Matrix_group->n; j++) {
			a = D[i * Matrix_group->n + j];
			ost << a << ",";
		}
	}
	if (Matrix_group->f_affine) {
		for (i = 0; i < Matrix_group->n; i++) {
			a = Elt[Matrix_group->offset_affine_vector + i];
			ost << a << ",";
		}
		if (Matrix_group->f_semilinear) {
			ost << Elt[Matrix_group->offset_frobenius] << ",";
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			ost << Elt[Matrix_group->offset_frobenius] << ",";
		}
	}
#endif

	FREE_int(Data);

}

void matrix_group_element::GL_print_for_make_element_no_commas(
		int *Elt, std::ostream &ost)
{


	int *Data;
	Data = NEW_int(Matrix_group->make_element_size);

	GL_code_for_make_element(
			Elt, Data);

	int i;
	int w;

	w = (int) Matrix_group->GFq->log10_of_q;

	for (i = 0; i < Matrix_group->make_element_size; i++) {
		ost << setw(w) << Data[i] << " ";
	}
	FREE_int(Data);

#if 0
	int i, j, a;
	int w;

	w = (int) Matrix_group->GFq->log10_of_q;
	for (i = 0; i < Matrix_group->n; i++) {
		for (j = 0; j < Matrix_group->n; j++) {
			a = Elt[i * Matrix_group->n + j];
			ost << setw(w) << a << " ";
		}
	}
	if (Matrix_group->f_affine) {
		for (i = 0; i < Matrix_group->n; i++) {
			a = Elt[Matrix_group->offset_affine_vector + i];
			ost << setw(w) << a << " ";
		}
		if (Matrix_group->f_semilinear) {
			ost << Elt[Matrix_group->offset_frobenius] << " ";
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			ost << Elt[Matrix_group->offset_frobenius] << " ";
		}
	}
#endif
}

void matrix_group_element::GL_print_easy_normalized(
		int *Elt, std::ostream &ost)
{
	int f_v = false;
    //int i, j, a;
    int w;

	if (f_v) {
		cout << "matrix_group_element::GL_print_easy_normalized" << endl;
	}

	w = (int) Matrix_group->GFq->log10_of_q;
	if (Matrix_group->f_projective) {
		int *D;
		D = NEW_int(Matrix_group->n * Matrix_group->n);
		Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);
		Matrix_group->GFq->Projective_space_basic->PG_element_normalize_from_front(
				D, 1, Matrix_group->n * Matrix_group->n);

		Int_matrix_print_width(ost, D, Matrix_group->n, Matrix_group->n, w);
		FREE_int(D);
	}
	else if (Matrix_group->f_affine) {
		Int_matrix_print_width(ost, Elt, Matrix_group->n, Matrix_group->n, w);
		Int_vec_print(ost, Elt + Matrix_group->offset_affine_vector, Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			ost << ", " << Elt[Matrix_group->offset_frobenius] << endl;
		}
	}
	else if (Matrix_group->f_general_linear) {
		Int_matrix_print_width(ost, Elt, Matrix_group->n, Matrix_group->n, w);
		if (Matrix_group->f_semilinear) {
			ost << ", " << Elt[Matrix_group->offset_frobenius] << endl;
		}
	}
	else {
		cout << "matrix_group_element::GL_print_easy_normalized "
				"unknown group type" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "matrix_group_element::GL_print_easy_normalized done" << endl;
	}
}

void matrix_group_element::GL_print_latex(
		int *Elt, std::ostream &ost)
{

	int *D;
	D = NEW_int(Matrix_group->n * Matrix_group->n);

	Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);

	if (Matrix_group->f_projective) {
		//GFq->PG_element_normalize_from_front(D, 1, n * n);
		Matrix_group->GFq->Projective_space_basic->PG_element_normalize(
				D, 1, Matrix_group->n * Matrix_group->n);
	}

	Matrix_group->GFq->Io->print_matrix_latex(
			ost, D, Matrix_group->n, Matrix_group->n);

	if (Matrix_group->f_affine) {
		Matrix_group->GFq->Io->print_matrix_latex(
				ost, Elt + Matrix_group->offset_affine_vector, 1, Matrix_group->n);
		//int_vec_print(ost, Elt + n * n, n);
		if (Matrix_group->f_semilinear) {
			ost << "_{" << Elt[Matrix_group->offset_frobenius] << "}" << endl;
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			ost << "_{" << Elt[Matrix_group->offset_frobenius] << "}" << endl;
		}
	}
	FREE_int(D);
}

std::string matrix_group_element::GL_stringify(
		int *Elt, std::string &options)
{

	string s;

	int *D;
	D = NEW_int(Matrix_group->n * Matrix_group->n);

	Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);

	if (Matrix_group->f_projective) {

		//GFq->PG_element_normalize_from_front(D, 1, n * n);
		Matrix_group->GFq->Projective_space_basic->PG_element_normalize(
				D, 1, Matrix_group->n * Matrix_group->n);
	}

	s += Matrix_group->GFq->Io->stringify_matrix_latex(
			D, Matrix_group->n, Matrix_group->n);



	if (Matrix_group->f_affine) {

		s +=
		Matrix_group->GFq->Io->stringify_matrix_latex(
				Elt + Matrix_group->offset_affine_vector, 1, Matrix_group->n);

		if (Matrix_group->f_semilinear) {
			s += "_{" + std::to_string(
					Elt[Matrix_group->offset_frobenius]) + "}\n";
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			s += "_{" + std::to_string(Elt[Matrix_group->offset_frobenius]) + "}\n";
		}
	}
	FREE_int(D);

	return s;
}



void matrix_group_element::GL_print_latex_with_print_point_function(
		int *Elt,
		std::ostream &ost,
		void (*point_label)(
				std::stringstream &sstr, int pt, void *data),
		void *point_label_data)
{
	cout << "matrix_group_element::GL_print_latex_with_print_point_function nyi" << endl;
#if 0
	int i, j, a;
	//int w;

	//w = (int) GFq->log10_of_q;

	int *D;
	D = NEW_int(n * n);

	int_vec_copy(Elt, D, n * n);

	if (f_projective) {
		GFq->PG_element_normalize_from_front(D, 1, n * n);
	}

	ost << "\\left[" << endl;
	ost << "\\begin{array}{*{" << n << "}{r}}" << endl;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = D[i * n + j];

#if 0
			if (is_prime(GFq->q)) {
				ost << setw(w) << a << " ";
			}
			else {
				ost << a;
				// GFq->print_element(ost, a);
			}
#else
			GFq->print_element(ost, a);
#endif

			if (j < n - 1)
				ost << " & ";
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	if (f_affine) {
		int_vec_print(ost, Elt + n * n, n);
		if (f_semilinear) {
			ost << "_{" << Elt[n * n + n] << "}" << endl;
		}
	}
	else {
		if (f_semilinear) {
			ost << "_{" << Elt[n * n] << "}" << endl;
		}
	}
	FREE_int(D);
#endif

}

void matrix_group_element::GL_print_easy_latex(
		int *Elt, std::ostream &ost)
{

	GL_print_easy_latex_with_option_numerical(Elt, false, ost);

}

void matrix_group_element::GL_print_easy_latex_with_option_numerical(
		int *Elt, int f_numerical, std::ostream &ost)
{
    int i, j, a;
    //int w;

	//w = (int) GFq->log10_of_q;
	int *D;
	D = NEW_int(Matrix_group->n * Matrix_group->n);

    if (Matrix_group->f_projective) {
		Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);
		Matrix_group->GFq->Projective_space_basic->PG_element_normalize_from_front(
				D, 1, Matrix_group->n * Matrix_group->n);
		//GFq->PG_element_normalize(D, 1, n * n);
    }
    else {
    	Int_vec_copy(Elt, D, Matrix_group->n * Matrix_group->n);
    }


	if (Matrix_group->GFq->q <= 9) {
		ost << "\\left[" << endl;
		ost << "\\begin{array}{c}" << endl;
		for (i = 0; i < Matrix_group->n; i++) {
			for (j = 0; j < Matrix_group->n; j++) {
				a = D[i * Matrix_group->n + j];


				if (f_numerical) {
					ost << a;
				}
				else {
					Matrix_group->GFq->Io->print_element(ost, a);
				}

				//if (j < n - 1)
				//	ost << " & ";
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
	}
	else {
		ost << "\\left[" << endl;
		ost << "\\begin{array}{*{" << Matrix_group->n << "}{r}}" << endl;
		for (i = 0; i < Matrix_group->n; i++) {
			for (j = 0; j < Matrix_group->n; j++) {
				a = D[i * Matrix_group->n + j];

				if (f_numerical) {
					ost << a;
				}
				else {
					Matrix_group->GFq->Io->print_element(ost, a);
				}


				if (j < Matrix_group->n - 1)
					ost << " & ";
			}
			ost << "\\\\" << endl;
		}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
	}
	if (Matrix_group->f_affine) {
		Int_vec_print(ost, Elt + Matrix_group->offset_affine_vector, Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			ost << "_{" << Elt[Matrix_group->offset_frobenius] << "}" << endl;
		}
	}
	else {
		if (Matrix_group->f_semilinear) {
			ost << "_{" << Elt[Matrix_group->offset_frobenius] << "}" << endl;
		}
	}
	FREE_int(D);

}

void matrix_group_element::decode_matrix(
		int *Elt, int n, unsigned char *elt)
{
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			Elt[i * n + j] = get_digit(elt, i, j);
		}
	}
}

int matrix_group_element::get_digit(
		unsigned char *elt, int i, int j)
{
	int h0 = (int) (i * Matrix_group->n + j) * Matrix_group->bits_per_digit;
	int h, h1, word, bit;
	uchar mask, d = 0;

	for (h = (int) Matrix_group->bits_per_digit - 1; h >= 0; h--) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((uchar) 1) << bit;
		d <<= 1;
		if (elt[word] & mask) {
			d |= 1;
		}
	}
	return d;
}

int matrix_group_element::decode_frobenius(
		unsigned char *elt)
{
	int h0;
	int h, h1, word, bit;
	uchar mask, d = 0;

	h0 = (int) (Matrix_group->offset_frobenius) * Matrix_group->bits_per_digit;
#if 0
	if (Matrix_group->f_affine) {
		h0 = (int) (Matrix_group->offset_frobenius) * Matrix_group->bits_per_digit;
	}
	else {
		h0 = (int) Matrix_group->offset_frobenius * Matrix_group->bits_per_digit;
	}
#endif
	for (h = (int) Matrix_group->bits_extension_degree - 1; h >= 0; h--) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((uchar) 1) << bit;
		d <<= 1;
		if (elt[word] & mask) {
			d |= 1;
		}
	}
	return d;
}

void matrix_group_element::encode_matrix(
		int *Elt, int n,
		unsigned char *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::encode_matrix" << endl;
	}
	int i, j;

	if (f_v) {
		cout << "matrix_group_element::encode_matrix n=" << n << endl;
	}
	if (elt == NULL) {
		cout << "matrix_group_element::encode_matrix elt == NULL" << endl;
		exit(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			put_digit(elt, i, j, Elt[i * n + j]);
		}
	}
	if (f_v) {
		cout << "matrix_group_element::encode_matrix done" << endl;
	}
}

void matrix_group_element::put_digit(
		unsigned char *elt, int i, int j, int d)
{
	int h0 = (int) (i * Matrix_group->n + j) * Matrix_group->bits_per_digit;
	int h, h1, word, bit;
	uchar mask;

	//cout << "put_digit() " << d << " bits_per_digit = "
	//		<< bits_per_digit << endl;
	for (h = 0; h < Matrix_group->bits_per_digit; h++) {
		h1 = h0 + h;
		word = h1 >> 3;
		//cout << "word = " << word << endl;
		bit = h1 & 7;
		mask = ((uchar) 1) << bit;
		if (d & 1) {
			elt[word] |= mask;
		}
		else {
			uchar not_mask = ~mask;
			elt[word] &= not_mask;
		}
		d >>= 1;
	}
}

void matrix_group_element::encode_frobenius(
		unsigned char *elt, int d)
{
	int h0;
	int h, h1, word, bit;
	uchar mask;

	h0 = (int) Matrix_group->offset_frobenius * Matrix_group->bits_per_digit;
#if 0
	if (Matrix_group->f_affine) {
		h0 = (int) (Matrix_group->offset_frobenius) * Matrix_group->bits_per_digit;
	}
	else {
		h0 = (int) Matrix_group->offset_frobenius * Matrix_group->bits_per_digit;
	}
#endif
	for (h = 0; h < Matrix_group->bits_extension_degree; h++) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((uchar) 1) << bit;
		if (d & 1) {
			elt[word] |= mask;
		}
		else {
			uchar not_mask = ~mask;
			elt[word] &= not_mask;
		}
		d >>= 1;
	}
}

void matrix_group_element::make_element(
		int *Elt,
		int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, a, b;

	if (f_v) {
		cout << "matrix_group_element::make_element" << endl;
	}
	if (f_vv) {
		cout << "data: ";
		Int_vec_print(cout, data, Matrix_group->elt_size_int_half);
		cout << endl;
	}
	for (i = 0; i < Matrix_group->elt_size_int_half; i++) {
		a = data[i];
		if (a < 0) {
			b = -a;
			//b = (GFq->q - 1) / a;
			a = Matrix_group->GFq->power(Matrix_group->GFq->alpha, b);
		}
		Elt[i] = a;
	}
	if (f_vv) {
		cout << "matrix_group_element::make_element "
				"calling GL_invert_internal" << endl;
	}
	GL_invert_internal(Elt, Elt + Matrix_group->elt_size_int_half, verbose_level - 2);
	if (f_vv) {
		cout << "matrix_group_element::make_element "
				"created the following element" << endl;
		GL_print_easy(Elt, cout);
		cout << endl;
	}
	if (f_v) {
		cout << "matrix_group_element::make_element done" << endl;
	}
}

void matrix_group_element::make_GL_element(
		int *Elt, int *A, int f)
{
	if (Matrix_group->f_projective) {
		Int_vec_copy(A, Elt, Matrix_group->n * Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = f % Matrix_group->GFq->e;
		}
	}
	else if (Matrix_group->f_affine) {
		Int_vec_copy(A, Elt, Matrix_group->n * Matrix_group->n + Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = f % Matrix_group->GFq->e;
		}
	}
	else if (Matrix_group->f_general_linear) {
		Int_vec_copy(A, Elt, Matrix_group->n * Matrix_group->n);
		if (Matrix_group->f_semilinear) {
			Elt[Matrix_group->offset_frobenius] = f % Matrix_group->GFq->e;
		}
	}
	else {
		cout << "matrix_group_element::make_GL_element "
				"unknown group type" << endl;
		exit(1);
	}
	GL_invert_internal(Elt, Elt + Matrix_group->elt_size_int_half, false);
}

int matrix_group_element::has_shape_of_singer_cycle(
		int *Elt)
{
	int i, j, a, a0;

	a0 = Elt[0 * Matrix_group->n + 1];
	for (i = 0; i < Matrix_group->n - 1; i++) {
		for (j = 0; j < Matrix_group->n; j++) {
			a = Elt[i * Matrix_group->n + j];
			if (j == i + 1) {
				if (a != a0) {
					return false;
				}
			}
			else {
				if (a) {
					return false;
				}
			}
		}
	}
	return true;
}

void matrix_group_element::matrix_minor(
		int *Elt,
		int *Elt1, matrix_group *mtx1, int f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *data;
	int n1;

	if (f_v) {
		cout << "matrix_group_element::matrix_minor" << endl;
	}
	if (Matrix_group->f_affine) {
		cout << "matrix_group_element::matrix_minor cannot be affine" << endl;
		exit(1);
	}
	n1 = mtx1->n;
	data = NEW_int(mtx1->elt_size_int_half);
	Matrix_group->GFq->Linear_algebra->matrix_minor(
			Matrix_group->f_semilinear, Elt, data, Matrix_group->n, f, n1);

	if (Matrix_group->f_semilinear) {
		mtx1->Element->make_GL_element(
			Elt1, data, Elt[Matrix_group->offset_frobenius]);
	}
	else {
		mtx1->Element->make_GL_element(
			Elt1, data, 0);
	}

	if (f_v) {
		cout << "matrix_group_element::matrix_minor done" << endl;
	}
}

void matrix_group_element::retrieve(
		int hdl, void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::retrieve" << endl;
	}

	if (elt == NULL) {
		cout << "matrix_group_element::retrieve "
				"elt == NULL" << endl;
		exit(1);
	}

	int *Elt = (int *) elt;
	uchar *p_elt;

	if (f_v) {
		cout << "matrix_group_element::retrieve "
				"overall_length = " << Page_storage->overall_length << endl;
	}
	if (hdl >= Page_storage->overall_length) {
		cout << "matrix_group_element::retrieve "
				"hdl = " << hdl << endl;
		cout << "matrix_group_element::retrieve "
				"overall_length = " << Page_storage->overall_length << endl;
		exit(1);
	}

	p_elt = Page_storage->s_i(hdl);
	GL_unpack(p_elt, Elt, verbose_level);
	if (f_v) {
		GL_print_easy(Elt, cout);
	}
	if (f_v) {
		cout << "matrix_group_element::retrieve done" << endl;
	}

}

int matrix_group_element::store(
		void *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::store" << endl;
	}

	int *Elt = (int *) elt;
	int hdl;

	GL_pack(Elt, elt1, verbose_level);
	hdl = Page_storage->store(elt1);
	if (f_v) {
		cout << "matrix_group_element::store "
				"hdl = " << hdl << endl;
	}
	return hdl;
}

void matrix_group_element::dispose(
		int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::dispose "
				"hdl = " << hdl << endl;
	}
	Page_storage->dispose(hdl);
}

void matrix_group_element::print_point(
		long int a, std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::print_point" << endl;
	}
	geometry::other_geometry::geometry_global Gg;

	if (Matrix_group->f_projective) {
		Matrix_group->GFq->Projective_space_basic->PG_element_unrank_modified_lint(
				v1, 1, Matrix_group->n, a);
	}
	else if (Matrix_group->f_affine) {
		Gg.AG_element_unrank(Matrix_group->GFq->q, v1, 1, Matrix_group->n, a);
	}
	else if (Matrix_group->f_general_linear) {
		Gg.AG_element_unrank(Matrix_group->GFq->q, v1, 1, Matrix_group->n, a);
	}
	else {
		cout << "matrix_group_element::print_point unknown group type" << endl;
		exit(1);
	}
	Int_vec_print(ost, v1, Matrix_group->n);
}

void matrix_group_element::unrank_point(
		long int rk, int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::unrank_point" << endl;
	}
	geometry::other_geometry::geometry_global Gg;

	if (Matrix_group->f_projective) {
		Matrix_group->GFq->Projective_space_basic->PG_element_unrank_modified(
				v, 1 /* stride */, Matrix_group->n, rk);
	}
	else if (Matrix_group->f_affine) {
		Gg.AG_element_unrank(Matrix_group->GFq->q, v, 1, Matrix_group->n, rk);
	}
	else if (Matrix_group->f_general_linear) {
		Gg.AG_element_unrank(Matrix_group->GFq->q, v, 1, Matrix_group->n, rk);
	}
	else {
		cout << "matrix_group_element::unrank_point unknown group type" << endl;
		exit(1);
	}
}

long int matrix_group_element::rank_point(
		int *v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group_element::rank_point" << endl;
	}
	long int rk;
	geometry::other_geometry::geometry_global Gg;

	if (Matrix_group->f_projective) {
		Matrix_group->GFq->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1 /* stride */, Matrix_group->n, rk);
	}
	else if (Matrix_group->f_affine) {
		rk = Gg.AG_element_rank(Matrix_group->GFq->q, v, 1, Matrix_group->n);
	}
	else if (Matrix_group->f_general_linear) {
		rk = Gg.AG_element_rank(Matrix_group->GFq->q, v, 1, Matrix_group->n);
	}
	else {
		cout << "matrix_group_element::rank_point unknown group type" << endl;
		exit(1);
	}
	return rk;
}






}}}}



