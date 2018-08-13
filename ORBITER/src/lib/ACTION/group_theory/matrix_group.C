// matrix_group.C
//
// Anton Betten
//
// started:  October 23, 2002
// last change:  November 11, 2005




#include "GALOIS/galois.h"
#include "action.h"


matrix_group::matrix_group()
{
	null();
}

matrix_group::~matrix_group()
{
	freeself();
}

void matrix_group::null()
{
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
	Elts = NULL;
	f_GFq_is_allocated = FALSE;
	GFq = NULL;
	C = NULL;
	f_kernel_is_diagonal_matrices = FALSE;
	low_level_point_size = 0;
	elt_size_INT = 0;
}

void matrix_group::freeself()
{
	INT verbose_level = 0;
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::freeself calling free_data" << endl;
		}
	free_data(verbose_level);
	if (f_v) {
		cout << "matrix_group::freeself destroying Elts" << endl;
		}
	if (Elts) {
		delete Elts;
		}
	if (f_v) {
		cout << "matrix_group::freeself destroying GFq" << endl;
		}
	if (f_GFq_is_allocated) {
		delete GFq;
		}
	if (C) {
		delete C;
		}
	null();
	if (f_v) {
		cout << "matrix_group::freeself finished" << endl;
		}
}

void matrix_group::init_projective_group(INT n,
		finite_field *F, INT f_semilinear, action *A,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT page_length_log = PAGE_LENGTH_LOG;

	if (f_v) {
		cout << "matrix_group::init_projective_group" << endl;
		cout << "n=" << n << endl;
		cout << "q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		}
	matrix_group::f_projective = TRUE;
	matrix_group::f_affine = FALSE;
	matrix_group::f_general_linear = FALSE;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = FALSE;
	low_level_point_size = n;
	make_element_size = n * n;
	if (f_semilinear) {
		make_element_size++;
	}
	f_kernel_is_diagonal_matrices = TRUE;
	degree = nb_PG_elements(n - 1, F->q);

	if (f_semilinear) {
		sprintf(label, "PGGL_%ld_%ld", n, F->q);
		sprintf(label_tex, "P\\Gamma L(%ld,%ld)", n, F->q);
		}
	else {
		sprintf(label, "PGL_%ld_%ld", n, F->q);
		sprintf(label_tex, "PGL(%ld,%ld)", n, F->q);
		}


	compute_elt_size(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_projective_group elt_size_INT = " << elt_size_INT << endl;
		}
	
	allocate_data(verbose_level);

	setup_page_storage(page_length_log, verbose_level - 1);




	if (f_vv) {
		cout << "matrix_group::init_projective_group before init_base" << endl;
		}
	init_base(A, verbose_level - 1);
	if (f_vv) {
		cout << "matrix_group::init_projective_group after init_base" << endl;
		}


	//init_gl_classes(verbose_level - 1);


	if (f_v) {
		cout << "matrix_group::init_projective_group finished" << endl;
		}
}

void matrix_group::init_affine_group(INT n,
		finite_field *F, INT f_semilinear, action *A,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT page_length_log = PAGE_LENGTH_LOG;

	if (f_vv) {
		cout << "matrix_group::init_affine_group" << endl;
		}
	matrix_group::f_projective = FALSE;
	matrix_group::f_affine = TRUE;
	matrix_group::f_general_linear = FALSE;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = FALSE;
	low_level_point_size = n;
	f_kernel_is_diagonal_matrices = FALSE;
	degree = nb_AG_elements(n, F->q);
	make_element_size = n * n + n;
	if (f_semilinear) {
		make_element_size++;
	}

	if (f_semilinear) {
		sprintf(label, "AGGL_%ld_%ld", n, F->q);
		sprintf(label_tex, "A\\Gamma L(%ld,%ld)", n, F->q);
		}
	else {
		sprintf(label, "AGL_%ld_%ld", n, F->q);
		sprintf(label_tex, "AGL(%ld,%ld)", n, F->q);
		}


	compute_elt_size(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_affine_group elt_size_INT = " << elt_size_INT << endl;
		}
	
	allocate_data(verbose_level);

	setup_page_storage(page_length_log, verbose_level - 1);




	if (f_vv) {
		cout << "matrix_group::init_affine_group before init_base" << endl;
		}
	init_base(A, verbose_level - 1);
	if (f_vv) {
		cout << "matrix_group::init_affine_group after init_base" << endl;
		}


	//init_gl_classes(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_affine_group finished" << endl;
		}
}

void matrix_group::init_general_linear_group(INT n,
		finite_field *F, INT f_semilinear, action *A,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT page_length_log = PAGE_LENGTH_LOG;

	if (f_vv) {
		cout << "matrix_group::init_general_linear_group" << endl;
		}
	matrix_group::f_projective = FALSE;
	matrix_group::f_affine = FALSE;
	matrix_group::f_general_linear = TRUE;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = FALSE;
	low_level_point_size = n;
	f_kernel_is_diagonal_matrices = FALSE;
	degree = nb_AG_elements(n, F->q);
	make_element_size = n * n;
	if (f_semilinear) {
		make_element_size++;
	}

	if (f_semilinear) {
		sprintf(label, "GGL_%ld_%ld", n, F->q);
		sprintf(label_tex, "\\Gamma L(%ld,%ld)", n, F->q);
		}
	else {
		sprintf(label, "GL_%ld_%ld", n, F->q);
		sprintf(label_tex, "GL(%ld,%ld)", n, F->q);
		}


	compute_elt_size(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_general_linear_group elt_size_INT = " << elt_size_INT << endl;
		}
	
	allocate_data(verbose_level);

	setup_page_storage(page_length_log, verbose_level - 1);




	if (f_vv) {
		cout << "matrix_group::init_general_linear_group before init_base" << endl;
		}
	init_base(A, verbose_level - 1);
	if (f_vv) {
		cout << "matrix_group::init_general_linear_group after init_base" << endl;
		}


	//init_gl_classes(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_general_linear_group finished" << endl;
		}
}

void matrix_group::allocate_data(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "matrix_group::allocate_data" << endl;
		}
	if (elt_size_INT == 0) {
		cout << "matrix_group::allocate_data elt_size_INT == 0" << endl;
		exit(1);
		}
	
	Elt1 = NEW_INT(elt_size_INT);
	Elt2 = NEW_INT(elt_size_INT);
	Elt3 = NEW_INT(elt_size_INT);
	Elt4 = NEW_INT(elt_size_INT);
	Elt5 = NEW_INT(elt_size_INT);
	tmp_M = NEW_INT(n * n);
	v1 = NEW_INT(2 * n);
	v2 = NEW_INT(2 * n);
	v3 = NEW_INT(2 * n);
	elt1 = new UBYTE[char_per_elt];
	elt2 = new UBYTE[char_per_elt];
	elt3 = new UBYTE[char_per_elt];
	base_cols = NEW_INT(n);
	
	if (f_v) {
		cout << "matrix_group::allocate_data done" << endl;
		}
}

void matrix_group::free_data(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "matrix_group::free_data" << endl;
		}
	if (Elt1) {
		if (f_v) {
			cout << "matrix_group::free_data freeing Elt1" << endl;
			}
		FREE_INT(Elt1);
		}
	if (Elt2) {
		if (f_v) {
			cout << "matrix_group::free_data freeing Elt2" << endl;
			}
		FREE_INT(Elt2);
		}
	if (Elt3) {
		if (f_v) {
			cout << "matrix_group::free_data freeing Elt3" << endl;
			}
		FREE_INT(Elt3);
		}
	if (Elt4) {
		if (f_v) {
			cout << "matrix_group::free_data freeing Elt4" << endl;
			}
		FREE_INT(Elt4);
		}
	if (Elt5) {
		if (f_v) {
			cout << "matrix_group::free_data freeing Elt5" << endl;
			}
		FREE_INT(Elt5);
		}
	if (tmp_M) {
		if (f_v) {
			cout << "matrix_group::free_data freeing tmp_M" << endl;
			}
		FREE_INT(tmp_M);
		}
	if (f_v) {
		cout << "matrix_group::free_data destroying v1-3" << endl;
		}
	if (v1) {
		FREE_INT(v1);
		}
	if (v2) {
		FREE_INT(v2);
		}
	if (v3) {
		FREE_INT(v3);
		}
	if (f_v) {
		cout << "matrix_group::free_data destroying elt1-3" << endl;
		}
	if (elt1) {
		delete [] elt1;
		}
	if (elt2) {
		delete [] elt2;
		}
	if (elt3) {
		delete [] elt3;
		}
	if (f_v) {
		cout << "matrix_group::free_data destroying base_cols" << endl;
		}
	if (base_cols) {
		FREE_INT(base_cols);
		}
	
	if (f_v) {
		cout << "matrix_group::free_data done" << endl;
		}
}

void matrix_group::setup_page_storage(INT page_length_log, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT hdl;
	
	if (f_v) {
		cout << "matrix_group::setup_page_storage" << endl;
		}
	if (Elts) {
		cout << "matrix_group::setup_page_storage Warning: Elts != NULL" << endl;
		delete Elts;
		}
	Elts = new page_storage;
	
	if (f_vv) {
		cout << "matrix_group::setup_page_storage calling Elts->init()" << endl;
		}
	Elts->init(char_per_elt /* entry_size */, page_length_log, verbose_level - 2);
	//Elts->add_elt_print_function(elt_print, (void *) this);
	
	
	if (f_vv) {
		cout << "matrix_group::setup_page_storage calling GL_one()" << endl;
		}
	GL_one(Elt1);
	GL_pack(Elt1, elt1);
	if (f_vv) {
		cout << "matrix_group::setup_page_storage calling Elts->store()" << endl;
		}
	hdl = Elts->store(elt1);
	if (f_vv) {
		cout << "identity element stored, hdl = " << hdl << endl;
		}
	if (f_v) {
		cout << "matrix_group::setup_page_storage done" << endl;
		}
}


void matrix_group::compute_elt_size(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::compute_elt_size" << endl;
		}
	if (f_semilinear && GFq->e > 1) {
		bits_extension_degree = INT_log2(GFq->e - 1);
		}
	else {
		bits_extension_degree = 0;
		}
	bits_per_digit = INT_log2(GFq->q - 1);
	if (f_projective) {
		bits_per_elt = n * n * bits_per_digit + bits_extension_degree;
		}
	else if (f_affine) {
		bits_per_elt = (n * n + n) * bits_per_digit + bits_extension_degree;
		}
	else if (f_general_linear) {
		bits_per_elt = n * n * bits_per_digit + bits_extension_degree;
		}
	else {
		cout << "matrix_group::compute_elt_size group type unknown" << endl;
		exit(1);
		}
	char_per_elt = bits_per_elt >> 3;
	if (bits_per_elt & 7) {
		char_per_elt++;
		}
	if (f_projective) {
		elt_size_INT = n * n;
		}
	else if (f_affine) {
		elt_size_INT = n * n + n;
		}
	else if (f_general_linear) {
		elt_size_INT = n * n;
		}
	else {
		cout << "matrix_group::compute_elt_size group type unknown" << endl;
		exit(1);
		}
	if (f_semilinear) {
		elt_size_INT++;
		}
	
	elt_size_INT_half = elt_size_INT;
	elt_size_INT *= 2;
	
	if (f_vv) {
		cout << "bits_per_digit = " << bits_per_digit << endl;
		cout << "bits_extension_degree = " << bits_extension_degree << endl;
		cout << "bits_per_elt = " << bits_per_elt << endl;
		cout << "char_per_elt = " << char_per_elt << endl;
		cout << "elt_size_INT_half = " << elt_size_INT_half << endl;
		cout << "elt_size_INT = " << elt_size_INT << endl;
		}
	if (f_v) {
		cout << "matrix_group::compute_elt_size done" << endl;
		}
}

void matrix_group::init_base(action *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	
	if (f_v) {
		cout << "matrix_group::init_base" << endl;
		}
	if (f_projective) {
		if (f_vv) {
			cout << "matrix_group::init_base before init_base_projective" << endl;
			}
		init_base_projective(A, verbose_level - 2);
		if (f_vv) {
			cout << "matrix_group::init_base after init_base_projective" << endl;
			}
		}
	else if (f_affine) {
		if (f_vv) {
			cout << "matrix_group::init_base before init_base_affine" << endl;
			}
		init_base_affine(A, verbose_level - 2);
		if (f_vv) {
			cout << "matrix_group::init_base after init_base_affine" << endl;
			}
		}
	else if (f_general_linear) {
		if (f_vv) {
			cout << "matrix_group::init_base before init_base_general_linear" << endl;
			}
		init_base_general_linear(A, verbose_level - 2);
		if (f_vv) {
			cout << "matrix_group::init_base after init_base_general_linear" << endl;
			}
		}
	else {
		cout << "matrix_group::init_base  group type unknown" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::init_base done" << endl;
		}
}

void matrix_group::init_base_projective(action *A, INT verbose_level)
// initializes base, base_len, degree, transversal_length, orbit, orbit_inv
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT q = GFq->q;
	
	if (f_v) {
		cout << "matrix_group::init_base_projective verbose_level=" << verbose_level << endl;
		}
	A->degree = degree;
	if (f_vv) {
		cout << "matrix_group::init_base_projective degree=" << degree << endl;
		}
	A->base_len = matrix_group_base_len_projective_group(
			n, q, f_semilinear, verbose_level - 1);
	if (f_vv) {
		cout << "matrix_group::init_base_projective base_len=" << A->base_len << endl;
		}

	A->allocate_base_data(A->base_len);

	if (f_vv) {
		cout << "matrix_group::init_base_projective before projective_matrix_group_base_and_orbits" << endl;
		}
	projective_matrix_group_base_and_orbits(n, 
		GFq, f_semilinear, 
		A->base_len, A->degree, 
		A->base, A->transversal_length, 
		A->orbit, A->orbit_inv, 
		verbose_level - 1);
		// in GALOIS/group_generators.C

	if (f_v) {
		cout << "matrix_group::init_base_projective: finished" << endl;
		}
}

void matrix_group::init_base_affine(action *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT q = GFq->q;
	
	if (f_v) {
		cout << "matrix_group::init_base_affine verbose_level=" << verbose_level << endl;
		}
	A->degree = degree;
	if (f_vv) {
		cout << "matrix_group::init_base_affine degree=" << degree << endl;
		}
	A->base_len = matrix_group_base_len_affine_group(
			n, q, f_semilinear, verbose_level - 1);
	if (f_vv) {
		cout << "matrix_group::init_base_affine base_len=" << A->base_len << endl;
		}

	A->allocate_base_data(A->base_len);

	if (f_vv) {
		cout << "matrix_group::init_base_affine before affine_matrix_group_base_and_orbits" << endl;
		}
	affine_matrix_group_base_and_transversal_length(n, 
		GFq, f_semilinear, 
		A->base_len, A->degree, 
		A->base, A->transversal_length, 
		verbose_level - 1);
		// in GALOIS/group_generators.C

	if (f_v) {
		cout << "matrix_group::init_base_affine: finished" << endl;
		}
}

void matrix_group::init_base_general_linear(action *A, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);
	INT q = GFq->q;
	
	if (f_v) {
		cout << "matrix_group::init_base_general_linear verbose_level=" << verbose_level << endl;
		}
	A->degree = degree;
	if (f_vv) {
		cout << "matrix_group::init_base_general_linear degree=" << degree << endl;
		}
	A->base_len = matrix_group_base_len_general_linear_group(
			n, q, f_semilinear, verbose_level - 1);
	// in GALOIS/group_generators.C
	if (f_vv) {
		cout << "matrix_group::init_base_general_linear base_len=" << A->base_len << endl;
		}

	A->allocate_base_data(A->base_len);

	if (f_vv) {
		cout << "matrix_group::init_base_general_linear before affine_matrix_group_base_and_orbits" << endl;
		}
	general_linear_matrix_group_base_and_transversal_length(n, 
		GFq, f_semilinear, 
		A->base_len, A->degree, 
		A->base, A->transversal_length, 
		verbose_level - 1);
		// in GALOIS/group_generators.C

	if (f_v) {
		cout << "matrix_group::init_base_affine: finished" << endl;
		}
}

void matrix_group::init_gl_classes(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::init_gl_classes" << endl;
		}
	if (GFq == NULL) {
		cout << "matrix_group::init_gl_classes GFq == NULL" << endl;
		exit(1);
		}
	if (GFq->e == 1) {
		// the following was added Dec 2, 2013:
		if (f_v) {
			cout << "matrix_group::init_gl_classes before init gl_classes n = "
					<< n << " before new gl_classes" << endl;
			}
		C = new gl_classes;
		if (f_v) {
			cout << "matrix_group::init_gl_classes after new gl_classes" << endl;
			}
		C->init(n, GFq, verbose_level);
		if (f_v) {
			cout << "matrix_group::init_gl_classes after init gl_classes" << endl;
			}
		}
	else {
		cout << "matrix_group::init_gl_classes the field is not a prime field" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::init_gl_classes done" << endl;
		}
}

// implementation functions for matrix group elements:

INT matrix_group::GL_element_entry_ij(INT *Elt, INT i, INT j)
{
	return Elt[i * n + j];
}

INT matrix_group::GL_element_entry_frobenius(INT *Elt)
{
	if (!f_semilinear) {
		cout << "matrix_group::GL_element_entry_frobenius fatal: !f_semilinear" << endl;
		exit(1);
		}
	if (f_projective) {
		return Elt[n * n];
		}
	else if (f_affine) {
		return Elt[n * n + n];
		}
	else if (f_general_linear) {
		return Elt[n * n];
		}
	else {
		cout << "matrix_group::GL_element_entry_frobenius unknown group type" << endl;
		exit(1);
		}
}

INT matrix_group::image_of_element(INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT b;
	
	if (f_v) {
		cout << "matrix_group::image_of_element" << endl;
		}
	if (f_projective) {
		b = GL_image_of_PG_element(Elt, a, verbose_level - 1);
		}
	else if (f_affine) {
		b = GL_image_of_AG_element(Elt, a, verbose_level - 1);
		}
	else if (f_general_linear) {
		b = GL_image_of_AG_element(Elt, a, verbose_level - 1);
		}
	else {
		cout << "matrix_group::image_of_element unknown group type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::image_of_element " << a << " maps to " << b << endl;
		}
	return b;
}


INT matrix_group::GL_image_of_PG_element(INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT b;
	
	if (f_v) {
		cout << "matrix_group::GL_image_of_PG_element" << endl;
		}
	PG_element_unrank_modified(*GFq, v1, 1, n, a);

	action_from_the_right_all_types(v1, Elt, v2, verbose_level - 1);
	
	PG_element_rank_modified(*GFq, v2, 1, n, b);

	if (f_v) {
		cout << "matrix_group::GL_image_of_PG_element done" << endl;
		}
	return b;
}

INT matrix_group::GL_image_of_AG_element(INT *Elt, INT a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT b;

	if (f_v) {
		cout << "matrix_group::GL_image_of_AG_element" << endl;
		}
	
	AG_element_unrank(GFq->q, v1, 1, n, a);

	action_from_the_right_all_types(v1, Elt, v2, verbose_level - 1);

	AG_element_rank(GFq->q, v2, 1, n, b);

	if (f_v) {
		cout << "matrix_group::GL_image_of_AG_element done" << endl;
		}
	return b;
}

void matrix_group::action_from_the_right_all_types(
		INT *v, INT *A, INT *vA, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::action_from_the_right_all_types" << endl;
		}
	if (f_projective) {
		projective_action_from_the_right(v, A, vA, verbose_level - 1);
		}
	else if (f_affine) {
		GFq->affine_action_from_the_right(f_semilinear, v, A, vA, n);
			// vA = (v * A)^{p^f} + b
			// where b = A + n * n
			// and f = A[n * n + n] if f_semilinear is TRUE
		}
	else if (f_general_linear) {
		general_linear_action_from_the_right(v, A, vA, verbose_level - 1);
		}
	else {
		cout << "matrix_group::action_from_the_right_all_types unknown group type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::action_from_the_right_all_types done" << endl;
		}
}

void matrix_group::projective_action_from_the_right(
		INT *v, INT *A, INT *vA, INT verbose_level)
// vA = (v * A)^{p^f} if f_semilinear,
// vA = v * A otherwise
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::projective_action_from_the_right"  << endl;
		}	
	GFq->projective_action_from_the_right(f_semilinear, v, A, vA, n, verbose_level - 1);
	if (f_v) {
		cout << "matrix_group::projective_action_from_the_right done"  << endl;
		}	
}

void matrix_group::general_linear_action_from_the_right(
		INT *v, INT *A, INT *vA, INT verbose_level)
// vA = (v * A)^{p^f} if f_semilinear,
// vA = v * A otherwise
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::general_linear_action_from_the_right"  << endl;
		}
	GFq->general_linear_action_from_the_right(f_semilinear, v, A, vA, n, verbose_level - 1);
	if (f_v) {
		cout << "matrix_group::general_linear_action_from_the_right done"  << endl;
		}	
}

void matrix_group::GL_one(INT *Elt)
{
	GL_one_internal(Elt);
	GL_one_internal(Elt + elt_size_INT_half);
}

void matrix_group::GL_one_internal(INT *Elt)
{
	INT i;
	
	if (f_projective) {
		GFq->identity_matrix(Elt, n);
		if (f_semilinear) {
			Elt[n * n] = 0;
			}
		}
	else if (f_affine) {
		GFq->identity_matrix(Elt, n);
		for (i = 0; i < n; i++) {
			Elt[n * n + i] = 0;
			}
		if (f_semilinear) {
			Elt[n * n + n] = 0;
			}
		}
	else {
		GFq->identity_matrix(Elt, n);
		if (f_semilinear) {
			Elt[n * n] = 0;
			}
		}
}

void matrix_group::GL_zero(INT *Elt)
{
	if (f_projective) {
		if (f_semilinear) {
			INT_vec_zero(Elt, n * n + 1);
			}
		else {
			INT_vec_zero(Elt, n * n);
			}
		}
	else if (f_affine) {
		if (f_semilinear) {
			INT_vec_zero(Elt, n * n + n + 1);
			}
		else {
			INT_vec_zero(Elt, n * n + n);
			}
		}
	if (f_general_linear) {
		if (f_semilinear) {
			INT_vec_zero(Elt, n * n + 1);
			}
		else {
			INT_vec_zero(Elt, n * n);
			}
		}
	else {
		cout << "matrix_group::GL_zero unknown group type" << endl;
		exit(1);
		}
	GL_copy_internal(Elt, Elt + elt_size_INT_half);
}

INT matrix_group::GL_is_one(INT *Elt)
{
	INT c;
	
	//cout << "matrix_group::GL_is_one" << endl;
	if (f_projective) {
		if (!GFq->is_scalar_multiple_of_identity_matrix(Elt, n, c)) {
			return FALSE;
			}
		if (f_semilinear) {
			if (Elt[n * n] != 0) {
				return FALSE;
				}
			}
		}
	else if (f_affine) {
		//cout << "matrix_group::GL_is_one f_affine" << endl;
		if (!GFq->is_identity_matrix(Elt, n)) {
			//cout << "matrix_group::GL_is_one not the identity matrix" << endl;
			//print_integer_matrix(cout, Elt, n, n);
			return FALSE;
			}
		if (!GFq->is_zero_vector(Elt + n * n, n)) {
			//cout << "matrix_group::GL_is_one not the zero vector" << endl;
			return FALSE;
			}
		if (f_semilinear) {
			if (Elt[n * n + n] != 0) {
				return FALSE;
				}
			}
		}
	else if (f_general_linear) {
		//cout << "matrix_group::GL_is_one f_general_linear" << endl;
		if (!GFq->is_identity_matrix(Elt, n)) {
			//cout << "matrix_group::GL_is_one not the identity matrix" << endl;
			//print_integer_matrix(cout, Elt, n, n);
			return FALSE;
			}
		if (f_semilinear) {
			if (Elt[n * n] != 0) {
				return FALSE;
				}
			}
		}
	else {
		cout << "matrix_group::GL_is_one unknown group type" << endl;
		exit(1);
		}
	return TRUE;
}

void matrix_group::GL_mult(INT *A, INT *B, INT *AB, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::GL_mult" << endl;
		}
	if (f_v) {
		cout << "matrix_group::GL_mult_verbose before GL_mult_internal (1)" << endl;
		}
	GL_mult_internal(A, B, AB, verbose_level - 1);
	if (f_v) {
		cout << "matrix_group::GL_mult_verbose before GL_mult_internal (2)" << endl;
		}
	GL_mult_internal(B + elt_size_INT_half, A + elt_size_INT_half, AB + elt_size_INT_half, verbose_level - 1);
	if (f_v) {
		cout << "matrix_group::GL_mult done" << endl;
		}
	
}

void matrix_group::GL_mult_internal(INT *A, INT *B, INT *AB, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::GL_mult_internal" << endl;
		}

	if (f_projective) {
		if (f_semilinear) {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->semilinear_matrix_mult" << endl;
				}
			//GFq->semilinear_matrix_mult(A, B, AB, n);
			GFq->semilinear_matrix_mult_memory_given(A, B, AB, tmp_M, n);
			}
		else {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->mult_matrix_matrix" << endl;
				}
			GFq->mult_matrix_matrix(A, B, AB, n, n, n);
			}
		}
	else if (f_affine) {
		if (f_semilinear) {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->semilinear_matrix_mult_affine" << endl;
				}
			GFq->semilinear_matrix_mult_affine(A, B, AB, n);
			}
		else {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->matrix_mult_affine" << endl;
				}
			GFq->matrix_mult_affine(A, B, AB, n, verbose_level - 1);
			}
		}
	else if (f_general_linear) {
		if (f_semilinear) {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->semilinear_matrix_mult" << endl;
				}
			//GFq->semilinear_matrix_mult(A, B, AB, n);
			GFq->semilinear_matrix_mult_memory_given(A, B, AB, tmp_M, n);
			}
		else {
			if (f_v) {
				cout << "matrix_group::GL_mult_internal before GFq->mult_matrix_matrix" << endl;
				}
			GFq->mult_matrix_matrix(A, B, AB, n, n, n);
			}
		}
	else {
		cout << "matrix_group::GL_mult_internal unknown group type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::GL_mult_internal done" << endl;
		}
}

void matrix_group::GL_copy(INT *A, INT *B)
{
	INT_vec_copy(A, B, elt_size_INT);
}

void matrix_group::GL_copy_internal(INT *A, INT *B)
{
	INT_vec_copy(A, B, elt_size_INT_half);
}

void matrix_group::GL_transpose(INT *A, INT *At, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::GL_transpose" << endl;
		}
	GL_transpose_internal(A, At, verbose_level);
	GL_transpose_internal(A + elt_size_INT_half, At + elt_size_INT_half, verbose_level);
	if (f_v) {
		cout << "matrix_group::GL_transpose done" << endl;
		}
}

void matrix_group::GL_transpose_internal(INT *A, INT *At, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "matrix_group::GL_transpose_internal" << endl;
		}
	if (f_affine) {
		cout << "matrix_group::GL_transpose_internal not yet implemented for affine groups" << endl;
		exit(1);
		}
	GFq->transpose_matrix(A, At, n, n);
	if (f_semilinear) {
		At[n * n] = A[n * n];
		}
	if (f_v) {
		cout << "matrix_group::GL_transpose_internal done" << endl;
		}
}

void matrix_group::GL_invert(INT *A, INT *Ainv)
{
	GL_copy_internal(A, Ainv + elt_size_INT_half);
	GL_copy_internal(A + elt_size_INT_half, Ainv);
}

void matrix_group::GL_invert_internal(INT *A, INT *Ainv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "matrix_group::GL_invert_internal" << endl;
		}
	if (f_projective) {
		if (f_semilinear) {
			if (f_vv) {
				cout << "matrix_group::GL_invert_internal calling GFq->semilinear_matrix_invert" << endl;
				}
			GFq->semilinear_matrix_invert(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		else {
			if (f_vv) {
				cout << "matrix_group::GL_invert_internal calling GFq->matrix_invert" << endl;
				}
			GFq->matrix_invert(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		}
	else if (f_affine) {
		if (f_semilinear) {
			if (f_vv) {
				cout << "matrix_group::semilinear_matrix_invert_affine calling GFq->semilinear_matrix_invert" << endl;
				}
			GFq->semilinear_matrix_invert_affine(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		else {
			if (f_vv) {
				cout << "matrix_group::matrix_invert_affine calling GFq->semilinear_matrix_invert" << endl;
				}
			GFq->matrix_invert_affine(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		}
	else if (f_general_linear) {
		if (f_semilinear) {
			if (f_vv) {
				cout << "matrix_group::GL_invert_internal calling GFq->semilinear_matrix_invert" << endl;
				}
			GFq->semilinear_matrix_invert(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		else {
			if (f_vv) {
				cout << "matrix_group::GL_invert_internal calling GFq->matrix_invert" << endl;
				}
			GFq->matrix_invert(A, Elt4, base_cols, Ainv, n, verbose_level - 2);
			}
		}
	else {
		cout << "matrix_group::GL_invert_internal unknown group type" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "matrix_group::GL_invert_internal done" << endl;
		}
	
}

void matrix_group::GL_unpack(UBYTE *elt, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j;
	
	if (f_v) {
		cout << "matrix_group::GL_unpack" << endl;
		}
	if (f_projective) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				Elt[i * n + j] = get_digit(elt, i, j);
				}
			}
		if (f_semilinear) {
			Elt[n * n] = get_digit_frobenius(elt);
			}
		}
	else if (f_affine) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				Elt[i * n + j] = get_digit(elt, i, j);
				}
			}
		for (i = 0; i < n; i++) {
			Elt[n * n + i] = get_digit(elt, n, i);
			}
		if (f_semilinear) {
			Elt[n * n] = get_digit_frobenius(elt);
			}
		}
	else if (f_general_linear) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				Elt[i * n + j] = get_digit(elt, i, j);
				}
			}
		if (f_semilinear) {
			Elt[n * n] = get_digit_frobenius(elt);
			}
		}
	else {
		cout << "matrix_group::GL_unpack unknown group type" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "GL_unpack read:" << endl;
		GL_print_easy(Elt, cout);
		cout << "GL_unpack calling GL_invert_internal" << endl;
		}
	GL_invert_internal(Elt, Elt + elt_size_INT_half, verbose_level - 2);
	if (f_v) {
		cout << "matrix_group::GL_unpack done" << endl;
		}
}

void matrix_group::GL_pack(INT *Elt, UBYTE *elt)
{
	INT i, j;
	
	if (f_projective) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				put_digit(elt, i, j, Elt[i * n + j]);
				}
			}
		if (f_semilinear) {
			put_digit_frobenius(elt, Elt[n * n]);
			}
		}
	else if (f_affine) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				put_digit(elt, i, j, Elt[i * n + j]);
				}
			}
		for (i = 0; i < n; i++) {
			put_digit(elt, n, i, Elt[n * n + i]);
			}
		if (f_semilinear) {
			put_digit_frobenius(elt, Elt[n * n + n]);
			}
		}
	else if (f_general_linear) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				put_digit(elt, i, j, Elt[i * n + j]);
				}
			}
		if (f_semilinear) {
			put_digit_frobenius(elt, Elt[n * n]);
			}
		}
	else {
		cout << "matrix_group::GL_pack unknown group type" << endl;
		exit(1);
		}
}

void matrix_group::GL_print_easy(INT *Elt, ostream &ost)
{
    INT i, j, a;
    int w;
	
	w = (int) GFq->log10_of_q;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = Elt[i * n + j];
			ost << setw(w) << a << " ";
			}
		ost << endl;
		}
	if (f_affine) {
		INT_vec_print(ost, Elt + n * n, n);
		if (f_semilinear) {
			ost << ", " << Elt[n * n + n] << endl;
			}
		}
	else {
		if (f_semilinear) {
			ost << ", " << Elt[n * n] << endl;
			}
		}
}

void matrix_group::GL_code_for_make_element(INT *Elt, INT *data)
{
	INT_vec_copy(Elt, data, n * n);
	if (f_affine) {
		INT_vec_copy(Elt + n * n, data + n * n, n);
		if (f_semilinear) {
			data[n * n + n] = Elt[n * n + n];
			}
		}
	else {
		if (f_semilinear) {
			data[n * n] = Elt[n * n];
			}
		}
}

void matrix_group::GL_print_for_make_element(INT *Elt, ostream &ost)
{
	INT i, j, a;
	int w;
	
	w = (int) GFq->log10_of_q;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = Elt[i * n + j];
			ost << setw(w) << a << ", ";
			}
		}
	if (f_affine) {
		for (i = 0; i < n; i++) {
			a = Elt[n * n + i];
			ost << setw(w) << a << ", ";
			}
		if (f_semilinear) {
			ost << Elt[n * n + n] << ", ";
			}
		}
	else {
		if (f_semilinear) {
			ost << Elt[n * n] << ", ";
			}
		}
}

void matrix_group::GL_print_for_make_element_no_commas(INT *Elt, ostream &ost)
{
	INT i, j, a;
	int w;
	
	w = (int) GFq->log10_of_q;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = Elt[i * n + j];
			ost << setw(w) << a << " ";
			}
		}
	if (f_affine) {
		for (i = 0; i < n; i++) {
			a = Elt[n * n + i];
			ost << setw(w) << a << " ";
			}
		if (f_semilinear) {
			ost << Elt[n * n + n] << " ";
			}
		}
	else {
		if (f_semilinear) {
			ost << Elt[n * n] << " ";
			}
		}
}

void matrix_group::GL_print_easy_normalized(INT *Elt, ostream &ost)
{
	INT f_v = FALSE;
    INT i, j, a;
    int w;
	
	if (f_v) {
		cout << "matrix_group::GL_print_easy_normalized" << endl;
		}

	w = (int) GFq->log10_of_q;
	if (f_projective) {
		INT *D;
		D = NEW_INT(n * n);
		INT_vec_copy(Elt, D, n * n);
		PG_element_normalize_from_front(*GFq, D, 1, n * n);
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = D[i * n + j];
				ost << setw(w) << a << " ";
				}
			ost << endl;
			}
		FREE_INT(D);
		}
	else if (f_affine) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = Elt[i * n + j];
				ost << setw(w) << a << ", ";
				}
			}
		INT_vec_print(ost, Elt + n * n, n);
		if (f_semilinear) {
			ost << ", " << Elt[n * n + n] << endl;
			}
		}
	else if (f_general_linear) {
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = Elt[i * n + j];
				ost << setw(w) << a << ", ";
				}
			}
		if (f_semilinear) {
			ost << ", " << Elt[n * n] << endl;
			}
		}
	else {
		cout << "matrix_group::GL_print_easy_normalized unknown group type" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "matrix_group::GL_print_easy_normalized done" << endl;
		}
}

void matrix_group::GL_print_latex(INT *Elt, ostream &ost)
{
	INT i, j, a;
	int w;
	
	w = (int) GFq->log10_of_q;

	INT *D;
	D = NEW_INT(n * n);

	INT_vec_copy(Elt, D, n * n);
	
	if (f_projective) {
		PG_element_normalize_from_front(*GFq, D, 1, n * n);
		}

	ost << "\\left[" << endl;
	ost << "\\begin{array}{*{" << n << "}{r}}" << endl;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			a = D[i * n + j];	

			if (is_prime(GFq->q)) {
				ost << setw(w) << a << " ";
				}
			else {
				ost << a;
				// GFq->print_element(ost, a);
				}
		
			if (j < n - 1)
				ost << " & ";
			}
		ost << "\\\\" << endl;
		}
	ost << "\\end{array}" << endl;
	ost << "\\right]" << endl;
	if (f_affine) {
		INT_vec_print(ost, Elt + n * n, n);
		if (f_semilinear) {
			ost << "_{" << Elt[n * n + n] << "}" << endl;
			}
		}
	else {
		if (f_semilinear) {
			ost << "_{" << Elt[n * n] << "}" << endl;
			}
		}
	FREE_INT(D);
}

void matrix_group::GL_print_easy_latex(INT *Elt, ostream &ost)
{
    INT i, j, a;
    int w;
	
	w = (int) GFq->log10_of_q;

	if (GFq->q <= 9) {
		ost << "\\left[" << endl;
		ost << "\\begin{array}{c}" << endl;
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = Elt[i * n + j];	

				if (is_prime(GFq->q)) {
					ost << a;
					}
				else {
					ost << a;
					//GFq->print_element(ost, a);
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
		ost << "\\begin{array}{*{" << n << "}{r}}" << endl;
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = Elt[i * n + j];	

				if (is_prime(GFq->q)) {
					ost << setw(w) << a << " ";
					}
				else {
					ost << a;
					// GFq->print_element(ost, a);
					}
			
				if (j < n - 1)
					ost << " & ";
				}
			ost << "\\\\" << endl;
			}
		ost << "\\end{array}" << endl;
		ost << "\\right]" << endl;
		}
	if (f_affine) {
		INT_vec_print(ost, Elt + n * n, n);
		if (f_semilinear) {
			ost << "_{" << Elt[n * n + n] << "}" << endl;
			}
		}
	else {
		if (f_semilinear) {
			ost << "_{" << Elt[n * n] << "}" << endl;
			}
		}
}

int matrix_group::get_digit(UBYTE *elt, INT i, INT j)
{
	int h0 = (int) (i * n + j) * bits_per_digit;
	int h, h1, word, bit;
	UBYTE mask, d = 0;
	
	for (h = (int) bits_per_digit - 1; h >= 0; h--) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((UBYTE) 1) << bit;
		d <<= 1;
		if (elt[word] & mask)
			d |= 1;
		}
	return d;
}

int matrix_group::get_digit_frobenius(UBYTE *elt)
{
	int h0;
	int h, h1, word, bit;
	UBYTE mask, d = 0;
	
	if (f_affine) {
		h0 = (int) (n * n + n) * bits_per_digit;
		}
	else {
		h0 = (int) n * n * bits_per_digit;
		}
	for (h = (int) bits_extension_degree - 1; h >= 0; h--) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((UBYTE) 1) << bit;
		d <<= 1;
		if (elt[word] & mask)
			d |= 1;
		}
	return d;
}

void matrix_group::put_digit(UBYTE *elt, INT i, INT j, INT d)
{
	int h0 = (int) (i * n + j) * bits_per_digit;
	int h, h1, word, bit;
	UBYTE mask;
	
	//cout << "put_digit() " << d << " bits_per_digit = " << bits_per_digit << endl;
	for (h = 0; h < bits_per_digit; h++) {
		h1 = h0 + h;
		word = h1 >> 3;
		//cout << "word = " << word << endl;
		bit = h1 & 7;
		mask = ((UBYTE) 1) << bit;
		if (d & 1) {
			elt[word] |= mask;
			}
		else {
			UBYTE not_mask = ~mask;
			elt[word] &= not_mask;
			}
		d >>= 1;
		}
}

void matrix_group::put_digit_frobenius(UBYTE *elt, INT d)
{
	int h0;
	int h, h1, word, bit;
	UBYTE mask;
	
	if (f_affine) {
		h0 = (int) (n * n + n) * bits_per_digit;
		}
	else {
		h0 = (int) n * n * bits_per_digit;
		}
	for (h = 0; h < bits_extension_degree; h++) {
		h1 = h0 + h;
		word = h1 >> 3;
		bit = h1 & 7;
		mask = ((UBYTE) 1) << bit;
		if (d & 1) {
			elt[word] |= mask;
			}
		else {
			UBYTE not_mask = ~mask;
			elt[word] &= not_mask;
			}
		d >>= 1;
		}
}

void matrix_group::make_element(INT *Elt, INT *data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, a, b;
	
	if (f_v) {
		cout << "matrix_group::make_element" << endl;
		}
	if (f_vv) {
		cout << "data: ";
		INT_vec_print(cout, data, elt_size_INT_half);
		cout << endl;
		}
	for (i = 0; i < elt_size_INT_half; i++) {
		a = data[i];
		if (a < 0) {
			b = -a;
			//b = (GFq->q - 1) / a;
			a = GFq->power(GFq->alpha, b);
			}
		Elt[i] = a;
		}
	if (f_vv) {
		cout << "matrix_group::make_element calling GL_invert_internal" << endl;
		}
	GL_invert_internal(Elt, Elt + elt_size_INT_half, verbose_level - 2);
	if (f_vv) {
		cout << "matrix_group::make_element created the following element" << endl;
		GL_print_easy(Elt, cout);
		cout << endl;
		}
	if (f_v) {
		cout << "matrix_group::make_element done" << endl;
		}
}

void matrix_group::make_GL_element(INT *Elt, INT *A, INT f)
{
	if (f_projective) {
		INT_vec_copy(A, Elt, n * n);
		if (f_semilinear) {
			Elt[n * n] = f % GFq->e;
			}
		}
	else if (f_affine) {
		INT_vec_copy(A, Elt, n * n + n);
		if (f_semilinear) {
			Elt[n * n + n] = f % GFq->e;
			}
		}
	else if (f_general_linear) {
		INT_vec_copy(A, Elt, n * n);
		if (f_semilinear) {
			Elt[n * n] = f % GFq->e;
			}
		}
	else {
		cout << "matrix_group::make_GL_element unknown group type" << endl;
		exit(1);
		}
	GL_invert_internal(Elt, Elt + elt_size_INT_half, FALSE);
}

void matrix_group::orthogonal_group_random_generator(action *A, orthogonal *O, 
	INT f_siegel, 
	INT f_reflection, 
	INT f_similarity,
	INT f_semisimilarity, 
	INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vvv = (verbose_level >= 3);
	INT *Mtx;

	if (f_v) {
		cout << "matrix_group::orthogonal_group_random_generator" << endl;
		cout << "f_siegel=" << f_siegel << endl;
		cout << "f_reflection=" << f_reflection << endl;
		cout << "f_similarity=" << f_similarity << endl;
		cout << "f_semisimilarity=" << f_semisimilarity << endl;
		cout << "n=" << n << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}

	Mtx = NEW_INT(n * n + 1);

	if (f_v) {
		cout << "matrix_group::orthogonal_group_random_generator before O->random_generator_for_orthogonal_group" << endl;
		}
	
	O->random_generator_for_orthogonal_group(
		f_semilinear /* f_action_is_semilinear */, 
		f_siegel, 
		f_reflection, 
		f_similarity,
		f_semisimilarity, 
		Mtx, verbose_level - 1);
	
	if (f_v) {
		cout << "matrix_group::orthogonal_group_random_generator after O->random_generator_for_orthogonal_group" << endl;
		cout << "Mtx=" << endl;
		INT_matrix_print(Mtx, n, n);
		}
	A->make_element(Elt, Mtx, verbose_level - 1);


	FREE_INT(Mtx);


	if (f_vvv) {
		cout << "matrix_group::orthogonal_group_random_generator random generator:" << endl;
		A->element_print_quick(Elt, cout);
		}
}


void matrix_group::matrices_without_eigenvector_one(sims *S,
		INT *&Sol, INT &cnt, INT f_path_select, INT select_value,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1;
	INT goi, rk, i, h;
	longinteger_object go;
	INT *Id;
	INT *Mtx1;
	INT *Mtx2;
	INT *Mtx3;
	INT *Mtx4;
	vector<INT> sol;

	S->group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "matrix_group::matrices_without_eigenvector_one testing group of order " << goi << endl;
		if (f_path_select) {
			cout << "path[0] = " << select_value << endl;
			}
		}
	Elt1 = NEW_INT(S->A->elt_size_in_INT);
	Id = NEW_INT(n * n);
	Mtx1 = NEW_INT(n * n);
	Mtx2 = NEW_INT(n * n);
	Mtx3 = NEW_INT(n * n);
	Mtx4 = NEW_INT(n * n);
	GFq->identity_matrix(Id, n);
	for (i = 0; i < n * n; i++) {
		Id[i] = GFq->negate(Id[i]);
		}
	if (f_v) {
		cout << "The negative Identity matrix is:" << endl;
		INT_matrix_print(Id, n, n);
		}
	cnt = 0;
	rk = 0;

	if (f_path_select) {
		S->path[0] = select_value;
		for (h = 1; h < S->A->base_len; h++) {
			S->path[h] = 0;
			}
		rk = S->path_rank_INT();
		}


	while (rk < goi) {
		S->element_unrank_INT(rk, Elt1);
		S->path_unrank_INT(rk);
		if (f_path_select && S->path[0] > select_value) {
			break;
			}
		if (FALSE) {
			cout << "testing matrix " << rk << " / " << goi << endl;
			INT_matrix_print(Elt1, n, n);
			S->path_unrank_INT(rk);
			cout << "path ";
			INT_vec_print(cout, S->path, S->A->base_len);
			cout << endl;
			}
		for (i = 0; i < n; i++) {
			INT_vec_copy(Elt1, Mtx1, (i + 1) * n);
			GFq->add_vector(Id, Mtx1, Mtx2, (i + 1) * n);
			if (FALSE) {
				cout << "testing level " << i << " / " << n << ":" << endl;
				INT_matrix_print(Mtx2, (i + 1), n);
				}
			if (GFq->rank_of_rectangular_matrix_memory_given(Mtx2,
					(i + 1), n, Mtx3, Mtx4, 0 /* verbose_level */) < i + 1) {
				if (FALSE) {
					cout << "failing level " << i << endl;
					}
				break;
				}
			}
		if (i < n) {
			S->path_unrank_INT(rk);
			while (i >= 0) {
				S->path[i]++;
				if (S->path[i] < S->orbit_len[i]) {
					break;
					}
				i--;
				}
			for (h = i + 1; h < S->A->base_len; h++) {
				S->path[h] = 0;
				}
			if (FALSE) {
				cout << "moving on to path ";
				INT_vec_print(cout, S->path, S->A->base_len);
				cout << endl;
				}
			rk = S->path_rank_INT();
			if (FALSE) {
				cout << "moving on to matrix " << rk << " / " << goi << endl;
				}
			if (rk == 0) {
				break;
				}
			}
		else {
			cnt++;
			if ((cnt % 10000) == 0) {
				double d;
				
				d = (double) rk / (double) goi * 100.;
				cout << "The matrix " << rk << " / " << goi << " (" << d << "%) has no eigenvector one, cnt=" << cnt << endl;
				S->path_unrank_INT(rk);
				cout << "path ";
				INT_vec_print(cout, S->path, S->A->base_len);
				cout << " : ";
				INT_vec_print(cout, S->orbit_len, S->A->base_len);
				if (f_path_select) {
					cout << " select_value = " << select_value;
					}
				cout << endl;
				INT_matrix_print(Elt1, n, n);
				}
			sol.push_back(rk);
			rk++;
			}
		}
	if (f_v) {
		cout << "We found " << cnt << " matrices without eigenvector one" << endl;
		}
	Sol = NEW_INT(cnt);
	for (i = 0; i < cnt; i++) {
		Sol[i] = sol[i];
		}
	FREE_INT(Elt1);
	FREE_INT(Id);
	FREE_INT(Mtx1);
	FREE_INT(Mtx2);
	FREE_INT(Mtx3);
	FREE_INT(Mtx4);
	if (f_v) {
		cout << "matrix_group::matrices_without_eigenvector_one done, found this many matrices: " << cnt << endl;
		}
}


void matrix_group::matrix_minor(INT *Elt, INT *Elt1, matrix_group *mtx1, INT f, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *data;
	INT n1;

	if (f_v) {
		cout << "matrix_group::matrix_minor" << endl;
		}
	if (f_affine) {
		cout << "matrix_group::matrix_minor cannot be affine" << endl;
		exit(1);
		}
	n1 = mtx1->n;
	data = NEW_INT(mtx1->elt_size_INT_half);
	GFq->matrix_minor(f_semilinear, Elt, data, n, f, n1);

	mtx1->make_GL_element(Elt1, data, Elt[n * n]);
	
	if (f_v) {
		cout << "matrix_group::matrix_minor done" << endl;
		}
}



