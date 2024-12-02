// matrix_group.cpp
//
// Anton Betten
//
// started:  October 23, 2002
// last change:  November 11, 2005




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace algebra {
namespace basic_algebra {


matrix_group::matrix_group()
{
	Record_birth();
	f_projective = false;
	f_affine = false;
	f_general_linear = false;
	n = 0;
	degree = 0;

	f_semilinear = false;
	f_kernel_is_diagonal_matrices = false;
	bits_per_digit = 0;
	bits_per_elt = 0;
	bits_extension_degree = 0;
	char_per_elt = 0;
	elt_size_int = 0;
	elt_size_int_half = 0;
	low_level_point_size = 0;
	make_element_size = 0;

	//std::string label;
	//std::string label_tex;

	f_GFq_is_allocated = false;
	GFq = NULL;
	data = NULL;
	C = NULL;

	Element = NULL;
}




matrix_group::~matrix_group()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::~matrix_group" << endl;
	}
	Record_death();
	if (Element) {
		FREE_OBJECT(Element);
	}
	if (f_GFq_is_allocated) {
		FREE_OBJECT(GFq);
	}
	if (C) {
		FREE_OBJECT(C);
	}
	if (f_v) {
		cout << "matrix_group::~matrix_group done" << endl;
	}
}


void matrix_group::init_projective_group(
		int n,
		field_theory::finite_field *F, int f_semilinear,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	geometry::other_geometry::geometry_global Gg;

	if (f_v) {
		cout << "matrix_group::init_projective_group" << endl;
		cout << "n=" << n << endl;
		cout << "q=" << F->q << endl;
		cout << "f_semilinear=" << f_semilinear << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	matrix_group::f_projective = true;
	matrix_group::f_affine = false;
	matrix_group::f_general_linear = false;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = false;
	low_level_point_size = n;
	make_element_size = n * n;
	if (f_semilinear) {
		make_element_size++;
	}
	f_kernel_is_diagonal_matrices = true;
	degree = Gg.nb_PG_elements(n - 1, F->q);


	other::data_structures::string_tools ST;

	ST.name_of_group_projective(
			label,
			label_tex,
			n, F->q,
			f_semilinear, false /* f_special */,
			verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"before compute_elt_size" << endl;
	}
	compute_elt_size(0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"after compute_elt_size" << endl;
	}

	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"elt_size_int = " << elt_size_int << endl;
	}
	
	Element = NEW_OBJECT(matrix_group_element);

	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"before Element->init" << endl;
	}
	Element->init(this, verbose_level);
	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"after Element->init" << endl;
	}



#if 0
	if (f_vv) {
		cout << "matrix_group::init_projective_group "
				"before init_base" << endl;
	}
	init_base(A, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "matrix_group::init_projective_group "
				"after init_base" << endl;
	}
#endif

	//init_gl_classes(verbose_level - 1);


	if (f_v) {
		cout << "matrix_group::init_projective_group "
				"finished" << endl;
	}
}

void matrix_group::init_affine_group(
		int n,
		field_theory::finite_field *F, int f_semilinear,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	geometry::other_geometry::geometry_global Gg;

	if (f_vv) {
		cout << "matrix_group::init_affine_group" << endl;
	}
	matrix_group::f_projective = false;
	matrix_group::f_affine = true;
	matrix_group::f_general_linear = false;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = false;
	low_level_point_size = n;
	f_kernel_is_diagonal_matrices = false;
	degree = Gg.nb_AG_elements(n, F->q);
	make_element_size = n * n + n;
	if (f_semilinear) {
		make_element_size++;
	}

	other::data_structures::string_tools ST;

	ST.name_of_group_affine(
			label,
			label_tex,
			n, F->q, f_semilinear, false /* f_special */,
			verbose_level - 1);


	compute_elt_size(0 /*verbose_level - 1*/);

	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"elt_size_int = " << elt_size_int << endl;
	}
	
#if 0
	allocate_data(verbose_level);

	setup_page_storage(
			page_length_log, 0 /*verbose_level - 1*/);
#endif

	Element = NEW_OBJECT(matrix_group_element);

	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"before Element->init" << endl;
	}
	Element->init(this, verbose_level);
	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"after Element->init" << endl;
	}



#if 0
	if (f_vv) {
		cout << "matrix_group::init_affine_group "
				"before init_base" << endl;
	}
	init_base(A, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "matrix_group::init_affine_group "
				"after init_base" << endl;
	}
#endif


	//init_gl_classes(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"finished" << endl;
	}
}

void matrix_group::init_general_linear_group(
		int n,
		field_theory::finite_field *F, int f_semilinear,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	geometry::other_geometry::geometry_global Gg;

	if (f_vv) {
		cout << "matrix_group::init_general_linear_group" << endl;
	}
	matrix_group::f_projective = false;
	matrix_group::f_affine = false;
	matrix_group::f_general_linear = true;
	matrix_group::f_semilinear = f_semilinear;
	matrix_group::n = n;
	matrix_group::GFq = F;
	f_GFq_is_allocated = false;
	low_level_point_size = n;
	f_kernel_is_diagonal_matrices = false;
	degree = Gg.nb_AG_elements(n, F->q);
	make_element_size = n * n;
	if (f_semilinear) {
		make_element_size++;
	}

	other::data_structures::string_tools ST;

	ST.name_of_group_general_linear(
			label,
			label_tex,
			n, F->q, f_semilinear, false /* f_special */,
			verbose_level - 1);



	compute_elt_size(0 /*verbose_level - 1*/);

	if (f_v) {
		cout << "matrix_group::init_general_linear_group "
				"elt_size_int = " << elt_size_int << endl;
	}
	
#if 0
	allocate_data(0 /*verbose_level*/);

	setup_page_storage(
			page_length_log, verbose_level - 1);
#endif
	Element = NEW_OBJECT(matrix_group_element);

	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"before Element->init" << endl;
	}
	Element->init(this, verbose_level);
	if (f_v) {
		cout << "matrix_group::init_affine_group "
				"after Element->init" << endl;
	}



#if 0
	if (f_vv) {
		cout << "matrix_group::init_general_linear_group "
				"before init_base" << endl;
	}
	init_base(A, 0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "matrix_group::init_general_linear_group "
				"after init_base" << endl;
	}
#endif

	//init_gl_classes(verbose_level - 1);

	if (f_v) {
		cout << "matrix_group::init_general_linear_group "
				"done" << endl;
	}
}



void matrix_group::compute_elt_size(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	number_theory::number_theory_domain NT;

	if (f_v) {
		cout << "matrix_group::compute_elt_size" << endl;
	}
	if (f_semilinear && GFq->e > 1) {
		bits_extension_degree = NT.int_log2(GFq->e - 1);
	}
	else {
		bits_extension_degree = 0;
	}
	bits_per_digit = NT.int_log2(GFq->q - 1);
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
		elt_size_int = n * n;
	}
	else if (f_affine) {
		elt_size_int = n * n + n;
	}
	else if (f_general_linear) {
		elt_size_int = n * n;
	}
	else {
		cout << "matrix_group::compute_elt_size "
				"group type unknown" << endl;
		exit(1);
	}
	if (f_semilinear) {
		elt_size_int++;
	}
	
	elt_size_int_half = elt_size_int;
	elt_size_int *= 2;
	
	if (f_vv) {
		cout << "bits_per_digit = " << bits_per_digit << endl;
		cout << "bits_extension_degree = " << bits_extension_degree << endl;
		cout << "bits_per_elt = " << bits_per_elt << endl;
		cout << "char_per_elt = " << char_per_elt << endl;
		cout << "elt_size_int_half = " << elt_size_int_half << endl;
		cout << "elt_size_int = " << elt_size_int << endl;
	}
	if (f_v) {
		cout << "matrix_group::compute_elt_size done" << endl;
	}
}

void matrix_group::init_gl_classes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

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
			cout << "matrix_group::init_gl_classes "
					"before init gl_classes n = "
					<< n << " before NEW_OBJECT gl_classes" << endl;
		}
		C = NEW_OBJECT(linear_algebra::gl_classes);
		if (f_v) {
			cout << "matrix_group::init_gl_classes "
					"after NEW_OBJECT gl_classes" << endl;
		}
		C->init(n, GFq, verbose_level);
		if (f_v) {
			cout << "matrix_group::init_gl_classes "
					"after init gl_classes" << endl;
		}
	}
	else {
		cout << "matrix_group::init_gl_classes the field "
				"is not a prime field" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group::init_gl_classes done" << endl;
	}
}

// implementation functions for matrix group elements:


#if 0

void matrix_group::matrices_without_eigenvector_one(
		sims *S,
		int *&Sol, int &cnt, int f_path_select, int select_value,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	long int goi, rk, i, h;
	ring_theory::longinteger_object go;
	int *Id;
	int *Mtx1;
	int *Mtx2;
	int *Mtx3;
	int *Mtx4;
	vector<int> sol;

	S->group_order(go);
	goi = go.as_int();
	if (f_v) {
		cout << "matrix_group::matrices_without_eigenvector_one "
				"testing group of order " << goi << endl;
		if (f_path_select) {
			cout << "path[0] = " << select_value << endl;
		}
	}
	Elt1 = NEW_int(S->A->elt_size_in_int);
	Id = NEW_int(n * n);
	Mtx1 = NEW_int(n * n);
	Mtx2 = NEW_int(n * n);
	Mtx3 = NEW_int(n * n);
	Mtx4 = NEW_int(n * n);
	GFq->Linear_algebra->identity_matrix(Id, n);
	for (i = 0; i < n * n; i++) {
		Id[i] = GFq->negate(Id[i]);
	}
	if (f_v) {
		cout << "The negative Identity matrix is:" << endl;
		Int_matrix_print(Id, n, n);
	}
	cnt = 0;
	rk = 0;

	if (f_path_select) {
		S->path[0] = select_value;
		for (h = 1; h < S->A->base_len(); h++) {
			S->path[h] = 0;
		}
		rk = S->path_rank_lint();
	}


	while (rk < goi) {
		S->element_unrank_lint(rk, Elt1);
		S->path_unrank_lint(rk);
		if (f_path_select && S->path[0] > select_value) {
			break;
		}
		if (false) {
			cout << "testing matrix " << rk << " / " << goi << endl;
			Int_matrix_print(Elt1, n, n);
			S->path_unrank_lint(rk);
			cout << "path ";
			Int_vec_print(cout, S->path, S->A->base_len());
			cout << endl;
		}
		for (i = 0; i < n; i++) {
			Int_vec_copy(Elt1, Mtx1, (i + 1) * n);
			GFq->Linear_algebra->add_vector(
					Id, Mtx1, Mtx2, (i + 1) * n);
			if (false) {
				cout << "testing level " << i << " / " << n << ":" << endl;
				Int_matrix_print(Mtx2, (i + 1), n);
			}
			if (GFq->Linear_algebra->rank_of_rectangular_matrix_memory_given(
					Mtx2,
					(i + 1), n, Mtx3, Mtx4,
					false /* f_complete */,
					0 /* verbose_level */) < i + 1) {
				if (false) {
					cout << "failing level " << i << endl;
				}
				break;
			}
		}
		if (i < n) {
			S->path_unrank_lint(rk);
			while (i >= 0) {
				S->path[i]++;
				if (S->path[i] < S->get_orbit_length(i)) {
					break;
				}
				i--;
			}
			for (h = i + 1; h < S->A->base_len(); h++) {
				S->path[h] = 0;
			}
			if (false) {
				cout << "moving on to path ";
				Int_vec_print(cout, S->path, S->A->base_len());
				cout << endl;
			}
			rk = S->path_rank_lint();
			if (false) {
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
				cout << "The matrix " << rk << " / " << goi
						<< " (" << d << "%) has no eigenvector one, "
								"cnt=" << cnt << endl;
				S->path_unrank_lint(rk);
				cout << "path ";
				Int_vec_print(cout, S->path, S->A->base_len());
				cout << " : ";
				for (int t = 0; t < S->A->base_len(); t++) {
					cout << S->get_orbit_length(t) << ", ";
				}
				//int_vec_print(cout, S->orbit_len, S->A->base_len());
				if (f_path_select) {
					cout << " select_value = " << select_value;
				}
				cout << endl;
				Int_matrix_print(Elt1, n, n);
			}
			sol.push_back(rk);
			rk++;
		}
	}
	if (f_v) {
		cout << "We found " << cnt << " matrices without "
				"eigenvector one" << endl;
	}
	Sol = NEW_int(cnt);
	for (i = 0; i < cnt; i++) {
		Sol[i] = sol[i];
	}
	FREE_int(Elt1);
	FREE_int(Id);
	FREE_int(Mtx1);
	FREE_int(Mtx2);
	FREE_int(Mtx3);
	FREE_int(Mtx4);
	if (f_v) {
		cout << "matrix_group::matrices_without_eigenvector_one "
				"done, found this many matrices: " << cnt << endl;
	}
}
#endif



int matrix_group::base_len(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int base_len;
	algebra::basic_algebra::group_generators_domain GG;


	if (f_v) {
		cout << "matrix_group::base_len" << endl;
	}
	if (f_projective) {
		if (f_v) {
			cout << "matrix_group::base_len "
					"before GG.matrix_group_base_len_projective_group" << endl;
		}
		base_len = GG.matrix_group_base_len_projective_group(
					n, GFq->q,
					f_semilinear, verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_len "
					"after GG.matrix_group_base_len_projective_group" << endl;
		}
	}
	else if (f_affine) {
		if (f_v) {
			cout << "matrix_group::base_len "
					"before GG.matrix_group_base_len_affine_group" << endl;
		}
		base_len = GG.matrix_group_base_len_affine_group(
					n, GFq->q,
					f_semilinear, verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_len "
					"after GG.matrix_group_base_len_affine_group" << endl;
		}
	}
	else if (f_general_linear) {
		if (f_v) {
			cout << "matrix_group::base_len "
					"before GG.matrix_group_base_len_general_linear_group" << endl;
		}
		base_len = GG.matrix_group_base_len_general_linear_group(
					n, GFq->q,
					f_semilinear, verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_len "
					"after GG.matrix_group_base_len_general_linear_group" << endl;
		}
	}
	else {
		cout << "matrix_group::base_len no type" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "matrix_group::base_len base_len = " << base_len << endl;
	}
	if (f_v) {
		cout << "matrix_group::base_len done" << endl;
	}
	return base_len;
}

void matrix_group::base_and_transversal_length(
		int base_len,
		long int *base, int *transversal_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "matrix_group::base_and_transversal_length" << endl;
	}
	if (f_projective) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"before GGD.projective_matrix_group_base_and_transversal_length" << endl;
		}
		GGD.projective_matrix_group_base_and_transversal_length(
				n, GFq,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"after GGD.projective_matrix_group_base_and_transversal_length" << endl;
		}
	}
	else if (f_affine) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"before GGD.affine_matrix_group_base_and_transversal_length" << endl;
		}
		GGD.affine_matrix_group_base_and_transversal_length(
				n, GFq,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"after GGD.affine_matrix_group_base_and_transversal_length" << endl;
		}
	}
	else if (f_general_linear) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"before GGD.general_linear_matrix_group_base_and_transversal_length" << endl;
		}
		GGD.general_linear_matrix_group_base_and_transversal_length(
				n, GFq,
			f_semilinear,
			base_len, degree,
			base, transversal_length,
			verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::base_and_transversal_length "
					"after GGD.general_linear_matrix_group_base_and_transversal_length" << endl;
		}
	}
	if (f_v) {
		cout << "matrix_group::base_and_transversal_length base_len = " << base_len << endl;
		cout << "matrix_group::base_and_transversal_length transversal_length = ";
		Int_vec_print(cout, transversal_length, base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "matrix_group::base_and_transversal_length done" << endl;
	}
}

void matrix_group::strong_generators_low_level(
		int *&data,
		int &size, int &nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "matrix_group::strong_generators_low_level" << endl;
	}
	if (f_projective) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"before GGD.strong_generators_for_projective_linear_group" << endl;
		}
		GGD.strong_generators_for_projective_linear_group(
			n, GFq,
			f_semilinear,
			data, size, nb_gens,
			verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"after GGD.strong_generators_for_projective_linear_group" << endl;
		}
	}
	else if (f_affine) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"before GGD.strong_generators_for_affine_linear_group" << endl;
		}
		GGD.strong_generators_for_affine_linear_group(
				n, GFq,
				f_semilinear,
				data, size, nb_gens,
				verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"after GGD.strong_generators_for_affine_linear_group" << endl;
		}
	}
	else if (f_general_linear) {

		algebra::basic_algebra::group_generators_domain GGD;

		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"before GGD.strong_generators_for_general_linear_group" << endl;
		}
		GGD.strong_generators_for_general_linear_group(
			n, GFq,
			f_semilinear,
			data, size, nb_gens,
			verbose_level - 3);
		if (f_v) {
			cout << "matrix_group::strong_generators_low_level "
					"after GGD.strong_generators_for_general_linear_group" << endl;
		}
	}
	if (f_v) {
		cout << "matrix_group::strong_generators_low_level done" << endl;
	}
}


}}}}


