// wreath_product.cpp
//
// Anton Betten
//
// started:  August 2, 2018




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


static void dimensions(int n, int &nb_rows, int &nb_cols);
static void place_binary(long int h, int &i, int &j);


wreath_product::wreath_product()
{
	M = NULL;
	A_mtx = NULL;
	F = NULL;
	q = 0;
	nb_factors = 0;

	//label[0] = 0;
	//label_tex[0] = 0;

	degree_of_matrix_group = 0;
	dimension_of_matrix_group = 0;
	dimension_of_tensor_action = 0;
	degree_of_tensor_action = 0;
	degree_overall = 0;
	low_level_point_size = 0;
	make_element_size = 0;

	P = NULL;
	elt_size_int = 0;

	perm_offset_i = NULL;
	mtx_size = NULL;
	index_set1 = NULL;
	index_set2 = NULL;
	u = NULL;
	v = NULL;
	w = NULL;
	A1 = NULL;
	A2 = NULL;
	A3 = NULL;
	tmp_Elt1 = NULL;
	tmp_perm1 = NULL;
	tmp_perm2 = NULL;
	induced_perm = NULL;
	bits_per_digit = 0;
	bits_per_elt = 0;
	char_per_elt = 0;

	elt1 = NULL;
	base_len_in_component = 0;
	base_for_component = NULL;
	tl_for_component = NULL;

	base_length = 0;
	the_base = NULL;
	the_transversal_length = NULL;

	Elts = NULL;

	rank_one_tensors = NULL;
	rank_one_tensors_in_PG = NULL;
	rank_one_tensors_in_PG_sorted = NULL;

	nb_rank_one_tensors = 0;

	TR = NULL;
	Prev = NULL;


	//null();
}

wreath_product::~wreath_product()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::~wreath_product" << endl;
		}
	if (mtx_size) {
		FREE_int(mtx_size);
	}
	if (perm_offset_i) {
		FREE_int(perm_offset_i);
	}
	if (index_set1) {
		FREE_int(index_set1);
	}
	if (index_set2) {
		FREE_int(index_set2);
	}
	if (u) {
		FREE_int(u);
	}
	if (v) {
		FREE_int(v);
	}
	if (w) {
		FREE_int(w);
	}
	if (A1) {
		FREE_int(A1);
	}
	if (A2) {
		FREE_int(A2);
	}
	if (A3) {
		FREE_int(A3);
	}
	if (tmp_Elt1) {
		FREE_int(tmp_Elt1);
	}
	if (tmp_perm1) {
		FREE_int(tmp_perm1);
	}
	if (tmp_perm2) {
		FREE_int(tmp_perm2);
	}
	if (induced_perm) {
		FREE_int(induced_perm);
	}
	if (P) {
		FREE_OBJECT(P);
	}
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (Elts) {
		FREE_OBJECT(Elts);
	}
	if (base_for_component) {
		FREE_lint(base_for_component);
	}
	if (tl_for_component) {
		FREE_int(tl_for_component);
	}
	if (the_base) {
		FREE_lint(the_base);
	}
	if (the_transversal_length) {
		FREE_int(the_transversal_length);
	}
	if (rank_one_tensors) {
		FREE_int((int *) rank_one_tensors);
	}
	if (rank_one_tensors_in_PG) {
		FREE_lint(rank_one_tensors_in_PG);
	}
	if (rank_one_tensors_in_PG_sorted) {
		FREE_lint(rank_one_tensors_in_PG_sorted);
	}
	if (TR) {
		FREE_char(TR);
	}
	if (Prev) {
		FREE_int((int *) Prev);
	}
	if (f_v) {
		cout << "wreath_product::~wreath_product finished" << endl;
		}
}

void wreath_product::init_tensor_wreath_product(matrix_group *M,
		actions::action *A_mtx, int nb_factors,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	number_theory::number_theory_domain NT;
	algebra::group_generators_domain GG;

	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product" << endl;
	}
	if (M->f_projective) {
		cout << "wreath_product::init_tensor_wreath_product "
				"the input group must be of type general linear "
				"(not projective)" << endl;
		exit(1);
	}
	if (M->f_affine) {
		cout << "wreath_product::init_tensor_wreath_product "
				"the input group must be of type general linear "
				"(not affine)" << endl;
		exit(1);
	}
	wreath_product::M = M;
	wreath_product::A_mtx = A_mtx;
	wreath_product::nb_factors = nb_factors;
	F = M->GFq;
	q = F->q;

	P = NEW_OBJECT(permutation_representation_domain);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product before P->init" << endl;
	}
	P->init(nb_factors, 10 /* page_length_log */, 0 /* verbose_level */);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product after P->init" << endl;
	}

	char str1[1000];
	char str2[1000];

	snprintf(str1, sizeof(str1), "_wreath_Sym%d", nb_factors);
	snprintf(str2, sizeof(str2), " \\wr {\\rm Sym}(%d)", nb_factors);

	label.assign(M->label);
	label.append(str1);
	label_tex.assign(M->label_tex);
	label_tex.append(str2);

	degree_of_matrix_group = M->degree;
	dimension_of_matrix_group = M->n;
	dimension_of_tensor_action =
			NT.i_power_j(dimension_of_matrix_group, nb_factors);
	low_level_point_size = dimension_of_tensor_action;
	make_element_size = nb_factors + nb_factors *
			dimension_of_matrix_group * dimension_of_matrix_group;
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"computing degree_of_tensor_action" << endl;
	}
	degree_of_tensor_action =
			(NT.i_power_j_lint_safe(q, dimension_of_tensor_action, verbose_level) - 1) / (q - 1);
		// warning: int overflow possible

	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"degree_of_tensor_action = " << degree_of_tensor_action << endl;
	}
	degree_overall = nb_factors + nb_factors *
			degree_of_matrix_group + degree_of_tensor_action;
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"degree_overall = " << degree_overall << endl;
	}
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"degree_of_matrix_group = "
				<< degree_of_matrix_group << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"dimension_of_matrix_group = "
				<< dimension_of_matrix_group << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"low_level_point_size = "
				<< low_level_point_size << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"dimension_of_tensor_action = "
				<< dimension_of_tensor_action << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"degree_of_tensor_action = "
				<< degree_of_tensor_action << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"degree_overall = " << degree_overall << endl;
	}
	perm_offset_i = NEW_int(nb_factors + 1);
		// one more so it can also be used to indicate
		// the start of the tensor product action also.
	perm_offset_i[0] = nb_factors;
	for (i = 1; i <= nb_factors; i++) {
		// note equality here!
		perm_offset_i[i] = perm_offset_i[i - 1] + degree_of_matrix_group;
	}
	mtx_size = NEW_int(nb_factors);
	for (i = 0; i < nb_factors; i++) {
		mtx_size[i] = NT.i_power_j(dimension_of_matrix_group, i + 1);
	}
	index_set1 = NEW_int(nb_factors);
	index_set2 = NEW_int(nb_factors);
	u = NEW_int(dimension_of_tensor_action);
	v = NEW_int(dimension_of_tensor_action);
	w = NEW_int(dimension_of_tensor_action);
	A1 = NEW_int(dimension_of_tensor_action * dimension_of_tensor_action);
	A2 = NEW_int(dimension_of_tensor_action * dimension_of_tensor_action);
	A3 = NEW_int(dimension_of_tensor_action * dimension_of_tensor_action);
	elt_size_int = M->elt_size_int * nb_factors + P->elt_size_int;
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"elt_size_int = " << elt_size_int << endl;
	}
	tmp_Elt1 = NEW_int(elt_size_int);
	tmp_perm1 = NEW_int(P->elt_size_int);
	tmp_perm2 = NEW_int(P->elt_size_int);
	induced_perm = NEW_int(dimension_of_tensor_action);

	bits_per_digit = M->bits_per_digit;
	bits_per_elt = nb_factors * dimension_of_matrix_group *
			dimension_of_matrix_group * bits_per_digit;
	char_per_elt = nb_factors + ((bits_per_elt + 7) >> 3);
	elt1 = NEW_uchar(char_per_elt);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"bits_per_digit = " << bits_per_digit << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"bits_per_elt = " << bits_per_elt << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"char_per_elt = " << char_per_elt << endl;
	}
	base_len_in_component = GG.matrix_group_base_len_general_linear_group(
			dimension_of_matrix_group, q,
			FALSE /*f_semilinear */, verbose_level - 1);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"base_len_in_component = " << base_len_in_component << endl;
	}
	base_for_component = NEW_lint(base_len_in_component);
	tl_for_component = NEW_int(base_len_in_component);


	algebra::group_generators_domain GGD;


	GGD.general_linear_matrix_group_base_and_transversal_length(
		dimension_of_matrix_group, F,
		FALSE /* f_semilinear */,
		base_len_in_component, degree_of_matrix_group,
		base_for_component, tl_for_component,
		verbose_level - 1);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"base_for_component = ";
		Lint_vec_print(cout, base_for_component, base_len_in_component);
		cout << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"tl_for_component = ";
		Int_vec_print(cout, tl_for_component, base_len_in_component);
		cout << endl;
	}

	Elts = NEW_OBJECT(data_structures::page_storage);
	Elts->init(char_per_elt /* entry_size */,
			10 /* page_length_log */, verbose_level);

	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"before compute_base_and_transversals" << endl;
	}
	compute_base_and_transversals(verbose_level);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"after compute_base_and_transversals" << endl;
	}
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"the_base = ";
		Lint_vec_print(cout, the_base, base_length);
		cout << endl;
		cout << "wreath_product::init_tensor_wreath_product "
				"the_transversal_length = ";
		Int_vec_print(cout, the_transversal_length, base_length);
		cout << endl;
	}


	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"before create_all_rank_one_tensors" << endl;
	}
	create_all_rank_one_tensors(
			rank_one_tensors,
			nb_rank_one_tensors,
			verbose_level);
	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product "
				"after create_all_rank_one_tensors" << endl;
	}

	save_rank_one_tensors(verbose_level);



	if (f_v) {
		cout << "wreath_product::init_tensor_wreath_product done" << endl;
	}
}

void wreath_product::compute_tensor_ranks(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::compute_tensor_ranks" << endl;
	}
	compute_tensor_ranks(TR, Prev, verbose_level);
}

void wreath_product::unrank_point(long int a, int *v, int verbose_level)
{
	if (a < nb_factors) {
		cout << "wreath_product::unrank_point "
					"we are in the permutation, cannot unrank" << endl;
		exit(1);
	}
	a -= nb_factors;
	if (a < nb_factors * M->degree) {
		cout << "wreath_product::unrank_point "
					"we are in the projected components, cannot unrank" << endl;
		exit(1);
	}
	a -= nb_factors * M->degree;
	F->PG_element_unrank_modified_lint(v, 1,
			dimension_of_tensor_action, a);
}

long int wreath_product::rank_point(int *v, int verbose_level)
{
	long int a, b;

	a = nb_factors + nb_factors * M->degree;
	F->PG_element_rank_modified_lint(v, 1,
			dimension_of_tensor_action, b);
	a += b;
	return a;
}

long int wreath_product::element_image_of(int *Elt, long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int f, a0, b, c;
	geometry::geometry_global Gg;

	if (f_v) {
		cout << "wreath_product::element_image_of" << endl;
	}
	a0 = a;
	b = 0;
	if (a < nb_factors) {
		if (f_v) {
			cout << "wreath_product::element_image_of "
					"we are in the permutation" << endl;
		}
		b = Elt[a];
	}
	else {
		a -= nb_factors;
		b += nb_factors;
		for (f = 0; f < nb_factors; f++) {
			if (a < M->degree) {
				if (f_v) {
					cout << "wreath_product::element_image_of "
							"we are in component " << f
							<< " reduced input a=" << a << endl;
				}
				Gg.AG_element_unrank(q, u, 1, M->n, a);
				F->Linear_algebra->mult_vector_from_the_left(u, Elt + offset_i(f), v,
						M->n, M->n);
				c = Gg.AG_element_rank(q, v, 1, M->n);
				if (f_v) {
					cout << "wreath_product::element_image_of "
							"we are in component " << f
							<< " reduced output c=" << c << endl;
				}
				b += c;
				break;
			}
			else {
				a -= M->degree;
				b += M->degree;
			}
		} // next f
		if (f == nb_factors) {
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"we are in the tensor product component "
						"reduced input a = " << a << endl;
			}
			F->PG_element_unrank_modified_lint(u, 1,
					dimension_of_tensor_action, a);
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"u = ";
				Int_vec_print(cout, u, dimension_of_tensor_action);
				cout << endl;
			}
			create_matrix(Elt, A3, 0 /* verbose_level */);
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"A3 = " << endl;
				Int_matrix_print(A3,
						dimension_of_tensor_action,
						dimension_of_tensor_action);
			}
			F->Linear_algebra->mult_vector_from_the_left(u, A3, v,
					dimension_of_tensor_action,
					dimension_of_tensor_action);
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"v = ";
				Int_vec_print(cout, v, dimension_of_tensor_action);
				cout << endl;
			}
			apply_permutation(Elt, v, w, verbose_level);
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"w = ";
				Int_vec_print(cout, w, dimension_of_tensor_action);
				cout << endl;
			}
			F->PG_element_rank_modified_lint(w, 1,
					dimension_of_tensor_action, c);
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"we are in tensor product component " << f
						<< " reduced output c=" << c << endl;
			}
			b += c;
			if (f_v) {
				cout << "wreath_product::element_image_of "
						"we are in tensor product component " << f
						<< " output b=" << b << endl;
			}
		}
	}
	if (f_v) {
		cout << "wreath_product::element_image_of " << a0
				<< " maps to " << b << endl;
	}
	return b;
}

void wreath_product::element_image_of_low_level(int *Elt,
		int *input, int *output, int verbose_level)
// we assume that we are in the tensor product domain
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::element_image_of_low_level" << endl;
	}
	if (f_v) {
		cout << "wreath_product::element_image_of_low_level "
				"input = ";
		Int_vec_print(cout, input, dimension_of_tensor_action);
		cout << endl;
	}
	create_matrix(Elt, A3, 0 /* verbose_level */);
	if (f_v) {
		cout << "wreath_product::element_image_of_low_level "
				"A3 = " << endl;
		Int_matrix_print(A3,
				dimension_of_tensor_action,
				dimension_of_tensor_action);
	}
	F->Linear_algebra->mult_vector_from_the_left(input, A3, v,
			dimension_of_tensor_action,
			dimension_of_tensor_action);
	if (f_v) {
		cout << "wreath_product::element_image_of_low_level "
				"v = ";
		Int_vec_print(cout, v, dimension_of_tensor_action);
		cout << endl;
	}
	apply_permutation(Elt, v, output, verbose_level - 1);
	if (f_v) {
		cout << "wreath_product::element_image_of_low_level "
				"output = ";
		Int_vec_print(cout, output, dimension_of_tensor_action);
		cout << endl;
	}
	if (f_v) {
		cout << "wreath_product::element_image_of_low_level done" << endl;
	}
}

void wreath_product::element_one(int *Elt)
{
	int f;

	P->one(Elt);
	for (f = 0; f < nb_factors; f++) {
		M->GL_one(Elt + offset_i(f));
	}
}

int wreath_product::element_is_one(int *Elt)
{
		int f; //, scalar;

		if (!P->is_one(Elt)) {
			return FALSE;
		}
		for (f = 0; f < nb_factors; f++) {
			if (!F->Linear_algebra->is_identity_matrix(
					Elt + offset_i(f), dimension_of_matrix_group)) {
				return FALSE;
			}
#if 0
			if (!F->is_scalar_multiple_of_identity_matrix(
					Elt + offset_i(f), dimension_of_matrix_group,
					scalar)) {
				return FALSE;
			}
#endif
		}
		return TRUE;
}

void wreath_product::element_mult(int *A, int *B, int *AB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f, g;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "wreath_product::element_mult" << endl;
	}
	Combi.perm_mult(A, B, AB, nb_factors);
	for (f = 0; f < nb_factors; f++) {
		g = A[f];
		M->GL_mult(A + offset_i(f),
				B + offset_i(g),
				AB + offset_i(f),
				0 /* verbose_level */);
	}
	if (f_v) {
		cout << "wreath_product::element_mult done" << endl;
	}
}

void wreath_product::element_move(int *A, int *B, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::element_move" << endl;
	}
	Int_vec_copy(A, B, elt_size_int);

#if 0
	if (f_v) {
		element_print_easy(B, cout);
	}
#endif

	if (f_v) {
		cout << "wreath_product::element_move done" << endl;
	}
}

void wreath_product::element_invert(int *A, int *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f, g;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "wreath_product::element_invert" << endl;
	}
	Combi.perm_inverse(A, Av, nb_factors);
	for (f = 0; f < nb_factors; f++) {
		g = A[f];
		M->GL_invert(A + offset_i(f), Av + offset_i(g));
	}
	if (f_v) {
		cout << "wreath_product::element_invert done" << endl;
	}
}

void wreath_product::compute_induced_permutation(int *Elt, int *perm)
{
	int i, j, h, k, a;
	geometry::geometry_global Gg;

	for (i = 0; i < dimension_of_tensor_action; i++) {
		Gg.AG_element_unrank(dimension_of_matrix_group,
				index_set1, 1, nb_factors, i);
		for (h = 0; h < nb_factors; h++) {
			a = index_set1[h];
			k = Elt[h];
			index_set2[k] = a;
		}
		j = Gg.AG_element_rank(dimension_of_matrix_group,
				index_set2, 1, nb_factors);
		perm[i] = j;
	}
}


void wreath_product::apply_permutation(int *Elt,
		int *v_in, int *v_out, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;

	if (f_v) {
		cout << "wreath_product::apply_permutation" << endl;
	}
	if (f_v) {
		cout << "wreath_product::apply_permutation perm=";
		Int_vec_print(cout, Elt, nb_factors);
		cout << endl;
	}
	if (f_v) {
		cout << "wreath_product::apply_permutation v_in=";
		Int_vec_print(cout, v_in, dimension_of_tensor_action);
		cout << endl;
	}
	Int_vec_zero(v_out, dimension_of_tensor_action);

	//perm_inverse(Elt, tmp_perm1, nb_factors);

	compute_induced_permutation(Elt, induced_perm);
	if (f_v) {
		cout << "wreath_product::apply_permutation induced_perm=";
		Int_vec_print(cout, induced_perm, dimension_of_tensor_action);
		cout << endl;
	}

	for (i = 0; i < dimension_of_tensor_action; i++) {
		j = induced_perm[i];
		v_out[j] = v_in[i];
	}

	if (f_v) {
		cout << "wreath_product::apply_permutation" << endl;

		cout << "i : in[i] : perm[i] : out[i] " << endl;
		for (i = 0; i < dimension_of_tensor_action; i++) {
			cout << setw(3) << i << " & " << setw(3) << v_in[i]
					<< " & " << setw(3) << induced_perm[i]
					<< " & " << setw(3) << v_out[i] << endl;
		}
	}
	if (f_v) {
		cout << "wreath_product::apply_permutation v_out=";
		Int_vec_print(cout, v_out, dimension_of_tensor_action);
		cout << endl;
	}
	if (f_v) {
		cout << "wreath_product::apply_permutation done" << endl;
	}
}

int wreath_product::offset_i(int f)
{
	return P->elt_size_int + f * M->elt_size_int;
}

void wreath_product::create_matrix(int *Elt, int *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f, N;

	if (f_v) {
		cout << "wreath_product::create_matrix" << endl;
	}
	for (f = 0; f < nb_factors; f++) {
		if (f == 0) {
			Int_vec_copy(Elt + offset_i(f), A1,
					dimension_of_matrix_group * dimension_of_matrix_group);
			N = dimension_of_matrix_group;
		}
		else {
			F->Linear_algebra->Kronecker_product_square_but_arbitrary(
					A1, Elt + offset_i(f),
					mtx_size[f - 1], dimension_of_matrix_group,
					A2, N, 0 /* verbose_level */);
			Int_vec_copy(A2, A1, N * N);
		}
		if (f_v) {
			cout << "wreath_product::create_matrix "
					"after step " << f << ":" << endl;
			Int_matrix_print(A1, N, N);
		}
	}
	Int_vec_copy(A1, A,
			dimension_of_tensor_action * dimension_of_tensor_action);
	if (f_v) {
		cout << "wreath_product::create_matrix done" << endl;
	}
}

void wreath_product::element_pack(int *Elt, uchar *elt)
{
	int i, j, f;

	for (f = 0; f < nb_factors; f++) {
		elt[f] = (uchar) Elt[f];
	}
	for (f = 0; f < nb_factors; f++) {
		for (i = 0; i < dimension_of_matrix_group; i++) {
			for (j = 0; j < dimension_of_matrix_group; j++) {
				put_digit(elt, f, i, j,
						(Elt + offset_i(f))[i * dimension_of_matrix_group + j]);
			}
		}
	}
}

void wreath_product::element_unpack(uchar *elt, int *Elt)
{
	int i, j, f;
	int *m;

	for (f = 0; f < nb_factors; f++) {
		Elt[f] = elt[f];
	}
	for (f = 0; f < nb_factors; f++) {
		m = Elt + offset_i(f);
		for (i = 0; i < dimension_of_matrix_group; i++) {
			for (j = 0; j < dimension_of_matrix_group; j++) {
				m[i * dimension_of_matrix_group + j] =
						get_digit(elt, f, i, j);
			}
		}
		M->GL_invert_internal(m, m + M->elt_size_int_half,
				0 /*verbose_level - 2*/);
	}
}

void wreath_product::put_digit(uchar *elt, int f, int i, int j, int d)
{
	data_structures::data_structures_global D;
	int h0 = (int) (f * dimension_of_matrix_group * dimension_of_matrix_group +
			(i * dimension_of_matrix_group + j)) * bits_per_digit;
	int h, h1, a;

	for (h = 0; h < bits_per_digit; h++) {
		h1 = h0 + h;

		if (d & 1) {
			a = 1;
		}
		else {
			a = 0;
		}
		D.bitvector_m_ii(elt + nb_factors, h1, a);
		d >>= 1;
	}
}

int wreath_product::get_digit(uchar *elt, int f, int i, int j)
{
	data_structures::data_structures_global D;
	int h0 = (int) (f * dimension_of_matrix_group * dimension_of_matrix_group +
			(i * dimension_of_matrix_group + j)) * bits_per_digit;
	int h, h1, a, d;

	d = 0;
	for (h = bits_per_digit - 1; h >= 0; h--) {
		h1 = h0 + h;

		a = D.bitvector_s_i(elt + nb_factors, h1);
		d <<= 1;
		if (a) {
			d |= 1;
		}
	}
	return d;
}

void wreath_product::make_element_from_one_component(int *Elt,
		int f, int *Elt_component)
{
		int g;

		P->one(Elt);
		for (g = 0; g < nb_factors; g++) {
			if (g == f) {
				M->GL_copy(Elt_component, Elt + offset_i(g));
			}
			else {
				M->GL_one(Elt + offset_i(g));
			}
		}
}

void wreath_product::make_element_from_permutation(int *Elt, int *perm)
{
		int f;

		for (f = 0; f < nb_factors; f++) {
			Elt[f] = perm[f];
		}
		for (f = 0; f < nb_factors; f++) {
			M->GL_one(Elt + offset_i(f));
		}
}

void wreath_product::make_element(int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f, offset;

	if (f_v) {
		cout << "wreath_product::make_element" << endl;
		}
	if (f_v) {
		cout << "wreath_product::make_element data:" << endl;
		Int_vec_print(cout, data, make_element_size);
		cout << endl;
	}
	for (f = 0; f < nb_factors; f++) {
		Elt[f] = data[f];
	}
	offset = nb_factors;
	for (f = 0; f < nb_factors; f++) {
		M->make_element(Elt + offset_i(f), data + offset,
				0 /* verbose_level */);
		offset += M->elt_size_int_half;
	}
	if (f_v) {
		cout << "wreath_product::make_element created this element:" << endl;
		element_print_easy(Elt, cout);
	}
	if (f_v) {
		cout << "wreath_product::make_element done" << endl;
		}
}

void wreath_product::element_print_for_make_element(int *Elt, ostream &ost)
{
	int f;

	for (f = 0; f < nb_factors; f++) {
		ost << Elt[f] << ",";
	}
	for (f = 0; f < nb_factors; f++) {
		M->GL_print_for_make_element(Elt + offset_i(f), ost);
	}
}

void wreath_product::element_print_easy(int *Elt, ostream &ost)
{
	int f;

	ost << "begin element of wreath product: " << endl;
	ost << "[";
	for (f = 0; f < nb_factors; f++) {
		ost << Elt[f];
		if (f < nb_factors - 1) {
			ost << ", ";
		}
	}
	ost << "]" << endl;
	for (f = 0; f < nb_factors; f++) {
		ost << "factor " << f << ":" << endl;
		M->GL_print_easy(Elt + offset_i(f), ost);
	}
	ost << "end element of wreath product" << endl;
}

void wreath_product::element_print_latex(int *Elt, ostream &ost)
{
	int f;
	combinatorics::combinatorics_domain Combi;

	ost << "\\left(";
	for (f = 0; f < nb_factors; f++) {
		M->GL_print_latex(Elt + offset_i(f), ost);
	}
	ost << "; \\;" << endl;
	Combi.perm_print(ost, Elt, nb_factors);
#if 0
	ost << "[";
	for (f = 0; f < nb_factors; f++) {
		ost << Elt[f];
		if (f < nb_factors - 1) {
			ost << ", ";
		}
	}
	ost << "]";
#endif
	ost << "\\right)" << endl;
}

void wreath_product::compute_base_and_transversals(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f, h;

	if (f_v) {
		cout << "wreath_product::compute_base_and_transversals" << endl;
		}
	base_length = 0;
	base_length += nb_factors;
	base_length += nb_factors * base_len_in_component;
	the_base = NEW_lint(base_length);

	h = 0;
	for (i = 0; i < nb_factors; i++, h++) {
		the_base[h] = i;
	}
	for (f = 0; f < nb_factors; f++) {
		for (i = 0; i < base_len_in_component; i++, h++) {
			the_base[h] = perm_offset_i[f] + base_for_component[i];
		}
	}
	if (h != base_length) {
		cout << "wreath_product::compute_base_and_transversals "
				"h != base_length (1)" << endl;
		exit(1);
	}
	the_transversal_length = NEW_int(base_length);
	h = 0;
	for (i = 0; i < nb_factors; i++, h++) {
		the_transversal_length[h] = nb_factors - i;
	}
	for (f = 0; f < nb_factors; f++) {
		for (i = 0; i < base_len_in_component; i++, h++) {
			the_transversal_length[h] = tl_for_component[i];
		}
	}
	if (h != base_length) {
		cout << "wreath_product::compute_base_and_transversals "
				"h != base_length (2)" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "wreath_product::compute_base_and_transversals done" << endl;
		}
}

void wreath_product::make_strong_generators_data(int *&data,
		int &size, int &nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *GL_data;
	int GL_size;
	int GL_nb_gens;
	int h, k, f, g;
	int *dat;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "wreath_product::make_strong_generators_data" << endl;
	}
	if (f_v) {
		cout << "wreath_product::make_strong_generators_data "
				"before strong_generators_for_general_linear_group" << endl;
	}

	algebra::group_generators_domain GGD;


	GGD.strong_generators_for_general_linear_group(
		dimension_of_matrix_group, F,
		FALSE /*M->f_semilinear*/,
		GL_data, GL_size, GL_nb_gens,
		verbose_level - 1);

	if (f_v) {
		cout << "wreath_product::make_strong_generators_data "
				"after strong_generators_for_general_linear_group" << endl;
	}
	nb_gens = nb_factors - 1 + nb_factors * GL_nb_gens;
	size = nb_factors + nb_factors *
			dimension_of_matrix_group * dimension_of_matrix_group;
	data = NEW_int(nb_gens * size);
	dat = NEW_int(size);

	h = 0;
	// generators for the components:
	for (f = nb_factors - 1; f >= 0; f--) {
		for (g = 0; g < GL_nb_gens; g++) {
			Combi.perm_identity(dat, nb_factors);
			for (k = 0; k < nb_factors; k++) {
				if (k == f) {
					Int_vec_copy(GL_data + g * GL_size,
							dat + nb_factors + k * M->elt_size_int_half,
							GL_size);
				}
				else {
					F->Linear_algebra->identity_matrix(
							dat + nb_factors + k * M->elt_size_int_half,
							dimension_of_matrix_group);
				}
			}
			Int_vec_copy(dat, data + h * size, size);
			h++;
		}
	}
#if 1
	// create the elementary swap permutations:
	for (k = nb_factors - 2; k >= 0; k--) {
		Combi.perm_elementary_transposition(dat, nb_factors, k);
		for (f = 0; f < nb_factors; f++) {
			F->Linear_algebra->identity_matrix(dat + nb_factors + f * M->elt_size_int_half,
					dimension_of_matrix_group);
		}
		Int_vec_copy(dat, data + h * size, size);
		h++;
	}
#endif
	if (h != nb_gens) {
		cout << "h != nb_gens" << endl;
		exit(1);
	}
	FREE_int(dat);
	if (f_v) {
		cout << "wreath_product::make_strong_generators_data done" << endl;
	}
}
void wreath_product::report_rank_one_tensors(
		ostream &ost, int verbose_level)
{
	verbose_level = 1;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	combinatorics::combinatorics_domain Combi;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	orbiter_kernel_system::latex_interface L;
	int *coords;
	int *Proj;
	int *projections;
	int *projections1;
	int *tensor;
	int *T;
	int i, j, h, a, len, N;
	int nb_rows, nb_cols;
	uint32_t b;

	if (f_v) {
		cout << "wreath_product::report_rank_one_tensors" << endl;
		cout << "wreath_product::report_rank_one_tensors "
				"dimension_of_tensor_action=" << dimension_of_tensor_action << endl;
	}


	dimensions(dimension_of_tensor_action, nb_rows, nb_cols);
	if (f_v) {
		cout << "wreath_product::report_rank_one_tensors nb_rows = " << nb_rows << endl;
		cout << "wreath_product::report_rank_one_tensors nb_cols = " << nb_cols << endl;
	}


	coords = NEW_int(nb_factors);
	Proj = NEW_int(nb_factors);
	projections = NEW_int(dimension_of_matrix_group * nb_factors);
	projections1 = NEW_int(dimension_of_matrix_group * nb_factors);
	tensor = NEW_int(dimension_of_tensor_action);
	T = NEW_int(dimension_of_tensor_action);
	N = NT.i_power_j(q, dimension_of_matrix_group) - 1;
	nb_rank_one_tensors = NT.i_power_j(N, nb_factors);
	if (f_v) {
		cout << "wreath_product::create_all_rank_one_tensors "
				"nb_rank_one_tensors = " << nb_rank_one_tensors << endl;
	}
	rank_one_tensors = (uint32_t *) NEW_int(nb_rank_one_tensors);
	len = dimension_of_matrix_group * nb_factors;

	ost << "{\\renewcommand{\\arraystretch}{1.1}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|r|r|r|}" << endl;
	ost << "\\hline" << endl;

	for (i = 0; i < nb_rank_one_tensors; i++) {

		if (i && (i % 10) == 0) {
			ost << "\\end{array}" << endl;
			ost << "$$" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|r|r|r|r|r|r|}" << endl;
			ost << "\\hline" << endl;
		}
		ost << i;

		Gg.AG_element_unrank(N, Proj, 1, nb_factors, i);
		for (j = 0; j < nb_factors; j++) {
			Gg.AG_element_unrank(q, projections + j * dimension_of_matrix_group,
					1, dimension_of_matrix_group, Proj[j] + 1);
		}
		F->Linear_algebra->transpose_matrix(
				projections, projections1, nb_factors,
				dimension_of_matrix_group);

		ost << " & " << endl;
		Int_vec_print(ost, Proj, nb_factors);
		ost << " & " << endl;

		ost << "\\left[" << endl;
		L.int_matrix_print_tex(ost, projections1,
				dimension_of_matrix_group, nb_factors);
		ost << "\\right]" << endl;


		if (f_vv) {
			cout << "wreath_product::create_all_rank_one_tensors " << i
					<< " / " << nb_rank_one_tensors << ":" << endl;
			cout << "projections: ";
			for (j = 0; j < len; j++) {
				cout << projections[j] << " ";
			}
			cout << endl;
		}

		Int_vec_zero(T, dimension_of_tensor_action);
		for (j = 0; j < dimension_of_tensor_action; j++) {
			Gg.AG_element_unrank(dimension_of_matrix_group, coords, 1, nb_factors, j);
			a = 1;
			for (h = 0; h < nb_factors; h++) {
				a = F->mult(a, projections[h * dimension_of_matrix_group + coords[h]]);
			}
			tensor[j] = a;
			if (a) {
				int u, v;
				place_binary(j, u, v);
				T[u * nb_cols + v] = 1;
			}
		}
		if (f_vv) {
			cout << "wreath_product::create_all_rank_one_tensors " << i
					<< " / " << nb_rank_one_tensors << ":" << endl;
			cout << "tensor: ";
			for (j = 0; j < dimension_of_tensor_action; j++) {
				cout << tensor[j] << " ";
			}
			cout << endl;
		}


		b = tensor[dimension_of_tensor_action - 1];
		for (j = 1; j < dimension_of_tensor_action; j++) {
			b <<= 1;
			if (tensor[dimension_of_tensor_action - 1 - j]) {
				b++;
			}
		}

		ost << " & " << endl;
		ost << "\\left[" << endl;
		L.int_matrix_print_tex(ost, T, nb_rows, nb_cols);
		ost << "\\right]" << endl;

		ost << " & ";
		ost << b;
		//ost << rank_one_tensors[i];
		ost << " & ";
		ost << rank_one_tensors_in_PG[i];
		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$}" << endl;

	FREE_int(coords);
	FREE_int(Proj);
	FREE_int(projections);
	FREE_int(projections1);
	FREE_int(tensor);
	FREE_int(T);

}


static void dimensions(int n, int &nb_rows, int &nb_cols)
{
	int i, j;

	place_binary(n - 1, i, j);
	nb_rows = i + 1;
	nb_cols = j + 1;
}

static void place_binary(long int h, int &i, int &j)
{
	int o[2];
	int c;

	o[0] = 1;
	o[1] = 0;
	i = 0;
	j = 0;
	for (c = 0; h; c++) {
		if (h % 2) {
			i += o[0];
			j += o[1];
		}
		h >>= 1;
		if (c % 2) {
			o[0] = o[1] << 1;
			o[1] = 0;
		}
		else {
			o[1] = o[0];
			o[0] = 0;
		}
	}
}



void wreath_product::create_all_rank_one_tensors(
		uint32_t *&rank_one_tensors,
		int &nb_rank_one_tensors, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	combinatorics::combinatorics_domain Combi;
	number_theory::number_theory_domain NT;
	geometry::geometry_global Gg;
	int *coords;
	int *Proj;
	int *projections;
	int *tensor;
	int i, j, h, a, len, N;
	uint32_t b;

	if (f_v) {
		cout << "wreath_product::create_all_rank_one_tensors" << endl;
	}
	if (q != 2) {
		cout << "wreath_product::create_all_rank_one_tensors requires q == 2" << endl;
		exit(1);
	}
	coords = NEW_int(nb_factors);
	Proj = NEW_int(nb_factors);
	projections = NEW_int(dimension_of_matrix_group * nb_factors);
	tensor = NEW_int(dimension_of_tensor_action);
	N = NT.i_power_j(q, dimension_of_matrix_group) - 1;
	nb_rank_one_tensors = NT.i_power_j(N, nb_factors);
	if (f_v) {
		cout << "wreath_product::create_all_rank_one_tensors "
				"nb_rank_one_tensors = " << nb_rank_one_tensors << endl;
	}
	rank_one_tensors = (uint32_t *) NEW_int(nb_rank_one_tensors);
	len = dimension_of_matrix_group * nb_factors;
	for (i = 0; i < nb_rank_one_tensors; i++) {

		Gg.AG_element_unrank(N, Proj, 1, nb_factors, i);
		for (j = 0; j < nb_factors; j++) {
			Gg.AG_element_unrank(q, projections + j * dimension_of_matrix_group,
					1, dimension_of_matrix_group, Proj[j] + 1);
		}
		if (f_vv) {
			cout << "wreath_product::create_all_rank_one_tensors " << i
					<< " / " << nb_rank_one_tensors << ":" << endl;
			cout << "projections: ";
			for (j = 0; j < len; j++) {
				cout << projections[j] << " ";
			}
			cout << endl;
		}

		for (j = 0; j < dimension_of_tensor_action; j++) {
			Gg.AG_element_unrank(dimension_of_matrix_group, coords, 1, nb_factors, j);
			a = 1;
			for (h = 0; h < nb_factors; h++) {
				a = F->mult(a, projections[h * dimension_of_matrix_group + coords[h]]);
			}
			tensor[j] = a;
		}
		if (f_vv) {
			cout << "wreath_product::create_all_rank_one_tensors " << i
					<< " / " << nb_rank_one_tensors << ":" << endl;
			cout << "tensor: ";
			for (j = 0; j < dimension_of_tensor_action; j++) {
				cout << tensor[j] << " ";
			}
			cout << endl;
		}
		b = tensor[dimension_of_tensor_action - 1];
		for (j = 1; j < dimension_of_tensor_action; j++) {
			b <<= 1;
			if (tensor[dimension_of_tensor_action - 1 - j]) {
				b++;
			}
		}
		rank_one_tensors[i] = b;
	}
	rank_one_tensors_in_PG = NEW_lint(nb_rank_one_tensors);
	rank_one_tensors_in_PG_sorted = NEW_lint(nb_rank_one_tensors);
	for (i = 0; i < nb_rank_one_tensors; i++) {
		rank_one_tensors_in_PG[i] = affine_rank_to_PG_rank(rank_one_tensors[i]);
	}
	Lint_vec_copy(rank_one_tensors_in_PG, rank_one_tensors_in_PG_sorted, nb_rank_one_tensors);

	data_structures::sorting Sorting;

	Sorting.lint_vec_heapsort(rank_one_tensors_in_PG_sorted, nb_rank_one_tensors);

	FREE_int(coords);
	FREE_int(Proj);
	FREE_int(projections);
	FREE_int(tensor);
	if (f_v) {
		cout << "wreath_product::create_all_rank_one_tensors done" << endl;
	}
}

uint32_t wreath_product::tensor_affine_rank(int *tensor)
{
	uint32_t b;
	int i;

	b = tensor[dimension_of_tensor_action - 1];
	for (i = 1; i < dimension_of_tensor_action; i++) {
		b <<= 1;
		if (tensor[dimension_of_tensor_action - 1 - i]) {
			b++;
		}
	}
	return b;
}

void wreath_product::tensor_affine_unrank(int *tensor, uint32_t rk)
{
	uint32_t b;
	int i;

	Int_vec_zero(tensor, dimension_of_tensor_action);

	b = rk;
	for (i = 0; i < dimension_of_tensor_action; i++) {
		if (b % 2) {
			tensor[i] = 1;
		}
		b >>= 1;
	}
}

long int wreath_product::tensor_PG_rank(int *tensor)
{
	long int b;

	F->PG_element_rank_modified_lint(tensor, 1, dimension_of_tensor_action, b);
	return b;
}

void wreath_product::tensor_PG_unrank(int *tensor, long int PG_rk)
{
	F->PG_element_unrank_modified_lint(tensor, 1, dimension_of_tensor_action, PG_rk);
}

long int wreath_product::affine_rank_to_PG_rank(uint32_t affine_rk)
{
	long int b;

	tensor_affine_unrank(u, affine_rk);
	F->PG_element_rank_modified_lint(u, 1, dimension_of_tensor_action, b);
	return b;
}

uint32_t wreath_product::PG_rank_to_affine_rank(long int PG_rk)
{
	uint32_t b;

	F->PG_element_unrank_modified_lint(u, 1, dimension_of_tensor_action, PG_rk);
	b = tensor_affine_rank(u);
	return b;
}

void wreath_product::save_rank_one_tensors(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::save_rank_one_tensors" << endl;
	}
	char fname[1000];

	snprintf(fname, sizeof(fname), "rank_one_tensors_q%d_f%d.txt", q, nb_factors);
	{
		ofstream fp(fname);
		int i;

		fp << nb_rank_one_tensors << endl;
		for (i = 0; i < nb_rank_one_tensors; i++) {
			fp << rank_one_tensors[i] << " ";
		}
		fp << endl;
		fp << -1 << endl;
	}
	if (f_v) {
		cout << "wreath_product::save_rank_one_tensors done" << endl;
	}
}

void wreath_product::compute_tensor_ranks(char *&TR, uint32_t *&Prev, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	long int i;
	int r;
	long int sz;
	long int nb_processed;
	std::deque<uint32_t> D;
	uint32_t a, b, c;
	long int one_percent;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "wreath_product::compute_tensor_ranks" << endl;
	}


	char fname1[1000];
	char fname2[1000];

	snprintf(fname1, sizeof(fname1), "tensor_q%d_w%d_ranks.bin", q, nb_factors);
	snprintf(fname2, sizeof(fname2), "tensor_q%d_w%d_ranks_prev.bin", q, nb_factors);


	if (f_v) {
		cout << "wreath_product::compute_tensor_ranks nb_rank_one_tensors = " << nb_rank_one_tensors << endl;
	}




	Prev = (uint32_t *) NEW_int(degree_of_tensor_action + 1);
	TR = NEW_char(degree_of_tensor_action + 1);


	if (Fio.file_size(fname1) > 0 && Fio.file_size(fname2) > 0) {
		cout << "reading tensor ranks from file" << endl;

		{
			ifstream fp(fname1, ios::binary);

			long int d;

			fp.read((char *) &d, sizeof(long int));
			if (d != degree_of_tensor_action + 1) {
				cout << "d != degree_of_tensor_action + 1" << endl;
				exit(1);
			}
			for (i = 0; i < d; i++) {
				fp.read((char *) &TR [i], sizeof(char));
			}
		}
		cout << "reading tensor ranks from file done" << endl;

		cout << "reading Prev from file:" << endl;
		{
			ifstream fp(fname2, ios::binary);

			long int d;

			fp.read((char *) &d, sizeof(long int));
			if (d != degree_of_tensor_action + 1) {
				cout << "d != degree_of_tensor_action + 1" << endl;
				exit(1);
			}
			for (i = 0; i < d; i++) {
				fp.read((char *) &Prev [i], sizeof(uint32_t));
			}
		}
		cout << "reading Prev from file done" << endl;

	}
	else {
		cout << "computing tensor ranks:" << endl;

		for (i = 0; i < degree_of_tensor_action + 1; i++) {
			TR[i] = -1;
		}
		D.push_back((uint32_t) 0);
		TR[0] = 0;
		Prev[0] = 0;
		sz = 1;
		nb_processed = 0;

		one_percent = (int)((double)(degree_of_tensor_action + 1) / (double)100) + 1;

		while (sz < degree_of_tensor_action + 1) {
			if (D.empty()) {
				cout << "wreath_product::compute_tensor_ranks sz < degree_of_tensor_action + 1 and D.empty()" << endl;
				cout << "sz=" << sz << endl;
				exit(1);
			}
			a = D.front();
			D.pop_front();
			r = TR[a];
			if (f_vv) {
				cout << "wreath_product::compute_tensor_ranks expanding " << a << " of rank " << r << ", sz=" << sz << endl;
			}

			for (i = 0; i < nb_rank_one_tensors; i++) {
				b = rank_one_tensors[i];
				c = a ^ b;
				if (f_vv) {
					cout << "wreath_product::compute_tensor_ranks expanding generator " << i << " = " << b << " maps " << a << " to " << c << endl;
				}
				if (TR[c] == -1) {
					if (f_vv) {
						cout << "wreath_product::compute_tensor_ranks expanding generator setting tensor rank of " << c << " to " << r + 1 << endl;
					}
					TR[c] = r + 1;
					Prev[c] = a;
					sz++;
					D.push_back(c);
					if (sz % one_percent == 0) {
						cout << "wreath_product::compute_tensor_ranks "
								<< sz / one_percent << " % of tree completed, size of "
								"queue is " << D.size() << " = "
								<< (D.size() / (double)(degree_of_tensor_action + 1)) * 100. << " %" << endl;
					}
				}
				else {
					if (f_vv) {
						cout << "wreath_product::compute_tensor_ranks expanding generator setting tensor rank of " << c << " is " << (int) TR[c] << " skipping" << endl;
					}

				}
			} // next i
			nb_processed++;
			if (nb_processed % one_percent == 0) {
				cout << "wreath_product::compute_tensor_ranks "
						<< nb_processed / one_percent << " % processed, size of "
						"queue is " << D.size() << " = "
						<< (D.size() / (double)(degree_of_tensor_action + 1)) * 100.
						<< " % tree at " << sz / one_percent << " %" << endl;
			}
		}
		if (f_vv) {
			cout << "wreath_product::compute_tensor_ranks TR:" << endl;
			for (i = 0; i < degree_of_tensor_action + 1; i++) {
				cout << i << " : " << (int) TR[i] << endl;
			}
		}


		cout << "computing tensor ranks done." << endl;


		cout << "writing TR to file:" << endl;
		{
			ofstream fp(fname1, ios::binary);

			long int d;

			d = degree_of_tensor_action + 1;
			fp.write((char *) &d, sizeof(long int));
			for (i = 0; i < d; i++) {
				fp.write((char *) &TR [i], sizeof(char));
			}
		}

		cout << "writing Prev to file:" << endl;
		{
			ofstream fp(fname2, ios::binary);

			long int d;

			d = degree_of_tensor_action + 1;
			fp.write((char *) &d, sizeof(long int));
			for (i = 0; i < d; i++) {
				fp.write((char *) &Prev [i], sizeof(uint32_t));
			}
		}
	}


	if (f_v) {
		cout << "computing maximum tensor rank:" << endl;
	}
	int m;

	m = 0;
	for (i = 0; i < degree_of_tensor_action + 1; i++) {
		m = MAXIMUM(m, (int) TR[i]);
	}
	if (f_v) {
		cout << "wreath_product::compute_tensor_ranks max tensor rank = " << m << endl;
	}
	long int *Nb_by_rank;

	if (f_v) {
		cout << "computing Nb_by_rank:" << endl;
	}
	Nb_by_rank = NEW_lint(m + 1);
	for (i = 0; i <= m; i++) {
		Nb_by_rank[i] = 0;
	}
	for (i = 0; i < degree_of_tensor_action + 1; i++) {
		r = (int) TR[i];
		Nb_by_rank[r]++;
	}
	if (f_v) {
		cout << "number of tensors by rank:" << endl;
		for (i = 0; i <= m; i++) {
			cout << i << " : " << Nb_by_rank[i] << endl;
		}
	}




	if (q == 2 && nb_factors == 5) {

		if (f_v) {
			cout << "q == 2 && nb_factors == 5" << endl;
		}

		knowledge_base K;


		int N = K.tensor_orbits_nb_reps(nb_factors);
		int *R;
		int *L;
		uint32_t *S;
		uint32_t s;
		int len;

		R = NEW_int(N);
		L = NEW_int(N);
		S = (uint32_t *) NEW_int(N);
		for (i = 0; i < N; i++) {
			a = K.tensor_orbits_rep(nb_factors, i)[1];
			len = K.tensor_orbits_rep(nb_factors, i)[2];
			//a = w5_reps[3 * i + 1];
			//len = w5_reps[3 * i + 2];
			s = PG_rank_to_affine_rank(a);
			S[i] = s;
			L[i] = len;
			R[i] = (int) TR[s];
		}

		cout << "tensor ranks of orbit representatives:" << endl;
		cout << "i : orbit length[i] : PG rank of rep[i] : tensor rank [i] : affine rank of rep(i) : rank one decomposition" << endl;
		for (i = 0; i < N; i++) {
			a = K.tensor_orbits_rep(nb_factors, i)[1];
			//a = w5_reps[3 * i + 1];
			cout << i << " : " << L[i] << " : " << a << " : " << R[i] << " : " << S[i] << " : ";
			s = S[i];
			while (true) {
				cout << (Prev[s] ^ s);
				s = Prev[s];
				cout << ",";
				if (s == 0) {
					break;
				}
				else {
				}
			}

			cout << endl;
		}

		data_structures::tally C;
		data_structures::set_of_sets *SoS;
		int *types;
		int nb_types;

		C.init(R, N, FALSE, 0);

		SoS = C.get_set_partition_and_types(types,
				nb_types, verbose_level);

		cout << "classification of orbit reps by tensor rank:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		for (int t = 0; t < nb_types; t++) {
			cout << "working on type " << t << " of value " << types[t] << ":" << endl;
			cout << "There are " << SoS->Set_size[t] << " orbits" << endl;
			int *L;
			int *Ago;

			L = NEW_int(SoS->Set_size[t]);
			Ago = NEW_int(SoS->Set_size[t]);
			for (s = 0; s < SoS->Set_size[t]; s++) {
				a = SoS->Sets[t][s];
				L[s] = K.tensor_orbits_rep(nb_factors, a)[2];
				//L[s] = w5_reps[3 * a + 2];
				Ago[s] = 933120 / L[s];
			}
			data_structures::tally C1;
			data_structures::tally C2;

			C1.init(L, SoS->Set_size[t], FALSE, 0);
			cout << "classification of orbit lengths for tensor rank " << types[t] << ":" << endl;
			C1.print_naked_tex(cout, TRUE);
			cout << endl;

			C2.init(Ago, SoS->Set_size[t], FALSE, 0);
			cout << "classification of ago for tensor rank " << types[t] << ":" << endl;
			C2.print_naked_tex(cout, TRUE);
			cout << endl;

			FREE_int(L);
			FREE_int(Ago);

		}

		FREE_int(v);
	}

	else if (q == 2 && nb_factors == 4) {

		if (f_v) {
			cout << "q == 2 && nb_factors == 4" << endl;
		}

		knowledge_base K;
		int N = K.tensor_orbits_nb_reps(nb_factors);
		int *R;
		int *L;
		uint32_t *S;
		uint32_t s;
		int len, ago;

		R = NEW_int(N);
		L = NEW_int(N);
		S = (uint32_t *) NEW_int(N);
		for (i = 0; i < N; i++) {
			a = K.tensor_orbits_rep(nb_factors, i)[1];
			len = K.tensor_orbits_rep(nb_factors, i)[2];
			//a = w4_reps[3 * i + 1];
			//len = w4_reps[3 * i + 2];
			s = PG_rank_to_affine_rank(a);
			S[i] = s;
			L[i] = len;
			R[i] = (int) TR[s];
			cout << "R[i]=" << R[i] << endl;
		}

		cout << "tensor ranks of orbit representatives:" << endl;
		cout << "i : orbit length[i] : ago : PG rank of rep[i] : tensor rank [i] : affine rank of rep(i) : rank one decomposition" << endl;
		for (i = 0; i < N; i++) {
			a = K.tensor_orbits_rep(nb_factors, i)[1];
			//a = w4_reps[3 * i + 1];
			ago = 31104 / L[i];
			cout << i << " : " << L[i] << " : " << ago << " : " << a << " : " << R[i] << " : " << S[i] << " : ";
			s = S[i];
			while (true) {
				cout << (Prev[s] ^ s);
				s = Prev[s];
				cout << ",";
				if (s == 0) {
					break;
				}
				else {
				}
			}

			cout << endl;
		}

		data_structures::tally C;
		data_structures::set_of_sets *SoS;
		int *types;
		int nb_types;

		C.init(R, N, FALSE, 0);

		SoS = C.get_set_partition_and_types(types,
				nb_types, verbose_level);

		cout << "classification of orbit reps by tensor rank:" << endl;
		C.print_naked(TRUE);
		cout << endl;
		for (int t = 0; t < nb_types; t++) {
			cout << "working on type " << t << " of value " << types[t] << ":" << endl;
			cout << "There are " << SoS->Set_size[t] << " orbits" << endl;
			int *L;
			int *Ago;

			L = NEW_int(SoS->Set_size[t]);
			Ago = NEW_int(SoS->Set_size[t]);
			for (s = 0; s < SoS->Set_size[t]; s++) {
				a = SoS->Sets[t][s];
				L[s] = K.tensor_orbits_rep(nb_factors, a)[2];
				//L[s] = w4_reps[3 * a + 2];
				Ago[s] = 31104 / L[s];
			}
			data_structures::tally C1;
			data_structures::tally C2;

			C1.init(L, SoS->Set_size[t], FALSE, 0);
			cout << "classification of orbit lengths for tensor rank " << types[t] << ":" << endl;
			C1.print_naked_tex(cout, TRUE);
			cout << endl;

			C2.init(Ago, SoS->Set_size[t], FALSE, 0);
			cout << "classification of ago for tensor rank " << types[t] << ":" << endl;
			C2.print_naked_tex(cout, TRUE);
			cout << endl;

			FREE_int(L);
			FREE_int(Ago);

		}

		FREE_int(v);
	}

	//exit(1);



	if (f_v) {
		cout << "wreath_product::compute_tensor_ranks done" << endl;
	}
}

void wreath_product::report(ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::report" << endl;
	}
	ost << "\\section*{Wreath product group}" << endl << endl;


	ost << "Group name: $" << label_tex << "$\\\\" << endl;
	ost << "Number of factors: " << nb_factors << "\\\\" << endl;
	ost << "Degree of matrix group: " << degree_of_matrix_group << "\\\\" << endl;
	ost << "Dimension of matrix group: " << dimension_of_matrix_group << "\\\\" << endl;
	ost << "Dimension of tensor action: " << dimension_of_tensor_action << "\\\\" << endl;
	ost << "Degree of tensor action: " << degree_of_tensor_action << "\\\\" << endl;
	ost << "Degree overall: " << degree_overall << "\\\\" << endl;
	ost << "Low level point size: " << low_level_point_size << "\\\\" << endl;
	ost << "Make element size: " << make_element_size << "\\\\" << endl;
	ost << "Number of rank one tensors: " << nb_rank_one_tensors << "\\\\" << endl;

	ost << "\\bigskip" << endl << endl;

	ost << "\\section*{Rank One Tensors}" << endl << endl;


#if 0
	//ost << "Rank one tensors: \\\\" << endl;

	int i;
	uint32_t a, b;
	int *tensor;

	tensor = NEW_int(dimension_of_tensor_action);

	for (i = 0; i < nb_rank_one_tensors; i++) {
		a = rank_one_tensors[i];
		b = affine_rank_to_PG_rank(a);
		tensor_PG_unrank(tensor, b);
		ost << i << " : ";
		int_vec_print(ost, tensor, dimension_of_tensor_action);
		ost << " : " << a << " : " << b << "\\\\" << endl;
	}

	FREE_int(tensor);
#endif


	report_rank_one_tensors(ost, verbose_level);

	//ost << "\\subsection*{The underlying matrix group}" << endl << endl;
	//A_mtx->report(ost, verbose_level);


	if (f_v) {
		cout << "wreath_product::report done" << endl;
	}
}

void wreath_product::compute_permutations_and_write_to_file(
		strong_generators* SG,
		actions::action *A,
		int*& result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	int *generator_stack;
	int **generators_transposed;
	int *perms;
	int mtx_n;
	int mtx_n2;
	int h;

	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file" << endl;
	}

	nb_gens = SG->gens->len;
	degree = degree_of_tensor_action;
	mtx_n = dimension_of_tensor_action;
	mtx_n2 = mtx_n * mtx_n;

	generator_stack = NEW_int(SG->gens->len * mtx_n2);
	generators_transposed = NEW_pint(SG->gens->len);
	perms = NEW_int(SG->gens->len * mtx_n);
	for (h = 0; h < SG->gens->len; h++) {
		if (f_v) {
			cout << "generator " << h << " / "
					<< SG->gens->len << " is: " << endl;
			A->element_print_quick(SG->gens->ith(h), cout);
			A->element_print_as_permutation(SG->gens->ith(h), cout);
		}

		create_matrix(SG->gens->ith(h), generator_stack + h * mtx_n2,
				0 /* verbose_level */);

		if (f_v) {
			cout << "wreath_product::compute_permutations_and_write_to_file matrix:" << endl;
			Int_matrix_print(generator_stack + h * mtx_n2, mtx_n, mtx_n);
		}
		generators_transposed[h] = NEW_int(mtx_n2);

		F->Linear_algebra->transpose_matrix(
				generator_stack + h * mtx_n2,
				generators_transposed[h], mtx_n, mtx_n);

		compute_induced_permutation(SG->gens->ith(h), perms + h * mtx_n);
	}

	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file generator_stack:" << endl;
		Int_matrix_print(generator_stack, SG->gens->len, mtx_n * mtx_n);
	}

#if 0
	cout << "wreath_product::compute_permutations_and_write_to_file generators transposed:" << endl;
	for (size_t h = 0; h < SG->gens->len; h++) {
		int_matrix_print(generators_transposed[h], mtx_n, mtx_n);
	}
#endif
	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file perms:" << endl;
		Int_matrix_print(perms, SG->gens->len, mtx_n);
		cout << "mtx_n=" << mtx_n << endl;
		cout << "SG->gens->len * mtx_n=" << SG->gens->len * mtx_n << endl;
	}

#if 0
	linalg::Matrix<int> v (mtx_n, 1);


	// matrix N contains the matrices of all projectivities
	// which generate the group, stacked on top of each other.
	// So, N has size (SG->gens->len * mtx_n) x mtx_n


	vector<linalg::Matrix<char>> N (SG->gens->len);
	for (size_t h = 0; h < N.size(); ++h) {
		N[h].INIT(mtx_n, mtx_n);

		for (size_t i=0; i < mtx_n; ++i)
			for (size_t j = 0; j < mtx_n; ++j) {
				N[h].matrix_[i*mtx_n+j] = generator_stack [h * mtx_n2 + i * mtx_n + j];
			}

	}

	// Print the matrices N
	for (size_t h=0; h<N.size(); ++h) {
		printf("=========================================================\n");
		printf("h = %ld\n", h);
		printf("=========================================================\n");

		linalg::print(N[h]);

		printf("=========================================================\n");
	}
#endif


	// result is the ranks of the images.
	// Each row of result is a permutation of the points of projective space
	// So, result is SG->gens->len x W->degree_of_tensor_action

	//result = NEW_int(SG->gens->len * W->degree_of_tensor_action);

	// perform the parallel matrix multiplication on the GPU:


//	int* v = NEW_int (MN.ncols);

	unsigned int w = (unsigned int) degree_of_tensor_action - 1;
	long int a;
	a = (long int) w;
	if (a != degree_of_tensor_action - 1) {
		cout << "W->degree_of_tensor_action - 1 does not fit into a unsigned int" << endl;
		exit(1);
	}
	else {
		if (f_v) {
			cout << "W->degree_of_tensor_action fits into a unsigned int, this is good" << endl;
		}
	}



	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file block_size=" << block_size << endl;
	}

	int nb_blocks = (degree_of_tensor_action + block_size - 1) / block_size;

	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file nb_blocks=" << nb_blocks << endl;
	}


	//cout << "allocating S, an unsigned int array of size " << W->degree_of_tensor_action << endl;

	//unsigned int* S = new unsigned int [W->degree_of_tensor_action];

	//for (unsigned int i=0; i<W->degree_of_tensor_action; ++i) S[i] = i;


	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file before allocating T, "
				"an unsigned int array of size " << block_size << endl;
	}

	unsigned int* T = new unsigned int [block_size];

//	memset(S, -1, sizeof(S)*W->degree_of_tensor_action);
	if (f_v) {
		cout << "wreath_product::compute_permutations_and_write_to_file after allocating T, "
				"an unsigned int array of size " << block_size << endl;
	}




	long int b;

	for (b = 0; b < nb_blocks; ++b) {
		if (f_v) {
			cout << "wreath_product::compute_permutations_and_write_to_file block b=" << b << " / " << nb_blocks << endl;
		}


		long int l;

		l = MINIMUM((b + 1) * block_size, degree_of_tensor_action) - b * block_size;
		if (f_v) {
			cout << "wreath_product::compute_permutations_and_write_to_file l=" << l << endl;
		}

		//linalg::Matrix<char> M  (l, mtx_n);

		data_structures::bitmatrix *M;

		M = NEW_OBJECT(data_structures::bitmatrix);
		M->init(mtx_n, l, 0 /*verbose_level*/);

		if (f_v) {
			cout << "wreath_product::compute_permutations "
					"unranking the elements of the PG to the bitmatrix" << endl;
		}
		M->unrank_PG_elements_in_columns_consecutively(
				F, (long int) b * (long int) block_size,
				0 /* verbose_level */);


#if 0
		cout << "wreath_product::compute_permutations_and_write_to_file unranking the elements of the PG" << endl;

		int l1 = l / 100;
		for (size_t i=0; i<l; ++i) {
			if ((i % l1) == 0) {
				cout << "block b=" << b << ", " << i / l1 << " % done unranking" << endl;
			}
			W->F->PG_element_unrank_modified_lint (v.matrix_, 1, mtx_n,
					(long int) b * (long int) block_size + (long int)i) ;
			for (size_t j=0; j<mtx_n; ++j)
				M(i,j) = v(j, 0);
		}
#endif

		if (f_v) {
			cout << "wreath_product::compute_permutations_and_write_to_file "
					"unranking the elements of the PG done" << endl;
		}

		//M->print();

		//linalg::Matrix<char> MN (l, mtx_n);

		data_structures::bitmatrix *NM;

		NM = NEW_OBJECT(data_structures::bitmatrix);
		NM->init(mtx_n, l, 0 /*verbose_level*/);


		for (h = 0; h < SG->gens->len; ++h) {
			if (f_v) {
				cout << "wreath_product::compute_permutations_and_write_to_file "
						"generator h=" << h << " / " << SG->gens->len << endl;
			}


			if (!test_if_file_exists(nb_factors, h, b)) {


				// Matrix Multiply
				//MN.reset_entries();
				NM->zero_out();



				//cout << "cuda multiplication" << endl;
				//linalg::cuda_mod_mat_mul (M, N[h], MN, W->q);
				//cout << "cuda multiplication done" << endl;
				//M.UninitializeOnGPU();
				//N[h].UninitializeOnGPU();
				//MN.UninitializeOnGPU();


				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"before CPU multiplication" << endl;
				}
				int t0, t1, dt;
				orbiter_kernel_system::os_interface Os;
				t0 = Os.os_ticks();
				//linalg::cpu_mod_mat_mul_block_AB(M, N[h], MN, W->q);
				M->mult_int_matrix_from_the_left(
						generators_transposed[h], mtx_n, mtx_n,
						NM, verbose_level);
				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"after CPU multiplication" << endl;
					t1 = Os.os_ticks();
					dt = t1 - t0;
					cout << "the multiplication took ";
					Os.time_check_delta(cout, dt);
					cout << endl;
				}

				//cout << "NM:" << endl;
				//NM->print();


				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"before ranking the elements of the PG" << endl;
				}
				NM->rank_PG_elements_in_columns(
						F, perms + h * mtx_n, T,
						verbose_level);

#if 0
				for (size_t i=0; i<l; ++i) {
					if ((i % l1) == 0) {
						cout << "h=" << h << ", b=" << b << ", " << i/l1 << " % done ranking" << endl;
					}
					for (size_t j=0; j<mtx_n; ++j) {
						int a = perms[h * mtx_n + j];
						v.matrix_[a*v.alloc_cols] = MN (i, j);

					}
					long int res;
					W->F->PG_element_rank_modified_lint (v.matrix_, 1, mtx_n, res);
					T [i] = (unsigned int) res;
				}
#endif
				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"after ranking the elements of the PG" << endl;
				}


				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"writing to file:" << endl;
				}
				std::string fname;

				make_fname(fname, nb_factors, h, b);
				{
					ofstream fp(fname, ios::binary);

					fp.write((char *) &l, sizeof(int));
					for (int i = 0; i < l; i++) {
						fp.write((char *) &T [i], sizeof(int));
					}
				}
				//file_io Fio;

				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"written file " << fname << endl;
					//" of size " << Fio.file_size(fname) << endl;
				}


			}
			else {
				if (f_v) {
					cout << "wreath_product::compute_permutations_and_write_to_file "
							"the case h=" << h << ", b=" << b
							<< " has already been done" << endl;
				}
			}

		} // next h

		FREE_OBJECT(M);
		FREE_OBJECT(NM);


	} // next b

#if 0
	int nb_orbits = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) ++nb_orbits;
	}
	cout << "nb_orbits: " << nb_orbits << endl;

	long int *orbit_length;
	long int *orbit_rep;

	orbit_length = NEW_lint(nb_orbits);
	orbit_rep = NEW_lint(nb_orbits);

	for (int i = 0; i < nb_orbits; i++) {
		orbit_length[i] = 0;
	}
	int j;
	j = 0;
	for (unsigned int i=0; i < W->degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			orbit_rep[j++] = i;
		}
	}

	cout << "the orbit representatives are: " << endl;
	for (int i = 0; i < nb_orbits; i++) {
		cout << i << " : " << orbit_rep[i] << endl;
	}
#endif



//	combinatorics_domain Combi;
//
//	for (size_t i = 0; i < SG->gens->len; i++) {
//		cout << "testing result " << i << " / " << SG->gens->len << ": ";
//		if (Combi.is_permutation(
//				result + i * W->degree_of_tensor_action,
//				W->degree_of_tensor_action)) {
//			cout << "OK" << endl;
//		}
//		else {
//			cout << "not OK" << endl;
//		}
//	}
//	cout << "We found " << SG->gens->len << " permutations of "
//			"degree " << W->degree_of_tensor_action << endl;
//
//
//	cout << __FILE__ << ":" << __LINE__ << endl;
//	//exit(0);
//
//	FREE_int(generator_stack);
//	FREE_int(perms);
//	cout << "wreath_product_orbits_CUDA done" << endl;


//#else
//	nb_gens = 0;
//	degree = 0;
//#endif

	if (f_v) {
		cout << "wreath_product::compute_permutations done" << endl;
	}

}

void wreath_product::make_fname(std::string &fname, int nb_factors, int h, int b)
{
	char str[1000];
	snprintf(str, sizeof(str), "w%d_h%d_b%d.bin", nb_factors, h, b);
	fname.assign(str);
}

int wreath_product::test_if_file_exists(int nb_factors, int h, int b)
{
	std::string fname;
	orbiter_kernel_system::file_io Fio;

	make_fname(fname, nb_factors, h, b);
	if (Fio.file_size(fname) > 0) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void wreath_product::orbits_using_files_and_union_find(
		strong_generators* SG,
		actions::action *A,
		int *&result,
		int &nb_gens, int &degree,
		int nb_factors,
		int verbosity)
{
	int f_v = (verbosity >= 1);

	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find" << endl;
	}
	long int i, b, h;
	long int j, r, orbit_idx, rep;
	long int nb_orbits = 0;
	data_structures::algorithms Algo;

	//int mtx_n;

	nb_gens = SG->gens->len;
	degree = degree_of_tensor_action;
	//mtx_n = dimension_of_tensor_action;

	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	if (f_v) {
		cout << "block_size=" << block_size << endl;
	}

	int nb_blocks = (degree_of_tensor_action + block_size - 1) / block_size;

	if (f_v) {
		cout << "nb_blocks=" << nb_blocks << endl;
	}


	if (f_v) {
		cout << "allocating S, an unsigned int array of size "
				<< degree_of_tensor_action << endl;
	}

	unsigned int* S = new unsigned int [degree_of_tensor_action];

	for (i = 0; i < degree_of_tensor_action; ++i) {
		S[i] = i;
	}


	if (f_v) {
		cout << "allocating T, an unsigned int array of size "
				<< degree_of_tensor_action << endl;
	}

	unsigned int* T = new unsigned int [degree_of_tensor_action];





	for (h = 0; h < SG->gens->len; ++h) {
		if (f_v) {
			cout << "wreath_product::orbits_using_files_and_union_find "
					"generator h=" << h << " / " << SG->gens->len << endl;
		}

		for (b = 0; b < nb_blocks; ++b) {
			if (f_v) {
				cout << "wreath_product::orbits_using_files_and_union_find "
						"block b=" << b << " / " << nb_blocks << endl;
			}


			long int l = MINIMUM((b + 1) * block_size, degree_of_tensor_action) - b * block_size;
			if (f_v) {
				cout << "wreath_product::orbits_using_files_and_union_find "
						"l=" << l << endl;
			}




			if (!test_if_file_exists(nb_factors, h, b)) {
				cout << "file does not exist h=" << h << " b=" << b << endl;
				exit(1);
			}
			else {
				std::string fname;

				make_fname(fname, nb_factors, h, b);
				if (f_v) {
					cout << "wreath_product::orbits_using_files_and_union_find "
							"reading from file " << fname << endl;
				}
				{
					ifstream fp(fname, ios::binary);

					int l1;
					fp.read((char *) &l1, sizeof(int));
					if (l1 != l) {
						cout << "l1 != l" << endl;
					}
					for (int i = 0; i < l; i++) {
						fp.read((char *) &T [b * block_size + i], sizeof(int));
					}
				}
				//file_io Fio;

				if (f_v) {
					cout << "wreath_product::orbits_using_files_and_union_find "
							"read file " << fname << endl;
					//" of size " << Fio.file_size(fname) << endl;
				}


			} // else
		} // next b

		if (f_v) {
			cout << "wreath_product::orbits_using_files_and_union_find "
					"performing the union-find for generator " << h
					<< " / " << SG->gens->len << ":" << endl;
		}


		for (i = 0; i < degree_of_tensor_action; ++i) {
			int l1;

			l1 = degree_of_tensor_action / 100;

			if ((i % l1) == 0) {
				cout << i/l1 << " % done with union-find" << endl;
			}
			int u = i;
			unsigned int t = T[i];
			unsigned int r1 = Algo.root_of_tree_uint32_t(S, u);
			unsigned int r2 = Algo.root_of_tree_uint32_t(S, t);

			if (r1 != r2) {
				if (r1 < r2) {
					S[r2] = r1;
				}
				else {
					S[r1] = r2;
				}
			}
		} // next i

	} // next h


	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find Done with the loop" << endl;
		cout << "wreath_product::orbits_using_files_and_union_find Computing the orbit representatives" << endl;
	}


	for (i = 0; i < degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			nb_orbits++;
		}
	}
	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find nb_orbits: " << nb_orbits << endl;
	}

	long int *orbit_length;
	long int *orbit_rep;

	orbit_length = NEW_lint(nb_orbits);
	orbit_rep = NEW_lint(nb_orbits);

	for (i = 0; i < nb_orbits; i++) {
		orbit_length[i] = 0;
	}
	j = 0;
	for (i = 0; i < degree_of_tensor_action; ++i) {
		if (S[i] == i) {
			orbit_rep[j++] = i;
		}
	}

	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find the orbit representatives are: " << endl;
		for (int i = 0; i < nb_orbits; i++) {
			cout << i << ", " << orbit_rep[i] << ", " << endl;
		}
	}
	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find Path compression:" << endl;
	}
	for (i = 0; i < degree_of_tensor_action; ++i) {
		r = Algo.root_of_tree_uint32_t(S, i);
		S[i] = r;
	}
	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find Path compression done" << endl;
	}

	uint32_t *Orbit;
	int goi;
	ring_theory::longinteger_object go;


	SG->group_order(go);
	goi = go.as_int();

	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find "
				"goi=" << goi << endl;
	}


	Orbit = (uint32_t *) NEW_int(goi);

	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find "
				"determining the orbits: " << endl;
	}
	for (orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {

		rep = orbit_rep[orbit_idx];
		uint32_t len = 0;

		if (f_v) {
			cout << "wreath_product::orbits_using_files_and_union_find "
					"determining orbit " << orbit_idx << " / " << nb_orbits
					<< " with rep " << rep << endl;
		}
		for (j = 0; j < degree_of_tensor_action; ++j) {
			if (S[j] == rep) {
				Orbit[len++] = j;
			}
		}
		orbit_length[orbit_idx] = len;
		if (f_v) {
			cout << "wreath_product::orbits_using_files_and_union_find "
					"orbit " << orbit_idx << " / " << nb_orbits << " has length " << len << endl;
		}
		char fname_orbit[1000];

		snprintf(fname_orbit, sizeof(fname_orbit), "wreath_q%d_w%d_orbit_%ld.bin", q, nb_factors, orbit_idx);
		if (f_v) {
			cout << "Writing the file " << fname_orbit << endl;
		}
		{
			ofstream fp(fname_orbit, ios::binary);

			fp.write((char *) &len, sizeof(uint32_t));
			for (i = 0; i < (long int) len; i++) {
				fp.write((char *) &Orbit[i], sizeof(uint32_t));
			}
		}
		if (f_v) {
			cout << "We are done writing the file " << fname_orbit << endl;
		}

	}
	FREE_int((int *) Orbit);
	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find the orbits are: " << endl;
		for (int orbit_idx = 0; orbit_idx < nb_orbits; orbit_idx++) {
			cout << orbit_idx << ", " << orbit_rep[orbit_idx] << ", " << orbit_length[orbit_idx] << ", " << endl;
		}
	}

	if (f_v) {
		cout << "wreath_product::orbits_using_files_and_union_find done" << endl;
	}
}


void wreath_product::orbits_restricted(
		strong_generators* SG,
		actions::action *A,
		int *&result,
		int &nb_gens, int &degree,
		int nb_factors,
		std::string &orbits_restricted_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "wreath_product::orbits_restricted "
				"orbits_restricted_fname=" << orbits_restricted_fname << endl;
	}

	orbiter_kernel_system::file_io Fio;
	data_structures::sorting Sorting;

	//int mtx_n;
	long int *Set;
	long int *Set_in_PG;
	int set_m, set_n;
	int nb_blocks;
	int *restr_first; // [nb_blocks]
	int *restr_length; // [nb_blocks]
	long int i, j, b, h;

	Fio.lint_matrix_read_csv(orbits_restricted_fname,
			Set, set_m, set_n, verbose_level);

	if (set_n != 1) {
		cout << "orbits_restricted set_n != 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Restricting to a set of size " << set_m << endl;
		cout << "converting points to PG point labels" << endl;
	}

	int *v;
	long int s;
	v = NEW_int(dimension_of_tensor_action);
	Set_in_PG = NEW_lint(set_m);
	for (i = 0; i < set_m; i++) {
		s = affine_rank_to_PG_rank(Set[i]);
		Set_in_PG[i] = s;
	}
	//FREE_int(v);
	Sorting.lint_vec_heapsort(Set_in_PG, set_m);
	if (f_v) {
		cout << "after sorting, Set_in_PG:" << endl;
		for (i = 0; i < set_m; i++) {
			cout << i << " : " << Set_in_PG[i] << endl;
		}
	}



	nb_gens = SG->gens->len;
	degree = degree_of_tensor_action;
	//mtx_n = dimension_of_tensor_action;

	int block_size = 1L << 28; // pow(2, 28) ints = 1024 MB

	if (f_v) {
		cout << "block_size=" << block_size << endl;
	}

	nb_blocks = (degree_of_tensor_action + block_size - 1) / block_size;

	if (f_v) {
		cout << "nb_blocks=" << nb_blocks << endl;
	}

	restr_first = NEW_int(nb_blocks);
	restr_length = NEW_int(nb_blocks);

	for (b = 0; b < nb_blocks; b++) {

		if (f_v) {
			cout << "block b=" << b << " / " << nb_blocks << endl;
		}


		int idx;
		Sorting.lint_vec_search(Set_in_PG, set_m, (long int) b * block_size,
					idx, 0 /*verbose_level*/);

		restr_first[b] = idx;
	}

	for (b = 0; b < nb_blocks; b++) {
		cout << b << " : " << restr_first[b] << endl;
	}

	for (b = nb_blocks - 1; b >= 0; b--) {
		if (f_v) {
			cout << "b=" << b << endl;
		}
		if (b == nb_blocks - 1) {
			restr_length[b] = set_m - restr_first[b];
		}
		else {
			restr_length[b] = restr_first[b + 1] - restr_first[b];
		}
	}

	for (b = 0; b < nb_blocks; b++) {
		cout << b << " : " << restr_first[b] << " : " << restr_length[b] << endl;
	}

	long int *Perms;

	Perms = NEW_lint(set_m * SG->gens->len);



	if (f_v) {
		cout << "allocating T, an unsigned int array of size " << block_size << endl;
	}

	unsigned int* T = new unsigned int [block_size];





	for (h = 0; h < SG->gens->len; ++h) {
		if (f_v) {
			cout << "generator h=" << h << " / " << SG->gens->len << endl;
		}

		for (int b = 0; b < nb_blocks; ++b) {
			if (f_v) {
				cout << "block b=" << b << " / " << nb_blocks << endl;
			}


			long int l = MINIMUM((b + 1) * block_size, degree_of_tensor_action) - b * block_size;
			if (f_v) {
				cout << "l=" << l << endl;
			}





			if (!test_if_file_exists(nb_factors, h, b)) {
				cout << "file does not exist h=" << h << " b=" << b << endl;
				exit(1);
			}
			std::string fname;

			make_fname(fname, nb_factors, h, b);
			if (f_v) {
				cout << "reading from file " << fname << endl;
			}
			{
				ifstream fp(fname, ios::binary);

				int l1;
				fp.read((char *) &l1, sizeof(int));
				if (l1 != l) {
					cout << "l1 != l" << endl;
				}
				for (int i = 0; i < l; i++) {
					fp.read((char *) &T [i], sizeof(int));
				}
			}
			if (f_v) {
				cout << "read file " << fname << endl;
				//" of size " << Fio.file_size(fname) << endl;
			}

			long int x, y;
			for (long int u = 0; u < restr_length[b]; u++) {
				i = restr_first[b] + u;
				x = Set_in_PG[i];
				if (x < b * block_size) {
					cout << "x < b * block_size" << endl;
					cout << "x=" << x << " b=" << b << endl;
					exit(1);
				}
				if (x >= (b + 1) * block_size) {
					cout << "x >= (b + 1) * block_size" << endl;
					cout << "x=" << x << " b=" << b << endl;
					exit(1);
				}
				y = T[x - b * block_size];

				int idx;
				if (!Sorting.lint_vec_search(Set_in_PG, set_m, y, idx, 0 /*verbose_level*/)) {
					cout << "did not find element y=" << y << " in Set_in_PG "
							"under generator h=" << h << ", something is wrong" << endl;
					cout << "x=" << x << endl;
					tensor_PG_unrank(v, x);
					s = tensor_affine_rank(v);
					cout << "tensor=";
					Int_vec_print(cout, v, dimension_of_tensor_action);
					cout << endl;
					cout << "affine rank s=" << s << endl;

					cout << "y=" << y << endl;
					tensor_PG_unrank(v, y);
					s = tensor_affine_rank(v);
					cout << "tensor=";
					Int_vec_print(cout, v, dimension_of_tensor_action);
					cout << endl;
					cout << "affine rank s=" << s << endl;

					exit(1);
				}
				j = idx;
				Perms[i * SG->gens->len + h] = j;
			} // next u

		} // next b

	} // next h

	string fname;
	data_structures::string_tools ST;

	fname.assign(orbits_restricted_fname);
	ST.chop_off_extension(fname);

	fname.append("_restricted_action.txt");
	Fio.lint_matrix_write_csv(fname, Perms, set_m, SG->gens->len);

	if (f_v) {
		cout << "wreath_product::orbits_restricted done" << endl;
	}
}

void wreath_product::orbits_restricted_compute(
		strong_generators* SG,
		actions::action *A,
		int *&result,
		int &nb_gens, int &degree,
		int nb_factors,
		std::string &orbits_restricted_fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "wreath_product::orbits_restricted_compute "
				"orbits_restricted_fname=" << orbits_restricted_fname << endl;
	}

	orbiter_kernel_system::file_io Fio;
	data_structures::sorting Sorting;

	long int *Set;
	long int *Set_in_PG;
	int set_m, set_n;
	int i;

	Fio.lint_matrix_read_csv(orbits_restricted_fname,
			Set, set_m, set_n, verbose_level);

	if (set_n != 1) {
		cout << "orbits_restricted set_n != 1" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "Restricting to a set of size " << set_m << endl;
		cout << "converting points to PG point labels" << endl;
	}

	//int *v;
	long int s;
	//v = NEW_int(dimension_of_tensor_action);
	Set_in_PG = NEW_lint(set_m);
	for (i = 0; i < set_m; i++) {
		s = affine_rank_to_PG_rank(Set[i]);
		Set_in_PG[i] = s;
	}
	//FREE_int(v);
	Sorting.lint_vec_heapsort(Set_in_PG, set_m);
	if (f_v) {
		cout << "after sorting, Set_in_PG:" << endl;
	}
#if 0
	for (i = 0; i < set_m; i++) {
		cout << i << " : " << Set_in_PG[i] << endl;
	}
#endif



	nb_gens = SG->gens->len;


	string fname;
	data_structures::string_tools ST;

	int *Perms;
	int perms_m, perms_n;

	fname.assign(orbits_restricted_fname);
	ST.chop_off_extension(fname);

	fname.append("_restricted_action.txt");
	Fio.int_matrix_read_csv(fname, Perms, perms_m, perms_n, verbose_level - 2);
	if (perms_n != SG->gens->len) {
		cout << "perms_n != SG->gens->len" << endl;
		exit(1);
	}
	if (perms_m != set_m) {
		cout << "perms_m != set_m" << endl;
		exit(1);
	}

	degree = perms_m;




	actions::action *A_perm;
	actions::action *A_perm_matrix;

	A_perm = NEW_OBJECT(actions::action);
	A_perm->init_permutation_representation(A,
			FALSE /* f_stay_in_the_old_action */,
			SG->gens,
			Perms, degree,
			verbose_level);
	if (f_v) {
		cout << "created A_perm = " << A_perm->label << endl;
	}

	A_perm_matrix = NEW_OBJECT(actions::action);
	A_perm_matrix->init_permutation_representation(A,
			TRUE /* f_stay_in_the_old_action */,
			SG->gens,
			Perms, degree,
			verbose_level);
	if (f_v) {
		cout << "created A_perm_matrix = " << A_perm_matrix->label << endl;
	}

	permutation_representation *Permutation_representation;

	Permutation_representation = A_perm->G.Permutation_representation;

	data_structures_groups::vector_ge *Gens;

	Gens = NEW_OBJECT(data_structures_groups::vector_ge);

	Gens->init(A_perm, verbose_level - 2);
	Gens->allocate(SG->gens->len, verbose_level - 2);
	for (i = 0; i < SG->gens->len; i++) {
		A_perm->element_move(
				Permutation_representation->Elts
					+ i * A_perm->elt_size_in_int,
				Gens->ith(i),
				verbose_level);
	}

	schreier *Sch;
	ring_theory::longinteger_object go;
	int orbit_idx;

	Sch = NEW_OBJECT(schreier);

	Sch->init(A_perm, verbose_level - 2);
	Sch->initialize_tables();
	Sch->init_generators(*Gens, verbose_level - 2);

	if (f_v) {
		cout << "before Sch->compute_all_point_orbits" << endl;
	}
	Sch->compute_all_point_orbits(0 /*verbose_level - 5*/);
	if (f_v) {
		cout << "after Sch->compute_all_point_orbits" << endl;
	}

	Sch->print_orbit_lengths_tex(cout);
	Sch->print_and_list_orbits_tex(cout);

	data_structures::set_of_sets *Orbits;
	Sch->orbits_as_set_of_sets(Orbits, verbose_level);

	A->group_order(go);
	if (f_v) {
		cout << "Action " << A->label << endl;
		cout << "group order " << go << endl;
		cout << "computing stabilizers:" << endl;
	}

	if (f_v) {
		cout << "wreath_product::orbits_restricted_compute looping over all orbits" << endl;
	}


	for (orbit_idx = 0; orbit_idx < Sch->nb_orbits; orbit_idx++) {
		if (f_v) {
			cout << "computing point stabilizer for orbit " << orbit_idx << ":" << endl;
		}

		int orb_rep;
		long int orbit_rep_in_PG;
		uint32_t orbit_rep_in_PG_uint;

		orb_rep = Sch->orbit[Sch->orbit_first[orbit_idx]];

		orbit_rep_in_PG = Set_in_PG[orb_rep];

		orbit_rep_in_PG_uint = PG_rank_to_affine_rank(orbit_rep_in_PG);

		int *tensor;

		tensor = NEW_int(dimension_of_tensor_action);

		tensor_PG_unrank(tensor, orbit_rep_in_PG);

		if (f_v) {
			cout << "orbit representative is " << orb_rep << " = " << orbit_rep_in_PG << " = " << orbit_rep_in_PG_uint << endl;
			cout << "tensor: ";
			Int_vec_print(cout, tensor, dimension_of_tensor_action);
			cout << endl;
		}
		sims *Stab;

		if (f_v) {
			cout << "before Sch->point_stabilizer in action " << A_perm_matrix->label << endl;
		}
		Sch->point_stabilizer(A_perm_matrix, go,
				Stab, orbit_idx, verbose_level - 5);
		if (f_v) {
			cout << "after Sch->point_stabilizer in action " << A_perm_matrix->label << endl;
		}

		strong_generators *gens;

		gens = NEW_OBJECT(strong_generators);
		gens->init(A_perm_matrix);
		gens->init_from_sims(Stab, verbose_level);


		if (f_v) {
			gens->print_generators_tex(cout);
		}

#if 1
		actions::action *A_on_orbit;

		if (f_v) {
			cout << "computing restricted action on the orbit:" << endl;
		}
		A_on_orbit = A_perm->restricted_action(Orbits->Sets[orbit_idx] + 1, Orbits->Set_size[orbit_idx] - 1,
				verbose_level);

		if (f_v) {
			cout << "generators restricted to the orbit of degree " << Orbits->Set_size[orbit_idx] - 1 << ":" << endl;
			gens->print_generators_MAGMA(A_on_orbit, cout);
		}


		sims *derived_group;
		ring_theory::longinteger_object d_go;

		derived_group = NEW_OBJECT(sims);

		if (f_v) {
			cout << "computing the derived subgroup:" << endl;
		}

		derived_group->init(A_perm_matrix, verbose_level - 2);
		derived_group->init_trivial_group(verbose_level - 1);
		derived_group->build_up_subgroup_random_process(Stab,
				choose_random_generator_derived_group,
				0 /*verbose_level*/);

		derived_group->group_order(d_go);
		if (f_v) {
			cout << "the derived subgroup has order: " << d_go << endl;
		}

		strong_generators *d_gens;

		d_gens = NEW_OBJECT(strong_generators);
		d_gens->init(A_perm_matrix);
		d_gens->init_from_sims(derived_group, 0 /*verbose_level*/);


		d_gens->print_generators_tex(cout);

		schreier *Sch_orbit;

		Sch_orbit = NEW_OBJECT(schreier);
		if (f_v) {
			cout << "computing orbits of stabilizer on the rest of the orbit:" << endl;
		}

		actions::action_global AcGl;

		AcGl.all_point_orbits_from_generators(A_on_orbit,
				*Sch_orbit,
				gens,
				0 /* verbose_level */);

		if (f_v) {
			cout << "Found " << Sch_orbit->nb_orbits << " orbits" << endl;
			Sch_orbit->print_orbit_lengths_tex(cout);
			Sch_orbit->print_and_list_orbits_tex(cout);
		}
#endif

		FREE_OBJECT(gens);
		FREE_OBJECT(Stab);
	}


	if (f_v) {
		cout << "wreath_product::orbits_restricted_compute done" << endl;
	}

}



}}}

