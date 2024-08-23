/*
 * polarity_extension.cpp
 *
 *  Created on: Jan 23, 2024
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace group_constructions {


polarity_extension::polarity_extension()
{
	M = NULL;
	P = NULL;
	Polarity = NULL;
	F = NULL;

	//std::string label;
	//std::string label_tex;

	A_on_points = NULL;
	A_on_hyperplanes = NULL;

	degree_of_matrix_group = 0;
	dimension_of_matrix_group = 0;
	degree_overall = 0;
	//low_level_point_size = 0;
	make_element_size = 0;
	elt_size_int = 0;

	element_coding_offset = NULL;
	perm_offset_i = NULL;
	tmp_Elt1 = NULL;
	tmp_matrix1 = NULL; // [n * n]
	tmp_matrix2 = NULL; // [n * n]
	tmp_vector = NULL; // [n]

	bits_per_digit = 0;

	bits_per_elt = 0;
	char_per_elt = 0;


	elt1 = NULL;
	base_len_in_component1 = 0;
	base_for_component1 = NULL;
	tl_for_component1 = NULL;

	base_len_in_component2 = 0;
	base_for_component2 = NULL;
	tl_for_component2 = NULL;

	base_length = 0;
	the_base = NULL;
	the_transversal_length = NULL;

	Page_storage = NULL;
}


polarity_extension::~polarity_extension()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::~polarity_extension" << endl;
	}
	if (element_coding_offset) {
		FREE_int(element_coding_offset);
	}
	if (perm_offset_i) {
		FREE_int(perm_offset_i);
	}
	if (tmp_Elt1) {
		FREE_int(tmp_Elt1);
	}
	if (tmp_matrix1) {
		FREE_int(tmp_matrix1);
	}
	if (tmp_matrix2) {
		FREE_int(tmp_matrix2);
	}
	if (tmp_vector) {
		FREE_int(tmp_vector);
	}
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (Page_storage) {
		FREE_OBJECT(Page_storage);
	}
	if (base_for_component1) {
		FREE_lint(base_for_component1);
	}
	if (tl_for_component1) {
		FREE_int(tl_for_component1);
	}
	if (base_for_component2) {
		FREE_lint(base_for_component2);
	}
	if (tl_for_component2) {
		FREE_int(tl_for_component2);
	}
	if (the_base) {
		FREE_lint(the_base);
	}
	if (the_transversal_length) {
		FREE_int(the_transversal_length);
	}
	if (f_v) {
		cout << "polarity_extension::~polarity_extension finished" << endl;
	}
}

void polarity_extension::init(
		actions::action *A,
		geometry::projective_space *P,
		geometry::polarity *Polarity,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::init" << endl;
	}


	algebra::matrix_group *M;


	if (!A->is_matrix_group()) {
		cout << "polarity_extension::init "
				"the given group is not a matrix group" << endl;
		exit(1);
	}
	M = A->get_matrix_group();


	A_on_points = A;
	polarity_extension::M = M;
	polarity_extension::P = P;
	polarity_extension::Polarity = Polarity;
	F = M->GFq;

	label = M->label + "_polarity_extension";
	label_tex = M->label_tex + " polarity extension";

	degree_of_matrix_group = M->degree;
	dimension_of_matrix_group = M->n;
	//low_level_point_size =
	//		dimension_of_matrix_group; // this does not work!

	make_element_size =
			M->make_element_size + 1;

	degree_overall =
			2 + Polarity->total_degree;


	//std::string stringify_rank_sequence();
	//std::string stringify_degree_sequence();
#if 0
	//geometry::polarity


	projective_space *P;

	int *Point_to_hyperplane; // [P->N_points]
	int *Hyperplane_to_point; // [P->N_points]

	int *f_absolute;  // [P->N_points]

	long int *Line_to_line; // [P->N_lines] only if n = 3
	int *f_absolute_line; // [P->N_lines] only if n = 3
	int nb_absolute_lines;
	int nb_self_dual_lines;

	int nb_ranks;
	int *rank_sequence;
	int *rank_sequence_opposite;
	long int *nb_objects;
	long int *offset;
	int total_degree;

	int *Mtx; // [d * d]
#endif


	actions::action_global AGlobal;

	if (f_v) {
		cout << "polarity_extension::init "
				"before AGlobal.create_action_on_k_subspaces" << endl;
	}
	A_on_hyperplanes = AGlobal.create_action_on_k_subspaces(
			A_on_points,
			M->n - 1 /* k */,
			verbose_level);
	if (f_v) {
		cout << "polarity_extension::init "
				"after AGlobal.create_action_on_k_subspaces" << endl;
	}



	if (f_v) {
		cout << "polarity_extension::init "
				"rank_sequence = " << Polarity->stringify_rank_sequence() << endl;
		cout << "polarity_extension::init "
				"degree_sequence = " << Polarity->stringify_degree_sequence() << endl;
		cout << "polarity_extension::init "
				"Polarity->total_degree = " << Polarity->total_degree << endl;
		cout << "polarity_extension::init "
				"degree_overall = " << degree_overall << endl;
	}

	element_coding_offset = NEW_int(2);
	element_coding_offset[0] = 0;
	element_coding_offset[1] = M->elt_size_int;

	perm_offset_i = NEW_int(3);
		// one more so it can also be used to indicated
		// the start of the product action.
	perm_offset_i[0] = 0;
	perm_offset_i[1] = 2;
	perm_offset_i[2] = perm_offset_i[1] + Polarity->total_degree;

	elt_size_int = M->elt_size_int + 1;

	if (f_v) {
		cout << "polarity_extension::init "
				"elt_size_int = " << elt_size_int << endl;
	}
	tmp_Elt1 = NEW_int(elt_size_int);
	tmp_matrix1 = NEW_int(M->n * M->n);
	tmp_matrix2 = NEW_int(M->n * M->n);
	tmp_vector = NEW_int(M->n);

	bits_per_digit = M->bits_per_digit;
	bits_per_elt = M->make_element_size * bits_per_digit;
	char_per_elt = ((bits_per_elt + 7) >> 3) + 1;
	elt1 = NEW_uchar(char_per_elt);
	if (f_v) {
		cout << "polarity_extension::init "
				"bits_per_digit = " << bits_per_digit << endl;
		cout << "polarity_extension::init "
				"bits_per_elt = " << bits_per_elt << endl;
		cout << "polarity_extension::init "
				"char_per_elt = " << char_per_elt << endl;
	}
	base_len_in_component1 = 1;
	if (f_v) {
		cout << "polarity_extension::init "
				"base_len_in_component1 = "
				<< base_len_in_component1 << endl;
	}
	base_len_in_component2 = M->base_len(verbose_level);
	if (f_v) {
		cout << "polarity_extension::init "
				"base_len_in_component1 = "
				<< base_len_in_component1 << endl;
		cout << "polarity_extension::init "
				"base_len_in_component2 = "
				<< base_len_in_component2 << endl;
	}
	base_for_component1 = NEW_lint(base_len_in_component1);
	tl_for_component1 = NEW_int(base_len_in_component1);
	base_for_component1[0] = 0;
	tl_for_component1[0] = 2;

	if (f_v) {
		cout << "polarity_extension::init "
				"base_for_component1 = ";
		Lint_vec_print(cout, base_for_component1,
				base_len_in_component1);
		cout << endl;
		cout << "polarity_extension::init "
				"tl_for_component1 = ";
		Int_vec_print(cout, tl_for_component1,
				base_len_in_component1);
		cout << endl;
	}
	base_for_component2 = NEW_lint(base_len_in_component2);
	tl_for_component2 = NEW_int(base_len_in_component2);
	if (f_v) {
		cout << "polarity_extension::init "
				"before M->base_and_transversal_length" << endl;
	}
	M->base_and_transversal_length(
			base_len_in_component2,
			base_for_component2, tl_for_component2,
			verbose_level - 1);
	if (f_v) {
		cout << "polarity_extension::init "
				"after M->base_and_transversal_length" << endl;
	}


	if (f_v) {
		cout << "polarity_extension::init base_for_component2 = ";
		Lint_vec_print(cout, base_for_component2, base_len_in_component2);
		cout << endl;
		cout << "polarity_extension::init tl_for_component2 = ";
		Int_vec_print(cout, tl_for_component2, base_len_in_component2);
		cout << endl;
	}

	Page_storage = NEW_OBJECT(data_structures::page_storage);
	if (f_v) {
		cout << "polarity_extension::init "
				"before Page_storage->init" << endl;
	}
	Page_storage->init(char_per_elt /* entry_size */,
			10 /* page_length_log */, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "polarity_extension::init "
				"after Page_storage->init" << endl;
	}

	if (f_v) {
		cout << "polarity_extension::init "
				"before compute_base_and_transversals" << endl;
	}
	compute_base_and_transversals(verbose_level);
	if (f_v) {
		cout << "polarity_extension::init "
				"after compute_base_and_transversals" << endl;
	}
	if (f_v) {
		cout << "polarity_extension::init the_base = ";
		Lint_vec_print(cout, the_base, base_length);
		cout << endl;
		cout << "polarity_extension::init the_transversal_length = ";
		Int_vec_print(cout, the_transversal_length, base_length);
		cout << endl;
	}

	if (f_v) {
		cout << "polarity_extension::init done" << endl;
	}
}

long int polarity_extension::element_image_of(
		int *Elt,
		long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a0, b, c;

	if (f_v) {
		cout << "polarity_extension::element_image_of" << endl;
	}
	a0 = a;
	b = 0;
	if (a < 2) {

		// we are in A:

		if (f_v) {
			cout << "polarity_extension::element_image_of "
					"we are in component " << 0
					<< " reduced input a=" << a << endl;
		}
		if (Elt[element_coding_offset[1]]) {
			c = 1 - a;
		}
		else {
			c = a;
		}
		if (f_v) {
			cout << "polarity_extension::element_image_of "
					"we are in component " << 0
					<< " reduced output c=" << c << endl;
		}
		b += c;
	}
	else {
		a -= 2;
		b += 2;
		if (a < Polarity->total_degree) {

			// we are in B:


			if (f_v) {
				cout << "polarity_extension::element_image_of "
						"we are in component " << 1
						<< " reduced input a=" << a << endl;
			}
			c = Polarity->image_of_element(
					Elt + element_coding_offset[0], Elt[element_coding_offset[1]], a,
					P,
					M,
					verbose_level - 1);
			if (f_v) {
				cout << "polarity_extension::element_image_of "
						"we are in component " << 1
						<< " reduced output c=" << c << endl;
			}
			b += c;
		}
		else {
			cout << "polarity_extension::element_image_of illegal input value" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "polarity_extension::element_image_of " << a0 << " maps to " << b << endl;
	}
	return b;
}

void polarity_extension::element_one(
		int *Elt)
{
	M->Element->GL_one(Elt + element_coding_offset[0]);
	Elt[element_coding_offset[1]] = 0; // no polarity present
}

int polarity_extension::element_is_one(
		int *Elt)
{
	if (!M->Element->GL_is_one(Elt + element_coding_offset[0])) {
		return false;
	}
	if (Elt[element_coding_offset[1]]) {
		return false;
	}
	return true;
}

void polarity_extension::element_mult(
		int *A, int *B, int *AB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_mult" << endl;
	}

	int *A_Elt;
	int *B_Elt;
	int *AB_Elt;

	A_Elt = A + element_coding_offset[0];
	B_Elt = B + element_coding_offset[0];
	AB_Elt = AB + element_coding_offset[0];

	if (A[element_coding_offset[1]]) {

		int *rho_B_rho;


		rho_B_rho = NEW_int(A_on_points->elt_size_in_int);


		if (f_v) {
			cout << "polarity_extension::element_mult "
					"before element_conjugate_by_polarity" << endl;
		}
		element_conjugate_by_polarity(
				B_Elt,
				rho_B_rho,
				verbose_level - 1);
		if (f_v) {
			cout << "polarity_extension::element_mult "
					"after element_conjugate_by_polarity" << endl;
		}


		if (f_v) {

			int offset = 0;
			int f_do_it_anyway_even_for_big_degree = true;
			int f_print_cycles_of_length_one = true;

			cout << "polarity_extension::element_mult "
					"A_Elt=" << endl;
			A_on_points->Group_element->element_print_quick(A_Elt, cout);
			A_on_points->Group_element->element_print_as_permutation_with_offset(
					A_Elt, cout,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0/*verbose_level*/);
			cout << endl;

			cout << "polarity_extension::element_mult "
					"rho_B_rho=" << endl;
			A_on_points->Group_element->element_print_quick(rho_B_rho, cout);
			A_on_points->Group_element->element_print_as_permutation_with_offset(
					rho_B_rho, cout,
				offset, f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0/*verbose_level*/);
			cout << endl;
		}


		if (f_v) {
			cout << "polarity_extension::element_mult "
					"before A_on_points->Group_element->element_mult" << endl;
		}
		A_on_points->Group_element->element_mult(A_Elt, rho_B_rho, AB_Elt, verbose_level);
		if (f_v) {
			cout << "polarity_extension::element_mult "
					"after A_on_points->Group_element->element_mult" << endl;
		}



		FREE_int(rho_B_rho);

	}
	else {

		if (f_v) {
			cout << "polarity_extension::element_mult "
					"before A_on_points->Group_element->mult" << endl;
		}
		A_on_points->Group_element->mult(A_Elt, B_Elt, AB_Elt);
		if (f_v) {
			cout << "polarity_extension::element_mult "
					"after A_on_points->Group_element->mult" << endl;
		}

	}

	AB[element_coding_offset[1]] =
			(A[element_coding_offset[1]] + B[element_coding_offset[1]]) % 2;

	if (f_v) {
		cout << "polarity_extension::element_mult done" << endl;
	}
}

void polarity_extension::element_move(
		int *A, int *B, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_move" << endl;
	}
	Int_vec_copy(A, B, elt_size_int);
	if (f_v) {
		cout << "polarity_extension::element_move done" << endl;
	}
}

void polarity_extension::compute_images_rho_A_rho(
		int *Mtx, int nb_rows, int *A_Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::compute_images_rho_A_rho" << endl;
	}

	int i;
	long int a, b, c, d;

	for (i = 0; i < M->n + 1; i++) {
		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"computing image of frame element i=" << i << endl;
		}

		M->GFq->Projective_space_basic->PG_element_rank_modified_lint(
				Mtx + i * M->n, 1, M->n, a);

		b = Polarity->Point_to_hyperplane[a];

		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"computing image of frame element a=" << a
					<< " -> (rho) " << b << endl;
		}

		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"before A_on_hyperplanes->Group_element->element_image_of" << endl;
		}
		c = A_on_hyperplanes->Group_element->element_image_of(
				b, A_Elt, 0 /*verbose_level*/);
		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"after A_on_hyperplanes->Group_element->element_image_of" << endl;
		}

		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"computing image of frame element a=" << a
					<< " -> (rho) " << b << " -> *A " << c << endl;
		}

		d = Polarity->Hyperplane_to_point[c];

		if (f_v) {
			cout << "polarity_extension::compute_images_rho_A_rho "
					"computing image of frame element a=" << a
					<< " -> (rho) " << b << " -> *A " << c << " -> (rho) " << d << endl;
		}

		M->GFq->Projective_space_basic->PG_element_unrank_modified_lint(
				Mtx + i * M->n, 1, M->n, d);

	}

	if (f_v) {
		cout << "polarity_extension::compute_images_rho_A_rho "
				"frame image:" << endl;
		Int_matrix_print(Mtx, M->n + 1, M->n);
	}

	if (f_v) {
		cout << "polarity_extension::compute_images_rho_A_rho done" << endl;
	}
}

void polarity_extension::create_rho_A_rho(
		int *A_Elt, int *data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho" << endl;
	}

	linear_algebra::linear_algebra_global LA;


	int *frame;

	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"before LA.create_frame" << endl;
	}
	LA.create_frame(
			frame, M->n, verbose_level - 1);
	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"after LA.create_frame" << endl;
	}



	// compute the image of the frame:


	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"before compute_images_rho_A_rho" << endl;
	}
	compute_images_rho_A_rho(
			frame, M->n + 1 /* nb_rows */, A_Elt,
			verbose_level - 1);
	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"after compute_images_rho_A_rho" << endl;
	}


	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"before LA.adjust_scalars_in_frame" << endl;
	}
	LA.adjust_scalars_in_frame(
			F,
			M->n, frame /* Image_of_basis_in_rows */,
			frame + M->n * M->n /* image_of_all_one */,
			verbose_level - 1);
	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho "
				"after LA.adjust_scalars_in_frame" << endl;
	}

	Int_vec_copy(frame, data, M->n * M->n);

	FREE_int(frame);

	if (f_v) {
		cout << "polarity_extension::create_rho_A_rho done" << endl;
	}
}

void polarity_extension::element_inverse_conjugate_by_polarity(
		int *A_Elt, int *rho_Av_rho, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity" << endl;
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"make_element_size = " << A_on_points->make_element_size << endl;
	}

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity A_Elt=" << endl;
		A_on_points->Group_element->element_print_quick(A_Elt, cout);
	}

	int *data1;
	int *data2;

	data1 = NEW_int(A_on_points->make_element_size);
	data2 = NEW_int(A_on_points->make_element_size);

	A_on_points->Group_element->code_for_make_element(
				data1, A_Elt);

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity element A: ";
		Int_vec_print(cout, data1, A_on_points->make_element_size);
		cout << endl;
	}

	int f_is_semilinear;

	f_is_semilinear = A_on_points->is_semilinear_matrix_group();
	if (f_is_semilinear) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"f_is_semilinear = " << f_is_semilinear << endl;
	}

	int save_frobenius = 0;

	if (f_is_semilinear) {

		if (A_on_points->make_element_size != M->n * M->n + 1) {
			cout << "A_on_points->make_element_size != M->n * M->n + 1" << endl;
			exit(1);
		}

		save_frobenius = data1[A_on_points->make_element_size - 1];
		data1[A_on_points->make_element_size - 1] = 0;

		if (f_v) {
			cout << "polarity_extension::element_inverse_conjugate_by_polarity "
					"element A after removing the Frobenius: ";
			Int_vec_print(cout, data1, A_on_points->make_element_size);
			cout << endl;
		}
	}


	int *A_Elt_copy;
	int *A_Elt_inv;

	A_Elt_copy = NEW_int(A_on_points->elt_size_in_int);
	A_Elt_inv = NEW_int(A_on_points->elt_size_in_int);

	A_on_points->Group_element->make_element(
			A_Elt_copy, data1, 0 /* verbose_level */);

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity A_Elt_copy=" << endl;
		A_on_points->Group_element->element_print_quick(A_Elt_copy, cout);
	}

	A_on_points->Group_element->invert(A_Elt_copy, A_Elt_inv);

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity A_Elt_inv=" << endl;
		A_on_points->Group_element->element_print_quick(A_Elt_inv, cout);
	}

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"before create_rho_A_rho" << endl;
	}
	create_rho_A_rho(
			A_Elt_inv, data2,
			verbose_level);
	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"after create_rho_A_rho" << endl;
	}

	if (f_is_semilinear) {
		data2[M->n * M->n] = save_frobenius;
	}

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"before A_on_points->Group_element->make_element" << endl;
	}
	A_on_points->Group_element->make_element(
			rho_Av_rho, data2, 0 /* verbose_level */);

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"after A_on_points->Group_element->make_element" << endl;
	}

	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity "
				"rho_Av_rho=" << endl;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = true;
		int f_print_cycles_of_length_one = true;
		A_on_points->Group_element->element_print_quick(rho_Av_rho, cout);
		A_on_points->Group_element->element_print_as_permutation_with_offset(
				rho_Av_rho, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}


	FREE_int(data1);
	FREE_int(data2);
	FREE_int(A_Elt_copy);
	FREE_int(A_Elt_inv);


	if (f_v) {
		cout << "polarity_extension::element_inverse_conjugate_by_polarity done" << endl;
	}
}



void polarity_extension::element_conjugate_by_polarity(
		int *A_Elt, int *rho_A_rho, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity" << endl;
		cout << "polarity_extension::element_conjugate_by_polarity "
				"make_element_size = " << A_on_points->make_element_size << endl;
	}

	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity A_Elt=" << endl;
		A_on_points->Group_element->element_print_quick(A_Elt, cout);
	}

	int *data2;

	data2 = NEW_int(A_on_points->make_element_size);


	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity "
				"before create_rho_A_rho" << endl;
	}
	create_rho_A_rho(
			A_Elt, data2,
			verbose_level);
	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity "
				"after create_rho_A_rho" << endl;
	}

	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity "
				"before A_on_points->Group_element->make_element" << endl;
	}
	A_on_points->Group_element->make_element(
			rho_A_rho, data2, verbose_level - 1);

	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity "
				"after A_on_points->Group_element->make_element" << endl;
	}

	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity "
				"rho_A_rho=" << endl;
		int offset = 0;
		int f_do_it_anyway_even_for_big_degree = true;
		int f_print_cycles_of_length_one = true;
		A_on_points->Group_element->element_print_quick(rho_A_rho, cout);
		A_on_points->Group_element->element_print_as_permutation_with_offset(
				rho_A_rho, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;
	}


	FREE_int(data2);


	if (f_v) {
		cout << "polarity_extension::element_conjugate_by_polarity done" << endl;
	}
}




void polarity_extension::element_invert(
		int *A, int *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_invert" << endl;
	}

	if (A[element_coding_offset[1]]) {

#if 0
		if (f_v) {
			cout << "polarity_extension::element_invert "
					"before M->Element->GL_invert_transpose" << endl;
		}
		M->Element->GL_invert_transpose(
				A + element_coding_offset[0],
				Av + element_coding_offset[0]);
		if (f_v) {
			cout << "polarity_extension::element_invert "
					"after M->Element->GL_invert_transpose" << endl;
		}
#endif

		if (f_v) {
			cout << "polarity_extension::element_invert "
					"before element_inverse_conjugate_by_polarity" << endl;
		}
		element_inverse_conjugate_by_polarity(
				A + element_coding_offset[0],
				Av + element_coding_offset[0],
				verbose_level);
		if (f_v) {
			cout << "polarity_extension::element_invert "
					"after element_inverse_conjugate_by_polarity" << endl;
		}

	}
	else {

		if (f_v) {
			cout << "polarity_extension::element_invert "
					"before M->Element->GL_invert" << endl;
		}
		M->Element->GL_invert(
				A + element_coding_offset[0],
				Av + element_coding_offset[0]);
		if (f_v) {
			cout << "polarity_extension::element_invert "
					"after M->Element->GL_invert" << endl;
		}

	}

	Av[element_coding_offset[1]] = A[element_coding_offset[1]];
	if (f_v) {
		cout << "polarity_extension::element_invert done" << endl;
	}
}


void polarity_extension::element_pack(
		int *Elt, uchar *elt)
{
	int i;

	for (i = 0; i < M->make_element_size; i++) {
		put_digit(elt, 0, i, (Elt + element_coding_offset[0])[i]);
	}
	for (i = 0; i < 1; i++) {
		put_digit(elt, 1, i, (Elt + element_coding_offset[1])[i]);
	}
}

void polarity_extension::element_unpack(
		uchar *elt, int *Elt)
{
	int i;
	int *m;

	//cout << "direct_product::element_unpack" << endl;
	m = Elt + element_coding_offset[0];
	for (i = 0; i < M->make_element_size; i++) {
		m[i] = get_digit(elt, 0, i);
	}
	M->Element->GL_invert_internal(m, m + M->elt_size_int_half,
			0 /*verbose_level - 2*/);
	m = Elt + element_coding_offset[1];
	for (i = 0; i < 1; i++) {
		m[i] = get_digit(elt, 1, i);
	}
	//cout << "after direct_product::element_unpack: " << endl;
	//element_print_easy(Elt, cout);
}

void polarity_extension::put_digit(
		uchar *elt, int f, int i, int d)
{
	int h0 = 0;
	int h, h1, a;
	int nb_bits = 0;
	data_structures::data_structures_global D;

	if (f == 0) {
		nb_bits = bits_per_digit;
	}
	else if (f == 1) {
		h0 += M->make_element_size * bits_per_digit;
		nb_bits = bits_per_digit;
	}
	h0 += i * nb_bits;
	for (h = 0; h < nb_bits; h++) {
		h1 = h0 + h;

		if (d & 1) {
			a = 1;
		}
		else {
			a = 0;
		}
		D.bitvector_m_ii(elt, h1, a);
		d >>= 1;
	}
}

int polarity_extension::get_digit(
		uchar *elt, int f, int i)
{
	int h0 = 0;
	int h, h1, a, d;
	int nb_bits = 0;
	data_structures::data_structures_global D;

	if (f == 0) {
		nb_bits = bits_per_digit;
	}
	else if (f == 1) {
		h0 += M->make_element_size * bits_per_digit;
		nb_bits = bits_per_digit;
	}
	h0 += i * nb_bits;
	d = 0;
	for (h = nb_bits - 1; h >= 0; h--) {
		h1 = h0 + h;

		a = D.bitvector_s_i(elt, h1);
		d <<= 1;
		if (a) {
			d |= 1;
		}
	}
	return d;
}

void polarity_extension::make_element(
		int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::make_element" << endl;
	}
	if (f_v) {
		cout << "polarity_extension::make_element data:" << endl;
		Int_vec_print(cout, data, make_element_size);
		cout << endl;
	}
	M->Element->make_element(Elt + element_coding_offset[0],
			data,
			0 /* verbose_level */);
	Elt[element_coding_offset[1]] =
			data[M->make_element_size];
	if (f_v) {
		cout << "polarity_extension::make_element "
				"created this element:" << endl;
		element_print_easy(Elt, cout);
	}
	if (f_v) {
		cout << "polarity_extension::make_element done" << endl;
	}
}

void polarity_extension::element_print_easy(
		int *Elt, std::ostream &ost)
{
	int f;

	ost << "begin element of type polarity extension: " << endl;
	for (f = 0; f < 2; f++) {
		ost << "component " << f << ":" << endl;
		if (f == 0) {
			M->Element->GL_print_easy(Elt + element_coding_offset[0], ost);
			//cout << endl;
		}
		else {
			ost << Elt[element_coding_offset[1]];
		}
	}
	ost << endl;
	ost << "end element of type polarity extension" << endl;
}

void polarity_extension::element_print_easy_latex(
		int *Elt, std::ostream &ost)
{
	int f;

	ost << "\\left(";
	for (f = 0; f < 2; f++) {
		//ost << "component " << f << ":" << endl;
		if (f == 0) {
			M->Element->GL_print_latex(Elt + element_coding_offset[0], ost);
			ost << ",";
		}
		else {
			ost << Elt[element_coding_offset[1]] << endl;
		}
	}
	ost << "\\right)";
	ost << "\\\\" << endl;
}

void polarity_extension::element_code_for_make_element(
		int *Elt, int *data)
{
	M->Element->GL_code_for_make_element(Elt + element_coding_offset[0], data);
	data[M->elt_size_int - 1] = Elt[element_coding_offset[1]];

}


void polarity_extension::element_print_for_make_element(
		int *Elt, std::ostream &ost)
{
	M->Element->GL_print_for_make_element(Elt + element_coding_offset[0], ost);
	ost << Elt[element_coding_offset[1]] << endl;
}

void polarity_extension::element_print_for_make_element_no_commas(
		int *Elt, std::ostream &ost)
{
	M->Element->GL_print_for_make_element_no_commas(Elt + element_coding_offset[0], ost);
	ost << Elt[element_coding_offset[1]] << endl;
}


void polarity_extension::compute_base_and_transversals(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h;

	if (f_v) {
		cout << "polarity_extension::compute_base_and_transversals" << endl;
	}
	base_length = 0;
	base_length += base_len_in_component1;
	base_length += base_len_in_component2;
	the_base = NEW_lint(base_length);

	h = 0;
	for (i = 0; i < base_len_in_component1; i++, h++) {
		the_base[h] = base_for_component1[i];
	}
	for (i = 0; i < base_len_in_component2; i++, h++) {
		the_base[h] = perm_offset_i[1] + base_for_component2[i];
	}
	if (h != base_length) {
		cout << "polarity_extension::compute_base_and_transversals "
				"h != base_length (1)" << endl;
		exit(1);
	}
	the_transversal_length = NEW_int(base_length);
	h = 0;
	for (i = 0; i < base_len_in_component1; i++, h++) {
		the_transversal_length[h] = tl_for_component1[i];
	}
	for (i = 0; i < base_len_in_component2; i++, h++) {
		the_transversal_length[h] = tl_for_component2[i];
	}
	if (h != base_length) {
		cout << "polarity_extension::compute_base_and_transversals "
				"h != base_length (2)" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "polarity_extension::compute_base_and_transversals done" << endl;
	}
}

void polarity_extension::make_strong_generators_data(
		int *&data,
		int &size, int &nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *GL_data;
	int GL_size;
	int GL_nb_gens;
	int h, g;
	int *dat;

	if (f_v) {
		cout << "polarity_extension::make_strong_generators_data" << endl;
	}
	if (f_v) {
		cout << "polarity_extension::make_strong_generators_data "
				"before strong_generators_for_general_linear_group" << endl;
	}

	M->strong_generators_low_level(
			GL_data, GL_size, GL_nb_gens,
			verbose_level - 1);

	if (f_v) {
		cout << "polarity_extension::make_strong_generators_data "
				"after strong_generators_for_general_linear_group" << endl;
	}
	nb_gens = 1 + GL_nb_gens;
	size = make_element_size;
	if (f_v) {
		cout << "size = " << size << endl;
		cout << "GL_size = " << GL_size << endl;
		cout << "M->make_element_size = " << M->make_element_size << endl;
	}
	if (size != GL_size + 1) {
		cout << "polarity_extension::make_strong_generators_data "
				"size != GL_size" << endl;
		exit(1);
	}
	data = NEW_int(nb_gens * size);
	dat = NEW_int(size);

	// the ordering of strong generators in the stabilizer chain is bottom up,
	// so the second component must come first:
	h = 0;

	// generators for the second component:
	for (g = 0; g < GL_nb_gens; g++) {
		Int_vec_zero(dat, size);
		Int_vec_copy(GL_data + g * GL_size,
					dat,
					GL_size);
		dat[M->make_element_size] = 0; // no polarity
		Int_vec_copy(dat, data + h * size, size);
		h++;
	}

	// generators for the first component must come last:
	Int_vec_zero(dat, size);
	Int_vec_copy(GL_data + 0 * GL_size,
				dat,
				GL_size);
	dat[M->make_element_size] = 1; // with polarity
	Int_vec_copy(dat, data + h * size, size);
	h++;


	if (h != nb_gens) {
		cout << "h != nb_gens" << endl;
		exit(1);
	}
	FREE_int(GL_data);
	FREE_int(dat);
	if (f_v) {
		cout << "polarity_extension::make_strong_generators_data data=" << endl;
		Int_matrix_print(data, nb_gens, size);
	}
	if (f_v) {
		cout << "polarity_extension::make_strong_generators_data done" << endl;
	}
}

void polarity_extension::unrank_point(
		long int rk, int *v, int verbose_level)
{
	//Int_vec_zero(v, A->low_level_point_size);
	//P->unrank_point(rk, v, verbose_level);
}

long int polarity_extension::rank_point(
		int *v, int verbose_level)
{
	return 0;
	//P->unrank_point(rk, v, verbose_level);

}

}}}
