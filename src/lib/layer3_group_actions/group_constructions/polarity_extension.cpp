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

	degree_of_matrix_group = 0;
	dimension_of_matrix_group = 0;
	degree_overall = 0;
	//low_level_point_size = 0;
	make_element_size = 0;
	elt_size_int = 0;

	element_coding_offset = NULL;
	perm_offset_i = NULL;
	tmp_Elt1 = NULL;

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

	Elts = NULL;
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
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (Elts) {
		FREE_OBJECT(Elts);
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
		algebra::matrix_group *M,
		geometry::projective_space *P,
		geometry::polarity *Polarity,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::init" << endl;
	}
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
	if (f_v) {
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
	perm_offset_i[1] = perm_offset_i[0] + 2;
	perm_offset_i[2] = perm_offset_i[1] + Polarity->total_degree;
	elt_size_int = M->elt_size_int + 1;
	if (f_v) {
		cout << "polarity_extension::init "
				"elt_size_int = " << elt_size_int << endl;
	}
	tmp_Elt1 = NEW_int(elt_size_int);

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

	Elts = NEW_OBJECT(data_structures::page_storage);
	Elts->init(char_per_elt /* entry_size */,
			10 /* page_length_log */, verbose_level);

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
	if (Elt + element_coding_offset[1]) {
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
	M->Element->GL_mult(
				A + element_coding_offset[0],
				B + element_coding_offset[0],
				AB + element_coding_offset[0],
				0 /* verbose_level */);
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

void polarity_extension::element_invert(
		int *A, int *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "polarity_extension::element_invert" << endl;
	}
	M->Element->GL_invert(
			A + element_coding_offset[0],
			Av + element_coding_offset[0]);
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

	ost << "begin element of direct product: " << endl;
	for (f = 0; f < 2; f++) {
		ost << "component " << f << ":" << endl;
		if (f == 0) {
			M->Element->GL_print_easy(Elt + element_coding_offset[0], ost);
			cout << endl;
		}
		else {
			cout << Elt[element_coding_offset[1]] << endl;
		}
	}
	ost << "end element of direct product" << endl;
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

void polarity_extension::element_print_for_make_element(
		int *Elt, std::ostream &ost)
{
	M->Element->GL_print_for_make_element(Elt + element_coding_offset[0], ost);
	ost << ", " << Elt[element_coding_offset[1]] << endl;
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
		cout << "polarity_extension::make_strong_generators_data done" << endl;
	}
}

#if 0
void polarity_extension::lift_generators(
		groups::strong_generators *SG1,
		groups::strong_generators *SG2,
		actions::action *A,
		groups::strong_generators *&SG3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	actions::action *A1;
	actions::action *A2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	data_structures_groups::vector_ge *gens;
	int i, len1, len2, len3;
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go1, go2, go3;

	if (f_v) {
		cout << "polarity_extension::lift_generators" << endl;
	}
	A1 = SG1->A;
	A2 = SG2->A;
	len1 = SG1->gens->len;
	len2 = SG2->gens->len;
	len3 = len1 + len2;

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(len3, verbose_level - 2);
	Elt1 = NEW_int(A1->elt_size_in_int);
	Elt2 = NEW_int(A2->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	A1->Group_element->element_one(Elt1, 0 /* verbose_level */);
	A2->Group_element->element_one(Elt2, 0 /* verbose_level */);
	for (i = 0; i < len1; i++) {
		A1->Group_element->element_move(SG1->gens->ith(i),
				Elt3, 0 /* verbose_level */);
		A2->Group_element->element_move(Elt2,
				Elt3 + A1->elt_size_in_int,
				0 /* verbose_level */);
		A->Group_element->element_move(Elt3, gens->ith(i), 0);
	}
	for (i = 0; i < len2; i++) {
		A1->Group_element->element_move(Elt1, Elt3,
				0 /* verbose_level */);
		A2->Group_element->element_move(SG2->gens->ith(i),
				Elt3 + A1->elt_size_in_int,
				0 /* verbose_level */);
		A->Group_element->element_move(Elt3, gens->ith(len1 + i), 0);
	}
	if (f_v) {
		cout << "polarity_extension::lift_generators "
				"the generators are:" << endl;
		gens->print_quick(cout);
	}
	SG1->group_order(go1);
	SG2->group_order(go2);
	D.mult(go1, go2, go3);
	A->generators_to_strong_generators(
		true /* f_target_go */, go3,
		gens, SG3,
		verbose_level);
	FREE_OBJECT(gens);
	if (f_v) {
		cout << "polarity_extension::lift_generators done" << endl;
	}
}
#endif


}}}
