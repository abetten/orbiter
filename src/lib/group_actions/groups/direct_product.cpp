// direct_product.cpp
//
// Anton Betten
//
// started:  August 12, 2018




#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {


direct_product::direct_product()
{
	null();
}

direct_product::~direct_product()
{
	freeself();
}

void direct_product::null()
{
	M1 = NULL;
	M2 = NULL;
	F1 = NULL;
	F2 = NULL;
	q1 = 0;
	q2 = 0;
	perm_offset_i = NULL;
	tmp_Elt1 = NULL;
	elt1 = NULL;
	Elts = NULL;
	base_for_component1 = NULL;
	tl_for_component1 = NULL;
	base_for_component2 = NULL;
	tl_for_component2 = NULL;
	the_base = NULL;
	the_transversal_length = NULL;
}

void direct_product::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::freeself" << endl;
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
	null();
	if (f_v) {
		cout << "direct_product::freeself finished" << endl;
		}
}

void direct_product::init(matrix_group *M1, matrix_group *M2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::init" << endl;
	}
	direct_product::M1 = M1;
	direct_product::M2 = M2;
	F1 = M1->GFq;
	F2 = M2->GFq;
	q1 = F1->q;
	q2 = F2->q;

	label.assign(M1->label);
	label.append("_");
	label.append(M2->label);
	label.append("_product");
	label_tex.assign(M1->label_tex);
	label_tex.append(" \\times ");
	label_tex.append(M2->label_tex);
	//sprintf(label, "%s_cross_%s", M1->label, M2->label);
	//sprintf(label_tex, "%s \\times %s", M1->label_tex, M2->label_tex);

	degree_of_matrix_group1 = M1->degree;
	dimension_of_matrix_group1 = M1->n;
	degree_of_matrix_group2 = M2->degree;
	dimension_of_matrix_group2 = M2->n;
	low_level_point_size =
			dimension_of_matrix_group1 + dimension_of_matrix_group2;
	make_element_size =
			M1->make_element_size + M2->make_element_size;
	degree_of_product_action =
			degree_of_matrix_group1 * degree_of_matrix_group2;
	degree_overall =
			degree_of_matrix_group1 + degree_of_matrix_group2
				+ degree_of_product_action;
	if (f_v) {
		cout << "direct_product::init "
				"degree_of_product_action = "
				<< degree_of_product_action << endl;
		cout << "direct_product::init "
				"degree_overall = " << degree_overall << endl;
	}
	perm_offset_i = NEW_int(3);
		// one more so it can also be used to indicated
		// the start of the product action.
	perm_offset_i[0] = 0;
	perm_offset_i[1] = perm_offset_i[0] + degree_of_matrix_group1;
	perm_offset_i[2] = perm_offset_i[1] + degree_of_matrix_group2;
	elt_size_int = M1->elt_size_int + M2->elt_size_int;
	if (f_v) {
		cout << "direct_product::init "
				"elt_size_int = " << elt_size_int << endl;
	}
	tmp_Elt1 = NEW_int(elt_size_int);

	bits_per_digit1 = M1->bits_per_digit;
	bits_per_digit2 = M2->bits_per_digit;
	bits_per_elt = M1->make_element_size * bits_per_digit1
			+ M2->make_element_size * bits_per_digit2;
	char_per_elt = ((bits_per_elt + 7) >> 3);
	elt1 = NEW_uchar(char_per_elt);
	if (f_v) {
		cout << "direct_product::init "
				"bits_per_digit1 = " << bits_per_digit1 << endl;
		cout << "direct_product::init "
				"bits_per_digit2 = " << bits_per_digit2 << endl;
		cout << "direct_product::init "
				"bits_per_elt = " << bits_per_elt << endl;
		cout << "direct_product::init "
				"char_per_elt = " << char_per_elt << endl;
	}
	base_len_in_component1 = M1->base_len(verbose_level);
	if (f_v) {
		cout << "direct_product::init "
				"base_len_in_component1 = "
				<< base_len_in_component1 << endl;
	}
	base_len_in_component2 = M2->base_len(verbose_level);
	if (f_v) {
		cout << "direct_product::init "
				"base_len_in_component1 = "
				<< base_len_in_component1 << endl;
		cout << "direct_product::init "
				"base_len_in_component2 = "
				<< base_len_in_component2 << endl;
	}
	base_for_component1 = NEW_lint(base_len_in_component1);
	tl_for_component1 = NEW_int(base_len_in_component1);

	M1->base_and_transversal_length(
			base_len_in_component1,
			base_for_component1, tl_for_component1,
			verbose_level - 1);
	if (f_v) {
		cout << "direct_product::init "
				"base_for_component1 = ";
		lint_vec_print(cout, base_for_component1,
				base_len_in_component1);
		cout << endl;
		cout << "direct_product::init "
				"tl_for_component1 = ";
		int_vec_print(cout, tl_for_component1,
				base_len_in_component1);
		cout << endl;
	}


	base_for_component2 = NEW_lint(base_len_in_component2);
	tl_for_component2 = NEW_int(base_len_in_component2);
	M2->base_and_transversal_length(
			base_len_in_component2,
			base_for_component2, tl_for_component2,
			verbose_level - 1);

	if (f_v) {
		cout << "direct_product::init base_for_component2 = ";
		lint_vec_print(cout, base_for_component2, base_len_in_component2);
		cout << endl;
		cout << "direct_product::init tl_for_component2 = ";
		int_vec_print(cout, tl_for_component2, base_len_in_component2);
		cout << endl;
	}

	Elts = NEW_OBJECT(page_storage);
	Elts->init(char_per_elt /* entry_size */,
			10 /* page_length_log */, verbose_level);

	if (f_v) {
		cout << "direct_product::init "
				"before compute_base_and_transversals" << endl;
	}
	compute_base_and_transversals(verbose_level);
	if (f_v) {
		cout << "direct_product::init "
				"after compute_base_and_transversals" << endl;
	}
	if (f_v) {
		cout << "direct_product::init the_base = ";
		lint_vec_print(cout, the_base, base_length);
		cout << endl;
		cout << "direct_product::init the_transversal_length = ";
		int_vec_print(cout, the_transversal_length, base_length);
		cout << endl;
	}

	if (f_v) {
		cout << "direct_product::init done" << endl;
	}
}

long int direct_product::element_image_of(int *Elt,
		long int a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a0, b, c, c1, c2, i, j;

	if (f_v) {
		cout << "direct_product::element_image_of" << endl;
	}
	a0 = a;
	b = 0;
	if (a < M1->degree) {
		if (f_v) {
			cout << "direct_product::element_image_of "
					"we are in component " << 0
					<< " reduced input a=" << a << endl;
		}
		c = M1->image_of_element(Elt + offset_i(0), a, 0 /* verbose_level */);
		if (f_v) {
			cout << "direct_product::element_image_of "
					"we are in component " << 0
					<< " reduced output c=" << c << endl;
		}
		b += c;
	}
	else {
		a -= M1->degree;
		b += M1->degree;
		if (a < M2->degree) {
			if (f_v) {
				cout << "direct_product::element_image_of "
						"we are in component " << 1
						<< " reduced input a=" << a << endl;
			}
			c = M2->image_of_element(Elt + offset_i(1), a, 0 /* verbose_level */);
			if (f_v) {
				cout << "direct_product::element_image_of "
						"we are in component " << 1
						<< " reduced output c=" << c << endl;
			}
			b += c;
		}
		else {
			a -= M2->degree;
			b += M2->degree;

			j = a % M2->degree;
			i = a / M2->degree;
			if (f_v) {
				cout << "direct_product::element_image_of "
						"we are in the product component "
						"reduced input a = " << a
						<< " i=" << i << " j=" << j << endl;
			}
			c1 = M1->image_of_element(Elt + offset_i(0), i, 0 /* verbose_level */);
			c2 = M2->image_of_element(Elt + offset_i(1), j, 0 /* verbose_level */);
			c = c1 * M2->degree + c2;
			if (f_v) {
				cout << "direct_product::element_image_of "
						"we are in the product component "
						" reduced output c=" << c << endl;
			}
			b += c;
			if (f_v) {
				cout << "direct_product::element_image_of "
						"we are in the product component "
						" output b=" << b << endl;
			}
		}
	}
	if (f_v) {
		cout << "direct_product::element_image_of " << a0 << " maps to " << b << endl;
	}
	return b;
}

void direct_product::element_one(int *Elt)
{
	M1->GL_one(Elt + offset_i(0));
	M2->GL_one(Elt + offset_i(1));
}

int direct_product::element_is_one(int *Elt)
{
	if (!M1->GL_is_one(Elt + offset_i(0))) {
		return FALSE;
	}
	if (!M2->GL_is_one(Elt + offset_i(1))) {
		return FALSE;
	}
	return TRUE;
}

void direct_product::element_mult(int *A, int *B, int *AB,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::element_mult" << endl;
	}
	M1->GL_mult(A + offset_i(0),
				B + offset_i(0),
				AB + offset_i(0),
				0 /* verbose_level */);
	M2->GL_mult(A + offset_i(1),
				B + offset_i(1),
				AB + offset_i(1),
				0 /* verbose_level */);

	if (f_v) {
		cout << "direct_product::element_mult done" << endl;
	}
}

void direct_product::element_move(int *A, int *B, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::element_move" << endl;
	}
	int_vec_copy(A, B, elt_size_int);
	if (f_v) {
		cout << "direct_product::element_move done" << endl;
	}
}

void direct_product::element_invert(int *A, int *Av, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::element_invert" << endl;
	}
	M1->GL_invert(A + offset_i(0), Av + offset_i(0));
	M2->GL_invert(A + offset_i(1), Av + offset_i(1));
	if (f_v) {
		cout << "direct_product::element_invert done" << endl;
	}
}


int direct_product::offset_i(int f)
{
	if (f == 0) {
		return 0;
	} else if (f == 1) {
		return M1->elt_size_int;
	} else {
		cout << "direct_product::offset_i illegal value of f" << endl;
		exit(1);
	}
}

void direct_product::element_pack(int *Elt, uchar *elt)
{
	int i;

	for (i = 0; i < M1->make_element_size; i++) {
		put_digit(elt, 0, i, (Elt + offset_i(0))[i]);
	}
	for (i = 0; i < M2->make_element_size; i++) {
		put_digit(elt, 1, i, (Elt + offset_i(1))[i]);
	}
}

void direct_product::element_unpack(uchar *elt, int *Elt)
{
	int i;
	int *m;

	//cout << "direct_product::element_unpack" << endl;
	m = Elt + offset_i(0);
	for (i = 0; i < M1->make_element_size; i++) {
		m[i] = get_digit(elt, 0, i);
	}
	M1->GL_invert_internal(m, m + M1->elt_size_int_half,
			0 /*verbose_level - 2*/);
	m = Elt + offset_i(1);
	for (i = 0; i < M2->make_element_size; i++) {
		m[i] = get_digit(elt, 1, i);
	}
	M2->GL_invert_internal(m, m + M2->elt_size_int_half,
			0 /*verbose_level - 2*/);
	//cout << "after direct_product::element_unpack: " << endl;
	//element_print_easy(Elt, cout);
}

void direct_product::put_digit(uchar *elt, int f, int i, int d)
{
	int h0 = 0;
	int h, h1, a;
	int nb_bits = 0;

	if (f == 0) {
		nb_bits = bits_per_digit1;
	} else if (f == 1) {
		h0 += M1->make_element_size * bits_per_digit1;
		nb_bits = bits_per_digit2;
	}
	h0 += i * nb_bits;
	for (h = 0; h < nb_bits; h++) {
		h1 = h0 + h;

		if (d & 1) {
			a = 1;
		} else {
			a = 0;
		}
		bitvector_m_ii(elt, h1, a);
		d >>= 1;
	}
}

int direct_product::get_digit(uchar *elt, int f, int i)
{
	int h0 = 0;
	int h, h1, a, d;
	int nb_bits = 0;

	if (f == 0) {
		nb_bits = bits_per_digit1;
	} else if (f == 1) {
		h0 += M1->make_element_size * bits_per_digit1;
		nb_bits = bits_per_digit2;
	}
	h0 += i * nb_bits;
	d = 0;
	for (h = nb_bits - 1; h >= 0; h--) {
		h1 = h0 + h;

		a = bitvector_s_i(elt, h1);
		d <<= 1;
		if (a) {
			d |= 1;
		}
	}
	return d;
}

void direct_product::make_element(int *Elt, int *data, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "direct_product::make_element" << endl;
		}
	if (f_v) {
		cout << "direct_product::make_element data:" << endl;
		int_vec_print(cout, data, make_element_size);
		cout << endl;
	}
	M1->make_element(Elt + offset_i(0),
			data,
			0 /* verbose_level */);
	M2->make_element(Elt + offset_i(1),
			data + M1->make_element_size,
			0 /* verbose_level */);
	if (f_v) {
		cout << "direct_product::make_element "
				"created this element:" << endl;
		element_print_easy(Elt, cout);
	}
	if (f_v) {
		cout << "direct_product::make_element done" << endl;
		}
}

void direct_product::element_print_easy(int *Elt, ostream &ost)
{
	int f;

	ost << "begin element of direct product: " << endl;
	if (M1->n == 1 && M2->n == 1) {
		cout << "(" << Elt[0] << "," << Elt[1] << ","
				<< Elt[4] << "," << Elt[5] << ")" << endl;
	}
	else {
		for (f = 0; f < 2; f++) {
			ost << "component " << f << ":" << endl;
			if (f == 0) {
				M1->GL_print_easy(Elt + offset_i(f), ost);
				cout << endl;
			} else {
				M2->GL_print_easy(Elt + offset_i(f), ost);
				cout << endl;
			}
		}
	}
	ost << "end element of direct product" << endl;
}

void direct_product::compute_base_and_transversals(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, h;

	if (f_v) {
		cout << "direct_product::compute_base_and_transversals" << endl;
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
		the_base[h] = degree_of_matrix_group1 + base_for_component1[i];
	}
	if (h != base_length) {
		cout << "direct_product::compute_base_and_transversals "
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
		cout << "direct_product::compute_base_and_transversals "
				"h != base_length (2)" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "direct_product::compute_base_and_transversals done" << endl;
		}
}

void direct_product::make_strong_generators_data(int *&data,
		int &size, int &nb_gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *GL1_data;
	int GL1_size;
	int GL1_nb_gens;
	int *GL2_data;
	int GL2_size;
	int GL2_nb_gens;
	int h, g;
	int *dat;

	if (f_v) {
		cout << "direct_product::make_strong_generators_data" << endl;
	}
	if (f_v) {
		cout << "direct_product::make_strong_generators_data "
				"before strong_generators_for_general_linear_group" << endl;
	}
	M1->strong_generators_low_level(
			GL1_data, GL1_size, GL1_nb_gens,
			verbose_level - 1);
	M2->strong_generators_low_level(
			GL2_data, GL2_size, GL2_nb_gens,
			verbose_level - 1);

	if (f_v) {
		cout << "direct_product::make_strong_generators_data "
				"after strong_generators_for_general_linear_group" << endl;
	}
	nb_gens = GL1_nb_gens + GL2_nb_gens;
	size = make_element_size;
	data = NEW_int(nb_gens * size);
	dat = NEW_int(size);

	h = 0;
	// generators for the second component:
	for (g = 0; g < GL2_nb_gens; g++) {
		int_vec_zero(dat, size);
		F1->identity_matrix(
					dat,
					dimension_of_matrix_group1);
		int_vec_copy(GL2_data + g * GL2_size,
					dat + M1->make_element_size,
					GL2_size);
		int_vec_copy(dat, data + h * size, size);
		h++;
	}
	// generators for the first component:
	for (g = 0; g < GL1_nb_gens; g++) {
		int_vec_zero(dat, size);
		int_vec_copy(GL1_data + g * GL1_size,
					dat + 0,
					GL1_size);
		F2->identity_matrix(
					dat + M1->make_element_size,
					dimension_of_matrix_group2);
		int_vec_copy(dat, data + h * size, size);
		h++;
	}
	if (h != nb_gens) {
		cout << "h != nb_gens" << endl;
		exit(1);
	}
	FREE_int(GL1_data);
	FREE_int(GL2_data);
	FREE_int(dat);
	if (f_v) {
		cout << "direct_product::make_strong_generators_data done" << endl;
	}
}

void direct_product::lift_generators(
		strong_generators *SG1,
		strong_generators *SG2,
		action *A, strong_generators *&SG3,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action *A1;
	action *A2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	vector_ge *gens;
	int i, len1, len2, len3;
	longinteger_domain D;
	longinteger_object go1, go2, go3;

	if (f_v) {
		cout << "direct_product::lift_generators" << endl;
	}
	A1 = SG1->A;
	A2 = SG2->A;
	len1 = SG1->gens->len;
	len2 = SG2->gens->len;
	len3 = len1 + len2;

	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(len3, verbose_level - 2);
	Elt1 = NEW_int(A1->elt_size_in_int);
	Elt2 = NEW_int(A2->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	A1->element_one(Elt1, 0 /* verbose_level */);
	A2->element_one(Elt2, 0 /* verbose_level */);
	for (i = 0; i < len1; i++) {
		A1->element_move(SG1->gens->ith(i),
				Elt3, 0 /* verbose_level */);
		A2->element_move(Elt2,
				Elt3 + A1->elt_size_in_int,
				0 /* verbose_level */);
		A->element_move(Elt3, gens->ith(i), 0);
	}
	for (i = 0; i < len2; i++) {
		A1->element_move(Elt1, Elt3,
				0 /* verbose_level */);
		A2->element_move(SG2->gens->ith(i),
				Elt3 + A1->elt_size_in_int,
				0 /* verbose_level */);
		A->element_move(Elt3, gens->ith(len1 + i), 0);
	}
	if (f_v) {
		cout << "direct_product::lift_generators "
				"the generators are:" << endl;
		gens->print_quick(cout);
	}
	SG1->group_order(go1);
	SG2->group_order(go2);
	D.mult(go1, go2, go3);
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, go3,
		gens, SG3,
		verbose_level);
	FREE_OBJECT(gens);
	if (f_v) {
		cout << "direct_product::lift_generators done" << endl;
	}
}

}}
