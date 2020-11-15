// finite_field_projective.cpp
//
// Anton Betten
//
// started:  April 2, 2003
//
// renamed from projective.cpp Nov 16, 2018



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {

void finite_field::PG_element_apply_frobenius(int n,
		int *v, int f)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = frobenius_power(v[i], f);
	}
}



void finite_field::number_of_conditions_satisfied(
		std::string &variety_label,
		int variety_nb_vars, int variety_degree,
		std::vector<std::string> &Variety_coeffs,
		monomial_ordering_type Monomial_ordering_type,
		std::string &number_of_conditions_satisfied_fname,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "finite_field::number_of_conditions_satisfied" << endl;
	}


	if (f_v) {
		cout << "Reading file " << number_of_conditions_satisfied_fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	Fio.read_set_from_file(number_of_conditions_satisfied_fname, Pts, nb_pts, verbose_level);

	int *Cnt;

	Cnt = NEW_int(nb_pts);
	int_vec_zero(Cnt, nb_pts);


	homogeneous_polynomial_domain *HPD;
	number_theory_domain NT;
	int h, i, a;
	long int rk;
	int *v;

	v = NEW_int(variety_nb_vars);

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, variety_nb_vars, variety_degree,
			FALSE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			0 /*verbose_level*/);

	HPD->print_monomial_ordering(cout);


	fname.assign(variety_label);
	//fname.append(".txt");



	for (h = 0; h < Variety_coeffs.size(); h++) {

		if (f_v) {
			cout << "finite_field::number_of_conditions_satisfied "
					"h=" << h << " / " << Variety_coeffs.size() << " : ";
			cout << Variety_coeffs[h] << endl;
		}

		int *coeff;

		coeff = HPD->read_from_string_coefficient_pairs(Variety_coeffs[h], verbose_level - 2);

		if (f_v) {
			cout << "finite_field::number_of_conditions_satisfied "
					"h=" << h << " / " << Variety_coeffs.size() << " coeff:";
			int_vec_print(cout, coeff, HPD->get_nb_monomials());
			cout << endl;
		}

		for (i = 0; i < nb_pts; i++) {
			rk = Pts[i];
			HPD->unrank_point(v, rk);
			a = HPD->evaluate_at_a_point(coeff, v);
			if (a == 0) {
				Cnt[i]++;
			}
		}

		FREE_int(coeff);


	} // next h


	tally T;

	T.init(Cnt, nb_pts, FALSE, 0);

	cout << "Number of conditions satisfied:" << endl;
	T.print_naked(TRUE);
	cout << endl;

	//T.save_classes_individually(fname);

	int f, l, t, j, pos;

	// go through classes in reverse order:
	for (i = T.nb_types - 1; i >= 0; i--) {

		f = T.type_first[i];
		l = T.type_len[i];
		t = T.data_sorted[f];


		string fname2;
		char str[10000];

		fname2.assign(fname);
		sprintf(str, "%d", t);
		fname2.append(str);
		fname2.append(".csv");



		long int *the_class;

		the_class = NEW_lint(l);
		for (j = 0; j < l; j++) {
			pos = T.sorting_perm_inv[f + j];
			the_class[j] = Pts[pos];
		}

		Fio.lint_vec_write_csv(the_class, l, fname2, "case");

		cout << "class of type " << t << " contains " << l << " elements:" << endl;
		display_table_of_projective_points(
				cout, the_class, l, variety_nb_vars);

		FREE_lint(the_class);

	}



	FREE_OBJECT(HPD);
	FREE_int(Cnt);

	FREE_int(v);

	if (f_v) {
		cout << "finite_field::number_of_conditions_satisfied done" << endl;
	}
}


void finite_field::create_intersection_of_zariski_open_sets(
		std::string &variety_label,
		int variety_nb_vars, int variety_degree,
		std::vector<std::string> &Variety_coeffs,
		monomial_ordering_type Monomial_ordering_type,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_intersection_of_zariski_open_sets" << endl;
	}
	homogeneous_polynomial_domain *HPD;
	number_theory_domain NT;
	int h;
	long int *Pts1;
	int sz1;
	long int *Pts2;
	int sz2;
	sorting Sorting;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, variety_nb_vars, variety_degree,
			FALSE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			verbose_level);

	HPD->print_monomial_ordering(cout);


	fname.assign(variety_label);
	fname.append(".txt");

	for (h = 0; h < Variety_coeffs.size(); h++) {

		if (f_v) {
			cout << "finite_field::create_intersection_of_zariski_open_sets "
					"h=" << h << " / " << Variety_coeffs.size() << " : ";
			cout << Variety_coeffs[h] << endl;
		}

		int *coeff;

		coeff = HPD->read_from_string_coefficient_pairs(Variety_coeffs[h], verbose_level - 2);
		if (f_v) {
			cout << "finite_field::create_intersection_of_zariski_open_sets "
					"h=" << h << " / " << Variety_coeffs.size() << " coeff:";
			int_vec_print(cout, coeff, HPD->get_nb_monomials());
			cout << endl;
		}

		Pts = NEW_lint(HPD->get_P()->N_points);

		if (f_v) {
			cout << "finite_field::create_intersection_of_zariski_open_sets "
					"before HPD->enumerate_points_zariski_open_set" << endl;
		}

		vector<long int> Points;


		HPD->enumerate_points_zariski_open_set(coeff, Points, verbose_level);

		FREE_int(coeff);

		if (h ==0) {
			int i;
			nb_pts = Points.size();
			Pts1 = NEW_lint(nb_pts);
			Pts2 = NEW_lint(nb_pts);
			for (i = 0; i < nb_pts; i++) {
				Pts1[i] = Points[i];
			}
			sz1 = nb_pts;
		}
		else {
			int i, idx;
			long int a;
			nb_pts = Points.size();
			sz2 = 0;
			for (i = 0; i < nb_pts; i++) {
				a = Points[i];
				if (Sorting.lint_vec_search(Pts1, sz1, a, idx, 0)) {
					Pts2[sz2++] = a;
				}
			}
			lint_vec_copy(Pts2, Pts1, sz2);
			sz1 = sz2;
		}
		if (f_v) {
			cout << "finite_field::create_intersection_of_zariski_open_sets "
					"after HPD->enumerate_points_zariski_open_set, "
					"nb_pts = " << nb_pts << endl;
		}
	} // next h

	nb_pts = sz1;
	Pts = NEW_lint(sz1);
	lint_vec_copy(Pts1, Pts, sz1);

	display_table_of_projective_points(
			cout, Pts, nb_pts, variety_nb_vars);

	FREE_OBJECT(HPD);
	FREE_lint(Pts1);
	FREE_lint(Pts2);



	if (f_v) {
		cout << "finite_field::create_intersection_of_zariski_open_sets done" << endl;
	}
}


void finite_field::create_projective_variety(
		std::string &variety_label,
		int variety_nb_vars, int variety_degree,
		std::string &variety_coeffs,
		monomial_ordering_type Monomial_ordering_type,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_projective_variety" << endl;
	}

	homogeneous_polynomial_domain *HPD;
	number_theory_domain NT;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, variety_nb_vars, variety_degree,
			FALSE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			verbose_level);

	HPD->print_monomial_ordering(cout);

	fname.assign(variety_label);
	fname.append(".txt");

	int *coeff;

	coeff = HPD->read_from_string_coefficient_pairs(variety_coeffs, verbose_level - 2);

	if (f_v) {
		cout << "finite_field::create_projective_variety coeff:";
		int_vec_print(cout, coeff, HPD->get_nb_monomials());
		cout << endl;
	}

	Pts = NEW_lint(HPD->get_P()->N_points);

	if (f_v) {
		cout << "finite_field::create_projective_variety "
				"before HPD->enumerate_points" << endl;
	}

	vector<long int> Points;
	int i;

	HPD->enumerate_points(coeff, Points, verbose_level);
	nb_pts = Points.size();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		Pts[i] = Points[i];
	}
	if (f_v) {
		cout << "finite_field::create_projective_variety "
				"after HPD->enumerate_points, nb_pts = " << nb_pts << endl;
	}

	display_table_of_projective_points(
			cout, Pts, nb_pts, variety_nb_vars);

	FREE_int(coeff);
	FREE_OBJECT(HPD);

	if (f_v) {
		cout << "finite_field::create_projective_variety done" << endl;
	}
}

void finite_field::create_projective_curve(
		std::string &variety_label,
		int curve_nb_vars, int curve_degree,
		std::string &curve_coeffs,
		monomial_ordering_type Monomial_ordering_type,
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_projective_curve" << endl;
	}

	homogeneous_polynomial_domain *HPD;
	int *coeff;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(this, 2, curve_degree,
			FALSE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			verbose_level);

	HPD->print_monomial_ordering(cout);

	coeff = NEW_int(HPD->get_nb_monomials());
	int_vec_zero(coeff, HPD->get_nb_monomials());

	fname.assign(variety_label);
	fname.append(".txt");
	int *coeffs;
	int len, i, j, a, b, c, s, t;
	int *v;
	int v2[2];

	int_vec_scan(curve_coeffs.c_str(), coeffs, len);
	if (len != curve_degree + 1) {
		cout << "finite_field::create_projective_curve "
				"len != curve_degree + 1" << endl;
		exit(1);
	}

	nb_pts = q + 1;

	v = NEW_int(curve_nb_vars);
	Pts = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		for (j = 0; j < curve_nb_vars; j++) {
			a = HPD->get_monomial(j, 0);
			b = HPD->get_monomial(j, 1);
			v[j] = mult3(coeffs[j], power(s, a), power(t, b));
		}
		PG_element_rank_modified(v, 1, curve_nb_vars, c);
		Pts[i] = c;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, curve_nb_vars);
			cout << " : " << setw(5) << c << endl;
			}
		}

	display_table_of_projective_points(
			cout, Pts, nb_pts, curve_nb_vars);


	FREE_int(v);
	FREE_int(coeffs);
	FREE_OBJECT(HPD);

	if (f_v) {
		cout << "finite_field::create_projective_curve done" << endl;
	}
}

int finite_field::test_if_vectors_are_projectively_equal(int *v1, int *v2, int len)
{
	int *w1, *w2;
	int i;
	int ret;

	w1 = NEW_int(len);
	w2 = NEW_int(len);

	int_vec_copy(v1, w1, len);
	int_vec_copy(v2, w2, len);
	PG_element_normalize(w1, 1, len);
	PG_element_normalize(w2, 1, len);
	for (i = 0; i < len; i++) {
		if (w1[i] != w2[i]) {
			ret = FALSE;
			goto finish;
		}
	}
	ret = TRUE;
finish:
	FREE_int(w1);
	FREE_int(w2);
	return ret;
}

void finite_field::PG_element_normalize(
		int *v, int stride, int len)
// last non-zero element made one
{
	int i, j, a;
	
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = inverse(a);
			v[i * stride] = 1;
			for (j = i - 1; j >= 0; j--) {
				v[j * stride] = mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "finite_field::PG_element_normalize zero vector" << endl;
	exit(1);
}

void finite_field::PG_element_normalize_from_front(
		int *v, int stride, int len)
// first non zero element made one
{
	int i, j, a;
	
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = inverse(a);
			v[i * stride] = 1;
			for (j = i + 1; j < len; j++) {
				v[j * stride] = mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "finite_field::PG_element_normalize_from_front "
			"zero vector" << endl;
	exit(1);
}

void finite_field::PG_elements_embed(
		long int *set_in, long int *set_out, int sz,
		int old_length, int new_length, int *v)
{
	int i;

	for (i = 0; i < sz; i++) {
		set_out[i] = PG_element_embed(set_in[i], old_length, new_length, v);
	}
}

long int finite_field::PG_element_embed(
		long int rk, int old_length, int new_length, int *v)
{
	long int a;
	PG_element_unrank_modified_lint(v, 1, old_length, rk);
	int_vec_zero(v + old_length, new_length - old_length);
	PG_element_rank_modified_lint(v, 1, new_length, a);
	return a;
}


void finite_field::PG_element_unrank_fining(int *v, int len, int a)
{
	int b, c;
	
	if (len != 3) {
		cout << "finite_field::PG_element_unrank_fining "
				"len != 3" << endl;
		exit(1);
		}
	if (a <= 0) {
		cout << "finite_field::PG_element_unrank_fining "
				"a <= 0" << endl;
		exit(1);
		}
	if (a == 1) {
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		return;
		}
	a--;
	if (a <= q) {
		if (a == 1) {
			v[0] = 0;
			v[1] = 1;
			v[2] = 0;
			return;
			}
		else {
			v[0] = 1;
			v[1] = a - 1;
			v[2] = 0;
			return;
			}
		}
	a -= q;
	if (a <= q * q) {
		if (a == 1) {
			v[0] = 0;
			v[1] = 0;
			v[2] = 1;
			return;
			}
		a--;
		a--;
		b = a % (q + 1);
		c = a / (q + 1);
		v[2] = c + 1;
		if (b == 0) {
			v[0] = 1;
			v[1] = 0;
			}
		else if (b == 1) {
			v[0] = 0;
			v[1] = 1;
			}
		else {
			v[0] = 1;
			v[1] = b - 1;
			}
		return;
		}
	else {
		cout << "finite_field::PG_element_unrank_fining "
				"a is illegal" << endl;
		exit(1);
		}
}

int finite_field::PG_element_rank_fining(int *v, int len)
{
	int a;

	if (len != 3) {
		cout << "finite_field::PG_element_rank_fining "
				"len != 3" << endl;
		exit(1);
		}
	//PG_element_normalize(v, 1, len);

	PG_element_normalize_from_front(v, 1, len);

	if (v[2] == 0) {
		if (v[0] == 1 && v[1] == 0) {
			return 1;
		}
		else if (v[0] == 0 && v[1] == 1) {
			return 2;
		}
		else {
			return 2 + v[1];
		}

	}
	else {
		if (v[0] == 0 && v[1] == 0) {
			return q + 2;
		}
		else {
			a = (q + 1) * v[2] + 2;
			if (v[0] == 1 && v[1] == 0) {
				return a;
			}
			else if (v[0] == 0 && v[1] == 1) {
				return a + 1;
			}
			else {
				return a + 1 + v[1];
			}
		}
	}
}

void finite_field::PG_element_unrank_gary_cook(
		int *v, int len, int a)
{
	int b, qm1o2, rk, i;
	
	if (len != 3) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"len != 3" << endl;
		exit(1);
		}
	if (q != 11) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"q != 11" << endl;
		exit(1);
		}
	qm1o2 = (q - 1) >> 1;
	if (a < 0) {
		cout << "finite_field::PG_element_unrank_gary_cook "
				"a < 0" << endl;
		exit(1);
		}
	if (a == 0) {
		v[0] = 0;
		v[1] = 0;
		v[2] = 1;
		}
	else {
		a--;
		if (a < q) {
			v[0] = 0;
			v[1] = 1;
			v[2] = -qm1o2 + a;
			}
		else {
			a -= q;
			rk = a;
			if (rk < q * q) {
				// (1, a, b) = 11a + b + 72, where a,b in -5..5
				b = rk % 11;
				a = (rk - b) / 11;
				v[0] = 1;
				v[1] = -qm1o2 + a;
				v[2] = -qm1o2 + b;
				}
			else {
				cout << "finite_field::PG_element_unrank_gary_cook "
						"a is illegal" << endl;
				exit(1);
				}
			}
		}
	for (i = 0; i < 3; i++) {
		if (v[i] < 0) {
			v[i] += q;
			}
		}
}

void finite_field::PG_element_rank_modified(
		int *v, int stride, int len, int &a)
{
	int i, j, q_power_j, b, sqj;
	int f_v = FALSE;

	if (len <= 0) {
		cout << "finite_field::PG_element_rank_modified "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_rank_modified++;
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	PG_element_normalize(v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	for (i = 0; i < len; i++) {
		if (v[i * stride])
			break;
		}
	if (i == len) {
		cout << "finite_field::PG_element_rank_modified "
				"zero vector" << endl;
		exit(1);
		}
	for (j = i + 1; j < len; j++) {
		if (v[j * stride])
			break;
		}
	if (j == len) {
		// we have the unit vector vector e_i
		a = i;
		return;
		}

	// test for the all one vector:
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1)
				break;
			}
		if (j == len) {
			a = len;
			return;
			}
		}


	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride])
			break;
		}
	if (i < 0) {
		cout << "finite_field::PG_element_rank_modified "
				"zero vector" << endl;
		exit(1);
		}
	if (v[i * stride] != 1) {
		cout << "finite_field::PG_element_rank_modified "
				"vector not normalized" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "i=" << i << endl;
		}

	b = 0;
	q_power_j = 1;
	sqj = 0;
	for (j = 0; j < i; j++) {
		b += q_power_j - 1;
		sqj += q_power_j;
		q_power_j *= q;
		}
	if (f_v) {
		cout << "b=" << b << endl;
		cout << "sqj=" << sqj << endl;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += v[j * stride];
		if (j > 0)
			a *= q;
		if (f_v) {
			cout << "j=" << j << ", a=" << a << endl;
			}
		}

	if (f_v) {
		cout << "a=" << a << endl;
		}

	// take care of 1111 vector being left out
	if (i == len - 1) {
		//cout << "sqj=" << sqj << endl;
		if (a >= sqj)
			a--;
		}

	a += b;
	a += len;
}

void finite_field::PG_element_unrank_modified(
		int *v, int stride, int len, int a)
{
	int n, l, ql, sql, k, j, r, a1 = a;
	
	n = len;
	if (n <= 0) {
		cout << "finite_field::PG_element_unrank_modified "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_unrank_modified++;
	if (a < n) {
		// unit vector:
		for (k = 0; k < n; k++) {
			if (k == a) {
				v[k * stride] = 1;
				}
			else {
				v[k * stride] = 0;
				}
			}
		return;
		}
	a -= n;
	if (a == 0) {
		// all one vector
		for (k = 0; k < n; k++) {
			v[k * stride] = 1;
			}
		return;
		}
	a--;
	
	l = 1;
	ql = q;
	sql = 1;
	// sql = q^0 + q^1 + \cdots + q^{l-1}
	while (l < n) {
		if (a >= ql - 1) {
			a -= (ql - 1);
			sql += ql;
			ql *= q;
			l++;
			continue;
			}
		v[l * stride] = 1;
		for (k = l + 1; k < n; k++) {
			v[k * stride] = 0;
			}
		a++; // take into account that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
				// take int account that the
				// vector 11111 has already been listed
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			v[j * stride] = r;
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++) {
			v[j * stride] = 0;
			}
		return;
		}
	cout << "finite_field::PG_element_unrank_modified "
			"a too large" << endl;
	cout << "len = " << len << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void finite_field::PG_element_rank_modified_lint(
		int *v, int stride, int len, long int &a)
{
	int i, j;
	long int q_power_j, b, sqj;
	int f_v = FALSE;

	if (len <= 0) {
		cout << "finite_field::PG_element_rank_modified_lint "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_rank_modified++;
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	PG_element_normalize(v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	for (i = 0; i < len; i++) {
		if (v[i * stride])
			break;
		}
	if (i == len) {
		cout << "finite_field::PG_element_rank_modified_lint "
				"zero vector" << endl;
		exit(1);
		}
	for (j = i + 1; j < len; j++) {
		if (v[j * stride])
			break;
		}
	if (j == len) {
		// we have the unit vector vector e_i
		a = i;
		return;
		}

	// test for the all one vector:
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1)
				break;
			}
		if (j == len) {
			a = len;
			return;
			}
		}


	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride])
			break;
		}
	if (i < 0) {
		cout << "finite_field::PG_element_rank_modified_lint "
				"zero vector" << endl;
		exit(1);
		}
	if (v[i * stride] != 1) {
		cout << "finite_field::PG_element_rank_modified_lint "
				"vector not normalized" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "i=" << i << endl;
		}

	b = 0;
	q_power_j = 1;
	sqj = 0;
	for (j = 0; j < i; j++) {
		b += q_power_j - 1;
		sqj += q_power_j;
		q_power_j *= q;
		}
	if (f_v) {
		cout << "b=" << b << endl;
		cout << "sqj=" << sqj << endl;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += v[j * stride];
		if (j > 0)
			a *= q;
		if (f_v) {
			cout << "j=" << j << ", a=" << a << endl;
			}
		}

	if (f_v) {
		cout << "a=" << a << endl;
		}

	// take care of 1111 vector being left out
	if (i == len - 1) {
		//cout << "sqj=" << sqj << endl;
		if (a >= sqj)
			a--;
		}

	a += b;
	a += len;
}

void finite_field::PG_elements_unrank_lint(
		int *M, int k, int n, long int *rank_vec)
{
	int i;

	for (i = 0; i < k; i++) {
		PG_element_unrank_modified_lint(M + i * n, 1, n, rank_vec[i]);
	}
}

void finite_field::PG_elements_rank_lint(
		int *M, int k, int n, long int *rank_vec)
{
	int i;

	for (i = 0; i < k; i++) {
		PG_element_rank_modified_lint(M + i * n, 1, n, rank_vec[i]);
	}
}

void finite_field::PG_element_unrank_modified_lint(
		int *v, int stride, int len, long int a)
{
	long int n, l, ql, sql, k, j, r, a1 = a;

	n = len;
	if (n <= 0) {
		cout << "finite_field::PG_element_unrank_modified_lint "
				"len <= 0" << endl;
		exit(1);
		}
	nb_calls_to_PG_element_unrank_modified++;
	if (a < n) {
		// unit vector:
		for (k = 0; k < n; k++) {
			if (k == a) {
				v[k * stride] = 1;
				}
			else {
				v[k * stride] = 0;
				}
			}
		return;
		}
	a -= n;
	if (a == 0) {
		// all one vector
		for (k = 0; k < n; k++) {
			v[k * stride] = 1;
			}
		return;
		}
	a--;

	l = 1;
	ql = q; // q to the power of l
	sql = 1;
	// sql = q^0 + q^1 + \cdots + q^{l-1}
	while (l < n) {
		if (a >= ql - 1) {
			a -= (ql - 1);
			sql += ql;
			ql *= q;
			l++;
			continue;
			}
		v[l * stride] = 1;
		for (k = l + 1; k < n; k++) {
			v[k * stride] = 0;
			}
		a++; // take into account that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
				// take int account that the
				// vector 11111 has already been listed
			}
		j = 0;
		while (a != 0) {
			r = a % q;
			v[j * stride] = r;
			j++;
			a -= r;
			a /= q;
			}
		for ( ; j < l; j++) {
			v[j * stride] = 0;
			}
		return;
		}
	cout << "finite_field::PG_element_unrank_modified_lint "
			"a too large" << endl;
	cout << "len = " << len << endl;
	cout << "l = " << l << endl;
	cout << "a = " << a1 << endl;
	cout << "q = " << q << endl;
	cout << "ql = " << ql << endl;
	cout << "sql = q^0 + q^1 + \\cdots + q^{l-1} = " << sql << endl;
	exit(1);
}

void finite_field::PG_element_rank_modified_not_in_subspace(
		int *v, int stride, int len, int m, long int &a)
{
	long int s, qq, i;

	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= q;
		s += qq;
		}
	s -= (m + 1);

	PG_element_rank_modified_lint(v, stride, len, a);
	if (a > len + s) {
		a -= s;
		}
	a -= (m + 1);
}

void finite_field::PG_element_unrank_modified_not_in_subspace(
		int *v, int stride, int len, int m, long int a)
{
	long int s, qq, i;

	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= q;
		s += qq;
		}
	s -= (m + 1);

	a += (m + 1);
	if (a > len) {
		a += s;
		}

	PG_element_unrank_modified_lint(v, stride, len, a);
}

int finite_field::evaluate_conic_form(int *six_coeffs, int *v3)
{
	//int a = 2, b = 0, c = 0, d = 4, e = 4, f = 4, val, val1;
	//int a = 3, b = 1, c = 2, d = 4, e = 1, f = 4, val, val1;
	int val, val1;

	val = 0;
	val1 = product3(six_coeffs[0], v3[0], v3[0]);
	val = add(val, val1);
	val1 = product3(six_coeffs[1], v3[1], v3[1]);
	val = add(val, val1);
	val1 = product3(six_coeffs[2], v3[2], v3[2]);
	val = add(val, val1);
	val1 = product3(six_coeffs[3], v3[0], v3[1]);
	val = add(val, val1);
	val1 = product3(six_coeffs[4], v3[0], v3[2]);
	val = add(val, val1);
	val1 = product3(six_coeffs[5], v3[1], v3[2]);
	val = add(val, val1);
	return val;
}

int finite_field::evaluate_quadric_form_in_PG_three(
		int *ten_coeffs, int *v4)
{
	int val, val1;

	val = 0;
	val1 = product3(ten_coeffs[0], v4[0], v4[0]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[1], v4[1], v4[1]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[2], v4[2], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[3], v4[3], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[4], v4[0], v4[1]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[5], v4[0], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[6], v4[0], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[7], v4[1], v4[2]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[8], v4[1], v4[3]);
	val = add(val, val1);
	val1 = product3(ten_coeffs[9], v4[2], v4[3]);
	val = add(val, val1);
	return val;
}

int finite_field::Pluecker_12(int *x4, int *y4)
{
	return Pluecker_ij(0, 1, x4, y4);
}

int finite_field::Pluecker_21(int *x4, int *y4)
{
	return Pluecker_ij(1, 0, x4, y4);
}

int finite_field::Pluecker_13(int *x4, int *y4)
{
	return Pluecker_ij(0, 2, x4, y4);
}

int finite_field::Pluecker_31(int *x4, int *y4)
{
	return Pluecker_ij(2, 0, x4, y4);
}

int finite_field::Pluecker_14(int *x4, int *y4)
{
	return Pluecker_ij(0, 3, x4, y4);
}

int finite_field::Pluecker_41(int *x4, int *y4)
{
	return Pluecker_ij(3, 0, x4, y4);
}

int finite_field::Pluecker_23(int *x4, int *y4)
{
	return Pluecker_ij(1, 2, x4, y4);
}

int finite_field::Pluecker_32(int *x4, int *y4)
{
	return Pluecker_ij(2, 1, x4, y4);
}

int finite_field::Pluecker_24(int *x4, int *y4)
{
	return Pluecker_ij(1, 3, x4, y4);
}

int finite_field::Pluecker_42(int *x4, int *y4)
{
	return Pluecker_ij(3, 1, x4, y4);
}

int finite_field::Pluecker_34(int *x4, int *y4)
{
	return Pluecker_ij(2, 3, x4, y4);
}

int finite_field::Pluecker_43(int *x4, int *y4)
{
	return Pluecker_ij(3, 2, x4, y4);
}

int finite_field::Pluecker_ij(int i, int j, int *x4, int *y4)
{
	return add(mult(x4[i], y4[j]), negate(mult(x4[j], y4[i])));
}


int finite_field::evaluate_symplectic_form(int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "finite_field::evaluate_symplectic_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
		}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = add(c, add(
				mult(x[2 * i + 0], y[2 * i + 1]),
				negate(mult(x[2 * i + 1], y[2 * i + 0]))
				));
		}
	return c;
}

int finite_field::evaluate_symmetric_form(int len, int *x, int *y)
{
	int i, n, c;

	if (ODD(len)) {
		cout << "finite_field::evaluate_symmetric_form "
				"len must be even" << endl;
		cout << "len=" << len << endl;
		exit(1);
		}
	c = 0;
	n = len >> 1;
	for (i = 0; i < n; i++) {
		c = add(c, add(
				mult(x[2 * i + 0], y[2 * i + 1]),
				mult(x[2 * i + 1], y[2 * i + 0])
				));
		}
	return c;
}

int finite_field::evaluate_quadratic_form_x0x3mx1x2(int *x)
{
	int a;

	a = add(mult(x[0], x[3]), negate(mult(x[1], x[2])));
	return a;
}

int finite_field::is_totally_isotropic_wrt_symplectic_form(
		int k, int n, int *Basis)
{
	int i, j;

	for (i = 0; i < k; i++) {
		for (j = i + 1; j < k; j++) {
			if (evaluate_symplectic_form(n, Basis + i * n, Basis + j * n)) {
				return FALSE;
				}
			}
		}
	return TRUE;
}

int finite_field::evaluate_monomial(int *monomial,
		int *variables, int nb_vars)
{
	int i, j, a, b, x;

	a = 1;
	for (i = 0; i < nb_vars; i++) {
		b = monomial[i];
		x = variables[i];
		for (j = 0; j < b; j++) {
			a = mult(a, x);
			}
		}
	return a;
}

void finite_field::projective_point_unrank(int n, int *v, int rk)
{
	PG_element_unrank_modified(v, 1 /* stride */,
			n + 1 /* len */, rk);
}

long int finite_field::projective_point_rank(int n, int *v)
{
	long int rk;

	PG_element_rank_modified_lint(v, 1 /* stride */, n + 1, rk);
	return rk;
}

void finite_field::create_BLT_point(
		int *v5, int a, int b, int c, int verbose_level)
// creates the point (-b/2,-c,a,-(b^2/4-ac),1)
// check if it satisfies x_0^2 + x_1x_2 + x_3x_4:
// b^2/4 + (-c)*a + -(b^2/4-ac)
// = b^2/4 -ac -b^2/4 + ac = 0
{
	int f_v = (verbose_level >= 1);
	int v0, v1, v2, v3, v4;
	int half, four, quarter, minus_one;

	if (f_v) {
		cout << "finite_field::create_BLT_point" << endl;
		}
	four = 4 % p;
	half = inverse(2);
	quarter = inverse(four);
	minus_one = negate(1);
	if (f_v) {
		cout << "finite_field::create_BLT_point "
				"four=" << four << endl;
		cout << "finite_field::create_BLT_point "
				"half=" << half << endl;
		cout << "finite_field::create_BLT_point "
				"quarter=" << quarter << endl;
		cout << "finite_field::create_BLT_point "
				"minus_one=" << minus_one << endl;
		}

	v0 = mult(minus_one, mult(b, half));
	v1 = mult(minus_one, c);
	v2 = a;
	v3 = mult(minus_one, add(
			mult(mult(b, b), quarter), negate(mult(a, c))));
	v4 = 1;
	int_vec_init5(v5, v0, v1, v2, v3, v4);
	if (f_v) {
		cout << "finite_field::create_BLT_point done" << endl;
		}

}

void finite_field::Segre_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, a, t6;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Segre_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::Segre_hyperoval needs e odd" << endl;
		exit(1);
		}

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t6 = power(t, 6);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = t6;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Segre_hyperoval "
				"q=" << q << " done" << endl;
		}
}


void finite_field::GlynnI_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, /*Gamma,*/ exponent;
	int *Mtx;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::GlynnI_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::GlynnI_hyperoval needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "finite_field::GlynnI_hyperoval "
				"did not find gamma" << endl;
		exit(1);
		}

	cout << "finite_field::GlynnI_hyperoval sigma = " << sigma
			<< " gamma = " << gamma << endl;
	//Gamma = i_power_j(2, gamma);
	Sigma = NT.i_power_j(2, sigma);

	exponent = 3 * Sigma + 4;

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::GlynnI_hyperoval "
				"q=" << q << " done" << endl;
		}
}

void finite_field::GlynnII_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, te, a;
	int sigma, gamma = 0, Sigma, Gamma, exponent;
	int *Mtx;
	number_theory_domain NT;

	if (f_v) {
		cout << "finite_field::GlynnII_hyperoval q=" << q << endl;
		}
	if (EVEN(e)) {
		cout << "finite_field::GlynnII_hyperoval "
				"needs e odd" << endl;
		exit(1);
		}

	sigma = e - 1;
	for (i = 0; i < e; i++) {
		if (((i * i) % e) == sigma) {
			gamma = i;
			break;
			}
		}
	if (i == e) {
		cout << "finite_field::GlynnII_hyperoval "
				"did not find gamma" << endl;
		exit(1);
		}

	cout << "finite_field::GlynnII_hyperoval "
			"sigma = " << sigma << " gamma = " << i << endl;
	Gamma = NT.i_power_j(2, gamma);
	Sigma = NT.i_power_j(2, sigma);

	exponent = Sigma + Gamma;

	nb_pts = N;

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		te = power(t, exponent);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = te;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::GlynnII_hyperoval "
				"q=" << q << " done" << endl;
		}
}


void finite_field::Subiaco_oval(
		long int *&Pts, int &nb_pts, int f_short, int verbose_level)
// following Payne, Penttila, Pinneri:
// Isomorphisms Between Subiaco q-Clan Geometries,
// Bull. Belg. Math. Soc. 2 (1995) 197-222.
// formula (53)
{
	int f_v = (verbose_level >= 1);
	int N = q + 1;
	int i, t, a, b, h, k, top, bottom;
	int omega, omega2;
	int t2, t3, t4, sqrt_t;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Subiaco_oval "
				"q=" << q << " f_short=" << f_short << endl;
		}

	nb_pts = N;
	k = (q - 1) / 3;
	if (k * 3 != q - 1) {
		cout << "Subiaco_oval k * 3 != q - 1" << endl;
		exit(1);
		}
	omega = power(alpha, k);
	omega2 = mult(omega, omega);
	if (add3(omega2, omega, 1) != 0) {
		cout << "finite_field::Subiaco_oval "
				"add3(omega2, omega, 1) != 0" << endl;
		exit(1);
		}
	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = mult(t, t);
		t3 = mult(t2, t);
		t4 = mult(t2, t2);
		sqrt_t = frobenius_power(t, e - 1);
		if (mult(sqrt_t, sqrt_t) != t) {
			cout << "finite_field::Subiaco_oval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}
		bottom = add3(t4, mult(omega2, t2), 1);
		if (f_short) {
			top = mult(omega2, add(t4, t));
			}
		else {
			top = add3(t3, t2, mult(omega2, t));
			}
		if (FALSE) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
			}
		a = mult(top, inverse(bottom));
		if (f_short) {
			b = sqrt_t;
			}
		else {
			b = mult(omega, sqrt_t);
			}
		h = add(a, b);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Subiaco_oval "
				"q=" << q << " done" << endl;
		}
}




void finite_field::Subiaco_hyperoval(
		long int *&Pts, int &nb_pts, int verbose_level)
// email 12/27/2014
//The o-polynomial of the Subiaco hyperoval is

//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3
// + d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

//where d has absolute trace 1.

//Best,
//Tim

//absolute trace of 1/d is 1 not d...

{
	int f_v = (verbose_level >= 1);
	int N = q + 2;
	int i, t, d, dv, d2, one_d_d2, a, h;
	int t2, t3, t4, sqrt_t;
	int top1, top2, top3, top4, top, bottom;
	int *Mtx;

	if (f_v) {
		cout << "finite_field::Subiaco_hyperoval q=" << q << endl;
		}

	nb_pts = N;
	for (d = 1; d < q; d++) {
		dv = inverse(d);
		if (absolute_trace(dv) == 1) {
			break;
			}
		}
	if (d == q) {
		cout << "finite_field::Subiaco_hyperoval "
				"cannot find element d" << endl;
		exit(1);
		}
	d2 = mult(d, d);
	one_d_d2 = add3(1, d, d2);

	Pts = NEW_lint(N);
	Mtx = NEW_int(N * 3);
	int_vec_zero(Mtx, N * 3);
	for (t = 0; t < q; t++) {
		t2 = mult(t, t);
		t3 = mult(t2, t);
		t4 = mult(t2, t2);
		sqrt_t = frobenius_power(t, e - 1);
		if (mult(sqrt_t, sqrt_t) != t) {
			cout << "finite_field::Subiaco_hyperoval "
					"mult(sqrt_t, sqrt_t) != t" << endl;
			exit(1);
			}


		bottom = add3(t4, mult(d2, t2), 1);

		//t^{1/2}+(d^2t^4 + d^2(1+d+d^2)t^3 +
		// d^2(1+d+d^2)t^2 + d^2t)/(t^4+d^2t^2+1)

		top1 = mult(d2,t4);
		top2 = mult3(d2, one_d_d2, t3);
		top3 = mult3(d2, one_d_d2, t2);
		top4 = mult(d2, t);
		top = add4(top1, top2, top3, top4);

		if (f_v) {
			cout << "t=" << t << " top=" << top
					<< " bottom=" << bottom << endl;
			}
		a = mult(top, inverse(bottom));
		h = add(a, sqrt_t);
		Mtx[t * 3 + 0] = 1;
		Mtx[t * 3 + 1] = t;
		Mtx[t * 3 + 2] = h;
		}
	t = q;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 1;
	Mtx[t * 3 + 2] = 0;
	t = q + 1;
	Mtx[t * 3 + 0] = 0;
	Mtx[t * 3 + 1] = 0;
	Mtx[t * 3 + 2] = 1;
	for (i = 0; i < N; i++) {
		PG_element_rank_modified(Mtx + i * 3, 1, 3, a);
		Pts[i] = a;
		}

	FREE_int(Mtx);
	if (f_v) {
		cout << "finite_field::Subiaco_hyperoval "
				"q=" << q << " done" << endl;
		}
}




// From Bill Cherowitzo's web page:
// In 1991, O'Keefe and Penttila [OKPe92]
// by means of a detailed investigation
// of the divisibility properties of the orders
// of automorphism groups
// of hypothetical hyperovals in this plane,
// discovered a n e w hyperoval.
// Its o-polynomial is given by:

//f(x) = x4 + x16 + x28 + beta*11(x6 + x10 + x14 + x18 + x22 + x26)
// + beta*20(x8 + x20) + beta*6(x12 + x24),
//where ß is a primitive root of GF(32) satisfying beta^5 = beta^2 + 1.
//The full automorphism group of this hyperoval has order 3.

int finite_field::OKeefe_Penttila_32(int t)
// needs the field generated by beta with beta^5 = beta^2+1
// From Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, e, beta6, beta11, beta20;

	t_powers = NEW_int(31);

	power_table(t, t_powers, 31);
	a = add3(t_powers[4], t_powers[16], t_powers[28]);
	b = add6(t_powers[6], t_powers[10], t_powers[14],
			t_powers[18], t_powers[22], t_powers[26]);
	c = add(t_powers[8], t_powers[20]);
	d = add(t_powers[12], t_powers[24]);

	beta6 = power(2, 6);
	beta11 = power(2, 11);
	beta20 = power(2, 20);

	b = mult(b, beta11);
	c = mult(c, beta20);
	d = mult(d, beta6);

	e = add4(a, b, c, d);

	FREE_int(t_powers);
	return e;
}



int finite_field::Subiaco64_1(int t)
// needs the field generated by beta with beta^6 = beta+1
// The first one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add6(t_powers[8], t_powers[12], t_powers[20],
			t_powers[22], t_powers[42], t_powers[52]);
	b = add6(t_powers[4], t_powers[10], t_powers[14],
			t_powers[16], t_powers[30], t_powers[38]);
	c = add6(t_powers[44], t_powers[48], t_powers[54],
			t_powers[56], t_powers[58], t_powers[60]);
	b = add3(b, c, t_powers[62]);
	c = add7(t_powers[2], t_powers[6], t_powers[26],
			t_powers[28], t_powers[32], t_powers[36], t_powers[40]);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int finite_field::Subiaco64_2(int t)
// needs the field generated by beta with beta^6 = beta+1
// The second one from Bill Cherowitzo's hyperoval page
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add3(t_powers[24], t_powers[30], t_powers[62]);
	b = add6(t_powers[4], t_powers[8], t_powers[10],
			t_powers[14], t_powers[16], t_powers[34]);
	c = add6(t_powers[38], t_powers[40], t_powers[44],
			t_powers[46], t_powers[52], t_powers[54]);
	b = add4(b, c, t_powers[58], t_powers[60]);
	c = add5(t_powers[6], t_powers[12], t_powers[18],
			t_powers[20], t_powers[26]);
	d = add5(t_powers[32], t_powers[36], t_powers[42],
			t_powers[48], t_powers[50]);
	c = add(c, d);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}

int finite_field::Adelaide64(int t)
// needs the field generated by beta with beta^6 = beta+1
{
	int *t_powers;
	int a, b, c, d, beta21, beta42;

	t_powers = NEW_int(65);

	power_table(t, t_powers, 65);
	a = add7(t_powers[4], t_powers[8], t_powers[14], t_powers[34],
			t_powers[42], t_powers[48], t_powers[62]);
	b = add8(t_powers[6], t_powers[16], t_powers[26], t_powers[28],
			t_powers[30], t_powers[32], t_powers[40], t_powers[58]);
	c = add8(t_powers[10], t_powers[18], t_powers[24], t_powers[36],
			t_powers[44], t_powers[50], t_powers[52], t_powers[60]);
	beta21 = power(2, 21);
	beta42 = mult(beta21, beta21);
	d = add3(a, mult(beta21, b), mult(beta42, c));
	FREE_int(t_powers);
	return d;
}



void finite_field::LunelliSce(int *pts18, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//const char *override_poly = "19";
	//finite_field F;
	//int n = 3;
	//int q = 16;
	int v[3];
	//int w[3];
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::LunelliSce" << endl;
		}
	//F.init(q), verbose_level - 2);
	//F.init_override_polynomial(q, override_poly, verbose_level);

#if 0
	int cubic1[100];
	int cubic1_size = 0;
	int cubic2[100];
	int cubic2_size = 0;
	int hoval[100];
	int hoval_size = 0;
#endif

	int a, b, i, sz, N;

	if (q != 16) {
		cout << "finite_field::LunelliSce "
				"field order must be 16" << endl;
		exit(1);
		}
	N = Gg.nb_PG_elements(2, 16);
	sz = 0;
	for (i = 0; i < N; i++) {
		PG_element_unrank_modified(v, 1, 3, i);
		//cout << "i=" << i << " v=";
		//int_vec_print(cout, v, 3);
		//cout << endl;

		a = LunelliSce_evaluate_cubic1(v);
		b = LunelliSce_evaluate_cubic2(v);

		// form the symmetric difference of the two cubics:
		if ((a == 0 && b) || (b == 0 && a)) {
			pts18[sz++] = i;
			}
		}
	if (sz != 18) {
		cout << "sz != 18" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the size of the LinelliSce hyperoval is " << sz << endl;
		cout << "the LinelliSce hyperoval is:" << endl;
		int_vec_print(cout, pts18, sz);
		cout << endl;
		}

#if 0
	cout << "the size of cubic1 is " << cubic1_size << endl;
	cout << "the cubic1 is:" << endl;
	int_vec_print(cout, cubic1, cubic1_size);
	cout << endl;
	cout << "the size of cubic2 is " << cubic2_size << endl;
	cout << "the cubic2 is:" << endl;
	int_vec_print(cout, cubic2, cubic2_size);
	cout << endl;
#endif

}

int finite_field::LunelliSce_evaluate_cubic1(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^3 XYZ
{
	int a, b, c, d, e, eta3;

	eta3 = power(2, 3);
	//eta12 = power(2, 12);
	a = power(v[0], 3);
	b = power(v[1], 3);
	c = power(v[2], 3);
	d = product4(eta3, v[0], v[1], v[2]);
	e = add4(a, b, c, d);
	return e;
}

int finite_field::LunelliSce_evaluate_cubic2(int *v)
// computes X^3 + Y^3 + Z^3 + \eta^{12} XYZ
{
	int a, b, c, d, e, eta12;

	//eta3 = power(2, 3);
	eta12 = power(2, 12);
	a = power(v[0], 3);
	b = power(v[1], 3);
	c = power(v[2], 3);
	d = product4(eta12, v[0], v[1], v[2]);
	e = add4(a, b, c, d);
	return e;
}


void finite_field::O4_isomorphism_4to2(
		int *At, int *As, int &f_switch, int *B,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a, b, c, d, e, f, g, h;
	int ev, fv;
	int P[4], Q[4], R[4], S[4];
	int Rx, Ry, Sx, Sy;
	int /*b11,*/ b12, b13, b14;
	int /*b21,*/ b22, b23, b24;
	int /*b31,*/ b32, b33, b34;
	int /*b41,*/ b42, b43, b44;

	if (f_v) {
		cout << "finite_field::O4_isomorphism_4to2" << endl;
		}
	//b11 = B[0 * 4 + 0];
	b12 = B[0 * 4 + 1];
	b13 = B[0 * 4 + 2];
	b14 = B[0 * 4 + 3];
	//b21 = B[1 * 4 + 0];
	b22 = B[1 * 4 + 1];
	b23 = B[1 * 4 + 2];
	b24 = B[1 * 4 + 3];
	//b31 = B[2 * 4 + 0];
	b32 = B[2 * 4 + 1];
	b33 = B[2 * 4 + 2];
	b34 = B[2 * 4 + 3];
	//b41 = B[3 * 4 + 0];
	b42 = B[3 * 4 + 1];
	b43 = B[3 * 4 + 2];
	b44 = B[3 * 4 + 3];
	O4_grid_coordinates_unrank(P[0], P[1], P[2], P[3],
			0, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (0,0) = ";
		int_vec_print(cout, P, 4);
		cout << endl;
		}
	O4_grid_coordinates_unrank(Q[0], Q[1], Q[2], Q[3],
			1, 0, verbose_level);
	if (f_vv) {
		cout << "grid point (1,0) = ";
		int_vec_print(cout, Q, 4);
		cout << endl;
		}
	mult_vector_from_the_left(P, B, R, 4, 4);
	mult_vector_from_the_left(Q, B, S, 4, 4);
	O4_grid_coordinates_rank(R[0], R[1], R[2], R[3],
			Rx, Ry, verbose_level);
	O4_grid_coordinates_rank(S[0], S[1], S[2], S[3],
			Sx, Sy, verbose_level);
	if (f_vv) {
		cout << "Rx=" << Rx << " Ry=" << Ry
				<< " Sx=" << Sx << " Sy=" << Sy << endl;
		}
	if (Ry == Sy) {
		f_switch = FALSE;
		}
	else {
		f_switch = TRUE;
		}
	if (f_vv) {
		cout << "f_switch=" << f_switch << endl;
		}
	if (f_switch) {
		if (b22 == 0 && b24 == 0 && b32 == 0 && b34 == 0) {
			a = 0;
			b = 1;
			f = b12;
			h = b14;
			e = b42;
			g = b44;
			if (e == 0) {
				fv = inverse(f);
				c = mult(fv, b33);
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				c = negate(mult(ev, b23));
				d = negate(mult(ev, b43));
				}
			}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = negate(b32);
			h = negate(b34);
			if (e == 0) {
				fv = inverse(f);
				b = mult(fv, b12);
				c = mult(fv, b33);
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				b = mult(ev, b42);
				c = negate(mult(ev, b23));
				d = negate(mult(ev, b43));
				}
			}
		}
	else {
		// no switch
		if (b22 == 0 && b24 == 0 && b42 == 0 && b44 == 0) {
			a = 0;
			b = 1;
			f = b12;
			h = b14;
			e = negate(b32);
			g = negate(b34);
			if (e == 0) {
				fv = inverse(f);
				c = negate(mult(fv, b43));
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				c = negate(mult(ev, b23));
				d = mult(ev, b33);
				}
			}
		else {
			a = 1;
			e = b22;
			g = b24;
			f = b42;
			h = b44;
			if (e == 0) {
				fv = inverse(f);
				b = mult(fv, b12);
				c = negate(mult(fv, b43));
				d = negate(mult(fv, b13));
				}
			else {
				ev = inverse(e);
				b = negate(mult(ev, b32));
				c = negate(mult(ev, b23));
				d = mult(ev, b33);
				}
			}
		}
	if (f_vv) {
		cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << endl;
		cout << "e=" << e << " f=" << f << " g=" << g << " h=" << h << endl;
		}
	At[0] = d;
	At[1] = b;
	At[2] = c;
	At[3] = a;
	As[0] = h;
	As[1] = f;
	As[2] = g;
	As[3] = e;
	if (f_v) {
		cout << "At:" << endl;
		print_integer_matrix_width(cout, At, 2, 2, 2, log10_of_q);
		cout << "As:" << endl;
		print_integer_matrix_width(cout, As, 2, 2, 2, log10_of_q);
		}

}

void finite_field::O4_isomorphism_2to4(
		int *At, int *As, int f_switch, int *B)
{
	int a, b, c, d, e, f, g, h;

	a = At[3];
	b = At[1];
	c = At[2];
	d = At[0];
	e = As[3];
	f = As[1];
	g = As[2];
	h = As[0];
	if (f_switch) {
		B[0 * 4 + 0] = mult(h, d);
		B[0 * 4 + 1] = mult(f, b);
		B[0 * 4 + 2] = negate(mult(f, d));
		B[0 * 4 + 3] = mult(h, b);
		B[1 * 4 + 0] = mult(g, c);
		B[1 * 4 + 1] = mult(e, a);
		B[1 * 4 + 2] = negate(mult(e, c));
		B[1 * 4 + 3] = mult(g, a);
		B[2 * 4 + 0] = negate(mult(h, c));
		B[2 * 4 + 1] = negate(mult(f, a));
		B[2 * 4 + 2] = mult(f, c);
		B[2 * 4 + 3] = negate(mult(h, a));
		B[3 * 4 + 0] = mult(g, d);
		B[3 * 4 + 1] = mult(e, b);
		B[3 * 4 + 2] = negate(mult(e, d));
		B[3 * 4 + 3] = mult(g, b);
		}
	else {
		B[0 * 4 + 0] = mult(h, d);
		B[0 * 4 + 1] = mult(f, b);
		B[0 * 4 + 2] = negate(mult(f, d));
		B[0 * 4 + 3] = mult(h, b);
		B[1 * 4 + 0] = mult(g, c);
		B[1 * 4 + 1] = mult(e, a);
		B[1 * 4 + 2] = negate(mult(e, c));
		B[1 * 4 + 3] = mult(g, a);
		B[2 * 4 + 0] = negate(mult(g, d));
		B[2 * 4 + 1] = negate(mult(e, b));
		B[2 * 4 + 2] = mult(e, d);
		B[2 * 4 + 3] = negate(mult(g, b));
		B[3 * 4 + 0] = mult(h, c);
		B[3 * 4 + 1] = mult(f, a);
		B[3 * 4 + 2] = negate(mult(f, c));
		B[3 * 4 + 3] = mult(h, a);
		}
}

void finite_field::O4_grid_coordinates_rank(
		int x1, int x2, int x3, int x4, int &grid_x, int &grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d, av, e;
	int v[2], w[2];

	a = x1;
	b = x4;
	c = negate(x3);
	d = x2;

	if (a) {
		if (a != 1) {
			av = inverse(a);
			b = mult(b, av);
			c = mult(c, av);
			d = mult(d, av);
			}
		v[0] = 1;
		w[0] = 1;
		v[1] = c;
		w[1] = b;
		e = mult(c, b);
		if (e != d) {
			cout << "finite_field::O4_grid_coordinates_rank "
					"e != d" << endl;
			exit(1);
			}
		}
	else if (b == 0) {
		v[0] = 0;
		v[1] = 1;
		w[0] = c;
		w[1] = d;
		}
	else {
		if (c) {
			cout << "a is zero, b and c are not" << endl;
			exit(1);
			}
		w[0] = 0;
		w[1] = 1;
		v[0] = b;
		v[1] = d;
		}
	PG_element_normalize_from_front(v, 1, 2);
	PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		int_vec_print(cout, v, 2);
		int_vec_print(cout, w, 2);
		cout << endl;
		}

	PG_element_rank_modified(v, 1, 2, grid_x);
	PG_element_rank_modified(w, 1, 2, grid_y);
}

void finite_field::O4_grid_coordinates_unrank(
		int &x1, int &x2, int &x3, int &x4,
		int grid_x, int grid_y,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, c, d;
	int v[2], w[2];

	PG_element_unrank_modified(v, 1, 2, grid_x);
	PG_element_unrank_modified(w, 1, 2, grid_y);
	PG_element_normalize_from_front(v, 1, 2);
	PG_element_normalize_from_front(w, 1, 2);
	if (f_v) {
		int_vec_print(cout, v, 2);
		int_vec_print(cout, w, 2);
		cout << endl;
		}

	a = mult(v[0], w[0]);
	b = mult(v[0], w[1]);
	c = mult(v[1], w[0]);
	d = mult(v[1], w[1]);
	x1 = a;
	x2 = d;
	x3 = negate(c);
	x4 = b;
}

void finite_field::O4_find_tangent_plane(
		int pt_x1, int pt_x2, int pt_x3, int pt_x4,
		int *tangent_plane,
		int verbose_level)
{
	//int A[4];
	int C[3 * 4];
	int size, x, y, z, xx, yy, zz, h, k;
	int x1, x2, x3, x4;
	int y1, y2, y3, y4;
	int f_special = FALSE;
	int f_complete = FALSE;
	int base_cols[4];
	int f_P = FALSE;
	int rk, det;
	int vec2[2];


	cout << "O4_find_tangent_plane pt_x1=" << pt_x1
		<< " pt_x2=" << pt_x2
		<< " pt_x3=" << pt_x3
		<< " pt_x4=" << pt_x4 << endl;
	size = q + 1;
#if 0
	A[0] = pt_x1;
	A[3] = pt_x2;
	A[2] = negate(pt_x3);
	A[1] = pt_x4;
#endif

	int *secants1;
	int *secants2;
	int nb_secants = 0;
	int *complement;
	int nb_complement = 0;

	secants1 = NEW_int(size * size);
	secants2 = NEW_int(size * size);
	complement = NEW_int(size * size);
	for (x = 0; x < size; x++) {
		for (y = 0; y < size; y++) {
			z = x * size + y;

			//cout << "trying grid point (" << x << "," << y << ")" << endl;
			//cout << "nb_secants=" << nb_secants << endl;
			O4_grid_coordinates_unrank(x1, x2, x3, x4, x, y, 0);

			//cout << "x1=" << x1 << " x2=" << x2
			//<< " x3=" << x3 << " x4=" << x4 << endl;




			for (k = 0; k < size; k++) {
				PG_element_unrank_modified(vec2, 1, 2, k);
				y1 = add(mult(pt_x1, vec2[0]), mult(x1, vec2[1]));
				y2 = add(mult(pt_x2, vec2[0]), mult(x2, vec2[1]));
				y3 = add(mult(pt_x3, vec2[0]), mult(x3, vec2[1]));
				y4 = add(mult(pt_x4, vec2[0]), mult(x4, vec2[1]));
				det = add(mult(y1, y2), mult(y3, y4));
				if (det != 0) {
					continue;
					}
				O4_grid_coordinates_rank(y1, y2, y3, y4, xx, yy, 0);
				zz = xx * size + yy;
				if (zz == z)
					continue;
				C[0] = pt_x1;
				C[1] = pt_x2;
				C[2] = pt_x3;
				C[3] = pt_x4;

				C[4] = x1;
				C[5] = x2;
				C[6] = x3;
				C[7] = x4;

				C[8] = y1;
				C[9] = y2;
				C[10] = y3;
				C[11] = y4;

				rk = Gauss_int(C, f_special, f_complete, base_cols,
					f_P, NULL, 3, 4, 4, 0);
				if (rk < 3) {
					secants1[nb_secants] = z;
					secants2[nb_secants] = zz;
					nb_secants++;
					}

				}

#if 0

			for (xx = 0; xx < size; xx++) {
				for (yy = 0; yy < size; yy++) {
					zz = xx * size + yy;
					if (zz == z)
						continue;
					O4_grid_coordinates_unrank(F, y1, y2, y3, y4, xx, yy, 0);
					//cout << "y1=" << y1 << " y2=" << y2
					//<< " y3=" << y3 << " y4=" << y4 << endl;
					C[0] = pt_x1;
					C[1] = pt_x2;
					C[2] = pt_x3;
					C[3] = pt_x4;

					C[4] = x1;
					C[5] = x2;
					C[6] = x3;
					C[7] = x4;

					C[8] = y1;
					C[9] = y2;
					C[10] = y3;
					C[11] = y4;

					rk = F.Gauss_int(C, f_special, f_complete, base_cols,
						f_P, NULL, 3, 4, 4, 0);
					if (rk < 3) {
						secants1[nb_secants] = z;
						secants2[nb_secants] = zz;
						nb_secants++;
						}
					}
				}
#endif


			}
		}
	cout << "nb_secants=" << nb_secants << endl;
	int_vec_print(cout, secants1, nb_secants);
	cout << endl;
	int_vec_print(cout, secants2, nb_secants);
	cout << endl;
	h = 0;
	for (zz = 0; zz < size * size; zz++) {
		if (secants1[h] > zz) {
			complement[nb_complement++] = zz;
			}
		else {
			h++;
			}
		}
	cout << "complement = tangents:" << endl;
	int_vec_print(cout, complement, nb_complement);
	cout << endl;

	int *T;
	T = NEW_int(4 * nb_complement);

	for (h = 0; h < nb_complement; h++) {
		z = complement[h];
		x = z / size;
		y = z % size;
		cout << setw(3) << h << " : " << setw(4) << z
				<< " : " << x << "," << y << " : ";
		O4_grid_coordinates_unrank(y1, y2, y3, y4,
				x, y, verbose_level);
		cout << "y1=" << y1 << " y2=" << y2
				<< " y3=" << y3 << " y4=" << y4 << endl;
		T[h * 4 + 0] = y1;
		T[h * 4 + 1] = y2;
		T[h * 4 + 2] = y3;
		T[h * 4 + 3] = y4;
		}


	rk = Gauss_int(T, f_special, f_complete, base_cols,
		f_P, NULL, nb_complement, 4, 4, 0);
	cout << "the rank of the tangent space is " << rk << endl;
	cout << "basis:" << endl;
	print_integer_matrix_width(cout, T, rk, 4, 4, log10_of_q);

	if (rk != 3) {
		cout << "rk = " << rk << " not equal to 3" << endl;
		exit(1);
		}
	int i;
	for (i = 0; i < 12; i++) {
		tangent_plane[i] = T[i];
		}
	FREE_int(secants1);
	FREE_int(secants2);
	FREE_int(complement);
	FREE_int(T);

#if 0
	for (h = 0; h < nb_secants; h++) {
		z = secants1[h];
		zz = secants2[h];
		x = z / size;
		y = z % size;
		xx = zz / size;
		yy = zz % size;
		cout << "(" << x << "," << y << "),(" << xx
				<< "," << yy << ")" << endl;
		O4_grid_coordinates_unrank(F, x1, x2, x3, x4,
				x, y, verbose_level);
		cout << "x1=" << x1 << " x2=" << x2
				<< " x3=" << x3 << " x4=" << x4 << endl;
		O4_grid_coordinates_unrank(F, y1, y2, y3, y4, xx, yy, verbose_level);
		cout << "y1=" << y1 << " y2=" << y2
				<< " y3=" << y3 << " y4=" << y4 << endl;
		}
#endif
}

void finite_field::oval_polynomial(
	int *S, unipoly_domain &D, unipoly_object &poly,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, v[3], x; //, y;
	int *map;

	if (f_v) {
		cout << "finite_field::oval_polynomial" << endl;
		}
	map = NEW_int(q);
	for (i = 0; i < q; i++) {
		PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[2 + i]);
		if (v[2] != 1) {
			cout << "finite_field::oval_polynomial "
					"not an affine point" << endl;
			exit(1);
			}
		x = v[0];
		//y = v[1];
		//cout << "map[" << i << "] = " << xx << endl;
		map[i] = x;
		}
	if (f_v) {
		cout << "the map" << endl;
		for (i = 0; i < q; i++) {
			cout << map[i] << " ";
			}
		cout << endl;
		}

	D.create_Dickson_polynomial(poly, map);

	FREE_int(map);
	if (f_v) {
		cout << "finite_field::oval_polynomial done" << endl;
		}
}


void finite_field::all_PG_elements_in_subspace(
		int *genma, int k, int n, long int *&point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i;
	long int a;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "finite_field::all_PG_elements_in_subspace" << endl;
		}
	message = NEW_int(k);
	word = NEW_int(n);
	nb_points = Combi.generalized_binomial(k, 1, q);
	point_list = NEW_lint(nb_points);

	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			int_vec_print(cout, message, k);
			cout << endl;
			}
		mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			int_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified_lint(word, 1, n, a);
		if (f_vv) {
			cout << "which has rank " << a << endl;
			}
		point_list[i] = a;
		}

	FREE_int(message);
	FREE_int(word);
	if (f_v) {
		cout << "finite_field::all_PG_elements_in_subspace "
				"done" << endl;
		}
}

void finite_field::all_PG_elements_in_subspace_array_is_given(
		int *genma, int k, int n, long int *point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i, j;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "finite_field::all_PG_elements_in_"
				"subspace_array_is_given" << endl;
		}
	message = NEW_int(k);
	word = NEW_int(n);
	nb_points = Combi.generalized_binomial(k, 1, q);
	//point_list = NEW_int(nb_points);

	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			int_vec_print(cout, message, k);
			cout << endl;
			}
		mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			int_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified(word, 1, n, j);
		if (f_vv) {
			cout << "which has rank " << j << endl;
			}
		point_list[i] = j;
		}

	FREE_int(message);
	FREE_int(word);
	if (f_v) {
		cout << "finite_field::all_PG_elements_in_"
				"subspace_array_is_given "
				"done" << endl;
		}
}

void finite_field::display_all_PG_elements(int n)
{
	int *v = NEW_int(n + 1);
	geometry_global Gg;
	int l = Gg.nb_PG_elements(n, q);
	int i, j, a;

	for (i = 0; i < l; i++) {
		PG_element_unrank_modified(v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified(v, 1, n + 1, a);
		cout << " : " << a << endl;
		}
	FREE_int(v);
}

void finite_field::display_all_PG_elements_not_in_subspace(int n, int m)
{
	int *v = NEW_int(n + 1);
	geometry_global Gg;
	long int l = Gg.nb_PG_elements_not_in_subspace(n, m, q);
	long int i, j, a;

	for (i = 0; i < l; i++) {
		PG_element_unrank_modified_not_in_subspace(v, 1, n + 1, m, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified_not_in_subspace(v, 1, n + 1, m, a);
		cout << " : " << a << endl;
		}
	FREE_int(v);
}

void finite_field::display_all_AG_elements(int n)
{
	int *v = NEW_int(n);
	geometry_global Gg;
	int l = Gg.nb_AG_elements(n, q);
	int i, j;

	for (i = 0; i < l; i++) {
		Gg.AG_element_unrank(q, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n; j++) {
			cout << v[j] << " ";
			}
		cout << endl;
		}
	FREE_int(v);
}


void finite_field::do_cone_over(int n,
	long int *set_in, int set_size_in, long int *&set_out, int &set_size_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	int *v;
	int d = n + 2;
	int h, u, a, b, cnt;

	if (f_v) {
		cout << "finite_field::do_cone_over" << endl;
		}
	P1 = NEW_OBJECT(projective_space);
	P2 = NEW_OBJECT(projective_space);

	P1->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(n + 1, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_int(d);

	set_size_out = 1 + q * set_size_in;
	set_out = NEW_lint(set_size_out);
	cnt = 0;

	// create the vertex:
	int_vec_zero(v, d);
	v[d - 1] = 1;
	b = P2->rank_point(v);
	set_out[cnt++] = b;


	// for each point, create the generator
	// which is the line connecting the point and the vertex
	// since we have created the vertex already,
	// we only need to create q points per line:

	for (h = 0; h < set_size_in; h++) {
		a = set_in[h];
		for (u = 0; u < q; u++) {
			P1->unrank_point(v, a);
			v[d - 1] = u;
			b = P2->rank_point(v);
			set_out[cnt++] = b;
			}
		}

	if (cnt != set_size_out) {
		cout << "finite_field::do_cone_over cnt != set_size_out" << endl;
		exit(1);
		}

	FREE_int(v);
	FREE_OBJECT(P1);
	FREE_OBJECT(P2);
}


void finite_field::do_blocking_set_family_3(int n,
	long int *set_in, int set_size,
	long int *&the_set_out, int &set_size_out,
	int verbose_level)
{
	projective_space *P;
	int h;

	if (n != 2) {
		cout << "finite_field::do_blocking_set_family_3 "
				"we need n = 2" << endl;
		exit(1);
		}
	if (ODD(q)) {
		cout << "finite_field::do_blocking_set_family_3 "
				"we need q even" << endl;
		exit(1);
		}
	if (set_size != q + 2) {
		cout << "finite_field::do_blocking_set_family_3 "
				"we need set_size == q + 2" << endl;
		exit(1);
		}
	P = NEW_OBJECT(projective_space);

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);


	int *idx;
	int p_idx[4];
	int line[6];
	int diag_pts[3];
	int diag_line;
	int nb, pt, sz;
	int i, j;
	int basis[6];
	combinatorics_domain Combi;

	fancy_set *S;

	S = NEW_OBJECT(fancy_set);

	S->init(P->N_lines, 0);
	S->k = 0;

	idx = NEW_int(set_size);

#if 1
	while (TRUE) {
		cout << "choosing random permutation" << endl;
		Combi.random_permutation(idx, set_size);

		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
			}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->intersection_of_two_lines(line[0], line[5]);
		diag_pts[1] = P->intersection_of_two_lines(line[1], line[4]);
		diag_pts[2] = P->intersection_of_two_lines(line[2], line[3]);

		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);
		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		P->unrank_line(basis, diag_line);
		int_matrix_print(basis, 2, 3);
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->is_incident(pt, diag_line)) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			break;
			}
		} // while
#endif

#if 0
	int fundamental_quadrangle[4] = {0,1,2,3};
	int basis[6];

	for (i = 0; i < 4; i++) {
		if (!int_vec_search_linear(set_in, set_size, fundamental_quadrangle[i], j)) {
			cout << "the point " << fundamental_quadrangle[i] << " is not contained in the hyperoval" << endl;
			exit(1);
			}
		idx[i] = j;
		}
	cout << "the fundamental quadrangle is contained, the positions are " << endl;
		cout << idx[0] << ", ";
		cout << idx[1] << ", ";
		cout << idx[2] << ", ";
		cout << idx[3] << endl;

		for (i = 0; i < 4; i++) {
			p_idx[i] = set_in[idx[i]];
			}

		line[0] = P->line_through_two_points(p_idx[0], p_idx[1]);
		line[1] = P->line_through_two_points(p_idx[0], p_idx[2]);
		line[2] = P->line_through_two_points(p_idx[0], p_idx[3]);
		line[3] = P->line_through_two_points(p_idx[1], p_idx[2]);
		line[4] = P->line_through_two_points(p_idx[1], p_idx[3]);
		line[5] = P->line_through_two_points(p_idx[2], p_idx[3]);
		diag_pts[0] = P->line_intersection(line[0], line[5]);
		diag_pts[1] = P->line_intersection(line[1], line[4]);
		diag_pts[2] = P->line_intersection(line[2], line[3]);

		diag_line = P->line_through_two_points(diag_pts[0], diag_pts[1]);
		cout << "The diagonal line is " << diag_line << endl;

		P->unrank_line(basis, diag_line);
		int_matrix_print(basis, 2, 3);

		if (diag_line != P->line_through_two_points(diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		nb = 0;
		for (i = 0; i < set_size; i++) {
			pt = set_in[i];
			if (P->Incidence[pt * P->N_lines + diag_line]) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			}
		else {
			cout << "error: the diagonal line is not external" << endl;
			exit(1);
			}

#endif

	S->add_element(diag_line);
	for (i = 4; i < set_size; i++) {
		pt = set_in[idx[i]];
		for (j = 0; j < P->r; j++) {
			h = P->Lines_on_point[pt * P->r + j];
			if (!S->is_contained(h)) {
				S->add_element(h);
				}
			}
		}

	cout << "we created a blocking set of lines of "
			"size " << S->k << ":" << endl;
	lint_vec_print(cout, S->set, S->k);
	cout << endl;


	int *pt_type;

	pt_type = NEW_int(P->N_points);

	P->point_types_of_line_set(S->set, S->k, pt_type, 0);

	tally C;

	C.init(pt_type, P->N_points, FALSE, 0);


	cout << "the point types are:" << endl;
	C.print_naked(FALSE /*f_backwards*/);
	cout << endl;

#if 0
	for (i = 0; i <= P->N_points; i++) {
		if (pt_type[i]) {
			cout << i << "^" << pt_type[i] << " ";
			}
		}
	cout << endl;
#endif

	sz = ((q * q) >> 1) + ((3 * q) >> 1) - 4;

	if (S->k != sz) {
		cout << "the size does not match the expected size" << endl;
		exit(1);
		}

	cout << "the size is OK" << endl;

	the_set_out = NEW_lint(sz);
	set_size_out = sz;

	for (i = 0; i < sz; i++) {
		j = S->set[i];
		the_set_out[i] = P->Polarity_hyperplane_to_point[j];
		}



	FREE_OBJECT(P);
}

void finite_field::create_hyperoval(
	int f_translation, int translation_exponent,
	int f_Segre, int f_Payne, int f_Cherowitzo, int f_OKeefe_Penttila,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int n = 2;
	int i, d;
	int *v;
	sorting Sorting;
	char str[1000];

	d = n + 1;
	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::create_hyperoval" << endl;
		}

	if (f_v) {
		cout << "finite_field::create_hyperoval before P->init" << endl;
		}
	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level /*MINIMUM(verbose_level - 1, 3)*/);
	if (f_v) {
		cout << "create_hyperoval after P->init" << endl;
		}

	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);

	sprintf(str, "_q%d.txt", q);

	if (f_translation) {
		P->create_translation_hyperoval(Pts, nb_pts,
				translation_exponent, verbose_level - 0);
		fname.assign("hyperoval_translation");
		fname.append(str);
		}
	else if (f_Segre) {
		P->create_Segre_hyperoval(Pts, nb_pts, verbose_level - 2);
		fname.assign("hyperoval_Segre");
		fname.append(str);
		}
	else if (f_Payne) {
		P->create_Payne_hyperoval(Pts, nb_pts, verbose_level - 2);
		fname.assign("hyperoval_Payne");
		fname.append(str);
		}
	else if (f_Cherowitzo) {
		P->create_Cherowitzo_hyperoval(Pts, nb_pts, verbose_level - 2);
		fname.assign("hyperoval_Cherowitzo");
		fname.append(str);
		}
	else if (f_OKeefe_Penttila) {
		P->create_OKeefe_Penttila_hyperoval_32(Pts, nb_pts,
				verbose_level - 2);
		fname.assign("hyperoval_OKeefe_Penttila");
		fname.append(str);
		}
	else {
		P->create_regular_hyperoval(Pts, nb_pts, verbose_level - 2);
		fname.assign("hyperoval_regular");
		fname.append(str);
		}

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				int_vec_print(cout, v, d);
				cout << endl;
				}
			}
		}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "create_hyperoval the set is not a set, "
				"something is wrong" << endl;
		exit(1);
		}

	FREE_OBJECT(P);
	FREE_int(v);
	//FREE_int(L);
}

void finite_field::create_subiaco_oval(
	int f_short,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	char str[1000];

	if (f_v) {
		cout << "finite_field::create_subiaco_oval" << endl;
		}

	sprintf(str, "_q%d.txt", q);
	Subiaco_oval(Pts, nb_pts, f_short, verbose_level);
	if (f_short) {
		fname.assign("oval_subiaco_short");
		fname.append(str);
		}
	else {
		fname.assign("oval_subiaco_long");
		fname.append(str);
		}


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;
		projective_space *P;

		v = NEW_int(d);
		P = NEW_OBJECT(projective_space);


		P->init(n, this,
			FALSE /* f_init_incidence_structure */,
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				int_vec_print(cout, v, d);
				cout << endl;
				}
			}
		FREE_int(v);
		FREE_OBJECT(P);
		}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "create_subiaco_oval the set is not a set, "
				"something is wrong" << endl;
		exit(1);
		}

}


void finite_field::create_subiaco_hyperoval(
		std::string &fname, int &nb_pts, long int *&Pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	sorting Sorting;
	char str[1000];

	if (f_v) {
		cout << "finite_field::create_subiaco_hyperoval" << endl;
		}

	Subiaco_hyperoval(Pts, nb_pts, verbose_level);

	sprintf(str, "_q%d.txt", q);

	fname.assign("subiaco_hyperoval");
	fname.append(str);


	if (f_v) {
		int i;
		int n = 2, d = n + 1;
		int *v;
		projective_space *P;

		v = NEW_int(d);
		P = NEW_OBJECT(projective_space);


		P->init(n, this,
			FALSE /* f_init_incidence_structure */,
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
		cout << "i : point : projective rank" << endl;
		for (i = 0; i < nb_pts; i++) {
			P->unrank_point(v, Pts[i]);
			if (f_v) {
				cout << setw(4) << i << " : ";
				int_vec_print(cout, v, d);
				cout << endl;
				}
			}
		FREE_int(v);
		FREE_OBJECT(P);
		}

	if (!Sorting.test_if_set_with_return_value_lint(Pts, nb_pts)) {
		cout << "finite_field::create_subiaco_hyperoval "
				"the set is not a set, "
				"something is wrong" << endl;
		exit(1);
		}

}

void finite_field::create_ovoid(
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int n = 3, epsilon = -1;
	int c1 = 1, c2 = 0, c3 = 0;
	int i, j, d, h;
	int *v, *w;
	geometry_global Gg;

	d = n + 1;
	P = NEW_OBJECT(projective_space);


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = Gg.nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_int(n + 1);
	w = NEW_int(n + 1);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	choose_anisotropic_form(c1, c2, c3, verbose_level);
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		for (h = 0; h < d; h++) {
			w[h] = v[h];
			}
		j = P->rank_point(w);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points on the ovoid:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];

	sprintf(str, "_q%d.txt", q);


	fname.assign("ovoid");
	fname.append(str);

	//write_set_to_file(fname, L, N, verbose_level);

	FREE_OBJECT(P);
	FREE_int(v);
	FREE_int(w);
	//FREE_int(L);
}

void finite_field::create_Baer_substructure(int n,
	finite_field *Fq,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
// the big field FQ is given
{
	projective_space *P2;
	int q = Fq->q;
	int Q = q;
	int sz;
	int *v;
	int d = n + 1;
	int i, j, a, b, index, f_is_in_subfield;
	number_theory_domain NT;

	//Q = q * q;
	P2 = NEW_OBJECT(projective_space);

	P2->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level);

	if (q != NT.i_power_j(p, e >> 1)) {
		cout << "q != i_power_j(p, e >> 1)" << endl;
		exit(1);
		}

	cout << "Q=" << Q << endl;
	cout << "q=" << q << endl;

	index = (Q - 1) / (q - 1);
	cout << "index=" << index << endl;

	v = NEW_int(d);
	Pts = NEW_lint(P2->N_points);
	sz = 0;
	for (i = 0; i < P2->N_points; i++) {
		PG_element_unrank_modified(v, 1, d, i);
		for (j = 0; j < d; j++) {
			a = v[j];
			b = log_alpha(a);
			f_is_in_subfield = FALSE;
			if (a == 0 || (b % index) == 0) {
				f_is_in_subfield = TRUE;
				}
			if (!f_is_in_subfield) {
				break;
				}
			}
		if (j == d) {
			Pts[nb_pts++] = i;
			}
		}
	cout << "the Baer substructure PG(" << n << "," << q
			<< ") inside PG(" << n << "," << Q << ") has size "
			<< sz << ":" << endl;
	for (i = 0; i < sz; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;



	char str[1000];
	sprintf(str, "_%d_%d.txt", n, Q);
	//write_set_to_file(fname, S, sz, verbose_level);


	fname.assign("Baer_substructure_in_PG");
	fname.append(str);

	FREE_int(v);
	//FREE_int(S);
	FREE_OBJECT(P2);
}

void finite_field::create_BLT_from_database(int f_embedded,
	int BLT_k,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int epsilon = 0;
	int n = 4;
	int c1 = 0, c2 = 0, c3 = 0;
	int d = 5;
	int *BLT;
	int *v;
	knowledge_base K;

	nb_pts = q + 1;

	BLT = K.BLT_representative(q, BLT_k);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "i : orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, BLT[i], 0 /* verbose_level */);
		if (f_embedded) {
			PG_element_rank_modified(v, 1, d, j);
			}
		else {
			j = BLT[i];
			}
		// recreate v:
		Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, BLT[i], 0 /* verbose_level */);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : " << setw(4) << BLT[i] << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	if (f_embedded) {
		sprintf(str, "%d_%d_embedded.txt", q, BLT_k);
		fname.assign("BLT_");
		fname.append(str);
		}
	else {
		sprintf(str, "%d_%d.txt", q, BLT_k);
		fname.assign("BLT_");
		fname.append(str);
		}
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	//FREE_int(L);
	//delete F;
}



void finite_field::create_orthogonal(int epsilon, int n,
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c1 = 1, c2 = 0, c3 = 0;
	int i, j;
	int d = n + 1;
	int *v;
	geometry_global Gg;

	nb_pts = Gg.nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (epsilon == -1) {
		choose_anisotropic_form(c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		PG_element_rank_modified(v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];

	algebra_global AG;

	sprintf(str, "Q%s_%d_%d.txt", AG.plus_minus_letter(epsilon), n, q);
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	//FREE_int(L);
}


void finite_field::create_hermitian(int n,
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int d = n + 1;
	int *v;
	hermitian *H;

	H = NEW_OBJECT(hermitian);
	H->init(this, d, verbose_level - 1);

	nb_pts = H->cnt_Sbar[d];

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "hermitian rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		H->Sbar_unrank(v, d, i, 0 /*verbose_level*/);
		PG_element_rank_modified(v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points:" << endl;
	cout << nb_pts << endl;
	for (i = 0; i < nb_pts; i++) {
		cout << Pts[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	sprintf(str, "H_%d_%d.txt", n, q);
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	FREE_OBJECT(H);
	//FREE_int(L);
}

void finite_field::create_cuspidal_cubic(
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int n = 2;
	long int i, a, d, s, t;
	int *v;
	int v2[2];

	d = n + 1;
	P = NEW_OBJECT(projective_space);


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = q + 1;

	v = NEW_int(d);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		v[0] = mult(power(s, 3), power(t, 0));
		v[1] = mult(power(s, 2), power(t, 1));
		v[2] = mult(power(s, 0), power(t, 3));
#if 0
		for (j = 0; j < d; j++) {
			v[j] = mult(power(s, n - j), power(t, j));
		}
#endif
		a = P->rank_point(v);
		Pts[i] = a;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << a << endl;
			}
		}

#if 0
	cout << "list of points on the cubic:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	sprintf(str, "cuspidal_cubic_%d.txt", q);
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	long int nCk;
	combinatorics_domain Combi;
	int k = 6;
	int rk;
	int idx[6];
	int *subsets;

	nCk = Combi.int_n_choose_k(nb_pts, k);
	subsets = NEW_int(nCk * k);
	for (rk = 0; rk < nCk; rk++) {
		Combi.unrank_k_subset(rk, idx, nb_pts, k);
		for (i = 0; i < k; i++) {
			subsets[rk * k + i] = Pts[idx[i]];
		}
	}

	string fname2;

	sprintf(str, "cuspidal_cubic_%d_subsets_%d.txt", q, k);
	fname2.assign(str);

	{

		ofstream fp(fname2);

		for (rk = 0; rk < nCk; rk++) {
			fp << k;
			for (i = 0; i < k; i++) {
				fp << " " << subsets[rk * k + i];
			}
			fp << endl;
		}
		fp << -1 << endl;

	}

	file_io Fio;

	cout << "Written file " << fname2 << " of size " << Fio.file_size(fname2) << endl;





	FREE_OBJECT(P);
	FREE_int(v);
	//FREE_int(L);
}

void finite_field::create_twisted_cubic(
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int n = 3;
	long int i, j, d, s, t;
	int *v;
	int v2[2];

	d = n + 1;
	P = NEW_OBJECT(projective_space);


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = q + 1;

	v = NEW_int(n + 1);
	Pts = NEW_lint(P->N_points);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(v2, 1, 2, i);
		s = v2[0];
		t = v2[1];
		v[0] = mult(power(s, 3), power(t, 0));
		v[1] = mult(power(s, 2), power(t, 1));
		v[2] = mult(power(s, 1), power(t, 2));
		v[3] = mult(power(s, 0), power(t, 3));
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points on the twisted cubic:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	sprintf(str, "twisted_cubic_%d.txt", q);
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);

	FREE_OBJECT(P);
	FREE_int(v);
	//FREE_int(L);
}


void finite_field::create_elliptic_curve(
	int elliptic_curve_b, int elliptic_curve_c,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int n = 2;
	long int i, a, d;
	int *v;
	elliptic_curve *E;

	d = n + 1;
	P = NEW_OBJECT(projective_space);


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = q + 1;

	E = NEW_OBJECT(elliptic_curve);
	v = NEW_int(n + 1);
	Pts = NEW_lint(P->N_points);

	E->init(this, elliptic_curve_b, elliptic_curve_c,
			verbose_level);

	nb_pts = E->nb;

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		PG_element_rank_modified_lint(E->T + i * d, 1, d, a);
		Pts[i] = a;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, E->T + i * d, d);
			cout << " : " << setw(5) << a << endl;
			}
		}

#if 0
	cout << "list of points on the elliptic curve:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	sprintf(str, "elliptic_curve_b%d_c%d_q%d.txt",
			elliptic_curve_b, elliptic_curve_c, q);
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_OBJECT(E);
	FREE_OBJECT(P);
	FREE_int(v);
	//FREE_int(L);
}

void finite_field::create_ttp_code(finite_field *Fq,
	int f_construction_A, int f_hyperoval, int f_construction_B,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
// this is FQ
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	long int i, j, d;
	int *v;
	int *H_subfield;
	int m, n;
	int f_elements_exponential = TRUE;
	string symbol_for_print_subfield;
	coding_theory_domain Codes;

	if (f_v) {
		cout << "finite_field::create_ttp_code" << endl;
		}

	symbol_for_print_subfield.assign("\\alpha");

	Codes.twisted_tensor_product_codes(
		H_subfield, m, n,
		this, Fq,
		f_construction_A, f_hyperoval,
		f_construction_B,
		verbose_level - 2);

	if (f_v) {
		cout << "H_subfield:" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		//f.latex_matrix(cout, f_elements_exponential,
		//symbol_for_print_subfield, H_subfield, m, n);
		}

	d = m;
	P = NEW_OBJECT(projective_space);


	P->init(d - 1, Fq,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = n;

	if (f_v) {
		cout << "H_subfield:" << endl;
		//print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		Fq->latex_matrix(cout, f_elements_exponential,
			symbol_for_print_subfield, H_subfield, m, n);
		}

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		for (j = 0; j < d; j++) {
			v[j] = H_subfield[j * n + i];
			}
		j = P->rank_point(v);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}

#if 0
	cout << "list of points for the ttp code:" << endl;
	cout << N << endl;
	for (i = 0; i < N; i++) {
		cout << L[i] << " ";
		}
	cout << endl;
#endif

	char str[1000];
	if (f_construction_A) {
		if (f_hyperoval) {
			snprintf(str, 1000, "ttp_code_Ah_%d.txt", Fq->q);
			}
		else {
			snprintf(str, 1000, "ttp_code_A_%d.txt", Fq->q);
			}
		}
	else if (f_construction_B) {
		snprintf(str, 1000, "ttp_code_B_%d.txt", Fq->q);
		}
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);

	FREE_OBJECT(P);
	FREE_int(v);
	FREE_int(H_subfield);
}


void finite_field::create_unital_XXq_YZq_ZYq(
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P2;
	int n = 2;
	int i, rk, d;
	int *v;

	d = n + 1;
	P2 = NEW_OBJECT(projective_space);


	P2->init(2, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_int(d);
	Pts = NEW_lint(P2->N_points);


	P2->create_unital_XXq_YZq_ZYq(Pts, nb_pts, verbose_level - 1);


	if (f_v) {
		cout << "i : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		rk = Pts[i];
		P2->unrank_point(v, rk);
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << rk << endl;
			}
		}


	char str[1000];
	sprintf(str, "unital_XXq_YZq_ZYq_Q%d.txt", q);
	fname.assign(str);

	FREE_OBJECT(P2);
	FREE_int(v);
}


void finite_field::create_whole_space(int n,
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int i; //, d;

	if (f_v) {
		cout << "finite_field::create_whole_space" << endl;
		}
	//d = n + 1;
	P = NEW_OBJECT(projective_space);


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	Pts = NEW_lint(P->N_points);
	nb_pts = P->N_points;
	for (i = 0; i < P->N_points; i++) {
		Pts[i] = i;
		}

	char str[1000];
	sprintf(str, "whole_space_PG_%d_%d.txt", n, q);
	fname.assign(str);

	FREE_OBJECT(P);
}


void finite_field::create_hyperplane(int n,
	int pt,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	long int i, d, a;
	int *v1;
	int *v2;

	if (f_v) {
		cout << "finite_field::create_hyperplane pt=" << pt << endl;
		}
	d = n + 1;
	P = NEW_OBJECT(projective_space);
	v1 = NEW_int(d);
	v2 = NEW_int(d);

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P->unrank_point(v1, pt);

	Pts = NEW_lint(P->N_points);
	nb_pts = 0;
	for (i = 0; i < P->N_points; i++) {
		P->unrank_point(v2, i);
		a = dot_product(d, v1, v2);
		if (a == 0) {
			Pts[nb_pts++] = i;
			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : ";
				int_vec_print(cout, v2, d);
				cout << " : " << setw(5) << i << endl;
				}
			}
		}

	char str[1000];
	sprintf(str, "hyperplane_PG_%d_%d_pt%d.txt", n, q, pt);
	fname.assign(str);

	FREE_OBJECT(P);
	FREE_int(v1);
	FREE_int(v2);
}


void finite_field::create_segre_variety(int a, int b,
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	projective_space *P3;
	int i, j, d, N1, N2, rk;
	int *v1;
	int *v2;
	int *v3;

	if (f_v) {
		cout << "finite_field::create_segre_variety" << endl;
		cout << "a=" << a << " (projective)" << endl;
		cout << "b=" << b << " (projective)" << endl;
		}
	d = (a + 1) * (b + 1);
	if (f_v) {
		cout << "d=" << d << " (vector space dimension)" << endl;
		}
	P1 = NEW_OBJECT(projective_space);
	P2 = NEW_OBJECT(projective_space);
	P3 = NEW_OBJECT(projective_space);
	v1 = NEW_int(a + 1);
	v2 = NEW_int(b + 1);
	v3 = NEW_int(d);

	P1->init(a, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(b, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	P3->init(d - 1, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);


	N1 = P1->N_points;
	N2 = P2->N_points;
	Pts = NEW_lint(N1 * N2);
	nb_pts = 0;
	for (i = 0; i < N1; i++) {
		P1->unrank_point(v1, i);
		for (j = 0; j < N2; j++) {
			P2->unrank_point(v2, j);
			mult_matrix_matrix(v1, v2, v3, a + 1, 1, b + 1,
					0 /* verbose_level */);
			rk = P3->rank_point(v3);
			Pts[nb_pts++] = rk;
			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : " << endl;
				int_matrix_print(v3, a + 1, b + 1);
				cout << " : " << setw(5) << rk << endl;
				}
			}
		}

	char str[1000];
	sprintf(str, "segre_variety_%d_%d_%d.txt", a, b, q);
	fname.assign(str);

	FREE_OBJECT(P1);
	FREE_OBJECT(P2);
	FREE_OBJECT(P3);
	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);
}

void finite_field::create_Maruta_Hamada_arc(
		std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int N;

	if (f_v) {
		cout << "finite_field::create_Maruta_Hamada_arc" << endl;
		}
	P = NEW_OBJECT(projective_space);

	P->init(2, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);


	N = P->N_points;
	Pts = NEW_lint(N);

	P->create_Maruta_Hamada_arc2(Pts, nb_pts, verbose_level);

	char str[1000];
	sprintf(str, "Maruta_Hamada_arc2_q%d.txt", q);
	fname.assign(str);

	FREE_OBJECT(P);
	//FREE_int(Pts);
}


void finite_field::create_desarguesian_line_spread_in_PG_3_q(
	finite_field *Fq,
	int f_embedded_in_PG_4_q,
	std::string &fname, int &nb_lines, long int *&Lines,
	int verbose_level)
// this is FQ
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	projective_space *P1, *P3;
	//finite_field *FQ, *Fq;
	int q = Fq->q;
	int Q = q * q;
	int j, h, rk, rk1, alpha, d;
	int *w1, *w2, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;

	P1 = NEW_OBJECT(projective_space);
	P3 = NEW_OBJECT(projective_space);

#if 0
	if (Q != FQ->q) {
		cout << "create_desarguesian_line_spread_in_PG_3_q "
				"Q != FQ->q" << endl;
		exit(1);
		}
#endif
	if (f_v) {
		cout << "create_desarguesian_line_spread_in_PG_3_q" << endl;
		cout << "f_embedded_in_PG_4_q=" << f_embedded_in_PG_4_q << endl;
		}

	P1->init(1, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	if (f_embedded_in_PG_4_q) {
		P3->init(4, Fq,
			TRUE /* f_init_incidence_structure */,
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

		d = 5;
		}
	else {
		P3->init(3, Fq,
			TRUE /* f_init_incidence_structure */,
			verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

		d = 4;
		}



	subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding, verbose_level - 3);

		// we think of FQ as two dimensional vector space
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have
		// pair_embedding[i * q + j] = x;
		// also,
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element
		// in FQ that corresponds to i

		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_vv) {
		print_embedding(*Fq,
			components, embedding, pair_embedding);
		}
	alpha = p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}


	nb_lines = Q + 1;
	Lines = NEW_lint(nb_lines);


	w1 = NEW_int(d);
	w2 = NEW_int(d);
	v2 = NEW_int(2);

	int ee;

	ee = e >> 1;
	if (f_vv) {
		cout << "ee=" << ee << endl;
		}


	int a, a0, a1;
	int b, b0, b1;

	if (f_v) {
		cout << "rk : w1,w2 : line rank" << endl;
		}
	for (rk = 0; rk < nb_lines; rk++) {
		if (f_vv) {
			cout << "rk=" << rk << endl;
			}
		P1->unrank_point(v2, rk);
			// w1[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2).
			// That way, w1[4] and w2[4] are a GF(q)-basis for the
			// 2-dimensional subspace v[2] (when viewed over GF(q)),
			// which is an element of the regular spread.
		if (f_vv) {
			cout << "v2=";
			int_vec_print(cout, v2, 2);
			cout << endl;
			}

		for (h = 0; h < 2; h++) {
			a = v2[h];
			a0 = components[a * 2 + 0];
			a1 = components[a * 2 + 1];
			b = mult(a, alpha);
			b0 = components[b * 2 + 0];
			b1 = components[b * 2 + 1];
			w1[2 * h + 0] = a0;
			w1[2 * h + 1] = a1;
			w2[2 * h + 0] = b0;
			w2[2 * h + 1] = b1;
			}
		if (f_embedded_in_PG_4_q) {
			w1[4] = 0;
			w2[4] = 0;
			}
		if (f_vv) {
			cout << "w1=";
			int_vec_print(cout, w1, 4);
			cout << "w2=";
			int_vec_print(cout, w2, 4);
			cout << endl;
			}

		for (j = 0; j < d; j++) {
			P3->Grass_lines->M[0 * d + j] = w1[j];
			P3->Grass_lines->M[1 * d + j] = w2[j];
			}
		if (f_vv) {
			cout << "before P3->Grass_lines->rank_int:" << endl;
			int_matrix_print(P3->Grass_lines->M, 2, 4);
			}
		rk1 = P3->Grass_lines->rank_lint(0 /* verbose_level*/);
		Lines[rk] = rk1;
		if (f_vv) {
			cout << setw(4) << rk << " : ";
			int_vec_print(cout, w1, d);
			cout << ", ";
			int_vec_print(cout, w2, d);
			cout << " : " << setw(5) << rk1 << endl;
			}
		}

	char str[1000];
	if (f_embedded_in_PG_4_q) {
		sprintf(str, "desarguesian_line_spread_"
				"in_PG_3_%d_embedded.txt", q);
		}
	else {
		sprintf(str, "desarguesian_line_spread_"
				"in_PG_3_%d.txt", q);
		}
	fname.assign(str);

	FREE_OBJECT(P1);
	FREE_OBJECT(P3);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(v2);
	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);
}




void finite_field::do_Klein_correspondence(int n,
		long int *set_in, int set_size,
		long int *&the_set_out, int &set_size_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	projective_space *P5;

	if (f_v) {
		cout << "finite_field::do_Klein_correspondence" << endl;
		}
	if (n != 3) {
		cout << "finite_field::do_Klein_correspondence n != 3" << endl;
		exit(1);
		}

	P = NEW_OBJECT(projective_space);

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	P5 = NEW_OBJECT(projective_space);

	P5->init(5, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	the_set_out = NEW_lint(set_size);
	set_size_out = set_size;

	P->klein_correspondence(P5,
		set_in, set_size, the_set_out, verbose_level);


	FREE_OBJECT(P);
	FREE_OBJECT(P5);
}

void finite_field::do_m_subspace_type(int n, int m,
		long int *set, int set_size,
	int f_show, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	int j, a, N;
	int d = n + 1;
	int *v;
	int *intersection_numbers;
	sorting Sorting;

	if (f_v) {
		cout << "finite_field::do_m_subspace_type" << endl;
		cout << "We will now compute the m_subspace type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "do_m_subspace_type before P->init" << endl;
		}


	P->init(n, this,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_m_subspace_type after P->init" << endl;
		}


	v = NEW_int(d);

	N = P->nb_rk_k_subspaces_as_lint(m + 1);
	if (f_v) {
		cout << "do_m_subspace_type N = " << N << endl;
		}

	intersection_numbers = NEW_int(N);
	if (f_v) {
		cout << "after allocating intersection_numbers" << endl;
		}
	if (m == 1) {
		P->line_intersection_type_basic(set, set_size,
				intersection_numbers, verbose_level - 1);
		}
	else if (m == 2) {
		P->plane_intersection_type_basic(set, set_size,
				intersection_numbers, verbose_level - 1);
		}
	else if (m == n - 1) {
		P->hyperplane_intersection_type_basic(set, set_size,
				intersection_numbers, verbose_level - 1);
		}
	else {
		cout << "finite_field::do_m_subspace_type m=" << m
				<< " not implemented" << endl;
		exit(1);
		}

	tally C;
	int f_second = FALSE;

	C.init(intersection_numbers, N, f_second, 0);
	if (f_v) {
		cout << "finite_field::do_m_subspace_type: " << m
				<< "-subspace intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}

	if (f_show) {
		int h, f, l, b;
		int *S;
		//int *basis;
		grassmann *G;

		G = NEW_OBJECT(grassmann);

		G->init(d, m + 1, this, 0 /* verbose_level */);

		//basis = NEW_int((m + 1) * d);
		S = NEW_int(N);
		for (h = 0; h < C.nb_types; h++) {
			f = C.type_first[h];
			l = C.type_len[h];
			a = C.data_sorted[f];
			if (f_v) {
				cout << a << "-spaces: ";
				}
			for (j = 0; j < l; j++) {
				b = C.sorting_perm_inv[f + j];
				S[j] = b;
				}
			Sorting.int_vec_quicksort_increasingly(S, l);
			if (f_v) {
				int_vec_print(cout, S, l);
				cout << endl;
				}


			for (j = 0; j < l; j++) {

				long int *intersection_set;
				int intersection_set_size;

				b = S[j];
				G->unrank_lint(b, 0);

				cout << "subspace " << j << " / " << l << " which is "
						<< b << " has a basis:" << endl;
				print_integer_matrix_width(cout,
						G->M, m + 1, d, d, P->F->log10_of_q);


				P->intersection_of_subspace_with_point_set(
					G, b, set, set_size,
					intersection_set,
					intersection_set_size,
					verbose_level);
				cout << "intersection set of size "
						<< intersection_set_size << ":" << endl;
				P->print_set(intersection_set, intersection_set_size);

				FREE_lint(intersection_set);
				}
			}
		FREE_int(S);
		//FREE_int(basis);
		FREE_OBJECT(G);
#if 0
		cout << "i : intersection number of plane i" << endl;
		for (i = 0; i < N_planes; i++) {
			cout << setw(4) << i << " : " << setw(3)
					<< intersection_numbers[i] << endl;
			}
#endif
		}

	FREE_int(v);
	FREE_int(intersection_numbers);
	FREE_OBJECT(P);
}

void finite_field::do_m_subspace_type_fast(int n, int m,
		long int *set, int set_size,
	int f_show, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	grassmann *G;
	int i;
	int N;
	int d = n + 1;
	int *v;
	longinteger_object *R;
	long int **Pts_on_plane;
	int *nb_pts_on_plane;
	int len;

	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast" << endl;
		cout << "We will now compute the m_subspace type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast before P->init" << endl;
		}

	P->init(n, this,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast after P->init" << endl;
		}


	v = NEW_int(d);

	N = P->nb_rk_k_subspaces_as_lint(m + 1);
	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast N = " << N << endl;
		}

	G = NEW_OBJECT(grassmann);

	G->init(n + 1, m + 1, this, 0 /* verbose_level */);

	if (m == 2) {
		P->plane_intersection_type_fast(G, set, set_size,
			R, Pts_on_plane, nb_pts_on_plane, len,
			verbose_level - 1);
		}
	else {
		cout << "finite_field::do_m_subspace_type m=" << m
				<< " not implemented" << endl;
		exit(1);
		}

	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast: We found "
				<< len << " planes." << endl;
#if 1
		for (i = 0; i < len; i++) {
			cout << setw(3) << i << " : " << R[i]
				<< " : " << setw(5) << nb_pts_on_plane[i] << " : ";
			lint_vec_print(cout, Pts_on_plane[i], nb_pts_on_plane[i]);
			cout << endl;
			}
#endif
		}

	tally C;
	int f_second = FALSE;

	C.init(nb_pts_on_plane, len, f_second, 0);
	if (f_v) {
		cout << "finite_field::do_m_subspace_type_fast: " << m
				<< "-subspace intersection type: ";
		C.print(FALSE /*f_backwards*/);
		}


	// we will now look at the subspaces that intersect in
	// the largest number of points:


	int *Blocks;
	int f, a, b, j, nb_planes, intersection_size, u;
	int *S;
	sorting Sorting;
	//int *basis;
	//grassmann *G;

	S = NEW_int(N);
	intersection_size = C.nb_types - 1;

	f = C.type_first[intersection_size];
	nb_planes = C.type_len[intersection_size];
	intersection_size = C.data_sorted[f];
	if (f_v) {
		cout << intersection_size << "-spaces: ";
		}
	for (j = 0; j < nb_planes; j++) {
		b = C.sorting_perm_inv[f + j];
		S[j] = b;
		}
	Sorting.int_vec_quicksort_increasingly(S, nb_planes);
	if (f_v) {
		int_vec_print(cout, S, nb_planes);
		cout << endl;
		}



	Blocks = NEW_int(nb_planes * intersection_size);


	for (i = 0; i < nb_planes; i++) {

		long int *intersection_set;

		b = S[i];
		G->unrank_longinteger(R[b], 0);

		cout << "subspace " << i << " / " << nb_planes << " which is "
			<< R[b] << " has a basis:" << endl;
		print_integer_matrix_width(cout,
				G->M, m + 1, d, d, P->F->log10_of_q);


		P->intersection_of_subspace_with_point_set_rank_is_longinteger(
			G, R[b], set, set_size,
			intersection_set, u, verbose_level);


		if (u != intersection_size) {
			cout << "u != intersection_size" << endl;
			cout << "u=" << u << endl;
			cout << "intersection_size=" << intersection_size << endl;
			exit(1);
			}
		cout << "intersection set of size " << intersection_size
				<< ":" << endl;
		P->print_set(intersection_set, intersection_size);

		for (j = 0; j < intersection_size; j++) {
			a = intersection_set[j];
			if (!Sorting.lint_vec_search_linear(set, set_size, a, b)) {
				cout << "did not find point" << endl;
				exit(1);
				}
			Blocks[i * intersection_size + j] = b;
			}

		FREE_lint(intersection_set);
		} // next i

	cout << "Blocks:" << endl;
	int_matrix_print(Blocks, nb_planes, intersection_size);

	int *Incma;
	int *ItI;
	int *IIt;
	int g;

	if (f_v) {
		cout << "Computing plane invariant for " << nb_planes
				<< " planes:" << endl;
		}
	Incma = NEW_int(set_size * nb_planes);
	ItI = NEW_int(nb_planes * nb_planes);
	IIt = NEW_int(set_size * set_size);
	for (i = 0; i < set_size * nb_planes; i++) {
		Incma[i] = 0;
		}
	for (u = 0; u < nb_planes; u++) {
		for (g = 0; g < intersection_size; g++) {
			i = Blocks[u * intersection_size + g];
			Incma[i * nb_planes + u] = 1;
			}
		}

	cout << "Incma:" << endl;
	int_matrix_print(Incma, set_size, nb_planes);

	for (i = 0; i < nb_planes; i++) {
		for (j = 0; j < nb_planes; j++) {
			a = 0;
			for (u = 0; u < set_size; u++) {
				a += Incma[u * nb_planes + i] *
						Incma[u * nb_planes + j];
				}
			ItI[i * nb_planes + j] = a;
			}
		}

	cout << "I^t*I:" << endl;
	int_matrix_print(ItI, nb_planes, nb_planes);

	for (i = 0; i < set_size; i++) {
		for (j = 0; j < set_size; j++) {
			a = 0;
			for (u = 0; u < nb_planes; u++) {
				a += Incma[i * nb_planes + u] *
						Incma[j * nb_planes + u];
				}
			IIt[i * set_size + j] = a;
			}
		}

	cout << "I*I^t:" << endl;
	int_matrix_print(IIt, set_size, set_size);


	FREE_int(Incma);
	FREE_int(ItI);
	FREE_int(IIt);
	FREE_int(Blocks);
	FREE_int(S);
	//FREE_int(basis);
	FREE_OBJECT(G);


	FREE_int(v);
	FREE_OBJECT(P);
}

void finite_field::do_line_type(int n,
		long int *set, int set_size,
	int f_show, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	projective_space *P;
	int j, a;
	int *v;
	int *intersection_numbers;

	if (f_v) {
		cout << "finite_field::do_line_type" << endl;
		cout << "We will now compute the line type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::do_line_type before P->init" << endl;
		}

	P->init(n, this,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_line_type after P->init" << endl;
		}


	v = NEW_int(n + 1);


	intersection_numbers = NEW_int(P->N_lines);
	if (f_v) {
		cout << "after allocating intersection_numbers" << endl;
		}
	P->line_intersection_type(set, set_size,
			intersection_numbers, 0 /* verbose_level */);

#if 0
	for (i = 0; i < P->N_lines; i++) {
		intersection_numbers[i] = 0;
		}
	for (i = 0; i < set_size; i++) {
		a = set[i];
		for (h = 0; h < P->r; h++) {
			j = P->Lines_on_point[a * P->r + h];
			//if (j == 17) {
			//	cout << "set point " << i << " which is "
			//<< a << " lies on line 17" << endl;
			//	}
			intersection_numbers[j]++;
			}
		}
#endif

	tally C;
	int f_second = FALSE;
	sorting Sorting;

	C.init(intersection_numbers, P->N_lines, f_second, 0);
	if (TRUE) {
		cout << "finite_field::do_line_type: line intersection type: ";
		C.print(TRUE /*f_backwards*/);
		}

	if (f_vv) {
		int h, f, l, b;
		int *S;
		int *basis;
		long int *I;

		I = NEW_lint(P->N_points);
		basis = NEW_int(2 * (P->n + 1));
		S = NEW_int(P->N_lines);
		for (h = 0; h < C.nb_types; h++) {
			f = C.type_first[h];
			l = C.type_len[h];
			a = C.data_sorted[f];
			if (f_v) {
				cout << a << "-lines: ";
				}
			for (j = 0; j < l; j++) {
				b = C.sorting_perm_inv[f + j];
				S[j] = b;
				}
			Sorting.int_vec_quicksort_increasingly(S, l);
			if (f_v) {
				int_vec_print(cout, S, l);
				cout << endl;
				}
			for (j = 0; j < l; j++) {
				b = S[j];
				P->unrank_line(basis, b);
				if (f_show) {
					cout << "line " << b << " has a basis:" << endl;
					print_integer_matrix_width(cout,
							basis, 2, P->n + 1, P->n + 1,
							P->F->log10_of_q);
					}
				int sz;

				P->intersect_with_line(set, set_size,
						a /* line_rk */, I, sz,
						0 /* verbose_level*/);

				if (f_show) {
					cout << "intersects in " << sz << " points : ";
					lint_vec_print(cout, I, sz);
					cout << endl;
					cout << "they are:" << endl;
					P->print_set(I, sz);
					}

				}
		}
		FREE_lint(I);
		FREE_int(S);
		FREE_int(basis);
#if 0
		cout << "i : intersection number of line i" << endl;
		for (i = 0; i < P->N_lines; i++) {
			cout << setw(4) << i << " : " << setw(3)
					<< intersection_numbers[i] << endl;
			}
#endif
		}

	FREE_int(v);
	FREE_int(intersection_numbers);
	FREE_OBJECT(P);
}

void finite_field::do_plane_type(int n,
		long int *set, int set_size,
	int *&intersection_type, int &highest_intersection_number,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	grassmann *G;
	projective_space *P;

	if (f_v) {
		cout << "finite_field::do_plane_type" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "do_plane_type before P->init" << endl;
		}

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_plane_type after P->init" << endl;
		}

	G = NEW_OBJECT(grassmann);

	G->init(n + 1, 3, this, 0 /*verbose_level - 2*/);

	P->plane_intersection_type(G,
		set, set_size,
		intersection_type, highest_intersection_number,
		verbose_level - 2);

	//FREE_int(intersection_type);
	FREE_OBJECT(G);
	FREE_OBJECT(P);
}

void finite_field::do_plane_type_failsafe(int n,
		long int *set, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int N_planes;
	int *type;

	if (f_v) {
		cout << "finite_field::do_plane_type_failsafe" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::do_plane_type_failsafe before P->init" << endl;
		}

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_plane_type_failsafe after P->init" << endl;
		}


	N_planes = P->nb_rk_k_subspaces_as_lint(3);
	type = NEW_int(N_planes);

	P->plane_intersection_type_basic(set, set_size,
		type,
		verbose_level - 2);


	tally C;

	C.init(type, N_planes, FALSE, 0);
	cout << "The plane type is:" << endl;
	C.print(FALSE /*f_backwards*/);

	FREE_int(type);
	FREE_OBJECT(P);
	if (f_v) {
		cout << "finite_field::do_plane_type_failsafe done" << endl;
		}
}

void finite_field::do_conic_type(int n,
	int f_randomized, int nb_times,
	long int *set, int set_size,
	int *&intersection_type, int &highest_intersection_number,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int f_save_largest_sets = FALSE;
	set_of_sets *largest_sets = NULL;

	if (f_v) {
		cout << "finite_field::do_conic_type" << endl;
		cout << "We will now compute the plane type" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::do_conic_type before P->init" << endl;
		}

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_conic_type after P->init" << endl;
		}


	P->conic_intersection_type(f_randomized, nb_times,
		set, set_size,
		intersection_type, highest_intersection_number,
		f_save_largest_sets, largest_sets,
		verbose_level - 2);

	FREE_OBJECT(P);
}

void finite_field::do_test_diagonal_line(int n,
		long int *set_in, int set_size,
	std::string &fname_orbits_on_quadrangles,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int h;

	if (f_v) {
		cout << "finite_field::do_test_diagonal_line" << endl;
		cout << "fname_orbits_on_quadrangles="
				<< fname_orbits_on_quadrangles << endl;
		}
	if (n != 2) {
		cout << "finite_field::do_test_diagonal_line we need n = 2" << endl;
		exit(1);
		}
	if (ODD(q)) {
		cout << "finite_field::do_test_diagonal_line we need q even" << endl;
		exit(1);
		}
	if (set_size != q + 2) {
		cout << "finite_field::do_test_diagonal_line "
				"we need set_size == q + 2" << endl;
		exit(1);
		}
	P = NEW_OBJECT(projective_space);

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);


	int f_casenumbers = FALSE;
	int *Casenumbers;
	int nb_cases;
	//char **data;
	long int **sets;
	int *set_sizes;
	char **Ago_ascii;
	char **Aut_ascii;
	file_io Fio;

	int *Nb;
	sorting Sorting;


#if 0
	read_and_parse_data_file(fname_orbits_on_quadrangles,
		nb_cases, data, sets, set_sizes);
#endif

	Fio.read_and_parse_data_file_fancy(fname_orbits_on_quadrangles,
		f_casenumbers,
		nb_cases,
		set_sizes, sets, Ago_ascii, Aut_ascii,
		Casenumbers,
		verbose_level);

	if (f_v) {
		cout << "read " << nb_cases << " orbits on qudrangles" << endl;
		}

	Nb = NEW_int(nb_cases);

	for (h = 0; h < nb_cases; h++) {


		int pt[4];
		int p_idx[4];
		int line[6];
		int diag_pts[3];
		int diag_line;
		int nb;
		int i, j;
		long int a;
		int basis[6];


		cout << "orbit " << h << " : ";
		lint_vec_print(cout, sets[h], set_sizes[h]);
		cout << endl;

		if (set_sizes[h] != 4) {
			cout << "size != 4" << endl;
			exit(1);
			}

		for (i = 0; i < 4; i++) {
			a = sets[h][i];
			pt[i] = a;
			if (!Sorting.lint_vec_search_linear(set_in, set_size, a, j)) {
				cout << "the point " << a << " is not contained "
						"in the hyperoval" << endl;
				exit(1);
				}
			p_idx[i] = j;
			}

		cout << "p_idx[4]: ";
		int_vec_print(cout, p_idx, 4);
		cout << endl;

		line[0] = P->line_through_two_points(pt[0], pt[1]);
		line[1] = P->line_through_two_points(pt[0], pt[2]);
		line[2] = P->line_through_two_points(pt[0], pt[3]);
		line[3] = P->line_through_two_points(pt[1], pt[2]);
		line[4] = P->line_through_two_points(pt[1], pt[3]);
		line[5] = P->line_through_two_points(pt[2], pt[3]);

		cout << "line[6]: ";
		int_vec_print(cout, line, 6);
		cout << endl;


		diag_pts[0] = P->intersection_of_two_lines(line[0], line[5]);
		diag_pts[1] = P->intersection_of_two_lines(line[1], line[4]);
		diag_pts[2] = P->intersection_of_two_lines(line[2], line[3]);

		cout << "diag_pts[3]: ";
		int_vec_print(cout, diag_pts, 3);
		cout << endl;


		diag_line = P->line_through_two_points(
				diag_pts[0], diag_pts[1]);
		cout << "The diagonal line is " << diag_line << endl;

		P->unrank_line(basis, diag_line);
		int_matrix_print(basis, 2, 3);

		if (diag_line != P->line_through_two_points(
				diag_pts[0], diag_pts[2])) {
			cout << "diaginal points not collinear!" << endl;
			exit(1);
			}
		nb = 0;
		for (i = 0; i < set_size; i++) {
			a = set_in[i];
			if (P->is_incident(a, diag_line)) {
				nb++;
				}
			}
		cout << "nb=" << nb << endl;
		Nb[h] = nb;

		if (nb == 0) {
			cout << "the diagonal line is external!" << endl;
			}
		else if (nb == 2) {
			cout << "the diagonal line is secant" << endl;
			}
		else {
			cout << "something else" << endl;
			}


		} // next h

	cout << "h : Nb[h]" << endl;
	for (h = 0; h < nb_cases; h++) {
		cout << setw(3) << h << " : " << setw(3) << Nb[h] << endl;
		}
	int l0, l2;
	int *V0, *V2;
	int i, a;

	l0 = 0;
	l2 = 0;
	V0 = NEW_int(nb_cases);
	V2 = NEW_int(nb_cases);
	for (i = 0; i < nb_cases; i++) {
		if (Nb[i] == 0) {
			V0[l0++] = i;
			}
		else {
			V2[l2++] = i;
			}
		}
	cout << "external orbits:" << endl;
	for (i = 0; i < l0; i++) {
		a = V0[i];
		cout << i << " : " << a << " : " << Ago_ascii[a] << endl;
		}
	cout << "secant orbits:" << endl;
	for (i = 0; i < l2; i++) {
		a = V2[i];
		cout << i << " : " << a << " : " << Ago_ascii[a] << endl;
		}
	cout << "So, there are " << l0 << " external diagonal orbits "
			"and " << l2 << " secant diagonal orbits" << endl;

	FREE_OBJECT(P);
}

void finite_field::do_andre(finite_field *Fq,
		long int *the_set_in, int set_size_in,
		long int *&the_set_out, int &set_size_out,
	int verbose_level)
// this is FQ
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	projective_space *P2, *P4;
	int a, a0, a1;
	int b, b0, b1;
	int i, h, k, alpha; //, d;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;

	if (f_v) {
		cout << "finite_field::do_andre for a set of size " << set_size_in << endl;
		}
	P2 = NEW_OBJECT(projective_space);
	P4 = NEW_OBJECT(projective_space);


	P2->init(2, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P4->init(4, Fq,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	//d = 5;


	if (f_v) {
		cout << "before subfield_embedding_2dimensional" << endl;
		}

	subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding, verbose_level);

		// we think of FQ as two dimensional vector space
		// over Fq with basis (1,alpha)
		// for i,j \in Fq, with x = i + j * alpha \in FQ, we have
		// pair_embedding[i * q + j] = x;
		// also,
		// components[x * 2 + 0] = i;
		// components[x * 2 + 1] = j;
		// also, for i \in Fq, embedding[i] is the element
		// in FQ that corresponds to i

		// components[Q * 2]
		// embedding[q]
		// pair_embedding[q * q]

	if (f_v) {
		cout << "after  subfield_embedding_2dimensional" << endl;
		}
	if (f_vv) {
		print_embedding(*Fq,
			components, embedding, pair_embedding);
		}
	alpha = p;
	if (f_vv) {
		cout << "alpha=" << alpha << endl;
		//FQ->print(TRUE /* f_add_mult_table */);
		}


	v = NEW_int(3);
	w1 = NEW_int(5);
	w2 = NEW_int(5);
	w3 = NEW_int(5);
	v2 = NEW_int(2);


	the_set_out = NEW_lint(P4->N_points);
	set_size_out = 0;

	for (i = 0; i < set_size_in; i++) {
		if (f_vv) {
			cout << "input point " << i << " is "
					<< the_set_in[i] << " : ";
			}
		P2->unrank_point(v, the_set_in[i]);
		PG_element_normalize(v, 1, 3);
		if (f_vv) {
			int_vec_print(cout, v, 3);
			cout << " becomes ";
			}

		if (v[2] == 0) {

			// we are dealing with a point on the line at infinity.
			// Such a point corresponds to a line of the spread.
			// We create the line and then create all
			// q + 1 points on that line.

			if (f_vv) {
				cout << endl;
				}
			// w1[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2]
			// w2[4] is the GF(q)-vector corresponding
			// to the GF(q^2)-vector v[2] * alpha
			// where v[2] runs through the points of PG(1,q^2).
			// That way, w1[4] and w2[4] are a GF(q)-basis for the
			// 2-dimensional subspace v[2] (when viewed over GF(q)),
			// which is an element of the regular spread.

			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				b = mult(a, alpha);
				b0 = components[b * 2 + 0];
				b1 = components[b * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				w2[2 * h + 0] = b0;
				w2[2 * h + 1] = b1;
				}
			if (FALSE) {
				cout << "w1=";
				int_vec_print(cout, w1, 4);
				cout << "w2=";
				int_vec_print(cout, w2, 4);
				cout << endl;
				}

			// now we create all points on the line spanned
			// by w1[4] and w2[4]:
			// There are q + 1 of these points.
			// We make sure that the coordinate vectors have
			// a zero in the last spot.

			for (h = 0; h < Fq->q + 1; h++) {
				Fq->PG_element_unrank_modified(v2, 1, 2, h);
				if (FALSE) {
					cout << "v2=";
					int_vec_print(cout, v2, 2);
					cout << " : ";
					}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]),
							Fq->mult(v2[1], w2[k]));
					}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					int_vec_print(cout, w3, 5);
					}
				a = P4->rank_point(w3);
				if (f_vv) {
					cout << " rank " << a << endl;
					}
				the_set_out[set_size_out++] = a;
				}
			}
		else {

			// we are dealing with an affine point:
			// We make sure that the coordinate vector
			// has a one in the last spot.


			for (h = 0; h < 2; h++) {
				a = v[h];
				a0 = components[a * 2 + 0];
				a1 = components[a * 2 + 1];
				w1[2 * h + 0] = a0;
				w1[2 * h + 1] = a1;
				}
			w1[4] = 1;
			if (f_vv) {
				//cout << "w1=";
				int_vec_print(cout, w1, 5);
				}
			a = P4->rank_point(w1);
			if (f_vv) {
				cout << " rank " << a << endl;
				}
			the_set_out[set_size_out++] = a;
			}
		}

	if (f_v) {
		for (i = 0; i < set_size_out; i++) {
			a = the_set_out[i];
			P4->unrank_point(w1, a);
			cout << setw(3) << i << " : " << setw(5) << a << " : ";
			int_vec_print(cout, w1, 5);
			cout << endl;
			}
		}

	FREE_OBJECT(P2);
	FREE_OBJECT(P4);
	FREE_int(v);
	FREE_int(w1);
	FREE_int(w2);
	FREE_int(w3);
	FREE_int(v2);
	FREE_int(components);
	FREE_int(embedding);
	FREE_int(pair_embedding);
}

void finite_field::do_print_lines_in_PG(int n,
		long int *set_in, int set_size)
{
	projective_space *P;
	int d = n + 1;
	long int h, a;
	int f_elements_exponential = TRUE;
	string symbol_for_print;

	P = NEW_OBJECT(projective_space);

	symbol_for_print.assign("\\alpha");

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P->Grass_lines->unrank_lint(a, 0 /* verbose_level */);
		cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
		latex_matrix(cout, f_elements_exponential,
			symbol_for_print, P->Grass_lines->M, 2, d);
		cout << endl;
		}
	FREE_OBJECT(P);
}

void finite_field::do_print_points_in_PG(int n,
		long int *set_in, int set_size)
{
	projective_space *P;
	int d = n + 1;
	long int h, a;
	//int f_elements_exponential = TRUE;
	string symbol_for_print;
	int *v;

	P = NEW_OBJECT(projective_space);

	symbol_for_print.assign("\\alpha");

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	v = NEW_int(d);
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P->unrank_point(v, a);
		cout << setw(5) << h << " : " << setw(5) << a << " : ";
		int_vec_print(cout, v, d);
		cout << " : ";
		int_vec_print_elements_exponential(cout,
				v, d, symbol_for_print);
		cout << endl;
		}
	FREE_int(v);
	FREE_OBJECT(P);
}

void finite_field::do_print_points_in_orthogonal_space(
	int epsilon, int n,
	long int *set_in, int set_size, int verbose_level)
{
	int d = n + 1;
	long int h, a;
	//int f_elements_exponential = TRUE;
	string symbol_for_print;
	int *v;
	orthogonal *O;

	symbol_for_print.assign("\\alpha");

	O = NEW_OBJECT(orthogonal);

	O->init(epsilon, d, this, verbose_level - 1);

	v = NEW_int(d);
	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		O->unrank_point(v, 1, a, 0);
		//cout << setw(5) << h << " : ";
		cout << setw(5) << a << " & ";
		if (e > 1) {
			int_vec_print_elements_exponential(cout,
					v, d, symbol_for_print);
			}
		else {
			int_vec_print(cout, v, d);
			}
		cout << "\\\\" << endl;
		}
	FREE_int(v);
	FREE_OBJECT(O);
}

void finite_field::do_print_points_on_grassmannian(
	int n, int k,
	long int *set_in, int set_size)
{
	grassmann *Grass;
	projective_space *P;
	int d = n + 1;
	int h, a;
	int f_elements_exponential = TRUE;
	string symbol_for_print;

	P = NEW_OBJECT(projective_space);
	Grass = NEW_OBJECT(grassmann);

	//N = generalized_binomial(n + 1, k + 1, q);
	//r = generalized_binomial(k + 1, 1, q);

	symbol_for_print.assign("\\alpha");


	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);
	Grass->init(n + 1, k + 1, this, 0);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		//cout << "unrank " << a << endl;
		Grass->unrank_lint(a, 0 /* verbose_level */);
		cout << setw(5) << h << " : " << setw(5) << a << " :" << endl;
		latex_matrix(cout, f_elements_exponential,
			symbol_for_print, Grass->M, k + 1, d);
		cout << endl;
		}
	FREE_OBJECT(P);
	FREE_OBJECT(Grass);
}

void finite_field::do_embed_orthogonal(
	int epsilon, int n,
	long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P;
	int *v;
	int d = n + 1;
	long int h, a, b;
	int c1 = 0, c2 = 0, c3 = 0;

	if (f_v) {
		cout << "finite_field::do_embed_orthogonal" << endl;
		}
	P = NEW_OBJECT(projective_space);

	P->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	if (epsilon == -1) {
		choose_anisotropic_form(c1, c2, c3, verbose_level);
		}

	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, a, 0 /* verbose_level */);
		b = P->rank_point(v);
		set_out[h] = b;
		}

	FREE_int(v);
	FREE_OBJECT(P);

}

void finite_field::do_embed_points(int n,
		long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	projective_space *P1;
	projective_space *P2;
	int *v;
	int d = n + 2;
	long int h, a, b;

	if (f_v) {
		cout << "finite_field::do_embed_points" << endl;
		}
	P1 = NEW_OBJECT(projective_space);
	P2 = NEW_OBJECT(projective_space);

	P1->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
	P2->init(n + 1, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		P1->unrank_point(v, a);
		v[d - 1] = 0;
		b = P2->rank_point(v);
		set_out[h] = b;
		}

	FREE_int(v);
	FREE_OBJECT(P1);
	FREE_OBJECT(P2);

}

void finite_field::do_draw_points_in_plane(
		long int *set, int set_size,
	std::string &fname_base, int f_point_labels,
	int f_embedded, int f_sideways,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	int n = 2;
	int rad = 17000;

	if (f_v) {
		cout << "finite_field::do_draw_points_in_plane" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "finite_field::do_draw_points_in_plane before P->init" << endl;
		}

	P->init(n, this,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);

	if (f_v) {
		cout << "finite_field::do_draw_points_in_plane after P->init" << endl;
		}

	P->draw_point_set_in_plane(
			fname_base, set, set_size,
			TRUE /*f_with_points*/, f_point_labels,
			f_embedded, f_sideways, rad,
			verbose_level - 2);
	FREE_OBJECT(P);

	if (f_v) {
		cout << "do_draw_points_in_plane done" << endl;
		}

}

void finite_field::do_ideal(int n,
		long int *set_in, int set_size, int degree,
		long int *&set_out, int &size_out,
		monomial_ordering_type Monomial_ordering_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	homogeneous_polynomial_domain *HPD;
	int *Kernel;
	int *w1;
	int *w2;
	long int *Pts;
	int nb_pts;
	int r, h, ns;
	geometry_global Gg;

	if (f_v) {
		cout << "finite_field::do_ideal" << endl;
	}

	size_out = 0;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	if (f_v) {
		cout << "finite_field::do_ideal before HPD->init" << endl;
	}
	HPD->init(this, n + 1, degree,
		FALSE /* f_init_incidence_structure */,
		Monomial_ordering_type,
		verbose_level - 2);
	if (f_v) {
		cout << "finite_field::do_ideal after HPD->init" << endl;
	}

	Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
	w1 = NEW_int(HPD->get_nb_monomials());
	w2 = NEW_int(HPD->get_nb_monomials());

	if (f_v) {
		cout << "finite_field::do_ideal before HPD->vanishing_ideal" << endl;
	}
	HPD->vanishing_ideal(set_in, set_size,
			r, Kernel, 0 /*verbose_level */);
	if (f_v) {
		cout << "finite_field::do_ideal after HPD->vanishing_ideal" << endl;
	}

	ns = HPD->get_nb_monomials() - r; // dimension of null space
	cout << "The system has rank " << r << endl;
	cout << "The ideal has dimension " << ns << endl;
	cout << "and is generated by:" << endl;
	int_matrix_print(Kernel, ns, HPD->get_nb_monomials());
	cout << "corresponding to the following basis "
			"of polynomials:" << endl;
	for (h = 0; h < ns; h++) {
		HPD->print_equation(cout, Kernel + h * HPD->get_nb_monomials());
		cout << endl;
		}

	cout << "looping over all generators of the ideal:" << endl;
	for (h = 0; h < ns; h++) {
		cout << "generator " << h << " / " << ns << ":" << endl;

		vector<long int> Points;
		int i;

		HPD->enumerate_points(Kernel + h * HPD->get_nb_monomials(),
				Points, verbose_level);
		nb_pts = Points.size();

		Pts = NEW_lint(nb_pts);
		for (i = 0; i < nb_pts; i++) {
			Pts[i] = Points[i];
		}


		cout << "We found " << nb_pts << " points on the generator of the ideal" << endl;
		cout << "They are : ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);


		if (h == 0) {
			size_out = HPD->get_nb_monomials();
			set_out = NEW_lint(size_out);
			//int_vec_copy(Kernel + h * HPD->nb_monomials, set_out, size_out);
			int u;
			for (u = 0; u < size_out; u++) {
				set_out[u] = Kernel[h * HPD->get_nb_monomials() + u];
			}
			//break;
		}
		FREE_lint(Pts);

	}

#if 0
	int N;
	int *Pts;
	cout << "looping over all elements of the ideal:" << endl;
	N = Gg.nb_PG_elements(ns - 1, q);
	for (h = 0; h < N; h++) {
		PG_element_unrank_modified(w1, 1, ns, h);
		cout << "element " << h << " / " << N << " w1=";
		int_vec_print(cout, w1, ns);
		mult_vector_from_the_left(w1, Kernel, w2, ns, HPD->get_nb_monomials());
		cout << " w2=";
		int_vec_print(cout, w2, HPD->get_nb_monomials());
		HPD->enumerate_points(w2, Pts, nb_pts, verbose_level);
		cout << " We found " << nb_pts << " points on this curve" << endl;
		}
#endif

	FREE_OBJECT(HPD);
	FREE_int(Kernel);
	FREE_int(w1);
	FREE_int(w2);
}


void finite_field::PG_element_modified_not_in_subspace_perm(int n, int m,
	long int *orbit, long int *orbit_inv,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v = NEW_int(n + 1);
	geometry_global Gg;
	long int l = Gg.nb_PG_elements(n, q);
	long int ll = Gg.nb_PG_elements_not_in_subspace(n, m, q);
	long int i, j1 = 0, j2 = ll, f_in, j;

	if (f_v) {
		cout << "finite_field::PG_element_modified_not_in_subspace_perm" << endl;
	}
	for (i = 0; i < l; i++) {
		PG_element_unrank_modified_lint(v, 1, n + 1, i);
		f_in = Gg.PG_element_modified_is_in_subspace(n, m, v);
		if (f_v) {
			cout << i << " : ";
			for (j = 0; j < n + 1; j++) {
				cout << v[j] << " ";
				}
			}
		if (f_in) {
			if (f_v) {
				cout << " is in the subspace" << endl;
				}
			orbit[j2] = i;
			orbit_inv[i] = j2;
			j2++;
			}
		else {
			if (f_v) {
				cout << " is not in the subspace" << endl;
				}
			orbit[j1] = i;
			orbit_inv[i] = j1;
			j1++;
			}
		}
	if (j1 != ll) {
		cout << "j1 != ll" << endl;
		exit(1);
		}
	if (j2 != l) {
		cout << "j2 != l" << endl;
		exit(1);
		}
	FREE_int(v);
	if (f_v) {
		cout << "finite_field::PG_element_modified_not_in_subspace_perm done" << endl;
	}
}

void finite_field::print_set_in_affine_plane(int len, long int *S)
{
	int *A;
	int i, j, x, y, v[3];


	A = NEW_int(q * q);
	for (x = 0; x < q; x++) {
		for (y = 0; y < q; y++) {
			A[(q - 1 - y) * q + x] = 0;
			}
		}
	for (i = 0; i < len; i++) {
		PG_element_unrank_modified(
				v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			//cout << "my_generator::print_set_in_affine_plane
			// not an affine point" << endl;
			cout << "(" << v[0] << "," << v[1]
					<< "," << v[2] << ")" << endl;
			continue;
			}
		x = v[0];
		y = v[1];
		A[(q - 1 - y) * q + x] = 1;
		}
	for (i = 0; i < q; i++) {
		for (j = 0; j < q; j++) {
			cout << A[i * q + j];
			}
		cout << endl;
		}
	FREE_int(A);
}

void finite_field::elliptic_curve_addition(int b, int c,
	int x1, int x2, int x3,
	int y1, int y2, int y3,
	int &z1, int &z2, int &z3, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, two, three, top, bottom, m;

	if (f_v) {
		cout << "finite_field::elliptic_curve_addition" << endl;
	}

	my_nb_calls_to_elliptic_curve_addition++;
	if (x3 == 0) {
		z1 = y1;
		z2 = y2;
		z3 = y3;
		goto done;
	}
	if (y3 == 0) {
		z1 = x1;
		z2 = x2;
		z3 = x3;
		goto done;
	}
	if (x3 != 1) {
		a = inverse(x3);
		x1 = mult(x1, a);
		x2 = mult(x2, a);
	}
	if (y3 != 1) {
		a = inverse(y3);
		y1 = mult(y1, a);
		y2 = mult(y2, a);
	}
	if (x1 == y1 && x2 != y2) {
		if (negate(x2) != y2) {
			cout << "x1 == y1 && x2 != y2 && negate(x2) != y2" << endl;
			exit(1);
		}
		z1 = 0;
		z2 = 1;
		z3 = 0;
		goto done;
	}
	if (x1 == y1 && x2 == 0 && y2 == 0) {
		z1 = 0;
		z2 = 1;
		z3 = 0;
		goto done;
	}
	if (x1 == y1 && x2 == y2) {
		two = add(1, 1);
		three = add(two, 1);
		top = add(mult(three, mult(x1, x1)), b);
		bottom = mult(two, x2);
		a = inverse(bottom);
		m = mult(top, a);
	}
	else {
		top = add(y2, negate(x2));
		bottom = add(y1, negate(x1));
		a = inverse(bottom);
		m = mult(top, a);
	}
	z1 = add(add(mult(m, m), negate(x1)), negate(y1));
	z2 = add(mult(m, add(x1, negate(z1))), negate(x2));
	z3 = 1;
done:
	if (f_v) {
		cout << "finite_field::elliptic_curve_addition done" << endl;
	}
}

void finite_field::elliptic_curve_point_multiple(int b, int c, int n,
	int x1, int y1, int z1,
	int &x3, int &y3, int &z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int bx, by, bz;
	int cx, cy, cz;
	int tx, ty, tz;

	if (f_v) {
		cout << "finite_field::elliptic_curve_point_multiple" << endl;
	}
	bx = x1;
	by = y1;
	bz = z1;
	cx = 0;
	cy = 1;
	cz = 0;
	while (n) {
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";

			elliptic_curve_addition(b, c,
				bx, by, bz,
				cx, cy, cz,
				tx, ty, tz, verbose_level - 1);
			cx = tx;
			cy = ty;
			cz = tz;
			//c = mult(b, c);
			//cout << c << endl;
		}
		elliptic_curve_addition(b, c,
			bx, by, bz,
			bx, by, bz,
			tx, ty, tz, verbose_level - 1);
		bx = tx;
		by = ty;
		bz = tz;
		//b = mult(b, b);
		n >>= 1;
		//cout << "finite_field::power: " << b << "^" << n << " * " << c << endl;
	}
	x3 = cx;
	y3 = cy;
	z3 = cz;
	if (f_v) {
		cout << "finite_field::elliptic_curve_point_multiple done" << endl;
	}
}

void finite_field::elliptic_curve_point_multiple_with_log(int b, int c, int n,
	int x1, int y1, int z1,
	int &x3, int &y3, int &z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int bx, by, bz;
	int cx, cy, cz;
	int tx, ty, tz;

	if (f_v) {
		cout << "finite_field::elliptic_curve_point_multiple_with_log" << endl;
	}
	bx = x1;
	by = y1;
	bz = z1;
	cx = 0;
	cy = 1;
	cz = 0;
	cout << "ECMultiple$\\Big((" << bx << "," << by << "," << bz << "),";
	cout << "(" << cx << "," << cy << "," << cz << "),"
			<< n << "," << b << "," << c << "," << p << "\\Big)$\\\\" << endl;

	while (n) {
		if (n % 2) {
			//cout << "finite_field::power: mult(" << b << "," << c << ")=";

			elliptic_curve_addition(b, c,
				bx, by, bz,
				cx, cy, cz,
				tx, ty, tz, verbose_level - 1);
			cx = tx;
			cy = ty;
			cz = tz;
			//c = mult(b, c);
			//cout << c << endl;
		}
		elliptic_curve_addition(b, c,
			bx, by, bz,
			bx, by, bz,
			tx, ty, tz, verbose_level - 1);
		bx = tx;
		by = ty;
		bz = tz;
		//b = mult(b, b);
		n >>= 1;
		cout << "=ECMultiple$\\Big((" << bx << "," << by << "," << bz << "),";
		cout << "(" << cx << "," << cy << "," << cz << "),"
				<< n << "," << b << "," << c << "," << p << "\\Big)$\\\\" << endl;
		//cout << "finite_field::power: " << b << "^" << n << " * " << c << endl;
	}
	x3 = cx;
	y3 = cy;
	z3 = cz;
	cout << "$=(" << x3 << "," << y3 << "," << z3 << ")$\\\\" << endl;
	if (f_v) {
		cout << "finite_field::elliptic_curve_point_multiple_with_log done" << endl;
	}
}

int finite_field::elliptic_curve_evaluate_RHS(int x, int b, int c)
{
	int x2, x3, e;

	x2 = mult(x, x);
	x3 = mult(x2, x);
	e = add(x3, mult(b, x));
	e = add(e, c);
	return e;
}

void finite_field::elliptic_curve_points(
		int b, int c, int &nb, int *&T, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//finite_field F;
	int x, y, n;
	int r, l;
	number_theory_domain NT;
	longinteger_domain D;

	if (f_v) {
		cout << "finite_field::elliptic_curve_points" << endl;
	}
	nb = 0;
	//F.init(p, verbose_level);
	for (x = 0; x < p; x++) {
		r = elliptic_curve_evaluate_RHS(x, b, c);
		if (r == 0) {
			if (f_v) {
				cout << nb << " : (" << x << "," << 0 << ",1)" << endl;
			}
			nb++;
		}
		else {
			if (p != 2) {
				if (e > 1) {
					cout << "finite_field::elliptic_curve_points odd characteristic and e > 1" << endl;
					exit(1);
				}
				l = NT.Legendre(r, p, verbose_level - 1);
				if (l == 1) {
					//y = sqrt_mod_involved(r, p);
					y = D.square_root_mod(r, p, 0 /* verbose_level*/);
					//y = NT.sqrt_mod_simple(r, p);
					if (f_v) {
						cout << nb << " : (" << x << "," << y << ",1)" << endl;
						cout << nb + 1 << " : (" << x << "," << negate(y) << ",1)" << endl;
					}
					nb += 2;
				}
			}
			else {
				y = frobenius_power(r, e - 1);
				if (f_v) {
					cout << nb << " : (" << x << "," << y << ",1)" << endl;
				}
				nb += 1;
			}
		}
	}
	if (f_v) {
		cout << nb << " : (0,1,0)" << endl;
	}
	nb++;
	if (f_v) {
		cout << "the curve has " << nb << " points" << endl;
	}
	T = NEW_int(nb * 3);
	n = 0;
	for (x = 0; x < p; x++) {
		r = elliptic_curve_evaluate_RHS(x, b, c);
		if (r == 0) {
			T[n * 3 + 0] = x;
			T[n * 3 + 1] = 0;
			T[n * 3 + 2] = 1;
			n++;
			//cout << nb++ << " : (" << x << "," << 0 << ",1)" << endl;
		}
		else {
			if (p != 2) {
				// odd characteristic:
				l = NT.Legendre(r, p, verbose_level - 1);
				if (l == 1) {
					//y = sqrt_mod_involved(r, p);
					//y = NT.sqrt_mod_simple(r, p);
					y = D.square_root_mod(r, p, 0 /* verbose_level*/);
					T[n * 3 + 0] = x;
					T[n * 3 + 1] = y;
					T[n * 3 + 2] = 1;
					n++;
					T[n * 3 + 0] = x;
					T[n * 3 + 1] = negate(y);
					T[n * 3 + 2] = 1;
					n++;
					//cout << nb++ << " : (" << x << "," << y << ",1)" << endl;
					//cout << nb++ << " : (" << x << "," << F.negate(y) << ",1)" << endl;
				}
			}
			else {
				// even characteristic
				y = frobenius_power(r, e - 1);
				T[n * 3 + 0] = x;
				T[n * 3 + 1] = y;
				T[n * 3 + 2] = 1;
				n++;
				//cout << nb++ << " : (" << x << "," << y << ",1)" << endl;
			}
		}
	}
	T[n * 3 + 0] = 0;
	T[n * 3 + 1] = 1;
	T[n * 3 + 2] = 0;
	n++;
	//print_integer_matrix_width(cout, T, nb, 3, 3, log10_of_q);
	if (f_v) {
		cout << "finite_field::elliptic_curve_points done" << endl;
		cout << "the curve has " << nb << " points" << endl;
	}
}

void finite_field::elliptic_curve_all_point_multiples(int b, int c, int &order,
	int x1, int y1, int z1,
	std::vector<std::vector<int> > &Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x2, y2, z2;
	int x3, y3, z3;

	if (f_v) {
		cout << "finite_field::elliptic_curve_all_point_multiples" << endl;
	}
	order = 1;

	x2 = x1;
	y2 = y1;
	z2 = z1;
	while (TRUE) {
		{
			vector<int> pts;

			pts.push_back(x2);
			pts.push_back(y2);
			pts.push_back(z2);

			Pts.push_back(pts);
		}
		if (z2 == 0) {
			return;
		}

		elliptic_curve_addition(b, c,
			x1, y1, z1,
			x2, y2, z2,
			x3, y3, z3, 0 /*verbose_level */);

		x2 = x3;
		y2 = y3;
		z2 = z3;

		order++;
	}
	if (f_v) {
		cout << "finite_field::elliptic_curve_all_point_multiples done" << endl;
	}
}

int finite_field::elliptic_curve_discrete_log(int b, int c,
	int x1, int y1, int z1,
	int x3, int y3, int z3,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int x2, y2, z2;
	int a3, b3, c3;
	int n;

	if (f_v) {
		cout << "finite_field::elliptic_curve_discrete_log" << endl;
	}
	n = 1;

	x2 = x1;
	y2 = y1;
	z2 = z1;
	while (TRUE) {
		if (x2 == x3 && y2 == y3 && z2 == z3) {
			break;
		}

		elliptic_curve_addition(b, c,
			x1, y1, z1,
			x2, y2, z2,
			a3, b3, c3, 0 /*verbose_level */);

		n++;

		x2 = a3;
		y2 = b3;
		z2 = c3;

	}
	if (f_v) {
		cout << "finite_field::elliptic_curve_discrete_log done" << endl;
	}
	return n;
}



void finite_field::simeon(int n, int len, long int *S, int s, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 1);
	int k, nb_rows, nb_cols, nb_r1, nb_r2, row, col;
	int *Coord;
	int *M;
	int *A;
	int *C;
	int *T;
	int *Ac; // no not free
	int *U;
	int *U1;
	int nb_A, nb_U;
	int a, u, ac, i, d, idx, mtx_rank;
	combinatorics_domain Combi;
	sorting Sorting;

	if (f_v) {
		cout << "finite_field::simeon s=" << s << endl;
	}
	k = n + 1;
	nb_cols = Combi.int_n_choose_k(len, k - 1);
	nb_r1 = Combi.int_n_choose_k(len, s);
	nb_r2 = Combi.int_n_choose_k(len - s, k - 2);
	nb_rows = nb_r1 * nb_r2;
	if (f_v) {
		cout << "nb_r1=" << nb_r1 << endl;
		cout << "nb_r2=" << nb_r2 << endl;
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
	}

	Coord = NEW_int(len * k);
	M = NEW_int(nb_rows * nb_cols);
	A = NEW_int(len);
	U = NEW_int(k - 2);
	U1 = NEW_int(k - 2);
	C = NEW_int(k - 1);
	T = NEW_int(k * k);

	int_vec_zero(M, nb_rows * nb_cols);


	// unrank all points of the arc:
	for (i = 0; i < len; i++) {
		//point_unrank(Coord + i * k, S[i]);
		PG_element_unrank_modified(Coord + i * k, 1 /* stride */, n + 1 /* len */, S[i]);
	}


	nb_A = Combi.int_n_choose_k(len, k - 2);
	nb_U = Combi.int_n_choose_k(len - (k - 2), k - 1);
	if (nb_A * nb_U != nb_rows) {
		cout << "nb_A * nb_U != nb_rows" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "nb_A=" << nb_A << endl;
		cout << "nb_U=" << nb_U << endl;
	}


	Ac = A + k - 2;

	row = 0;
	for (a = 0; a < nb_A; a++) {
		if (f_vv) {
			cout << "a=" << a << " / " << nb_A << ":" << endl;
		}
		Combi.unrank_k_subset(a, A, len, k - 2);
		Combi.set_complement(A, k - 2, Ac, ac, len);
		if (ac != len - (k - 2)) {
			cout << "arc_generator::simeon ac != len - (k - 2)" << endl;
			exit(1);
		}
		if (f_vv) {
			cout << "Ac=";
			int_vec_print(cout, Ac, ac);
			cout << endl;
		}


		for (u = 0; u < nb_U; u++, row++) {

			Combi.unrank_k_subset(u, U, len - (k - 2), k - 1);
			for (i = 0; i < k - 1; i++) {
				U1[i] = Ac[U[i]];
			}
			if (f_vv) {
				cout << "U1=";
				int_vec_print(cout, U1, k - 1);
				cout << endl;
			}

			for (col = 0; col < nb_cols; col++) {
				if (f_vv) {
					cout << "row=" << row << " / " << nb_rows
							<< " col=" << col << " / "
							<< nb_cols << ":" << endl;
				}
				Combi.unrank_k_subset(col, C, len, k - 1);
				if (f_vv) {
					cout << "C: ";
					int_vec_print(cout, C, k - 1);
					cout << endl;
				}


				// test if A is a subset of C:
				for (i = 0; i < k - 2; i++) {
					if (!Sorting.int_vec_search_linear(C, k - 1, A[i], idx)) {
						//cout << "did not find A[" << i << "] in C" << endl;
						break;
					}
				}
				if (i == k - 2) {
					d = BallChowdhury_matrix_entry(
							Coord, C, U1, k, s /*sz_U */,
						T, 0 /*verbose_level*/);
					if (f_vv) {
						cout << "d=" << d << endl;
					}

					M[row * nb_cols + col] = d;
				} // if
			} // next col
		} // next u
	} // next a

	if (f_v) {
		cout << "simeon, the matrix M is:" << endl;
		//int_matrix_print(M, nb_rows, nb_cols);
	}

	//print_integer_matrix_with_standard_labels(cout, M,
	//nb_rows, nb_cols, TRUE /* f_tex*/);
	//int_matrix_print_tex(cout, M, nb_rows, nb_cols);

	if (f_v) {
		cout << "nb_rows=" << nb_rows << endl;
		cout << "nb_cols=" << nb_cols << endl;
		cout << "s=" << s << endl;
	}

	mtx_rank = Gauss_easy(M, nb_rows, nb_cols);
	if (f_v) {
		cout << "mtx_rank=" << mtx_rank << endl;
		//cout << "simeon, the reduced matrix M is:" << endl;
		//int_matrix_print(M, mtx_rank, nb_cols);
	}


	FREE_int(Coord);
	FREE_int(M);
	FREE_int(A);
	FREE_int(C);
	//FREE_int(E);
#if 0
	FREE_int(A1);
	FREE_int(C1);
	FREE_int(E1);
	FREE_int(A2);
	FREE_int(C2);
#endif
	FREE_int(T);
	if (f_v) {
		cout << "finite_field::simeon s=" << s << " done" << endl;
	}
}


void finite_field::wedge_to_klein(int *W, int *K)
{
	K[0] = W[0]; // 12
	K[1] = W[5]; // 34
	K[2] = W[1]; // 13
	K[3] = negate(W[4]); // 24
	K[4] = W[2]; // 14
	K[5] = W[3]; // 23
}

void finite_field::klein_to_wedge(int *K, int *W)
{
	W[0] = K[0];
	W[1] = K[2];
	W[2] = K[4];
	W[3] = K[5];
	W[4] = negate(K[3]);
	W[5] = K[1];
}


void finite_field::isomorphism_to_special_orthogonal(int *A4, int *A6, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "finite_field::isomorphism_to_special_orthogonal" << endl;
	}
	int i, j;
	int Basis1[] = {
			1,0,0,0,0,0,
			0,1,0,0,0,0,
			0,0,1,0,0,0,
			0,0,0,1,0,0,
			0,0,0,0,1,0,
			0,0,0,0,0,1,
	};
	int Basis2[36];
	int An2[37];
	int v[6];
	int w[6];
	int C[36];
	int D[36];
	int B[] = {
			1,0,0,0,0,0,
			0,0,0,2,0,0,
			1,3,0,0,0,0,
			0,0,0,1,3,0,
			1,0,2,0,0,0,
			0,0,0,2,0,4,
	};
	int Bv[36];
	sorting Sorting;

	for (i = 0; i < 6; i++) {
		klein_to_wedge(Basis1 + i * 6, Basis2 + i * 6);
	}

	matrix_inverse(B, Bv, 6, 0 /* verbose_level */);




	exterior_square(A4, An2, 4, 0 /*verbose_level*/);

	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"exterior_square :" << endl;
		int_matrix_print(An2, 6, 6);
		cout << endl;
	}


	for (j = 0; j < 6; j++) {
		mult_vector_from_the_left(Basis2 + j * 6, An2, v, 6, 6);
				// v[m], A[m][n], vA[n]
		wedge_to_klein(v, w);
		int_vec_copy(w, C + j * 6, 6);
	}


	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"orthogonal matrix :" << endl;
		int_matrix_print(C, 6, 6);
		cout << endl;
	}

	mult_matrix_matrix(Bv, C, D, 6, 6, 6, 0 /*verbose_level */);
	mult_matrix_matrix(D, B, A6, 6, 6, 6, 0 /*verbose_level */);

	PG_element_normalize_from_front(A6, 1, 36);

	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"orthogonal matrix in the special form:" << endl;
		int_matrix_print(A6, 6, 6);
		cout << endl;
	}

	if (f_v) {
		cout << "finite_field::isomorphism_to_special_orthogonal done" << endl;
	}

}


void finite_field::minimal_orbit_rep_under_stabilizer_of_frame_characteristic_two(int x, int y,
		int &a, int &b, int verbose_level)
{
	int X[6];
	int i, i0;

	X[0] = x;
	X[1] = add(x, 1);
	X[2] = mult(x, inverse(y));
	X[3] = mult(add(x, y), inverse(add(y, 1)));
	X[4] = mult(1, inverse(y));
	X[5] = mult(add(x, y), inverse(x));
	i0 = 0;
	for (i = 1; i < 6; i++) {
		if (X[i] < X[i0]) {
			i0 = i;
		}
	}
	a = X[i0];
	if (i0 == 0) {
		b = y;
	}
	else if (i0 == 1) {
		b = add(y, 1);
	}
	else if (i0 == 2) {
		b = mult(1, inverse(y));
	}
	else if (i0 == 3) {
		b = mult(y, inverse(add(y, 1)));
	}
	else if (i0 == 4) {
		b = mult(x, inverse(y));
	}
	else if (i0 == 5) {
		b = mult(add(x, 1), inverse(x));
	}
}


}}

