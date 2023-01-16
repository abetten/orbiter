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
namespace layer1_foundations {
namespace field_theory {


void finite_field::PG_element_apply_frobenius(int n,
		int *v, int f)
{
	int i;

	for (i = 0; i < n; i++) {
		v[i] = frobenius_power(v[i], f);
	}
}



int finite_field::test_if_vectors_are_projectively_equal(int *v1, int *v2, int len)
{
	int *w1, *w2;
	int i;
	int ret;

	w1 = NEW_int(len);
	w2 = NEW_int(len);

#if 0
	cout << "finite_field::test_if_vectors_are_projectively_equal:" << endl;
	Orbiter->Int_vec.print(cout, v1, 20);
	cout << endl;
	cout << "finite_field::test_if_vectors_are_projectively_equal:" << endl;
	Orbiter->Int_vec.print(cout, v2, 20);
	cout << endl;
#endif

	Int_vec_copy(v1, w1, len);
	Int_vec_copy(v2, w2, len);
	PG_element_normalize(w1, 1, len);
	PG_element_normalize(w2, 1, len);

#if 0
	cout << "finite_field::test_if_vectors_are_projectively_equal:" << endl;
	Orbiter->Int_vec.print(cout, w1, 20);
	cout << endl;
	cout << "finite_field::test_if_vectors_are_projectively_equal:" << endl;
	Orbiter->Int_vec.print(cout, w2, 20);
	cout << endl;
#endif

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
			if (a == 1) {
				return;
			}
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
			if (a == 1) {
				return;
			}
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

void finite_field::PG_element_normalize_from_a_given_position(
		int *v, int stride, int len, int idx)
{
	int j, a, av;

	a = v[idx * stride];
	if (a) {
		if (a == 1) {
			return;
		}
		av = inverse(a);
		for (j = 0; j < len; j++) {
			v[j * stride] = mult(v[j * stride], av);
		}
		return;
	}
	else {
		cout << "finite_field::PG_element_normalize_from_a_given_position "
				"zero vector" << endl;
		exit(1);
	}
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
	Int_vec_zero(v + old_length, new_length - old_length);
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
		if (v[i * stride]) {
			break;
		}
	}
	if (i == len) {
		cout << "finite_field::PG_element_rank_modified "
				"zero vector" << endl;
		exit(1);
	}
	for (j = i + 1; j < len; j++) {
		if (v[j * stride]) {
			break;
		}
	}
	if (j == len) {
		// we have the unit vector vector e_i
		a = i;
		return;
	}

	// test for the all one vector:
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1) {
				break;
			}
		}
		if (j == len) {
			a = len;
			return;
		}
	}


	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride]) {
			break;
		}
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
		if (j > 0) {
			a *= q;
		}
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
		if (a >= sqj) {
			a--;
		}
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
		a++; // take into account the fact that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
				// take into account the fact that the
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
		if (v[i * stride]) {
			break;
		}
	}
	if (i == len) {
		cout << "finite_field::PG_element_rank_modified_lint "
				"zero vector" << endl;
		exit(1);
	}
	for (j = i + 1; j < len; j++) {
		if (v[j * stride]) {
			break;
		}
	}
	if (j == len) {
		// we have the unit vector vector e_i
		a = i;
		return;
	}

	// test for the all one vector:
	if (i == 0 && v[i * stride] == 1) {
		for (j = i + 1; j < len; j++) {
			if (v[j * stride] != 1) {
				break;
			}
		}
		if (j == len) {
			a = len;
			return;
		}
	}


	for (i = len - 1; i >= 0; i--) {
		if (v[i * stride]) {
			break;
		}
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
		if (j > 0) {
			a *= q;
		}
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
		if (a >= sqj) {
			a--;
		}
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
		a++; // take into account the fact that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++;
				// take into account the fact that the
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




void finite_field::all_PG_elements_in_subspace(
		int *genma, int k, int n,
		long int *&point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i;
	long int a;
	combinatorics::combinatorics_domain Combi;

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
			Int_vec_print(cout, message, k);
			cout << endl;
		}
		Linear_algebra->mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			Int_vec_print(cout, word, n);
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
		int *genma, int k, int n,
		long int *point_list, int &nb_points,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int *message;
	int *word;
	int i, j;
	combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "finite_field::all_PG_elements_in_subspace_array_is_given" << endl;
	}
	message = NEW_int(k);
	word = NEW_int(n);
	nb_points = Combi.generalized_binomial(k, 1, q);
	//point_list = NEW_int(nb_points);

	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			Int_vec_print(cout, message, k);
			cout << endl;
		}
		Linear_algebra->mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			Int_vec_print(cout, word, n);
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
		cout << "finite_field::all_PG_elements_in_subspace_array_is_given "
				"done" << endl;
	}
}

void finite_field::display_all_PG_elements(int n)
{
	int *v = NEW_int(n + 1);
	geometry::geometry_global Gg;
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
	geometry::geometry_global Gg;
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
	geometry::geometry_global Gg;
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





}}}


