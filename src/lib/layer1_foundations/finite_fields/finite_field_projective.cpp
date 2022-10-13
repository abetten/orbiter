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
		int *genma, int k, int n, long int *&point_list, int &nb_points,
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
		int *genma, int k, int n, long int *point_list, int &nb_points,
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


void finite_field::do_cone_over(int n,
	long int *set_in, int set_size_in, long int *&set_out, int &set_size_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::do_cone_over" << endl;
	}
	//projective_space *P1;
	//projective_space *P2;
	int *v;
	int d = n + 2;
	int h, u, cnt;
	long int a, b;

#if 0
	P1 = NEW_OBJECT(projective_space);
	P2 = NEW_OBJECT(projective_space);

	P1->init(n, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);

	P2->init(n + 1, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level - 2  /*MINIMUM(verbose_level - 1, 3)*/);
#endif

	v = NEW_int(d);

	set_size_out = 1 + q * set_size_in;
	set_out = NEW_lint(set_size_out);
	cnt = 0;

	// create the vertex:
	Int_vec_zero(v, d);
	v[d - 1] = 1;
	//b = P2->rank_point(v);
	PG_element_rank_modified_lint(v, 1, n + 2, b);
	set_out[cnt++] = b;


	// for each point, create the generator
	// which is the line connecting the point and the vertex
	// since we have created the vertex already,
	// we only need to create q points per line:

	for (h = 0; h < set_size_in; h++) {
		a = set_in[h];
		for (u = 0; u < q; u++) {
			//P1->unrank_point(v, a);
			PG_element_unrank_modified_lint(v, 1, n + 1, a);

			v[d - 1] = u;

			//b = P2->rank_point(v);
			PG_element_rank_modified_lint(v, 1, n + 2, b);

			set_out[cnt++] = b;
		}
	}

	if (cnt != set_size_out) {
		cout << "finite_field::do_cone_over cnt != set_size_out" << endl;
		exit(1);
	}

	FREE_int(v);
	//FREE_OBJECT(P1);
	//FREE_OBJECT(P2);
	if (f_v) {
		cout << "finite_field::do_cone_over done" << endl;
	}
}


void finite_field::do_blocking_set_family_3(int n,
	long int *set_in, int set_size,
	long int *&the_set_out, int &set_size_out,
	int verbose_level)
// creates projective_space PG(2,q)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::do_blocking_set_family_3" << endl;
	}
	geometry::projective_space *P;
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
	P = NEW_OBJECT(geometry::projective_space);

	P->projective_space_init(n, this,
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
	combinatorics::combinatorics_domain Combi;

	data_structures::fancy_set *S;

	S = NEW_OBJECT(data_structures::fancy_set);

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
		Int_matrix_print(basis, 2, 3);
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
			cout << "diagonal points not collinear!" << endl;
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
			h = P->Implementation->Lines_on_point[pt * P->r + j];
			if (!S->is_contained(h)) {
				S->add_element(h);
			}
		}
	}

	cout << "we created a blocking set of lines of "
			"size " << S->k << ":" << endl;
	Lint_vec_print(cout, S->set, S->k);
	cout << endl;


	int *pt_type;

	pt_type = NEW_int(P->N_points);

	P->point_types_of_line_set(S->set, S->k, pt_type, 0);

	data_structures::tally C;

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
		the_set_out[i] = P->Standard_polarity->Hyperplane_to_point[j];
	}



	FREE_OBJECT(P);
}





void finite_field::create_orthogonal(int epsilon, int n,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_orthogonal" << endl;
	}
	int c1 = 1, c2 = 0, c3 = 0;
	int i, j;
	int d = n + 1;
	int *v;
	geometry::geometry_global Gg;

	nb_pts = Gg.nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (epsilon == -1) {
		Linear_algebra->choose_anisotropic_form(c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
		}
	}
	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
	}
	for (i = 0; i < nb_pts; i++) {
		Orthogonal_indexing->Q_epsilon_unrank(v, 1, epsilon, n,
				c1, c2, c3, i, 0 /* verbose_level */);
		PG_element_rank_modified(v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			Int_vec_print(cout, v, d);
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

	algebra::algebra_global AG;

	sprintf(str, "Q%s_%d_%d.txt", AG.plus_minus_letter(epsilon), n, q);
	label_txt.assign(str);

	sprintf(str, "Q%s\\_%d\\_%d.txt", AG.plus_minus_letter(epsilon), n, q);
	label_tex.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	//FREE_int(L);
}


void finite_field::create_hermitian(int n,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// creates hermitian
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_hermitian" << endl;
	}
	int i, j;
	int d = n + 1;
	int *v;
	geometry::hermitian *H;

	H = NEW_OBJECT(geometry::hermitian);
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
			Int_vec_print(cout, v, d);
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
	label_txt.assign(str);

	sprintf(str, "H\\_%d\\_%d.txt", n, q);
	label_tex.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);


	FREE_int(v);
	FREE_OBJECT(H);
	//FREE_int(L);
}

void finite_field::create_ttp_code(finite_field *Fq_subfield,
	int f_construction_A, int f_hyperoval, int f_construction_B,
	std::string &fname, int &nb_pts, long int *&Pts,
	int verbose_level)
// this is FQ
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_ttp_code" << endl;
	}
	geometry::projective_space *P;
	long int i, j, d;
	int *v;
	int *H_subfield;
	int m, n;
	int f_elements_exponential = TRUE;
	string symbol_for_print_subfield;
	coding_theory::ttp_codes Ttp_codes;

	if (f_v) {
		cout << "finite_field::create_ttp_code" << endl;
	}

	symbol_for_print_subfield.assign("\\alpha");

	Ttp_codes.twisted_tensor_product_codes(
		H_subfield, m, n,
		this, Fq_subfield,
		f_construction_A, f_hyperoval,
		f_construction_B,
		verbose_level - 2);

	if (f_v) {
		cout << "H_subfield:" << endl;
		cout << "m=" << m << endl;
		cout << "n=" << n << endl;
		Int_vec_print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		//f.latex_matrix(cout, f_elements_exponential,
		//symbol_for_print_subfield, H_subfield, m, n);
	}

	d = m;
	P = NEW_OBJECT(geometry::projective_space);


	P->projective_space_init(d - 1, Fq_subfield,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);
	nb_pts = n;

	if (f_v) {
		cout << "H_subfield:" << endl;
		//print_integer_matrix_width(cout, H_subfield, m, n, n, 2);
		Fq_subfield->latex_matrix(cout, f_elements_exponential,
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
			Int_vec_print(cout, v, d);
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
			snprintf(str, 1000, "ttp_code_Ah_%d.txt", Fq_subfield->q);
		}
		else {
			snprintf(str, 1000, "ttp_code_A_%d.txt", Fq_subfield->q);
		}
	}
	else if (f_construction_B) {
		snprintf(str, 1000, "ttp_code_B_%d.txt", Fq_subfield->q);
	}
	fname.assign(str);
	//write_set_to_file(fname, L, N, verbose_level);

	FREE_OBJECT(P);
	FREE_int(v);
	FREE_int(H_subfield);
}





void finite_field::create_segre_variety(int a, int b,
		std::string &label_txt,
		std::string &label_tex,
		int &nb_pts, long int *&Pts,
	int verbose_level)
// The Segre map goes from PG(a,q) cross PG(b,q) to PG((a+1)*(b+1)-1,q)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "finite_field::create_segre_variety" << endl;
	}
	int d;
	long int N1, N2;
	long int rk1, rk2, rk3;
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


	v1 = NEW_int(a + 1);
	v2 = NEW_int(b + 1);
	v3 = NEW_int(d);


	geometry::geometry_global GG;

	N1 = GG.nb_PG_elements(a, q);
	N2 = GG.nb_PG_elements(b, q);

	Pts = NEW_lint(N1 * N2);
	nb_pts = 0;


	for (rk1 = 0; rk1 < N1; rk1++) {
		//P1->unrank_point(v1, rk1);
		PG_element_unrank_modified_lint(v1, 1, a + 1, rk1);


		for (rk2 = 0; rk2 < N2; rk2++) {
			//P2->unrank_point(v2, rk2);
			PG_element_unrank_modified_lint(v2, 1, b + 1, rk2);


			Linear_algebra->mult_matrix_matrix(v1, v2, v3, a + 1, 1, b + 1,
					0 /* verbose_level */);

			//rk3 = P3->rank_point(v3);
			PG_element_rank_modified_lint(v3, 1, d, rk3);

			Pts[nb_pts++] = rk3;

			if (f_v) {
				cout << setw(4) << nb_pts - 1 << " : " << endl;
				Int_matrix_print(v3, a + 1, b + 1);
				cout << " : " << setw(5) << rk3 << endl;
			}
		}
	}

	char str[1000];

	sprintf(str, "segre_variety_%d_%d_%d", a, b, q);
	label_txt.assign(str);

	sprintf(str, "segre\\_variety\\_%d\\_%d\\_%d", a, b, q);
	label_tex.assign(str);

	FREE_int(v1);
	FREE_int(v2);
	FREE_int(v3);
}



void finite_field::do_andre(finite_field *Fq,
		long int *the_set_in, int set_size_in,
		long int *&the_set_out, int &set_size_out,
	int verbose_level)
// creates PG(2,Q) and PG(4,q)
// this is FQ
// this functions is not called from anywhere right now
// it needs a pair of finite fields
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "finite_field::do_andre for a set of size " << set_size_in << endl;
	}

	geometry::projective_space *P2, *P4;
	int a, a0, a1;
	int b, b0, b1;
	int i, h, k, alpha;
	int *v, *w1, *w2, *w3, *v2;
	int *components;
	int *embedding;
	int *pair_embedding;


	P2 = NEW_OBJECT(geometry::projective_space);
	P4 = NEW_OBJECT(geometry::projective_space);


	P2->projective_space_init(2, this,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	P4->projective_space_init(4, Fq,
		FALSE /* f_init_incidence_structure */,
		verbose_level  /*MINIMUM(verbose_level - 1, 3)*/);

	//d = 5;


	if (f_v) {
		cout << "finite_field::do_andre before subfield_embedding_2dimensional" << endl;
	}

	subfield_embedding_2dimensional(*Fq,
		components, embedding, pair_embedding,
		verbose_level);

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
		cout << "finite_field::do_andre after subfield_embedding_2dimensional" << endl;
	}
	if (f_vv) {
		print_embedding(*Fq,
			components, embedding, pair_embedding);
	}
	alpha = p;
	if (f_vv) {
		cout << "finite_field::do_andre alpha=" << alpha << endl;
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
			cout << "finite_field::do_andre input point " << i << " is "
					<< the_set_in[i] << " : ";
		}
		P2->unrank_point(v, the_set_in[i]);
		PG_element_normalize(v, 1, 3);
		if (f_vv) {
			Int_vec_print(cout, v, 3);
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
				Int_vec_print(cout, w1, 4);
				cout << "w2=";
				Int_vec_print(cout, w2, 4);
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
					Int_vec_print(cout, v2, 2);
					cout << " : ";
				}
				for (k = 0; k < 4; k++) {
					w3[k] = Fq->add(Fq->mult(v2[0], w1[k]),
							Fq->mult(v2[1], w2[k]));
				}
				w3[4] = 0;
				if (f_vv) {
					cout << " ";
					Int_vec_print(cout, w3, 5);
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
				Int_vec_print(cout, w1, 5);
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
			Int_vec_print(cout, w1, 5);
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

	if (f_v) {
		cout << "finite_field::do_andre done" << endl;
	}
}


void finite_field::do_embed_orthogonal(
	int epsilon, int n,
	long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int d = n + 1;
	long int h, a, b;
	int c1 = 0, c2 = 0, c3 = 0;

	if (f_v) {
		cout << "finite_field::do_embed_orthogonal" << endl;
	}

	if (epsilon == -1) {
		Linear_algebra->choose_anisotropic_form(c1, c2, c3, verbose_level);
	}

	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];
		Orthogonal_indexing->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, a, 0 /* verbose_level */);
		//b = P->rank_point(v);
		PG_element_rank_modified_lint(v, 1, n + 1, b);
		set_out[h] = b;
	}

	FREE_int(v);
	if (f_v) {
		cout << "finite_field::do_embed_orthogonal done" << endl;
	}

}

void finite_field::do_embed_points(int n,
		long int *set_in, long int *&set_out, int set_size,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *v;
	int d = n + 2;
	int h;
	long int a, b;

	if (f_v) {
		cout << "finite_field::do_embed_points" << endl;
	}

	v = NEW_int(d);
	set_out = NEW_lint(set_size);

	for (h = 0; h < set_size; h++) {
		a = set_in[h];

		PG_element_unrank_modified_lint(v, 1, n + 1, a);

		v[d - 1] = 0;

		PG_element_rank_modified_lint(v, 1, n + 2, b);

		set_out[h] = b;
	}

	FREE_int(v);
	if (f_v) {
		cout << "finite_field::do_embed_points done" << endl;
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
	combinatorics::combinatorics_domain Combi;
	data_structures::sorting Sorting;

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

	Int_vec_zero(M, nb_rows * nb_cols);


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
			Int_vec_print(cout, Ac, ac);
			cout << endl;
		}


		for (u = 0; u < nb_U; u++, row++) {

			Combi.unrank_k_subset(u, U, len - (k - 2), k - 1);
			for (i = 0; i < k - 1; i++) {
				U1[i] = Ac[U[i]];
			}
			if (f_vv) {
				cout << "U1=";
				Int_vec_print(cout, U1, k - 1);
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
					Int_vec_print(cout, C, k - 1);
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
					d = Linear_algebra->BallChowdhury_matrix_entry(
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

	mtx_rank = Linear_algebra->Gauss_easy(M, nb_rows, nb_cols);
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
	data_structures::sorting Sorting;

	for (i = 0; i < 6; i++) {
		klein_to_wedge(Basis1 + i * 6, Basis2 + i * 6);
	}

	Linear_algebra->matrix_inverse(B, Bv, 6, 0 /* verbose_level */);




	Linear_algebra->exterior_square(A4, An2, 4, 0 /*verbose_level*/);

	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"exterior_square :" << endl;
		Int_matrix_print(An2, 6, 6);
		cout << endl;
	}


	for (j = 0; j < 6; j++) {
		Linear_algebra->mult_vector_from_the_left(Basis2 + j * 6, An2, v, 6, 6);
				// v[m], A[m][n], vA[n]
		wedge_to_klein(v, w);
		Int_vec_copy(w, C + j * 6, 6);
	}


	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"orthogonal matrix :" << endl;
		Int_matrix_print(C, 6, 6);
		cout << endl;
	}

	Linear_algebra->mult_matrix_matrix(Bv, C, D, 6, 6, 6, 0 /*verbose_level */);
	Linear_algebra->mult_matrix_matrix(D, B, A6, 6, 6, 6, 0 /*verbose_level */);

	PG_element_normalize_from_front(A6, 1, 36);

	if (f_vv) {
		cout << "finite_field::isomorphism_to_special_orthogonal "
				"orthogonal matrix in the special form:" << endl;
		Int_matrix_print(A6, 6, 6);
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

int finite_field::evaluate_Fermat_cubic(int *v)
// used to create the Schlaefli graph
{
	int a, i;

	a = 0;
	for (i = 0; i < 4; i++) {
		a = add(a, power(v[i], 3));
	}
	return a;
}



}}}


