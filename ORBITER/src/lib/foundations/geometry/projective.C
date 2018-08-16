// projective.C
//
// Anton Betten
//
// started:  April 2, 2003




#include "foundations.h"

INT nb_PG_elements(INT n, INT q)
// $\frac{q^{n+1} - 1}{q-1} = \sum_{i=0}^{n} q^i $
{
	INT qhl, l, deg;
	
	l = 0;
	qhl = 1;
	deg = 0;
	while (l <= n) {
		deg += qhl;
		qhl *= q;
		l++;
		}	
	return deg;
}

INT nb_PG_elements_not_in_subspace(INT n, INT m, INT q)
// |PG_n(q)| - |PG_m(q)|
{
	INT a, b;
	
	a = nb_PG_elements(n, q);
	b = nb_PG_elements(m, q);
	return a - b;
}

INT nb_AG_elements(INT n, INT q)
// $q^n$
{
	return i_power_j(q, n);
}

void all_PG_elements_in_subspace(finite_field *F, INT *genma, INT k, INT n, INT *&point_list, INT &nb_points, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT *message;
	INT *word;
	INT i, j;

	if (f_v) {
		cout << "all_PG_elements_in_subspace" << endl;
		}
	message = NEW_INT(k);
	word = NEW_INT(n);
	nb_points = generalized_binomial(k, 1, F->q);
	point_list = NEW_INT(nb_points);
	
	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(*F, message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			INT_vec_print(cout, message, k);
			cout << endl;
			}
		F->mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			INT_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified(*F, word, 1, n, j);
		if (f_vv) {
			cout << "which has rank " << j << endl;
			}
		point_list[i] = j;
		}
	if (f_v) {
		cout << "before FREE_INT(message);" << endl;
		}

	FREE_INT(message);
	FREE_INT(word);
	if (f_v) {
		cout << "all_PG_elements_in_subspace done" << endl;
		}
}

void all_PG_elements_in_subspace_array_is_given(finite_field *F, INT *genma, INT k, INT n, INT *point_list, INT &nb_points, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; //(verbose_level >= 2);
	INT *message;
	INT *word;
	INT i, j;

	if (f_v) {
		cout << "all_PG_elements_in_subspace_array_is_given" << endl;
		}
	message = NEW_INT(k);
	word = NEW_INT(n);
	nb_points = generalized_binomial(k, 1, F->q);
	//point_list = NEW_INT(nb_points);
	
	for (i = 0; i < nb_points; i++) {
		PG_element_unrank_modified(*F, message, 1, k, i);
		if (f_vv) {
			cout << "message " << i << " / " << nb_points << " is ";
			INT_vec_print(cout, message, k);
			cout << endl;
			}
		F->mult_vector_from_the_left(message, genma, word, k, n);
		if (f_vv) {
			cout << "yields word ";
			INT_vec_print(cout, word, n);
			cout << endl;
			}
		PG_element_rank_modified(*F, word, 1, n, j);
		if (f_vv) {
			cout << "which has rank " << j << endl;
			}
		point_list[i] = j;
		}
	if (f_v) {
		cout << "before FREE_INT(message);" << endl;
		}

	FREE_INT(message);
	FREE_INT(word);
	if (f_v) {
		cout << "all_PG_elements_in_subspace_array_is_given done" << endl;
		}
}

void display_all_PG_elements(INT n, finite_field &GFq)
{
	INT *v = NEW_INT(n + 1);
	INT l = nb_PG_elements(n, GFq.q);
	INT i, j, a;
	
	for (i = 0; i < l; i++) {
		PG_element_unrank_modified(GFq, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified(GFq, v, 1, n + 1, a);
		cout << " : " << a << endl;
		}
	FREE_INT(v);
}

void display_all_PG_elements_not_in_subspace(INT n, INT m, finite_field &GFq)
{
	INT *v = NEW_INT(n + 1);
	INT l = nb_PG_elements_not_in_subspace(n, m, GFq.q);
	INT i, j, a;
	
	for (i = 0; i < l; i++) {
		PG_element_unrank_modified_not_in_subspace(GFq, v, 1, n + 1, m, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		PG_element_rank_modified_not_in_subspace(GFq, v, 1, n + 1, m, a);
		cout << " : " << a << endl;
		}
	FREE_INT(v);
}

void display_all_AG_elements(INT n, finite_field &GFq)
{
	INT *v = NEW_INT(n);
	INT l = nb_AG_elements(n, GFq.q);
	INT i, j;
	
	for (i = 0; i < l; i++) {
		AG_element_unrank(GFq.q, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n; j++) {
			cout << v[j] << " ";
			}
		cout << endl;
		}
	FREE_INT(v);
}

void PG_element_apply_frobenius(INT n, finite_field &GFq, INT *v, INT f)
{
	INT i;
	
	for (i = 0; i < n; i++) {
		v[i] = GFq.frobenius_power(v[i], f);
		}
}

void PG_element_normalize(finite_field &GFq, INT *v, INT stride, INT len)
// last non-zero element made one
{
	INT i, j, a;
	
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = GFq.inverse(a);
			v[i * stride] = 1;
			for (j = i - 1; j >= 0; j--) {
				v[j * stride] = GFq.mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "PG_element_normalize() zero vector()" << endl;
	exit(1);
}

void PG_element_normalize_from_front(finite_field &GFq, INT *v, INT stride, INT len)
// first non zero element made one
{
	INT i, j, a;
	
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (a) {
			if (a == 1)
				return;
			a = GFq.inverse(a);
			v[i * stride] = 1;
			for (j = i + 1; j < len; j++) {
				v[j * stride] = GFq.mult(v[j * stride], a);
				}
			return;
			}
		}
	cout << "PG_element_normalize() zero vector()" << endl;
	exit(1);
}

void PG_element_rank_modified(finite_field &GFq, INT *v, INT stride, INT len, INT &a)
{
	INT i, j, q_power_j, b, sqj;
	INT f_v = FALSE;
	
	if (len <= 0) {
		cout << "PG_element_rank_modified() len <= 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	PG_element_normalize(GFq, v, stride, len);
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
		cout << "PG_element_rank_modified() zero vector" << endl;
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
		cout << "PG_element_rank_modified() zero vector" << endl;
		exit(1);
		}
	if (v[i * stride] != 1) {
		cout << "PG_element_rank_modified() vector not normalized" << endl;
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
		q_power_j *= GFq.q;
		}
	if (f_v) {
		cout << "b=" << b << endl;
		cout << "sqj=" << sqj << endl;
		}


	a = 0;
	for (j = i - 1; j >= 0; j--) {
		a += v[j * stride];
		if (j > 0)
			a *= GFq.q;
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

void PG_element_unrank_fining(finite_field &GFq, INT *v, INT len, INT a)
{
	INT b, c, q;
	
	q = GFq.q;
	if (len != 3) {
		cout << "PG_element_unrank_fining len != 3" << endl;
		exit(1);
		}
	if (a <= 0) {
		cout << "PG_element_unrank_fining a <= 0" << endl;
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
		cout << "PG_element_unrank_fining a is illegal" << endl;
		exit(1);
		}
}

void PG_element_unrank_gary_cook(finite_field &GFq, INT *v, INT len, INT a)
{
	INT b, q, qm1o2, rk, i;
	
	q = GFq.q;
	if (len != 3) {
		cout << "PG_element_unrank_gary_cook len != 3" << endl;
		exit(1);
		}
	if (q != 11) {
		cout << "PG_element_unrank_gary_cook q != 11" << endl;
		exit(1);
		}
	qm1o2 = (q - 1) >> 1;
	if (a < 0) {
		cout << "PG_element_unrank_gary_cook a < 0" << endl;
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
				cout << "PG_element_unrank_gary_cook a is illegal" << endl;
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

void PG_element_unrank_modified(finite_field &GFq, INT *v, INT stride, INT len, INT a)
{
	INT n, l, ql, sql, k, j, r, a1 = a;
	
	n = len;
	if (n <= 0) {
		cout << "PG_element_unrank_modified() len <= 0" << endl;
		exit(1);
		}
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
	ql = GFq.q;
	sql = 1;
	// sql = q^0 + q^1 + \cdots + q^{l-1}
	while (l < n) {
		if (a >= ql - 1) {
			a -= (ql - 1);
			sql += ql;
			ql *= GFq.q;
			l++;
			continue;
			}
		v[l * stride] = 1;
		for (k = l + 1; k < n; k++) {
			v[k * stride] = 0;
			}
		a++; // take into account that we do not want 00001000
		if (l == n - 1 && a >= sql) {
			a++; // take int account that the vector 11111 has already been listed
			}
		j = 0;
		while (a != 0) {
			r = a % GFq.q;
			v[j * stride] = r;
			j++;
			a -= r;
			a /= GFq.q;
			}
		for ( ; j < l; j++) {
			v[j * stride] = 0;
			}
		return;
		}
	cout << "PG_element_unrank_modified() a too large" << endl;
	cout << "len = " << len << endl;
	cout << "a = " << a1 << endl;
	exit(1);
}

void PG_element_rank_modified_not_in_subspace(finite_field &GFq, INT *v, INT stride, INT len, INT m, INT &a)
{
	INT s, qq, i;
	
	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= GFq.q;
		s += qq;
		}
	s -= (m + 1);
	
	PG_element_rank_modified(GFq, v, stride, len, a);
	if (a > len + s) {
		a -= s;
		}
	a -= (m + 1);
}

void PG_element_unrank_modified_not_in_subspace(finite_field &GFq, INT *v, INT stride, INT len, INT m, INT a)
{
	INT s, qq, i;
	
	qq = 1;
	s = qq;
	for (i = 0; i < m; i++) {
		qq *= GFq.q;
		s += qq;
		}
	s -= (m + 1);
	
	a += (m + 1);
	if (a > len) {
		a += s;
		}
	
	PG_element_unrank_modified(GFq, v, stride, len, a);
}

void AG_element_rank(INT q, INT *v, INT stride, INT len, INT &a)
{
	INT i;
	
	if (len <= 0) {
		cout << "AG_element_rank() len <= 0" << endl;
		exit(1);
		}
	a = 0;
	for (i = len - 1; i >= 0; i--) {
		a += v[i * stride];
		if (i > 0) {
			a *= q;
			}
		}
}

void AG_element_unrank(INT q, INT *v, INT stride, INT len, INT a)
{
	INT i, b;
	
#if 1
	if (len <= 0) {
		cout << "AG_element_unrank() len <= 0" << endl;
		exit(1);
		}
#endif
	for (i = 0; i < len; i++) {
		b = a % q;
		v[i * stride] = b;
		a /= q;
		}
}

void AG_element_rank_longinteger(INT q, INT *v, INT stride, INT len, longinteger_object &a)
{
	longinteger_domain D;
	longinteger_object Q, a1;
	INT i;
	
	if (len <= 0) {
		cout << "AG_element_rank_longinteger() len <= 0" << endl;
		exit(1);
		}
	a.create(0);
	Q.create(q);
	for (i = len - 1; i >= 0; i--) {
		a.add_INT(v[i * stride]);
		//cout << "AG_element_rank_longinteger after add_INT " << a << endl;
		if (i > 0) {
			D.mult(a, Q, a1);
			a.swap_with(a1);
			//cout << "AG_element_rank_longinteger after mult " << a << endl;
			}
		}
}

void AG_element_unrank_longinteger(INT q, INT *v, INT stride, INT len, longinteger_object &a)
{
	INT i, r;
	longinteger_domain D;
	longinteger_object Q, a1;
	
	if (len <= 0) {
		cout << "AG_element_unrank_longinteger() len <= 0" << endl;
		exit(1);
		}
	for (i = 0; i < len; i++) {
		D.integral_division_by_INT(a, q, a1, r);
		//r = a % q;
		v[i * stride] = r;
		//a /= q;
		a.swap_with(a1);
		}
}


INT PG_element_modified_is_in_subspace(INT n, INT m, INT *v)
{
	INT j;
	
	for (j = m + 1; j < n + 1; j++) {
		if (v[j]) {
			return FALSE;
			}
		}
	return TRUE;
}

void PG_element_modified_not_in_subspace_perm(INT n, INT m, 
	finite_field &GFq, INT *orbit, INT *orbit_inv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *v = NEW_INT(n + 1);
	INT l = nb_PG_elements(n, GFq.q);
	INT ll = nb_PG_elements_not_in_subspace(n, m, GFq.q);
	INT i, j1 = 0, j2 = ll, f_in, j;
	
	for (i = 0; i < l; i++) {
		PG_element_unrank_modified(GFq, v, 1, n + 1, i);
		f_in = PG_element_modified_is_in_subspace(n, m, v);
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
	FREE_INT(v);
}

INT PG2_line_on_point_unrank(finite_field &GFq, INT *v1, INT rk)
{
	INT v2[3];
	
	PG2_line_on_point_unrank_second_point(GFq, v1, v2, rk);
	return PG2_line_rank(GFq, v1, v2, 1);
}

void PG2_line_on_point_unrank_second_point(finite_field &GFq, INT *v1, INT *v2, INT rk)
{
	INT V[2];
	INT idx0, idx1, idx2;
	
	PG_element_normalize(GFq, v1, 1/* stride */, 3);
	if (v1[2] == 1) {
		idx0 = 2;
		idx1 = 0;
		idx2 = 1;
		}
	else if (v1[1] == 1) {
		idx0 = 1;
		idx1 = 0;
		idx2 = 2;
		}
	else {
		idx0 = 0;
		idx1 = 1;
		idx2 = 2;
		}
	PG_element_unrank_modified(GFq, V, 1/* stride */, 2, rk);
	v2[idx0] = v1[idx0];
	v2[idx1] = GFq.add(v1[idx1], V[0]);
	v2[idx2] = GFq.add(v1[idx2], V[1]);
}

INT PG2_line_rank(finite_field &GFq, INT *v1, INT *v2, INT stride)
{
	INT A[9];
	INT base_cols[3];
	INT kernel_m, kernel_n;
	INT kernel[9];
	INT rk, line_rk;
	
	A[0] = v1[0];
	A[1] = v1[1];
	A[2] = v1[2];
	A[3] = v2[0];
	A[4] = v2[1];
	A[5] = v2[2];
	rk = GFq.Gauss_INT(A, FALSE /* f_special */, TRUE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL /*P*/, 2, 3, 3, 0 /* verbose_level */);
	if (rk != 2) {
		cout << "PG2_line_rank rk != 2" << endl;
		exit(1);
		}
	GFq.matrix_get_kernel(A, 2, 3, base_cols, rk /*nb_base_cols*/, 
		kernel_m, kernel_n, kernel);
	if (kernel_m != 3) {
		cout << "PG2_line_rank kernel_m != 3" << endl;
		exit(1);
		}
	if (kernel_n != 1) {
		cout << "PG2_line_rank kernel_n != 1" << endl;
		exit(1);
		}
	PG_element_rank_modified(GFq, kernel, 1 /* stride*/, 3, line_rk);
	return line_rk;
}

void PG2_line_unrank(finite_field &GFq, INT *v1, INT *v2, INT stride, INT line_rk)
{
	INT A[9];
	INT base_cols[3];
	INT kernel_m, kernel_n;
	INT kernel[9];
	INT rk;
	
	PG_element_unrank_modified(GFq, A, 1 /* stride*/, 3, line_rk);
	rk = GFq.Gauss_INT(A, FALSE /* f_special */, TRUE /* f_complete */, 
		base_cols, 
		FALSE /* f_P */, NULL /*P*/, 1, 3, 3, 0 /* verbose_level */);
	if (rk != 1) {
		cout << "PG2_line_unrank rk != 1" << endl;
		exit(1);
		}
	GFq.matrix_get_kernel(A, 1, 3, base_cols, rk /*nb_base_cols*/, 
		kernel_m, kernel_n, kernel);
	if (kernel_m != 3) {
		cout << "PG2_line_rank kernel_m != 3" << endl;
		exit(1);
		}
	if (kernel_n != 2) {
		cout << "PG2_line_rank kernel_n != 2" << endl;
		exit(1);
		}
	v1[0] = kernel[0];
	v2[0] = kernel[1];
	v1[1] = kernel[2];
	v2[1] = kernel[3];
	v1[2] = kernel[4];
	v2[2] = kernel[5];
}

void test_PG(INT n, INT q)
{
	finite_field GFq;
	INT m;
	INT verbose_level = 1;
	
	GFq.init(q, verbose_level);
	
	cout << "all elements of PG_" << n << "(" << q << ")" << endl;
	display_all_PG_elements(n, GFq);
	
	for (m = 0; m < n; m++) {
		cout << "all elements of PG_" << n << "(" << q << "), not in a subspace of dimension " << m << endl;
		display_all_PG_elements_not_in_subspace(n, m, GFq);
		}
	
}


void line_through_two_points(finite_field &GFq, INT len, INT pt1, INT pt2, INT *line)
{
	INT v1[100], v2[100], v3[100], alpha, a, ii;
	
	if (len > 100) {
		cout << "line_through_two_points() len >= 100" << endl;
		exit(1);
		}
	PG_element_unrank_modified(GFq, v1, 1 /* stride */, len, pt1);
	PG_element_unrank_modified(GFq, v2, 1 /* stride */, len, pt2);
	line[0] = pt1;
	line[1] = pt2;
	for (alpha = 1; alpha < GFq.q; alpha++) {
		for (ii = 0; ii < len; ii++) {
			a = GFq.mult(v1[ii], alpha);
			v3[ii] = GFq.add(a, v2[ii]);
			}
		PG_element_normalize(GFq, v3, 1 /* stride */, len);
		PG_element_rank_modified(GFq, v3, 1 /* stride */, len, line[1 + alpha]);
		}
}

void print_set_in_affine_plane(finite_field &GFq, INT len, INT *S)
{
	INT *A;
	INT i, j, x, y, v[3];
	
	
	A = NEW_INT(GFq.q * GFq.q);
	for (x = 0; x < GFq.q; x++) {
		for (y = 0; y < GFq.q; y++) {
			A[(GFq.q - 1 - y) * GFq.q + x] = 0;
			}
		}
	for (i = 0; i < len; i++) {
		PG_element_unrank_modified(GFq, v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			//cout << "my_generator::print_set_in_affine_plane() not an affine point" << endl;
			cout << "(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
			continue;
			}
		x = v[0];
		y = v[1];
		A[(GFq.q - 1 - y) * GFq.q + x] = 1;
		}
	for (i = 0; i < GFq.q; i++) {
		for (j = 0; j < GFq.q; j++) {
			cout << A[i * GFq.q + j];
			}
		cout << endl;
		}
	FREE_INT(A);
}

INT consecutive_ones_property_in_affine_plane(ostream &ost, finite_field &GFq, INT len, INT *S)
{
	INT i, y, v[3];
	
	
	for (i = 0; i < len; i++) {
		PG_element_unrank_modified(GFq, v, 1 /* stride */, 3 /* len */, S[i]);
		if (v[2] != 1) {
			cout << "my_generator::consecutive_ones_property_in_affine_plane() not an affine point" << endl;
			ost << "(" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
			exit(1);
			}
		y = v[1];
		if (y != i)
			return FALSE;
		}
	return TRUE;
}

void oval_polynomial(finite_field &GFq, INT *S, unipoly_domain &D, unipoly_object &poly, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, v[3], x; //, y;
	INT *map;
	
	if (f_v) {
		cout << "oval_polynomial" << endl;
		}
	map = NEW_INT(GFq.q);
	for (i = 0; i < GFq.q; i++) {
		PG_element_unrank_modified(GFq, v, 1 /* stride */, 3 /* len */, S[2 + i]);
		if (v[2] != 1) {
			cout << "oval_polynomial not an affine point" << endl;
			exit(1);
			}
		x = v[0];
		//y = v[1];
		//cout << "map[" << i << "] = " << xx << endl;
		map[i] = x;
		}
	if (f_v) {
		cout << "the map" << endl;
		for (i = 0; i < GFq.q; i++) {
			cout << map[i] << " ";
			}
		cout << endl;
		}
	
	D.create_Dickson_polynomial(poly, map);
	
	FREE_INT(map);
	if (f_v) {
		cout << "oval_polynomial done" << endl;
		}
}


INT line_intersection_with_oval(finite_field &GFq, INT *f_oval_point, INT line_rk, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT line[3], base_cols[3], K[6], points[6], rk, kernel_m, kernel_n;
	INT j, w[2], a, b, pt[3], nb = 0;
	
	if (f_v) {
		cout << "intersecting line " << line_rk << " with the oval" << endl;
		}
	PG_element_unrank_modified(GFq, line, 1, 3, line_rk);
	rk = GFq.Gauss_INT(line, FALSE /* f_special */, TRUE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL /* P */, 1 /* m */, 3 /* n */, 0 /* Pn */, 
		0 /* verbose_level */);
	if (f_vv) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, line, 1, 3);
		}
	GFq.matrix_get_kernel(line, 1, 3, base_cols, rk, 
		kernel_m, kernel_n, K);
	INT_matrix_transpose(K, kernel_m, kernel_n, points);
	if (f_vv) {
		cout << "points:" << endl;
		print_integer_matrix(cout, points, 2, 3);
		}
	for (j = 0; j < GFq.q + 1; j++) {
		PG_element_unrank_modified(GFq, w, 1, 2, j);
		a = GFq.mult(points[0], w[0]);
		b = GFq.mult(points[3], w[1]);
		pt[0] = GFq.add(a, b);
		a = GFq.mult(points[1], w[0]);
		b = GFq.mult(points[4], w[1]);
		pt[1] = GFq.add(a, b);
		a = GFq.mult(points[2], w[0]);
		b = GFq.mult(points[5], w[1]);
		pt[2] = GFq.add(a, b);
		PG_element_rank_modified(GFq, pt, 1, 3, rk);
		//cout << j << " : " << pt[0] << "," << pt[1] << "," << pt[2] << " : " << rk << " : ";
		if (f_oval_point[rk]) {
			//cout << "yes" << endl;
			if (f_v) {
				cout << "found oval point" << endl;
				}
			nb++;
			}
		else {
			//cout << "no" << endl;
			}
		}
	return nb;
}

INT get_base_line(finite_field &GFq, INT plane1, INT plane2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "get_base_line()" << endl;
		}
	INT planes[8], base_cols[4], rk;
	INT intersection[16], intersection2[16], lines[6], line[3], kernel_m, kernel_n, line_rk;

	AG_element_unrank(GFq.q, planes, 1, 3, plane1);
	planes[3] = 1;
	AG_element_unrank(GFq.q, planes + 4, 1, 3, plane2);
	planes[7] = 1;
	if (f_v) {
		cout << "planes:" << endl;
		print_integer_matrix(cout, planes, 2, 4);
		}
	rk = GFq.Gauss_INT(planes, FALSE /* f_special */, TRUE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL /* P */, 2 /* m */, 4 /* n */, 0 /* Pn */, 
		0 /* verbose_level */);
	if (f_v) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, planes, 2, 4);
		}
	GFq.matrix_get_kernel(planes, 2, 4, base_cols, rk, 
		kernel_m, kernel_n, intersection);
	INT_matrix_transpose(intersection, kernel_m, kernel_n, intersection2);
	if (f_v) {
		cout << "kernel:" << endl;
		print_integer_matrix(cout, intersection2, kernel_n, kernel_m);
		}
	lines[0] = intersection2[0];
	lines[1] = intersection2[1];
	lines[2] = intersection2[2];
	lines[3] = intersection2[4];
	lines[4] = intersection2[5];
	lines[5] = intersection2[6];
	if (f_v) {
		cout << "lines:" << endl;
		print_integer_matrix(cout, lines, 2, 3);
		}
	rk = GFq.Gauss_INT(lines, FALSE /* f_special */, TRUE /* f_complete */, base_cols, 
		FALSE /* f_P */, NULL /* P */, 2 /* m */, 3 /* n */, 0 /* Pn */, 
		0 /* verbose_level */);
	if (rk != 2) {
		cout << "rk != 2" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "after Gauss:" << endl;
		print_integer_matrix(cout, lines, 2, 3);
		}
	GFq.matrix_get_kernel(lines, 2, 3, base_cols, rk, kernel_m, kernel_n, line);
	if (f_v) {
		cout << "the line:" << endl;
		print_integer_matrix(cout, line, 1, 3);
		}
	PG_element_rank_modified(GFq, line, 1, 3, line_rk);
	if (f_v) {
		cout << "has rank " << line_rk << endl;
		}
	return line_rk;
}

INT PHG_element_normalize(finite_ring &R, INT *v, INT stride, INT len)
// last unit element made one
{
	INT i, j, a;
	
	for (i = len - 1; i >= 0; i--) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = len - 1; j >= 0; j--) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "PHG_element_normalize() vector is not free" << endl;
	exit(1);
}


INT PHG_element_normalize_from_front(finite_ring &R, INT *v, INT stride, INT len)
// first non unit element made one
{
	INT i, j, a;
	
	for (i = 0; i < len; i++) {
		a = v[i * stride];
		if (R.is_unit(a)) {
			if (a == 1)
				return i;
			a = R.inverse(a);
			for (j = 0; j < len; j++) {
				v[j * stride] = R.mult(v[j * stride], a);
				}
			return i;
			}
		}
	cout << "PHG_element_normalize_from_front() vector is not free" << endl;
	exit(1);
}

INT PHG_element_rank(finite_ring &R, INT *v, INT stride, INT len)
{
	INT i, j, idx, a, b, r1, r2, rk, N;
	INT f_v = FALSE;
	INT *w;
	INT *embedding;
	
	if (len <= 0) {
		cout << "PHG_element_rank() len <= 0" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "the vector before normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	idx = PHG_element_normalize(R, v, stride, len);
	if (f_v) {
		cout << "the vector after normalization is ";
		for (i = 0; i < len; i++) {
			cout << v[i * stride] << " ";
			}
		cout << endl;
		}
	w = NEW_INT(len - 1);
	embedding = NEW_INT(len - 1);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
			}
		embedding[i] = j;
		}
	for (i = 0; i < len - 1; i++) {
		w[i] = v[embedding[i] * stride];
		}
	for (i = 0; i < len - 1; i++) {
		a = w[i];
		b = a % R.p;
		v[embedding[i] * stride] = b;
		w[i] = (a - b) / R.p;
		}
	if (f_v) {
		cout << "w=";
		INT_vec_print(cout, w, len - 1);
		cout << endl;
		}
	AG_element_rank(R.e, w, 1, len - 1, r1);
	PG_element_rank_modified(*R.Fp, v, stride, len, r2);

	N = nb_PG_elements(len - 1, R.p);
	rk = r1 * N + r2;

	FREE_INT(w);
	FREE_INT(embedding);

	return rk;
}

void PHG_element_unrank(finite_ring &R, INT *v, INT stride, INT len, INT rk)
{
	INT i, j, idx, r1, r2, N;
	INT f_v = FALSE;
	INT *w;
	INT *embedding;
	
	if (len <= 0) {
		cout << "PHG_element_unrank() len <= 0" << endl;
		exit(1);
		}

	w = NEW_INT(len - 1);
	embedding = NEW_INT(len - 1);

	N = nb_PG_elements(len - 1, R.p);
	r2 = rk % N;
	r1 = (rk - r2) / N;
	
	AG_element_unrank(R.e, w, 1, len - 1, r1);
	PG_element_unrank_modified(*R.Fp, v, stride, len, r2);

	if (f_v) {
		cout << "w=";
		INT_vec_print(cout, w, len - 1);
		cout << endl;
		}

	idx = PHG_element_normalize(R, v, stride, len);
	for (i = 0, j = 0; i < len - 1; i++, j++) {
		if (i == idx) {
			j++;
			}
		embedding[i] = j;
		}
	
	for (i = 0; i < len - 1; i++) {
		v[embedding[i] * stride] += w[i] * R.p;
		}



	FREE_INT(w);
	FREE_INT(embedding);

}

INT nb_PHG_elements(INT n, finite_ring &R)
{
	INT N1, N2;
	
	N1 = nb_PG_elements(n, R.p);
	N2 = nb_AG_elements(n, R.e);
	return N1 * N2;
}

void display_all_PHG_elements(INT n, INT q)
{
	INT *v = NEW_INT(n + 1);
	INT l;
	INT i, j, a;
	finite_ring R;

	R.init(q, 0);
	l = nb_PHG_elements(n, R);
	for (i = 0; i < l; i++) {
		PHG_element_unrank(R, v, 1, n + 1, i);
		cout << i << " : ";
		for (j = 0; j < n + 1; j++) {
			cout << v[j] << " ";
			}
		a = PHG_element_rank(R, v, 1, n + 1);
		cout << " : " << a << endl;
		}
	FREE_INT(v);
}


void display_table_of_projective_points(ostream &ost, finite_field *F, INT *v, INT nb_pts, INT len)
{
	INT i;
	INT *coords;
	
	coords = NEW_INT(len);
	ost << "{\\renewcommand*{\\arraystretch}{1.5}" << endl;
	ost << "$$" << endl;
	ost << "\\begin{array}{|c|c|c|}" << endl;
	ost << "\\hline" << endl;
	ost << "i & a_i & P_{a_i}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_pts; i++) {
		PG_element_unrank_modified(*F, coords, 1, 3, v[i]);
		ost << i << " & " << v[i] << " & ";
		INT_vec_print(ost, coords, len);
		ost << "\\\\" << endl;
		if (((i + 1) % 30) == 0) {
			ost << "\\hline" << endl;
			ost << "\\end{array}" << endl;
			ost << "$$}%" << endl;
			ost << "$$" << endl;
			ost << "\\begin{array}{|c|c|c|}" << endl;
			ost << "\\hline" << endl;
			ost << "i & a_i & P_{a_i}\\\\" << endl;
			ost << "\\hline" << endl;
			ost << "\\hline" << endl;
			}
		}
	ost << "\\hline" << endl;
	ost << "\\end{array}" << endl;
	ost << "$$}%" << endl;
	FREE_INT(coords);
}



