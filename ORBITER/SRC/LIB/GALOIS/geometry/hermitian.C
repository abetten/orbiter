// hermitian.C
// 
// Anton Betten
// 3/19/2010
//
// 
//
//

#include "galois.h"

INT hermitian::cntr_new = 0;
INT hermitian::cntr_objects = 0;
INT hermitian::f_debug_memory = FALSE;

void *hermitian::operator new(size_t bytes)
{
	cntr_new++;
	cntr_objects++;
	if (f_debug_memory) {
		cout << "hermitian::operator new bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void *hermitian::operator new[](size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(hermitian);
	cntr_new++;
	cntr_objects += n;
	if (f_debug_memory) {
		cout << "hermitian::operator new[] n=" << n 
			<< " bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	return malloc(bytes);
}

void hermitian::operator delete(void *ptr, size_t bytes)
{
	if (f_debug_memory) {
		cout << "hermitian::operator delete bytes=" << bytes 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects--;
	return free(ptr);
}

void hermitian::operator delete[](void *ptr, size_t bytes)
{
	INT n;
	
	n = bytes / sizeof(hermitian);
	if (f_debug_memory) {
		cout << "hermitian::operator delete[] n=" << n 
			<< " cntr_new=" << cntr_new 
			<< " cntr_objects=" << cntr_objects 
			<< endl;
		}
	cntr_new--;
	cntr_objects -= n;
	return free(ptr);
}

hermitian::hermitian()
{
	null();
}

hermitian::~hermitian()
{
	if (cnt_N) {
		FREE_INT(cnt_N);
		}
	if (cnt_N1) {
		FREE_INT(cnt_N1);
		}
	if (cnt_S) {
		FREE_INT(cnt_S);
		}
	if (cnt_Sbar) {
		FREE_INT(cnt_Sbar);
		}
	if (norm_one_elements) {
		FREE_INT(norm_one_elements);
		}
	if (index_of_norm_one_element) {
		FREE_INT(index_of_norm_one_element);
		}
	if (log_beta) {
		FREE_INT(log_beta);
		}
	if (beta_power) {
		FREE_INT(beta_power);
		}
	null();
}

void hermitian::null()
{
	F = NULL;
	cnt_N = NULL;
	cnt_N1 = NULL;
	cnt_S = NULL;
	cnt_Sbar = NULL;
	norm_one_elements = NULL;
	index_of_norm_one_element = NULL;
	log_beta = NULL;
	beta_power = NULL;
}


void hermitian::init(finite_field *F, INT nb_vars, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, a;
	
	hermitian::F = F;
	hermitian::Q = F->q;
	hermitian::q = i_power_j(F->p, F->e >> 1);
	hermitian::k = nb_vars;
	if (f_v) {
		cout << "hermitian::init Q=" << F->q << " q=" << q << " nb_vars=" << nb_vars << endl;
		}
	if (F->e % 2) {
		cout << "hermitian::init field must have a quadratic subfield" << endl;
		exit(1);
		}
	cnt_N = NEW_INT(k + 1);
	cnt_N1 = NEW_INT(k + 1);
	cnt_S = NEW_INT(k + 1);
	cnt_Sbar = NEW_INT(k + 1);
	cnt_N[0] = 0;
	cnt_N1[0] = 0;
	cnt_S[0] = 0;
	cnt_Sbar[0] = 0;
	cnt_N[1] = Q - 1;
	cnt_N1[1] = q + 1;
	cnt_S[1] = 1;
	cnt_Sbar[1] = 0;
	for (i = 2; i <= k; i++) {
		cnt_N[i] = cnt_N[i - 1] * (Q - q - 1) + cnt_S[i - 1] * (Q - 1);
		cnt_S[i] = cnt_N[i - 1] * (q + 1) + cnt_S[i - 1];
		cnt_N1[i] = cnt_N[i] / (q - 1);
		cnt_Sbar[i] = (cnt_S[i] - 1) / (Q - 1);
		}
	cout << "  i :   N1[i] :    N[i] :    S[i] : Sbar[i]" << endl;
	for (i = 1; i <= k; i++) {
		cout << setw(3) << i << " : ";
		cout << setw(7) << cnt_N1[i] << " : ";
		cout << setw(7) << cnt_N[i] << " : ";
		cout << setw(7) << cnt_S[i] << " : ";
		cout << setw(7) << cnt_Sbar[i] << endl;
		}

	norm_one_elements = NEW_INT(q + 1);
	index_of_norm_one_element = NEW_INT(Q);
	log_beta = NEW_INT(Q);
	beta_power = NEW_INT(q);
	for (i = 0; i < Q; i++) {
		index_of_norm_one_element[i] = -1;
		log_beta[i] = -1;
		}
	for (i = 0; i < q + 1; i++) {
		a = F->alpha_power(i * (q - 1));
		norm_one_elements[i] = a;
		index_of_norm_one_element[a] = i;
		}
	if (f_v) {
		cout << "the norm one elements are: ";
		INT_vec_print(cout, norm_one_elements, q + 1);
		cout << endl;
		}
	cout << "i : norm_one_elements[i] : F->N2(norm_one_elements[i])" << endl;
	for (i = 0; i < q + 1; i++) {
		cout << i << " : " << norm_one_elements[i] << " : " << F->N2(norm_one_elements[i]) << endl;
		}
	alpha = F->p;
	beta = F->alpha_power(q + 1);
	for (i = 0; i < q - 1; i++) {
		j = F->power(beta, i);
		beta_power[i] = j;
		log_beta[j] = i;
		}
}

INT hermitian::nb_points()
{
	return cnt_Sbar[k];
}

void hermitian::unrank_point(INT *v, INT rk)
{
	Sbar_unrank(v, k, rk, 0 /*verbose_level*/);
}

INT hermitian::rank_point(INT *v)
{
	INT rk;

	rk = Sbar_rank(v, k, 0 /*verbose_level*/);
	return rk;
}

void hermitian::list_of_points_embedded_in_PG(INT *&Pts, INT &nb_pts, INT verbose_level)
{
	INT i, rk;
	INT *v;

	v = NEW_INT(k);
	nb_pts = nb_points();
	Pts = NEW_INT(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, i);
		PG_element_rank_modified(*F, v, 1, k, rk);
		Pts[i] = rk;
		}
}

void hermitian::list_all_N(INT verbose_level)
{
	INT *v;
	INT i, j, val0, val;

	cout << "list_all_N:" << endl;
	v = NEW_INT(k);
	for (i = 0; i < cnt_N[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 165) {verbose_level += 2;}
		N_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		INT_vec_print(cout, v, k);
		cout << " : " << val0;
		cout << " : " << val << endl;
		if (val == 0) {
			cout << "error" << endl;
			exit(1);
			}
		j = N_rank(v, k, verbose_level - 2);
		if (j != i) {
			cout << "error in ranking, i=" << i << " j=" << j << endl;
			exit(1);
			}
		}
}

void hermitian::list_all_N1(INT verbose_level)
{
	INT *v;
	INT i, j, val0, val;

	cout << "list_all_N1:" << endl;
	v = NEW_INT(k);
	for (i = 0; i < cnt_N1[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 15) {verbose_level += 2;}
		N1_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		INT_vec_print(cout, v, k);
		cout << " : " << val0;
		cout << " : " << val << endl;
		if (val != 1) {
			cout << "error" << endl;
			exit(1);
			}
		j = N1_rank(v, k, verbose_level - 2);
		if (j != i) {
			cout << "error in ranking, i=" << i << " j=" << j << endl;
			exit(1);
			}
		}
}

void hermitian::list_all_S(INT verbose_level)
{
	INT *v;
	INT i, j, val0, val;

	cout << "list_all_S:" << endl;
	v = NEW_INT(k);
	for (i = 0; i < cnt_S[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 6) {verbose_level += 2;}
		S_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		INT_vec_print(cout, v, k);
		cout << " : " << val0;
		cout << " : " << val << endl;
		if (val) {
			cout << "error" << endl;
			exit(1);
			}
		j = S_rank(v, k, verbose_level - 2);
		if (j != i) {
			cout << "error in ranking, i=" << i << " j=" << j << endl;
			exit(1);
			}
		}
}

void hermitian::list_all_Sbar(INT verbose_level)
{
	INT *v;
	INT i, j, a, h, val0, val;

	cout << "list_all_Sbar:" << endl;
	v = NEW_INT(k);
	for (i = 0; i < cnt_Sbar[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 6) {verbose_level += 2;}

		for (h = 0; h < q - 1; h++) {
			// loop over all elements in the subfield F_q:
			a = F->alpha_power(h * (q + 1));
	

			Sbar_unrank(v, k, i, 0 /*verbose_level*/);
			F->scalar_multiply_vector_in_place(a, v, k);
#if 0
			for (u = 0; u < k; u++) {
				v[u] = F->mult(a, v[u]);
				}
#endif
			val0 = evaluate_hermitian_form(v, k - 1);
			val = evaluate_hermitian_form(v, k);
			cout << setw(5) << i << "," << h << " : ";
			INT_vec_print(cout, v, k);
			cout << " : " << val0;
			cout << " : " << val << endl;
			if (val) {
				cout << "error" << endl;
				exit(1);
				}
			j = Sbar_rank(v, k, 0 /*verbose_level*/);
			if (j != i) {
				cout << "error in ranking, i=" << i << " j=" << j << endl;
				exit(1);
				}
			}
		}
}


INT hermitian::evaluate_hermitian_form(INT *v, INT len)
{
	INT i, a, b;

	a = 0;
	for (i = 0; i < len; i++) {
		b = F->N2(v[i]);
		a = F->add(a, b);
		//cout << "b=" << b << " a=" << a << endl;
		}
	//cout << "hermitian::evaluate_hermitian_form ";
	//INT_vec_print(cout, v, len);
	//cout << "val=" << a << endl;
	return a;
}

void hermitian::N_unrank(INT *v, INT len, INT rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk1, coset, rk0, coset0, A, val, m_val, log;
	
	if (f_v) {
		cout << "N_unrank len=" << len << " rk=" << rk << endl;
		}
	if (rk >= cnt_N[len]) {
		cout << "hermitian::N_unrank fatal: rk >= cnt_N[len]" << endl;
		exit(1);
		}
	if (len == 1) {
		v[0] = rk + 1;
		if (f_v) {
			cout << "N_unrank len=" << len << " done: ";
			INT_vec_print(cout, v, len);
			cout << endl;
			}
		return;
		}
	A = Q - q - 1;
	if (rk < A * cnt_N[len - 1]) {
		if (f_v) {
			cout << "N_unrank case 1" << endl;
			}
		coset = rk / cnt_N[len - 1];
		rk1 = rk % cnt_N[len - 1];
		N_unrank(v, len - 1, rk1, verbose_level - 1);
		if (coset == 0) {
			v[len - 1] = 0;
			}
		else {
			coset--;
			val = evaluate_hermitian_form(v, len - 1);
			if (f_v) {
				cout << "N_unrank case 1 val=" << val << endl;
				}
			coset0 = coset / (q + 1);
			rk0 = coset % (q + 1);
			if (f_v) {
				cout << "N_unrank case 1 coset0=" << coset0 << " rk0=" << rk0 << endl;
				}
			m_val = F->negate(val);
			if (f_v) {
				cout << "N_unrank case 1 m_val=" << m_val << endl;
				}
			log = log_beta[m_val];
			if (f_v) {
				cout << "N_unrank case 1 log=" << log << endl;
				}
			if (log == -1) {
				cout << "hermitian::N_unrank fatal: log == -1" << endl;
				exit(1);
				}
			if (coset0 >= log) {
				coset0++;
				}
			if (f_v) {
				cout << "N_unrank case 1 coset0=" << coset0 << endl;
				}
			v[len - 1] = F->mult(F->alpha_power(coset0), norm_one_elements[rk0]);
			}
		}
	else {
		if (f_v) {
			cout << "N_unrank case 2" << endl;
			}
		rk -= A * cnt_N[len - 1];

		coset = rk / cnt_S[len - 1];
		if (f_v) {
			cout << "N_unrank case 2 coset=" << coset << endl;
			}
		rk1 = rk % cnt_S[len - 1];
		if (f_v) {
			cout << "N_unrank case 2 rk1=" << rk1 << endl;
			}
		S_unrank(v, len - 1, rk1, verbose_level - 1);
		v[len - 1] = 1 + coset;
		}
	if (f_v) {
		cout << "N_unrank len=" << len << " done: ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
}

INT hermitian::N_rank(INT *v, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk, rk1, coset, rk0, coset0, val, m_val, log, a;
	
	if (f_v) {
		cout << "N_rank len=" << len << endl;
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (len == 1) {
		rk = v[0] - 1;
		if (f_v) {
			cout << "N_rank len=" << len << " done, rk=" << rk << endl;
			}
		return rk;
		}
	val = evaluate_hermitian_form(v, len - 1);
	if (val) {
		if (f_v) {
			cout << "N_rank case 1" << endl;
			}
		rk1 = N_rank(v, len - 1, verbose_level - 1);
		// case 1
		if (v[len - 1] == 0) {
			coset = 0;
			}
		else {
			m_val = F->negate(val);
			if (f_v) {
				cout << "N_rank case 1 m_val=" << m_val << endl;
				}
			log = log_beta[m_val];
			if (f_v) {
				cout << "N_rank case 1 log=" << log << endl;
				}
			if (log == -1) {
				cout << "hermitian::N_rank fatal: log == -1" << endl;
				exit(1);
				}
			a = F->N2(v[len - 1]);
			coset0 = log_beta[a];
			if (f_v) {
				cout << "N_rank case 1 coset0=" << coset0 << endl;
				}
			a = F->mult(v[len - 1], F->inverse(F->alpha_power(coset0)));
			if (coset0 > log) {
				coset0--;
				}
			if (f_v) {
				cout << "N_rank case 1 coset0=" << coset0 << endl;
				}
			rk0 = index_of_norm_one_element[a];
			if (rk0 == -1) {
				cout << "N_rank not an norm one element" << endl;
				exit(1);
				}
			if (f_v) {
				cout << "N_rank case 1 rk0=" << rk0 << endl;
				}
			coset = coset0 * (q + 1) + rk0;
			coset++;
			}
		rk = coset * cnt_N[len - 1] + rk1;
		}
	else {
		if (f_v) {
			cout << "N_rank case 2" << endl;
			}
		rk = (Q - q - 1) * cnt_N[len - 1];
		coset = v[len - 1] - 1;
		if (f_v) {
			cout << "    case 2 coset=" << coset << endl;
			}
		rk1 = S_rank(v, len - 1, verbose_level - 1);
		if (f_v) {
			cout << "N_rank case 2 rk1=" << rk1 << endl;
			}
		rk += coset * cnt_S[len - 1] + rk1;
		}
	if (f_v) {
		cout << "N_rank len=" << len << " done, rk=" << rk << endl;
		}
	return rk;
}

void hermitian::N1_unrank(INT *v, INT len, INT rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT coset, rk2, coset2, rk1, coset1, val, new_val, log, A, a, i;
	
	if (f_v) {
		cout << "N1_unrank len=" << len << " rk=" << rk << endl;
		}
	if (rk >= cnt_N1[len]) {
		cout << "hermitian::N1_unrank fatal: rk >= cnt_N1[len]" << endl;
		exit(1);
		}
	if (len == 1) {
		v[0] = norm_one_elements[rk];
		if (f_v) {
			cout << "N1_unrank len=" << len << " done: ";
			INT_vec_print(cout, v, len);
			cout << endl;
			}
		return;
		}
	if (rk < cnt_N1[len - 1]) {
		if (f_v) {
			cout << "N1_unrank case 0" << endl;
			}
		N1_unrank(v, len - 1, rk, verbose_level - 1);
		v[len - 1] = 0;
		if (f_v) {
			cout << "N1_unrank len=" << len << " done: ";
			INT_vec_print(cout, v, len);
			cout << endl;
			}
		return;
		}
	rk -= cnt_N1[len - 1];
	//A = (q + 1) * (cnt_N[len - 1] - cnt_N1[len - 1]);
	A = (q + 1) * (q - 2) * cnt_N1[len - 1];
	if (rk < A) {
		if (f_v) {
			cout << "N1_unrank case 1" << endl;
			}
		coset1 = rk / ((q - 2) * cnt_N1[len - 1]);
		rk1 = rk % ((q - 2) * cnt_N1[len - 1]);
		coset2 = rk1 / cnt_N1[len - 1];
		rk2 = rk1 % cnt_N1[len - 1];
		if (f_v) {
			cout << "N1_unrank case 1 coset1=" << coset1 << " rk1=" << rk1 << endl;
			}
		if (f_v) {
			cout << "N1_unrank case 1 coset2=" << coset2 << " rk2=" << rk2 << endl;
			}
		
		N1_unrank(v, len - 1, rk2, verbose_level - 1);
		val = evaluate_hermitian_form(v, len - 1);
		if (f_v) {
			cout << "N1_unrank case 1 val=" << val << endl;
			}
		if (val != 1) {
			cout << "N1_unrank case 1 error val=" << val << " should be 1" << endl;
			exit(1);
			}
		coset2++;
		if (f_v) {
			cout << "N1_unrank case 1 coset2=" << coset2 << endl;
			}
		a = F->alpha_power(coset2);
		if (f_v) {
			cout << "N1_unrank case 1 a=" << a << endl;
			}
		for (i = 0; i < len - 1; i++) {
			v[i] = F->mult(a, v[i]);
			}
		val = evaluate_hermitian_form(v, len - 1);
		if (f_v) {
			cout << "N1_unrank case 1 val=" << val << endl;
			}
		new_val = F->add(1, F->negate(val));
		if (f_v) {
			cout << "N1_unrank case 1 new_val=" << new_val << endl;
			}
		log = log_beta[new_val];
		if (f_v) {
			cout << "N_unrank case 1 log=" << log << endl;
			}
		if (log == -1) {
			cout << "hermitian::N_unrank fatal: log == -1" << endl;
			exit(1);
			}

		v[len - 1] = F->mult(F->alpha_power(log), norm_one_elements[coset1]);
		}
	else {
		if (f_v) {
			cout << "N1_unrank case 2" << endl;
			}
		rk -= A;

		coset = rk / cnt_S[len - 1];
		rk1 = rk % cnt_S[len - 1];
		if (f_v) {
			cout << "N1_unrank case 2 coset=" << coset << " rk1=" << rk1 << endl;
			}
		S_unrank(v, len - 1, rk1, verbose_level - 1);
		v[len - 1] = norm_one_elements[coset];
		}
	if (f_v) {
		cout << "N1_unrank len=" << len << " done: ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
}

INT hermitian::N1_rank(INT *v, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk, coset, rk2, coset2, rk1, coset1, val, new_val, log, A, a, av, i, log1;
	
	if (f_v) {
		cout << "N1_rank len=" << len << " : ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (len == 1) {
		rk = index_of_norm_one_element[v[0]];
		if (f_v) {
			cout << "N1_rank len=" << len << " done, rk=" << rk << endl;
			}
		return rk;
		}
	if (v[len - 1] == 0) {
		if (f_v) {
			cout << "N1_rank case 0" << endl;
			}
		rk = N1_rank(v, len - 1, verbose_level - 1);
		if (f_v) {
			cout << "N1_rank len=" << len << " done, rk=" << rk << endl;
			}
		return rk;
		}
	rk = cnt_N1[len - 1];


	//A = (q + 1) * (cnt_N[len - 1] - cnt_N1[len - 1]);
	A = (q + 1) * (q - 2) * cnt_N1[len - 1];
	val = evaluate_hermitian_form(v, len - 1);
	if (val) {
		if (f_v) {
			cout << "N1_rank case 1" << endl;
			}
		coset2 = log_beta[val];
		a = F->alpha_power(coset2);
		av = F->inverse(a);
		if (f_v) {
			cout << "N1_rank case 1 a=" << a << endl;
			}
		for (i = 0; i < len - 1; i++) {
			v[i] = F->mult(av, v[i]);
			}
		rk2 = N1_rank(v, len - 1, verbose_level - 1);
#if 0
		val = evaluate_hermitian_form(v, len - 1);
		if (val != 1) {
			cout << "N1_rank val != 1" << endl;
			exit(1);
			}
#endif
		coset2--;

		new_val = F->add(1, F->negate(val));
		if (f_v) {
			cout << "N1_rank case 1 new_val=" << new_val << endl;
			}
		log = log_beta[new_val];
		if (f_v) {
			cout << "N1_rank case 1 log=" << log << endl;
			}
		if (log == -1) {
			cout << "hermitian::N1_rank fatal: log == -1" << endl;
			exit(1);
			}
		a = F->N2(v[len - 1]);
		log1 = log_beta[a];
		if (log1 != log) {
			cout << "hermitian::N1_rank fatal: log1 != log" << endl;
			exit(1);
			}
		a = F->inverse(F->alpha_power(log));
		a = F->mult(a, v[len - 1]);
		coset1 = index_of_norm_one_element[a];
		if (coset1 == -1) {
			cout << "hermitian::N1_rank fatal: coset1 == -1" << endl;
			exit(1);
			}
		rk1 = coset2 * cnt_N1[len - 1] + rk2;
		rk += coset1 * ((q - 2) * cnt_N1[len - 1]) + rk1;
		}
	else {
		if (f_v) {
			cout << "N1_rank case 2" << endl;
			}
		rk += A;

		rk1 = S_rank(v, len - 1, verbose_level - 1);
		coset = index_of_norm_one_element[v[len - 1]];
		if (f_v) {
			cout << "N1_rank case 2 coset=" << coset << " rk1=" << rk1 << endl;
			}

		rk += coset * cnt_S[len - 1] + rk1;
		}


	if (f_v) {
		cout << "N1_rank len=" << len << " done, rk=" << rk << endl;
		}
	return rk;
}

void hermitian::S_unrank(INT *v, INT len, INT rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk1, coset, log, val, m_val;
	
	if (rk >= cnt_S[len]) {
		cout << "hermitian::S_unrank fatal: rk >= cnt_S[len]" << endl;
		exit(1);
		}
	if (len == 1) {
		v[0] = 0;
		return;
		}
	if (rk < (q + 1) * cnt_N[len - 1]) {
		if (f_v) {
			cout << "S_unrank case 1" << endl;
			}
		coset = rk / cnt_N[len - 1];
		rk1 = rk % cnt_N[len - 1];
		if (f_v) {
			cout << "S_unrank case 1 coset=" << coset << " rk1=" << rk1 << endl;
			}
		N_unrank(v, len - 1, rk1, verbose_level);
		val = evaluate_hermitian_form(v, len - 1);
		if (f_v) {
			cout << "S_unrank case 1 val=" << val << endl;
			}
		m_val = F->negate(val);
		if (f_v) {
			cout << "S_unrank case 1 m_val=" << m_val << endl;
			}
		log = log_beta[m_val];
		if (f_v) {
			cout << "S_unrank case 1 log=" << log << endl;
			}
		if (log == -1) {
			cout << "hermitian::S_unrank fatal: log == -1" << endl;
			exit(1);
			}
		v[len - 1] = F->mult(F->alpha_power(log), norm_one_elements[coset]);
		}
	else {
		if (f_v) {
			cout << "S_unrank case 2" << endl;
			}
		rk -= (q + 1) * cnt_N[len - 1];
		S_unrank(v, len - 1, rk, verbose_level);
		v[len - 1] = 0;
		}
	if (f_v) {
		cout << "S_unrank len=" << len << " done: ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	
}

INT hermitian::S_rank(INT *v, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk, rk1, coset, log, val, m_val, a, log1;
	
	if (f_v) {
		cout << "S_rank len=" << len << ": ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (len == 1) {
		if (v[0]) {
			cout << "hermitian::S_rank v[0]" << endl;
			exit(1);
			}
		return 0;
		}
	if (v[len - 1]) {
		if (f_v) {
			cout << "S_rank case 1" << endl;
			}
		rk1 = N_rank(v, len - 1, verbose_level);
		val = evaluate_hermitian_form(v, len - 1);
		if (f_v) {
			cout << "S_rank case 1 val=" << val << endl;
			}
		m_val = F->negate(val);
		if (f_v) {
			cout << "S_rank case 1 m_val=" << m_val << endl;
			}
		log = log_beta[m_val];
		if (f_v) {
			cout << "S_rank case 1 log=" << log << endl;
			}
		if (log == -1) {
			cout << "hermitian::S_rank fatal: log == -1" << endl;
			exit(1);
			}
		a = F->N2(v[len - 1]);
		log1 = log_beta[a];
		if (log1 != log) {
			cout << "hermitian::S_rank fatal: log1 != log" << endl;
			exit(1);
			}
		a = F->mult(v[len - 1], F->inverse(F->alpha_power(log)));
		coset = index_of_norm_one_element[a];
		rk = coset * cnt_N[len - 1] + rk1;
		}
	else {
		if (f_v) {
			cout << "S_rank case 2" << endl;
			}
		rk = S_rank(v, len - 1, verbose_level);
		rk += (q + 1) * cnt_N[len - 1];
		}
	if (f_v) {
		cout << "S_rank len=" << len << " done, rk=" << rk << endl;
		}
	return rk;
}


void hermitian::Sbar_unrank(INT *v, INT len, INT rk, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT log, a, b, i;
	
	if (rk >= cnt_Sbar[len]) {
		cout << "hermitian::Sbar_unrank fatal: rk >= cnt_Sbar[len]" << endl;
		exit(1);
		}
	if (len == 1) {
		cout << "hermitian::Sbar_unrank fatal: len == 1" << endl;
		exit(1);
		}
	if (rk < cnt_Sbar[len - 1]) {
		if (f_v) {
			cout << "Sbar_unrank case 1" << endl;
			}
		Sbar_unrank(v, len - 1, rk, verbose_level);
		v[len - 1] = 0;
		}
	else {
		if (f_v) {
			cout << "Sbar_unrank case 2" << endl;
			}
		rk -= cnt_Sbar[len - 1];

		N1_unrank(v, len - 1, rk, verbose_level);
		a = F->negate(1);
		log = log_beta[a];
		b = F->alpha_power(log);
		if (f_v) {
			cout << "Sbar_unrank case 2 log=" << log << endl;
			}
		if (log == -1) {
			cout << "hermitian::Sbar_unrank fatal: log == -1" << endl;
			exit(1);
			}
		for (i = 0; i < len - 1; i++) {
			v[i] = F->mult(b, v[i]);
			}
		v[len - 1] = 1;
		//v[len - 1] = F->mult(F->alpha_power(log), norm_one_elements[0]);
		}
	if (f_v) {
		cout << "Sbar_unrank len=" << len << " done: ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	
}

INT hermitian::Sbar_rank(INT *v, INT len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk, val, a, b, bv, log, i;
	
	if (f_v) {
		cout << "Sbar_rank len=" << len << " : ";
		INT_vec_print(cout, v, len);
		cout << endl;
		}
	if (len == 1) {
		cout << "hermitian::Sbar_rank fatal: len == 1" << endl;
		exit(1);
		}
	if (v[len - 1] == 0) {
		if (f_v) {
			cout << "Sbar_rank case 1" << endl;
			}
		rk = Sbar_rank(v, len - 1, verbose_level);
		}
	else {
		if (f_v) {
			cout << "Sbar_rank case 2" << endl;
			}

		PG_element_normalize(*F, v, 1, len);
		rk = cnt_Sbar[len - 1];

		val = evaluate_hermitian_form(v, len - 1); 
				// val must be minus_one
		a = F->negate(1);
		if (val != a) {
			cout << "Sbar_rank case 2 val != F->negate(1)" << endl;
			exit(1);
			}
		log = log_beta[val];
		b = F->alpha_power(log);
		bv = F->inverse(b);
		for (i = 0; i < len - 1; i++) {
			v[i] = F->mult(v[i], bv);
			}
		val = evaluate_hermitian_form(v, len - 1);
		if (val != 1) {
			cout << "Sbar_rank case 2 val != 1" << endl;
			exit(1);
			}

		rk += N1_rank(v, len - 1, verbose_level);
		}
	
	if (f_v) {
		cout << "Sbar_rank done, rk=" << rk << endl;
		}
	return rk;
}




