// hermitian.cpp
// 
// Anton Betten
// 3/19/2010
//
// 
//
//

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace geometry {
namespace other_geometry {


hermitian::hermitian()
{
	Record_birth();
	F = NULL;
	Q = 0;
	q = 0;
	k = 0;
	cnt_N = NULL;
	cnt_N1 = NULL;
	cnt_S = NULL;
	cnt_Sbar = NULL;
	norm_one_elements = NULL;
	index_of_norm_one_element = NULL;
	alpha = 0;
	beta = 0;
	log_beta = NULL;
	beta_power = NULL;
}

hermitian::~hermitian()
{
	Record_death();
	if (cnt_N) {
		FREE_int(cnt_N);
	}
	if (cnt_N1) {
		FREE_int(cnt_N1);
	}
	if (cnt_S) {
		FREE_int(cnt_S);
	}
	if (cnt_Sbar) {
		FREE_int(cnt_Sbar);
	}
	if (norm_one_elements) {
		FREE_int(norm_one_elements);
	}
	if (index_of_norm_one_element) {
		FREE_int(index_of_norm_one_element);
	}
	if (log_beta) {
		FREE_int(log_beta);
	}
	if (beta_power) {
		FREE_int(beta_power);
	}
}


void hermitian::init(
		algebra::field_theory::finite_field *F,
		int nb_vars, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, a;
	algebra::number_theory::number_theory_domain NT;
	
	hermitian::F = F;
	hermitian::Q = F->q;
	hermitian::q = NT.i_power_j(F->p, F->e >> 1);
	hermitian::k = nb_vars;
	if (f_v) {
		cout << "hermitian::init Q=" << F->q << " q=" << q
				<< " nb_vars=" << nb_vars << endl;
	}
	if (F->e % 2) {
		cout << "hermitian::init field must have a "
				"quadratic subfield" << endl;
		exit(1);
	}
	cnt_N = NEW_int(k + 1);
	cnt_N1 = NEW_int(k + 1);
	cnt_S = NEW_int(k + 1);
	cnt_Sbar = NEW_int(k + 1);
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

	norm_one_elements = NEW_int(q + 1);
	index_of_norm_one_element = NEW_int(Q);
	log_beta = NEW_int(Q);
	beta_power = NEW_int(q);
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
		Int_vec_print(cout, norm_one_elements, q + 1);
		cout << endl;
	}
	cout << "i : norm_one_elements[i] : "
			"F->N2(norm_one_elements[i])" << endl;
	for (i = 0; i < q + 1; i++) {
		cout << i << " : " << norm_one_elements[i] << " : "
				<< F->N2(norm_one_elements[i]) << endl;
	}
	alpha = F->p;
	beta = F->alpha_power(q + 1);
	for (i = 0; i < q - 1; i++) {
		j = F->power(beta, i);
		beta_power[i] = j;
		log_beta[j] = i;
	}
}

int hermitian::nb_points()
{
	return cnt_Sbar[k];
}

void hermitian::unrank_point(
		int *v, int rk)
{
	Sbar_unrank(v, k, rk, 0 /*verbose_level*/);
}

int hermitian::rank_point(
		int *v)
{
	int rk;

	rk = Sbar_rank(v, k, 0 /*verbose_level*/);
	return rk;
}

void hermitian::list_of_points_embedded_in_PG(
	long int *&Pts, int &nb_pts, int verbose_level)
{
	long int i, rk;
	int *v;

	v = NEW_int(k);
	nb_pts = nb_points();
	Pts = NEW_lint(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		unrank_point(v, i);
		F->Projective_space_basic->PG_element_rank_modified(
				v, 1, k, rk);
		Pts[i] = rk;
	}
}

void hermitian::list_all_N(
		int verbose_level)
{
	int *v;
	int i, j, val0, val;

	cout << "list_all_N:" << endl;
	v = NEW_int(k);
	for (i = 0; i < cnt_N[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 165) {verbose_level += 2;}
		N_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		Int_vec_print(cout, v, k);
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

void hermitian::list_all_N1(
		int verbose_level)
{
	int *v;
	int i, j, val0, val;

	cout << "list_all_N1:" << endl;
	v = NEW_int(k);
	for (i = 0; i < cnt_N1[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 15) {verbose_level += 2;}
		N1_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		Int_vec_print(cout, v, k);
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

void hermitian::list_all_S(
		int verbose_level)
{
	int *v;
	int i, j, val0, val;

	cout << "list_all_S:" << endl;
	v = NEW_int(k);
	for (i = 0; i < cnt_S[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 6) {verbose_level += 2;}
		S_unrank(v, k, i, verbose_level - 2);
		val0 = evaluate_hermitian_form(v, k - 1);
		val = evaluate_hermitian_form(v, k);
		cout << setw(5) << i << " : ";
		Int_vec_print(cout, v, k);
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

void hermitian::list_all_Sbar(
		int verbose_level)
{
	int *v;
	int i, j, a, h, val0, val;

	cout << "list_all_Sbar:" << endl;
	v = NEW_int(k);
	for (i = 0; i < cnt_Sbar[k]; i++) {
		//cout << "i=" << i << endl;
		//if (i == 6) {verbose_level += 2;}

		for (h = 0; h < q - 1; h++) {
			// loop over all elements in the subfield F_q:
			a = F->alpha_power(h * (q + 1));
	

			Sbar_unrank(v, k, i, 0 /*verbose_level*/);
			F->Linear_algebra->scalar_multiply_vector_in_place(a, v, k);
#if 0
			for (u = 0; u < k; u++) {
				v[u] = F->mult(a, v[u]);
			}
#endif
			val0 = evaluate_hermitian_form(v, k - 1);
			val = evaluate_hermitian_form(v, k);
			cout << setw(5) << i << "," << h << " : ";
			Int_vec_print(cout, v, k);
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


int hermitian::evaluate_hermitian_form(
		int *v, int len)
// \sum_{i=0}^{len-1} X_i^{q+1}
{
	int i, a, b;

	a = 0;
	for (i = 0; i < len; i++) {
		b = F->N2(v[i]);
		a = F->add(a, b);
		//cout << "b=" << b << " a=" << a << endl;
	}
	//cout << "hermitian::evaluate_hermitian_form ";
	//int_vec_print(cout, v, len);
	//cout << "val=" << a << endl;
	return a;
}

void hermitian::N_unrank(
		int *v, int len, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk1, coset, rk0, coset0, A, val, m_val, log;
	
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
			Int_vec_print(cout, v, len);
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
				cout << "N_unrank case 1 coset0=" << coset0
						<< " rk0=" << rk0 << endl;
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
			v[len - 1] = F->mult(F->alpha_power(coset0),
					norm_one_elements[rk0]);
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
		Int_vec_print(cout, v, len);
		cout << endl;
	}
}

int hermitian::N_rank(
		int *v, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, rk1, coset, rk0, coset0, val, m_val, log, a;
	
	if (f_v) {
		cout << "N_rank len=" << len << endl;
		Int_vec_print(cout, v, len);
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

void hermitian::N1_unrank(
		int *v, int len, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int coset, rk2, coset2, rk1, coset1, val, new_val, log, A, a, i;
	
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
			Int_vec_print(cout, v, len);
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
			Int_vec_print(cout, v, len);
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
			cout << "N1_unrank case 1 coset1=" << coset1
					<< " rk1=" << rk1 << endl;
		}
		if (f_v) {
			cout << "N1_unrank case 1 coset2=" << coset2
					<< " rk2=" << rk2 << endl;
		}
		
		N1_unrank(v, len - 1, rk2, verbose_level - 1);
		val = evaluate_hermitian_form(v, len - 1);
		if (f_v) {
			cout << "N1_unrank case 1 val=" << val << endl;
		}
		if (val != 1) {
			cout << "N1_unrank case 1 error val=" << val
					<< " should be 1" << endl;
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

		v[len - 1] = F->mult(F->alpha_power(log),
				norm_one_elements[coset1]);
	}
	else {
		if (f_v) {
			cout << "N1_unrank case 2" << endl;
		}
		rk -= A;

		coset = rk / cnt_S[len - 1];
		rk1 = rk % cnt_S[len - 1];
		if (f_v) {
			cout << "N1_unrank case 2 coset=" << coset
					<< " rk1=" << rk1 << endl;
		}
		S_unrank(v, len - 1, rk1, verbose_level - 1);
		v[len - 1] = norm_one_elements[coset];
	}
	if (f_v) {
		cout << "N1_unrank len=" << len << " done: ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
}

int hermitian::N1_rank(
		int *v, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, coset, rk2, coset2, rk1, coset1, val;
	int new_val, log, A, a, av, i, log1;
	
	if (f_v) {
		cout << "N1_rank len=" << len << " : ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	if (len == 1) {
		rk = index_of_norm_one_element[v[0]];
		if (f_v) {
			cout << "N1_rank len=" << len << " done, "
					"rk=" << rk << endl;
		}
		return rk;
	}
	if (v[len - 1] == 0) {
		if (f_v) {
			cout << "N1_rank case 0" << endl;
		}
		rk = N1_rank(v, len - 1, verbose_level - 1);
		if (f_v) {
			cout << "N1_rank len=" << len << " done, "
					"rk=" << rk << endl;
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
			cout << "N1_rank case 2 coset=" << coset
					<< " rk1=" << rk1 << endl;
		}

		rk += coset * cnt_S[len - 1] + rk1;
	}


	if (f_v) {
		cout << "N1_rank len=" << len << " done, rk=" << rk << endl;
	}
	return rk;
}

void hermitian::S_unrank(
		int *v,
		int len, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk1, coset, log, val, m_val;
	
	if (rk >= cnt_S[len]) {
		cout << "hermitian::S_unrank fatal: "
				"rk >= cnt_S[len]" << endl;
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
			cout << "S_unrank case 1 coset=" << coset
					<< " rk1=" << rk1 << endl;
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
		v[len - 1] = F->mult(F->alpha_power(log),
				norm_one_elements[coset]);
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
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	
}

int hermitian::S_rank(
		int *v, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, rk1, coset, log, val, m_val, a, log1;
	
	if (f_v) {
		cout << "S_rank len=" << len << ": ";
		Int_vec_print(cout, v, len);
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


void hermitian::Sbar_unrank(
		int *v,
		int len, int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int log, a, b, i;
	
	if (rk >= cnt_Sbar[len]) {
		cout << "hermitian::Sbar_unrank fatal: "
				"rk >= cnt_Sbar[len]" << endl;
		exit(1);
	}
	if (len == 1) {
		cout << "hermitian::Sbar_unrank fatal: "
				"len == 1" << endl;
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
			cout << "hermitian::Sbar_unrank fatal: "
					"log == -1" << endl;
			exit(1);
		}
		for (i = 0; i < len - 1; i++) {
			v[i] = F->mult(b, v[i]);
		}
		v[len - 1] = 1;
		//v[len - 1] = F->mult(F->alpha_power(log),
		// norm_one_elements[0]);
	}
	if (f_v) {
		cout << "Sbar_unrank len=" << len << " done: ";
		Int_vec_print(cout, v, len);
		cout << endl;
	}
	
}

int hermitian::Sbar_rank(
		int *v, int len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk, val, a, b, bv, log, i;
	
	if (f_v) {
		cout << "Sbar_rank len=" << len << " : ";
		Int_vec_print(cout, v, len);
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

		F->Projective_space_basic->PG_element_normalize(
				v, 1, len, 0 /* verbose_level */);
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

void hermitian::create_latex_report(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "hermitian::create_latex_report" << endl;
	}

	{

		string fname;
		string author;
		string title;
		string extra_praeamble;



		fname = "H_" + std::to_string(k - 1) + "_" + std::to_string(Q) + ".tex";
		title = "Hermitian Variety  ${\\rm H}(" + std::to_string(k - 1) + "," + std::to_string(Q) + ")$";



		{
			ofstream ost(fname);
			other::l1_interfaces::latex_interface L;

			L.head(ost,
					false /* f_book*/,
					true /* f_title */,
					title, author,
					false /* f_toc */,
					false /* f_landscape */,
					true /* f_12pt */,
					true /* f_enlarged_page */,
					true /* f_pagenumbers */,
					extra_praeamble /* extra_praeamble */);


			if (f_v) {
				cout << "hermitian::create_latex_report "
						"before report" << endl;
			}
			report(ost, verbose_level);
			if (f_v) {
				cout << "hermitian::create_latex_report "
						"after report" << endl;
			}


			L.foot(ost);

		}
		other::orbiter_kernel_system::file_io Fio;

		cout << "written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "hermitian::create_latex_report done" << endl;
	}
}

void hermitian::report(
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hermitian::report" << endl;
	}

	//report_schemes(ost);

	report_points(ost, verbose_level);

	//report_points_by_type(ost, verbose_level);

	//report_lines(ost, verbose_level);

	if (f_v) {
		cout << "hermitian::report done" << endl;
	}
}

void hermitian::report_points(
		std::ostream &ost, int verbose_level)
{
	long int rk;
	long int *rk_in_PG;
	long int nb_pts;

	int *v;

	v = NEW_int(k);
	nb_pts = nb_points();

	rk_in_PG = NEW_lint(nb_pts);


	ost << "The Hermitian variety ${\\rm H}(" << k - 1 << "," << Q << ")$ "
			"contains " << nb_pts << " points:\\\\" << endl;
	ost << "\\begin{multicols}{2}" << endl;
	ost << "\\noindent" << endl;
	for (rk = 0; rk < nb_pts; rk++) {
		unrank_point(v, rk);
		F->Projective_space_basic->PG_element_rank_modified_lint(
				v, 1, k, rk_in_PG[rk], 0 /* verbose_level */);
		ost << "$P_{" << rk << "} = ";
		Int_vec_print(ost, v, k);
		ost << "=" << rk_in_PG[rk] << "$\\\\" << endl;
	}
	ost << "\\end{multicols}" << endl;
	ost << "All points: ";
	Lint_vec_print(ost, rk_in_PG, nb_pts);
	ost << "\\\\" << endl;

	FREE_int(v);
	FREE_lint(rk_in_PG);
}



}}}}


