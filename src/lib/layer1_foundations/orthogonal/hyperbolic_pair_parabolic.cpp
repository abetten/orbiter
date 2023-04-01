/*
 * hyperbolic_pair_parabolic.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: betten
 */

#include "foundations.h"

using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace orthogonal_geometry {





// #############################################################################
// orthogonal_parabolic.cpp
// #############################################################################

//##############################################################################
// ranking / unranking points according to the partition:
//##############################################################################

int hyperbolic_pair::parabolic_type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
			"type=" << type << " index=" << index
			<< " epsilon=" << epsilon << " n=" << n << endl;
	}
	if (type == 3) {
		int field, sub_index, len;

		len = alpha;
		field = index / len;
		sub_index = index % len;
		field++;
		if (f_vv) {
			cout << "field=" << field
					<< " sub_index=" << sub_index << endl;
		}
		O->subspace->Hyperbolic_pair->unrank_point(
				v_tmp2, 1, sub_index, verbose_level - 1);
		v_tmp2[n - 2] = 0;
		v_tmp2[n - 1] = field;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	else if (type == 4) {
		int field, sub_index, len;

		len = alpha;
		field = index / len;
		sub_index = index % len;
		field++;
		if (f_vv) {
			cout << "field=" << field << " sub_index=" << sub_index << endl;
		}
		O->subspace->Hyperbolic_pair->unrank_point(
				v_tmp2, 1, sub_index, verbose_level - 1);
		v_tmp2[n - 2] = field;
		v_tmp2[n - 1] = 0;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	else if (type == 5) {
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index << endl;
			cout << "parabolic_type_and_index_to_point_rk "
				"before subspace->unrank_point" << endl;
		}
		if (O->subspace == NULL) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
				"subspace == NULL" << endl;
			exit(1);
		}
		O->subspace->Hyperbolic_pair->unrank_point(
				v_tmp2, 1, index, verbose_level /*- 1*/);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
					"after subspace->unrank_point" << endl;
		}
		v_tmp2[n - 2] = 0;
		v_tmp2[n - 1] = 0;
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	else if (type == 6) {
		if (index < 1) {
			rk = pt_Q;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
					"type=" << type << " index=" << index
					<< " rk=" << rk << endl;
			}
			return rk;
		}
		else {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk, "
					"illegal index" << endl;
			exit(1);
		}
	}
	else if (type == 7) {
		if (index < 1) {
			rk = pt_P;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk "
					"type=" << type << " index=" << index
					<< " rk=" << rk << endl;
			}
			return rk;
		}
		else {
			cout << "hyperbolic_pair::parabolic_type_and_index_to_point_rk, "
				"illegal index" << endl;
			exit(1);
		}
	}
	else {
		if (O->Quadratic_form->f_even) {
			return parabolic_even_type_and_index_to_point_rk(
					type, index, verbose_level);
		}
		else {
			return parabolic_odd_type_and_index_to_point_rk(
					type, index, verbose_level);
		}
	}
}

int hyperbolic_pair::parabolic_even_type_and_index_to_point_rk(
		int type, int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int rk;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_even_type_and_index_to_point_rk "
			"type=" << type << " index=" << index << endl;
	}
	if (type == 1) {
		parabolic_even_type1_index_to_point(index, v_tmp2);
		rk = rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_even_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	else if (type == 2) {
		parabolic_even_type2_index_to_point(index, v_tmp2);
		rk = rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_even_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	cout << "hyperbolic_pair::parabolic_even_type_and_index_to_point_rk "
			"illegal type " << type << endl;
	exit(1);
}

void hyperbolic_pair::parabolic_even_type1_index_to_point(
		int index, int *v)
{
	int a, b;

	if (index >= p1) {
		cout << "hyperbolic_pair::parabolic_even_type1_index_to_point, "
				"index >= p1" << endl;
		exit(1);
	}
	O->zero_vector(v + 1, 1, 2 * (m - 1));
	a = 1 + index;
	b = F->inverse(a);
	v[0] = 1;
	v[1 + 2 * (m - 1) + 0] = a;
	v[1 + 2 * (m - 1) + 1] = b;
}

void hyperbolic_pair::parabolic_even_type2_index_to_point(
		int index, int *v)
{
	int a, b, c, d, l, ll, lll, field1, field2;
	int sub_index, sub_sub_index;

	l = (q - 1) * N1_mm1;
	if (index < l) {
		field1 = index / N1_mm1;
		sub_index = index % N1_mm1;
		v[0] = 0;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		a = 1 + field1;
		b = 1;
		c = a;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		O->change_form_value(v + 1, 1, m - 1, c);
		//int_vec_print(cout, v, n);
		return;
	}
	index -= l;
	ll = S_mm1 - 1;
	l = (q - 1) * ll;
	if (index < l) {
		field1 = index / ll;
		sub_index = index % ll;
		lll = Sbar_mm1;
		field2 = sub_index / lll;
		sub_sub_index = sub_index % lll;
		v[0] = 1;
		unrank_Sbar(v + 1, 1, m - 1, sub_sub_index);
		O->scalar_multiply_vector(
				v + 1, 1, n - 3, 1 + field2);
		a = 1 + field1;
		b = F->inverse(a);
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		return;
	}
	index -= l;
	l = (q - 2) * (q - 1) * N1_mm1;
	if (index < l) {
		ll = (q - 1) * N1_mm1;
		field1 = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		//cout << "field1=" << field1 << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		v[0] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		a = 2 + field1;
		b = 1 + field2;
		c = F->mult(a, F->inverse(b));
		v[1 + 2 * (m - 1) + 0] = b;
		v[1 + 2 * (m - 1) + 1] = c;
		d = F->add(1, a);
		O->change_form_value(v + 1, 1, m - 1, d);
		return;
	}
	else {
		cout << "hyperbolic_pair::parabolic_even_type2_index_to_point "
				"illegal index" << endl;
		exit(1);
	}
}

long int hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk(
		long int type, long int index,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	long int rk;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk "
			"type=" << type << " index=" << index << endl;
	}
	if (type == 1) {
		parabolic_odd_type1_index_to_point(
				index, v_tmp2, verbose_level);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk created ";
			Int_vec_print(cout, v_tmp2, n);
			cout << endl;
		}
		rk = rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	else if (type == 2) {
		parabolic_odd_type2_index_to_point(
				index, v_tmp2, verbose_level);
		rk = rank_point(v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk "
				"type=" << type << " index=" << index
				<< " rk=" << rk << endl;
		}
		return rk;
	}
	cout << "hyperbolic_pair::parabolic_odd_type_and_index_to_point_rk "
			"illegal type " << type << endl;
	exit(1);
}

void hyperbolic_pair::parabolic_odd_type1_index_to_point(
		long int index, int *v, int verbose_level)
{
	long int a, b, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_odd_type1_index_to_point "
				"m = " << m << " index = " << index << endl;
	}
	if (index >= p1) {
		cout << "hyperbolic_pair::parabolic_odd_type1_index_to_point, "
				"index >= p1" << endl;
		exit(1);
	}
	l = (q - 1) / 2 * N1_mm1;
	if (index < l) {
		ms_idx = index / N1_mm1;
		sub_index = index % N1_mm1;
		field1 = O->SN->minus_squares[ms_idx];
		if (f_v) {
			cout << "case a) ms_idx = " << ms_idx
				<< " sub_index=" << sub_index
				<< " field1 = " << field1 << endl;
		}
		v[0] = 0;
		v[1 + 2 * (m - 1) + 0] = field1;
		v[1 + 2 * (m - 1) + 1] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		c = F->negate(field1);
		O->change_form_value(v + 1, 1, m - 1, c);
		return;
	}
	index -= l;
	l = (q - 1) * S_mm1;
	if (index < l) {
		field1 = index / S_mm1;
		sub_index = index % S_mm1;
		if (f_v) {
			cout << "case b) sub_index=" << sub_index
					<< " field1 = " << field1 << endl;
		}
		if (sub_index == 0) {
			a = 1 + field1;
			b = F->mult(F->inverse(a), F->negate(1));
			v[0] = 1;
			v[1 + 2 * (m - 1) + 0] = a;
			v[1 + 2 * (m - 1) + 1] = b;
			O->zero_vector(v + 1, 1, n - 3);
			return;
		}
		else {
			sub_index--;
			field2 = sub_index / Sbar_mm1;
			sub_sub_index = sub_index % Sbar_mm1;
			//cout << "field1=" << field1 << " field2=" << field2
			//<< " sub_sub_index=" << sub_sub_index << endl;
			a = 1 + field1;
			b = F->mult(F->inverse(a), F->negate(1));
			v[0] = 1;
			v[1 + 2 * (m - 1) + 0] = a;
			v[1 + 2 * (m - 1) + 1] = b;
			unrank_Sbar(v + 1, 1, m - 1, sub_sub_index);
			O->scalar_multiply_vector(
					v + 1, 1, n - 3, 1 + field2);
			return;
		}
	}
	index -= l;
	l = ((q - 1) / 2 - 1) * (q - 1) * N1_mm1;
	ll = (q - 1) * N1_mm1;
	//cout << "index = " << index << " l=" << l << endl;
	if (index < l) {
		ms_idx = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		field1 = O->SN->minus_squares_without[ms_idx];
		if (f_v) {
			cout << "case c) ms_idx = " << ms_idx
				<< " sub_index=" << sub_index
				<< " field2 = " << field2
				<< " sub_sub_index=" << sub_sub_index
				<< " field1 = " << field1
				<< endl;
		}
		a = 1 + field2;
		b = F->mult(F->inverse(a), field1);
		v[0] = 1;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		c = F->negate(F->add(1, field1));
		O->change_form_value(v + 1, 1, m - 1, c);
		return;
	}
	else {
		cout << "hyperbolic_pair::parabolic_odd_type1_index_to_point "
				"illegal index" << endl;
		exit(1);
	}
}

void hyperbolic_pair::parabolic_odd_type2_index_to_point(
		long int index, int *v, int verbose_level)
{
	long int a, b, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_odd_type2_index_to_point "
				"index = " << index << endl;
	}
	if (index >= p1) {
		cout << "hyperbolic_pair::parabolic_odd_type2_index_to_point, "
				"index >= p1" << endl;
		exit(1);
	}
	l = (q - 1) / 2 * N1_mm1;
	if (index < l) {
		ms_idx = index / N1_mm1;
		sub_index = index % N1_mm1;
		field1 = O->SN->minus_nonsquares[ms_idx];
		if (f_v) {
			cout << "case 1 ms_idx=" << ms_idx
					<< " field1=" << field1
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 0;
		v[1 + 2 * (m - 1) + 0] = field1;
		v[1 + 2 * (m - 1) + 1] = 1;
		unrank_N1(v + 1, 1, m - 1, sub_index);
		c = F->negate(field1);
		O->change_form_value(v + 1, 1, m - 1, c);
		return;
	}
	index -= l;
	l = (q - 1) / 2 * (q - 1) * N1_mm1;
	ll = (q - 1) * N1_mm1;
	if (index < l) {
		ms_idx = index / ll;
		sub_index = index % ll;
		field2 = sub_index / N1_mm1;
		sub_sub_index = sub_index % N1_mm1;
		field1 = O->SN->minus_nonsquares[ms_idx];
		if (f_v) {
			cout << "case 2 ms_idx=" << ms_idx
				<< " field1=" << field1 << " field2=" << field2
				<< " sub_sub_index=" << sub_sub_index << endl;
		}
		//cout << "ms_idx=" << ms_idx << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		a = 1 + field2;
		b = F->mult(F->inverse(a), field1);
		v[0] = 1;
		v[1 + 2 * (m - 1) + 0] = a;
		v[1 + 2 * (m - 1) + 1] = b;
		unrank_N1(v + 1, 1, m - 1, sub_sub_index);
		c = F->negate(F->add(1, field1));
		O->change_form_value(v + 1, 1, m - 1, c);
		return;
	}
	cout << "hyperbolic_pair::parabolic_odd_type2_index_to_point "
			"illegal index" << endl;
	exit(1);
}

void hyperbolic_pair::parabolic_point_rk_to_type_and_index(
		long int rk, long int &type, long int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
				"rk = " << rk << endl;
	}
	if (rk == pt_Q) {
		type = 6;
		index = 0;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
		}
		return;
	}
	if (rk == pt_P) {
		type = 7;
		index = 0;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
		}
		return;
	}
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index created vector ";
		Int_vec_print(cout, v_tmp2, n);
		cout << endl;
	}
	if (v_tmp2[n - 2] == 0 && v_tmp2[n - 1]) {
		long int field, sub_index, len;
		type = 3;
		len = alpha;
		parabolic_normalize_point_wrt_subspace(v_tmp2, 1);
		field = v_tmp2[n - 1];
		sub_index = O->subspace->Hyperbolic_pair->rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "field=" << field
					<< " sub_index=" << sub_index << endl;
		}
		index = (field - 1) * len + sub_index;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
					"rk = " << rk
					<< " type = " << type
					<< " index = " << index << endl;
		}
		return;
	}
	else if (v_tmp2[n - 2] && v_tmp2[n - 1] == 0) {
		long int field, sub_index, len;
		type = 4;
		len = alpha;
		parabolic_normalize_point_wrt_subspace(v_tmp2, 1);
		field = v_tmp2[n - 2];
		sub_index = O->subspace->Hyperbolic_pair->rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_vv) {
			cout << "field=" << field
					<< " sub_index=" << sub_index << endl;
		}
		index = (field - 1) * len + sub_index;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
		}
		return;
	}
	else if (v_tmp2[n - 2] == 0 && v_tmp2[n - 1] == 0) {
		type = 5;
		index = O->subspace->Hyperbolic_pair->rank_point(
				v_tmp2, 1, verbose_level - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_rk_to_type_and_index "
					"rk = " << rk << " type = " << type
					<< " index = " << index << endl;
		}
		return;
	}
	if (O->Quadratic_form->f_even) {
		parabolic_even_point_rk_to_type_and_index(rk,
				type, index, verbose_level);
	}
	else {
		parabolic_odd_point_rk_to_type_and_index(rk,
				type, index, verbose_level);
	}
}

void hyperbolic_pair::parabolic_even_point_rk_to_type_and_index(
		long int rk, long int &type,
		long int &index, int verbose_level)
{
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	parabolic_even_point_to_type_and_index(v_tmp2,
			type, index, verbose_level);
}

void hyperbolic_pair::parabolic_even_point_to_type_and_index(
		int *v, long int &type,
		long int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_start_with_one, value_middle, value_end, f_middle_is_zero;
	long int a, b, /*c,*/ l, ll, lll, field1, field2, sub_index, sub_sub_index;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index:";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[0] != 0 && v[0] != 1) {
		cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index: "
				"error in unrank_point" << endl;
		exit(1);
	}
	parabolic_point_properties(v, 1, n,
		f_start_with_one, value_middle, value_end, verbose_level);
	if (value_middle == 0) {
		f_middle_is_zero = O->is_zero_vector(v + 1, 1, n - 3);
	}
	else {
		f_middle_is_zero = FALSE;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index: "
				"f_start_with_one=" << f_start_with_one
				<< " value_middle=" << value_middle
				<< " f_middle_is_zero=" << f_middle_is_zero
				<< " value_end=" << value_end << endl;
	}
	if (f_start_with_one &&
			value_middle == 0 &&
			f_middle_is_zero &&
			value_end == 1) {
		type = 1;
		a = v[1 + 2 * (m - 1) + 0];
		b = v[1 + 2 * (m - 1) + 1];
		index = a - 1;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
				"type = " << type << " index = " << index << endl;
		}
		return;
	}
	else if (value_end) {
		type = 2;
		index = 0;
		if (!f_start_with_one) {
			O->change_form_value(
					v + 1, 1, m - 1, F->inverse(value_middle));
			sub_index = rank_N1(v + 1, 1, m - 1);
			a = v[1 + 2 * (m - 1) + 0];
			b = v[1 + 2 * (m - 1) + 1];
			field1 = a - 1;
			index += field1 * N1_mm1 + sub_index;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
			return;
		}
		index += (q - 1) * N1_mm1;
		ll = S_mm1 - 1;
		l = (q - 1) * ll;
		if (value_middle == 0) {
			a = v[1 + 2 * (m - 1) + 0];
			b = v[1 + 2 * (m - 1) + 1];
			field2 = O->last_non_zero_entry(
					v + 1, 1, n - 3);
			O->scalar_multiply_vector(
					v + 1, 1, n - 3, F->inverse(field2));
			sub_sub_index = rank_Sbar(
					v + 1, 1, m - 1);
			field2--;
			lll = Sbar_mm1;
			sub_index = field2 * lll + sub_sub_index;
			field1 = a - 1;
			index += field1 * ll + sub_index;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
			return;
		}
		index += l;
		l = (q - 2) * (q - 1) * N1_mm1;
		O->change_form_value(
				v + 1, 1, m - 1, F->inverse(value_middle));
		sub_sub_index = rank_N1(
				v + 1, 1, m - 1);
		a = F->add(1, value_middle);
		b = v[1 + 2 * (m - 1) + 0];
		//c = v[1 + 2 * (m - 1) + 1];
		if (a == 0 || a == 1) {
			cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
					"a == 0 || a == 1" << endl;
			exit(1);
		}
		if (b == 0) {
			cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
					"b == 0" << endl;
			exit(1);
		}
		field2 = b - 1;
		field1 = a - 2;
		//cout << "field1=" << field1 << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		ll = (q - 1) * N1_mm1;
		index += field1 * ll + sub_index;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
		}
		return;
	}
	else {
		cout << "hyperbolic_pair::parabolic_even_point_to_type_and_index, "
				"unknown type, type = " << type << endl;
		exit(1);
	}
}

void hyperbolic_pair::parabolic_odd_point_rk_to_type_and_index(
		long int rk, long int &type,
		long int &index, int verbose_level)
{
	unrank_point(v_tmp2, 1, rk, verbose_level - 1);
	parabolic_odd_point_to_type_and_index(v_tmp2,
			type, index, verbose_level);
}

void hyperbolic_pair::parabolic_odd_point_to_type_and_index(
		int *v, long int &type,
		long int &index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_start_with_one, value_middle, value_end;
	int f_middle_is_zero, f_end_value_is_minus_square;
	int a, c, l, ll, ms_idx, field1, field2, sub_index, sub_sub_index;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index:";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[0] != 0 && v[0] != 1) {
		cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index: "
				"error in unrank_point" << endl;
		exit(1);
	}
	parabolic_point_properties(v, 1, n,
		f_start_with_one, value_middle, value_end, verbose_level);
	if (f_v) {
		cout << "f_start_with_one=" << f_start_with_one
				<< " value_middle=" << value_middle
				<< " value_end=" << value_end << endl;
	}
	if (value_middle == 0) {
		f_middle_is_zero = O->is_zero_vector(v + 1, 1, n - 3);
	}
	else {
		f_middle_is_zero = FALSE;
	}
	if (f_v) {
		cout << "f_middle_is_zero=" << f_middle_is_zero << endl;
	}
	f_end_value_is_minus_square = O->SN->f_is_minus_square[value_end];
	if (f_v) {
		cout << "f_end_value_is_minus_square="
				<< f_end_value_is_minus_square << endl;
	}

	if (f_end_value_is_minus_square) {
		type = 1;
		index = 0;
		l = (q - 1) / 2 * N1_mm1;
		if (!f_start_with_one) {
			ms_idx = O->SN->index_minus_square[value_end];
			if (ms_idx == -1) {
				cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index: "
						"ms_idx == -1" << endl;
			}
			c = F->negate(value_end);
			O->change_form_value(
					v + 1, 1, m - 1, F->inverse(c));
			sub_index = rank_N1(
					v + 1, 1, m - 1);
			index += ms_idx * N1_mm1 + sub_index;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
			}
			return;
		}
		index += l;
		l = (q - 1) * S_mm1;
		if (value_middle == 0) {
			if (f_middle_is_zero) {
				a = v[1 + 2 * (m - 1) + 0];
				field1 = a - 1;
				sub_index = 0;
				index += field1 * S_mm1 + sub_index;
				if (f_v) {
					cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
						"type = " << type << " index = " << index << endl;
				}
				return;
			}
			else {
				a = v[1 + 2 * (m - 1) + 0];
				//b = v[1 + 2 * (m - 1) + 1];
				field1 = a - 1;
				field2 = O->last_non_zero_entry(v + 1, 1, n - 3);
				O->scalar_multiply_vector(
						v + 1, 1, n - 3, F->inverse(field2));
				sub_sub_index = rank_Sbar(
						v + 1, 1, m - 1);
				field2--;
				//cout << "field1=" << field1 << " field2=" << field2
				//<< " sub_sub_index=" << sub_sub_index << endl;
				sub_index = field2 * Sbar_mm1 + sub_sub_index + 1;
				index += field1 * S_mm1 + sub_index;
				if (f_v) {
					cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
							"type = " << type << " index = " << index << endl;
				}
				return;
			}
		}
		index += l;
		l = ((q - 1) / 2 - 1) * (q - 1) * N1_mm1;
		ll = (q - 1) * N1_mm1;
		ms_idx = O->SN->index_minus_square_without[value_end];
		if (ms_idx == -1) {
			cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index: "
					"ms_idx == -1" << endl;
		}
		field1 = O->SN->minus_squares_without[ms_idx];
		c = F->negate(F->add(1, field1));
		O->change_form_value(
				v + 1, 1, m - 1, F->inverse(c));
		sub_sub_index = rank_N1(
				v + 1, 1, m - 1);
		a = v[1 + 2 * (m - 1) + 0];
		field2 = a - 1;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		index += ms_idx * ll + sub_index;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
		}
		return;
	}
	else if (value_end) {
		type = 2;
		l = (q - 1) / 2 * N1_mm1;
		index = 0;
		if (!f_start_with_one) {
			ms_idx = O->SN->index_minus_nonsquare[value_end];
			if (ms_idx == -1) {
				cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index: "
						"ms_idx == -1" << endl;
			}
			c = F->negate(value_end);
			O->change_form_value(
					v + 1, 1, m - 1, F->inverse(c));
			sub_index = rank_N1(
					v + 1, 1, m - 1);
			index += ms_idx * N1_mm1 + sub_index;
			if (f_v) {
				cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
						"type = " << type << " index = " << index << endl;
			}
			return;
		}
		index += l;
		l = (q - 1) / 2 * (q - 1) * N1_mm1;
		ll = (q - 1) * N1_mm1;
		ms_idx = O->SN->index_minus_nonsquare[value_end];
		if (ms_idx == -1) {
			cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index: "
					"ms_idx == -1" << endl;
		}
		//field1 = minus_nonsquares[ms_idx];
		//c = F->negate(F->add(1, field1));
		O->change_form_value(
				v + 1, 1, m - 1, F->inverse(value_middle));
		sub_sub_index = rank_N1(
				v + 1, 1, m - 1);
		a = v[1 + 2 * (m - 1) + 0];
		field2 = a - 1;
		//cout << "ms_idx=" << ms_idx << " field2=" << field2
		//<< " sub_sub_index=" << sub_sub_index << endl;
		sub_index = field2 * N1_mm1 + sub_sub_index;
		index += ms_idx * ll + sub_index;
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index "
					"type = " << type << " index = " << index << endl;
		}
		return;
	}
	cout << "hyperbolic_pair::parabolic_odd_point_to_type_and_index, "
			"unknown type, type = " << type << endl;
	exit(1);
}

//##############################################################################
// ranking / unranking neighbors of the favorite point:
//##############################################################################

void hyperbolic_pair::parabolic_neighbor51_odd_unrank(
		long int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor51_odd_unrank "
				"index=" << index << endl;
	}
	O->subspace->Hyperbolic_pair->parabolic_odd_type1_index_to_point(
			index, O->subspace->Hyperbolic_pair->v_tmp2, verbose_level);
	v[0] = O->subspace->Hyperbolic_pair->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		v[2 + i] = O->subspace->Hyperbolic_pair->v_tmp2[i];
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor51_odd_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor51_odd_rank(
		int *v, int verbose_level)
{
	int i;
	long int type, index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor51_odd_rank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[2]) {
		cout << "hyperbolic_pair::parabolic_neighbor51_odd_rank v[2]" << endl;
		exit(1);
	}
	O->subspace->Hyperbolic_pair->v_tmp2[0] = v[0];
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		O->subspace->Hyperbolic_pair->v_tmp2[i] = v[2 + i];
	}
	O->subspace->normalize_point(
			O->subspace->Hyperbolic_pair->v_tmp2, 1);
	if (f_v) {
		cout << "normalized and in subspace: ";
		Int_vec_print(cout,
				O->subspace->Hyperbolic_pair->v_tmp2,
				O->subspace->Quadratic_form->n);
		cout << endl;
	}
	O->subspace->Hyperbolic_pair->parabolic_odd_point_to_type_and_index(
			O->subspace->Hyperbolic_pair->v_tmp2, type, index, verbose_level);
	if (type != 1) {
		cout << "hyperbolic_pair::parabolic_neighbor51_odd_rank type != 1" << endl;
		exit(1);
	}
	return index;
}


void hyperbolic_pair::parabolic_neighbor52_odd_unrank(
		long int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_odd_unrank index=" << index << endl;
	}
	O->subspace->Hyperbolic_pair->parabolic_odd_type2_index_to_point(
			index, O->subspace->Hyperbolic_pair->v_tmp2, verbose_level);
	v[0] = O->subspace->Hyperbolic_pair->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		v[2 + i] = O->subspace->Hyperbolic_pair->v_tmp2[i];
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_odd_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor52_odd_rank(
		int *v, int verbose_level)
{
	long int i, type, index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_odd_rank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[2]) {
		cout << "hyperbolic_pair::parabolic_neighbor52_odd_rank v[2]" << endl;
		exit(1);
	}
	O->subspace->Hyperbolic_pair->v_tmp2[0] = v[0];
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		O->subspace->Hyperbolic_pair->v_tmp2[i] = v[2 + i];
	}
	O->subspace->normalize_point(
			O->subspace->Hyperbolic_pair->v_tmp2, 1);
	O->subspace->Hyperbolic_pair->parabolic_odd_point_to_type_and_index(
			O->subspace->Hyperbolic_pair->v_tmp2, type, index, verbose_level);
	if (type != 2) {
		cout << "hyperbolic_pair::parabolic_neighbor52_odd_rank "
				"type != 2" << endl;
		exit(1);
	}
	return index;
}

void hyperbolic_pair::parabolic_neighbor52_even_unrank(
		long int index, int *v, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_even_unrank index=" << index << endl;
	}
	O->subspace->Hyperbolic_pair->parabolic_even_type2_index_to_point(
			index, O->subspace->Hyperbolic_pair->v_tmp2);
	v[0] = O->subspace->Hyperbolic_pair->v_tmp2[0];
	v[1] = 0;
	v[2] = 0;
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		v[2 + i] = O->subspace->Hyperbolic_pair->v_tmp2[i];
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_even_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor52_even_rank(
		int *v, int verbose_level)
{
	long int i, type, index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor52_even_rank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[2]) {
		cout << "hyperbolic_pair::parabolic_neighbor52_even_rank v[2]" << endl;
		exit(1);
	}
	O->subspace->Hyperbolic_pair->v_tmp2[0] = v[0];
	for (i = 1; i < O->subspace->Quadratic_form->n; i++) {
		O->subspace->Hyperbolic_pair->v_tmp2[i] = v[2 + i];
	}
	O->subspace->normalize_point(
			O->subspace->Hyperbolic_pair->v_tmp2, 1);
	O->subspace->Hyperbolic_pair->parabolic_even_point_to_type_and_index(
			O->subspace->Hyperbolic_pair->v_tmp2, type, index, verbose_level);
	if (type != 2) {
		cout << "hyperbolic_pair::parabolic_neighbor52_even_rank "
				"type != 1" << endl;
		exit(1);
	}
	return index;
}

void hyperbolic_pair::parabolic_neighbor34_unrank(
		long int index, int *v, int verbose_level)
{
	long int len, sub_len, a, av, b, sub_index;
	long int sub_sub_index, multiplier;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor34_unrank "
				"index=" << index << endl;
	}
	len = S_mm2;
	if (index < len) {
		// case 1:
		if (f_v) {
			cout << "case 1 index=" << index << endl;
		}
		v[0] = 0;
		v[n - 2] = 1;
		v[n - 1] = 0;
		v[1] = 0;
		v[2] = F->negate(1);
		unrank_S(v + 3, 1, m - 2, index);
		goto finish;
	}
	index -= len;
	len = (q - 1) * N1_mm2;
	if (index < len) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 a=" << a
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 0;
		v[n - 2] = 1;
		v[n - 1] = 0;
		v[1] = a;
		v[2] = F->negate(1);
		unrank_N1(v + 3, 1, m - 2, sub_index);
		O->change_form_value(v + 3, 1, m - 2, a);
		goto finish;
	}
	index -= len;
	len = (q - 1) * N1_mm2;
	if (index < len) {
		// case 3:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 3 a=" << a
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 1;
		v[1] = 0;
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		O->change_form_value(v + 3, 1, m - 2, F->negate(1));
		goto finish;
	}
	index -= len;
	len = (q - 1) * S_mm2;
	if (index < len) {
		// case 4:
		a = index / S_mm2;
		sub_index = index % S_mm2;
		a++;
		if (f_v) {
			cout << "case 4 a=" << a
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 1;
		v[1] = F->inverse(a);
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_S(v + 3, 1, m - 2, sub_index);
		goto finish;
	}
	index -= len;
	len = (q - 1) * (q - 2) * N1_mm2;
	if (index < len) {
		// case 5:
		sub_len = (q - 2) * N1_mm2;
		a = index / sub_len;
		sub_index = index % sub_len;
		b = sub_index / N1_mm2;
		sub_sub_index = sub_index % N1_mm2;
		a++;
		av = F->inverse(a);
		b++;
		if (b >= av) {
			b++;
		}
		if (f_v) {
			cout << "case 5 a=" << a << " b=" << b
					<< " sub_sub_index=" << sub_sub_index << endl;
		}
		v[0] = 1;
		v[1] = b;
		v[2] = F->negate(a);
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_sub_index);
		multiplier = F->add(F->negate(1), F->mult(a, b));
		if (f_v) {
			cout << "case 5 multiplyer=" << multiplier << endl;
		}
		O->change_form_value(
				v + 3, 1, m - 2, multiplier);
		goto finish;
	}
	cout << "hyperbolic_pair::parabolic_neighbor34_unrank "
			"index illegal" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor34_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor34_rank(
		int *v, int verbose_level)
{
	int len1, len2, len3, len4, /*len5,*/ av;
	int index, sub_len, a, b, sub_index, sub_sub_index, multiplyer;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor34_rank " << endl;
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	O->normalize_point(v, 1);
	if (v[n - 1]) {
		cout << "hyperbolic_pair::parabolic_neighbor34_rank "
				"v[n - 1]" << endl;
		exit(1);
	}
	if (v[n - 2] == 0) {
		cout << "hyperbolic_pair::parabolic_neighbor34_rank "
				"v[n - 2] == 0" << endl;
		exit(1);
	}

	len1 = S_mm2;
	len2 = (q - 1) * N1_mm2;
	len3 = (q - 1) * N1_mm2;
	len4 = (q - 1) * S_mm2;
	//len5 = (q - 1) * (q - 2) * N1_mm2;

	if (v[0] == 0) {
		if (v[2] != F->negate(1)) {
			cout << "hyperbolic_pair::parabolic_neighbor34_rank "
					"v[2] != F->negate(1)" << endl;
			exit(1);
		}
		a = v[1];
		if (a == 0) {
			// case 1:
			index = rank_S(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 1 index=" << index << endl;
			}
			goto finish;
		}
		else {
			// case 2:
			O->change_form_value(v + 3, 1, m - 2, F->inverse(a));
			sub_index = rank_N1(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 2 a=" << a
						<< " sub_index=" << sub_index << endl;
			}
			index = (a - 1) * N1_mm2 + sub_index;
			index += len1;
			goto finish;
		}
	}
	else {
		if (v[0] != 1) {
			cout << "hyperbolic_pair::parabolic_neighbor34_rank "
					"v[1] != 1" << endl;
			exit(1);
		}
		a = v[n - 2];
		if (v[2] != F->negate(a)) {
			cout << "hyperbolic_pair::parabolic_neighbor34_rank "
					"v[2] != F->negate(a)" << endl;
			exit(1);
		}
		if (v[1] == 0) {
			// case 3:
			O->change_form_value(v + 3, 1, m - 2, F->negate(1));
			sub_index = rank_N1(v + 3, 1, m - 2);
			if (f_v) {
				cout << "case 3 a=" << a
						<< " sub_index=" << sub_index << endl;
			}
			index = (a - 1) * N1_mm2 + sub_index;
			index += len1;
			index += len2;
			goto finish;
		}
		else {
			av = F->inverse(a);
			if (v[1] == av) {
				// case 4:
				sub_index = rank_S(v + 3, 1, m - 2);
				if (f_v) {
					cout << "case 4 a=" << a
							<< " sub_index=" << sub_index << endl;
				}
				index = (a - 1) * S_mm2 + sub_index;
				index += len1;
				index += len2;
				index += len3;
				goto finish;
			}
			else {
				// case 5:
				sub_len = (q - 2) * N1_mm2;
				b = v[1];
				if (b == av) {
					cout << "hyperbolic_pair::parabolic_neighbor34_rank "
							"b = av" << endl;
					exit(1);
				}
				multiplyer = F->add(F->negate(1), F->mult(a, b));
				if (f_v) {
					cout << "case 5 multiplyer=" << multiplyer << endl;
				}
				O->change_form_value(
						v + 3, 1, m - 2, F->inverse(multiplyer));
				sub_sub_index = rank_N1(
						v + 3, 1, m - 2);
				if (f_v) {
					cout << "case 5 a=" << a << " b=" << b
							<< " sub_sub_index=" << sub_sub_index << endl;
				}
				if (b >= av) {
					b--;
				}
				b--;
				sub_index = b * N1_mm2 + sub_sub_index;
				index = (a - 1) * sub_len + sub_index;
				index += len1;
				index += len2;
				index += len3;
				index += len4;
				goto finish;
			}
		}
	}
	cout << "hyperbolic_pair::parabolic_neighbor34_rank "
			"illegal point" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor34_rank "
				"index = " << index << endl;
	}
	return index;
}


void hyperbolic_pair::parabolic_neighbor53_unrank(
		long int index, int *v, int verbose_level)
{
	long int a, sub_index;
	int f_v = (verbose_level >= 1);
	int len1, len2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor53_unrank "
				"index=" << index << endl;
	}
	len1 = (q - 1) * Sbar_mm2;
	len2 = (q - 1) * N1_mm2;
	if (index < len1) {
		// case 1:
		a = index / Sbar_mm2;
		sub_index = index % Sbar_mm2;
		a++;
		if (f_v) {
			cout << "case 1 index=" << index << " a=" << a
					<< " sub_index=" << sub_index << endl;
		}

		v[0] = 0;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = 0;
		v[n - 1] = a;
		unrank_Sbar(v + 3, 1, m - 2, sub_index);
		goto finish;
	}
	index -= len1;
	if (index < len2) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 index=" << index << " a=" << a
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = 0;
		v[n - 1] = a;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		O->change_form_value(v + 3, 1, m - 2, F->negate(1));
		goto finish;
	}
	cout << "hyperbolic_pair::parabolic_neighbor53_unrank "
			"index illegal" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor53_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor53_rank(
		int *v, int verbose_level)
{
	int len1; //, len2;
	int index, a, sub_index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor53_rank " << endl;
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	parabolic_normalize_point_wrt_subspace(v, 1);
	if (v[n - 2]) {
		cout << "hyperbolic_pair::parabolic_neighbor53_rank "
				"v[n - 2]" << endl;
		exit(1);
	}
	if (v[n - 1] == 0) {
		cout << "hyperbolic_pair::parabolic_neighbor53_rank "
				"v[n - 1] == 0" << endl;
		exit(1);
	}
	a = v[n - 1];

	len1 = (q - 1) * Sbar_mm2;
	//len2 = (q - 1) * N1_mm2;

	if (v[0] == 0) {
		// case 1
		sub_index = rank_Sbar(v + 3, 1, m - 2);
		index = (a - 1) * Sbar_mm2 + sub_index;
		goto finish;
	}
	else {
		// case 2
		O->change_form_value(v + 3, 1, m - 2, F->negate(1));
		sub_index = rank_N1(v + 3, 1, m - 2);
		index = len1 + (a - 1) * N1_mm2 + sub_index;
		goto finish;
	}

	cout << "hyperbolic_pair::parabolic_neighbor53_rank "
			"illegal point" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor53_rank "
				"index = " << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_neighbor54_unrank(
		long int index, int *v, int verbose_level)
{
	long int a, sub_index;
	int f_v = (verbose_level >= 1);
	long int len1, len2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor54_unrank "
				"index=" << index << endl;
	}
	len1 = (q - 1) * Sbar_mm2;
	len2 = (q - 1) * N1_mm2;
	if (index < len1) {
		// case 1:
		a = index / Sbar_mm2;
		sub_index = index % Sbar_mm2;
		a++;
		if (f_v) {
			cout << "case 1 index=" << index
					<< " a=" << a
					<< " sub_index=" << sub_index << endl;
		}

		v[0] = 0;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_Sbar(v + 3, 1, m - 2, sub_index);
		goto finish;
	}
	index -= len1;
	if (index < len2) {
		// case 2:
		a = index / N1_mm2;
		sub_index = index % N1_mm2;
		a++;
		if (f_v) {
			cout << "case 2 index=" << index
					<< " a=" << a
					<< " sub_index=" << sub_index << endl;
		}
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
		v[n - 2] = a;
		v[n - 1] = 0;
		unrank_N1(v + 3, 1, m - 2, sub_index);
		O->change_form_value(
				v + 3, 1, m - 2, F->negate(1));
		goto finish;
	}
	cout << "hyperbolic_pair::parabolic_neighbor54_unrank "
			"index illegal" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor54_unrank ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
}

long int hyperbolic_pair::parabolic_neighbor54_rank(
		int *v, int verbose_level)
{
	long int len1; //, len2;
	long int index, a, sub_index;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor54_rank " << endl;
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	parabolic_normalize_point_wrt_subspace(v, 1);
	if (f_v) {
		cout << "normalized wrt subspace " << endl;
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	if (v[n - 1]) {
		cout << "hyperbolic_pair::parabolic_neighbor54_rank "
				"v[n - 2]" << endl;
		exit(1);
	}
	if (v[n - 2] == 0) {
		cout << "hyperbolic_pair::parabolic_neighbor54_rank "
				"v[n - 1] == 0" << endl;
		exit(1);
	}
	a = v[n - 2];

	len1 = (q - 1) * Sbar_mm2;
	//len2 = (q - 1) * N1_mm2;

	if (v[0] == 0) {
		// case 1
		sub_index = rank_Sbar(v + 3, 1, m - 2);
		index = (a - 1) * Sbar_mm2 + sub_index;
		goto finish;
	}
	else {
		// case 2
		O->change_form_value(v + 3, 1, m - 2, F->negate(1));
		sub_index = rank_N1(v + 3, 1, m - 2);
		index = len1 + (a - 1) * N1_mm2 + sub_index;
		goto finish;
	}

	cout << "hyperbolic_pair::parabolic_neighbor54_rank "
			"illegal point" << endl;
	exit(1);

finish:
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_neighbor54_rank "
				"index = " << index << endl;
	}
	return index;
}


//##############################################################################
// ranking / unranking lines:
//##############################################################################

void hyperbolic_pair::parabolic_unrank_line(
		long int &p1, long int &p2,
		long int rk, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line "
				"rk=" << rk << endl;
	}
	if (m == 0) {
		cout << "hyperbolic_pair::parabolic_unrank_line "
				"Witt index zero, there is no line to unrank" << endl;
		exit(1);
	}
	if (rk < l1) {
		if (O->Quadratic_form->f_even) {
			parabolic_unrank_line_L1_even(
					p1, p2, rk, verbose_level);
		}
		else {
			parabolic_unrank_line_L1_odd(
					p1, p2, rk, verbose_level);
		}
		return;
	}
	rk -= l1;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line "
				"reducing rk to " << rk << " l2=" << l2 << endl;
	}
	if (rk < l2) {
		if (O->Quadratic_form->f_even) {
			parabolic_unrank_line_L2_even(
					p1, p2, rk, verbose_level);
		}
		else {
			parabolic_unrank_line_L2_odd(
					p1, p2, rk, verbose_level);
		}
		return;
	}
	rk -= l2;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line "
				"reducing rk to " << rk << " l3=" << l3 << endl;
	}
	if (rk < l3) {
		parabolic_unrank_line_L3(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l3;
	if (rk < l4) {
		parabolic_unrank_line_L4(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l4;
	if (rk < l5) {
		parabolic_unrank_line_L5(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l5;
	if (rk < l6) {
		parabolic_unrank_line_L6(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l6;
	if (rk < l7) {
		parabolic_unrank_line_L7(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l7;
	if (rk < l8) {
		parabolic_unrank_line_L8(
				p1, p2, rk, verbose_level);
		return;
	}
	rk -= l8;
	cout << "hyperbolic_pair::parabolic_unrank_line, "
			"rk too big" << endl;
	exit(1);
}

long int hyperbolic_pair::parabolic_rank_line(
		long int p1, long int p2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int p1_type, p2_type, p1_index, p2_index, type;
	long int cp1, cp2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	point_rk_to_type_and_index(p1,
			p1_type, p1_index, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line "
				"p1_type=" << p1_type
				<< " p1_index=" << p1_index << endl;
	}
	point_rk_to_type_and_index(p2,
			p2_type, p2_index, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line "
				"p2_type=" << p2_type
				<< " p2_index=" << p2_index << endl;
	}
	type = parabolic_line_type_given_point_types(
			p1, p2, p1_type, p2_type, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line "
				"line type = " << type << endl;
	}
	parabolic_canonical_points_of_line(type,
			p1, p2, cp1, cp2, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line "
				"cp1=" << cp1 << " cp2=" << cp2 << endl;
	}

	if (type == 1) {
		if (O->Quadratic_form->f_even) {
			return parabolic_rank_line_L1_even(
					cp1, cp2, verbose_level);
		}
		else {
			return parabolic_rank_line_L1_odd(
					cp1, cp2, verbose_level);
		}
	}
	else if (type == 2) {
		if (O->Quadratic_form->f_even) {
			return l1 +
					parabolic_rank_line_L2_even(
							cp1, cp2, verbose_level);
		}
		else {
			return l1 +
					parabolic_rank_line_L2_odd(
							cp1, cp2, verbose_level);
		}
	}
	else if (type == 3) {
		return l1 + l2 +
				parabolic_rank_line_L3(
						cp1, cp2, verbose_level);
	}
	else if (type == 4) {
		return l1 + l2 + l3 +
				parabolic_rank_line_L4(
						cp1, cp2, verbose_level);
	}
	else if (type == 5) {
		return l1 + l2 + l3 + l4 +
				parabolic_rank_line_L5(
						cp1, cp2, verbose_level);
	}
	else if (type == 6) {
		return l1 + l2 + l3 + l4 + l5 +
				parabolic_rank_line_L6(
						cp1, cp2, verbose_level);
	}
	else if (type == 7) {
		return l1 + l2 + l3 + l4 + l5 + l6 +
				parabolic_rank_line_L7(
						cp1, cp2, verbose_level);
	}
	else if (type == 8) {
		return l1 + l2 + l3 + l4 + l5 + l6 + l7 +
				parabolic_rank_line_L8(
						cp1, cp2, verbose_level);
	}
	else {
		cout << "hyperbolic_pair::parabolic_rank_line type nyi" << endl;
		exit(1);
	}
}

void hyperbolic_pair::parabolic_unrank_line_L1_even(
		long int &p1, long int &p2,
		long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, sub_idx;

	if (index >= l1) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_even "
				"index too large" << endl;
	}
	idx = index / (q - 1);
	sub_idx = index % (q - 1);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_even "
				"index=" << index << " idx=" << idx
				<< " sub_idx=" << sub_idx << endl;
	}
	p1 = type_and_index_to_point_rk(
			5, idx, verbose_level);
	if (f_vv) {
		cout << "p1=" << p1 << endl;
	}
	p2 = type_and_index_to_point_rk(
			1, sub_idx, verbose_level);
	if (f_vv) {
		cout << "p2=" << p2 << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_even "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L1_even(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, type, idx, sub_idx;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_even "
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
	point_rk_to_type_and_index(
			p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_even "
				"p1 must be in P5" << endl;
		exit(1);
	}
	point_rk_to_type_and_index(
			p2, type, sub_idx, verbose_level);
	if (type != 1) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_even "
				"p2 must be in P1" << endl;
		exit(1);
	}
	index = idx * (q - 1) + sub_idx;
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L1_odd(
		long int &p1, long int &p2,
		long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_odd "
				"index=" << index << " l1=" << l1
				<< " a51=" << a51 << endl;
	}
	if (index >= l1) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_odd "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a51;
	index2 = index % a51;

	rk1 = type_and_index_to_point_rk(
			5, 0, verbose_level);
	if (f_v) {
		cout << "rk1=" << rk1 << endl;
	}
	p1 = type_and_index_to_point_rk(
			5, idx, verbose_level);
	if (f_v) {
		cout << "p1=" << p1 << endl;
	}

	parabolic_neighbor51_odd_unrank(
			index2, v3, verbose_level);

	O->Orthogonal_group->Siegel_move_forward_by_index(
			rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L1_odd "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L1_odd(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, type, idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_odd "
				"p1=" << p1 << " p2=" << p2 << endl;
	}

	rk1 = type_and_index_to_point_rk(
			5, 0, verbose_level);

	point_rk_to_type_and_index(
			p1, type, idx, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_odd "
				"type=" << type << " idx=" << idx << endl;
	}
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_odd "
				"point 1 must be of type 5" << endl;
		exit(1);
	}
	unrank_point(
			v4, 1, p2, verbose_level - 1);
	O->Orthogonal_group->Siegel_move_backward_by_index(
			rk1, p1, v4, v3, verbose_level);

	index2 = parabolic_neighbor51_odd_rank(v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_odd "
				"idx=" << idx << " index2=" << index2 << endl;
	}

	index = idx * a51 + index2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L1_odd "
				"index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L2_even(
		long int &p1, long int &p2,
		long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_even "
				"index=" << index << endl;
	}
	if (index >= l2) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_even "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a52a;
	index2 = index % a52a;

	rk1 = type_and_index_to_point_rk(
			5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(
			5, idx, verbose_level);
	parabolic_neighbor52_even_unrank(
			index2, v3, FALSE);

	O->Orthogonal_group->Siegel_move_forward_by_index(
			rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);


	if (f_vv) {
		cout << "p2=" << p2 << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_even "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
	}
}

void hyperbolic_pair::parabolic_unrank_line_L2_odd(
		long int &p1, long int &p2,
		long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_odd "
				"index=" << index << endl;
	}
	if (index >= l2) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_odd "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a52a;
	index2 = index % a52a;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_odd "
				"idx=" << idx << " index2=" << index2 << endl;
	}

	rk1 = type_and_index_to_point_rk(
			5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(
			5, idx, verbose_level);
	parabolic_neighbor52_odd_unrank(
			index2, v3, FALSE);

	O->Orthogonal_group->Siegel_move_forward_by_index(
			rk1, p1, v3, v4, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_forward_by_index";
		Int_vec_print(cout, v4, n);
		cout << endl;
	}
	p2 = rank_point(v4, 1, verbose_level - 1);


	if (f_vv) {
		cout << "p2=" << p2 << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L2_odd "
				"index=" << index << " p1=" << p1
				<< " p2=" << p2 << endl;
	}
}

int hyperbolic_pair::parabolic_rank_line_L2_even(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, type, idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_even "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);

	point_rk_to_type_and_index(
			p1, type, idx, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_even "
				"type=" << type << " idx=" << idx << endl;
	}
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_even "
				"point 1 must be of type 5" << endl;
		exit(1);
	}
	unrank_point(v4, 1, p2, verbose_level - 1);
	O->Orthogonal_group->Siegel_move_backward_by_index(
			rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		Int_vec_print(cout, v3, n);
		cout << endl;
	}
	index2 = parabolic_neighbor52_even_rank(v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_even "
				"idx=" << idx
				<< " index2=" << index2 << endl;
	}

	index = idx * a52a + index2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_even "
				"index=" << index << endl;
	}
	return index;
}

long int hyperbolic_pair::parabolic_rank_line_L2_odd(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, type, idx, index2, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_odd "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);

	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_odd "
				"type=" << type
				<< " idx=" << idx << endl;
	}
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_odd "
				"point 1 must be of type 5" << endl;
		exit(1);
	}
	unrank_point(v4, 1, p2, verbose_level - 1);
	O->Orthogonal_group->Siegel_move_backward_by_index(
			rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		Int_vec_print(cout, v3, n);
		cout << endl;
	}
	index2 = parabolic_neighbor52_odd_rank(v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_odd "
				"idx=" << idx
				<< " index2=" << index2 << endl;
	}

	index = idx * a52a + index2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L2_odd "
				"index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L3(
		long int &p1, long int &p2,
		long int index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, index2, idx2, field, rk1, rk2, a, b, c, multiplier, i;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L3 "
				"index=" << index << endl;
	}
	if (index >= l3) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L3 "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a32b;
	index2 = index % a32b;
	idx2 = idx / (q - 1);
	field = idx % (q - 1);
	field++;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L3 idx=" << idx
				<< " index2=" << index2 << " idx2=" << idx2
				<< " field=" << field << endl;
	}

	rk1 = type_and_index_to_point_rk(
			3, 0, verbose_level);
	rk2 = type_and_index_to_point_rk(
			5, idx2, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L3 rk1=" << rk1
				<< " rk2=" << rk2 << " idx2=" << idx2
				<< " field=" << field << endl;
	}
	unrank_point(v1, 1, rk1, verbose_level - 1);
	unrank_point(v2, 1, rk2, verbose_level - 1);
	v2[n - 1] = 1;


	if (f_v) {
		Int_vec_print(cout, v1, n); cout << endl;
		Int_vec_print(cout, v2, n); cout << endl;
	}

	parabolic_neighbor34_unrank(
			index2, v3, verbose_level);

	O->Orthogonal_group->Siegel_move_forward(
			v1, v2, v3, v4, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_forward" << endl;
		Int_vec_print(cout, v3, n); cout << endl;
		Int_vec_print(cout, v4, n); cout << endl;
	}
	a = O->subspace->Quadratic_form->evaluate_bilinear_form(
			v1, v3, 1);
	b = O->subspace->Quadratic_form->evaluate_bilinear_form(
			v2, v4, 1);
	if (f_v) {
		cout << "a=" << a << " b=" << b << endl;
	}
	if (a != b) {
		if (a == 0) {
			cout << "a != b but a = 0" << endl;
			exit(1);
		}
		if (b == 0) {
			cout << "a != b but b = 0" << endl;
			exit(1);
		}
		multiplier = F->mult(a, F->inverse(b));
		if (f_v) {
			cout << "multiplier=" << multiplier << endl;
		}
		for (i = 0; i < n - 2; i++) {
			v4[i] = F->mult(v4[i], multiplier);
		}
		if (f_v) {
			cout << "after scaling" << endl;
			Int_vec_print(cout, v4, n); cout << endl;
		}
		c = O->subspace->Quadratic_form->evaluate_bilinear_form(
				v2, v4, 1);
		if (f_v) {
			cout << "c=" << c << endl;
		}
		if (c != a) {
			cout << "c != a" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "now changing the last components:" << endl;
	}

	v2[n - 2] = 0;
	v2[n - 1] = field;
	O->normalize_point(v2, 1);
	p1 = rank_point(v2, 1, verbose_level - 1);
	v4[n - 2] = F->mult(v4[n - 2], F->inverse(field));
	p2 = rank_point(v4, 1, verbose_level - 1);



	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L3 "
				"index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L3(
		long int p1, long int p2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, idx, index2, idx2, field;
	long int rk1, rk2, type, a, b, c, i, multiplier;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L3 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}


	rk1 = type_and_index_to_point_rk(3, 0, verbose_level);

	unrank_point(v1, 1, rk1, verbose_level - 1);
	unrank_point(v2, 1, p1, verbose_level - 1);
	if (f_v) {
		Int_vec_print(cout, v1, n);
		cout << endl;
		Int_vec_print(cout, v2, n);
		cout << endl;
	}

	parabolic_normalize_point_wrt_subspace(v2, 1);
	if (f_v) {
		cout << "after parabolic_normalize_point_wrt_subspace ";
		Int_vec_print(cout, v2, n);
		cout << endl;
	}
	field = v2[n - 1];
	if (f_v) {
		cout << "field=" << field << endl;
	}
	v2[n - 1] = 0;
	rk2 = rank_point(v2, 1, verbose_level - 1);
	parabolic_point_rk_to_type_and_index(rk2,
			type, idx2, verbose_level);
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L3 "
				"rk1=" << rk1 << " rk2=" << rk2
				<< " idx2=" << idx2 << " field=" << field << endl;
	}
	if (type != 5) {
		cout << "parabolic_rank_line_L3  type != 5" << endl;
		exit(1);
	}
	v2[n - 1] = 1;


	unrank_point(v4, 1, p2, verbose_level - 1);
	v4[n - 2] = F->mult(v4[n - 2], field);

	idx = idx2 * (q - 1) + (field - 1);

	O->Orthogonal_group->Siegel_move_backward(
			v1, v2, v4, v3, verbose_level);
	if (f_v) {
		cout << "after Siegel_move_backward" << endl;
		Int_vec_print(cout, v3, n); cout << endl;
		Int_vec_print(cout, v4, n); cout << endl;
	}
	a = O->subspace->Quadratic_form->evaluate_bilinear_form(v1, v3, 1);
	b = O->subspace->Quadratic_form->evaluate_bilinear_form(v2, v4, 1);
	if (f_v) {
		cout << "a=" << a << " b=" << b << endl;
	}
	if (a != b) {
		if (a == 0) {
			cout << "a != b but a = 0" << endl;
			exit(1);
		}
		if (b == 0) {
			cout << "a != b but b = 0" << endl;
			exit(1);
		}
		multiplier = F->mult(b, F->inverse(a));
		if (f_v) {
			cout << "multiplier=" << multiplier << endl;
		}
		for (i = 0; i < n - 2; i++) {
			v3[i] = F->mult(v3[i], multiplier);
		}
		if (f_v) {
			cout << "after scaling" << endl;
			Int_vec_print(cout, v3, n); cout << endl;
		}
		c = O->subspace->Quadratic_form->evaluate_bilinear_form(v1, v3, 1);
		if (f_v) {
			cout << "c=" << c << endl;
		}
		if (c != b) {
			cout << "c != a" << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "after scaling" << endl;
		Int_vec_print(cout, v3, n); cout << endl;
		Int_vec_print(cout, v4, n); cout << endl;
	}

	index2 = parabolic_neighbor34_rank(v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L3 idx=" << idx
				<< " index2=" << index2 << " idx2=" << idx2
				<< " field=" << field << endl;
	}

	index = idx * a32b + index2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L3 "
				"index=" << index << endl;
	}

	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L4(
		long int &p1, long int &p2,
		long int index, int verbose_level)
// from P5 to P3
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, neighbor_idx, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L4 "
				"index=" << index << endl;
	}
	if (index >= l4) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L4 "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a53;
	neighbor_idx = index % a53;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L4 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
	}

	rk1 = type_and_index_to_point_rk(
			5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(
			5, idx, verbose_level);
	parabolic_neighbor53_unrank(
			neighbor_idx, v3, verbose_level);

	O->Orthogonal_group->Siegel_move_forward_by_index(
			rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);

	if (f_v) {
		unrank_point(v5, 1, p1, verbose_level - 1);
		cout << "p1=" << p1 << " ";
		Int_vec_print(cout, v5, n);
		cout << endl;
		cout << "p2=" << p2 << " ";
		Int_vec_print(cout, v4, n);
		cout << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L4 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L4(
		long int p1, long int p2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, idx, neighbor_idx, rk1, type;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L4 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);

	point_rk_to_type_and_index(
			p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L4 "
				"type != 5" << endl;
		exit(1);
	}

	unrank_point(v4, 1, p2, verbose_level - 1);
	O->Orthogonal_group->Siegel_move_backward_by_index(
			rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		Int_vec_print(cout, v3, n);
		cout << endl;
	}
	neighbor_idx = parabolic_neighbor53_rank(
			v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L4 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
	}

	index = idx * a53 + neighbor_idx;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L4 index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L5(
		long int &p1, long int &p2,
		long int index, int verbose_level)
// from P5 to P4
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int idx, neighbor_idx, rk1;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L5 "
				"index=" << index << endl;
	}
	if (index >= l5) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L5 "
				"index too large" << endl;
		exit(1);
	}
	idx = index / a54;
	neighbor_idx = index % a54;
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L5 "
				"idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
	}

	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);
	p1 = type_and_index_to_point_rk(5, idx, verbose_level);
	parabolic_neighbor54_unrank(neighbor_idx, v3, verbose_level);

	O->Orthogonal_group->Siegel_move_forward_by_index(
			rk1, p1, v3, v4, verbose_level);
	p2 = rank_point(v4, 1, verbose_level - 1);

	if (f_v) {
		unrank_point(v5, 1, p1, verbose_level - 1);
		cout << "p1=" << p1 << " ";
		Int_vec_print(cout, v5, n);
		cout << endl;
		cout << "p2=" << p2 << " ";
		Int_vec_print(cout, v4, n);
		cout << endl;
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L5 "
				"index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L5(
		long int p1, long int p2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int index, idx, neighbor_idx, rk1, type;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L5 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	rk1 = type_and_index_to_point_rk(5, 0, verbose_level);

	point_rk_to_type_and_index(p1, type, idx, verbose_level);
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L5 "
				"type != 5" << endl;
		exit(1);
	}

	unrank_point(v4, 1, p2, verbose_level - 1);
	O->Orthogonal_group->Siegel_move_backward_by_index(
			rk1, p1, v4, v3, verbose_level);

	if (f_v) {
		cout << "after Siegel_move_backward_by_index";
		Int_vec_print(cout, v3, n);
		cout << endl;
	}
	neighbor_idx = parabolic_neighbor54_rank(v3, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L5 idx=" << idx
				<< " neighbor_idx=" << neighbor_idx << endl;
	}

	index = idx * a54 + neighbor_idx;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L5 index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L6(
		long int &p1, long int &p2,
		long int index, int verbose_level)
// within P5
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int pt1, pt2;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L6 "
				"index=" << index << endl;
	}
	if (index >= l6) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L6 "
				"index too large" << endl;
		exit(1);
	}
	O->subspace->Hyperbolic_pair->parabolic_unrank_line(
			pt1, pt2, index, verbose_level);
	O->subspace->Hyperbolic_pair->unrank_point(
			v1, 1, pt1, verbose_level - 1);
	O->subspace->Hyperbolic_pair->unrank_point(
			v2, 1, pt2, verbose_level - 1);
	v1[n - 2] = 0;
	v1[n - 1] = 0;
	v2[n - 2] = 0;
	v2[n - 1] = 0;
	p1 = rank_point(v1, 1, verbose_level - 1);
	p2 = rank_point(v2, 1, verbose_level - 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L6 "
				"index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L6(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int pt1, pt2;
	long int index;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L6 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	unrank_point(v1, 1, p1, verbose_level - 1);
	unrank_point(v2, 1, p2, verbose_level - 1);
	if (v1[n - 2] || v1[n - 1] || v2[n - 2] || v2[n - 1]) {
		cout << "hyperbolic_pair::parabolic_rank_line_L6 "
				"points not in subspace" << endl;
		exit(1);
	}
	pt1 = O->subspace->Hyperbolic_pair->rank_point(
			v1, 1, verbose_level - 1);
	pt2 = O->subspace->Hyperbolic_pair->rank_point(
			v2, 1, verbose_level - 1);
	index = O->subspace->Hyperbolic_pair->parabolic_rank_line(
			pt1, pt2, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L6 "
				"index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L7(
		long int &p1, long int &p2,
		long int index, int verbose_level)
// from P6 = {Q}  to P5 via P3
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L7 "
				"index=" << index << endl;
	}
	if (index >= l7) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L7 "
				"index too large" << endl;
		exit(1);
	}
	p1 = pt_Q;
	p2 = type_and_index_to_point_rk(5, index, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L7 "
				"index=" << index << " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L7(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int type, index;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L7 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	if (p1 != pt_Q) {
		cout << "hyperbolic_pair::parabolic_rank_line_L7 "
				"p1 != pt_Q" << endl;
		exit(1);
	}
	point_rk_to_type_and_index(
			p2, type, index, verbose_level);
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L7 "
				"type != 5" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L7 "
				"index=" << index << endl;
	}
	return index;
}

void hyperbolic_pair::parabolic_unrank_line_L8(
		long int &p1, long int &p2,
		long int index, int verbose_level)
// from P7 = {P}  to P5 via P4
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L8 "
				"index=" << index << endl;
	}
	if (index >= l8) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L8 "
				"index too large" << endl;
		exit(1);
	}
	p1 = pt_P;
	p2 = type_and_index_to_point_rk(5, index, verbose_level);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_unrank_line_L8 index=" << index
				<< " p1=" << p1 << " p2=" << p2 << endl;
	}
}

long int hyperbolic_pair::parabolic_rank_line_L8(
		long int p1, long int p2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	long int type, index;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L8 "
				"p1=" << p1 << " p2=" << p2 << endl;
	}
	if (p1 != pt_P) {
		cout << "hyperbolic_pair::parabolic_rank_line_L8 "
				"p1 != pt_P" << endl;
		exit(1);
	}
	point_rk_to_type_and_index(
			p2, type, index, verbose_level);
	if (type != 5) {
		cout << "hyperbolic_pair::parabolic_rank_line_L8 "
				"type != 5" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_rank_line_L8 "
				"index=" << index << endl;
	}
	return index;
}

long int hyperbolic_pair::parabolic_line_type_given_point_types(
		long int pt1, long int pt2,
		long int pt1_type, long int pt2_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "hyperbolic_pair::parabolic_line_type_given_point_types "
				"pt1=" << pt1 << " pt2=" << pt2 << endl;
	}
	if (pt1_type > pt2_type) {
		return parabolic_line_type_given_point_types(
				pt2, pt1, pt2_type, pt1_type, verbose_level);
	}

	// from now on, we assume pt1_type <= pt2_type

	if (pt1_type == 1) {
		if (O->Quadratic_form->f_even) {
			return 1;
		}
		else {
			if (pt2_type == 1) {
				return parabolic_decide_P11_odd(pt1, pt2);
			}
			else if (pt2_type == 2) {
				return 3;
			}
			else if (pt2_type == 3) {
				return 3;
			}
			else if (pt2_type == 4) {
				return 3;
			}
			else if (pt2_type == 5) {
				return 1;
			}
		}
	}
	else if (pt1_type == 2) {
		if (O->Quadratic_form->f_even) {
			if (pt2_type == 2) {
				return parabolic_decide_P22_even(pt1, pt2);
			}
			else if (pt2_type == 3) {
				return 3;
			}
			else if (pt2_type == 4) {
				return 3;
			}
			else if (pt2_type == 5) {
				return parabolic_decide_P22_even(pt1, pt2);
			}
		}
		else {
			if (pt2_type == 2) {
				return parabolic_decide_P22_odd(pt1, pt2);
			}
			else if (pt2_type == 3) {
				return 3;
			}
			else if (pt2_type == 4) {
				return 3;
			}
			else if (pt2_type == 5) {
				return 2;
			}
		}
	}
	else if (pt1_type == 3) {
		if (O->Quadratic_form->f_even) {
			if (pt2_type == 3) {
				return parabolic_decide_P33(pt1, pt2);
			}
			else if (pt2_type == 4) {
				return 3;
			}
			else if (pt2_type == 5) {
				return parabolic_decide_P35(pt1, pt2);
			}
			else if (pt2_type == 6) {
				return 7;
			}
		}
		else {
			if (pt2_type == 3) {
				return parabolic_decide_P33(pt1, pt2);
			}
			else if (pt2_type == 4) {
				return 3;
			}
			else if (pt2_type == 5) {
				return parabolic_decide_P35(pt1, pt2);
			}
			else if (pt2_type == 6) {
				return 7;
			}
		}
	}
	else if (pt1_type == 4) {
		if (O->Quadratic_form->f_even) {
			if (pt2_type == 4) {
				return parabolic_decide_P44(pt1, pt2);
			}
			else if (pt2_type == 5) {
				return parabolic_decide_P45(pt1, pt2);
			}
			else if (pt2_type == 7) {
				return 8;
			}
		}
		else {
			if (pt2_type == 4) {
				return parabolic_decide_P44(pt1, pt2);
			}
			else if (pt2_type == 5) {
				return parabolic_decide_P45(pt1, pt2);
			}
			else if (pt2_type == 7) {
				return 8;
			}
		}
	}
	else if (pt1_type == 5) {
		if (pt2_type == 5) {
			return 6;
		}
		else if (pt2_type == 6) {
			return 7;
		}
		else if (pt2_type == 7) {
			return 8;
		}
	}
	cout << "hyperbolic_pair::parabolic_line_type_given_point_types "
			"illegal combination" << endl;
	cout << "pt1_type = " << pt1_type << endl;
	cout << "pt2_type = " << pt2_type << endl;
	exit(1);
}

int hyperbolic_pair::parabolic_decide_P11_odd(
		long int pt1, long int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P11_odd" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;

	if (O->is_ending_dependent(v1, v2)) {
		return 1;
	}
	else {
		return 3;
	}
}

int hyperbolic_pair::parabolic_decide_P22_even(
		long int pt1, long int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P22_even" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;


	if (O->is_ending_dependent(v1, v2)) {
		//cout << "ending is dependent, i.e. 1 or 2" << endl;
		if (parabolic_is_middle_dependent(v1, v2)) {
			return 1;
		}
		else {
			return 2;
		}
	}
	else {
		//cout << "ending is not dependent, hence 3" << endl;
		return 3;
	}
}

int hyperbolic_pair::parabolic_decide_P22_odd(
		long int pt1, long int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	if (O->is_ending_dependent(v1, v2)) {
		return 2;
	}
	else {
		return 3;
	}
}

int hyperbolic_pair::parabolic_decide_P33(
		long int pt1, long int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	//cout << "parabolic_decide_P33" << endl;
	if (O->is_ending_dependent(v1, v2)) {
		//cout << "ending is dependent" << endl;
		if (O->triple_is_collinear(pt1, pt2, pt_Q)) {
			return 7;
		}
		else {
			return 4;
		}
	}
	else {
		cout << "hyperbolic_pair::parabolic_decide_P33 "
				"ending is not dependent" << endl;
		exit(1);
	}
}

int hyperbolic_pair::parabolic_decide_P35(
		long int pt1, long int pt2)
{
	//cout << "parabolic_decide_P35 pt1 = " << pt1
	//<< " pt2=" << pt2 << endl;
	//unrank_point(v1, 1, pt1, verbose_level - 1);
	//unrank_point(v2, 1, pt2, verbose_level - 1);
	if (O->triple_is_collinear(pt1, pt2, pt_Q)) {
		return 7;
	}
	else {
		return 4;
	}
}

int hyperbolic_pair::parabolic_decide_P45(
		long int pt1, long int pt2)
{
	//unrank_point(v1, 1, pt1, verbose_level - 1);
	//unrank_point(v2, 1, pt2, verbose_level - 1);
	if (O->triple_is_collinear(pt1, pt2, pt_P)) {
		return 8;
	}
	else {
		return 5;
	}
}

int hyperbolic_pair::parabolic_decide_P44(
		long int pt1, long int pt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	if (O->is_ending_dependent(v1, v2)) {
		if (O->triple_is_collinear(pt1, pt2, pt_P)) {
			return 8;
		}
		else {
			return 5;
		}
	}
	else {
		cout << "hyperbolic_pair::parabolic_decide_P44 "
				"ending is not dependent" << endl;
		exit(1);
	}
}

void hyperbolic_pair::find_root_parabolic_xyz(
		long int rk2, int *x, int *y, int *z, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	int i;

	for (i = 0; i < n; i++) {
		x[i] = 0;
		z[i] = 0;
	}
	x[1] = 1;

	if (f_v) {
		cout << "hyperbolic_pair::find_root_parabolic_xyz rk2=" << rk2 << endl;
	}
	unrank_point(y, 1, rk2, verbose_level - 1);
	if (f_v) {
		Int_vec_print(cout, y, n);
		cout << endl;
	}
	if (y[1]) {
		z[2] = 1;
		return;
	}
	if (y[2] && y[0] == 0) {
		z[0] = 1;
		z[1] = 1;
		z[2] = F->negate(1);
		return;
	}
	if (n == 3) {
		cout << "hyperbolic_pair::find_root_parabolic_xyz n == 3, "
				"we should not be in this case" << endl;
		exit(1);
	}
	// now y[2] = 0 or y = (*0*..) and
	// m > 1 and y_i \neq 0 for some i \ge 3
	for (i = 3; i < n; i++) {
		if (y[i]) {
			if (EVEN(i)) {
				z[2] = 1;
				z[i - 1] = 1;
				return;
			}
			else {
				z[2] = 1;
				z[i + 1] = 1;
				return;
			}
		}
	}
	cout << "hyperbolic_pair::find_root_parabolic_xyz error" << endl;
	exit(1);
}

long int hyperbolic_pair::find_root_parabolic(
		long int rk2, int verbose_level)
// m = Witt index
{
	int f_v = (verbose_level >= 1);
	long int root, u, v;

	if (f_v) {
		cout << "hyperbolic_pair::find_root_parabolic rk2=" << rk2 << endl;
	}
	if (rk2 == 0) {
		cout << "hyperbolic_pair::find_root_parabolic: rk2 must not be 0" << endl;
		exit(1);
	}
#if 0
	if (m == 1) {
		cout << "find_root_parabolic: m must not be 1" << endl;
		exit(1);
	}
#endif
	find_root_parabolic_xyz(rk2,
			O->Orthogonal_group->find_root_x,
			O->Orthogonal_group->find_root_y,
			O->Orthogonal_group->find_root_z, verbose_level);
	if (f_v) {
		cout << "found root: ";
		Int_vec_print(cout, O->Orthogonal_group->find_root_x, n);
		Int_vec_print(cout, O->Orthogonal_group->find_root_y, n);
		Int_vec_print(cout, O->Orthogonal_group->find_root_z, n);
		cout << endl;
	}
	u = O->Quadratic_form->evaluate_parabolic_bilinear_form(
			O->Orthogonal_group->find_root_z,
			O->Orthogonal_group->find_root_x, 1, m);
	if (u == 0) {
		cout << "hyperbolic_pair::find_root_parabolic u=" << u << endl;
		exit(1);
	}
	v = O->Quadratic_form->evaluate_parabolic_bilinear_form(
			O->Orthogonal_group->find_root_z,
			O->Orthogonal_group->find_root_y, 1, m);
	if (v == 0) {
		cout << "hyperbolic_pair::find_root_parabolic v=" << v << endl;
		exit(1);
	}
	root = rank_point(O->Orthogonal_group->find_root_z, 1, verbose_level - 1);
	if (f_v) {
		cout << "hyperbolic_pair::find_root_parabolic root=" << root << endl;
	}
	return root;
}

void hyperbolic_pair::parabolic_canonical_points_of_line(
		int line_type, long int pt1, long int pt2,
		long int &cpt1, long int &cpt2, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_canonical_points_of_line "
				"line_type=" << line_type
				<< " pt1=" << pt1 << " pt2=" << pt2 << endl;
	}
	if (line_type == 1) {
		if (O->Quadratic_form->f_even) {
			parabolic_canonical_points_L1_even(pt1, pt2, cpt1, cpt2);
		}
		else {
			parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
		}
	}
	else if (line_type == 2) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
	}
	else if (line_type == 3) {
		parabolic_canonical_points_L3(pt1, pt2, cpt1, cpt2);
	}
	else if (line_type == 4) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
	}
	else if (line_type == 5) {
		parabolic_canonical_points_separate_P5(pt1, pt2, cpt1, cpt2);
	}
	else if (line_type == 6) {
		cpt1 = pt1;
		cpt2 = pt2;
	}
	else if (line_type == 7) {
		parabolic_canonical_points_L7(pt1, pt2, cpt1, cpt2);
	}
	else if (line_type == 8) {
		parabolic_canonical_points_L8(pt1, pt2, cpt1, cpt2);
	}
	if (f_v) {
		cout << "hyperbolic_pair::parabolic_canonical_points_of_line "
				"of type " << line_type << endl;
		cout << "pt1=" << pt1 << " pt2=" << pt2 << endl;
		cout << "cpt1=" << cpt1 << " cpt2=" << cpt2 << endl;
	}
}

void hyperbolic_pair::parabolic_canonical_points_L1_even(
		long int pt1, long int pt2,
		long int &cpt1, long int &cpt2)
{
	int verbose_level = 0;
	int i;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);

	//cout << "parabolic_canonical_points_L1_even" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;

	O->Gauss_step(v2, v1, n, n - 1);


	//cout << "after Gauss_step n - 1" << endl;
	//int_vec_print(cout, v1, n); cout << endl;
	//int_vec_print(cout, v2, n); cout << endl;

	if (!O->is_zero_vector(v1 + n - 2, 1, 2)) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L1_even ending "
				"of v1 is not zero" << endl;
		exit(1);
	}
	for (i = 1; i < n - 2; i++) {
		if (v2[i]) {
			O->Gauss_step(v1, v2, n, i);
			//cout << "after Gauss_step " << i << endl;
			//int_vec_print(cout, v1, n); cout << endl;
			//int_vec_print(cout, v2, n); cout << endl;

			if (!O->is_zero_vector(v2 + 1, 1, n - 3)) {
				cout << "hyperbolic_pair::parabolic_canonical_points_L1_even "
						"not zero" << endl;
				exit(1);
			}
			break;
		}
	}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void hyperbolic_pair::parabolic_canonical_points_separate_P5(
		long int pt1, long int pt2,
		long int &cpt1, long int &cpt2)
{
	int verbose_level = 0;
	int i;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
#if 0
	cout << "hyperbolic_pair::parabolic_canonical_points_separate_P5" << endl;
	cout << "v1=";
	int_vec_print(cout, v1, n);
	cout << "v2=";
	int_vec_print(cout, v2, n);
	cout << endl;
#endif
	for (i = n - 2; i < n; i++)
		if (v1[i]) {
			break;
		}
	if (i < n) {
		O->Gauss_step(v2, v1, n, i);
	}
#if 0
	cout << "after Gauss_step" << endl;
	cout << "v1=";
	int_vec_print(cout, v1, n);
	cout << "v2=";
	int_vec_print(cout, v2, n);
	cout << endl;
#endif
	if (!O->is_zero_vector(v1 + n - 2, 1, 2)) {
		cout << "hyperbolic_pair::parabolic_canonical_points_separate_P5 "
				"ending of v1 is not zero" << endl;
		cout << "v1=";
		Int_vec_print(cout, v1, n);
		cout << endl;
		exit(1);
	}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void hyperbolic_pair::parabolic_canonical_points_L3(
		long int pt1, long int pt2,
		long int &cpt1, long int &cpt2)
{
	int verbose_level = 0;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	O->Gauss_step(v2, v1, n, n - 2);
	if (v1[n - 2]) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L3 v1[n - 2]" << endl;
		exit(1);
	}
	O->Gauss_step(v1, v2, n, n - 1);
	if (v2[n - 1]) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L3 v2[n - 1]" << endl;
		exit(1);
	}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	return;
}

void hyperbolic_pair::parabolic_canonical_points_L7(
		long int pt1, long int pt2,
		long int &cpt1, long int &cpt2)
{
	int verbose_level = 0;
	int i;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	O->Gauss_step(v1, v2, n, n - 1);
	if (!O->is_zero_vector(v2 + n - 2, 1, 2)) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L7 "
				"ending of v2 is not zero" << endl;
		exit(1);
	}
	// now v2 is a point in P5

	for (i = 0; i < n - 2; i++) {
		if (v1[i]) {
			O->Gauss_step(v2, v1, n, i);
			if (!O->is_zero_vector(v1, 1, n - 2)) {
				cout << "hyperbolic_pair::parabolic_canonical_points_L7 "
						"not zero" << endl;
				exit(1);
			}
			break;
		}
	}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	if (cpt1 != pt_Q) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L7 "
				"cpt1 != pt_Q" << endl;
		exit(1);
	}
	return;
}

void hyperbolic_pair::parabolic_canonical_points_L8(
		long int pt1, long int pt2,
		long int &cpt1, long int &cpt2)
{
	int verbose_level = 0;
	int i;

	unrank_point(v1, 1, pt1, verbose_level - 1);
	unrank_point(v2, 1, pt2, verbose_level - 1);
	O->Gauss_step(v1, v2, n, n - 2);
	if (!O->is_zero_vector(v2 + n - 2, 1, 2)) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L8 "
				"ending of v2 is not zero" << endl;
		exit(1);
	}
	// now v2 is a point in P5

	for (i = 0; i < n - 2; i++) {
		if (v1[i]) {
			O->Gauss_step(v2, v1, n, i);
			if (!O->is_zero_vector(v1, 1, n - 2)) {
				cout << "hyperbolic_pair::parabolic_canonical_points_L8 "
						"not zero" << endl;
				exit(1);
			}
			break;
		}
	}
	cpt1 = rank_point(v1, 1, verbose_level - 1);
	cpt2 = rank_point(v2, 1, verbose_level - 1);
	if (cpt1 != pt_P) {
		cout << "hyperbolic_pair::parabolic_canonical_points_L8 "
				"cpt1 != pt_P" << endl;
		exit(1);
	}
	return;
}



void hyperbolic_pair::parabolic_point_normalize(
		int *v, int stride, int n)
{
	if (v[0]) {
		if (v[0] != 1) {
			F->Projective_space_basic->PG_element_normalize_from_front(
					v, stride, n);
		}
	}
	else {
		F->Projective_space_basic->PG_element_normalize(
				v, stride, n);
	}
}

void hyperbolic_pair::parabolic_normalize_point_wrt_subspace(
		int *v, int stride)
{
	int i, a, av;

	if (v[0]) {
		F->Projective_space_basic->PG_element_normalize_from_front(
				v, stride, n);
		return;
	}
	for (i = n - 3; i >= 0; i--) {
		if (v[i * stride]) {
			break;
		}
	}
	if (i < 0) {
		cout << "hyperbolic_pair::parabolic_normalize_point_wrt_subspace i < 0" << endl;
		exit(1);
	}
	a = v[i * stride];
	//cout << "parabolic_normalize_point_wrt_subspace "
	// "a=" << a << " in position " << i << endl;
	av = F->inverse(a);
	for (i = 0; i < n; i++) {
		v[i * stride] = F->mult(av, v[i * stride]);
	}
}

void hyperbolic_pair::parabolic_point_properties(
		int *v, int stride, int n,
	int &f_start_with_one, int &middle_value, int &end_value,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int m;

	if (f_v) {
		cout << "hyperbolic_pair::parabolic_point_properties ";
		Int_vec_print(cout, v, n);
		cout << endl;
	}
	m = (n - 1) / 2;
	if (v[0]) {
		if (v[0] != 1) {
			cout << "hyperbolic_pair::parabolic_point_properties: "
					"v[0] != 1" << endl;
			exit(1);
		}
		f_start_with_one = TRUE;
	}
	else {
		f_start_with_one = FALSE;
		F->Projective_space_basic->PG_element_normalize(
				v + 1, stride, n - 1);
		if (f_v) {
			cout << "hyperbolic_pair::parabolic_point_properties "
					"after normalization: ";
			Int_vec_print(cout, v, n);
			cout << endl;
		}
	}
	middle_value = O->Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
			v + 1 * stride, stride, m - 1);
	end_value = O->Quadratic_form->evaluate_hyperbolic_quadratic_form_with_m(
			v + (1 + 2 * (m - 1)) * stride, stride, 1);
}

int hyperbolic_pair::parabolic_is_middle_dependent(
		int *vec1, int *vec2)
{
	int i, j, *V1, *V2, a, b;

	V1 = NULL;
	V2 = NULL;
	for (i = 1; i < n - 2; i++) {
		if (vec1[i] == 0 && vec2[i] == 0) {
			continue;
		}
		if (vec1[i] == 0) {
			V1 = vec2;
			V2 = vec1;
		}
		else {
			V1 = vec1;
			V2 = vec2;
		}
		a = F->mult(V2[i], F->inverse(V1[i]));
		for (j = i; j < n - 2; j++) {
			b = F->add(F->mult(a, V1[j]), V2[j]);
			V2[j] = b;
		}
		break;
	}
	return O->is_zero_vector(V2 + 1, 1, n - 3);
}


}}}

