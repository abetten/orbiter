/*
 * decomposition_with_fuse.cpp
 *
 *  Created on: Dec 26, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace geometry_builder {


decomposition_with_fuse::decomposition_with_fuse()
{
	Record_birth();
	gg = NULL;

	nb_fuse = 0;
	Fuse_first = NULL;
	Fuse_len = NULL;
	K0 = NULL;
	KK = NULL;
	K1 = NULL;
	F_last_k_in_col = NULL;


	Conf = NULL;

	row_partition = NULL;
	col_partition = NULL;
	Partition = NULL;
	Partition_fixing_last = NULL;


}

decomposition_with_fuse::~decomposition_with_fuse()
{
	Record_death();
	if (Fuse_first) {
		FREE_int(Fuse_first);
	}
	if (Fuse_len) {
		FREE_int(Fuse_len);
	}
	if (K0) {
		FREE_int(K0);
	}
	if (KK) {
		FREE_int(KK);
	}
	if (K1) {
		FREE_int(K1);
	}
	if (F_last_k_in_col) {
		FREE_int(F_last_k_in_col);
	}

	if (Conf) {
		FREE_OBJECTS(Conf);
	}

	if (row_partition) {
		FREE_int(row_partition);
	}
	if (col_partition) {
		FREE_int(col_partition);
	}
	if (Partition) {
		int i;

		for (i = 0; i <= gg->GB->V; i++) {
			FREE_int(Partition[i]);
		}
		FREE_pint(Partition);
	}
	if (Partition_fixing_last) {
		int i;

		for (i = 0; i <= gg->GB->V; i++) {
			FREE_int(Partition_fixing_last[i]);
		}
		FREE_pint(Partition_fixing_last);
	}

}

gen_geo_conf *decomposition_with_fuse::get_conf_IJ(
		int I, int J)
{
	return Conf + I * gg->GB->b_len + J;
}


void decomposition_with_fuse::init(
		gen_geo *gg, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_with_fuse::init" << endl;
	}

	decomposition_with_fuse::gg = gg;

	if (f_v) {
		cout << "decomposition_with_fuse::init before init_fuse" << endl;
	}
	init_fuse(verbose_level);
	if (f_v) {
		cout << "decomposition_with_fuse::init after init_fuse" << endl;
	}

	if (f_v) {
		cout << "decomposition_with_fuse::init before TDO_init" << endl;
	}
	TDO_init(gg->GB->v, gg->GB->b, gg->GB->TDO, verbose_level);
	if (f_v) {
		cout << "decomposition_with_fuse::init after TDO_init" << endl;
	}

	if (f_v) {
		cout << "decomposition_with_fuse::init before init_partition" << endl;
	}
	init_partition(verbose_level);
	if (f_v) {
		cout << "decomposition_with_fuse::init after init_partition" << endl;
	}


	if (f_v) {
		cout << "decomposition_with_fuse::init done" << endl;
	}
}

void decomposition_with_fuse::TDO_init(
		int *v, int *b, int *theTDO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init" << endl;
	}
	int I, fuse_idx, f, l;

	Conf = NEW_OBJECTS(gen_geo_conf, gg->GB->v_len * gg->GB->b_len);


	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init before loops" << endl;
	}
	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		f = Fuse_first[fuse_idx];
		l = Fuse_len[fuse_idx];
		if (f_v) {
			cout << "decomposition_with_fuse::TDO_init fuse_idx=" << fuse_idx
					<< " f=" << f << " l=" << l << endl;
		}
		for (I = f; I < f + l; I++) {
			if (f_v) {
				cout << "decomposition_with_fuse::TDO_init fuse_idx=" << fuse_idx
						<< " f=" << f << " l=" << l
						<< " I=" << I << " v[I]=" << v[I] << endl;
			}
			init_tdo_line(fuse_idx,
					I /* tdo_line */, v[I] /* v */, b,
					theTDO + I * gg->GB->b_len /* r */,
					verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init after loops" << endl;
	}


	print_conf();

	gg->inc->print_param();



	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init before init_k" << endl;
	}
	init_k(verbose_level - 1);
	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init after init_k" << endl;
	}
	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init before conf_init_last_non_zero_flag" << endl;
	}
	conf_init_last_non_zero_flag(verbose_level - 1);
	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init after conf_init_last_non_zero_flag" << endl;
	}
	if (f_v) {
		cout << "decomposition_with_fuse::TDO_init done" << endl;
	}
}


void decomposition_with_fuse::init_tdo_line(
		int fuse_idx, int tdo_line,
		int v, int *b, int *r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j; //, rr;

	if (f_v) {
		cout << "decomposition_with_fuse::init_tdo_line tdo_line=" << tdo_line << endl;
		cout << "r=";
		Int_vec_print(cout, r, gg->GB->b_len);
		cout << endl;
	}
	if (tdo_line >= gg->GB->v_len) {
		cout << "decomposition_with_fuse::init_tdo_line tdo_line >= GB->v_len" << endl;
		exit(1);
	}



	for (j = 0; j < gg->GB->b_len; j++) {

		if (f_v) {
			cout << "decomposition_with_fuse::init_tdo_line "
					"tdo_line=" << tdo_line << " j=" << j << endl;
		}

		gen_geo_conf *C = get_conf_IJ(tdo_line, j);

		C->fuse_idx = fuse_idx;

		C->v = v;
		C->b = b[j];
		C->r = r[j];

		if (j == 0) {
			C->j0 = 0;
			C->r0 = 0;
		}
		else {
			gen_geo_conf *C_left = get_conf_IJ(tdo_line, j - 1);

			C->j0 = C_left->j0 + C_left->b;
			C->r0 = C_left->r0 + C_left->r;
		}

		if (tdo_line == 0) {
			C->i0 = 0;
		}
		else {
			gen_geo_conf *C_top = get_conf_IJ(tdo_line - 1, j);

			C->i0 = C_top->i0 + C_top->v;
		}
		//i0 = C->i0;

#if 0
		if (j == gg->GB->b_len - 1) {
			rr = C->r0 + C->r;
		}
#endif
	}


	if (f_v) {
		cout << "decomposition_with_fuse::init_tdo_line done" << endl;
	}
}

void decomposition_with_fuse::print_conf()
{
	int I, J;

	for (I = 0; I < gg->GB->v_len; I++) {
		for (J = 0; J < gg->GB->b_len; J++) {
			cout << "I=" << I << " J=" << J << ":" << endl;
			Conf[I * gg->GB->b_len + J].print(cout);
		}
	}
}

void decomposition_with_fuse::init_fuse(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_with_fuse::init_fuse" << endl;
	}
	Fuse_first = NEW_int(gg->GB->v_len);
	Fuse_len = NEW_int(gg->GB->v_len);
	int f, i;

	nb_fuse = gg->GB->fuse_len;
	f = 0;
	for (i = 0; i < gg->GB->fuse_len; i++) {
		Fuse_first[i] = f;
		Fuse_len[i] = gg->GB->fuse[i];
		f += gg->GB->fuse[i];
	}

	if (f_v) {
		cout << "decomposition_with_fuse::init_fuse done" << endl;
	}

}

void decomposition_with_fuse::init_k(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_with_fuse::init_k" << endl;
	}
	int I, J, fuse_idx, f, l, k, s, b;


	K0 = NEW_int(gg->GB->v_len * gg->GB->b_len);
	KK = NEW_int(gg->GB->v_len * gg->GB->b_len);
	K1 = NEW_int(gg->GB->v_len * gg->GB->b_len);
	F_last_k_in_col = NEW_int(gg->GB->v_len * gg->GB->b_len);

	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		for (J = 0; J < gg->GB->b_len; J++) {
			if (fuse_idx == 0) {
				K0[fuse_idx * gg->GB->b_len + J] = 0;
			}
			F_last_k_in_col[fuse_idx * gg->GB->b_len + J] = false;
		}
	}
	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		f = Fuse_first[fuse_idx];
		l = Fuse_len[fuse_idx];
		s = 0;
		for (J = 0; J < gg->GB->b_len; J++) {
			if (fuse_idx) {
				K0[fuse_idx * gg->GB->b_len + J] = K1[(fuse_idx - 1) * gg->GB->b_len + J];
			}
			s = 0;
			for (I = f; I < f + l; I++) {
				gen_geo_conf *C = get_conf_IJ(I, J);
				s += C->v * C->r;
				b = C->b;
			}
			k = s / b;
			if (k * b != s) {
				cout << "geo_init_k b does not divide s ! fuse_idx = " << fuse_idx
						<< " J = " << J << " s = " << s << " b = " << b << endl;
				exit(1);
			}
			KK[fuse_idx * gg->GB->b_len + J] = k;
			K1[fuse_idx * gg->GB->b_len + J] = K0[fuse_idx * gg->GB->b_len + J] + k;
		}
	}
	for (J = 0; J < gg->GB->b_len; J++) {
		for (fuse_idx = nb_fuse - 1; fuse_idx >= 0; fuse_idx--) {
			k = KK[fuse_idx * gg->GB->b_len + J];
			if (k) {
				F_last_k_in_col[fuse_idx * gg->GB->b_len + J] = true;
				break;
			}
		}
	}
	if (f_v) {
		cout << "KK:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < gg->GB->b_len; J++) {
				cout << setw(3) << KK[fuse_idx * gg->GB->b_len + J] << " ";
			}
			cout << endl;
		}
		cout << "K0:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < gg->GB->b_len; J++) {
				cout << setw(3) << K0[fuse_idx * gg->GB->b_len + J] << " ";
			}
			cout << endl;
		}

		cout << "K1:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < gg->GB->b_len; J++) {
				cout << setw(3) << K1[fuse_idx * gg->GB->b_len + J] << " ";
			}
			cout << endl;
		}

		cout << "F_last_k_in_col:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < gg->GB->b_len; J++) {
				cout << setw(3) << F_last_k_in_col[fuse_idx * gg->GB->b_len + J] << " ";
			}
			cout << endl;
		}
	}
	if (f_v) {
		cout << "decomposition_with_fuse::init_k done" << endl;
	}
}

void decomposition_with_fuse::conf_init_last_non_zero_flag(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition_with_fuse::conf_init_last_non_zero_flag" << endl;
	}
	int fuse_idx, ff, fl, i, I, r;

	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		ff = Fuse_first[fuse_idx];
		fl = Fuse_len[fuse_idx];
		for (i = fl - 1; i >= 0; i--) {
			I = ff + i;
			Conf[I * gg->GB->b_len + 0].f_last_non_zero_in_fuse = false;
		}
		for (i = fl - 1; i >= 0; i--) {
			I = ff + i;
			r = Conf[I * gg->GB->b_len + 0].r;
			if (r > 0) {
				Conf[I * gg->GB->b_len + 0].f_last_non_zero_in_fuse = true;
				break;
			}
		}
	}

	if (f_v) {
		cout << "f_last_non_zero_in_fuse:" << endl;
		for (I = 0; I < gg->GB->v_len; I++) {
			i = Conf[I * gg->GB->b_len + 0].f_last_non_zero_in_fuse;
			cout << setw(3) << i << " ";
		}
		cout << endl;
	}
	if (f_v) {
		cout << "decomposition_with_fuse::conf_init_last_non_zero_flag" << endl;
	}
}

void decomposition_with_fuse::init_partition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, I, J;

	if (f_v) {
		cout << "decomposition_with_fuse::init_partition" << endl;
	}
	row_partition = NEW_int(gg->GB->V);
	col_partition = NEW_int(gg->GB->B);

	for (i = 0; i < gg->GB->V; i++) {
		row_partition[i] = 1;
	}
	for (j = 0; j < gg->GB->B; j++) {
		col_partition[j] = 1;
	}


	for (I = 0; I < gg->GB->v_len; I++) {


		gen_geo_conf *C = get_conf_IJ(I, 0);
		i = C->i0 + C->v - 1;

		if (f_v) {
			cout << "I=" << I << " i=" << i << endl;
		}

		row_partition[i] = 0;

	}


	for (J = 0; J < gg->GB->b_len; J++) {

		gen_geo_conf *C = get_conf_IJ(0, J);
		j = C->j0 + C->b - 1;

		if (f_v) {
			cout << "J=" << J << " j=" << j << endl;
		}

		col_partition[j] = 0;

	}

	if (f_v) {
		cout << "row_partition: ";
		Int_vec_print(cout, row_partition, gg->GB->V);
		cout << endl;

		cout << "col_partition: ";
		Int_vec_print(cout, col_partition, gg->GB->B);
		cout << endl;
	}

	Partition = NEW_pint(gg->GB->V + 1);

	for (i = 0; i <= gg->GB->V; i++) {
		Partition[i] = NEW_int(i + gg->GB->B);
		Int_vec_copy(row_partition, Partition[i], i);
		if (i) {
			Partition[i][i - 1] = 0;
		}
		Int_vec_copy(col_partition, Partition[i] + i, gg->GB->B);

		if (f_v) {
			cout << "Partition[" << i << "]: ";
			Int_vec_print(cout, Partition[i], i + gg->GB->B);
			cout << endl;
		}
	}

	Partition_fixing_last = NEW_pint(gg->GB->V + 1);

	for (i = 0; i <= gg->GB->V; i++) {
		Partition_fixing_last[i] = NEW_int(i + gg->GB->B);
		Int_vec_copy(row_partition, Partition_fixing_last[i], i);
		if (i) {
			Partition_fixing_last[i][i - 1] = 0;
		}
		if (i >= 2) {
			Partition_fixing_last[i][i - 2] = 0;
		}
		Int_vec_copy(col_partition, Partition_fixing_last[i] + i, gg->GB->B);

		if (f_v) {
			cout << "Partition_fixing_last[" << i << "]: ";
			Int_vec_print(cout, Partition_fixing_last[i], i + gg->GB->B);
			cout << endl;
		}
	}


	if (f_v) {
		cout << "decomposition_with_fuse::init_partition done" << endl;
	}

}



}}}}



