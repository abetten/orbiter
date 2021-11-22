/*
 * gen_geo.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


gen_geo::gen_geo()
{
	GB = NULL;

	nb_fuse = 0;
	Fuse_first = NULL;
	Fuse_len = NULL;
	K0 = NULL;
	KK = NULL;
	K1 = NULL;
	F_last_k_in_col = NULL;


	Conf = NULL;

	inc = NULL;

	f_vbar = NULL;
	vbar = NULL;
	hbar = NULL;

	f_do_iso_test = FALSE;
	f_do_aut_group = FALSE;
	f_do_aut_group_in_iso_type_without_vhbars = FALSE;
	forget_ivhbar_in_last_isot = FALSE;
	gen_print_intervall = FALSE;

	//std::string inc_file_name;

	Girth_test = NULL;
}

gen_geo::~gen_geo()
{
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

	if (f_vbar) {
		FREE_int(f_vbar);
	}
	if (vbar) {
		FREE_int(vbar);
	}
	if (hbar) {
		FREE_int(hbar);
	}

	if (inc) {
		FREE_OBJECT(inc);
	}

	if (Girth_test) {
		FREE_OBJECT(Girth_test);
	}
}

void gen_geo::init(geometry_builder *GB,
	int f_do_iso_test,
	int f_do_aut_group,
	int f_do_aut_group_in_iso_type_without_vhbars,
	int gen_print_intervall,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init" << endl;
	}
	gen_geo::GB = GB;
	gen_geo::f_do_iso_test = f_do_iso_test;
	gen_geo::f_do_aut_group = f_do_aut_group;
	gen_geo::f_do_aut_group_in_iso_type_without_vhbars = f_do_aut_group_in_iso_type_without_vhbars;
	gen_geo::gen_print_intervall = gen_print_intervall;

	inc = NEW_OBJECT(incidence);

	if (f_v) {
		cout << "gen_geo::init before inc->init" << endl;
	}
	inc->init(this, GB->V, GB->B, GB->R, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after inc->init" << endl;
	}

	forget_ivhbar_in_last_isot = FALSE;




	if (f_v) {
		cout << "gen_geo::init before init_fuse" << endl;
	}
	init_fuse(verbose_level);
	if (f_v) {
		cout << "gen_geo::init after init_fuse" << endl;
	}

	if (f_v) {
		cout << "gen_geo::init before TDO_init" << endl;
	}
	TDO_init(GB->v, GB->b, GB->TDO, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after TDO_init" << endl;
	}


	if (f_v) {
		cout << "gen_geo::init before init_bars" << endl;
	}
	init_bars(verbose_level);
	if (f_v) {
		cout << "gen_geo::init before init_bars done" << endl;
	}


	// init_ISO();
	// init_ISO2();

	if (GB->Descr->f_girth_test) {

		Girth_test = NEW_OBJECT(girth_test);

		Girth_test->init(this, GB->Descr->girth, verbose_level);

	}

	if (f_v) {
		cout << "gen_geo::init done" << endl;
	}

}

void gen_geo::TDO_init(int *v, int *b, int *theTDO, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::TDO_init" << endl;
	}
	int I, fuse_idx, f, l;

	Conf = NEW_OBJECTS(gen_geo_conf, GB->v_len * GB->b_len);


	if (f_v) {
		cout << "gen_geo::TDO_init before loops" << endl;
	}
	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		f = Fuse_first[fuse_idx];
		l = Fuse_len[fuse_idx];
		if (f_v) {
			cout << "gen_geo::TDO_init fuse_idx=" << fuse_idx << " f=" << f << " l=" << l << endl;
		}
		for (I = f; I < f + l; I++) {
			if (f_v) {
				cout << "gen_geo::TDO_init fuse_idx=" << fuse_idx << " f=" << f << " l=" << l
						<< " I=" << I << " v[I]=" << v[I] << endl;
			}
			init_tdo_line(fuse_idx,
					I /* tdo_line */, v[I] /* v */, b,
					theTDO + I * GB->b_len /* r */,
					verbose_level - 1);
		}
	}
	if (f_v) {
		cout << "gen_geo::TDO_init after loops" << endl;
	}


	print_conf();

	inc->print_param();



	if (f_v) {
		cout << "gen_geo::TDO_init before init_k" << endl;
	}
	init_k(verbose_level - 1);
	if (f_v) {
		cout << "gen_geo::TDO_init after init_k" << endl;
	}
	if (f_v) {
		cout << "gen_geo::TDO_init before conf_init_last_non_zero_flag" << endl;
	}
	conf_init_last_non_zero_flag(verbose_level - 1);
	if (f_v) {
		cout << "gen_geo::TDO_init after conf_init_last_non_zero_flag" << endl;
	}
	if (f_v) {
		cout << "gen_geo::TDO_init done" << endl;
	}
}


void gen_geo::init_tdo_line(int fuse_idx, int tdo_line,
		int v, int *b, int *r, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i0, j, rr;

	if (f_v) {
		cout << "gen_geo::init_tdo_line tdo_line=" << tdo_line << endl;
		cout << "r=";
		Orbiter->Int_vec.print(cout, r, GB->b_len);
		cout << endl;
	}
	if (tdo_line >= GB->v_len) {
		cout << "gen_geo::init_tdo_line tdo_line >= GB->v_len" << endl;
		exit(1);
	}



	//int V, B;

	//V = 0;
	for (j = 0; j < GB->b_len; j++) {

		if (f_v) {
			cout << "gen_geo::init_tdo_line tdo_line=" << tdo_line << " j=" << j << endl;
		}

		Conf[tdo_line * GB->b_len + j].fuse_idx = fuse_idx;

		Conf[tdo_line * GB->b_len + j].v = v;
		Conf[tdo_line * GB->b_len + j].b = b[j];
		Conf[tdo_line * GB->b_len + j].r = r[j];

		if (j == 0) {
			Conf[tdo_line * GB->b_len + j].j0 = 0;
			Conf[tdo_line * GB->b_len + j].r0 = 0;
		}
		else {
			Conf[tdo_line * GB->b_len + j].j0 =
					Conf[tdo_line * GB->b_len + j - 1].j0 +
					Conf[tdo_line * GB->b_len + j - 1].b;
			Conf[tdo_line * GB->b_len + j].r0 =
					Conf[tdo_line * GB->b_len + j - 1].r0 +
					Conf[tdo_line * GB->b_len + j - 1].r;
		}

		if (tdo_line == 0) {
			Conf[tdo_line * GB->b_len + j].i0 = 0;
		}
		else {
			Conf[tdo_line * GB->b_len + j].i0 =
					Conf[(tdo_line - 1) * GB->b_len + j].i0 +
					Conf[(tdo_line - 1) * GB->b_len + j].v;
		}
		i0 = Conf[tdo_line * GB->b_len + j].i0;

		if (j == GB->b_len - 1) {
			rr = Conf[tdo_line * GB->b_len + j].r0 + Conf[tdo_line * GB->b_len + j].r;
			if (rr >= MAX_R) {
				cout << "geo_tdo_init rr >= MAX_R" << endl;
				exit(1);
			}
			//max_r = MAXIMUM(max_r, rr);
		}
#if 0
		for (i = i0; i < i0 + v; i++) {
			//inc->R[i] = rr;
			R[i] = rr;
		}
#endif
		//V += v;
	}

#if 0
	if (f_v) {
		cout << "gen_geo::init_tdo_line computing B" << endl;
	}
	inc->Encoding->b = 0;
	for (j = 0; j < GB->b_len; j++) {
		inc->Encoding->b += b[j];
	}
	B = inc->Encoding->b;
#endif

	if (f_v) {
		cout << "gen_geo::init_tdo_line done" << endl;
	}
}

void gen_geo::print_conf()
{
	int I, J;

	for (I = 0; I < GB->v_len; I++) {
		for (J = 0; J < GB->b_len; J++) {
			cout << "I=" << I << " J=" << J << ":" << endl;
			Conf[I * GB->b_len + J].print(cout);
		}
	}
}

void gen_geo::init_bars(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "gen_geo::init_bars" << endl;
	}
	f_vbar = NEW_int(GB->V * inc->Encoding->dim_n);

	for (i = 0; i < GB->V * inc->Encoding->dim_n; i++) {
		f_vbar[i] = FALSE;
	}

	hbar = NEW_int(GB->V + 1);
	for (i = 0; i <= GB->V; i++) {
		hbar[i] = INT_MAX;
	}

	vbar = NEW_int(GB->B + 1);
	for (i = 0; i <= GB->B; i++) {
		vbar[i] = INT_MAX;
	}

	if (f_v) {
		cout << "gen_geo::init_bars before inc->init_bars" << endl;
	}
	inc->init_bars(verbose_level);
	if (f_v) {
		cout << "gen_geo::init_bars after inc->init_bars" << endl;
	}

	if (f_v) {
		cout << "gen_geo::init_bars done" << endl;
	}
}

void gen_geo::init_fuse(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init_fuse" << endl;
	}
	Fuse_first = NEW_int(GB->v_len);
	Fuse_len = NEW_int(GB->v_len);
	int f, i;

	nb_fuse = GB->fuse_len;
	f = 0;
	for (i = 0; i < GB->fuse_len; i++) {
		Fuse_first[i] = f;
		Fuse_len[i] = GB->fuse[i];
		f += GB->fuse[i];
	}

	if (f_v) {
		cout << "gen_geo::init_fuse done" << endl;
	}

}

void gen_geo::init_k(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init_k" << endl;
	}
	int I, J, fuse_idx, f, l, k, s, b;


	K0 = NEW_int(GB->v_len * GB->b_len);
	KK = NEW_int(GB->v_len * GB->b_len);
	K1 = NEW_int(GB->v_len * GB->b_len);
	F_last_k_in_col = NEW_int(GB->v_len * GB->b_len);

	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		for (J = 0; J < GB->b_len; J++) {
			if (fuse_idx == 0) {
				K0[fuse_idx * GB->b_len + J] = 0;
			}
			F_last_k_in_col[fuse_idx * GB->b_len + J] = FALSE;
		}
	}
	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		f = Fuse_first[fuse_idx];
		l = Fuse_len[fuse_idx];
		s = 0;
		for (J = 0; J < GB->b_len; J++) {
			if (fuse_idx) {
				K0[fuse_idx * GB->b_len + J] = K1[(fuse_idx - 1) * GB->b_len + J];
			}
			s = 0;
			for (I = f; I < f + l; I++) {
				s += Conf[I * GB->b_len + J].v * Conf[I * GB->b_len + J].r;
				b = Conf[I * GB->b_len + J].b;
			}
			k = s / b;
			if (k * b != s) {
				cout << "geo_init_k b does not divide s ! fuse_idx = " << fuse_idx << " J = " << J << " s = " << s << " b = " << b << endl;
				exit(1);
			}
			KK[fuse_idx * GB->b_len + J] = k;
			K1[fuse_idx * GB->b_len + J] = K0[fuse_idx * GB->b_len + J] + k;
		}
	}
	for (J = 0; J < GB->b_len; J++) {
		for (fuse_idx = nb_fuse - 1; fuse_idx >= 0; fuse_idx--) {
			k = KK[fuse_idx * GB->b_len + J];
			if (k) {
				F_last_k_in_col[fuse_idx * GB->b_len + J] = TRUE;
				break;
			}
		}
	}
	if (f_v) {
		cout << "KK:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < GB->b_len; J++) {
				cout << setw(3) << KK[fuse_idx * GB->b_len + J] << " ";
			}
			cout << endl;
		}
		cout << "K0:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < GB->b_len; J++) {
				cout << setw(3) << K0[fuse_idx * GB->b_len + J] << " ";
			}
			cout << endl;
		}

		cout << "K1:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < GB->b_len; J++) {
				cout << setw(3) << K1[fuse_idx * GB->b_len + J] << " ";
			}
			cout << endl;
		}

		cout << "F_last_k_in_col:" << endl;
		for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
			for (J = 0; J < GB->b_len; J++) {
				cout << setw(3) << F_last_k_in_col[fuse_idx * GB->b_len + J] << " ";
			}
			cout << endl;
		}
	}
	if (f_v) {
		cout << "gen_geo::init_k done" << endl;
	}
}

void gen_geo::conf_init_last_non_zero_flag(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::conf_init_last_non_zero_flag" << endl;
	}
	int fuse_idx, ff, fl, i, I, r;

	for (fuse_idx = 0; fuse_idx < nb_fuse; fuse_idx++) {
		ff = Fuse_first[fuse_idx];
		fl = Fuse_len[fuse_idx];
		for (i = fl - 1; i >= 0; i--) {
			I = ff + i;
			Conf[I * GB->b_len + 0].f_last_non_zero_in_fuse = FALSE;
		}
		for (i = fl - 1; i >= 0; i--) {
			I = ff + i;
			r = Conf[I * GB->b_len + 0].r;
			if (r > 0) {
				Conf[I * GB->b_len + 0].f_last_non_zero_in_fuse = TRUE;
				break;
			}
		}
	}

	if (f_v) {
		cout << "f_last_non_zero_in_fuse:" << endl;
		for (I = 0; I < GB->v_len; I++) {
			i = Conf[I * GB->b_len + 0].f_last_non_zero_in_fuse;
			cout << setw(3) << i << " ";
		}
		cout << endl;
	}
	if (f_v) {
		cout << "gen_geo::conf_init_last_non_zero_flag" << endl;
	}
}


void gen_geo::print_pairs(int line)
{
	int i1, i2, a;

	for (i1 = 0; i1 < line; i1++) {
		cout << "line " << i1 << " : ";
		for (i2 = 0; i2 < i1; i2++) {
			a = inc->pairs[i1][i2];
			cout << a;
		}
		cout << endl;
	}
}

void gen_geo::main2(int &nb_GEN, int &nb_GEO, int &ticks, int &tps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	os_interface Os;

	if (f_v) {
		cout << "gen_geo::main2, verbose_level = " << verbose_level << endl;
	}
	int t0, t1, user_time, V;

	t0 = Os.os_ticks();

	if (f_v) {
		cout << "gen_geo::main2 before generate_all" << endl;
	}
	generate_all(verbose_level);
	if (f_v) {
		cout << "gen_geo::main2 after generate_all" << endl;
	}





	t1 = Os.os_ticks();

	V = inc->Encoding->v;
	iso_type *it;

	it = inc->iso_type_at_line[V - 1];
	{
		string fname;
		fname.assign(inc_file_name);
		fname.append(".inc");
		it->write_inc_file(fname, verbose_level);

		file_io Fio;

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	it = inc->iso_type_at_line[V - 1];

	if (it->Canonical_forms->B.size()) {

		{
			string fname;
			fname.assign(inc_file_name);
			fname.append(".blocks_long");
			it->write_blocks_file_long(fname, verbose_level);

			file_io Fio;

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		}

		object_in_projective_space *OiP;

		OiP = (object_in_projective_space *) it->Canonical_forms->Objects[0];

		if (inc->is_block_tactical(V, OiP->set)) {

			string fname;
			fname.assign(inc_file_name);
			fname.append(".blocks");
			it->write_blocks_file(fname, verbose_level);

			file_io Fio;

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}
	}


	if (inc->iso_type_no_vhbars) {
		it = inc->iso_type_no_vhbars;
		{
			string fname;
			fname.assign(inc_file_name);
			fname.append("_resolved.inc");
			it->write_inc_file(fname, verbose_level);
		}
	}


	inc->print(cout, V, V);

	int i, a;

	for (i = 0; i < GB->Descr->print_at_line.size(); i++) {
		a = GB->Descr->print_at_line[i];
		inc->iso_type_at_line[a - 1]->print_geos(verbose_level);
	}

	// Anton Betten AB 150200:
	it = inc->iso_type_at_line[V - 1];
	cout << "generated: " << it->sum_nb_GEN
			<< " nb_TDO = " << it->sum_nb_TDO
			<< " nb_GEO = " << it->sum_nb_GEO << endl;

	if (inc->iso_type_no_vhbars) {
		it = inc->iso_type_no_vhbars;
		cout << "resolved: generated: " << it->nb_GEN
				<< " nb_TDO = " << it->nb_TDO
				<< " nb_GEO = " << it->nb_GEO << endl;
	}


	nb_GEN = inc->gl_nb_GEN;
	nb_GEO = it->nb_GEO;
	user_time = t1 - t0;
	ticks = user_time;


	tps = Os.os_ticks_per_second();

	if (f_v) {
		cout << "gen_geo::main2 done" << endl;
	}

	Os.time_check_delta(cout, user_time);

}

void gen_geo::generate_all(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);

	if (f_v) {
		cout << "gen_geo::generate_all, verbose_level = " << verbose_level << endl;
	}

	int ret = FALSE;
	int f_already_there;
	int s_nb_i_vbar, s_nb_i_hbar;
	iso_type *it0, *it1;

	if (f_v) {
		if (GB->Descr->f_lambda) {
			cout << "lambda = " << GB->Descr->lambda << endl;
		}
	}

	if (f_v) {
		cout << "gen_geo::generate_all before it0 = ..." << endl;
	}

	it0 = inc->iso_type_at_line[inc->Encoding->v - 1];

	if (it0 == NULL) {
		cout << "please install a test at line " << inc->Encoding->v << endl;
		exit(1);
	}

	if (f_v) {
		cout << "gen_geo::generate_all before it1 = ..." << endl;
	}

	it1 = inc->iso_type_no_vhbars;


	if (it1 && forget_ivhbar_in_last_isot) {
		cout << "gen_geo::generate_all inc.iso_type_no_vhbars && forget_ivhbar_in_last_isot" << endl;
		goto l_exit;
	}

	inc->gl_nb_GEN = 0;
	if (!GeoFst(verbose_level - 5)) {
		ret = TRUE;
		goto l_exit;
	}
	while (TRUE) {

		inc->gl_nb_GEN++;
		if (FALSE) {
			cout << "gen_geo::generate_all nb_GEN=" << inc->gl_nb_GEN << endl;
			if ((inc->gl_nb_GEN % gen_print_intervall) == 0) {
				//inc->print(cout, inc->Encoding->v);
			}
		}
		//cout << "*** do_geo *** geometry no. " << gg->inc.gl_nb_GEN << endl;
#if 0
		if (incidence_back_test(&gg->inc,
			TRUE /* f_verbose */,
			FALSE /* f_very_verbose */)) {
			}
#endif

		if (forget_ivhbar_in_last_isot) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;
		}
		if (FALSE) {
			cout << "gen_geo::generate_all before isot_add for it0" << endl;
		}

		it0->add_geometry(inc->Encoding,
			inc->Encoding->v, inc,
			f_already_there,
			0 /*verbose_level*/);

		if (FALSE) {
			cout << "gen_geo::generate_all after isot_add for it0" << endl;
		}

		if (f_vv && it0->Canonical_forms->B.size() % 1 == 0) {
			cout << it0->Canonical_forms->B.size() << endl;
			inc->print(cout, inc->Encoding->v, inc->Encoding->v);
		}

		if (forget_ivhbar_in_last_isot) {
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!f_already_there && inc->iso_type_no_vhbars) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;

			if (FALSE) {
				cout << "gen_geo::generate_all before isot_add for it1" << endl;
			}
			it1->add_geometry(inc->Encoding,
					inc->Encoding->v, inc,
					f_already_there,
					0 /*verbose_level*/);

			if (FALSE) {
				cout << "gen_geo::generate_all after isot_add for it1" << endl;
			}
			if (!f_already_there) {
				//save_theX(inc->iso_type_no_vhbars->fp);
			}
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!f_already_there && !inc->iso_type_no_vhbars) {
		}

		if (!GeoNxt(verbose_level - 5)) {
			break;
		}
	}
l_exit:
	if (f_v) {
		cout << "gen_geo::generate_all done" << endl;
	}
}

void gen_geo::print_I_m(int I, int m)
{
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i1;

	i1 = C->i0 + m;
	inc->print(cout, i1 + 1, i1 + 1);
}

void gen_geo::print(int v)
{
	inc->print(cout, v, v);
}

int gen_geo::GeoFst(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "GeoFst" << endl;
	}
	int I;

	I = 0;
	while (TRUE) {
		while (TRUE) {
			if (I >= GB->v_len) {
				return TRUE;
			}
			if (!GeoRowFst(I, verbose_level)) {
				break;
			}
			I++;
		}
		// I-th element could not initialize, move on
		while (TRUE) {
			if (I == 0) {
				return FALSE;
			}
			I--;
			if (GeoRowNxt(I, verbose_level)) {
				break;
			}
		}
		// I-th element has been incremented. Initialize elements after it:
		I++;
	}
}

int gen_geo::GeoNxt(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoNxt" << endl;
	}
	int I;

	I = GB->v_len - 1;
	while (TRUE) {
		while (TRUE) {
			if (GeoRowNxt(I, verbose_level)) {
				break;
			}
			if (I == 0) {
				return FALSE;
			}
			I--;
		}
		// I-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (I >= GB->v_len - 1) {
				return TRUE;
			}
			I++;
			if (!GeoRowFst(I, verbose_level)) {
				break;
			}
		}
		// I-th element could not initialize, move on
		I--;
	}
}

int gen_geo::GeoRowFst(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len;

	if (f_v) {
		cout << "gen_geo::GeoRowFst I=" << I << endl;
	}

	int m;

	m = 0;
	while (TRUE) {
		while (TRUE) {
			if (m >= C->v) {
				return TRUE;
			}
			if (!GeoLineFstRange(I, m, verbose_level)) {
				break;
			}
			m++;
		}
		// m-th element could not initialize, move on
		while (TRUE) {
			if (m == 0) {
				return FALSE;
			}
			m--;
			if (GeoLineNxtRange(I, m, verbose_level)) {
				break;
			}
		}
		// m-th element has been incremented. Initialize elements after it:
		m++;
	}
}

int gen_geo::GeoRowNxt(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len;

	if (f_v) {
		cout << "gen_geo::GeoRowNxt I=" << I << endl;
	}

	int m;

	m = C->v - 1;
	while (TRUE) {
		while (TRUE) {
			if (GeoLineNxtRange(I, m, verbose_level)) {
				break;
			}
			if (m == 0) {
				return FALSE;
			}
			m--;
		}
		// m-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (m >= C->v - 1) {
				return TRUE;
			}
			m++;
			if (!GeoLineFstRange(I, m, verbose_level)) {
				break;
			}
		}
		// m-th element could not initialize, move on
		m--;
	}
}

#define GEO_LINE_RANGE

int gen_geo::GeoLineFstRange(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineFstRange I=" << I << " m=" << m << endl;
	}

#ifdef GEO_LINE_RANGE
	iso_type *it;
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i1;

	i1 = C->i0 + m;
	it = inc->iso_type_at_line[i1];
	if (it && it->f_range) {
		if (it->nb_GEO == it->range_first + it->range_len - 1) {
			return FALSE;
		}
		if (it->nb_GEO > it->range_first + it->range_len - 1) {
			return FALSE;
		}
	}
	if (!GeoLineFst0(I, m, verbose_level)) {
		return FALSE;
	}
	if (it && it->f_range) {
		if (it->nb_GEO < it->range_first) {
			while (it->nb_GEO < it->range_first) {
				cout << "gen_geo::GeoLineFstRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
				if (!GeoLineNxt0(I, m, verbose_level)) {
					return FALSE;
				}
			}
		}
		cout << "gen_geo::GeoLineFstRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
	}
	return TRUE;
#else
	return GeoLineFst0(I, m, verbose_level);
#endif
}

int gen_geo::GeoLineNxtRange(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineNxtRange I=" << I << " m=" << m << endl;
	}
#ifdef GEO_LINE_RANGE
	iso_type *it;
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i1;

	i1 = C->i0 + m;
	it = inc->iso_type_at_line[i1];
	if (it && it->f_range) {
		if (it->nb_GEO == it->range_first + it->range_len - 1) {
			if (f_v) {
				cout << "gen_geo::GeoLineNxtRange: line " << i1 + 1
						<< " case end at " << it->range_first + it->range_len
						<<  " reached" << endl;
			}
			return FALSE;
		}
		if (it->nb_GEO > it->range_first + it->range_len - 1) {
			return FALSE;
		}
	}
	if (!GeoLineNxt0(I, m, verbose_level)) {
		return FALSE;
	}
	if (it && it->f_range) {
		while (it->nb_GEO < it->range_first) {
			if (f_v) {
				cout << "gen_geo::GeoLineNxtRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
			}
			if (!GeoLineNxt0(I, m, verbose_level)) {
				return FALSE;
			}
		}
		if (f_v) {
			cout << "gen_geo::GeoLineNxtRange: "
					"line " << i1 + 1 << " case " << it->nb_GEO << ":" << endl;
		}
	}
	return TRUE;
#else
	return GeoLineNxt0(gg, I, m, verbose_level);
#endif
}

int gen_geo::geo_back_test(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::geo_back_test I=" << I << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i0, i1, m, f_already_there, control_line;
	iso_type *it;

	i0 = C->i0;
	control_line = i0 + C->v - 1;
	for (m = 0; m < C->v - 1; m++) {
		i1 = i0 + m;
		it = inc->iso_type_at_line[i1];
		if (it && it->f_generate_first && !it->f_beginning_checked) {

			it->add_geometry(inc->Encoding,
					i1 + 1, inc,
					f_already_there,
					0 /*verbose_level*/);

			if (!f_already_there) {
				it->f_beginning_checked = TRUE;
				continue;
			}
			inc->back_to_line = i1;
			return FALSE;
		}
	}
	return TRUE;
}


int gen_geo::GeoLineFst0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineFst0 I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int f_already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!GeoLineFst(I, m, verbose_level)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		it->f_beginning_checked = FALSE;
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!GeoLineNxt(I, m, verbose_level)) {
				return FALSE;
			}
			cout << "gen_geo::GeoLineFst0 back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		// survived the back test,
		// and now one test of the first kind:
	}
	if (i1 == inc->Encoding->v - 1) {
		// a new geometry is completed
		// let the main routine add it
		return TRUE;
	}
	if (it) {
		// test of the first kind
		while (TRUE) {
			if (f_v) {
				cout << "gen_geo::GeoLineFst0 I=" << I << " m=" << m << " before isot_add" << endl;
				inc->print(cout, i1 + 1, i1 + 1);
			}

			it->add_geometry(inc->Encoding,
					i1 + 1, inc, f_already_there,
					0 /*verbose_level*/);

			if (f_v) {
				cout << "gen_geo::GeoLineFst0 I=" << I << " m=" << m << " after isot_add" << endl;
			}
			if (it->f_print_mod) {
				if ((it->nb_GEN % it->print_mod) == 0) {
					//inc->print(cout, i1 + 1);
					// geo_print_pairs(gg, i1 + 1);
				}
			}
			if (!f_already_there) {
				break;
			}
			if (!GeoLineNxt(I, m, verbose_level)) {
				return FALSE;
			}
		}
		// now: a new geometry has been produced,
		// f_already_there is FALSE
	}
	return TRUE;
}

int gen_geo::GeoLineNxt0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineNxt0 I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int f_already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!GeoLineNxt(I, m, verbose_level)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		it->f_beginning_checked = FALSE;
#if 0
		gg->inc.nb_GEO[i1] = ((ISO_TYPE *)
		gg->inc.iso_type[control_line])->nb_GEO;
#endif
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!GeoLineNxt(I, m, verbose_level)) {
				return FALSE;
			}
			cout << "gen_geo::GeoLineNxt0 back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		// survived the back test,
		// and now one test of the first kind:
	}
	if (i1 == inc->Encoding->v - 1) {
		// a new geometry is completed
		// let the main routine add it
		return TRUE;
	}
	if (it) {
		while (TRUE) {
			it->add_geometry(inc->Encoding,
				i1 + 1, inc, f_already_there,
				0 /*verbose_level*/);

#if 0
			if (it->f_print_mod) {
				if ((it->nb_GEN % it->print_mod) == 0) {
					//inc->print(cout, i1 + 1);
					// geo_print_pairs(gg, i1 + 1);
				}
			}
#endif

			if (!f_already_there) {
				break;
			}
			if (!GeoLineNxt(I, m, verbose_level)) {
				return FALSE;
			}
		}
		// now: a new geometry has been produced,
		// f_already_there is FALSE
	}
	return TRUE;
}

int gen_geo::GeoLineFst(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len;
	int J, i1;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "gen_geo::GeoLineFst I=" << I << " m=" << m << " i1=" << i1 << endl;
	}

	girth_Floyd(i1, verbose_level);

	J = 0;
	while (TRUE) {
		while (TRUE) {
			if (J >= GB->b_len) {
				return TRUE;
			}
			if (!GeoConfFst(I, m, J, verbose_level)) {
				break;
			}
			J++;
		}
		// J-th element could not initialize, move on
		while (TRUE) {
			if (J == 0) {
				return FALSE;
			}
			J--;
			if (GeoConfNxt(I, m, J, verbose_level)) {
				break;
			}
		}
		// J-th element has been incremented. Initialize elements after it:
		J++;
	}
}

int gen_geo::GeoLineNxt(int I, int m, int verbose_level)
{
	gen_geo_conf *C = Conf + I * GB->b_len;
	int J, i1;

	i1 = C->i0 + m;
	if (inc->back_to_line != -1 && inc->back_to_line < i1) {
		GeoLineClear(I, m);
		return FALSE;
	}
	if (inc->back_to_line != -1 && inc->back_to_line == i1) {
		inc->back_to_line = -1;
	}
	J = GB->b_len - 1;
	while (TRUE) {
		while (TRUE) {
			if (GeoConfNxt(I, m, J, verbose_level)) {
				break;
			}
			if (J == 0) {
				return FALSE;
			}
			J--;
		}
		// J-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (J >= GB->b_len - 1) {
				return TRUE;
			}
			J++;
			if (!GeoConfFst(I, m, J, verbose_level)) {
				break;
			}
		}
		// J-th element could not initialize, move on
		J--;
	}
}

void gen_geo::GeoLineClear(int I, int m)
{
	int J;

	for (J = GB->b_len - 1; J >= 0; J--) {
		GeoConfClear(I, m, J);
	}
}

int gen_geo::GeoConfFst(int I, int m, int J, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoConfFst I=" << I << " m=" << m << " J=" << J << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int n, i1;

	if (J == 0) {
		i1 = C->i0 + m;
		if (m == 0) {
			hbar[i1] = -1;
			// initial hbar
		}
		else {
			hbar[i1] = GB->b_len;
			// no hbar
		}
	}
	n = 0;
	while (TRUE) {
		while (TRUE) {
			if (n >= C->r) {
				if (f_v) {
					cout << "gen_geo::GeoConfFst I=" << I << " m=" << m << " J=" << J << " returns TRUE" << endl;
				}
				return TRUE;
			}
			if (!GeoXFst(I, m, J, n, verbose_level)) {
				break;
			}
			n++;
		}
		// n-th element could not initialize, move on
		while (TRUE) {
			if (n == 0) {
				if (f_v) {
					cout << "gen_geo::GeoConfFst I=" << I << " m=" << m << " J=" << J << " returns FALSE" << endl;
				}
				return FALSE;
			}
			n--;
			if (GeoXNxt(I, m, J, n, verbose_level)) {
				break;
			}
		}
		// n-th element has been incremented. Initialize elements after it:
		n++;
	}
}

int gen_geo::GeoConfNxt(int I, int m, int J, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoConfNxt I=" << I << " m=" << m << " J=" << J << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int n, i1;

	i1 = C->i0 + m;
	if (hbar[i1] > J) {
		hbar[i1] = J;
	}
	if (C->r == 0) {
		return FALSE;
	}
	n = C->r - 1;
	while (TRUE) {
		while (TRUE) {
			if (GeoXNxt(I, m, J, n, verbose_level)) {
				break;
			}
			if (n == 0) {
				return FALSE;
				if (f_v) {
					cout << "gen_geo::GeoConfNxt I=" << I << " m=" << m << " J=" << J << " returns FALSE" << endl;
				}
			}
			n--;
		}
		// n-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (n >= C->r - 1) {
				if (f_v) {
					cout << "gen_geo::GeoConfNxt I=" << I << " m=" << m << " J=" << J << " returns TRUE" << endl;
				}
				return TRUE;
			}
			n++;
			if (!GeoXFst(I, m, J, n, verbose_level)) {
				break;
			}
		}
		// n-th element could not initialize, move on
		n--;
	}
}

void gen_geo::GeoConfClear(int I, int m, int J)
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int n;

	if (C->r == 0) {
		return;
	}
	for (n = C->r - 1; n >= 0; n--) {
		GeoXClear(I, m, J, n);
	}
}

int gen_geo::GeoXFst(int I, int m, int J, int n, int verbose_level)
// maintains hbar[], vbar[], f_vbar[][], theX[][], K[]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoXFst I=" << I << " m=" << m << " J=" << J << " n=" << n << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int i1, j0, r, j;

	i1 = C->i0 + m; // current row
	r = C->r0 + n; // current incidence index
	j0 = C->j0;
	if (hbar[i1] <= J) {
		// hbar exists, which means that the left part of the row differs from the row above.
		// The next incidence must be tried starting from the leftmost position.
		// We cannot copy over from the previous row.
		if (n == 0) {
			// first incidence inside the block?
			j = 0;
		}
		else {

			// start out one to the right of the previous incidence:

			j = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] - j0 + 1;
		}
	}
	else {
		if (m == 0) {
			cout << "gen_geo::GeoXFst hbar[i1] > J && m == 0" << endl;
			exit(1);
		}
		// no hbar means that the left parts agree.

		// pick the incidence according to the previous row:
		j = inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r] - j0;
	}
	int ret;

	ret = X_Fst(I, m, J, n, j, verbose_level);

	if (f_v) {
		cout << "gen_geo::GeoXFst I=" << I << " m=" << m << " J=" << J << " n=" << n << " returns " << ret << endl;
	}

	return ret;
}

int gen_geo::GeoXNxt(int I, int m, int J, int n, int verbose_level)
// maintains: hbar[], vbar[], f_vbar[][], theX[][], K[]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int old_x;
	int fuse_idx, i1, j0, j1, r, j, k;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m; // current row
	r = C->r0 + n; // current incidence index
	j0 = C->j0;

	old_x = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r];

	girth_test_delete_incidence(i1, r, old_x);

	inc->K[old_x]--;
	if (GB->Descr->f_lambda) {
		k = inc->K[old_x];
		inc->theY[old_x][k] = -1;

		decrement_pairs_point(i1, old_x, k);

	}

	// remove vbar:
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		vbar[old_x + 1] = MAX_V;
		f_vbar[i1 * inc->Encoding->dim_n + r] = FALSE;
	}

	// possibly create new vbar on the left:
	if (n > 0) {
		if (vbar[old_x] > i1 &&
			inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] == old_x - 1) {
			vbar[old_x] = i1;
			f_vbar[i1 * inc->Encoding->dim_n + r - 1] = TRUE;
		}
	}

#if 0
	// diese Stelle ist gefaehrlich!
	if (J == 0) {
		if (n == 0)
			return FALSE;
		}
#endif
	// new version, works with FUSE:
	if (J == 0 && n == 0) {
		if (C->f_last_non_zero_in_fuse) {
			return FALSE;
		}
	}

	for (j = old_x - j0 + 1; j < C->b; j++) {

		if (inc->K[j0 + j] >= K1[fuse_idx * GB->b_len + J]) {
			// this column is already full. We cannot put the incidence here.
			if (f_v) {
				cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of column sum wrt fuse partition" << endl;
			}
			continue;
		}

		if (vbar[j0 + j] > i1) {
			if (f_v) {
				cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of vbar" << endl;
			}
			continue;
		}

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j0 + j;
			// must be set before calling find_square



		k = inc->K[j0 + j];
		inc->theY[j0 + j][k] = i1;
		j1 = j0 + j;

		if (!apply_tests(I, m, J, n, j, verbose_level - 2)) {
			if (f_v) {
				cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of test" << endl;
			}
			continue;
		}


		// the incidence passes the tests:


		// increase column sum
		inc->K[j0 + j]++;


		// generate new vbar to the left of this incidence:
		if (vbar[j0 + j + 1] == i1) {
			cout << "gen_geo::GeoXNxt vbar[j0 + j + 1] == i1" << endl;
			exit(1);
		}
		if (vbar[j0 + j + 1] > i1) {
			f_vbar[i1 * inc->Encoding->dim_n + r] = TRUE;
			vbar[j0 + j + 1] = i1;
		}

		if (hbar[i1] > J) {
			if (m == 0) {
				cout << "gen_geo::GeoXNxt no hbar && m == 0" << endl;
				exit(1);
			}
			if (j0 + j != inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r]) {
				// generate new hbar:
				hbar[i1] = J;
			}
		}

		if (f_v) {
			cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << " returns TRUE" << endl;
		}
		return TRUE;
	}
	if (f_v) {
		cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J << " n=" << n << " returns FALSE" << endl;
	}
	return FALSE;
}

void gen_geo::GeoXClear(int I, int m, int J, int n)
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int old_x;
	int i1, j0, r, k;

	i1 = C->i0 + m;
		// current row
	r = C->r0 + n;
		// index of current incidence
	j0 = C->j0;
	old_x = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r];
	inc->K[old_x]--;


	girth_test_delete_incidence(i1, r, old_x);


	// remove old vbar:
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		vbar[old_x + 1] = MAX_V;
		f_vbar[i1 * inc->Encoding->dim_n + r] = FALSE;
	}
	// recover vbar to the left if necessary:
	if (n > 0) {
		if (vbar[old_x] > i1 &&
			inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] == old_x - 1) {
			vbar[old_x] = i1;
			f_vbar[i1 * inc->Encoding->dim_n + r - 1] = TRUE;
		}
	}
	inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
	if (GB->Descr->f_lambda) {
		k = inc->K[old_x];
		inc->theY[old_x][k] = -1;

		decrement_pairs_point(i1, old_x, k);

	}
}

int gen_geo::X_Fst(int I, int m, int J, int n, int j, int verbose_level)
// Try placing an incidence, starting from column j and moving to the right
// j is local coordinate
// maintains hbar[], vbar[], f_vbar[][], theX[][], K[]
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int fuse_idx, i1, j0, j1, r, k;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// current incidence index within the row

	j0 = C->j0;

	// f_vbar must be off:
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		cout << "I = " << I << " m = " << m << ", J = " << J
				<< ", n = " << n << ", i1 = " << i1
				<< ", r = " << r << ", j0 = " << j0 << endl;
		cout << "X_Fst f_vbar[i1][r]" << endl;
		exit(1);
	}

	for (; j < C->b; j++) {

		if (inc->K[j0 + j] >= K1[fuse_idx * GB->b_len + J]) {
			// this column is full
			if (f_v) {
				cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of column sum" << endl;
			}
			continue;
		}

		if (vbar[j0 + j] > i1) {
			// no vbar, skip
			if (f_v) {
				cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of vbar" << endl;
			}
			continue;
		}

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j0 + j;
		// incidence must be recorded before we call find_square




		j1 = j0 + j;
		k = inc->K[j0 + j];
		inc->theY[j0 + j][k] = i1;

		// and now come the tests:

		if (!apply_tests(I, m, J, n, j, verbose_level - 2)) {
			if (f_v) {
				cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " skipped because of test" << endl;
			}
			continue;
		}


		// the incidence passes the tests:

		// increase column sum:

		inc->K[j0 + j]++;


		// manage vbar:

		if (vbar[j0 + j] == i1) {
			if (n == 0) {
				cout << "gen_geo::X_Fst n == 0" << endl;
				exit(1);
			}
			if (inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] != j0 + j - 1) {

				// previous incidence:
				cout << "gen_geo::X_Fst theX[i1 * inc.max_r + r - 1] != j0 + j - 1" << endl;
				exit(1);
			}
			if (!f_vbar[i1 * inc->Encoding->dim_n + r - 1]) {
				cout << "gen_geo::X_Fst !f_vbar[i1 * inc->Encoding->dim_n + r - 1]" << endl;
				exit(1);
			}
			f_vbar[i1 * inc->Encoding->dim_n + r - 1] = FALSE;
			vbar[j0 + j] = MAX_V;
			// the value MAX_V indicates that there is no vbar
		}

		// create new vbar to the right:

		if (vbar[j0 + j + 1] == i1) {
			cout << "gen_geo::X_Fst vbar[j0 + j + 1] == i1" << endl;
			exit(1);
		}
		if (vbar[j0 + j + 1] > i1) {
			f_vbar[i1 * inc->Encoding->dim_n + r] = TRUE;
			vbar[j0 + j + 1] = i1;
		}

		if (hbar[i1] > J) {
			if (m == 0) {
				cout << "gen_geo::X_Fst no hbar && m == 0" << endl;
				exit(1);
			}
			if (j0 + j != inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r]) {
				// create new hbar:
				hbar[i1] = J;
			}
		}

		return TRUE;

	} // next j

	return FALSE;
}

int gen_geo::apply_tests(int I, int m, int J, int n, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fuse_idx;
	int i1, r, k, j0, j1;

	if (f_v) {
		cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// current incidence index within the row

	j0 = C->j0;

	j1 = j0 + j;
	k = inc->K[j0 + j];

	if (GB->Descr->f_lambda) {

		// another test if this was the last incidence in this column:
		// we check that there are no repeated columns !

		if (GB->Descr->f_simple) { /* JS 180100 */

			if (F_last_k_in_col[fuse_idx * GB->b_len + J] &&
					k == K1[fuse_idx * GB->b_len + J] - 1) {
				int ii;
				for (ii = 0; ii <= k; ii++) {
					if (inc->theY[j1 - 1][ii] != inc->theY[j1][ii]) {
						break; // is OK, columns differ !
					}
				}
				if (ii > k) {
					// not OK !
					inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
					inc->theY[j0 + j][k] = -1;
					if (f_v) {
						cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " rejected because not simple" << endl;
					}
					return FALSE;
				}
			}
		} // JS


		// check whether all scalar products
		// for previous rows with a one in column j0 + j
		// are less than to \lambda:
		// this is so that there is space for one more pair
		// created by the present incidence in row i1 and column j0 + j

		int ii, ii1;

		for (ii = 0; ii < k; ii++) {
			ii1 = inc->theY[j0 + j][ii];
			if (inc->pairs[i1][ii1] >= GB->Descr->lambda) {
				inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
				inc->theY[j0 + j][k] = -1;
				break;
			}
		}


		if (ii < k) {

			// there is not enough space. So, we cannot put the incidence here:

			if (f_v) {
				cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " rejected because of column sum" << endl;
			}

			return FALSE;
		}

		// increment the pairs counter:

		increment_pairs_point(i1, j0 + j, k);

		// check all scalar products for previous rows:
		if (J == GB->b_len - 1 && n == C->r - 1) {
			for (ii = 0; ii < i1; ii++) {
				if (inc->pairs[i1][ii] != GB->Descr->lambda) {
					break;
				}
			}
			if (ii < i1) {

				decrement_pairs_point(i1, j0 + j, k);

				if (f_v) {
					cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " rejected because at least one pair is not covered lambda times" << endl;
				}
				return FALSE;
			}
		}
	}
	else {
		if (GB->Descr->f_find_square) { /* JS 120100 */
			if (inc->find_square(i1, r)) {
				// cannot be placed here
				inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
				if (f_v) {
					cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " rejected because of square condition" << endl;
				}
				return FALSE;
			}
		}
	}

	girth_test_add_incidence(i1, r, j0 + j);

	if (!check_girth_condition(i1, r, j0 + j, 0 /*verbose_level*/)) {

		girth_test_delete_incidence(i1, r, j0 + j);

		if (GB->Descr->f_lambda) {

			decrement_pairs_point(i1, j0 + j, k);

		}
		if (f_v) {
			cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J << " n=" << n << " j=" << j << " rejected because of girth condition" << endl;
		}
		return FALSE;
	}



	return TRUE;
}

void gen_geo::increment_pairs_point(int i1, int col, int k)
{
	int ii, ii1;

	for (ii = 0; ii < k; ii++) {
		ii1 = inc->theY[col][ii];
		inc->pairs[i1][ii1]++;
	}
}

void gen_geo::decrement_pairs_point(int i1, int col, int k)
{
	int ii, ii1;

	for (ii = 0; ii < k; ii++) {
		ii1 = inc->theY[col][ii];
		inc->pairs[i1][ii1]--;
	}
}

void gen_geo::girth_test_add_incidence(int i, int j_idx, int j)
{
	if (Girth_test) {
		Girth_test->add_incidence(i, j_idx, j);
	}
}

void gen_geo::girth_test_delete_incidence(int i, int j_idx, int j)
{
	if (Girth_test) {
		Girth_test->delete_incidence(i, j_idx, j);
	}
}

void gen_geo::girth_Floyd(int i, int verbose_level)
{
	if (Girth_test) {
		Girth_test->Floyd(i, verbose_level);
	}
}

int gen_geo::check_girth_condition(int i, int j_idx, int j, int verbose_level)
{
	if (Girth_test) {
#if 0
		inc->print(cout, i + 1, GB->V);
		Girth_test->print_Si(i);
		Girth_test->print_Di(i);
#endif
		return Girth_test->check_girth_condition(i, j_idx, j, verbose_level);
	}
	else {
		return TRUE;
	}
}

}}



