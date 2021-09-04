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

	K = NULL;

	f_vbar = NULL;
	vbar = NULL;
	hbar = NULL;

	f_do_iso_test = FALSE;
	f_do_aut_group = FALSE;
	f_do_aut_group_in_iso_type_without_vhbars = FALSE;
	forget_ivhbar_in_last_isot = FALSE;
	gen_print_intervall = FALSE;

	//std::string inc_file_name;

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

	if (K) {
		FREE_int(K);
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
					verbose_level);
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
	init_k();
	if (f_v) {
		cout << "gen_geo::TDO_init after init_k" << endl;
	}
	if (f_v) {
		cout << "gen_geo::TDO_init before conf_init_last_non_zero_flag" << endl;
	}
	conf_init_last_non_zero_flag();
	if (f_v) {
		cout << "gen_geo::TDO_init after conf_init_last_non_zero_flag" << endl;
	}
	if (f_v) {
		cout << "gen_geo::TDO_init done" << endl;
	}
}


void gen_geo::init_tdo_line(int fuse_idx, int tdo_line, int v, int *b, int *r, int verbose_level)
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
			Conf[tdo_line * GB->b_len + j].j0 = Conf[tdo_line * GB->b_len + j - 1].j0 + Conf[tdo_line * GB->b_len + j - 1].b;
			Conf[tdo_line * GB->b_len + j].r0 = Conf[tdo_line * GB->b_len + j - 1].r0 + Conf[tdo_line * GB->b_len + j - 1].r;
		}

		if (tdo_line == 0) {
			Conf[tdo_line * GB->b_len + j].i0 = 0;
		}
		else {
			Conf[tdo_line * GB->b_len + j].i0 = Conf[(tdo_line - 1) * GB->b_len + j].i0 + Conf[(tdo_line - 1) * GB->b_len + j].v;
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
void gen_geo::init_k()
{
	int I, J, j, fuse_idx, f, l, k, s, b;

	K = NEW_int(GB->B);
	for (j = 0; j < GB->B; j++) {
		K[j] = 0;
	}

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

void gen_geo::conf_init_last_non_zero_flag()
{
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

	cout << "f_last_non_zero_in_fuse:" << endl;
	for (I = 0; I < GB->v_len; I++) {
		i = Conf[I * GB->b_len + 0].f_last_non_zero_in_fuse;
		cout << setw(3) << i << " ";
	}
	cout << endl;
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

	if (f_v) {
		cout << "gen_geo::main2, verbose_level = " << verbose_level << endl;
	}
	int t0, t1, user_time, V;

	t0 = os_ticks();

	if (f_v) {
		cout << "gen_geo::main2 before generate_all" << endl;
	}
	generate_all(verbose_level);
	if (f_v) {
		cout << "gen_geo::main2 after generate_all" << endl;
	}





	t1 = os_ticks();

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


#if 0
	/* alle Geometrien noch
	 * einmal ausgeben: */
	isot_print(it,
		FALSE /* f_with_tdo */, V, &gg->inc);
#endif
	//f_inc_fst_time_printed = TRUE;
	inc->print(cout, V, V);

	int i, a;

	for (i = 0; i < GB->Descr->print_at_line.size(); i++) {
		a = GB->Descr->print_at_line[i];
		inc->iso_type_at_line[a - 1]->print_geos(verbose_level);
	}

	// Anton Betten AB 150200:
	it = inc->iso_type_at_line[V - 1];
	cout << "generated: " << it->sum_nb_GEN << " nb_TDO = " << it->sum_nb_TDO << " nb_GEO = " << it->sum_nb_GEO << endl;

	if (inc->iso_type_no_vhbars) {
		it = inc->iso_type_no_vhbars;
		cout << "resolved: generated: " << it->nb_GEN << " nb_TDO = " << it->nb_TDO << " nb_GEO = " << it->nb_GEO << endl;
	}


	nb_GEN = inc->gl_nb_GEN;
	nb_GEO = it->nb_GEO;
	user_time = t1 - t0;
	ticks = user_time;
	tps = os_ticks_per_second();

	if (f_v) {
		cout << "gen_geo::main2 done" << endl;
	}

	char str[256];

	strcpy(str, "Running time : ");
	print_delta_time(user_time, str);
	printf("%s\n", str);

}

void gen_geo::generate_all(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::generate_all" << endl;
	}

	int ret = FALSE;
	int already_there;
	int s_nb_i_vbar, s_nb_i_hbar;
	iso_type *it0, *it1;

	if (f_v) {
		cout << "gen_geo::generate_all this = " << this << endl;
		cout << "gen_geo::generate_all inc = " << inc << endl;
		if (inc->f_lambda) {
			cout << "lambda = " << inc->lambda << endl;
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
		/* printf("*** do_geo *** geometry no. %d\n", gg->inc.gl_nb_GEN); */
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
			&already_there,
			0 /*verbose_level*/);

		if (FALSE) {
			cout << "gen_geo::generate_all after isot_add for it0" << endl;
		}

		if (it0->Canonical_forms->B.size() % 1 == 0) {
			cout << it0->Canonical_forms->B.size() << endl;
			inc->print(cout, inc->Encoding->v, inc->Encoding->v);
		}

		if (forget_ivhbar_in_last_isot) {
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!already_there && inc->iso_type_no_vhbars) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;

			if (FALSE) {
				cout << "gen_geo::generate_all before isot_add for it1" << endl;
			}
			it1->add_geometry(inc->Encoding,
					inc->Encoding->v, inc,
					&already_there,
					0 /*verbose_level*/);

			if (FALSE) {
				cout << "gen_geo::generate_all after isot_add for it1" << endl;
			}
			if (!already_there) {
				//save_theX(inc->iso_type_no_vhbars->fp);
			}
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!already_there && !inc->iso_type_no_vhbars) {
			// if (GEO_fp) {
			// 	geo_save_into_file(gg, GEO_fp);
			//	}
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
		/* I-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige weiterzaehlen/zurueckbauen. */
		while (TRUE) {
			if (I == 0) {
				return FALSE;
			}
			I--;
			if (GeoRowNxt(I, verbose_level)) {
				break;
			}
		}
		/* I-tes Element wurde gerade erhoeht.
		 * I == II - 1 ist moeglich.
		 * Dann sind wir fertig.
		 * Nachfolgende Elemente initialisieren. */
		I++;
	}
}

int gen_geo::GeoNxt(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "GeoNxt" << endl;
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
		/* I-tes Element wurde gerade erhoeht. */
		while (TRUE) {
			if (I >= GB->v_len - 1) {
				return TRUE;
			}
			I++;
			if (!GeoRowFst(I, verbose_level)) {
				break;
			}
		}
		/* I-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige zurueckbauen. */
		I--;
	}
}

int gen_geo::GeoRowFst(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len;

	if (f_v) {
		cout << "GeoRowFst I=" << I << endl;
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
		/* m-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige weiterzaehlen/zurueckbauen. */
		while (TRUE) {
			if (m == 0) {
				return FALSE;
			}
			m--;
			if (GeoLineNxtRange(I, m, verbose_level)) {
				break;
			}
		}
		/* m-tes Element wurde gerade erhoeht.
		 * m == v - 1 ist moeglich.
		 * Dann sind wir fertig.
		 * Nachfolgende Elemente initialisieren. */
		m++;
	}
}

int gen_geo::GeoRowNxt(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Conf + I * GB->b_len;

	if (f_v) {
		cout << "GeoRowNxt I=" << I << endl;
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
		/* m-tes Element wurde gerade erhoeht. */
		while (TRUE) {
			if (m >= C->v - 1) {
				return TRUE;
			}
			m++;
			if (!GeoLineFstRange(I, m, verbose_level)) {
				break;
			}
		}
		/* m-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige zurueckbauen. */
		m--;
	}
}

#define GEO_LINE_RANGE

int gen_geo::GeoLineFstRange(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "GeoLineFstRange I=" << I << " m=" << m << endl;
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
				cout << "GeoLineFstRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
				if (!GeoLineNxt0(I, m, verbose_level)) {
					return FALSE;
				}
			}
		}
		cout << "GeoLineFstRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
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
		cout << "GeoLineNxtRange I=" << I << " m=" << m << endl;
	}
#ifdef GEO_LINE_RANGE
	iso_type *it;
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i1;

	i1 = C->i0 + m;
	it = inc->iso_type_at_line[i1];
	if (it && it->f_range) {
		if (it->nb_GEO == it->range_first + it->range_len - 1) {
			cout << "GeoLineNxtRange: line " << i1 + 1 << " case end at " << it->range_first + it->range_len <<  " reached" << endl;
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
			cout << "GeoLineNxtRange: line " << i1 + 1 << " case " << it->nb_GEO << endl;
			if (!GeoLineNxt0(I, m, verbose_level)) {
				return FALSE;
			}
		}
		printf("\nGeoLineNxtRange(): "
			"line %d case %d:\n",
			i1 + 1, it->nb_GEO); fflush(stdout);
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
		cout << "geo_back_test I=" << I << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int i0, i1, m, already_there, control_line;
	iso_type *it;

	i0 = C->i0;
	control_line = i0 + C->v - 1;
	for (m = 0; m < C->v - 1; m++) {
		i1 = i0 + m;
		it = inc->iso_type_at_line[i1];
		if (it && it->f_generate_first && !it->f_beginning_checked) {
			/* Der Stand der i1-ten Zeile
			 * ist realisierbar
			 * (bis control line),
			 * wir haben jetzt eine Realisierung
			 * vor uns. */
			/* Nur falls ein Ansatz realisierbar
			 * ist, soll getestet werden,
			 * ob dieser Anfang bereits
			 * einmal da war (Stuetze 2-ter Art);
			 * dies geschieht nun: */
			/* nb_GEO[i1] == nb_GEO
			 * zeigt an, dass die i1-te
			 * Zeile noch nicht
			 * getestet worden ist: */
			/* printf("geo_back_test "
			"for line %d:\n", (i1 + 1)); */
			/* printf("geo_back_test() vor isot_add()\n"); */

			it->add_geometry(inc->Encoding,
					i1 + 1, inc,
					&already_there,
					0 /*verbose_level*/);

			/* printf("geo_back_test() nach isot_add()\n"); */
			if (!already_there) {
				/* wir sind in einem neuen Zweig */
				/* setze nb_GEO auf unmoeglichen Wert,
				 * um anzuzeigen, dass das
				 * Anfangsstueck jetzt nicht
				 * mehr getestet werden muss: */
				/* gg->inc.nb_GEO[i1] = -1; */
				it->f_beginning_checked = TRUE;
				//inc->flush(m);
				continue;
			}
			inc->back_to_line = i1;
			/* printf("going back to line %d\n", (gg->inc.back_to_line + 1)); */
			return FALSE;
		}
	}
	return TRUE;
}


int gen_geo::GeoLineFst0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "GeoLineFst0 I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!GeoLineFst(I, m)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		/* Stuetze 2-ter Art:
		 * es wird jetzt NICHT geprueft,
		 * ob der Ansatz bereits einmal da war,
		 * sondern erst in der control line,
		 * wenn der Ansatz bis dahin
		 * realisierbar ist. Setze jetzt
		 * nb_GEO[i1] auf nb_GEO der control line,
		 * um anzuzeigen, dass dieses
		 * Anfangsstueck noch nicht
		 * getestet wurde: */
		it->f_beginning_checked = FALSE;
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!GeoLineNxt(I, m)) {
				return FALSE;
			}
			/* Fehler: wenn back_test
			 * FALSE ist, so muss
			 * zurueckgegangen werden
			 * (back_to_line ist gesetzt).
			 * In diesem Falle muss
			 * GeoLineNxt() FALSE geben. */
			cout << "GeoLineFst0(): back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		/* back test ueberstanden,
		 * jetzt noch einen Test erster Art: */
	}
	if (i1 == inc->Encoding->v - 1) {
		/* a full geometry;
		 * let the main routine add it: */
		return TRUE;
	}
	if (it) {
		/* Stuetze erster Art: */
		while (TRUE) {
			if (f_v) {
				cout << "GeoLineFst0 I=" << I << " m=" << m << " before isot_add" << endl;
				inc->print(cout, i1 + 1, i1 + 1);
			}

			it->add_geometry(inc->Encoding,
					i1 + 1, inc, &already_there,
					0 /*verbose_level*/);

			if (f_v) {
				cout << "GeoLineFst0 I=" << I << " m=" << m << " after isot_add" << endl;
			}
			if (it->f_print_mod) {
				if ((it->nb_GEN % it->print_mod) == 0) {
					//inc->print(cout, i1 + 1);
					// geo_print_pairs(gg, i1 + 1);
				}
			}
			if (!already_there) {
				//inc->flush(i1);
				break;
			}
			if (!GeoLineNxt(I, m)) {
				return FALSE;
			}
		}
		/* now: a new geometry is reached
		 * (not already_there). */
	}
	return TRUE;
}

int gen_geo::GeoLineNxt0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "GeoLineNxt0 I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = Conf + I * GB->b_len;
	int already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!GeoLineNxt(I, m)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		/* Anfangsstueck muss noch geprueft
		 * werden, falls realisierbar: */
		it->f_beginning_checked = FALSE;
#if 0
		gg->inc.nb_GEO[i1] = ((ISO_TYPE *)
		gg->inc.iso_type[control_line])->nb_GEO;
#endif
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!GeoLineNxt(I, m)) {
				return FALSE;
			}
			/* Fehler: wenn back_test
			 * FALSE ist, so muss
			 * zurueckgegangen werden
			 * (back_to_line ist gesetzt).
			 * In diesem Falle muss
			 * GeoLineNxt() FALSE geben. */
			cout << "GeoLineNxt0 back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		/* back test ueberstanden,
		 * jetzt noch einen Test erster Art: */
	}
	if (i1 == inc->Encoding->v - 1) {
		/* a full geometry;
		 * let the main routine add it: */
		return TRUE;
	}
	if (it) {
		while (TRUE) {
			it->add_geometry(inc->Encoding,
				i1 + 1, inc, &already_there,
				0 /*verbose_level*/);

#if 0
			if (it->f_print_mod) {
				if ((it->nb_GEN % it->print_mod) == 0) {
					//inc->print(cout, i1 + 1);
					// geo_print_pairs(gg, i1 + 1);
				}
			}
#endif

			if (!already_there) {
				//inc->flush(i1);
				break;
			}
			if (!GeoLineNxt(I, m)) {
				return FALSE;
			}
		}
		/* now: a new geometry is reached
		 * (not already_there). */
	}
	return TRUE;
}

int gen_geo::GeoLineFst(int I, int m)
{
	int J;

	J = 0;
	while (TRUE) {
		while (TRUE) {
			if (J >= GB->b_len) {
				return TRUE;
			}
			if (!GeoConfFst(I, m, J)) {
				break;
			}
			J++;
		}
		/* J-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige weiterzaehlen/zurueckbauen. */
		while (TRUE) {
			if (J == 0) {
				return FALSE;
			}
			J--;
			if (GeoConfNxt(I, m, J)) {
				break;
			}
		}
		/* J-tes Element wurde gerade erhoeht.
		 * J == JJ - 1 ist moeglich.
		 * Dann sind wir fertig.
		 * Nachfolgende Elemente initialisieren. */
		J++;
	}
}

int gen_geo::GeoLineNxt(int I, int m)
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
			if (GeoConfNxt(I, m, J)) {
				break;
			}
			if (J == 0) {
				return FALSE;
			}
			J--;
		}
		/* J-tes Element wurde gerade erhoeht. */
		while (TRUE) {
			if (J >= GB->b_len - 1) {
				return TRUE;
			}
			J++;
			if (!GeoConfFst(I, m, J)) {
				break;
			}
		}
		/* J-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige zurueckbauen. */
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

int gen_geo::GeoConfFst(int I, int m, int J)
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int n, i1;

	if (J == 0) {
		i1 = C->i0 + m;
		if (m == 0) {
			hbar[i1] = -1;
			/* initialer hbar */
		}
		else {
			hbar[i1] = GB->b_len;
			/* kein hbar */
		}
	}
	n = 0;
	while (TRUE) {
		while (TRUE) {
			if (n >= C->r) {
				return TRUE;
			}
			if (!GeoXFst(I, m, J, n)) {
				break;
			}
			n++;
		}
		/* n-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige weiterzaehlen/zurueckbauen. */
		while (TRUE) {
			if (n == 0) {
				return FALSE;
			}
			n--;
			if (GeoXNxt(I, m, J, n)) {
				break;
			}
		}
		/* n-tes Element wurde gerade erhoeht.
		 * n == r - 1 ist moeglich.
		 * Dann sind wir fertig.
		 * Nachfolgende Elemente initialisieren. */
		n++;
	}
}

int gen_geo::GeoConfNxt(int I, int m, int J)
{
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
			if (GeoXNxt(I, m, J, n)) {
				break;
			}
			if (n == 0) {
				return FALSE;
			}
			n--;
		}
		/* n-tes Element wurde gerade erhoeht. */
		while (TRUE) {
			if (n >= C->r - 1) {
				return TRUE;
			}
			n++;
			if (!GeoXFst(I, m, J, n)) {
				break;
			}
		}
		/* n-tes Element konnte sich
		 * in dieser Situation nicht initialisieren:
		 * Vorige zurueckbauen. */
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

int gen_geo::GeoXFst(int I, int m, int J, int n)
/* Verwaltet: hbar[], vbar[],
 * f_vbar[][], theX[][], K[] */
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	//int f_new_situation1;
	int i1, j0, r, j;

	i1 = C->i0 + m; /* aktuelle Zeile */
	r = C->r0 + n; /* aktuelles Kreuz */
	j0 = C->j0;
	if (hbar[i1] <= J) {
		/* hbar existiert,
		 * der Anfang der Zeile ist
		 * anders als der der letzten Zeile. */
		/* versuche linksbuendig zu setzen: */
		/* rechts vom Vorgaengerkreuz: */
		/* (j ist lokale Koordinate) */
		if (n == 0) {
			/* erstes Kreuz des Kaestchens ? */
			j = 0;
		}
		else {
			j = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] - j0 + 1;
		}
	}
	else {
		if (m == 0) {
			cout << "GeoXFst hbar[i1] > J && m == 0" << endl;
			exit(1);
		}
		/* kein hbar,
		 * der Zeilenanfang stimmt mit dem
		 * der letzten Zeile ueberein. */
		/* versuche, den Stand der letzten Zeile
		 * zu uebernehmen: */
		j = inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r] - j0;
	}
	return X_Fst(I, m, J, n, j);
}

int gen_geo::GeoXNxt(int I, int m, int J, int n)
/* Verwaltet: hbar[], vbar[],
 * f_vbar[][], theX[][], K[] */
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int old_x;
	int fuse_idx, i1, j0, j1, r, j, k, ii, ii1;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m; /* aktuelle Zeile */
	r = C->r0 + n; /* aktuelles Kreuz */
	j0 = C->j0;
	old_x = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r];
	// printf("XNxt() line = %d n=%d old_x = %d ", i1, n, old_x); fflush(stdout);
	K[old_x]--;
	if (inc->f_lambda) {
		k = K[old_x];
		inc->theY[old_x * inc->Encoding->dim_n + k] = -1;
		for (ii = 0; ii < k; ii++) {
			ii1 = inc->theY[old_x * inc->Encoding->dim_n + ii];
			if (inc->pairs[i1][ii1] <= 0) {
				cout << "GeoXClear pairs[i1][ii1] <= 0" << endl;
				exit(1);
			}
			inc->pairs[i1][ii1]--;
		}
	}

	/* alten vbar des Kreuzchens austragen: */
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		vbar[old_x + 1] = MAX_V;
		f_vbar[i1 * inc->Encoding->dim_n + r] = FALSE;
	}
	/* evtl. neuen vbar
	 * des linken Nachbarkreuzchens
	 * generieren: */
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

		if (K[j0 + j] >= K1[fuse_idx * GB->b_len + J]) {
			/* Spalte bereits voll */
			continue;
		}

		if (vbar[j0 + j] > i1) {
			continue;
			/* weil die Spalte
			 * sich bislang noch nicht
			 * von ihrer linken
			 * Nachbarspalte unterscheidet,
			 * setzen wir OE alle
			 * Kreuzchen linksbuendig. */
		}

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j0 + j;
			/* muss vor find_square gesetzt sein */

		k = K[j0 + j];
		inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = i1;
		j1 = j0 + j;
		if (inc->f_lambda) {

			// another test if this was the last X in this column:
			// we check that there are no repeated columns !
			if (inc->f_simple == TRUE) { /* JS 180100 */
				if (F_last_k_in_col[fuse_idx * GB->b_len + J] && k == K1[fuse_idx * GB->b_len + J] - 1) {
					for (ii = 0; ii <= k; ii++) {
						if (inc->theY[(j1 - 1) * inc->Encoding->dim_n + ii] != inc->theY[j1 * inc->Encoding->dim_n + ii]) {
							break; // is OK, columns differ !
						}
					}
					if (ii > k) {
						// not OK !
						inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
						inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = -1;
						continue;
					}
				}
			}

			// check if each scalarproduct
			// with a previous row is $\le \lambda$
			for (ii = 0; ii < k; ii++) {
				ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
				if (inc->pairs[i1][ii1] >= inc->lambda) {
					inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
					inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = -1;
					break;
				}
			}
			if (ii < k) {
				continue;
			}
			for (ii = 0; ii < k; ii++) {
				ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
				inc->pairs[i1][ii1]++;
			}

			// check scalarproduct for all previous rows:
			if (J == GB->b_len - 1 && n == C->r - 1) {
				for (ii = 0; ii < i1; ii++) {
					if (inc->pairs[i1][ii] != inc->lambda) {
						break;
					}
				}
				if (ii < i1) {
					for (ii = 0; ii < k; ii++) {
						ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
						inc->pairs[i1][ii1]--;
					}
					continue;
				}
			}
		}
		else {
		  if (inc->f_find_square) { /* JS 120100 */
			if (inc->find_square(i1, r)) {
				/* kann hier nicht plazieren: */
				inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
				continue;
				}
			}
		}
		/* Spaltensumme erhoehen: */
		K[j0 + j]++;

		/* neuen vbar rechts
		 * vom Kreuzchen erzeugen: */
		if (vbar[j0 + j + 1] == i1) {
			cout << "XNxt vbar[j0 + j + 1] == i1" << endl;
			exit(1);
		}
		if (vbar[j0 + j + 1] > i1) {
			f_vbar[i1 * inc->Encoding->dim_n + r] = TRUE;
			vbar[j0 + j + 1] = i1;
		}

		if (hbar[i1] > J) {
			/* hbar erzeugen ? */
			/* bislang noch kein hbar,
			 * also stimmt der Anfang der Zeile
			 * mit dem der
			 * Vorgaengerzeile ueberein. */
			if (m == 0) {
				cout << "GeoXNxt no hbar && m == 0" << endl;
				exit(1);
			}
			if (j0 + j != inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r]) {
				/* neuen hbar anlegen: */
				hbar[i1] = J;
			}
		}

#if 0
		printf("OK at = %d\n", j0 + j);
			incidence_print(&gg->inc,
			(int *)gg->inc.theX, TRUE, i1 + 1);
#endif
		return TRUE;
	}
	// printf("not OK\n");
	return FALSE;
}

void gen_geo::GeoXClear(int I, int m, int J, int n)
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int old_x;
	int i1, j0, r, k, ii, ii1;

	i1 = C->i0 + m;
		/* aktuelle Zeile */
	r = C->r0 + n;
		/* aktuelles Kreuz */
	j0 = C->j0;
	old_x = inc->Encoding->theX[i1 * inc->Encoding->dim_n + r];
	K[old_x]--;
	/* alten vbar
	 * des Kreuzchens austragen: */
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		vbar[old_x + 1] = MAX_V;
		f_vbar[i1 * inc->Encoding->dim_n + r] = FALSE;
	}
	/* evtl. neuen vbar
	 * des linken Nachbarkreuzchens
	 * generieren: */
	if (n > 0) {
		if (vbar[old_x] > i1 &&
			inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] == old_x - 1) {
			vbar[old_x] = i1;
			f_vbar[i1 * inc->Encoding->dim_n + r - 1] = TRUE;
		}
	}
	inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
	if (inc->f_lambda) {
		k = K[old_x];
		inc->theY[old_x * inc->Encoding->dim_n + k] = -1;
		for (ii = 0; ii < k; ii++) {
			ii1 = inc->theY[old_x * inc->Encoding->dim_n + ii];
			if (inc->pairs[i1][ii1] <= 0) {
				cout << "GeoXClear pairs[i1][ii1] <= 0" << endl;
				exit(1);
			}
			inc->pairs[i1][ii1]--;
		}
	}
}

int gen_geo::X_Fst(int I, int m, int J, int n, int j)
/* Versuche Kreuz zu plazieren,
 * beginne an Position j. */
/* (j ist lokale Koordinate) */
/* Verwaltet: hbar[], vbar[],
 * f_vbar[][], theX[][], K[] */
{
	gen_geo_conf *C = Conf + I * GB->b_len + J;
	int fuse_idx, i1, j0, j1, r, k, ii, ii1;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m; /* aktuelle Zeile */
	// printf("X_Fst() line = %d j = %d\n", i1, j); fflush(stdout);
	r = C->r0 + n; /* aktuelles Kreuz */
	j0 = C->j0;

	/* f_vbar muss zu Beginn
	 * ausgetragen sein: */
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		cout << "I = " << I << " m = " << m << ", J = " << J
				<< ", n = " << n << ", i1 = " << i1
				<< ", r = " << r << ", j0 = " << j0 << endl;
		cout << "X_Fst f_vbar[i1][r]" << endl;
		exit(1);
	}

	for (; j < C->b; j++) {

		if (K[j0 + j] >= K1[fuse_idx * GB->b_len + J]) {
			/* Spalte bereits voll */
			continue;
		}

		if (vbar[j0 + j] > i1) {
			/* kein vbar,
			 * hier wird nicht plaziert: */
			continue;
		}
		/* vbar[j0 + j] < i1:
		 *    die Spalte muss verwendet werden,
		 *    da sie sich in den oberen Eintraegen
		 *    von ihrer linken
		 *    Nachbarspalte unterscheidet
		 *  ==  i1: Anbau an linke Nachbarkreuze. */

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j0 + j;
		/* muss vor find_square gesetzt sein */

		j1 = j0 + j;
		k = K[j0 + j];
		inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = i1;
		if (inc->f_lambda) {

			// another test if this was the last X in this column:
			// we check that there are no repeated columns !
			if (inc->f_simple) { /* JS 180100 */
				if (F_last_k_in_col[fuse_idx * GB->b_len + J] && k == K1[fuse_idx * GB->b_len + J] - 1) {
					for (ii = 0; ii <= k; ii++) {
						if (inc->theY[(j1 - 1) * inc->Encoding->dim_n + ii] != inc->theY[j1 * inc->Encoding->dim_n + ii]) {
							break; // is OK, columns differ !
						}
					}
					if (ii > k) {
						// not OK !
						inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
						inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = -1;
						continue;
					}
				}
			} // JS

			// check if each scalarproduct
			// with a previous row is $\le \lambda$
			for (ii = 0; ii < k; ii++) {
				ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
				if (inc->pairs[i1][ii1] >= inc->lambda) {
					inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
					inc->theY[(j0 + j) * inc->Encoding->dim_n + k] = -1;
					break;
				}
			}
			if (ii < k) {
				continue;
			}
			for (ii = 0; ii < k; ii++) {
				ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
				inc->pairs[i1][ii1]++;
			}
			// check scalarproduct for all previous rows:
			if (J == GB->b_len - 1 && n == C->r - 1) {
				for (ii = 0; ii < i1; ii++) {
					if (inc->pairs[i1][ii] != inc->lambda) {
						break;
					}
				}
				if (ii < i1) {
					for (ii = 0; ii < k; ii++) {
						ii1 = inc->theY[(j0 + j) * inc->Encoding->dim_n + ii];
						inc->pairs[i1][ii1]--;
					}
					continue;
				}
			}
		}
		else {
			if (inc->f_find_square) { /* JS 120100 */
				if (inc->find_square(i1, r)) {
					/* kann hier nicht plazieren: */
					inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
					continue;
				}
			}
		}
		/* Spaltensumme erhoehen: */
		K[j0 + j]++;

		/* vbar verwalten: */
		if (vbar[j0 + j] == i1) {
			/* ein vbar, der durch das
			 * linke Nachbarkreuz
			 * erzeugt wurde; austragen: */
			if (n == 0) {
				/* vom TDO Schema
				 * herruehrende vbars
				 * haben die Kennung -1
				 * (und nicht i1). */
				cout << "X_Fst(): n == 0" << endl;
				exit(1);
			}
			if (inc->Encoding->theX[i1 * inc->Encoding->dim_n + r - 1] != j0 + j - 1) {
				/* Vorgaengerkreuz */
				cout << "X_Fst theX[i1 * inc.max_r + r - 1] != j0 + j - 1" << endl;
				exit(1);
			}
			if (!f_vbar[i1 * inc->Encoding->dim_n + r - 1]) {
				cout << "X_Fst !f_vbar[i1 * inc->Encoding->dim_n + r - 1]" << endl;
				exit(1);
			}
			f_vbar[i1 * inc->Encoding->dim_n + r - 1] = FALSE;
			vbar[j0 + j] = MAX_V;
			/* MAX_V heisst: kein vbar hier. */
		}
		/* neuen vbar rechts
		 * vom Kreuzchen generieren: */
		if (vbar[j0 + j + 1] == i1) {
			cout << "X_Fst vbar[j0 + j + 1] == i1" << endl;
			exit(1);
		}
		if (vbar[j0 + j + 1] > i1) {
			/* noch kein vbar rechts ? */
			/* es war bislang noch kein vbar da,
			 * wir legen einen an: */
			/* Beachte: vbars werden
			 * immer in der rechten Nachbarspalte
			 * angelegt, da sie sich
			 * auf die linke Seite
			 * einer Spalte beziehen. */
			f_vbar[i1 * inc->Encoding->dim_n + r] = TRUE;
			vbar[j0 + j + 1] = i1;
		}

		if (hbar[i1] > J) {
			/* hbar erzeugen ? */
			/* bislang noch kein hbar,
			 * also stimmt der Anfang der Zeile
			 * mit dem der Vorgaengerzeile ueberein. */
			if (m == 0) {
				cout << "X_Fst no hbar && m == 0" << endl;
				exit(1);
			}
			if (j0 + j != inc->Encoding->theX[(i1 - 1) * inc->Encoding->dim_n + r]) {
				/* neuen hbar anlegen: */
				hbar[i1] = J;
			}
		}

		// printf("OK at %d\n", j0 + j);
		return TRUE;

	} // next j

	return FALSE;
}



}}



