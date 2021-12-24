/*
 * iso_type.cpp
 *
 *  Created on: Aug 15, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


#define MAX_GEO 100
#define MAX_TDO 16



iso_type::iso_type()
{
	v = 0;
	sum_R = 0;
	inc = NULL;

	f_orderly = FALSE;

	f_transpose_it = FALSE;
	f_snd_TDO = FALSE;
	f_ddp = FALSE;
	f_ddb = FALSE;

	f_generate_first = FALSE;
	f_beginning_checked = FALSE;

	f_split = FALSE;
	split_remainder = 0;
	split_modulo = 1;


	f_flush_line = FALSE;

	//std::string fname;

	sum_nb_GEN = 0;
	sum_nb_GEO = 0;
	sum_nb_TDO = 0;
	nb_GEN = 0;
	nb_GEO = 0;
	nb_TDO = 0;
	dim_GEO = 0;
	dim_TDO = 0;
	theGEO1 = NULL;
	theGEO2 = NULL;
	GEO_TDO_idx = NULL;
	//theTDO = NULL;

	Canonical_forms = NULL;

	f_print_mod = TRUE;
	print_mod = 1;

}

iso_type::~iso_type()
{
	int i;

#if 0
	if (dim_TDO) {
		for (i = 0; i < nb_TDO; i++) {
			delete theTDO[i];
			}
		delete [] theTDO;
		theTDO = NULL;
	}
#endif
	if (dim_GEO) {
		delete [] GEO_TDO_idx;
		for (i = 0; i < nb_GEO; i++) {
			FREE_int(theGEO1[i]);
			FREE_int(theGEO2[i]);
		}
		delete [] theGEO1;
		delete [] theGEO2;
	}
}

void iso_type::init(int v, incidence *inc, int tdo_flags, int f_orderly, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::init v=" << v << " tdo_flags=" << tdo_flags << endl;
	}
	int i;

	scan_tdo_flags(tdo_flags);

	iso_type::v = v;
	iso_type::inc = inc;
	iso_type::f_orderly = f_orderly;

	sum_R = 0;
	for (i = 0; i < v; i++) {
		sum_R += inc->Encoding->R[i];
	}

	if (f_v) {
		cout << "iso_type::init v=" << v << " sum_R=" << sum_R << endl;
	}
	sum_nb_GEN = 0;
	sum_nb_GEO = 0;
	sum_nb_TDO = 0;
	nb_GEN = 0;
	nb_GEO = 0;
	nb_TDO = 0;
	if (f_v) {
		cout << "iso_type::init v=" << v << " before init2" << endl;
	}
	init2();
	if (f_v) {
		cout << "iso_type::init v=" << v << " after init2" << endl;
	}

	Canonical_forms = NEW_OBJECT(classify_using_canonical_forms);

	if (f_v) {
		cout << "iso_type::init done" << endl;
	}
}

void iso_type::init2()
{
	int i;

	nb_GEN = 0;
	nb_GEO = 0;
	nb_TDO = 0;
	dim_GEO = MAX_GEO;
	dim_TDO = MAX_TDO;

	theGEO1 = new pint[dim_GEO];
	theGEO2 = new pint[dim_GEO];
	GEO_TDO_idx = new int[dim_GEO];

	//theTDO = new ptdo_scheme [dim_TDO];

	for (i = 0; i < dim_GEO; i++) {
		theGEO1[i] = NULL;
		theGEO2[i] = NULL;
		GEO_TDO_idx[i] = -1;
	}
#if 0
	for (i = 0; i < dim_TDO; i++) {
		theTDO[i] = NULL;
	}
#endif
}

#if 0
int iso_type::find_geometry(
	inc_encoding *Encoding,
	int v, incidence *inc,
	int verbose_level)
// returns index of the geometry if found, otherwise -1.
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::find_geometry" << endl;
	}

	if (f_v) {
		cout << "iso_type::find_geometry v=" << v << " geo: ";
		inc->print_geo(cout, v, Encoding->theX);
		cout << endl;
	}

	int *theY = NULL;
	tdo_scheme *tdos = NULL;
	int tdo_idx, geo_idx;
	int f_found;

	if (f_v) {
		cout << "iso_type::find_geometry before calc_theY_and_tdos_override_v" << endl;
	}
	calc_theY_and_tdos_override_v(Encoding, inc, v, theY, tdos, verbose_level - 5);
	if (f_v) {
		cout << "iso_type::find_geometry after calc_theY_and_tdos_override_v" << endl;
		tdos->print();
	}

	if (f_v) {
		cout << "iso_type::find_geometry before find_tdos" << endl;
	}
	find_tdos(tdos, &tdo_idx, &f_found);
	if (f_v) {
		cout << "iso_type::find_geometry after find_tdos" << endl;
	}

	if (!f_found) {
		// new TDO implies new isomorphism type
		geo_idx = -1;
	}
	else {
		if (f_v) {
			cout << "iso_type::find_geometry before find_geo" << endl;
		}
		geo_idx = find_geo(v, inc, tdos, theY, tdo_idx, verbose_level);
		if (f_v) {
			cout << "iso_type::find_geometry after find_geo" << endl;
		}
	}
	if (theY) {
		delete [] theY;
	}
	if (f_v) {
		cout << "iso_type::find_geometry done" << endl;
	}
	return geo_idx;
}
#endif

void iso_type::add_geometry(
	inc_encoding *Encoding,
	int v, incidence *inc,
	int &f_already_there,
	int verbose_level)
{

	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << endl;

		//inc->print(cout, v);
		//print_geometry(Encoding, v, inc);

	}


	static long int count = 0;

	if (((1L << 18) - 1 & count) == 0) {
		cout << "iso_type::add_geometry v=" << v << " count = " << count << endl;

		iso_type *it;
		int V = inc->gg->GB->V;

		it = inc->iso_type_at_line[V - 1];

		inc->print(cout, V, v);

	}

	count++;

#if 0

#if 0
	if (v == 4) {
		verbose_level = 2;
	}
#endif

	int *theY = NULL;
	tdo_scheme *tdos = NULL;
	int tdo_idx, geo_idx;
	int f_found;
	int status = 0;

	nb_GEN++;
	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " nb_GEN=" << nb_GEN << endl;
	}

	if (f_v) {
		cout << "iso_type::add_geometry before calc_theY_and_tdos_override_v" << endl;
	}

	int vl;
	if (v == 10) {
		vl = 3;
	}
	else {
		vl = 0;
	}
	calc_theY_and_tdos_override_v(Encoding, inc, v, theY, tdos, vl);

	if (f_v) {
		cout << "iso_type::add_geometry after calc_theY_and_tdos_override_v" << endl;
	}
	if (f_snd_TDO) {
		cout << "iso_type::calc_theY_and_tdos_override_v the second tdo scheme is:" << endl;
		tdos->print();
	}

	if (f_v) {
		cout << "iso_type::add_geometry the tdo scheme is " << endl;
		tdos->print();
	}

	if (f_v) {
		cout << "iso_type::add_geometry before isot_find_tdos" << endl;
	}
	find_tdos(tdos, &tdo_idx, &f_found);

	if (f_v) {
		cout << "iso_type::add_geometry after isot_find_tdos" << endl;
	}
	if (!f_found) {
		// new TDO implies new isomorphism type
		*already_there = FALSE;

		if (f_v) {
			cout << "iso_type::add_geometry before add_tdos_and_geo" << endl;
		}
		add_tdos_and_geo(tdos, tdo_idx, Encoding->theX, theY, verbose_level);
		status = 1;
		goto l_exit;
	}

	//int f_new_object;

	if (!f_do_iso_test) {
		cout << "iso_type::add_geometry warning: no iso test" << endl;
		geo_idx = -1;
	}
	else {
		if (f_v) {
			cout << "iso_type::add_geometry before isot_find_geo" << endl;
		}
		geo_idx = find_geo(v, inc, tdos, theY, tdo_idx, verbose_level);
		if (f_v) {
			cout << "iso_type::add_geometry after isot_find_geo" << endl;
		}
	}

	if (geo_idx >= 0) {
		*already_there = TRUE;
		status = 2;
		goto l_exit;
	}
	*already_there = FALSE;
	status = 3;

	if (f_v) {
		cout << "iso_type::add_geometry before isot_add_geo" << endl;
	}

	int *theX;

	theX = NEW_int(v * Encoding->dim_n);
	Orbiter->Int_vec.copy(Encoding->theX, theX, v * Encoding->dim_n);

	add_geo(tdo_idx, theX, theY);
	if (f_v) {
		cout << "iso_type::add_geometry after isot_add_geo" << endl;
	}
	theX = NULL;
	theY = NULL;

#if 0
calc_aut:
	if (f_print_isot_small || f_print_isot || f_v) {
		if (!f_found) {
			cout << endl << "new TDO at line " << v << " : ";
		}
		else {
			cout << endl << "new GEO at line " << v << " : ";
		}
		print_status(cout, FALSE /* f_with_flags */);
		cout << "  ";
	}
	if (f_print_isot) {
		cout << endl;
		inc->print(cout, v);
	}
	if (f_aut_group) {
		geo_idx = nb_GEO - 1;
		if (f_v) {
			cout << "iso_type::add_geometry before recalc_autgroup" << endl;
		}
		recalc_autgroup(v, inc, tdo_idx, geo_idx, f_print_isot_small, f_print_isot, verbose_level);
		if (f_v) {
			cout << "iso_type::add_geometry after recalc_autgroup" << endl;
		}
	}
#endif

l_exit:
	if (f_v) {
		cout << endl;
		cout << "iso_type::add_geometry v=" << v << " count=" << count << " status = " << status << " geo: ";
		inc->print_geo(cout, v, Encoding->theX);
		cout << endl;
	}
	count++;
#else
	int f_new_object;

	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " before find_and_add_geo" << endl;
	}
	find_and_add_geo(
		v, inc,
		inc->Encoding->theX, f_new_object, verbose_level - 1);
	if (f_v) {
		cout << "iso_type::add_geometry v=" << v << " after find_and_add_geo" << endl;
	}

	if (f_new_object) {
		f_already_there = FALSE;
	}
	else {
		f_already_there = TRUE;
	}

#endif
	if (f_v) {
		cout << "iso_type::add_geometry done" << endl;
	}
}

#if 0
void iso_type::recalc_autgroup(
	int v, incidence *inc,
	int tdo_idx, int geo_idx,
	int f_print_isot_small,
	int f_print_isot, int verbose_level)
{
	int aut_group_order;

	do_aut_group(v, inc,
		theTDO[tdo_idx],
		theGEO2[geo_idx],
		&aut_group_order,
		f_print_isot_small, f_print_isot, verbose_level);

	//set_aut_group_order(theGEO[geo_idx], aut_group_order);
}

void iso_type::calc_theY_and_tdos_override_v(
	inc_encoding *Encoding, incidence *inc, int v,
	int *&theY, tdo_scheme *&tdos, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v" << endl;
	}
	//int *pc1;
	short *ddp = NULL, *ddb = NULL;
	cperm tdo_p, tdo_q;
	//int *theY; // [MAX_V * MAX_R]

	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v before geo_calc_tdos" << endl;
	}
	tdos = geo_calc_tdos(
		Encoding, inc, v,
		ddp, ddb, &tdo_p, &tdo_q,
		verbose_level);

	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v after geo_calc_tdos" << endl;
		cout << "iso_type::calc_theY_and_tdos_override_v the tdo scheme is:" << endl;
		tdos->print();
	}

	theY = new int [v * Encoding->dim_n];

	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v before Encoding->apply_permutation" << endl;
	}
#if 0
	Orbiter->Int_vec.copy(Encoding->theX, theY, v * Encoding->dim_n);

#else
	Encoding->apply_permutation(inc, v, theY, &tdo_p, &tdo_q, verbose_level);
	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v theX=" << endl;
		Orbiter->Int_vec.print(cout, Encoding->theX, v * Encoding->dim_n);
		cout << endl;
		cout << "iso_type::calc_theY_and_tdos_override_v theY=" << endl;
		Orbiter->Int_vec.print(cout, theY, v * Encoding->dim_n);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v after Encoding->apply_permutation" << endl;
	}


	if (f_v) {
		cout << "iso_type::calc_theY_and_tdos_override_v done" << endl;
	}
}

tdo_scheme *iso_type::geo_calc_tdos(
	inc_encoding *Encoding,
	incidence *inc,
	int v,
	short *&ddp, short *&ddb,
	cperm *tdo_p, cperm *tdo_q,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	tactical_decomposition *tdo = NULL;
	short *ddp_mult = NULL;
	short *ddb_mult = NULL;
	//int ddb_N, ddp_N;
	tdo_scheme *tdos;

	if (f_v) {
		cout << "iso_type::geo_calc_tdos" << endl;
	}

	tdo = NEW_OBJECT(tactical_decomposition);

#if 0
	if (f_v) {
		cout << "iso_type::geo_calc_tdos before tdo->init" << endl;
	}
	tdo->init(Encoding, verbose_level);
	if (f_v) {
		cout << "iso_type::geo_calc_tdos after tdo->init" << endl;
	}
#endif

	if (f_v) {
		cout << "iso_type::geo_calc_tdos before tdo->tdo_calc" << endl;
	}

	tdo->tdo_calc(Encoding, inc, v, f_snd_TDO, verbose_level);

	if (f_v) {
		cout << "iso_type::geo_calc_tdos after tdo->tdo_calc" << endl;
	}


#if 0
	ddp = NULL;
	ddp_mult = NULL;
	ddb = NULL;
	ddb_mult = NULL;
	if (f_v) {
		cout << "iso_type::geo_calc_tdos before tdo->tdo_dd" << endl;
	}
	tdo->tdo_dd(
			v,
		f_ddp, f_ddb,
		ddp, ddp_N, ddp_mult,
		ddb, ddb_N, ddb_mult, verbose_level);

	if (f_v) {
		cout << "iso_type::geo_calc_tdos after tdo->tdo_dd" << endl;
	}
#endif

	tdos = tdo->tdos;
	tdo->tdos = NULL;

	ddp_mult = NULL;
	ddb_mult = NULL;

	tdo_p->init_and_identity(tdo->p.l);
	tdo_q->init_and_identity(tdo->q.l);
	tdo->p.move_to(tdo_p);
	tdo->q.move_to(tdo_q);

	FREE_OBJECT(tdo);

	if (f_v) {
		cout << "iso_type::geo_calc_tdos done" << endl;
	}
	return tdos;
}

int iso_type::find_geo(
	int v, incidence *inc, tdo_scheme *tdos,
	int *theY, int tdo_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, ret, f_iso;

	if (f_v) {
		cout<< "iso_type::find_geo" << endl;
		inc->print_geo(cout, v, theY);
	}


	ret = -1;
	for (i = 0; i < nb_GEO; i++) {
		if (GEO_TDO_idx[i] != tdo_idx) {
			continue;
		}

		if (f_v) {
			cout << "iso_type::find_geo v=" << v
					<< " i=" << i << " / " << nb_GEO
					<< " find_geo: ";
			inc->print_geo(cout, v, theGEO2[i]);
			cout << endl;
		}



#if 0
		if (v == 4) {
			vl = 5;
		}
		else {
			vl = 0;
		}
#endif
		f_iso = isomorphic(v, inc, tdos, theY, theGEO2[i], 0 /* verbose_level */);
		if (f_v) {
			cout<< "iso_type::find_geo v=" << v
					<< " i=" << i << " / " << nb_GEO
					<< " after isomorphic, f_iso=" << f_iso << endl;
		}
		if (f_iso) {
			ret = i;
			break;
		}
	}

	if (f_v) {
		cout<< "iso_type::find_geo done" << endl;
	}
	return ret;
}
#endif


void iso_type::find_and_add_geo(
	int v, incidence *inc,
	int *theY, int &f_new_object, int verbose_level)
{

	//verbose_level = 5;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout<< "iso_type::find_and_add_geo" << endl;
		inc->print_geo(cout, v, theY);
		cout << endl;
	}


	object_with_canonical_form *OwCF;
	long int *theInc;
	int nb_flags;

	nb_flags = sum_R;

	theInc = NEW_lint(nb_flags);

	inc->geo_to_inc(v, theY, theInc, nb_flags);

	OwCF = NEW_OBJECT(object_with_canonical_form);

	OwCF->init_incidence_geometry(
		theInc, nb_flags, v, inc->Encoding->b, nb_flags,
		verbose_level - 2);

	if (f_v) {
		cout<< "iso_type::find_and_add_geo setting partition" << endl;
	}

	OwCF->f_partition = TRUE;
	OwCF->partition = inc->Partition[v];


	if (f_orderly) {
		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"before Canonical_forms->orderly_test" << endl;
		}
		Canonical_forms->orderly_test(OwCF,
				f_new_object, verbose_level - 2);

		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"after Canonical_forms->orderly_test" << endl;
		}

	}
	else {
		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"before Canonical_forms->add_object" << endl;
		}
		Canonical_forms->add_object(OwCF,
				f_new_object, verbose_level);

		if (f_v) {
			cout << "iso_type::find_and_add_geo "
					"after Canonical_forms->add_object" << endl;
		}

	}

	if (f_v) {
		cout<< "iso_type::find_and_add_geo done" << endl;
	}
}

#if 0
int iso_type::isomorphic(
	int v, incidence *inc, tdo_scheme *tdos,
	int *pcA, int *pcB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::isomorphic" << endl;
	}
	iso_info info;

	info.v = v;
	info.b = inc->Encoding->b;
	info.max_r = inc->Encoding->dim_n;

	info.R = inc->Encoding->R;
#if 0
	info.nb_i_vbar = inc->nb_i_vbar;
	info.nb_i_hbar = inc->nb_i_hbar;
	info.i_vbar = inc->i_vbar;
	info.i_hbar = inc->i_hbar;
#endif

	info.init_tdo(tdos);

	info.init_A_int(get_theX(pcA), FALSE /* f_full */);
	info.init_B_int(get_theX(pcB), FALSE /* f_full */);

	info.nb_isomorphisms = 0;
	info.f_break_after_fst = TRUE;
	info.f_verbose = FALSE;
	info.f_very_verbose = FALSE;
	info.f_transpose_it = f_transpose_it;

	info.f_use_d = FALSE;
	info.f_use_ddp = FALSE;
	info.f_use_ddb = FALSE;

#if 0
	iso_info_init_ddp(&info, isot->f_ddp, get_ddp(pcA), get_ddp(pcB));
	iso_info_init_ddb(&info, isot->f_ddb, get_ddb(pcA), get_ddb(pcB));
#endif

	info.iso_test(verbose_level);

	if (f_v) {
		cout << "iso_type::isomorphic done" << endl;
	}
	if (info.nb_isomorphisms) {
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void iso_type::do_aut_group(
	int v, incidence *inc, tdo_scheme *tdos,
	int *pc, int *aut_group_order,
	int f_print_isot_small, int f_print_isot,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::do_aut_group" << endl;
	}
	iso_info info;

	info.v = v;
	info.b = inc->Encoding->b;
	info.max_r = inc->Encoding->dim_n;

	info.R = inc->Encoding->R;
#if 0
	info.nb_i_vbar = inc->nb_i_vbar;
	info.nb_i_hbar = inc->nb_i_hbar;
	info.i_vbar = inc->i_vbar;
	info.i_hbar = inc->i_hbar;
#endif

	info.init_tdo(tdos);

	info.init_A_int(get_theX(pc), FALSE /* f_full */);
	info.init_B_int(get_theX(pc), FALSE /* f_full */);

	info.nb_isomorphisms = 0;
	info.f_break_after_fst = FALSE;


	/* HERE we define if we want to see the
	 * PERMUTATIONS !!! */
	info.f_verbose = TRUE;

	info.f_very_verbose = FALSE;
	info.f_transpose_it = f_transpose_it;

	info.f_use_d = FALSE;
	info.f_use_ddp = FALSE;
	info.f_use_ddb = FALSE;
#if 0
	iso_info_init_ddp(&info, isot->f_ddp, get_ddp(pc), get_ddp(pc));
	iso_info_init_ddb(&info, isot->f_ddb, get_ddb(pc), get_ddb(pc));
#endif

	info.iso_test(verbose_level);

	cout << "ago = " << info.nb_isomorphisms << " ";

	*aut_group_order = info.nb_isomorphisms;
	if (f_v) {
		cout << "iso_type::do_aut_group done" << endl;
	}
}
#endif

void iso_type::scan_tdo_flags(int tdo_flags)
{
	if (tdo_flags & 8)
		f_transpose_it = TRUE;
	else
		f_transpose_it = FALSE;
	if (tdo_flags & 4)
		f_snd_TDO = TRUE;
	else
		f_snd_TDO = FALSE;
	if (tdo_flags & 2)
		f_ddp = TRUE;
	else
		f_ddp = FALSE;
	if (tdo_flags & 1)
		f_ddb = TRUE;
	else
		f_ddb = FALSE;
}

void iso_type::second()
{
	f_generate_first = TRUE;
	f_beginning_checked = FALSE;
}

void iso_type::set_split(int remainder, int modulo)
{
	f_split = TRUE;
	split_remainder = remainder;
	split_modulo = modulo;
}

void iso_type::set_flush_line()
{
	f_flush_line = TRUE;
}

void iso_type::flush()
{
	if (nb_GEO) {
		sum_nb_GEN += nb_GEN;
		sum_nb_GEO += nb_GEO;
		sum_nb_TDO += nb_TDO;
	}
}

#if 0
void iso_type::TDO_realloc()
{
	tdo_scheme **tmp;
	int new_dim_tdo;
	int i;

	new_dim_tdo = dim_TDO + MAX_TDO;

	tmp = new ptdo_scheme [new_dim_tdo];

	for (i = 0; i < nb_TDO; i++) {
		tmp[i] = theTDO[i];
	}
	for (i = nb_TDO; i < new_dim_tdo; i++) {
		tmp[i] = NULL;
	}
	delete [] theTDO;

	theTDO = tmp;
	dim_TDO = new_dim_tdo;
}

void iso_type::find_tdos(tdo_scheme *tdos, int *tdo_idx, int *f_found)
{
	int res, i;

	*f_found = FALSE;
	for (i = 0; i < nb_TDO; i++) {

		res = tdos_cmp(tdos, theTDO[i], 0 /* verbose_level */);

		if (res < 0) {
			break;
		}
		if (res == 0) {
			*f_found = TRUE;
			break;
		}
	}
	*tdo_idx = i;
}

void iso_type::add_tdos_and_geo(tdo_scheme *tdos, int tdo_idx,
		int *theX, int *theY, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int j;

	if (f_v) {
		cout << "iso_type::add_tdos_and_geo" << endl;
	}
	// Insert a new TDO at position tdo_idx

	if (nb_TDO >= dim_TDO) {
		TDO_realloc();
	}
	for (j = nb_TDO - 1; j >= tdo_idx; j--) {
		theTDO[j + 1] = theTDO[j];
	}
	theTDO[tdo_idx] = tdos;
	nb_TDO++;
	for (j = 0; j < nb_GEO; j++) {
		if (GEO_TDO_idx[j] >= tdo_idx) {
			GEO_TDO_idx[j]++;
		}
	}
	if (f_v) {
		cout << "iso_type::add_tdos_and_geo before add_geo" << endl;
	}
	add_geo(tdo_idx, theX, theY);
	if (f_v) {
		cout << "iso_type::add_tdos_and_geo after add_geo" << endl;
	}
	if (f_v) {
		cout << "iso_type::add_tdos_and_geo done" << endl;
	}
}


void iso_type::add_geo(int tdo_idx, int *theX, int *theY)
{

	if (nb_GEO >= dim_GEO) {

		int **tmp1;
		int **tmp2;
		int *tmp3;
		int new_dim;
		int i;

		new_dim = dim_GEO + MAX_GEO;

		tmp1 = new pint[new_dim];
		tmp2 = new pint[new_dim];
		tmp3 = new int[new_dim];

		for (i = 0; i < nb_GEO; i++) {
			tmp1[i] = theGEO1[i];
			tmp2[i] = theGEO2[i];
			tmp3[i] = GEO_TDO_idx[i];
		}

		for (i = nb_GEO; i < new_dim; i++) {
			tmp1[i] = NULL;
			tmp2[i] = NULL;
			tmp3[i] = 0;
		}

		delete [] theGEO1;
		delete [] theGEO2;
		delete [] GEO_TDO_idx;

		theGEO1 = tmp1;
		theGEO2 = tmp2;
		GEO_TDO_idx = tmp3;
		dim_GEO = new_dim;
	}

	theGEO1[nb_GEO] = theX;
	theGEO2[nb_GEO] = theY;
	GEO_TDO_idx[nb_GEO] = tdo_idx;

	nb_GEO++;
}
#endif




int *iso_type::get_theX(int *theGEO)
{
	return theGEO /*+ 3*/;
}

void iso_type::geo_free(int *theGEO)
{
#if 0
	short *ddp, *ddb;

	ddp = get_ddp(theGEO);
	ddb = get_ddb(theGEO);
	if (ddp) {
		my_free(ddp);
		}
	if (ddb) {
		my_free(ddb);
		}
#endif
	delete theGEO;
	//my_free(theGEO, "isot_geo_free");
	//MEM_free(theGEO);
}

void iso_type::print_geos(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::print_geos" << endl;
	}
	{
		int h;
		long int nb_geo;

		nb_geo = Canonical_forms->B.size();

		cout << v << " " << inc->Encoding->b << " " << sum_R << endl;
		for (h = 0; h < nb_geo; h++) {

			cout << h << " / " << nb_geo << ":" << endl;
#if 0
			inc->print_override_theX(cout, theGEO1[h], v, v);

			inc->print_geo(cout, v, theGEO1[h]);
			cout << endl;
#else
			object_with_canonical_form *OiP;

			OiP = (object_with_canonical_form *) Canonical_forms->Objects[h];


			inc->print_inc(cout, v, OiP->set);

			inc->inc_to_geo(v, OiP->set, inc->Encoding->theX, sum_R);


			inc->print(cout, inc->Encoding->v, v);

			cout << endl;
#endif

		}
		cout << -1 << " " << Canonical_forms->B.size() << endl;

		tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo /*nb_GEO*/; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, FALSE, 0);
		T.print_file(cout, TRUE /* f_backwards*/);
		cout << endl;
	}
	if (f_v) {
		cout << "iso_type::print_geos done" << endl;
	}
}


void iso_type::write_inc_file(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_inc_file" << endl;
	}
	{
		ofstream ost(fname);
		int h;
		long int nb_geo;

		nb_geo = Canonical_forms->B.size();

		ost << v << " " << inc->Encoding->b << " " << sum_R << endl;
		for (h = 0; h < nb_geo /*nb_GEO*/; h++) {

			//inc->print_geo(ost, v, theGEO1[h]);

			object_with_canonical_form *OiP;

			OiP = (object_with_canonical_form *) Canonical_forms->Objects[h];
			inc->print_inc(ost, v, OiP->set);

			ost << endl;
		}
		ost << -1 << " " << Canonical_forms->B.size() << endl;

		tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo /*nb_GEO*/; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, FALSE, 0);
		T.print_file(ost, TRUE /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_inc_file done" << endl;
	}
}

void iso_type::write_blocks_file(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_blocks_file" << endl;
	}
	{
		ofstream ost(fname);
		int h, k;
		long int nb_geo;



		nb_geo = Canonical_forms->B.size();


		if (nb_geo) {
			object_with_canonical_form *OiP;

			OiP = (object_with_canonical_form *) Canonical_forms->Objects[0];

			k = inc->compute_k(v, OiP->set);




			ost << v << " " << inc->Encoding->b << " " << k << endl;
			for (h = 0; h < nb_geo /*nb_GEO*/; h++) {

				//inc->print_geo(ost, v, theGEO1[h]);

				object_with_canonical_form *OiP;

				OiP = (object_with_canonical_form *) Canonical_forms->Objects[h];
				inc->print_blocks(ost, v, OiP->set);

				ost << endl;
			}
		}
		ost << -1 << " " << Canonical_forms->B.size() << endl;

		tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo /*nb_GEO*/; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, FALSE, 0);
		T.print_file(ost, TRUE /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_blocks_file done" << endl;
	}
}

void iso_type::write_blocks_file_long(std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_type::write_blocks_file_long" << endl;
	}
	{
		ofstream ost(fname);
		int h;
		long int nb_geo;



		nb_geo = Canonical_forms->B.size();


		if (nb_geo) {

			long int *theInc;
			long int *Blocks;
			int b;

			b = inc->Encoding->b;








			ost << v << " " << b << endl;
			for (h = 0; h < nb_geo /*nb_GEO*/; h++) {

				int *K;
				int i, j, a;

				object_with_canonical_form *OiP;

				OiP = (object_with_canonical_form *) Canonical_forms->Objects[0];

				theInc = OiP->set;

				inc->compute_blocks(Blocks, K, v, theInc);

				ost << "geometry " << h << " : " << endl;

				for (i = 0; i < b; i++) {
					//ost << K[i] << " ";
					for (j = 0; j < K[i]; j++) {
						a = Blocks[i * v + j];

						if (v == 18 && b == 39) {
							if (i >= 3) {
								if (a >= 12) {
									a -= 12;
								}
								else if (a >= 6) {
									a -= 6;
								}
							}
							a++;
						}
						ost << a;
						if (j < K[i] - 1) {
							ost << ", ";
						}
					}
					ost << "\\\\" << endl;
				}


				FREE_int(K);
				FREE_lint(Blocks);

			}
		}
		ost << -1 << " " << Canonical_forms->B.size() << endl;

		tally T;
		long int *Ago;

		Ago = NEW_lint(nb_geo);
		for (h = 0; h < nb_geo /*nb_GEO*/; h++) {
			Ago[h] = Canonical_forms->Ago[h];
		}

		T.init_lint(Ago, nb_geo, FALSE, 0);
		T.print_file(ost, TRUE /* f_backwards*/);
		ost << endl;
	}
	if (f_v) {
		cout << "iso_type::write_blocks_file_long done" << endl;
	}
}



#if 0
void iso_type::print(std::ostream &ost, int f_with_TDO, int v, incidence *inc)
{
	int i, tdo_idx;
	int *pc;

	ost << "The isomorphism types are:" << endl;
	for (i = 0; i < nb_GEO; i++) {
		ost << "GEO nr. " << i << ":";
		pc = theGEO1[i];
		print_GEO(pc, v, inc);
		tdo_idx = GEO_TDO_idx[i];
		ost << " with TDO_idx = " << tdo_idx << ":" << endl;
	}
	if (f_with_TDO) {
		for (i = 0; i < nb_TDO; i++) {
			ost << "TDO nr. " << i << ":";
			theTDO[i]->print();
		}
	}
	print_status(ost, FALSE);
	ost << endl;
}
#endif

void iso_type::print_GEO(int *theY, int v, incidence *inc)
{
	//int aut_group_order;

	//aut_group_order = get_aut_group_order(pc);
	inc->print_override_theX(cout, theY, v, v);
	//cout << "automorphism group order = " << aut_group_order << endl;
}

void iso_type::print_status(std::ostream &ost, int f_with_flags)
{
#if 1
	ost << setw(3) << v << " : " << setw(7) << Canonical_forms->B.size();

#else

	if (sum_nb_GEO > 0) {
		ost << " " << setw(3) << sum_nb_GEN << " || " << setw(3) << sum_nb_TDO << " / " << setw(3) << sum_nb_GEO << " ";
	}

	ost << " " << setw(3) << nb_GEN << " || " << setw(3) << nb_TDO << " / " << setw(3) << nb_GEO << " ";
	if (f_with_flags) {
		ost << " ";
		print_flags(ost);
	}
	if (f_generate_first) {
		if (f_beginning_checked) {
			ost << " +";
		}
		else {
			ost << " -";
		}
	}
	else {
		ost << " .";
	}
	if (f_flush_line) {
		ost << " flush";
	}
#endif
}

void iso_type::print_flags(std::ostream &ost)
{
	if (f_transpose_it)
		ost << "T";
	else
		ost << "F";
	if (f_snd_TDO)
		ost << "T";
	else
		ost << "F";
	if (f_ddp)
		ost << "T";
	else
		ost << "F";
	if (f_ddb)
		ost << "T";
	else
		ost << "F";
	ost << " ";
#if 0
	if (f_range) {
		ost << " range[" << range_first << "-" << range_first + range_len - 1 << "]";
	}
#endif
	if (f_split) {
		ost << " split[" << split_remainder << " % " << split_modulo << "]";
	}
}


void iso_type::print_geometry(inc_encoding *Encoding, int v, incidence *inc)
{
	cout << "geo" << endl;
	Encoding->print_partitioned(cout, v, v, inc, FALSE /* f_print_isot */);
	cout << "end geo" << endl;
}




}}

