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



#define MAX_V 300



gen_geo::gen_geo()
{
	GB = NULL;

	Decomposition_with_fuse = NULL;

	inc = NULL;

	//f_do_iso_test = FALSE;
	//f_do_aut_group = FALSE;
	//f_do_aut_group_in_iso_type_without_vhbars = FALSE;
	forget_ivhbar_in_last_isot = FALSE;
	//gen_print_intervall = FALSE;

	//std::string inc_file_name;


	//std::string fname_search_tree;
	ost_search_tree = NULL;
	//std::string fname_search_tree_flags;
	ost_search_tree_flags = NULL;

	Girth_test = NULL;

	Test_semicanonical = NULL;
}

gen_geo::~gen_geo()
{
	if (Decomposition_with_fuse) {
		FREE_OBJECT(Decomposition_with_fuse);
	}

	if (inc) {
		FREE_OBJECT(inc);
	}

	if (Girth_test) {
		FREE_OBJECT(Girth_test);
	}
	if (Test_semicanonical) {
		FREE_OBJECT(Test_semicanonical);
	}
}

void gen_geo::init(geometry_builder *GB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init" << endl;
	}
	gen_geo::GB = GB;

	inc = NEW_OBJECT(incidence);

	if (f_v) {
		cout << "gen_geo::init before inc->init" << endl;
	}
	inc->init(this, GB->V, GB->B, GB->R, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after inc->init" << endl;
	}

	forget_ivhbar_in_last_isot = FALSE;


	Decomposition_with_fuse = NEW_OBJECT(decomposition_with_fuse);

	if (f_v) {
		cout << "gen_geo::init before Decomposition_with_fuse->init" << endl;
	}
	Decomposition_with_fuse->init(this, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after Decomposition_with_fuse->init" << endl;
	}


	if (f_v) {
		cout << "gen_geo::init before init_semicanonical" << endl;
	}
	init_semicanonical(verbose_level);
	if (f_v) {
		cout << "gen_geo::init before init_semicanonical done" << endl;
	}


	if (GB->Descr->f_girth_test) {

		Girth_test = NEW_OBJECT(girth_test);

		Girth_test->init(this, GB->Descr->girth, verbose_level);

	}

	if (f_v) {
		cout << "gen_geo::init done" << endl;
	}

}


void gen_geo::init_semicanonical(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init_semicanonical" << endl;
	}

	Test_semicanonical = NEW_OBJECT(test_semicanonical);

	if (f_v) {
		cout << "gen_geo::init_semicanonical before Test_semicanonical->init" << endl;
	}
	Test_semicanonical->init(this, MAX_V, verbose_level);
	if (f_v) {
		cout << "gen_geo::init_semicanonical after Test_semicanonical->init" << endl;
	}

	if (f_v) {
		cout << "gen_geo::init_semicanonical done" << endl;
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

void gen_geo::main2(int &nb_GEN, int &nb_GEO,
		int &ticks, int &tps, int verbose_level)
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

		object_with_canonical_form *OiP;

		OiP = (object_with_canonical_form *) it->Canonical_forms->Objects[0];

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
	//int s_nb_i_vbar, s_nb_i_hbar;
	iso_type *it0, *it1;

	if (f_v) {
		if (GB->Descr->f_lambda) {
			cout << "lambda = " << GB->Descr->lambda << endl;
		}
	}


	setup_output_files(verbose_level);

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

		cout << "GeoFst returns FALSE, no geometry exists. This is perhaps a bit unusual." << endl;
		goto l_exit;
	}


	while (TRUE) {


		inc->gl_nb_GEN++;
		if (f_v) {
			cout << "gen_geo::generate_all nb_GEN=" << inc->gl_nb_GEN << endl;
			inc->print(cout, inc->Encoding->v, inc->Encoding->v);
			//cout << "pairs:" << endl;
			//inc->print_pairs(inc->Encoding->v);

#if 0
			if ((inc->gl_nb_GEN % gen_print_intervall) == 0) {
				//inc->print(cout, inc->Encoding->v);
			}
#endif
		}


		//cout << "*** do_geo *** geometry no. " << gg->inc.gl_nb_GEN << endl;
#if 0
		if (incidence_back_test(&gg->inc,
			TRUE /* f_verbose */,
			FALSE /* f_very_verbose */)) {
			}
#endif



#if 0
		if (forget_ivhbar_in_last_isot) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;
		}
#endif
		if (FALSE) {
			cout << "gen_geo::generate_all before isot_add for it0" << endl;
		}

		it0->add_geometry(inc->Encoding,
			inc->Encoding->v, inc,
			f_already_there,
			verbose_level - 2);



		record_tree(inc->Encoding->v, f_already_there);


		if (FALSE) {
			cout << "gen_geo::generate_all after isot_add for it0" << endl;
		}

		if (f_vv && it0->Canonical_forms->B.size() % 1 == 0) {
			cout << it0->Canonical_forms->B.size() << endl;
			inc->print(cout, inc->Encoding->v, inc->Encoding->v);
		}

#if 0
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
					verbose_level - 2);

			//record_tree(inc->Encoding->v, f_already_there);

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
#endif

		if (!GeoNxt(verbose_level - 5)) {
			if (f_v) {
				cout << "gen_geo::generate_all GeoNxt returns FALSE, finished" << endl;
			}
			break;
		}
	}


	close_output_files(verbose_level);


l_exit:
	if (f_v) {
		cout << "gen_geo::generate_all done" << endl;
	}
}

void gen_geo::setup_output_files(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (GB->Descr->f_search_tree) {
		fname_search_tree.assign(inc_file_name);
		fname_search_tree.append("_tree.txt");

		if (f_v) {
			cout << "gen_geo::setup_output_files, opening file " << fname_search_tree << endl;
		}

		ost_search_tree = new ofstream;
		ost_search_tree->open (fname_search_tree, std::ofstream::out);
	}
	else {
		ost_search_tree = NULL;
	}

	if (GB->Descr->f_search_tree_flags) {
		fname_search_tree_flags.assign(inc_file_name);
		fname_search_tree_flags.append("_tree_flags.txt");

		if (f_v) {
			cout << "gen_geo::setup_output_files, opening file " << fname_search_tree << endl;
		}

		ost_search_tree_flags = new ofstream;
		ost_search_tree_flags->open (fname_search_tree_flags, std::ofstream::out);
	}
	else {
		ost_search_tree_flags = NULL;
	}

}

void gen_geo::close_output_files(int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	if (GB->Descr->f_search_tree) {
		*ost_search_tree << "-1" << endl;
		ost_search_tree->close();
	}

	if (GB->Descr->f_search_tree_flags) {
		*ost_search_tree_flags << "-1" << endl;
		ost_search_tree_flags->close();
	}

}

void gen_geo::record_tree(int i1, int f_already_there)
{
	int color;

	if (f_already_there) {
		color = COLOR_RED;
	}
	else {
		color = COLOR_GREEN;
	}


	if (ost_search_tree) {

		int row;
		long int rk;

		*ost_search_tree << i1;
		for (row = 0; row < i1; row++) {
			rk = inc->Encoding->rank_row(row);
			*ost_search_tree << " " << rk;
		}
		*ost_search_tree << " " << color;
		*ost_search_tree << endl;
	}

	if (GB->Descr->f_search_tree_flags) {

		std::vector<int> flags;
		int h;

		inc->Encoding->get_flags(i1, flags);

		*ost_search_tree_flags << flags.size();

		for (h = 0; h < flags.size(); h++) {
			*ost_search_tree_flags << " " << flags[h];

		}
		*ost_search_tree_flags << " " << color;
		*ost_search_tree_flags << endl;

	}

}


void gen_geo::print_I_m(int I, int m)
{
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);

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
			if (!GeoLineFstSplit(I, m, verbose_level)) {
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
			if (GeoLineNxtSplit(I, m, verbose_level)) {
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);

	if (f_v) {
		cout << "gen_geo::GeoRowNxt I=" << I << endl;
	}

	int m;

	m = C->v - 1;
	while (TRUE) {
		while (TRUE) {
			if (GeoLineNxtSplit(I, m, verbose_level)) {
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
			if (!GeoLineFstSplit(I, m, verbose_level)) {
				break;
			}
		}
		// m-th element could not initialize, move on
		m--;
	}
}

#define GEO_LINE_SPLIT

int gen_geo::GeoLineFstSplit(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineFstSplit I=" << I << " m=" << m << endl;
	}

#ifdef GEO_LINE_SPLIT
	iso_type *it;
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;
	it = inc->iso_type_at_line[i1];
	if (it && it->f_split) {
		if ((it->nb_GEO % it->split_modulo) != it->split_remainder) {
			return FALSE;
		}
	}
	if (!GeoLineFst0(I, m, verbose_level)) {
		return FALSE;
	}
	return TRUE;
#else
	return GeoLineFst0(I, m, verbose_level);
#endif
}

int gen_geo::GeoLineNxtSplit(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::GeoLineNxtSplit I=" << I << " m=" << m << endl;
	}
#ifdef GEO_LINE_SPLIT
	iso_type *it;
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;
	it = inc->iso_type_at_line[i1];
	if (it && it->f_split) {
		if ((it->nb_GEO % it->split_modulo) != it->split_remainder) {
			return FALSE;
		}
	}
	if (!GeoLineNxt0(I, m, verbose_level)) {
		return FALSE;
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
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
					verbose_level - 2);



			record_tree(i1 + 1, f_already_there);


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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);

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
					verbose_level - 2);

			record_tree(i1 + 1, f_already_there);

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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);

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
				verbose_level - 2);

			record_tree(i1 + 1, f_already_there);

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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
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
				if (f_v) {
					cout << "gen_geo::GeoLineFst" << endl;
					inc->print(cout, i1 + 1, inc->Encoding->v);
				}
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
	int f_v = (verbose_level >= 1);
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
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
				if (f_v) {
					cout << "gen_geo::GeoLineNxt" << endl;
					inc->print(cout, i1 + 1, inc->Encoding->v);
				}
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
		cout << "gen_geo::GeoConfFst I=" << I << " m=" << m
				<< " J=" << J << endl;
	}
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	int n, i1;

	if (J == 0) {
		i1 = C->i0 + m;

		Test_semicanonical->row_init(I, m, J,
					i1,
					verbose_level);

	}
	n = 0;
	while (TRUE) {
		while (TRUE) {
			if (n >= C->r) {
				if (f_v) {
					cout << "gen_geo::GeoConfFst I=" << I << " m=" << m
							<< " J=" << J << " returns TRUE" << endl;
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
					cout << "gen_geo::GeoConfFst I=" << I << " m=" << m
							<< " J=" << J << " returns FALSE" << endl;
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	int n, i1;

	i1 = C->i0 + m;


	Test_semicanonical->row_test_continue(I, m, J, i1);


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
					cout << "gen_geo::GeoConfNxt I=" << I << " m=" << m
							<< " J=" << J << " returns FALSE" << endl;
				}
			}
			n--;
		}
		// n-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (n >= C->r - 1) {
				if (f_v) {
					cout << "gen_geo::GeoConfNxt I=" << I << " m=" << m
							<< " J=" << J << " returns TRUE" << endl;
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
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
		cout << "gen_geo::GeoXFst I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << endl;
	}
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	int i1, j0, r, j;

	i1 = C->i0 + m; // current row
	r = C->r0 + n; // current incidence index
	j0 = C->j0;


	j = Test_semicanonical->row_starter(I, m, J, n,
			i1, j0, r,
			verbose_level);


	int ret;

	ret = X_Fst(I, m, J, n, j, verbose_level);

	if (f_v) {
		cout << "gen_geo::GeoXFst I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << " returns " << ret << endl;
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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
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


	Test_semicanonical->col_marker_remove(I, m, J, n,
				i1, j0, r, old_x);



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

		j1 = j0 + j;
		if (inc->K[j1] >= Decomposition_with_fuse->K1[fuse_idx * GB->b_len + J]) {
			// this column is already full. We cannot put the incidence here.
			if (f_v) {
				cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " skipped "
								"because of column sum wrt fuse partition" << endl;
			}
			continue;
		}

		if (Test_semicanonical->col_marker_test(j0, j, i1)) {
			continue;
		}

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j1;
			// must be set before calling find_square



		k = inc->K[j1];
		inc->theY[j1][k] = i1;

		if (!apply_tests(I, m, J, n, j, verbose_level - 2)) {
			if (f_v) {
				cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " skipped because of test" << endl;
			}
			continue;
		}


		// the incidence passes the tests:


		// increase column sum
		inc->K[j1]++;



		Test_semicanonical->marker_move_on(I, m, J, n, j,
				i1, j0, r,
				verbose_level);



		if (f_v) {
			cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J
					<< " n=" << n << " returns TRUE" << endl;
		}
		return TRUE;
	}
	if (f_v) {
		cout << "gen_geo::GeoXNxt I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << " returns FALSE" << endl;
	}
	return FALSE;
}

void gen_geo::GeoXClear(int I, int m, int J, int n)
{
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
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



	Test_semicanonical->col_marker_remove(I, m, J, n,
				i1, j0, r, old_x);



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
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	int fuse_idx, i1, j0, j1, r, k;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// index of current incidence within the row

	j0 = C->j0;

#if 0
	// f_vbar must be off:
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		cout << "I = " << I << " m = " << m << ", J = " << J
				<< ", n = " << n << ", i1 = " << i1
				<< ", r = " << r << ", j0 = " << j0 << endl;
		cout << "X_Fst f_vbar[i1][r]" << endl;
		exit(1);
	}
#endif

	for (; j < C->b; j++) {

		j1 = j0 + j;

		if (inc->K[j1] >= Decomposition_with_fuse->K1[fuse_idx * GB->b_len + J]) {

			// column j1 is full, move on

			if (f_v) {
				cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " skipped because of column sum" << endl;
			}
			continue;
		}


		if (Test_semicanonical->col_marker_test(j0, j, i1)) {
			continue;
		}

		inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = j0 + j;
		// incidence must be recorded before we call find_square




		k = inc->K[j1];
		inc->theY[j1][k] = i1;

		// and now come the tests:

		if (!apply_tests(I, m, J, n, j, verbose_level - 2)) {
			if (f_v) {
				cout << "gen_geo::X_Fst I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " skipped because of test" << endl;
			}
			continue;
		}


		// the incidence passes the tests:

		// increase column sum:

		inc->K[j1]++;


		// ToDo: col_markers_update:


		Test_semicanonical->markers_update(I, m, J, n, j,
				i1, j0, r,
				verbose_level);


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
		cout << "gen_geo::apply_tests I=" << I << " m=" << m
				<< " J=" << J << " n=" << n << " j=" << j << endl;
	}
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// index of current incidence within the row

	j0 = C->j0;

	j1 = j0 + j;

	k = inc->K[j1];

	// We want to place an incidence in (i1,j1).

	if (GB->Descr->f_lambda) {

		// We check that there are no repeated columns
		// in the incidence matrix of the design that we create.


		// If this was the last incidence in column j1,
		// make sure the column is different from the previous column.
		// Do this test based on theY[], which lists the incidences in the column.

		// JS = Jochim Selzer


		if (GB->Descr->f_simple) { /* JS 180100 */

			// Note that K[j1] does not take into account
			// the incidence that we want to place in column j1.

			if (Decomposition_with_fuse->F_last_k_in_col[fuse_idx * GB->b_len + J] &&
					k == Decomposition_with_fuse->K1[fuse_idx * GB->b_len + J] - 1) {
				int ii;

				// check whether the column is
				// different from the previous column:

				for (ii = 0; ii <= k; ii++) {
					if (inc->theY[j1 - 1][ii] != inc->theY[j1][ii]) {
						break; // yes, columns differ !
					}
				}
				if (ii > k) {

					// not OK, columns are equal.
					// The design is not simple.

					inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
					inc->theY[j1][k] = -1;
					if (f_v) {
						cout << "gen_geo::apply_tests I=" << I << " m=" << m
								<< " J=" << J << " n=" << n << " j=" << j
								<< " rejected because not simple" << endl;
					}
					return FALSE;
				}
			}
		} // JS


		// check whether there is enough capacity in the lambda array.
		// This means, check if all scalar products
		// for previous rows with a one in column j1
		// are strictly less than \lambda:
		// This means that there is room for one more pair
		// created by the present incidence in row i1 and column j1

		int ii, ii1;

		for (ii = 0; ii < k; ii++) {

			ii1 = inc->theY[j1][ii];

			if (inc->pairs[i1][ii1] >= GB->Descr->lambda) {

				// no, fail

				inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
				inc->theY[j1][k] = -1;
				break;
			}
		}


		if (ii < k) {

			// There is not enough capacity in the lambda array.
			// So, we cannot put the incidence at (i1,j1):

			if (f_v) {
				cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j
						<< " rejected because of pair test" << endl;
			}

			return FALSE;
		}

		// increment the pairs counter:

		increment_pairs_point(i1, j1, k);

		// additional test that applies if the row is complete:

		// In this case, all scalar products with previous rows must be equal to lambda:

		if (J == GB->b_len - 1 && n == C->r - 1) {
			for (ii = 0; ii < i1; ii++) {
				if (inc->pairs[i1][ii] != GB->Descr->lambda) {
					// no, fail
					break;
				}
			}
			if (ii < i1) {

				// reject this incidence:

				decrement_pairs_point(i1, j1, k);

				if (f_v) {
					cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J
							<< " n=" << n << " j=" << j << " rejected because at least "
									"one pair is not covered lambda times" << endl;
				}
				return FALSE;
			}
		}
	}
	else {

		if (GB->Descr->f_find_square) { /* JS 120100 */

			// Test the square condition.
			// The square condition tests whether there is
			// a pair of points that is contained in two different blocks
			// (corresponding to a square in the incidence matrix, hence the name).
			// We need only consider the pairs of points that include i1.

			if (inc->find_square(i1, r)) {

				// fail, cannot place incidence here

				inc->Encoding->theX[i1 * inc->Encoding->dim_n + r] = -1;
				if (f_v) {
					cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J
							<< " n=" << n << " j=" << j << " rejected because "
									"of square condition" << endl;
				}
				return FALSE;
			}
		}
	}

	girth_test_add_incidence(i1, r, j1);

	if (!check_girth_condition(i1, r, j1, 0 /*verbose_level*/)) {

		girth_test_delete_incidence(i1, r, j1);

		if (GB->Descr->f_lambda) {

			decrement_pairs_point(i1, j1, k);

		}
		if (f_v) {
			cout << "gen_geo::apply_tests I=" << I << " m=" << m << " J=" << J
					<< " n=" << n << " j=" << j << " rejected because "
							"of girth condition" << endl;
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



