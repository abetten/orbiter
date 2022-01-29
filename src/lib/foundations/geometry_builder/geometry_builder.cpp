/*
 * geometry_builder.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */


#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



geometry_builder::geometry_builder()
{
	Descr = NULL;

	//II = 0;
	//JJ = 0;

	v = NULL;
	v_len = 0;

	b = NULL;
	b_len = 0;

	fuse = NULL;
	fuse_len = 0;

	TDO = NULL;
	TDO_len = 0;

	V = 0;
	B = 0;
	R = NULL;

	f_transpose_it = FALSE;
	f_save_file = FALSE;
	//std::string fname;

	//control_file_name;
	no = 0;
	flag_numeric = 0;
	f_no_inc_files = FALSE;

	gg = NULL;


}

geometry_builder::~geometry_builder()
{
	if (v) {
		FREE_int(v);
	}
	if (b) {
		FREE_int(b);
	}
	if (fuse) {
		FREE_int(fuse);
	}
	if (TDO) {
		FREE_int(TDO);
	}
	if (R) {
		FREE_int(R);
	}
	if (gg) {
		FREE_OBJECT(gg);
	}
}

void geometry_builder::init_description(geometry_builder_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "geometry_builder::init_description" << endl;
	}
	geometry_builder::Descr = Descr;

	if (!Descr->f_V) {
		cout << "please use option -V to specify the row partition" << endl;
		exit(1);
	}
	Int_vec_scan(Descr->V_text, v, v_len);
	V = 0;
	for (i = 0; i < v_len; i++) {
		V += v[i];
	}

	if (!Descr->f_B) {
		cout << "please use option -B to specify the column partition" << endl;
		exit(1);
	}
	Int_vec_scan(Descr->B_text, b, b_len);
	B = 0;
	for (i = 0; i < b_len; i++) {
		B += b[i];
	}


	if (f_v) {
		cout << "geometry_builder::init_description V=" << V << endl;
		cout << "geometry_builder::init_description B=" << B << endl;
	}

	if (!Descr->f_TDO) {
		cout << "please use option -TDO to specify the TDO row-scheme" << endl;
		exit(1);
	}
	Int_vec_scan(Descr->TDO_text, TDO, TDO_len);

	if (Descr->f_fuse) {
		Int_vec_scan(Descr->fuse_text, fuse, fuse_len);
		int f;

		f = 0;
		for (i = 0; i < fuse_len; i++) {
			f += fuse[i];
		}
		if (f != v_len) {
			cout << "the sum of the fuse values must equal the number of rows of the TDO" << endl;
			cout << "f=" << f << endl;
			cout << "v_len=" << v_len << endl;
			exit(1);
		}
	}

	f_transpose_it = FALSE;
	f_save_file = FALSE;


	if (f_v) {
		cout << "geometry_builder::init_description before compute_VBR" << endl;
	}

	compute_VBR(verbose_level);

	if (f_v) {
		cout << "geometry_builder::init_description after compute_VBR" << endl;
	}


	gg = NEW_OBJECT(gen_geo);

	if (f_v) {
		cout << "geometry_builder::init_description before gg->init" << endl;
	}
	gg->init(this, verbose_level);
	if (f_v) {
		cout << "geometry_builder::init_description after gg->init" << endl;
	}




	if (f_v) {
		cout << "geometry_builder::init_description set_flush_to_inc_file" << endl;
	}


	//gg->print_conf();



	if (Descr->f_fname_GEO) {

		gg->inc_file_name.assign(Descr->fname_GEO);


		if (f_v) {
			cout << "geometry_builder::init_description inc_file_name = " << gg->inc_file_name << endl;
		}
	}

	if (f_v) {
		cout << "geometry_builder::init_description setting up tests:" << endl;
		for (i = 0; i < Descr->test_lines.size(); i++) {
			cout << "-test " << Descr->test_lines[i] << endl;
		}
	}


	if (f_v) {
		cout << "geometry_builder::init_description allocating arrays" << endl;
	}

	int *s_type = NULL;
	int *s_flag = NULL;

	s_type = NEW_int(V + 1);
	s_flag = NEW_int(V + 1);

	for (i = 0; i <= V; i++) {
		s_type[i] = 0;
		s_flag[i] = 0;
	}


	if (f_v) {
		cout << "geometry_builder::init_description reading test_lines" << endl;
	}

	for (i = 0; i < Descr->test_lines.size(); i++) {
		int *lines;
		int lines_len;
		int a, j;

		//cout << "-test " << Descr->test_lines[i] << " " << Descr->test_flags[i] << endl;

		orbiter_kernel_system::Orbiter->get_vector_from_label(Descr->test_lines[i], lines, lines_len, 0 /* verbose_level*/);
		//Orbiter->Int_vec.scan(Descr->test_lines[i], lines, lines_len);

		for (j = 0; j < lines_len; j++) {
			a = lines[j];
			s_type[a] = 1;
		}
	}


	if (f_v) {
		cout << "geometry_builder::init_description reading test2_lines" << endl;
	}

	for (i = 0; i < Descr->test2_lines.size(); i++) {
		int *lines;
		int lines_len;
		int a, j;

		//cout << "-test " << Descr->test_lines[i] << " " << Descr->test_flags[i] << endl;
		Int_vec_scan(Descr->test2_lines[i], lines, lines_len);
		//flags = true_false_string_numeric(Descr->test_flags[i].c_str());
		for (j = 0; j < lines_len; j++) {
			a = lines[j];
		}
	}

	if (f_v) {
		cout << "geometry_builder::init_description installing tests" << endl;
	}

	for (i = 1; i <= V; i++) {
		if (f_v) {
			cout << "geometry_builder::init_description installing test on line " << i << endl;
		}

		if (s_type[i] == 1) {
			isot(i, verbose_level);
		}
		else if (s_type[i] == 2) {
			isot2(i, verbose_level);
		}

	}


	if (Descr->f_split) {
		if (f_v) {
			cout << "geometry_builder::init_description installing split on line " << Descr->split_line << endl;
			cout << "geometry_builder::init_description remainder " << Descr->split_remainder << endl;
			cout << "geometry_builder::init_description modulo " << Descr->split_modulo << endl;
		}
		gg->inc->set_split(Descr->split_line, Descr->split_remainder, Descr->split_modulo);

	}




	if (f_v) {
		cout << "geometry_builder::init_description done" << endl;
	}
}



void geometry_builder::compute_VBR(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, row, h;

	if (f_v) {
		cout << "geometry_builder::compute_VBR v_len = " << v_len << " b_len = " << b_len << endl;
	}
	B = 0;
	for (j = 0; j < b_len; j++) {
		B += b[j];
	}
	V = 0;
	for (i = 0; i < v_len; i++) {
		V += v[i];
	}

	R = NEW_int(V);
	row = 0;
	for (i = 0; i < v_len; i++) {
		for (h = 0; h < v[i]; h++, row++) {
			R[row] = 0;
			for (j = 0; j < b_len; j++) {
				R[row] += TDO[i * b_len + j];
			}
		}
	}

	print_tdo();

	if (f_v) {
		cout << "geometry_builder::compute_VBR done" << endl;
	}
}

void geometry_builder::print_tdo()
{
	int i, j;

	cout << "   | ";
	for (j = 0; j < b_len; j++) {
		cout << setw(2) << b[j] << " ";
	}
	cout << endl;
	cout << "---| ";
	for (j = 0; j < b_len; j++) {
		cout << "---";
	}
	cout << endl;
	for (i = 0; i < v_len; i++) {
		cout << setw(2) << v[i] << " ";
		for (j = 0; j < b_len; j++) {
			cout << setw(2) << TDO[i * b_len + j] << " ";
		}
		cout << endl;
	}
}

void geometry_builder::isot(int line, int verbose_level)
{
	gg->inc->install_isomorphism_test_after_a_given_row(
			line, Descr->f_orderly, verbose_level);
}

void geometry_builder::isot_no_vhbars(int verbose_level)
{
	gg->inc->iso_type_no_vhbars = new iso_type;
	gg->inc->iso_type_no_vhbars->init(gg, V, Descr->f_orderly, verbose_level);
}

void geometry_builder::isot2(int line, int verbose_level)
{
	gg->inc->install_isomorphism_test_of_second_kind_after_a_given_row(
			line, Descr->f_orderly, verbose_level);
}

void geometry_builder::set_split(int line, int remainder, int modulo)
{
	gg->inc->set_split(line, remainder, modulo);
}



}}}

