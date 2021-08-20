/*
 * geometry_builder.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */


#include "geo.h"

using namespace std;




geometry_builder::geometry_builder()
{
	II = 0;
	JJ = 0;

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
	if (R) {
		delete R;
	}
	if (gg) {
		delete gg;
	}
}


void geometry_builder::init(const char *control_file_name, int no,
		int flag_numeric, int f_no_inc_files,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "geometry_builder::init" << endl;
	}

	geometry_builder::control_file_name.assign(control_file_name);
	geometry_builder::no = no;
	geometry_builder::flag_numeric = flag_numeric;
	geometry_builder::f_no_inc_files = f_no_inc_files;

	for (i = 0; i < MAX_V; i++) {
		GV[i] = 0;
	}

	f_transpose_it = FALSE;
	f_save_file = FALSE;

	gg = new gen_geo;


	if (f_v) {
		cout << "geometry_builder::init before read_control_file" << endl;
	}
	read_control_file(verbose_level);
	if (f_v) {
		cout << "geometry_builder::init after read_control_file" << endl;
	}

	if (f_v) {
		cout << "geometry_builder::init done" << endl;
	}
}


#define BUFSIZE 10000

void geometry_builder::read_control_file(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	char buf[BUFSIZE];
	char type[64];
	FILE *fp;
	int b_len = 0, *B = NIL, b = 0;
	int v_len = 0, *V = NIL, v = 0;
	int *tdo = NIL;
	int *s_type = NIL, *s_flag = NIL;
	int *r_type = NIL, *r_from = NIL, *r_len = NIL;
	int *f_flush = NIL;
	int nb_fuse = 0, *fuse = NIL;
	int i, j, a, f, l, t;
	int f_do_iso_test = FALSE;
	int f_do_aut_group = FALSE;
	int f_transpose = FALSE;
	int gen_print_interval = 1;
	int f_resolve = FALSE;
	int f_resolve_with_aut_group = FALSE;
	int f_lambda = FALSE;
	int lambda = FALSE;
	int f_find_square = TRUE; /* JS 120100 */
	int f_simple = FALSE; /* JS 180100 */
	iso_type *it;

	if (f_v) {
		cout << "geometry_builder::read_control_file" << endl;
	}
	fp = fopen(control_file_name.c_str(), "r");

	gg->inc_file_name.assign("");
	gg->GEO_fname.assign("");

	if (no != 0) {
		printf("searching for input case no %d\n", no);
		while (TRUE) {
			if (fgets(buf, BUFSIZE, fp) == NIL)
				break;
			l = strlen(buf);
			if (l)
				buf[l - 1] = 0; /* delete newline */
			if (buf[0] == '#')
				continue;
			if (strncmp(buf, "STARTINGPOINT", 13) == 0) {
				sscanf(buf + 13, "%d", &l);
				if (l == no) {
					printf("found\n");
					break;
					}
				}
			}
		}
	while (TRUE) {
		if (fgets(buf, BUFSIZE, fp) == NIL) {
			break;
		}
		l = strlen(buf);
		if (l) {
			buf[l - 1] = 0; /* delete newline */
		}

		if (buf[0] == '#') {
			printf("%s\n", buf);
			continue;
		}

		if (strncmp(buf, "STARTINGPOINT", 13) == 0) {
			printf("WARNING: read %s, ignoring\n", buf);
			continue;
		}

		if (strncmp(buf, "B", 1) == 0) {
			fscanf(fp, "%d", &b_len);
			B = new int[b_len];
			b = 0;
			for (i = 0; i < b_len; i++) {
				fscanf(fp, "%d", &B[i]);
				b += B[i];
				}
			printf("B: ");
			for (i = 0; i < b_len; i++) {
				printf("%d ", B[i]);
				}
			printf(" (b = %d)\n", b);
			}

		if (strncmp(buf, "V", 1) == 0) {
			fscanf(fp, "%d", &v_len);
			V = new int [v_len];
			v = 0;
			for (i = 0; i < v_len; i++) {
				fscanf(fp, "%d", &V[i]);
				v += V[i];
				}
			printf("V: ");
			for (i = 0; i < v_len; i++) {
				printf("%d ", V[i]);
				}
			printf(" (v = %d)\n", v);


			s_type = new int[v + 1];
			s_flag = new int[v + 1];
			for (i = 0; i <= v; i++) {
				s_type[i] = 0;
				s_flag[i] = 0;
				}
			s_type[v - 1] = 1;
			s_flag[v - 1] = flag_numeric;

			r_type = new int [v + 1];
			r_from = new int [v + 1];
			r_len = new int [v + 1];
			f_flush = new int [v + 1];
			for (i = 0; i <= v; i++) {
				r_type[i] = 0;
				r_from[i] = 0;
				r_len[i] = 0;
				f_flush[i] = 0;
				}
			}

		if (strncmp(buf, "FUSE", 4) == 0) {
			if (v_len == 0) {
				printf("FUSE: V not specified !\n");
				exit(1);
				}
			fscanf(fp, "%d", &nb_fuse);
			fuse = new int[nb_fuse];
			f = 0;
			for (i = 0; i < nb_fuse; i++) {
				fscanf(fp, "%d", &fuse[i]);
				f += fuse[i];
				}
			if (f != v_len) {
				printf("FUSE: f != v_len !\n");
				exit(1);
				}
			printf("fuse: ");
			for (i = 0; i < nb_fuse; i++) {
				printf("%d ", fuse[i]);
				}
			printf("\n");
			}

		if (strncmp(buf, "FLUSH", 5) == 0) {
			sscanf(buf + 6, "%d ", &i);
			f_flush[i - 1] = TRUE;
			printf("FLUSH at line %d\n", i);
			}

		if (strncmp(buf, "R", 1) == 0) {
			sscanf(buf + 1, "%d %d %d", &i, &f, &l);
			r_type[i - 1] = 1;
			r_from[i - 1] = f;
			r_len[i - 1] = l;
			printf("R: %d [%d-%d]\n", i, f, f + l - 1);
			}

		if (strncmp(buf, "S1", 2) == 0 ||
			strncmp(buf, "S2", 2) == 0) {
			sscanf(buf + 2, "%d %s", &l, type);
			t = 0;
			if (type[0] == 'T')
				t += 8;
			if (type[1] == 'T')
				t += 4;
			if (type[2] == 'T')
				t += 2;
			if (type[3] == 'T')
				t += 1;
			if (buf[1] == '1')
				s_type[l - 1] = 1;
			else
				s_type[l - 1] = 2;
			s_flag[l - 1] = t;
			printf("S%d: line %d type %s = %d\n", s_type[l - 1], l, type, t);
			}

		if (strncmp(buf, "TDO", 3) == 0) {
			if (b_len == 0) {
				printf("B not specified !\n");
				exit(1);
				}
			if (v_len == 0) {
				printf("V not specified !\n");
				exit(1);
				}
			tdo = new int[v_len * b_len];
			for (i = 0; i < v_len; i++) {
				for (j = 0; j < b_len; j++) {
					fscanf(fp, "%d", &tdo[i * b_len + j]);
					}
				}
			printf("TDO: \n");
			for (i = 0; i < v_len; i++) {
				for (j = 0; j < b_len; j++) {
					printf("%d ", tdo[i * b_len + j]);
					}
				printf("\n");
				}
			printf("\n");
			}

		if (strncmp(buf, "GEO_file", 8) == 0) {
			fgets(buf, BUFSIZE, fp);
			l = strlen(buf);
			if (l)
				buf[l - 1] = 0; /* delete newline */
			if (!f_no_inc_files) {
				gg->inc_file_name.assign(buf);
				cout << "inc_file_name = " << gg->inc_file_name << endl;
				}
			}

		if (strcmp(buf, "end") == 0) {
			break;
			}


		if (strcmp(buf, "f_do_iso_test") == 0) {
			f_do_iso_test = TRUE;
			printf("f_do_iso_test\n");
			}

		if (strcmp(buf, "f_do_aut_group") == 0) {
			f_do_aut_group = TRUE;
			printf("f_do_aut_group\n");
			}

		if (strcmp(buf, "f_transpose") == 0) {
			f_transpose = TRUE;
			printf("f_transpose\n");
			}

		if (strcmp(buf, "print_interval") == 0) {
			fscanf(fp, "%d", &gen_print_interval);
			printf("gen_print_interval = %d\n", gen_print_interval);
			}

		if (strcmp(buf, "f_resolve") == 0) {
			f_resolve = TRUE;
			printf("f_resolve\n");
			}

		if (strcmp(buf, "f_resolve_with_aut_group") == 0) {
			f_resolve_with_aut_group = TRUE;
			f_resolve = TRUE;
			printf("f_resolve_with_aut_group\n");
			}

		if (strcmp(buf, "f_lambda") == 0) {
			f_lambda = TRUE;
			fscanf(fp, "%d", &lambda);
			printf("lambda = %d\n", lambda);
			}
		if (strcmp(buf, "f_ignore_square") == 0) {
			f_find_square = FALSE;
			puts("f_ignore_square");
			}
		if (strcmp(buf, "f_simple") == 0) {
			f_simple = TRUE;
			puts("f_simple");
			}
		}
	fclose(fp);


	if (f_v) {
		cout << "i_geo_from_file done reading input file" << endl;
	}


	if (v_len == 0) {
		printf("v_len == 0\n");
		exit(1);
		}
	if (b_len == 0) {
		printf("b_len == 0\n");
		exit(1);
		}
	if (nb_fuse == 0) {
		nb_fuse = v_len;
		fuse = new int [nb_fuse];
		for (i = 0; i < nb_fuse; i++) {
			fuse[i] = 1;
			}
		printf("fuse: ");
		for (i = 0; i < nb_fuse; i++) {
			printf("%d ", fuse[i]);
			}
		printf("\n");
		}


	if (f_v) {
		cout << "geometry_builder::read_control_file before init2" << endl;
	}

	init2(V, B, tdo, v_len, b_len,
		nb_fuse, fuse,
		f_do_iso_test,
		f_do_aut_group,
		f_resolve_with_aut_group,
		gen_print_interval,
		f_transpose,
		NIL /* file_name */,
		verbose_level);


	if (f_v) {
		cout << "geometry_builder::read_control_file after init2" << endl;
	}


	if (f_lambda) {
		gg->inc->f_lambda = TRUE;
		gg->inc->lambda = lambda;
	}

	gg->inc->f_find_square = f_find_square;
	gg->inc->f_simple = f_simple;
#if 0
	if (f_resolve) {
		ggg->gg.forget_ivhbar_in_last_isot = TRUE;
		}
	else {
		ggg->gg.forget_ivhbar_in_last_isot = FALSE;
		}
#endif

#if 0
	if (V)
		my_free(V);
	if (B)
		my_free(B);
	if (tdo)
		my_free(tdo);
#endif

	for (i = 0; i < v; i++) {
		if (s_type[i] == 1) {
			isot(i + 1, s_flag[i], verbose_level);
		}
		else if (s_type[i] == 2) {
			isot2(i + 1, s_flag[i]);
		}

	}
	for (i = 0; i < v; i++) {
		if (r_type[i] == 1) {
			range(i + 1, r_from[i], r_len[i]);
			it = gg->inc->iso_type_at_line[i];
			it->f_print_mod = TRUE;
			it->print_mod = 1;
		}
	}
	for (i = 0; i < v; i++) {
		if (f_flush[i]) {
			gg->inc->set_flush_line(i + 1);
		}
	}
	// ggg_isot(ggg, ggg->gg.inc.V, FTFF);
	if (gg->inc_file_name.length()) {
		gg->inc->set_flush_to_inc_file(v, gg->inc_file_name);
	}


#if 0
	if (s_type[v]) {
		string fname_resolved;
		int l;
		iso_type *it;
		incidence *inc = gg->inc;

		fname_resolved.assign(gg->inc_file_name);

#if 0
		l = strlen(fname_resolved);
		if (l > 4 && strcmp(fname_resolved + l - 4, ".inc") == 0) {
			fname_resolved[l - 4] = 0;
		}
#endif

		fname_resolved.append("_resolved.inc");
		cout << "fname_resolved =" << fname_resolved << endl;

		isot_no_vhbars(s_flag[v], verbose_level);

		it = inc->iso_type_no_vhbars;
		it->open_inc_file(inc, fname_resolved);

	}
#endif

	if (f_v) {
		cout << "geometry_builder::read_control_file done" << endl;
	}
}

void geometry_builder::init2(
	int *v, int *b, int *the_tdo,
	int II, int JJ,
	int nb_fuse, int *fuse,
	int f_do_iso_test,
	int f_do_aut_group,
	int f_do_aut_group_in_iso_type_without_vhbars,
	int gen_print_intervall,
	int f_transpose_it, char *file_name, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f;

	if (f_v) {
		cout << "geometry_builder::init2" << endl;
	}
	gg->nb_fuse = nb_fuse;
	f = 0;
	for (i = 0; i < nb_fuse; i++) {
		gg->fuse_first[i] = f;
		gg->fuse_len[i] = fuse[i];
		f += fuse[i];
	}


	if (f_v) {
		cout << "geometry_builder::init2 before init_tdo" << endl;
	}
	init_tdo(v, b, the_tdo, II, JJ, verbose_level);
	if (f_v) {
		cout << "geometry_builder::init2 after init_tdo" << endl;
	}

	if (f_v) {
		cout << "geometry_builder::init2 before gg->init" << endl;
	}
	gg->init(
		V, B, R,
		II /* II */,
		JJ /* JJ */,
		f_do_iso_test,
		f_do_aut_group,
		f_do_aut_group_in_iso_type_without_vhbars,
		gen_print_intervall, verbose_level);
	if (f_v) {
		cout << "geometry_builder::init2 after gg->init" << endl;
	}

	if (f_v) {
		cout << "geometry_builder::init2 before TDO_init" << endl;
	}
	TDO_init(verbose_level);
	if (f_v) {
		cout << "geometry_builder::init2 after TDO_init" << endl;
	}

	if (f_v) {
		cout << "geometry_builder::init2 before gg->inc->init" << endl;
	}
	gg->inc->init(V, B, gg->R, verbose_level);
	if (f_v) {
		cout << "geometry_builder::init2 after gg->inc->init" << endl;
	}

	if (file_name) {
		init_file_name(f_transpose_it, file_name, verbose_level);
	}
	if (f_v) {
		cout << "geometry_builder::init2 done" << endl;
	}
}


void geometry_builder::calc_PV_GV(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometry_builder::calc_PV_GV" << endl;
	}
	int i, j, k, l, m;

	for (i = 0; i < MAX_V; i++) {
		PV[i] = 0;
		GV[i] = 0;
	}
	V = 0;
	B = 0;
	for (i = 0; i < II; i++) {
		V += v[i];
	}
	for (j = 0; j < JJ; j++) {
		B += b[j];
	}
	if (V >= MAX_V) {
		cout << "geometry_builder::calc_PV_GV V >= MAX_V" << endl;
		exit(1);
	}
	if (B >= MAX_V) {
		cout << "geometry_builder::calc_PV_GV V >= MAX_V" << endl;
		exit(1);
	}

	for (i = 0; i < II; i++) {
		k = 0;
		for (j = 0; j < JJ; j++) {
			k += theTDO[i][j];
		}
		PV[k] += v[i];
	}

	for (j = 0; j < JJ; j++) {
		k = 0;
		for (i = 0; i < II; i++) {
			l = theTDO[i][j] * v[i];
			m = l / b[j];
			k += m;
		}
		GV[m] += b[j];
	}

	for (i = 0; i < MAX_V; i++) {
		if (PV[i]) {
			cout << "number of points with " << i << " lines on them is " << PV[i] << endl;
		}
	}
	cout << endl;

	for (i = 0; i < MAX_V; i++) {
		if (GV[i]) {
			cout << "number of lines with " << i << " points on them is " << GV[i] << endl;
		}
	}
	cout << endl;

	if (f_v) {
		cout << "geometry_builder::calc_PV_GV done" << endl;
	}
}

void geometry_builder::init_tdo(
	int *v, int *b, int *the_tdo,
	int II, int JJ,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, row, h;

	if (f_v) {
		cout << "geometry_builder::init_tdo II = " << II << " JJ = " << JJ << endl;
	}
	geometry_builder::II = II;
	geometry_builder::JJ = JJ;
	B = 0;
	for (j = 0; j < JJ; j++) {
		geometry_builder::b[j] = b[j];
		B += b[j];
	}
	V = 0;
	for (i = 0; i < II; i++) {
		geometry_builder::v[i] = v[i];
		V += v[i];
	}
	for (i = 0; i < II; i++) {
		for (j = 0; j < JJ; j++) {
			theTDO[i][j] = the_tdo[i * JJ + j];
		}
	}
	R = new int[V];
	row = 0;
	for (i = 0; i < II; i++) {
		for (h = 0; h < v[i]; h++, row++) {
			for (j = 0; j < JJ; j++) {
				R[row] += the_tdo[i * JJ + j];
			}
		}
	}

	print_tdo();
	if (f_v) {
		cout << "geometry_builder::init_tdo done" << endl;
	}
}

void geometry_builder::print_tdo()
{
	int i, j;

	printf("   | ");
	for (j = 0; j < JJ; j++) {
		printf("%2d ", b[j]);
	}
	printf("\n");
	printf("---| ");
	for (j = 0; j < JJ; j++) {
		printf("---");
	}
	printf("\n");
	for (i = 0; i < II; i++) {
		printf("%2d | ", v[i]);
		for (j = 0; j < JJ; j++) {
			printf("%2d ",
			theTDO[i][j]);
		}
		printf("\n");
	}
}

void geometry_builder::TDO_init(int verbose_level)
{
	gg->TDO_init(v, b, theTDO, verbose_level);
}

void geometry_builder::isot(int line,
	int tdo_flags, int verbose_level)
{
	gg->inc->stuetze_nach_zeile(line, tdo_flags, verbose_level);
}

void geometry_builder::isot_no_vhbars(int tdo_flags, int verbose_level)
{
	gg->inc->iso_type_no_vhbars = new iso_type;
	gg->inc->iso_type_no_vhbars->init(V, gg->inc, tdo_flags, verbose_level);
}

void geometry_builder::isot2(int line, int tdo_flags)
{
	gg->inc->stuetze2_nach_zeile(line, tdo_flags);
}

void geometry_builder::range(int line, int first, int len)
{
	gg->inc->set_range(line, first, len);
}

void geometry_builder::flush_line(int line)
{
	gg->inc->set_flush_line(line);
}

void geometry_builder::init_file_name(int f_transpose_it, const char *file_name, int verbose_level)
{
	/* GV[] ist in i_geo()
	 * mit Nullen gefuellt worden. */
	/* ggg->GV[3] = 26; */
	geometry_builder::f_transpose_it = f_transpose_it;
	geometry_builder::f_save_file = TRUE;
	fname.assign(file_name);
	calc_PV_GV(verbose_level);
}

