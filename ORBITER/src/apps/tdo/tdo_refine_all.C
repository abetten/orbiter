// tdo_refine_all.C
// Anton Betten
//
// started:  Dec 26 2006

#include "orbiter.h"

INT t0;

#define MY_BUFSIZE 1000000

BYTE buf[MY_BUFSIZE];

int part[10000];
int entries[100000000];
int new_part[10000];
int new_entries[100000000];

void refine(BYTE *label_in, ofstream &g, int *part, int *entries, int nb_parts, int nb_entries, 
	int row_level, int col_level, int lambda_level, int verbose_level, 
	int &nb_tdo_written, int &nb_tactical_tdo_written, INT &cnt_second_system);
int min_breadth_test(int *line, int *mult, int len, int v, int nb);
void print_table(int *line, int *mult, int len, int tdos, int tacticals);
void tex_print_table(int I, int *line, int *mult, int len, int tdos, int tacticals);
void tdo_write(ofstream &g, BYTE *label,
int *part, int nb_parts, int *entries, int nb_entries,
	INT row_level, INT col_level, INT lambda_level);
void print_distribution(ostream &ost,
	INT *types, INT nb_types, INT type_len,
	INT *distributions, INT nb_distributions);
void do_all_row_refinements(BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *point_types, INT nb_point_types, INT point_type_len,
	INT *distributions, INT nb_distributions, INT &nb_tactical, INT verbose_level);
void do_all_column_refinements(BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *line_types, INT nb_line_types, INT line_type_len,
	INT *distributions, INT nb_distributions, INT &nb_tactical, INT verbose_level);
INT do_row_refinement(INT t, BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *point_types, INT nb_point_types, INT point_type_len,
	INT *distributions, INT nb_distributions, INT verbose_level);
INT do_column_refinement(INT t, BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *line_types, INT nb_line_types, INT line_type_len,
	INT *distributions, INT nb_distributions, INT verbose_level);

int main(int argc, char **argv)
{
	INT cnt;
	BYTE *p_buf;
	BYTE str[1000];
	BYTE ext[1000];
	BYTE *base_fname;
	//BYTE fname_base[1000];
	BYTE fname_out[1000];
	char fname_in[1000];
	int verbose_level = 0;
	INT row_level, col_level, lambda_level;
	INT i, j, a;
	int I;
	INT nb_written, nb_written_tactical;
	BYTE label_in[1000];
	//BYTE label_out[1000];
	INT nb_parts, nb_entries;
	INT f_v, f_vv, f_vvv;
	int nb_files;
	int *P, *mult, *line_type;
	int line_type_size;
	int f_tdos;
	P = new int[10000];
	mult = new int[10000];
	line_type = new int[10000];
	int nb_tdo_written, nb_tactical_tdo_written;
	INT cnt_second_system = 0;


	base_fname = argv[argc - 2];
	//here is the input that tells how many files we have:
	nb_files = atoi(argv[argc - 1]);

	t0 = os_ticks();
	for (i = 1; i < argc - 2; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
	}

	cout << "opening all files that starts with " << base_fname << " for reading ... " << endl;
	cout << setw(20) << "number of TDOs :: " << setw(20) << " number of tacticals :: " << setw(20) << " line type " << endl;

	for (I = 0; I < nb_files; I++) {
		nb_written = 0;
		nb_written_tactical = 0;
		f_tdos = TRUE;
		sprintf(fname_in, "%sL%d.tdo", base_fname, I);
		cout << "opening file number " << I << " which called " << fname_in << " in " << base_fname << "for reading" << endl;
		{
			ifstream f(fname_in);
			strcpy(str, fname_in);
			get_extension_if_present(str, ext);
			chop_off_extension_if_present(str, ext);

			f_v = (verbose_level >= 1);
			f_vv = (verbose_level >= 2);
			f_vvv = (verbose_level >= 3);

			sprintf(fname_out, "%srL%d.tdo", base_fname, I);
			{
				cout << "opening file " << fname_out << " for writing" << endl;
				ofstream g(fname_out);

				cnt = 0;
				while (TRUE) {

					if (f.eof()) {
						break;
					}
					f.getline(buf, MY_BUFSIZE, '\n');

					if (buf[0] == '#')
						continue;


					p_buf = buf;

					s_scan_token_arbitrary(&p_buf, label_in);

					if (strcmp(label_in, "-1") == 0) {
					//	cout << " --> found a complete file with " << cnt << " solutions" << endl;
						break;
					}
					i = 0;
					while (TRUE) {
						s_scan_int(&p_buf, &a);

						part[i] = (int) a;
						P[i] = a;
						i++;
						if (a == -1) {
							nb_parts = i - 1;
							break;
						}
					}

					i = 0;
					line_type_size = 0;
					while (TRUE) {
						s_scan_int(&p_buf, &a);

						entries[i] = (int) a;
						if ( i % 4 == 3 ){
							line_type[line_type_size] = a;
							line_type_size++;
						}
						i++;
						if (a == -1) {
							nb_entries = (i - 1) / 4;
							break;
						}
					}

// 					for ( i = 0; i <= line_type_size; i++ ){
// 						cout << P[i] << " ";
// 					}
// 					cout << endl;
// 					for ( i = 0; i < line_type_size; i++ ){
// 						cout << line_type[i] << " ";
// 					}
// 					cout << endl;

					//getting the table info:
					if ( line_type_size > 1 ){
						for ( i = 1; i < line_type_size; i++ ){
							mult[i - 1] = P[i + 1] - P[i];
						}
						mult[line_type_size - 1] = P[0] - P[line_type_size];
					}
					else mult[0] = P[0] - P[1];
					int v = P[1];
					if ( !min_breadth_test(line_type, mult, line_type_size, v, I) ){
						nb_tdo_written = 0;
						nb_tactical_tdo_written = 0;
						f_tdos = FALSE;
						break;
					}

					s_scan_int(&p_buf, &row_level);
					s_scan_int(&p_buf, &col_level);
					s_scan_int(&p_buf, &lambda_level);

					if (f_v) {
						cout << "read TDO " << cnt << " " << label_in << endl;
					}
					if (f_vv) {
						cout << "row_level=" << row_level << endl;
						cout << "col_level=" << col_level << endl;
						cout << "lambda_level=" << lambda_level << endl;
					}
					if (FALSE) {
						cout << "nb_parts=" << nb_parts << endl;
						cout << "nb_entries=" << nb_entries << endl;
						cout << "part:" << endl;
						for (i = 0; i < nb_parts; i++)
							cout << part[i] << " ";
						cout << endl;
						cout << "entries:" << endl;
						for (i = 0; i < nb_entries; i++) {
							for (j = 0; j < 4; j++) {
								cout << entries[i * 4 + j] << " ";
							}
							cout << endl;
						}
						cout << endl;
					}
					
					refine(label_in, g, part, entries, nb_parts, nb_entries, 
						row_level, col_level, lambda_level, verbose_level, 
						nb_tdo_written, nb_tactical_tdo_written, cnt_second_system);
					if (f_v) {
						cout << "after refine: nb_tdo_written+ " << nb_tdo_written << " nb_tactical_tdo_written=" << nb_tactical_tdo_written << endl;
						}
					nb_written += nb_tdo_written;
					nb_written_tactical += nb_tactical_tdo_written;
					
					cnt++;
				} // while
				g << -1 << " " << nb_tdo_written << " TDOs, with " << nb_tactical_tdo_written << " being tactical" << endl;
				cout << nb_tdo_written << " TDOs, with " << nb_tactical_tdo_written << " being tactical" << endl;
				print_table(line_type, mult, line_type_size, nb_tdo_written, nb_tactical_tdo_written);
				if ( f_tdos )
					tex_print_table(I, line_type, mult, line_type_size, nb_tdo_written, nb_tactical_tdo_written);
			}	//cleaning g (ofstream)
		}	//cleaning f (ifstream)
	//	cout << endl << endl;
	}	//for

	delete [] P;
	delete [] mult;
	delete [] line_type;

	cout << "time: ";
	time_check(cout, t0);
	cout << endl;
}

void refine(BYTE *label_in, ofstream &g, int *part, int *entries, int nb_parts, int nb_entries, 
	int row_level, int col_level, int lambda_level, int verbose_level, 
	int &nb_tdo_written, int &nb_tactical_tdo_written, INT &cnt_second_system)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	INT nb_tactical;
	tdo_scheme G;
	partitionstack P;

	G.part = part;
	G.part_length = nb_parts;
	G.entries = entries;
	G.nb_entries = nb_entries;
	G.row_level = row_level;
	G.col_level = col_level;
	G.lambda_level = lambda_level;
	G.level[ROW] = row_level;
	G.level[COL] = col_level;
	G.level[LAMBDA] = lambda_level;

	G.init_partition_stack(verbose_level);

	nb_tdo_written = 0;
	nb_tactical_tdo_written = 0;
	
	if (col_level > row_level) {
		INT *point_types, nb_point_types, point_type_len;
		INT *distributions, nb_distributions;

		if (f_v) {
			cout << "row refinement" << endl;
			}
		if (G.refine_rows(verbose_level - 3, P,
			point_types, nb_point_types, point_type_len,
			distributions, nb_distributions, cnt_second_system, NULL)) {

			if (f_vvv) {
				print_distribution(cout,
					point_types, nb_point_types, point_type_len,
					distributions, nb_distributions);
			}

			do_all_row_refinements(label_in, g, G,
				point_types, nb_point_types, point_type_len,
				distributions, nb_distributions, nb_tactical, verbose_level - 1);

			delete [] point_types;
			delete [] distributions;

			nb_tdo_written = nb_distributions;
			nb_tactical_tdo_written = nb_tactical;
			}
		}
	else if (col_level < row_level) {
		INT *line_types, nb_line_types, line_type_len;
		INT *distributions, nb_distributions;

		if (f_v) {
			cout << "column refinement" << endl;
			}
		if (G.refine_columns(verbose_level - 3, P,
			line_types, nb_line_types, line_type_len,
			distributions, nb_distributions, cnt_second_system, NULL)) {

			if (f_vvv) {
				print_distribution(cout,
					line_types, nb_line_types, line_type_len,
					distributions, nb_distributions);
				}
			do_all_column_refinements(label_in, g, G,
				line_types, nb_line_types, line_type_len,
				distributions, nb_distributions, nb_tactical, verbose_level - 1);
								
			delete [] line_types;
			delete [] distributions;

			nb_tdo_written = nb_distributions;
			nb_tactical_tdo_written = nb_tactical;
			}
		}
	else {
		if (f_v) {
			cout << "copying tactical TDO" << endl;
			}
		tdo_write(g, label_in, part, nb_parts, entries, nb_entries,
			G.row_level, G.col_level, G.lambda_level);
		if (f_vv) {
			cout << label_in << " written" << endl;
			}
		nb_tdo_written = 1;
		nb_tactical_tdo_written = 1;
		}
}

int min_breadth_test(int *line, int *mult, int len, int v, int nb)
{
	int i, k, K, l, s, m;
	i = 0;
	s = 0;
	for ( k = 0; k <= len; k++ ){
		K = line[k];
		for ( l = 0; l < mult[k]; l++ ){
			m = MAXIMUM(K - i, 0);
			s = s + m;
			if ( s > v ){
				cout << nb << " & $(";
				for ( i = 0; i < len; i++ ){
					if ( mult[i] > 1 ){
						cout << " " << line[i] << "^{" << mult[i] << "}";
						if ( i < len - 1 )
							cout << ", ";
					}
					else{
						if ( i < len - 1 )
							cout << line[i] << ", ";
						else cout << line[i] << " ";
					}
				}
				cout << ")$ & $-$ & $-$ \\\\ %% TEX " << endl;

				return FALSE;
			}
			i++;
		}
	}
	return TRUE;
}

void print_table(int *line, int *mult, int len, int tdos, int tacticals)
{
	int i;
	cout << setw(15) << tdos << " : "
			<< setw(20) << tacticals << " : "
			<< setw(20);
	for ( i = 0; i < len; i++ ){
		if ( mult[i] > 1 ){
			cout << line[i] << "^(" << mult[i] << ")";
			if ( i < len - 1 )
				cout << ", ";
		}
		else{
			if ( i < len - 1 )
				cout << line[i] << ", ";
			else cout << line[i] << " ";
		}
	}
	cout << endl;
}

void tex_print_table(int I, int *line, int *mult, int len, int tdos, int tacticals)
{
	int i;
	cout << I << " & $(";
	for ( i = 0; i < len; i++ ){
		if ( mult[i] > 1 ){
			cout << " " << line[i] << "^{" << mult[i] << "}";
			if ( i < len - 1 )
				cout << ", ";
		}
		else{
			if ( i < len - 1 )
				cout << line[i] << ", ";
			else cout << line[i] << " ";
		}
	}
	cout << ")$ & " <<  tdos << " & " << tacticals << " \\\\ " << "  %% TEX" << endl;
}

void tdo_write(ofstream &g, BYTE *label,
	int *part, int nb_parts, int *entries, int nb_entries,
	INT row_level, INT col_level, INT lambda_level)
{
	INT i;

	g << label << " ";
	for (i = 0; i < nb_parts; i++)
		g << part[i] << " ";
	g << " -1 ";
	for (i = 0; i < 4 * nb_entries; i++)
		g << entries[i] << " ";
	g << " -1 ";
	g << row_level << " " << col_level << " " << lambda_level << endl;
}

void print_distribution(ostream &ost,
	INT *types, INT nb_types, INT type_len,
	INT *distributions, INT nb_distributions)
{
	int i, j;


	ost << "types:" << endl;
	for (i = 0; i < nb_types; i++) {
		ost << setw(3) << i << " : ";
		for (j = 0; j < type_len; j++) {
			ost << setw(3) << types[i * type_len + j];
			}
		ost << endl;
		}
	ost << endl;

	ost << "distributions:" << endl;
	for (i = 0; i < nb_distributions; i++) {
		ost << setw(3) << i << " : ";
		for (j = 0; j < nb_types; j++) {
			ost << setw(3) << distributions[i * nb_types + j];
			}
		ost << endl;
		}
	ost << endl;
}

void do_all_row_refinements(BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *point_types, INT nb_point_types, INT point_type_len,
	INT *distributions, INT nb_distributions, INT &nb_tactical, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, t;

	if (f_v) {
		cout << "do_all_row_refinements nb_distributions = " << nb_distributions << endl;
		}
	nb_tactical = 0;
	for (i = 0; i < G.part_length; i++)
		new_part[i] = G.part[i];
	for (i = 0; i < 4 * G.nb_entries; i++)
		new_entries[i] = entries[i];

	for (t = 0; t < nb_distributions; t++) {

		if (do_row_refinement(t, label_in, g, G, point_types, nb_point_types,
			point_type_len, distributions, nb_distributions, verbose_level - 1))
			nb_tactical++;

		}
	if (f_v) {
		cout << "found " << nb_distributions << " row refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
		}

}

void do_all_column_refinements(BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *line_types, INT nb_line_types, INT line_type_len,
	INT *distributions, INT nb_distributions, INT &nb_tactical, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, t;


	if (f_v) {
		cout << "do_all_row_refinements nb_distributions = " << nb_distributions << endl;
		}
	nb_tactical = 0;
	for (i = 0; i < G.part_length; i++)
		new_part[i] = G.part[i];
	for (i = 0; i < 4 * G.nb_entries; i++)
		new_entries[i] = entries[i];

	for (t = 0; t < nb_distributions; t++) {

		if (do_column_refinement(t, label_in, g, G, line_types, nb_line_types,
			line_type_len, distributions, nb_distributions, verbose_level - 1))
			nb_tactical++;

		}
	if (f_v) {
		cout << "found " << nb_distributions << " column refinements, out of which "
			<< nb_tactical << " are tactical" << endl;
		}
}


INT do_row_refinement(INT t, BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *point_types, INT nb_point_types, INT point_type_len,
	INT *distributions, INT nb_distributions, INT verbose_level)
{
	INT r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	INT *type_index;
	BYTE label_out[1000];
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT f_tactical;

	if (f_v) {
		cout << "do_row_refinement t=" << t << endl;
		}

	type_index = new INT[nb_point_types];

	new_nb_parts = G.part_length;
	R = G.nb_row_classes[ROW];
	i = 0;
	h = 0;
	S = 0;
	for (r = 0; r < R; r++) {
		l = G.row_classes_len[ROW][r];
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
			}
		while (TRUE) {
			a = distributions[t * nb_point_types + i];
			if (a == 0) {
				i++;
				continue;
				}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
				}
			type_index[h++] = i;
			if (s == 0) {
				}
			else {
				new_part[new_nb_parts++] = S + s;
				}
			s += a;
			i++;
			if (s == l)
				break;
			if (s > l) {
				cout << "do_row_refinement: s > l" << endl;
				exit(1);
				}
			}
		S += l;
		}
	if (S != G.m) {
		cout << "do_row_refinement: S != G.m" << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++)
			cout << new_part[i] << " ";
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++)
			cout << type_index[i] << " ";
		cout << endl;
		}



	{
		tdo_scheme G2;

		G2.part = new_part;
		G2.part_length = new_nb_parts;
		G2.entries = entries;
		G2.nb_entries = G.nb_entries;
		G2.row_level = new_nb_parts;
		G2.col_level = G.col_level;
		G2.lambda_level = G.lambda_level;
		G2.level[ROW] = new_nb_parts;
		G2.level[COL] = G.col_level;
		G2.level[LAMBDA] = G.lambda_level;

		G2.init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a " << G2.nb_row_classes[ROW] << " x " << G2.nb_col_classes[ROW] << " scheme" << endl;
			}
		new_nb_entries = G.nb_entries;
		for (i = 0; i < G2.nb_row_classes[ROW]; i++) {
			c1 = G2.row_classes[ROW][i];
			for (j = 0; j < G2.nb_col_classes[ROW]; j++) {
				c2 = G2.col_classes[ROW][j];
				idx = type_index[i];
				a = point_types[idx * point_type_len + j];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
					}
				new_entries[new_nb_entries * 4 + 0] = new_nb_parts;
				new_entries[new_nb_entries * 4 + 1] = c1;
				new_entries[new_nb_entries * 4 + 2] = c2;
				new_entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
				}
			}

		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << new_entries[i * 4 + j] << " ";
					}
				cout << endl;
				}
			}

		sprintf(label_out, "%s.%ld", label_in, t);

		tdo_write(g, label_out, new_part, new_nb_parts, new_entries, new_nb_entries,
			new_nb_parts, G.col_level, G.lambda_level);
		if (f_vv) {
			cout << label_out << " written" << endl;
			}
		if (new_nb_parts == G.col_level)
			f_tactical = TRUE;
		else
			f_tactical = FALSE;
	}

	delete [] type_index;
	return f_tactical;
}

INT do_column_refinement(INT t, BYTE *label_in, ofstream &g, tdo_scheme &G,
	INT *line_types, INT nb_line_types, INT line_type_len,
	INT *distributions, INT nb_distributions, INT verbose_level)
{
	INT r, i, j, h, a, l, R, c1, c2, S, s, idx, new_nb_parts, new_nb_entries;
	INT *type_index;
	BYTE label_out[1000];
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 2);
	INT f_tactical;

	if (f_v) {
		cout << "do_column_refinement t=" << t << endl;
		}

	type_index = new INT[nb_line_types];

	new_nb_parts = G.part_length;
	R = G.nb_col_classes[COL];
	i = 0;
	h = 0;
	S = G.m;
	for (r = 0; r < R; r++) {
		l = G.col_classes_len[COL][r];
		s = 0;
		if (f_vv) {
			cout << "r=" << r << " l=" << l << endl;
			}
		while (TRUE) {
			a = distributions[t * nb_line_types + i];
			if (a == 0) {
				i++;
				continue;
				}
			if (f_vv) {
				cout << "h=" << h << " i=" << i << " a=" << a << " s=" << s << " S=" << S << endl;
				}
			type_index[h++] = i;
			if (s == 0) {
				}
			else {
				new_part[new_nb_parts++] = S + s;
				}
			s += a;
			i++;
			if (s == l)
				break;
			if (s > l) {
				cout << "do_column_refinement: s > l" << endl;
				exit(1);
				}
			}
		S += l;
		}
	if (S != G.m + G.n) {
		cout << "do_column_refinement: S != G.m + G.n" << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "new_part:" << endl;
		for (i = 0; i < new_nb_parts; i++)
			cout << new_part[i] << " ";
		cout << endl;
		cout << "type_index:" << endl;
		for (i = 0; i < h; i++)
			cout << type_index[i] << " ";
		cout << endl;
		}

	{
		tdo_scheme *G2;

		G2 = new tdo_scheme;

		G2->part = new_part;
		G2->part_length = new_nb_parts;
		G2->entries = entries;
		G2->nb_entries = G.nb_entries;
		G2->row_level = G.row_level;
		G2->col_level = new_nb_parts;
		G2->lambda_level = G.lambda_level;
		G2->level[ROW] = G.row_level;
		G2->level[COL] = new_nb_parts;
		G2->level[LAMBDA] = G.lambda_level;

		G2->init_partition_stack(verbose_level - 2);

		if (f_v) {
			cout << "found a " << G2->nb_row_classes[COL] << " x " << G2->nb_col_classes[COL] << " scheme" << endl;
			}
		new_nb_entries = G.nb_entries;
		for (i = 0; i < G2->nb_row_classes[COL]; i++) {
			c1 = G2->row_classes[COL][i];
			for (j = 0; j < G2->nb_col_classes[COL]; j++) {
				c2 = G2->col_classes[COL][j];
				idx = type_index[j];
				a = line_types[idx * line_type_len + i];
				if (f_vv) {
					cout << "i=" << i << " j=" << j << " idx=" << idx << " a=" << a << endl;
					}
				new_entries[new_nb_entries * 4 + 0] = new_nb_parts;
				new_entries[new_nb_entries * 4 + 1] = c2;
				new_entries[new_nb_entries * 4 + 2] = c1;
				new_entries[new_nb_entries * 4 + 3] = a;
				new_nb_entries++;
				}
			}

		if (f_vvv) {
			for (i = 0; i < new_nb_entries; i++) {
				for (j = 0; j < 4; j++) {
					cout << setw(2) << new_entries[i * 4 + j] << " ";
					}
				cout << endl;
				}
			}

		sprintf(label_out, "%s.%ld", label_in, t);

		tdo_write(g, label_out, new_part, new_nb_parts, new_entries, new_nb_entries,
			G.row_level, new_nb_parts, G.lambda_level);

		if (f_vv) {
			cout << label_out << " written" << endl;
			}
		if (new_nb_parts == G.row_level)
			f_tactical = TRUE;
		else
			f_tactical = FALSE;
		delete G2;
		}

	delete [] type_index;
	return f_tactical;
}

