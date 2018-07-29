// plot_xy.C
//
// Anton Betten
// February 6, 2018

#include "GALOIS/galois.h"
#include "math.h"



int main(int argc, char **argv)
{
	//INT t0 = os_ticks();
	INT verbose_level;
	INT i;
	
	INT f_file = FALSE;
	const BYTE *fname = NULL;
	INT f_x = FALSE;
	const BYTE *x_str = NULL;
	INT f_y = FALSE;
	const BYTE *y_str = NULL;
	INT f_x_label = FALSE;
	const BYTE *x_label = "";
	INT f_y_label = FALSE;
	const BYTE *y_label = "";
	INT f_title = FALSE;
	const BYTE *title = "";
	INT f_x_multiplyer = FALSE;
	double x_multiplyer = 1.;
	INT f_y_multiplyer = FALSE;
	double y_multiplyer = 1.;
	INT f_name = FALSE;
	const BYTE *name = "";
	INT f_series = FALSE;
	INT series_from = 0;
	INT series_len = 0;
	const BYTE *series_column_mask = NULL;
	INT f_series_by_values = FALSE;
	const BYTE *series_by_values_str = 0;
	const BYTE *series_by_values_column_mask = NULL;
	INT f_logscale = FALSE;
	const BYTE *logscale_xy = NULL;
	INT f_key_position = FALSE;
	const BYTE *key_position = NULL;
	INT f_png = TRUE;
	INT f_tikz = TRUE;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[i + 1]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			i++;
			fname = argv[i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-x") == 0) {
			f_x = TRUE;
			x_str = argv[++i];
			cout << "-x " << x_str << endl;
			}
		else if (strcmp(argv[i], "-y") == 0) {
			f_y = TRUE;
			y_str = argv[++i];
			cout << "-y " << y_str << endl;
			}
		else if (strcmp(argv[i], "-x_label") == 0) {
			f_x_label = TRUE;
			x_label = argv[++i];
			cout << "-x " << x_label << endl;
			}
		else if (strcmp(argv[i], "-y_label") == 0) {
			f_y_label = TRUE;
			y_label = argv[++i];
			cout << "-y " << y_label << endl;
			}
		else if (strcmp(argv[i], "-title") == 0) {
			f_title = TRUE;
			title = argv[++i];
			cout << "-title " << title << endl;
			}
		else if (strcmp(argv[i], "-x_multiplyer") == 0) {
			f_x_multiplyer = TRUE;
			sscanf(argv[++i], "%lf", &x_multiplyer);
			cout << "-x_multiplyer " << x_multiplyer << endl;
			}
		else if (strcmp(argv[i], "-y_multiplyer") == 0) {
			f_y_multiplyer = TRUE;
			sscanf(argv[++i], "%lf", &y_multiplyer);
			cout << "-y_multiplyer " << y_multiplyer << endl;
			}
		else if (strcmp(argv[i], "-name") == 0) {
			f_name = TRUE;
			name = argv[++i];
			cout << "-name " << name << endl;
			}
		else if (strcmp(argv[i], "-series") == 0) {
			f_series = TRUE;
			series_from = atoi(argv[++i]);
			series_len = atoi(argv[++i]);
			series_column_mask = argv[++i];
			cout << "-series " << series_from << " " << series_len << " " << series_column_mask << endl;
			}
		else if (strcmp(argv[i], "-series_by_values") == 0) {
			f_series_by_values = TRUE;
			series_by_values_str = argv[++i];
			series_by_values_column_mask = argv[++i];
			cout << "-series_by_values " << series_by_values_str << " " << series_by_values_column_mask << endl;
			}
		else if (strcmp(argv[i], "-logscale") == 0) {
			f_logscale = TRUE;
			logscale_xy = argv[++i];
			cout << "-logscale " << logscale_xy << endl;
			}
		else if (strcmp(argv[i], "-key_position") == 0) {
			f_key_position = TRUE;
			key_position = argv[++i];
			cout << "-key_position " << key_position << endl;
			}
		else if (strcmp(argv[i], "-png") == 0) {
			f_png = TRUE;
			f_tikz = FALSE;
			cout << "-png " << endl;
			}
		else if (strcmp(argv[i], "-tikz") == 0) {
			f_png = FALSE;
			f_tikz = TRUE;
			cout << "-tikz " << endl;
			}
		}
	
	//BYTE fname_out[1000];
	BYTE prefix[1000];
	BYTE ext[1000];

	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}

	//sprintf(fname_out, "plot_%s", fname);

	cout << "Reading data from file " << fname << " of size " << file_size(fname) << endl;

	spreadsheet *Sp;
	INT *Idx1, idx2;
	INT m, n;
		INT *values;
		
	
	Sp = new spreadsheet;
	Sp->read_spreadsheet(fname, 0 /*verbose_level*/);
	Sp->print_table(cout, TRUE);
	if (f_series) {
		Idx1 = NEW_INT(series_len);
		BYTE label[1000];

		for (i = 0; i < series_len; i++) {
			sprintf(label, series_column_mask, series_from + i);
			Idx1[i] = Sp->find_by_column(label);
			}
		idx2 = -1;
		}
	else if (f_series_by_values) {

		INT_vec_scan(series_by_values_str, values, series_len);
		Idx1 = NEW_INT(series_len);
		BYTE label[1000];

		for (i = 0; i < series_len; i++) {
			sprintf(label, series_by_values_column_mask, values[i]);
			Idx1[i] = Sp->find_by_column(label);
			}
		idx2 = -1;
		}
	else {
		Idx1 = NEW_INT(1);
		Idx1[0] = Sp->find_by_column(x_str);
		idx2 = Sp->find_by_column(y_str);
		}
	m = Sp->nb_rows - 1;
	n = Sp->nb_cols;

#if 0
	INT *Data;
	INT m, n;
	
	INT_matrix_read_csv(fname, Data, m, n, verbose_level);
	
	while (m) {
		if (Data[(m - 1) * n + n - 1] == 0) {
			m--;
			}
		else {
			break;
			}
		}
	INT_matrix_print(Data, m, n);
#endif

#if 0
	INT *Series[9];
	INT h, u, v;

	for (h = 0; h < 9; h++) {
		INT *S;
		
		S = NEW_INT(m * 2);
		INT_vec_zero(S, m * 2);
		for (i = 0; i < m; i++) {
			S[i * 2 + 0] = Data[i * n + h];
			if (h < 9 - 1) {
				S[i * 2 + 1] = Data[i * n + 9 + h];
				}
			else {
				v = 0;
				for (u = 0; u < 8; u++) {
					v = MAXIMUM(v, Data[i * n + 9 + u]);
					}
				S[i * 2 + 1] = v;
				}
			}
		cout << "Series " << h << ":" << endl;
		INT_matrix_print(S, m, 2);

		Series[h] = S;
		}
#else
	double *S;
	INT *f_NA;
	INT h;

	if (f_series) {
		S = new double [series_len * (m + 1)];
		f_NA = new INT [series_len * (m + 1)];
		//INT_vec_zero(S, m * 2);
		for (h = 0; h < m; h++) {
			for (i = 0; i < series_len; i++) {
				f_NA[i * (m + 1) + 0] = FALSE;
				S[i * (m + 1) + 0] = i;

				Sp->get_value_double_or_NA(h + 1, Idx1[i], S[i * (m + 1) + 1 + h], f_NA[i * (m + 1) + 1 + h]);
				//S[i * (m + 1) + 1 + h] = Sp->get_double(h + 1, Idx1[i]);
				if (f_y_multiplyer && !f_NA[i * (m + 1) + 1 + h]) {
					S[i * (m + 1) + 1 + h] *= y_multiplyer;
					}
				}
			}
		}
	else if (f_series_by_values) {
		S = new double [series_len * (m + 1)];
		f_NA = new INT [series_len * (m + 1)];
		//INT_vec_zero(S, m * 2);
		for (h = 0; h < m; h++) {
			for (i = 0; i < series_len; i++) {
				S[i * (m + 1) + 0] = values[i];
				f_NA[i * (m + 1) + 0] = FALSE;
				//S[i * (m + 1) + 0] = values[i];
				Sp->get_value_double_or_NA(h + 1, Idx1[i], S[i * (m + 1) + 1 + h], f_NA[i * (m + 1) + 1 + h]);
				//S[i * (m + 1) + 1 + h] = Sp->get_double(h + 1, Idx1[i]);
				if (f_y_multiplyer && !f_NA[i * (m + 1) + 1 + h]) {
					S[i * (m + 1) + 1 + h] *= y_multiplyer;
					}
				}
			}
		}
	else {
		S = new double [m * 2];
		f_NA = new INT [m * 2];
		//INT_vec_zero(S, m * 2);
		for (i = 0; i < m; i++) {
			Sp->get_value_double_or_NA(i + 1, Idx1[0], S[i * 2 + 0], f_NA[i * 2 + 0]);
			//S[i * 2 + 0] = Sp->get_double(i + 1, Idx1[0]);
			if (f_x_multiplyer && !f_NA[i * 2 + 0]) {
				S[i * 2 + 0] *= x_multiplyer;
				}
			Sp->get_value_double_or_NA(i + 1, idx2, S[i * 2 + 1], f_NA[i * 2 + 1]);
			//S[i * 2 + 1] = Sp->get_double(i + 1, idx2);
			if (f_y_multiplyer && !f_NA[i * 2 + 1]) {
				S[i * 2 + 1] *= y_multiplyer;
				}
			}
		while (m > 0) {
			if (ABS(S[(m - 1) * 2 + 0]) < 0.00001 && ABS(S[(m - 1) * 2 + 1]) < 0.00001) {
				m--;
				cout << "reducing m to " << m << endl;
				}
			else {
				break;
				}
			}
		}
#endif

	if (f_name) {
		strcpy(prefix, name);
		}
	else {
		strcpy(prefix, fname);
		get_extension_if_present(prefix, ext);
		chop_off_extension_if_present(prefix, ext);
		}
	cout << "prefix=" << prefix << endl;




	BYTE fname_dat[1000];
	BYTE fname_txt[1000];
	BYTE fname_out[1000];
	BYTE cmd[1000];

	sprintf(fname_dat, "%s_data.dat", prefix);
	sprintf(fname_txt, "%s_gnuplot.txt", prefix);


	if (f_png) {
		sprintf(fname_out, "%s_gnuplot.png", prefix);
		}
	else if (f_tikz) {
		sprintf(fname_out, "%s_gnuplot.tex", prefix);
		}


	cout << "writing data file " << fname_dat << endl;

	{
		ofstream fp(fname_dat);

		fp << "# " << fname_dat << endl;
		if (f_series || f_series_by_values) {

			for (h = 0; h < m; h++) {
				fp << "# data block " << h << endl;
				fp << "# X Y" << endl;
				double x, y;

				for (i = 0; i < series_len; i++) {
					if (!f_NA[i * (m + 1) + 0] && !f_NA[i * (m + 1) + 1 + h]) {
						x = (double) S[i * (m + 1) + 0];
						y = (double) S[i * (m + 1) + 1 + h];
						fp << setw(12) << x << setw(12) << y << endl;
						}
					}
				fp << endl;
				fp << endl;
				}
			}
		else {

			fp << "# data block " << endl;
			fp << "# X Y" << endl;
			double x, y;

			for (i = 0; i < m; i++) {
				if (!f_NA[i * 2 + 0] && !f_NA[i * 2 + 1]) {
					x = (double) S[i * 2 + 0];
					y = (double) S[i * 2 + 1];
					fp << setw(12) << x << setw(12) << y << endl;
					}
				}
			fp << endl;
			fp << endl;
			}
	}
	cout << "Written file " << fname_dat << " of size " << file_size(fname_dat) << endl;


	cout << "writing gnuplot control file " << fname_txt << endl;

	{
		ofstream fp(fname_txt);

		fp << "# " << fname_txt << endl;
		if (f_png) {
			fp << "set terminal png" << endl;
			}
		else if (f_tikz) {
			fp << "set terminal lua tikz" << endl;
			}
		if (f_series || f_series_by_values) {
			fp << "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue" << endl;
			fp << "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red" << endl;
			fp << "set style line 3 lc rgb '#18dd1f' lt 1 lw 2 pt 4 ps 1.5   # " << endl;
			fp << "#set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 6 ps 1.5   # --- blue" << endl;
			}
		else {
			fp << "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue" << endl;
			}
		if (f_logscale) {
			fp << "set logscale " << logscale_xy << endl;
			}
		fp << "set xlabel '" << x_label << "'" << endl;
		fp << "set ylabel '" << y_label << "'" << endl;
		if (f_key_position) {
			fp << "set key " << key_position << endl;
			}
		else {
			fp << "set key bottom right" << endl;
			}
		if (f_series || f_series_by_values) {
			for (h = 0; h < m; h++) {
				BYTE my_title[1000];
				sprintf(my_title, title, h + 1);
				if (h == 0) {
					fp << "plot '" << fname_dat << "' index " << h << " with linespoints  ls " << (h % 3) + 1 << " title '" << my_title << "'";
					if (h < m - 1) {
						fp << ", \\" << endl;
						}
					}
				else {
					fp << "     '' index " << h << " with linespoints  ls " << (h % 3) + 1 << " title '" << my_title << "'";
					if (h < m - 1) {
						fp << ", \\" << endl;
						}
					}
				}
			}
		else {
			fp << "plot '" << fname_dat << "' using 1:2 with points ls 1 title '" << title << "'" << endl;
			}
	}
	cout << "Written file " << fname_txt << " of size " << file_size(fname_txt) << endl;

	sprintf(cmd, "gnuplot %s >%s", fname_txt, fname_out);
	cout << "Executing command: " << cmd << endl;
	system(cmd);
	
//plot 'BLT_stats_data.dat' index 0 with linespoints ls 1 title 'Create cases', \
//     ''                   index 1 with linespoints ls 2 title 'Create graphs', \
//     ''                   index 2 with linespoints ls 3 title 'Clique search'
#if 0
	{
		ofstream fp(fname2);

		fp << "# " << fname2 << endl;
		fp << "set terminal png" << endl;
		fp << "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue" << endl;
		fp << "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red" << endl;
		fp << "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 4 ps 1.5   # --- red" << endl;
		fp << "set logscale xy" << endl;
		fp << "set xlabel 'Memory (in GB)'" << endl;
		fp << "set ylabel 'Time (in hrs)'" << endl;
		fp << "set key bottom right" << endl;
		fp << "plot '" << fname1 << "' index 0 with linespoints ls 1 title 'Create cases', \\" << endl;
		fp << "     ''                   index 1 with linespoints ls 2 title 'Create graphs', \\" << endl;
		fp << "     ''                   index 2 with linespoints ls 3 title 'Clique search'" << endl;
	}
	cout << "Written file " << fname2 << " of size " << file_size(fname2) << endl;

	sprintf(cmd, "gnuplot %s >%s", fname2, fname4);

	cout << "Executing command: " << cmd << endl;
	system(cmd);
#endif

	

#if 0
# plotting_data3.dat
# First data block (index 0)
# X   Y
  1   2
  2   3


# Second index block (index 1)
# X   Y
  3   2
  4   1
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red
plot 'plotting-data3.dat' index 0 with linespoints ls 1, \
     ''                   index 1 with linespoints ls 2
#endif




}





