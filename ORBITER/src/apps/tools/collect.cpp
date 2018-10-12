// collect.C
//
// Anton Betten
// May 27, 2018

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

int main(int argc, char **argv);
int int_vec_sum(int *v, int len);

int main(int argc, char **argv)
{
	int verbose_level = 0;
	int f_fname_in_mask = FALSE;
	const char *fname_in_mask = NULL;
	int f_N = FALSE;
	int N = 0;
	int f_save = FALSE;
	const char *save_fname_csv = NULL;
	const char *save_col_label = NULL;
	int f_graph_number_of_vertices = FALSE;
	int i, j;

	t0 = os_ticks();
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-graph_number_of_vertices") == 0) {
			f_graph_number_of_vertices = TRUE;
			cout << "-graph_number_of_vertices " << endl;
			}
		else if (strcmp(argv[i], "-fname_in_mask") == 0) {
			f_fname_in_mask = TRUE;
			fname_in_mask = argv[++i];
			cout << "-fname_in_mask " << fname_in_mask << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			save_fname_csv = argv[++i];
			save_col_label = argv[++i];
			cout << "-save " << save_fname_csv << " " << save_col_label << endl;
			}
		}
	if (!f_fname_in_mask) {
		cout << "please use option -fname_in_mask <fname_in_mask>" << endl;
		exit(1);
		}
	if (!f_N) {
		cout << "please use option -N <N>" << endl;
		exit(1);
		}

	//int f_v = (verbose_level >= 1);

	char ext[1000];
	char fname[1000];
	//char fname_out[1000];


	get_extension_if_present(fname_in_mask, ext);
	cout << "extension: " << ext << endl;

	int **Data;
	int **Ago;
	int *Nb_rows;
	int *Nb_cols;
	int f_has_ago = FALSE;

	Data = NEW_pint(N);
	Ago = NEW_pint(N);
	Nb_rows = NEW_int(N);
	Nb_cols = NEW_int(N);
	for (i = 0; i < N; i++) {
		sprintf(fname, fname_in_mask, i);
		cout << "trying to read file " << i << " / " << N
			<< " named " << fname << ":" << endl;
		if (f_graph_number_of_vertices) {

			colored_graph G;

			G.load(fname, verbose_level);
			Nb_rows[i] = G.nb_points;
			Nb_cols[i] = 0;
			}
		else if (strcmp(ext, ".csv") == 0) {
			int_matrix_read_csv(fname, Data[i], 
				Nb_rows[i], Nb_cols[i], verbose_level);
			cout << "read file " << fname << " with "
				<< Nb_rows[i] << " rows" << endl;
			}
		else if (strcmp(ext, ".txt") == 0) {

			f_has_ago = TRUE;
			char **data;
			int *Set_sizes;
			int **Sets;
			char **Ago_ascii;
			char **Aut_ascii;
			int *Casenumbers;
			
#if 0
			//int **sets;
			//int *set_sizes;
			cout << "before read_and_parse_data_file " << fname << endl;
			read_and_parse_data_file(fname, Nb_rows[i], 
				data, sets, set_sizes, 
				verbose_level);
#endif

			cout << "before try_to_read_file" << endl;
			if (try_to_read_file(fname, Nb_rows[i],
					data, 0 /* verbose_level */)) {
				cout << "read file " << fname
						<< " nb_cases = " << Nb_rows[i] << endl;
				}
			else {
				cout << "couldn't read file " << fname << endl;
				exit(1);
				}

			cout << "before parse_sets" << endl;
			parse_sets(Nb_rows[i], data, FALSE /* f_casenumbers */, 
				Set_sizes, Sets, Ago_ascii, Aut_ascii, 
				Casenumbers, 
				verbose_level);
			cout << "after parse_sets, scanning Ago[i]" << endl;
			Ago[i] = NEW_int(Nb_rows[i]);
			for (j = 0; j < Nb_rows[i]; j++) {
				Ago[i][j] = atoi(Ago_ascii[j]);
				}
			cout << "after scanning Ago[i]" << endl;
			
			//int_matrix_read_csv(fname, Data[i], 
			//	Nb_rows[i], Nb_cols[i], verbose_level);
			//cout << "read file " << fname << " with "
			//<< Nb_rows[i] << " rows" << endl;
			}
		}
	classify C;

	C.init(Nb_rows, N, FALSE, 0);

	cout << "classification:" << endl;
	C.print_naked(TRUE);
	cout << endl;

	int total;

	total = int_vec_sum(Nb_rows, N);
	cout << "total = " << total << endl;

	if (f_save) {
		int_vec_write_csv(Nb_rows, N, save_fname_csv, save_col_label);
		cout << "Written file " << save_fname_csv << " of size "
			<< file_size(save_fname_csv) << endl;
		}

	if (f_has_ago) {
		int *T;
		int h;

		cout << "creating table T" << endl;

		T = NEW_int(total * 3);
		h = 0;
		for (i = 0; i < N; i++) {
			for (j = 0; j < Nb_rows[i]; j++, h++) {
				T[h * 3 + 0] = i;
				T[h * 3 + 1] = j;
				T[h * 3 + 2] = Ago[i][j];
				}
			}
		if (f_save) {

			char fname[1000];
			char ext[1000];
			strcpy(fname, save_fname_csv);
			get_extension_if_present_and_chop_off(fname, ext);
			strcat(fname, "_ago");
			strcat(fname, ext);
			const char *column_label[] = {"i", "j", "ago"};

			cout << "saving file " << fname << endl;
			int_matrix_write_csv_with_labels(fname, 
				T, total, 3, column_label);
			
			//int_vec_write_csv(Nb_rows, N, fname, "Nb");
			cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
			}
		FREE_int(T);
		}

}

int int_vec_sum(int *v, int len)
{
	int s = 0;
	int i;

	for (i = 0; i < len; i++) {
		s += v[i];
		}
	return s;
}

