// split_spreadsheet.C
//
// Anton Betten
// September 11, 2018
//
//
//

#include "orbiter.h"


using namespace std;


using namespace orbiter;



// global data:

int t0; // the system time when the program started



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;

	int f_file = FALSE;
	const char *fname;

	int f_save_mask = FALSE;
	char *save_mask = NULL;

	int f_col = FALSE;
	char *col_str = NULL;

	int f_split_mod = FALSE;
	int split_mod = 0;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			i++;
			fname = argv[i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-save_mask") == 0) {
			f_save_mask = TRUE;
			i++;
			save_mask = argv[i];
			cout << "-save_mask " << save_mask << endl;
			}
		else if (strcmp(argv[i], "-col") == 0) {
			f_col = TRUE;
			i++;
			col_str = argv[i];
			cout << "-col " << col_str << endl;
			}
		else if (strcmp(argv[i], "-split_mod") == 0) {
			f_split_mod = TRUE;
			split_mod = atoi(argv[++i]);
			cout << "-split_mod " << split_mod << endl;
			}
		}

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);


	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}
	if (!f_save_mask) {
		cout << "please use option -save_mask <fname_mask>" << endl;
		exit(1);
		}
	if (!f_col) {
		cout << "please use option -col <col_label>" << endl;
		exit(1);
		}
	if (!f_split_mod) {
		cout << "please use option -split_mod <split_mod>" << endl;
		exit(1);
		}

	spreadsheet *S;
	int col_idx;


	S = NEW_OBJECT(spreadsheet);

	cout << "Reading table " << fname << endl;
	S->read_spreadsheet(fname, 0 /*verbose_level*/);
	cout << "Table " << fname << " has been read" << endl;

	col_idx = S->find_by_column(col_str);
	cout << "column " << col_str << " has index " << col_idx << endl;


	int *f_selected;
	int r, a;

	f_selected = NEW_int(S->nb_rows);
	for (r = 0; r < split_mod; r++) {
		char fname_save[1000];

		sprintf(fname_save, save_mask, r);

		{
			ofstream fp(fname_save);

			int_vec_zero(f_selected, S->nb_rows);
			f_selected[0] = TRUE;
			for (i = 0; i < S->nb_rows - 1; i++) {
				a = S->get_int(i + 1, col_idx);
				if ((a % split_mod) == r) {
					f_selected[1 + i] = TRUE;
				}
			}
			cout << "saving file " << fname_save << endl;
			S->print_table_with_row_selection(f_selected, fp);
			fp << "END" << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_save << " of size "
				<< Fio.file_size(fname_save) << endl;
	}

#if 0
	{
	ofstream fp(save_fname);
	S->print_table_latex(fp, f_col_select, FALSE /* f_enclose_in_parentheses */);
	}
#endif



}


