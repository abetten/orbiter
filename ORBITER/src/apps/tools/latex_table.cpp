// latex_table.C
// 
// Anton Betten
// October 27, 2016
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

	int f_save = FALSE;
	char *save_fname = NULL;

	int f_range = FALSE;
	int range_first = 0;
	int range_len = 0;

	int f_selection = FALSE;
	const char *selection_str = NULL;
	int *selection = NULL;
	int selection_len = 0;

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
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			i++;
			save_fname = argv[i];
			cout << "-save " << save_fname << endl;
			}
		else if (strcmp(argv[i], "-selection") == 0) {
			f_selection = TRUE;
			i++;
			selection_str = argv[i];
			cout << "-selection " << selection_str << endl;
			}
		else if (strcmp(argv[i], "-range") == 0) {
			f_range = TRUE;
			range_first = atoi(argv[++i]);
			range_len = atoi(argv[++i]);
			cout << "-range " << range_first << " " << range_len << endl;
			}
		}

	//int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);


	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}
	if (!f_save) {
		cout << "please use option -save <fname>" << endl;
		exit(1);
		}

	if (f_selection) {
		int_vec_scan(selection_str, selection, selection_len);
		cout << "The selected columns are: ";
		int_vec_print(cout, selection, selection_len);
		cout << endl;
		}
	
	spreadsheet *S;
	int *f_col_select;

	S = NEW_OBJECT(spreadsheet);

	cout << "Reading table " << fname << endl;
	S->read_spreadsheet(fname, verbose_level);
	cout << "Table " << fname << " has been read" << endl;


	f_col_select = NEW_int(S->nb_cols);
	if (f_range) {
		for (i = 0; i < S->nb_cols; i++) {
			f_col_select[i] = FALSE;
			}
		for (i = 0; i < range_len; i++) {
			f_col_select[range_first + i] = TRUE;
			}
		}
	else if (f_selection) {
		for (i = 0; i < S->nb_cols; i++) {
			f_col_select[i] = FALSE;
			}
		for (i = 0; i < selection_len; i++) {
			f_col_select[selection[i]] = TRUE;
			}
		}
	else {
		for (i = 0; i < S->nb_cols; i++) {
			f_col_select[i] = TRUE;
			}
		}

	cout << "f_col_select = ";
	int_vec_print(cout, f_col_select, S->nb_cols);
	cout << endl;

	{
	ofstream fp(save_fname);
	S->print_table_latex(fp, f_col_select,
		FALSE /* f_enclose_in_parentheses */);
	}

	cout << "Written file " << save_fname << " of size "
		<< file_size(save_fname) << endl;
	
	
}


