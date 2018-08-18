// join_sets.C
// 
// Anton Betten
// March 11, 2018
//
// 
//

#include "orbiter.h"

#define MAX_LINES 1000

int main(int argc, char **argv)
{
	INT i, j;
	INT verbose_level = 0;
	INT f_file_mask = FALSE;
	const BYTE *file_mask = NULL;
	INT f_N = FALSE;
	INT N = 0;
	INT f_out_mask = FALSE;
	const BYTE *out_mask = NULL;
	INT f_save = FALSE;
	const BYTE *save_fname = NULL;


	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file_mask") == 0) {
			f_file_mask = TRUE;
			file_mask = argv[++i];
			cout << "-file_mask " << file_mask << endl;
			}
		else if (strcmp(argv[i], "-out_mask") == 0) {
			f_out_mask = TRUE;
			out_mask = argv[++i];
			cout << "-out_mask " << out_mask << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			save_fname = argv[++i];
			cout << "-save " << save_fname << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		}
	if (!f_file_mask) {
		cout << "please use -file_mask <file_mask>" << endl;
		exit(1);
		}
	if (!f_out_mask) {
		cout << "please use -out_mask <out_mask>" << endl;
		exit(1);
		}
	if (!f_N) {
		cout << "please use -N <N>" << endl;
		exit(1);
		}

	cout << "N=" << N << endl;

	INT **Sets;
	INT *Set_sz;

	Sets = NEW_PINT(N);
	Set_sz = NEW_INT(N);

	INT M, h;

	M = 0;

	for (i = 0; i < N; i++) {
		BYTE fname[1000];

		sprintf(fname, file_mask, i, i, i, i, i);
		
		cout << "Reading file " << fname << endl;
		read_set_from_file(fname, Sets[i], Set_sz[i], 0 /* verbose_level */);

		M += Set_sz[i];
		}

	cout << "Read all sets from files, M=" << M << endl;
	
	spreadsheet *S;

	S = new spreadsheet;

	S->init_empty_table(M + 1 /*nb_rows*/, 5 /* nb_cols */);

	h = 0;
	S->fill_entry_with_text(0 /* row_idx */, 0 /* col_idx */, "C0");
	S->fill_entry_with_text(0 /* row_idx */, 1 /* col_idx */, "C1");
	S->fill_entry_with_text(0 /* row_idx */, 2 /* col_idx */, "C2");
	S->fill_entry_with_text(0 /* row_idx */, 3 /* col_idx */, "C3");
	S->fill_entry_with_text(0 /* row_idx */, 4 /* col_idx */, "C4");
	for (i = 0; i < N; i++) {
		for (j = 0; j < Set_sz[i]; j++, h++) {
			BYTE str[1000];

			sprintf(str, "%ld", h);
			S->fill_entry_with_text(1 + h /* row_idx */, 0 /* col_idx */, str);
			sprintf(str, "%ld", i);
			S->fill_entry_with_text(1 + h /* row_idx */, 1 /* col_idx */, str);
			sprintf(str, "%ld", j);
			S->fill_entry_with_text(1 + h /* row_idx */, 2 /* col_idx */, str);
			sprintf(str, "%ld", Sets[i][j]);
			S->fill_entry_with_text(1 + h /* row_idx */, 3 /* col_idx */, str);
			sprintf(str, out_mask, i, Sets[i][j]);
			S->fill_entry_with_text(1 + h /* row_idx */, 4 /* col_idx */, str);
			}
		}
	cout << "Saving table to file " << save_fname << endl;
	S->save(save_fname, verbose_level);
	cout << "Written file " << save_fname << " of size " << file_size(save_fname) << endl;

}


