// concatenate_files.C
// 
// Anton Betten
// March 14, 2018
//
// 
//

#include "orbiter.h"

int t0;

int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int f_N = FALSE;
	int N = 0;
	int f_fname_in_mask = FALSE;
	const char *fname_in_mask = NULL;
	int f_save = FALSE;
	const char *fname_out = NULL;
	int f_EOF_marker = FALSE;
	const char *EOF_marker = NULL;
	int f_title_line = FALSE;
	int f_loop = FALSE;
	int loop_from = 0;
	int loop_to = 0;
	
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-fname_in_mask") == 0) {
			f_fname_in_mask = TRUE;
			fname_in_mask = argv[++i];
			cout << "-fname_in_mask " << fname_in_mask << endl;
			}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			fname_out = argv[++i];
			cout << "-save " << fname_out << endl;
			}
		else if (strcmp(argv[i], "-EOF_marker") == 0) {
			f_EOF_marker = TRUE;
			EOF_marker = argv[++i];
			cout << "-EOF_marker " << EOF_marker << endl;
			}
		else if (strcmp(argv[i], "-title_line") == 0) {
			f_title_line = TRUE;
			cout << "-title_line " << endl;
			}
		else if (strcmp(argv[i], "-loop") == 0) {
			f_loop = TRUE;
			loop_from = atoi(argv[++i]);
			loop_to = atoi(argv[++i]);
			cout << "-loop " << loop_from << " " << loop_to << endl;
			}
		}
	if (!f_N) {
		cout << "Please use option -N <N>" << endl;
		exit(1);
		}
	if (!f_fname_in_mask) {
		cout << "Please use option -fname_in_mask <mask>" << endl;
		exit(1);
		}
	if (!f_save) {
		cout << "Please use option -save <fname>" << endl;
		exit(1);
		}
	if (!f_EOF_marker) {
		cout << "Please use option -EOF_marker <EOF_marker>" << endl;
		exit(1);
		}

	if (f_loop) {
		int h;

		for (h = loop_from; h <= loop_to; h++) {
			char fname_in_mask_processed[1000];
			char fname_out_processed[1000];

			sprintf(fname_in_mask_processed, fname_in_mask, h);
			sprintf(fname_out_processed, fname_out, h);
			cout << "h=" << h << " fname_in_mask_processed=" << fname_in_mask_processed << endl;
			concatenate_files(fname_in_mask_processed, N, 
				fname_out_processed, EOF_marker, f_title_line, 
				verbose_level);
			}
		}
	else {
		concatenate_files(fname_in_mask, N, 
			fname_out, EOF_marker, f_title_line, 
			verbose_level);
		}
	the_end(t0);
}



