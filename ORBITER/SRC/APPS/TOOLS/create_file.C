// create_file.C
// 
// Anton Betten
// February 20, 2018
//
// 
//

#include "orbiter.h"
#include "discreta.h"

#define MAX_LINES 1000

int main(int argc, char **argv)
{
	INT i, j;
	INT verbose_level = 0;
	INT f_file_mask = FALSE;
	const BYTE *file_mask = NULL;
	INT f_N = FALSE;
	INT N = 0;
	INT nb_lines = 0;
	const BYTE *lines[MAX_LINES];
	INT f_repeat = FALSE;
	INT repeat_N = 0;
	const BYTE *repeat_mask;
	INT f_split = FALSE;
	INT split_m = 0;

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
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-line") == 0) {
			lines[nb_lines] = argv[++i];
			cout << "-line " << lines[nb_lines] << endl;
			nb_lines++;
			}
		else if (strcmp(argv[i], "-repeat") == 0) {
			f_repeat = TRUE;
			repeat_N = atoi(argv[++i]);
			repeat_mask = argv[++i];
			cout << "-repeat " << repeat_N << " " << repeat_mask << endl;
			}
		else if (strcmp(argv[i], "-split") == 0) {
			f_split = TRUE;
			split_m = atoi(argv[++i]);
			cout << "-split " << split_m << endl;
			}
		}
	if (!f_file_mask) {
		cout << "please use -file_mask <file_mask>" << endl;
		exit(1);
		}
	if (!f_N) {
		cout << "please use -N <N>" << endl;
		exit(1);
		}

	BYTE fname[1000];
	BYTE str[1000];
	INT r;

	
	for (i = 0; i < N; i++) {

		sprintf(fname, file_mask, i);
		

		{
		ofstream fp(fname);
		
		for (j = 0; j < nb_lines; j++) {
			sprintf(str, lines[j], i);
			fp << str << endl;
			}
		if (f_repeat) {
			if (f_split) {
				for (r = 0; r < split_m; r++) {
					for (j = 0; j < repeat_N; j++) {
						if ((j % split_m) == r) {
							sprintf(str, repeat_mask, j);
							fp << str << endl;
							}
						}
					fp << endl;
					}
				}
			else {
				}
			}
		}
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		
		}
}


