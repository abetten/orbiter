// loop.C
// 
// Anton Betten
// January 7, 2017
//
// 
//

#include "orbiter.h"

int main(int argc, char **argv)
{
	INT i;
	INT verbose_level = 0;
	INT f_command_mask = FALSE;
	const BYTE *command_mask = NULL;
	INT f_N = FALSE;
	INT N = 0;
	INT f_nb = FALSE;
	INT nb = 0;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-command_mask") == 0) {
			f_command_mask = TRUE;
			command_mask = argv[++i];
			cout << "-command_mask " << command_mask << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-nb") == 0) {
			f_nb = TRUE;
			nb = atoi(argv[++i]);
			cout << "-nb " << nb << endl;
			}
		}
	if (!f_command_mask) {
		cout << "please use -command_mask <command_mask>" << endl;
		exit(1);
		}
	if (!f_N) {
		cout << "please use -N <N>" << endl;
		exit(1);
		}
	if (!f_nb) {
		cout << "please use -nb <nb>" << endl;
		exit(1);
		}

	BYTE str[1000];

	
	for (i = 0; i < N; i++) {
		if (nb == 1) {
			sprintf(str, command_mask, i);
			}
		else if (nb == 2) {
			sprintf(str, command_mask, i, i);
			}
		else if (nb == 3) {
			sprintf(str, command_mask, i, i, i);
			}
		else {
			cout << "nb unrecognized" << endl;
			exit(1);
			}
		cout << "executing command: '" << str << "'" << endl;
		system(str);
		
		}
}


