// prepare_frames.C
// 
// Anton Betten
// February 4, 2018
//
//
// 
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, const char **argv);


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	
	{

	INT verbose_level = 0;
	INT nb_inputs = FALSE;
	INT input_first[1000];
	INT input_last[1000];
	const BYTE *input_mask[1000];
	INT f_o = FALSE;
	const BYTE *output_mask = NULL;


	INT i;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-i") == 0) {
			input_first[nb_inputs] = atoi(argv[++i]);
			input_last[nb_inputs] = atoi(argv[++i]);
			input_mask[nb_inputs] = argv[++i];
			cout << "-i " << input_first[nb_inputs] << " " << input_last[nb_inputs] << " " << input_mask[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-o") == 0) {
			f_o = TRUE;
			output_mask = argv[++i];
			cout << "-o " << output_mask << endl;
			}
		}
	INT nb_frames;
	INT j, h;
	BYTE input_fname[1000];
	BYTE output_fname[1000];
	BYTE cmd[1000];

	nb_frames = 0;
	for (i = 0; i < nb_inputs; i++) {
		nb_frames += input_last[i] - input_first[i] + 1;
		}

	cout << "nb_frames = " << nb_frames << endl;
	h = 0;
	for (i = 0; i < nb_inputs; i++) {
		cout << "input " << i << " / " << nb_inputs << endl;
		for (j = input_first[i]; j <= input_last[i]; j++) {
			cout << "input frame " << j << " / " << input_last[i] << endl;
			sprintf(input_fname, input_mask[i], (int) j);
			sprintf(output_fname, output_mask, (int) h);
			sprintf(cmd, "cp %s %s", input_fname, output_fname);
			system(cmd);
			h++;
			}
		}
	if (h != nb_frames) {
		cout << "h != nb_frames" << endl;
		exit(1);
		}
	cout << "copied " << h << " files, we are done" << endl;
	}
}


