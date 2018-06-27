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
	INT nb_inputs = 0;
	INT input_first[1000];
	INT input_len[1000];
	const BYTE *input_mask[1000];
	INT f_o = FALSE;
	const BYTE *output_mask = NULL;
	INT f_output_starts_at = FALSE;
	INT output_starts_at = 0;


	INT i;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-i") == 0) {
			input_first[nb_inputs] = atoi(argv[++i]);
			input_len[nb_inputs] = atoi(argv[++i]);
			input_mask[nb_inputs] = argv[++i];
			cout << "-i " << input_first[nb_inputs] << " " << input_len[nb_inputs] << " " << input_mask[nb_inputs] << endl;
			nb_inputs++;
			}
		else if (strcmp(argv[i], "-o") == 0) {
			f_o = TRUE;
			output_mask = argv[++i];
			cout << "-o " << output_mask << endl;
			}
		else if (strcmp(argv[i], "-output_starts_at") == 0) {
			f_output_starts_at = TRUE;
			output_starts_at = atoi(argv[++i]);
			cout << "-output_starts_at " << output_starts_at << endl;
			}
		}
	INT nb_frames;
	INT j, h, u;
	BYTE input_fname[1000];
	BYTE output_fname[1000];
	BYTE cmd[1000];

	nb_frames = 0;
	for (i = 0; i < nb_inputs; i++) {
		nb_frames += input_len[i];
		}

	cout << "nb_frames = " << nb_frames << endl;
	h = output_starts_at;
	for (i = 0; i < nb_inputs; i++) {
		cout << "input " << i << " / " << nb_inputs << endl;
		for (u = 0; u < input_len[i]; u++) {
			j = input_first[i] + u;
			cout << "input " << i << " / " << nb_inputs << " frame " << j << " / " << input_len[i] << endl;
			sprintf(input_fname, input_mask[i], (int) j);
			sprintf(output_fname, output_mask, (int) h);
			sprintf(cmd, "cp %s %s", input_fname, output_fname);
			system(cmd);
			h++;
			}
		}
	cout << "nb_frames = " << nb_frames << " copied" << endl;
#if 0
	if (h != nb_frames) {
		cout << "h != nb_frames" << endl;
		exit(1);
		}
#endif
	cout << "copied " << h - output_starts_at << " files, we are done" << endl;
	}
}


