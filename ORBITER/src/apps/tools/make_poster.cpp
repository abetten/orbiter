// make_poster.cpp


#include "orbiter.h"

using namespace std;
using namespace orbiter;


void do_it(int verbose_level, int N, int nb_frames, const char *file_mask,
		int f_file_mask_array, const char **file_mask_array);

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int f_N = FALSE;
	int N = 0;
	int nb_frames = 0;
	const char *file_mask = NULL;
	int f_file_mask_array = FALSE;
	const char **file_mask_array = NULL;


	int i, j;

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
		else if (strcmp(argv[i], "-nb_frames") == 0) {
			nb_frames = atoi(argv[++i]);
			cout << "-nb_frames " << nb_frames << endl;
			}
		else if (strcmp(argv[i], "-file_mask") == 0) {
			file_mask = argv[++i];
			cout << "-file_mask " << file_mask << endl;
			}
		else if (strcmp(argv[i], "-file_mask_array") == 0) {
			f_file_mask_array = TRUE;
			if (!f_N) {
				cout << "need -N first" << endl;
				exit(1);
			}
			file_mask_array = new const char *[N];
			for (j = 0; j < N; j++) {
				file_mask_array[j] = argv[++i];
			}
			cout << "-file_mask_array " << endl;
			}
		}

	//int f_v = (verbose_level >= 1);

	do_it(verbose_level, N, nb_frames, file_mask, f_file_mask_array, file_mask_array);
}

void do_it(int verbose_level, int N, int nb_frames, const char *file_mask,
		int f_file_mask_array, const char **file_mask_array)
{
	int nb_rows, nb_cols;
	char fname[1000];
	int i, j, h, orbit;

	nb_rows = sqrt(N) + 1;
	nb_cols = sqrt(N) + 1;

	cout << "nb_rows=" << nb_rows << endl;
	cout << "nb_cols=" << nb_cols << endl;

	{
	ofstream fp("makefile_poster");
		fp << "all:";
		for (h = 0; h < nb_frames; h++ ) {
			fp << " poster" << h;
		}
		fp << endl;
		fp << endl;
		for (h = 0; h < nb_frames; h++ ) {
		fp << "poster" << h << ":" << endl;
			for (i = 0; i < nb_rows; i++) {
				fp << "\tconvert ";
				for (j = 0; j < nb_cols; j++) {
					orbit = i * nb_cols + j;

					if (orbit < N) {
						if (f_file_mask_array) {
							sprintf(fname, file_mask_array[i * nb_cols + j], h);
						}
						else {
							sprintf(fname, file_mask, h, i * nb_cols + j);
						}
						fp << " " << fname;
					}
				}
				fp << " +append a" << i << ".png" << endl;
			} // next i
			fp << "\tconvert";
			for (i = 0; i < nb_rows; i++) {
				fp << " a" << i << ".png";
			}

			char str[1000];

			sprintf(str, "poster%02d.png", h);
			fp << " -append " << str << endl;
			fp << endl;
		} // next h
	}

}

#if 0
hilbert_%02ld_%03ld_$1.pov

poster:
	convert POSTER/frame_000.png POSTER/frame_001.png POSTER/frame_002.png POSTER/frame_003.png POSTER/frame_004.png +append a1.png 
	convert POSTER/frame_005.png POSTER/frame_006.png POSTER/frame_007.png POSTER/frame_008.png POSTER/frame_009.png +append a2.png 
	convert POSTER/frame_010.png POSTER/frame_011.png POSTER/frame_012.png POSTER/frame_013.png POSTER/frame_014.png +append a3.png 
	convert POSTER/frame_015.png POSTER/frame_016.png POSTER/frame_017.png POSTER/frame_018.png POSTER/frame_019.png +append a4.png 
	convert POSTER/frame_020.png POSTER/frame_021.png POSTER/frame_022.png POSTER/frame_023.png POSTER/frame_024.png +append a5.png 
	convert POSTER/frame_025.png POSTER/frame_026.png POSTER/frame_027.png POSTER/frame_028.png POSTER/frame_029.png +append a6.png 
	convert POSTER/frame_030.png POSTER/frame_031.png POSTER/frame_032.png POSTER/frame_033.png POSTER/frame_034.png +append a7.png 
	convert POSTER/frame_035.png POSTER/frame_036.png POSTER/frame_037.png POSTER/frame_038.png POSTER/frame_039.png +append a8.png 
	convert POSTER/frame_040.png POSTER/frame_041.png POSTER/frame_042.png POSTER/frame_043.png POSTER/frame_044.png +append a9.png 
	convert a1.png a2.png a3.png a4.png a5.png a6.png a7.png a8.png a9.png -append poster.png 
#endif






