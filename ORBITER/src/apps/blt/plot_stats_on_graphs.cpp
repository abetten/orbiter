// plot_stats_on_graphs.C
//
// Anton Betten
// February 7, 2018

#include "orbiter.h"
#include "math.h"

using namespace orbiter;



int main(int argc, char **argv)
{
	//int t0 = os_ticks();
	int verbose_level;
	int i;

	int f_q = FALSE;
	int q = 0;
	
	int f_file = FALSE;
	char *fname = NULL;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[i + 1]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[i + 1]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			i++;
			fname = argv[i];
			cout << "-file " << fname << endl;
			}
		}
	
	char fname_out[1000];
	char prefix[1000];
	char ext[1000];

	if (!f_file) {
		cout << "please use option -file <fname>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	sprintf(fname_out, "plot_%s", fname);

	cout << "Reading data from file " << fname << " of size " << file_size(fname) << endl;


	int *Data;
	int m, n;
	
	int_matrix_read_csv(fname, Data, m, n, verbose_level);
	
	while (m) {
		if (Data[(m - 1) * n + n - 1] == 0) {
			m--;
			}
		else {
			break;
			}
		}
	int_matrix_print(Data, m, n);


	int *Series;
		
	Series = NEW_int(m * 2);
	int_vec_zero(Series, m * 2);
	for (i = 0; i < m; i++) {
		Series[i * 2 + 0] = Data[i * n + 1];
		Series[i * 2 + 1] = Data[i * n + 7];
		}
	cout << "Series:" << endl;
	int_matrix_print(Series, m, 2);

	strcpy(prefix, fname);
	get_extension_if_present(prefix, ext);
	chop_off_extension_if_present(prefix, ext);
	
	cout << "prefix=" << prefix << endl;




	char fname1[1000];
	char fname2[1000];
	char fname3[1000];
	char cmd[1000];

	sprintf(fname1, "%s_data.dat", prefix);
	sprintf(fname2, "%s_gnuplot.txt", prefix);
	sprintf(fname3, "%s_gnuplot.png", prefix);
	{
		ofstream fp(fname1);

		fp << "# " << fname1 << endl;
		fp << "# data block " << endl;
		fp << "# X Y" << endl;
		double t1, t2;

		for (i = 0; i < m; i++) {
			t1 = (double) Series[i * 2 + 1];
			t2 = (double) Series[i * 2 + 0];
			fp << setw(12) << t1 << setw(12) << t2 << endl;
			}
		fp << endl;
		fp << endl;
	}
	cout << "Written file " << fname1 << " of size " << file_size(fname1) << endl;

	{
		ofstream fp(fname2);

		fp << "# " << fname2 << endl;
		fp << "set terminal png" << endl;
		//fp << "set style dots 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue" << endl;
		fp << "set logscale xy" << endl;
		fp << "set xlabel 'Time to create the graph (in 1/100 sec)'" << endl;
		fp << "set ylabel 'Time to find all cliques (in 1/100 sec)'" << endl;
		fp << "set key bottom right" << endl;
		fp << "plot '" << fname1 << "' using 1:2 title 'BLT(" << q << ")'" << endl;
	}
	cout << "Written file " << fname2 << " of size " << file_size(fname2) << endl;

	sprintf(cmd, "gnuplot %s >%s", fname2, fname3);
	cout << "Executing command: " << cmd << endl;
	system(cmd);
	


	

#if 0
# plotting_data3.dat
# First data block (index 0)
# X   Y
  1   2
  2   3


# Second index block (index 1)
# X   Y
  3   2
  4   1
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5   # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5   # --- red
plot 'plotting-data3.dat' index 0 with linespoints ls 1, \
     ''                   index 1 with linespoints ls 2
#endif




}





