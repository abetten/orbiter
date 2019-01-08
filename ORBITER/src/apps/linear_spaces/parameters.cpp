// parameters.C
// 
// Anton Betten
// 12/6/2010
//
// 
//
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

void do_maximal_arc(int q, int r, int verbose_level);
void write_widor(char *fname, int m, int n, int *v, int *b, int *aij);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_maximal_arc = FALSE;
	int q;
	int r;
	
	t0 = os_ticks();



 	

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-maximal_arc") == 0) {
			f_maximal_arc = TRUE;
			q = atoi(argv[++i]);
			r = atoi(argv[++i]);
			cout << "-maximal_arc " << q << " " << r << endl;
			}
		}
	
	if (f_maximal_arc) {
		do_maximal_arc(q, r, verbose_level);
		}

	the_end(t0);
}

void do_maximal_arc(int q, int r, int verbose_level)
{
	int m = 2, n = 2;
	int v[2], b[2], aij[4];
	int Q;
	char fname[1000];
	
	Q = q * q;
	v[0] = q * (r - 1) + r;
	v[1] = Q + q * (2 - r) - r + 1;
	b[0] = Q - Q / r + q * 2 - q / r + 1;
	b[1] = Q / r + q / r - q;
	aij[0] = q + 1;
	aij[1] = 0;
	aij[2] = q - q / r + 1;
	aij[3] = q / r; 
	sprintf(fname, "max_arc_q%d_r%d.widor", q, r);
	write_widor(fname, m, n, v, b, aij);
}


void write_widor(char *fname, int m, int n, int *v, int *b, int *aij)
{
	int i, j;
	
	{
	ofstream f(fname);
	

	f << "<HTDO type=pt ptanz=" << m << " btanz=" << n << " fuse=simple>" << endl;
	f << "        ";
	for (j = 0; j < n; j++) {
		f << setw(8) << b[j] << " ";
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << setw(8) << v[i];
		for (j = 0; j < n; j++) {
			f << setw(8) << aij[i * n + j] << " ";
			}
		f << endl;
		}
	f << endl;
	for (i = 0; i < m; i++) {
		f << setw(3) << 1;
		}
	f << endl;
	f << "</HTDO>" << endl;
	}
	cout << "written file " << fname << " of size " << file_size(fname) << endl;
}



