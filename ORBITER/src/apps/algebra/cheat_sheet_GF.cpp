// cheat_sheet_GF.C
// 
// Anton Betten
// 3/14/2010
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started



void cheat_sheet_GF(int q,
		int f_override_polynomial,
		char *my_override_polynomial,
		int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_override_poly = FALSE;
	char *my_override_poly = NULL;
	int f_q = FALSE;
	int q = 0;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_override_poly = TRUE;
			my_override_poly = argv[++i];
			cout << "-poly " << my_override_poly << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	
	if (!f_q) {
		cout << "please use -q option to specify q" << endl;
		exit(1);
		}
	cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	
	
	the_end(t0);
}



void cheat_sheet_GF(int q,
		int f_override_polynomial,
		char *my_override_polynomial,
		int verbose_level)
{
	const char *override_poly;
	char fname[1000];
	char title[1000];
	char author[1000];
	
	sprintf(fname, "GF_%d.tex", q);
	sprintf(title, "Cheat Sheet GF($%d$)", q);
	//sprintf(author, "");
	author[0] = 0;
	if (f_override_polynomial) {
		override_poly = my_override_polynomial;
		}
	else {
		override_poly = NULL;
		}
	finite_field F;

	{
	ofstream f(fname);
	
	//F.init(q), verbose_level - 2);
	F.init_override_polynomial(q, override_poly, verbose_level);
	latex_head(f, FALSE /* f_book*/, TRUE /* f_title */, 
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */, 
			TRUE /* f_enlarged_page */, 	
			TRUE /* f_pagenumbers */, 
			NULL /* extra_praeamble */);


	F.cheat_sheet(f, verbose_level);
	
	F.cheat_sheet_tables(f, verbose_level);

	latex_foot(f);
	}
	cout << "written file " << fname << " of size " << file_size(fname) << endl;


	//F.compute_subfields(verbose_level);
}


