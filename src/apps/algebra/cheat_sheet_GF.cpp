// cheat_sheet_GF.cpp
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




int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_poly = FALSE;
	char *poly = NULL;
	int f_q = FALSE;
	int q = 0;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
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

	char fname[1000];
	char title[1000];
	char author[1000];

	sprintf(fname, "GF_%d.tex", q);
	sprintf(title, "Cheat Sheet GF($%d$)", q);
	//sprintf(author, "");
	author[0] = 0;

	finite_field F;

	if (f_poly) {
		F.init_override_polynomial(q, poly, verbose_level);
	}
	else {
		F.init(q, 0 /* verbose_level */);
	}

	{
	ofstream f(fname);

	
	//algebra_global AG;

	//AG.cheat_sheet_GF(q, f_override_poly, my_override_poly, verbose_level);
	latex_interface L;

	//F.init(q), verbose_level - 2);

	L.head(f, FALSE /* f_book*/, TRUE /* f_title */,
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	F.cheat_sheet(f, verbose_level);

	F.cheat_sheet_tables(f, verbose_level);

	int *power_table;
	int t;
	int len = q;

	t = F.primitive_root();

	power_table = NEW_int(len);
	F.power_table(t, power_table, len);

	f << "\\begin{multicols}{2}" << endl;
	f << "\\noindent" << endl;
	for (i = 0; i < len; i++) {
		if (F.e == 1) {
			f << "$" << t << "^{" << i << "} \\equiv " << power_table[i] << "$\\\\" << endl;
		}
		else {
			f << "$" << t << "^{" << i << "} = " << power_table[i] << "$\\\\" << endl;
		}
	}
	f << "\\end{multicols}" << endl;
	FREE_int(power_table);


	
	L.foot(f);
	}

	file_io Fio;

	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

	the_end(t0);
}





