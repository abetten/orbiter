// cheat_sheet_PG.cpp
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
	int f_override_poly = FALSE;
	char *my_override_poly = NULL;
	int f_n = FALSE;
	int n = 0;
	int f_q = FALSE;
	int q = 0;
	int f_surface = FALSE;
	os_interface Os;

 	t0 = Os.os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_override_poly = TRUE;
			my_override_poly = argv[++i];
			cout << "-poly " << my_override_poly << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-surface") == 0) {
			f_surface = TRUE;
			cout << "-surface " << endl;
			}
		}
	
	if (!f_n) {
		cout << "please use -n option to specify n" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option to specify q" << endl;
		exit(1);
		}
	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init_override_polynomial(q, my_override_poly, 0);

	//F->cheat_sheet_PG(n, f_surface, verbose_level);

	//const char *override_poly;
	char fname[1000];
	char title[1000];
	char author[1000];
	//int f_with_group = FALSE;
	//int f_semilinear = FALSE;
	//int f_basis = TRUE;
	//int q = F->q;

	sprintf(fname, "PG_%d_%d.tex", n, q);
	sprintf(title, "Cheat Sheet PG($%d,%d$)", n, q);
	//sprintf(author, "");
	author[0] = 0;
	projective_space *P;

	P = NEW_OBJECT(projective_space);
	cout << "before P->init" << endl;
	P->init(n, F,
		TRUE /* f_init_incidence_structure */,
		verbose_level/*MINIMUM(2, verbose_level)*/);


	{
	ofstream f(fname);
	latex_interface L;

	L.head(f,
			FALSE /* f_book*/,
			TRUE /* f_title */,
			title, author,
			FALSE /* f_toc */,
			FALSE /* f_landscape */,
			TRUE /* f_12pt */,
			TRUE /* f_enlarged_page */,
			TRUE /* f_pagenumbers */,
			NULL /* extra_praeamble */);


	P->report(f);

	if (f_surface) {
		surface_domain *S;

		S = NEW_OBJECT(surface_domain);
		S->init(F, verbose_level + 2);

		f << "\\clearpage" << endl << endl;
		f << "\\section{Surface}" << endl;
		f << "\\subsection{Steiner Trihedral Pairs}" << endl;
		S->latex_table_of_trihedral_pairs(f);

		f << "\\clearpage" << endl << endl;
		f << "\\subsection{Eckardt Points}" << endl;
		S->latex_table_of_Eckardt_points(f);

#if 1
		long int *Lines;

		cout << "creating S_{3,1}:" << endl;
		Lines = NEW_lint(27);
		S->create_special_double_six(Lines,
				3 /*a*/, 1 /*b*/, 0 /* verbose_level */);
		S->create_remaining_fifteen_lines(Lines,
				Lines + 12, 0 /* verbose_level */);
		P->Grass_lines->print_set(Lines, 27);

		FREE_lint(Lines);
#endif
		FREE_OBJECT(S);
		}


	L.foot(f);
	}
	file_io Fio;

	cout << "written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	
	FREE_OBJECT(P);

	FREE_OBJECT(F);

	the_end(t0);
}

