// cheat_sheet_PG.C
// 
// Anton Betten
// 3/14/2010
//
//
// 
//
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started



void cheat_sheet_PG(INT n, finite_field *F, INT f_surface, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_override_poly = FALSE;
	BYTE *my_override_poly = NULL;
	INT f_n = FALSE;
	INT n = 0;
	INT f_q = FALSE;
	INT q = 0;
	INT f_surface = FALSE;
	
 	t0 = os_ticks();
	
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
	F = new finite_field;
	F->init_override_polynomial(q, my_override_poly, 0);
	cheat_sheet_PG(n, F, f_surface, verbose_level);
	
	the_end(t0);
}

void cheat_sheet_PG(INT n, finite_field *F, INT f_surface, INT verbose_level)
{
	//const BYTE *override_poly;
	BYTE fname[1000];
	BYTE title[1000];
	BYTE author[1000];
	//INT f_with_group = FALSE;
	//INT f_semilinear = FALSE;
	//INT f_basis = TRUE;
	INT q = F->q;
	
	sprintf(fname, "PG_%ld_%ld.tex", n, q);
	sprintf(title, "Cheat Sheet PG($%ld,%ld$)", n, q);
	//sprintf(author, "");
	author[0] = 0;
	projective_space *P;

	P = new projective_space;
	cout << "before P->init" << endl;
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		verbose_level/*MINIMUM(2, verbose_level)*/);

	
	{
	ofstream f(fname);
	
	latex_head(f, FALSE /* f_book*/, TRUE /* f_title */, 
		title, author, FALSE /* f_toc */, FALSE /* f_landscape */,
			TRUE /* f_12pt */, 
			TRUE /* f_enlarged_page */, 
			TRUE /* f_pagenumbers */, 
			NULL /* extra_praeamble */);

	f << "\\small" << endl;
	f << "\\arraycolsep=2pt" << endl;
	f << "\\parindent=0pt" << endl;
	f << "$q = " << F->q << "$\\\\" << endl;
	f << "$p = " << F->p << "$\\\\" << endl;
	f << "$e = " << F->e << "$\\\\" << endl;
	f << "$n = " << n << "$\\\\" << endl;
	f << "Number of points = " << P->N_points << "\\\\" << endl;
	f << "Number of lines = " << P->N_lines << "\\\\" << endl;
	f << "Number of lines on a point = " << P->r << "\\\\" << endl;
	f << "Number of points on a line = " << P->k << "\\\\" << endl;

	f << "\\clearpage" << endl << endl;
	f << "\\section{The Finite Field with $" << q << "$ Elements}" << endl;
	F->cheat_sheet(f, verbose_level);

	if (n == 2) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{The Plane}" << endl;
		BYTE fname_base[1000];
		INT *set;
		INT i;
		INT rad = 17000;

		set = NEW_INT(P->N_points);
		for (i = 0; i < P->N_points; i++) {
			set[i] = i;
			}
		sprintf(fname_base, "plane_of_order_%ld", F->q);
		P->draw_point_set_in_plane(fname_base, set, P->N_points, TRUE /*f_with_points*/, TRUE /*f_point_labels*/, FALSE /*f_embedded*/, FALSE /*f_sideways*/, rad, 0 /* verbose_level */);
		FREE_INT(set);
		f << "{\\scriptsize" << endl;
		f << "$$" << endl;
		f << "\\input " << fname_base << ".tex" << endl;
		f << "$$" << endl;
		f << "}%%" << endl;
		}

	f << "\\clearpage" << endl << endl;
	f << "\\section{Points and Lines}" << endl;
	P->cheat_sheet_points(f, verbose_level);
	
	P->cheat_sheet_point_table(f, verbose_level);
	


	f << "\\clearpage" << endl << endl;
	P->cheat_sheet_points_on_lines(f, verbose_level);
	
	f << "\\clearpage" << endl << endl;
	P->cheat_sheet_lines_on_points(f, verbose_level);


	// report subspaces:
	INT k;
	
	for (k = 1; k < n; k++) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Subspaces of dimension " << k << "}" << endl;
		P->cheat_sheet_subspaces(f, k, verbose_level);
		}
	


	if (n >= 2 && P->N_lines < 25) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Line intersections}" << endl;
		P->cheat_sheet_line_intersection(f, verbose_level);
		}


	if (n >= 2 && P->N_points < 25) {
		f << "\\clearpage" << endl << endl;
		f << "\\section{Line through point-pairs}" << endl;
		P->cheat_sheet_line_through_pairs_of_points(f, verbose_level);
		}

	if (f_surface) {
		surface *S;
	
		S = new surface;
		S->init(F, verbose_level + 2);
		
		f << "\\clearpage" << endl << endl;
		f << "\\section{Surface}" << endl;
		f << "\\subsection{Steiner Trihedral Pairs}" << endl;
		S->latex_table_of_trihedral_pairs(f);

		f << "\\clearpage" << endl << endl;
		f << "\\subsection{Eckardt Points}" << endl;
		S->latex_table_of_Eckardt_points(f);

#if 1
		INT *Lines;

		cout << "creating S_{3,1}:" << endl;
		Lines = NEW_INT(27);
		S->create_special_double_six(Lines, 3 /*a*/, 1 /*b*/, 0 /* verbose_level */);
		S->create_remaining_fifteen_lines(Lines, Lines + 12, 0 /* verbose_level */);
		P->Grass_lines->print_set(Lines, 27);

		FREE_INT(Lines);
#endif
		delete S;		
		}

	latex_foot(f);
	}
	cout << "written file " << fname << " of size " << file_size(fname) << endl;
}




