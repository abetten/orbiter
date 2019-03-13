/*
 * classify_cubic_curves.cpp
 *
 *  Created on: Mar 7, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started



int main(int argc, const char **argv)
{

	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;

	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
	}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
	}

	const char *starter_directory_name = "";
	char base_fname[1000];

	sprintf(base_fname, "cubic_curves_%d", q);

	int f_v = (verbose_level >= 1);

	int f_semilinear = FALSE;

	if (!is_prime(q)) {
		f_semilinear = TRUE;
	}
	finite_field *F;

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	cubic_curve *CC;

	CC = NEW_OBJECT(cubic_curve);

	CC->init(F, verbose_level);


	cubic_curve_with_action *CCA;

	CCA = NEW_OBJECT(cubic_curve_with_action);

	CCA->init(CC, f_semilinear, verbose_level);

	classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(classify_cubic_curves);


	CCC->init(CCA,
			starter_directory_name,
			base_fname,
			argc, argv,
			verbose_level);

	CCC->compute_starter(verbose_level);

	CCC->test_orbits(verbose_level);

	CCC->do_classify(verbose_level);

	int f_with_stabilizers = TRUE;


	if (f_v) {
		cout << "surface_classify writing cheat sheet "
				"on double sixes" << endl;
		}
	{
	char fname[1000];
	char title[1000];
	char author[1000];
	int *Pts;
	int *singular_Pts;
	int *type;

	Pts = NEW_int(CCA->CC->P->N_points);
	singular_Pts = NEW_int(CCA->CC->P->N_points);
	type = NEW_int(CCA->CC->P->N_lines);

	sprintf(title, "Cubic Curves in PG$(2,%d)$", q);
	sprintf(author, "");
	sprintf(fname, "Cubic_curves_q%d.tex", q);

		{
		ofstream fp(fname);

		//latex_head_easy(fp);
		latex_head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);

#if 0
		CCC->Curves->print_latex(fp, title, f_with_stabilizers);
#else
		fp << "\\subsection*{" << title << "}" << endl;



		{

		fp << "The order of the group is ";
		CCC->Curves->go.print_not_scientific(fp);
		fp << "\\\\" << endl;

		fp << "\\bigskip" << endl;
		}

		fp << "The group has " << CCC->Curves->nb_orbits << " orbits: \\\\" << endl;

		int i;
		longinteger_domain D;
		longinteger_object go1, ol, Ol;
		Ol.create(0);

		for (i = 0; i < CCC->Curves->nb_orbits; i++) {

			if (f_v) {
				cout << "Curve " << i << " / " << CCC->Curves->nb_orbits << ":" << endl;
			}

			CCC->Curves->Orbit[i].gens->group_order(go1);

			if (f_v) {
				cout << "stab order " << go1 << endl;
				}

			D.integral_division_exact(CCC->Curves->go, go1, ol);

			if (f_v) {
				cout << "orbit length " << ol << endl;
				}

			int *data;
			int *eqn;
			int nb_pts;
			int nb_singular_pts;

			data = CCC->Curves->Rep + i * CCC->Curves->representation_sz;
			eqn = data + 9;

			fp << "\\subsection*{Curve " << i << " / " << CCC->Curves->nb_orbits << "}" << endl;
			//fp << "$" << i << " / " << CCC->Curves->nb_orbits << "$ $" << endl;

			fp << "$";
			int_set_print_tex_for_inline_text(fp,
					data,
					9 /*CCC->Curves->representation_sz*/);
			fp << "_{";
			go1.print_not_scientific(fp);
			fp << "}$ orbit length $";
			ol.print_not_scientific(fp);
			fp << "$\\\\" << endl;

			fp << "\\begin{eqnarray*}" << endl;
			fp << "&&";


			CCA->CC->Poly->enumerate_points(eqn, Pts, nb_pts,
					verbose_level - 2);


			CC->Poly->print_equation_with_line_breaks_tex(fp,
					eqn,
					5 /* nb_terms_per_line */,
					"\\\\\n&&");
			fp << "\\end{eqnarray*}" << endl;

			fp << "The curve has " << nb_pts << " points.\\\\" << endl;


			CC->compute_singular_points(
					eqn, singular_Pts, nb_singular_pts,
					verbose_level);

			fp << "The curve has " << nb_singular_pts << " singular points.\\\\" << endl;


			CCA->CC->P->line_intersection_type(
					Pts, nb_pts /* set_size */,
					type, 0 /*verbose_level*/);
			// type[N_lines]

			fp << "The line type is $";
			classify C;
			C.init(type, CCA->CC->P->N_lines, FALSE, 0);
			C.print_naked_tex(fp, TRUE /* f_backwards*/);
			fp << ".$ \\\\" << endl;


			if (f_with_stabilizers) {
				//ost << "Strong generators are:" << endl;
				CCC->Curves->Orbit[i].gens->print_generators_tex(fp);
				D.add_in_place(Ol, ol);
				}


			}

		fp << "The overall number of objects is: " << Ol << "\\\\" << endl;
#endif

		latex_foot(fp);
		FREE_int(Pts);
		FREE_int(singular_Pts);
		FREE_int(type);
		}
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify writing cheat sheet on "
				"double sixes done" << endl;
		}




	the_end(t0);
}
