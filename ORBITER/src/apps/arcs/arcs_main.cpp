// arcs_main.cpp
// 
// Anton Betten
//
// previous version Dec 6, 2004
// revised June 19, 2006
// revised Aug 17, 2008
//
// Searches for arcs in desarguesian projective planes
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	int i;
	int verbose_level = 0;
	int f_draw_poset = FALSE;
	int f_embedded = FALSE;
	int f_report = FALSE;

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-Report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
			}
	}
	{
	arc_generator *Gen;
	finite_field *F;
	action *A;

	
	Gen = NEW_OBJECT(arc_generator);

	cout << argv[0] << endl;
	cout << "before Gen->read_arguments" << endl;
	Gen->read_arguments(argc, argv);
	

	cout << "before creating the finite field" << endl;
	F = NEW_OBJECT(finite_field);

	if (Gen->f_poly) {
		F->init_override_polynomial(Gen->q,
				Gen->poly, 0 /*verbose_level*/);
		}
	else {
		F->init(Gen->q, 0);
		}


	A = NEW_OBJECT(action);

	vector_ge *nice_gens;

	int f_semilinear = TRUE;
	number_theory_domain NT;

	if (NT.is_prime(F->q)) {
		f_semilinear = FALSE;
		}


	A->init_projective_group(3, F,
			f_semilinear, TRUE /*f_basis*/,
			nice_gens,
			0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);


	cout << "before Gen->init" << endl;
	Gen->init(F, 
		A,
		Gen->ECA->input_prefix, 
		Gen->ECA->base_fname,
		Gen->ECA->starter_size, 
		argc, argv, 
		Gen->verbose_level);
	cout << "after Gen->init" << endl;
	


	cout << "before Gen->main" << endl;
	Gen->main(Gen->verbose_level);
	cout << "after Gen->main" << endl;

	if (Gen->f_starter) {
			cout << "preparing level spreadsheet" << endl;
			{
			spreadsheet *Sp;
			Gen->gen->make_spreadsheet_of_level_info(
					Sp, Gen->ECA->starter_size, Gen->verbose_level);
			char fname_csv[1000];
			sprintf(fname_csv, "arcs_%d_%d_level.csv",
					Gen->q, Gen->ECA->starter_size);
			Sp->save(fname_csv, Gen->verbose_level);
			delete Sp;
			}
			cout << "preparing orbit spreadsheet" << endl;
			{
			spreadsheet *Sp;
			Gen->gen->make_spreadsheet_of_orbit_reps(
					Sp, Gen->ECA->starter_size);
			char fname_csv[1000];
			sprintf(fname_csv, "arcs_%d_%d.csv",
					Gen->q, Gen->ECA->starter_size);
			Sp->save(fname_csv, Gen->verbose_level);
			delete Sp;
			}
			cout << "preparing orbit spreadsheet done" << endl;

	}

	if (f_draw_poset) {
		cout << "f_draw_poset verbose_level=" << verbose_level << endl;
		{
		char fname_poset[1000];
		sprintf(fname_poset, "arcs_%d_poset_%d",
				Gen->q, Gen->ECA->starter_size);
		Gen->gen->draw_poset(fname_poset,
				Gen->ECA->starter_size /*depth*/,
				0 /* data1 */,
				f_embedded /* f_embedded */,
				FALSE /* f_sideways */,
				verbose_level);
		}
	}
	if (f_report) {
		cout << "doing a report" << endl;

		file_io Fio;

		{
		char fname[1000];
		char title[1000];
		char author[1000];
		//int f_with_stabilizers = TRUE;

		sprintf(title, "Arcs over GF(%d) ", Gen->q);
		sprintf(author, "Orbiter");
		sprintf(fname, "Arcs_q%d.tex", Gen->q);

			{
			ofstream fp(fname);
			latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				FALSE /* f_book */,
				TRUE /* f_title */,
				title, author,
				FALSE /*f_toc */,
				FALSE /* f_landscape */,
				FALSE /* f_12pt */,
				TRUE /*f_enlarged_page */,
				TRUE /* f_pagenumbers*/,
				NULL /* extra_praeamble */);

			fp << "\\section{The field of order " << Gen->q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< Gen->q
					<< "}$ :\\\\" << endl;
			Gen->F->cheat_sheet(fp, verbose_level);

			Gen->gen->report(fp);

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
	}


	FREE_OBJECT(Gen);
	FREE_OBJECT(F);
	
	}
	cout << "Memory usage = " << os_memory_usage()
			<<  " Time = " << delta_time(t0)
			<< " tps = " << os_ticks_per_second() << endl;
	the_end(t0);
	//the_end_quietly(t0);
}



