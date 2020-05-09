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
	os_interface Os;
	t0 = Os.os_ticks();
	int i;
	int verbose_level = 0;
	int f_draw_poset = FALSE;
	int f_draw_full_poset = FALSE;
	int f_embedded = FALSE;
	int f_report = FALSE;
	int f_linear = FALSE;
	linear_group_description *Descr = NULL;
	linear_group *LG = NULL;
	int f_sylow = FALSE;
	int f_recognize = FALSE;
	const char *recognize_set_ascii = NULL;

	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-linear" << endl;
		}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
		}
		else if (strcmp(argv[i], "-draw_full_poset") == 0) {
			f_draw_full_poset = TRUE;
			cout << "-draw_full_poset " << endl;
		}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
		}
		else if (strcmp(argv[i], "-sylow") == 0) {
			f_sylow = TRUE;
			cout << "-sylow " << endl;
		}
		else if (strcmp(argv[i], "-recognize") == 0) {
			f_recognize = TRUE;
			recognize_set_ascii = argv[++i];
			cout << "-recognize " << recognize_set_ascii << endl;
		}
	}


	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}


	finite_field *F;
	int f_v = (verbose_level >= 1);
	file_io Fio;
	int q;


	F = NEW_OBJECT(finite_field);
	q = Descr->input_q;

	if (f_v) {
		cout << "arcs_main q=" << q << endl;
	}

	if (Descr->f_override_polynomial) {
		cout << "creating finite field of order q=" << Descr->input_q
				<< " using override polynomial " << Descr->override_polynomial << endl;
		F->init_override_polynomial(Descr->input_q,
				Descr->override_polynomial, verbose_level);
	}
	else {
		cout << "creating finite field of order q=" << Descr->input_q << endl;
		F->init(Descr->input_q, 0);
	}
	Descr->F = F;
	//q = Descr->input_q;



	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "arcs_main before LG->init, "
				"creating the group" << endl;
	}

	LG->init(Descr, verbose_level - 1);

	if (f_v) {
		cout << "arcs_main after LG->init" << endl;
	}

	action *A;

	A = LG->A2;

	if (f_v) {
		cout << "arcs_main created group " << LG->prefix << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "arcs_main the group is not a matrix group " << endl;
		exit(1);
	}

	int f_semilinear;

	f_semilinear = A->is_semilinear_matrix_group();
	if (f_v) {
		cout << "arcs_main f_semilinear=" << f_semilinear << endl;
	}

	matrix_group *M;

	M = A->get_matrix_group();
	int dim = M->n;

	if (f_v) {
		cout << "arcs_main dim=" << dim << endl;
	}



	{
	arc_generator *Gen;
	int f_poset_classification_control = TRUE;
	poset_classification_control *Control;

	Control = NEW_OBJECT(poset_classification_control);

	//finite_field *F;
	//action *A;

	
	Gen = NEW_OBJECT(arc_generator);

	cout << argv[0] << endl;
	cout << "before Gen->read_arguments" << endl;
	Gen->read_arguments(argc, argv);
	

	cout << "before Gen->init" << endl;
	Gen->init(f_poset_classification_control, Control,
			F,
		A, LG->Strong_gens,
		Gen->ECA->input_prefix, 
		Gen->ECA->base_fname,
		Gen->ECA->starter_size, 
		//argc, argv,
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

		Gen->gen->draw_poset_fname_base_poset_lvl(fname_poset, Gen->ECA->starter_size);
#if 0
		sprintf(fname_poset, "arcs_%d_poset_%d",
				Gen->q, Gen->ECA->starter_size);
#endif
		Gen->gen->draw_poset(fname_poset,
				Gen->ECA->starter_size /*depth*/,
				0 /* data1 */,
				f_embedded /* f_embedded */,
				FALSE /* f_sideways */,
				verbose_level);
		}
	}
	if (f_draw_full_poset) {
		cout << "f_draw_full_poset verbose_level=" << verbose_level << endl;
		{
		char fname_flag_orbits[1000];

		Gen->gen->draw_poset_fname_base_poset_lvl(fname_flag_orbits, Gen->ECA->starter_size);
		strcat(fname_flag_orbits, "_flag_orbits");

		Gen->gen->make_flag_orbits_on_relations(
				Gen->ECA->starter_size, fname_flag_orbits, verbose_level);
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

		sprintf(title, "Arcs over GF(%d) ", q);
		sprintf(author, "Orbiter");
		sprintf(fname, "Arcs_q%d.tex", q);

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

			fp << "\\section{The field of order " << q << "}" << endl;
			fp << "\\noindent The field ${\\mathbb F}_{"
					<< Gen->q
					<< "}$ :\\\\" << endl;
			Gen->F->cheat_sheet(fp, verbose_level);

			fp << "\\section{The plane PG$(2, " << q << ")$}" << endl;

			fp << "The points in the plane PG$(2, " << q << ")$:\\\\" << endl;

			fp << "\\bigskip" << endl;


			Gen->P->cheat_sheet_points(fp, 0 /*verbose_level*/);


			int f_group_table = FALSE;
			double tikz_global_scale = 0.3;
			double tikz_global_line_width = 1.;
			int factor1000 = 1000;

			LG->report(fp, f_sylow, f_group_table,
					tikz_global_scale, tikz_global_line_width, factor1000,
					verbose_level);

			fp << endl;
			fp << "\\section{Poset Classification}" << endl;
			fp << endl;


			Gen->gen->report(fp);

			L.foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
		}
		if (f_recognize) {
			cout << "recognizing the set " << recognize_set_ascii << endl;
			long int *recognize_set;
			int recognize_set_sz;
			int *transporter;
			int *transporter_inv;
			int f_implicit_fusion = TRUE;
			int final_node = 0;

			lint_vec_scan(recognize_set_ascii, recognize_set, recognize_set_sz);
			cout << "set=";
			lint_vec_print(cout, recognize_set, recognize_set_sz);
			cout << endl;

			transporter = NEW_int(A->elt_size_in_int);
			transporter_inv = NEW_int(A->elt_size_in_int);
			Gen->gen->recognize(
					recognize_set, recognize_set_sz, transporter, f_implicit_fusion,
					final_node, verbose_level);
			cout << "final_node = " << final_node << endl;

			A->element_invert(transporter, transporter_inv, 0);

			cout << "transporter=" << endl;
			A->element_print(transporter, cout);
			cout << endl;

			cout << "transporter_inv=" << endl;
			A->element_print(transporter_inv, cout);
			cout << endl;

		}
	}


	FREE_OBJECT(Gen);
	//FREE_OBJECT(A);
	
	}

	FREE_OBJECT(F);

	cout << "Memory usage = " << Os.os_memory_usage()
			<<  " Time = " << Os.delta_time(t0)
			<< " tps = " << Os.os_ticks_per_second() << endl;
	the_end(t0);
	//the_end_quietly(t0);
}



