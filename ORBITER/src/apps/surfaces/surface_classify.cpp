// surface_classify.cpp
// 
// Anton Betten
// September 1, 2016
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

int main(int argc, const char **argv)
{
	t0 = os_ticks();
	

	//start_memory_debug();
	
	int f_memory_dump_at_end = FALSE;
	const char *memory_dump_at_end_fname = NULL;


	{
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	linear_group_description *Descr;
	linear_group *LG;


	int verbose_level = 0;
	int f_linear = FALSE;
	int f_report = FALSE;
	int f_report_5p1 = FALSE;
	int f_read_double_sixes = FALSE;
	int f_double_sixes_only = FALSE;
	int f_read_surfaces = FALSE;
	int q;
	int f_semilinear = FALSE;
	int f_draw_poset = FALSE;
	int f_draw_poset_full = FALSE;
	int f_automatic_memory_dump = FALSE;
	int automatic_dump_interval = 0;
	const char *automatic_dump_mask = NULL;
	int f_memory_dump_at_peak = FALSE;
	const char *memory_dump_at_peak_fname = NULL;
	int f_identify_Sa = FALSE;
	int f_isomorph = FALSE;
	surface_create_description *surface_descr_isomorph1 = NULL;
	surface_create_description *surface_descr_isomorph2 = NULL;
	int f_recognize = FALSE;
	surface_create_description *surface_descr_recognize = NULL;


	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = NEW_OBJECT(linear_group_description);
			i += Descr->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-isomorph") == 0) {
			f_isomorph = TRUE;
			cout << "-isomorph reading description of first surface" << endl;
			surface_descr_isomorph1 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph1->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			i += 2;
			cout << "the current argument is " << argv[i] << endl;
			cout << "-isomorph reading description of second surface" << endl;
			surface_descr_isomorph2 = NEW_OBJECT(surface_create_description);
			i += surface_descr_isomorph2->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			cout << "-isomorph" << endl;
			}
		else if (strcmp(argv[i], "-recognize") == 0) {
			f_recognize = TRUE;
			cout << "-recognize reading description of surface" << endl;
			surface_descr_recognize = NEW_OBJECT(surface_create_description);
			i += surface_descr_recognize->
					read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;
			i += 2;
			cout << "-recognize" << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
			}
		else if (strcmp(argv[i], "-report_5p1") == 0) {
			f_report_5p1 = TRUE;
			cout << "-report_5p1" << endl;
			}
		else if (strcmp(argv[i], "-read_double_sixes") == 0) {
			f_read_double_sixes = TRUE;
			cout << "-read_double_sixes" << endl;
			}
		else if (strcmp(argv[i], "-double_sixes_only") == 0) {
			f_double_sixes_only = TRUE;
			cout << "-double_sixes_only" << endl;
			}
		else if (strcmp(argv[i], "-read_surfaces") == 0) {
			f_read_surfaces = TRUE;
			cout << "-read_surfaces" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset" << endl;
			}
		else if (strcmp(argv[i], "-draw_poset_full") == 0) {
			f_draw_poset_full = TRUE;
			cout << "-draw_poset_full" << endl;
			}
		else if (strcmp(argv[i], "-memory_debug") == 0) {
			f_memory_debug = TRUE;
			cout << "-memory_debug" << endl;
			}
		else if (strcmp(argv[i], "-memory_debug_verbose_level") == 0) {
			memory_debug_verbose_level = atoi(argv[++i]);
			cout << "-memory_debug_verbose_level "
					<< memory_debug_verbose_level << endl;
			}
		else if (strcmp(argv[i], "-automatic_memory_dump") == 0) {
			f_automatic_memory_dump = TRUE;
			automatic_dump_interval = atoi(argv[++i]);
			automatic_dump_mask = argv[++i];
			cout << "-automatic_memory_dump " << automatic_dump_interval
				<< " " << automatic_dump_mask << endl;
			}
		else if (strcmp(argv[i], "-memory_dump_at_peak") == 0) {
			f_memory_dump_at_peak = TRUE;
			memory_dump_at_peak_fname = argv[++i];
			cout << "-memory_dump_at_peak "
					<< memory_dump_at_peak_fname << endl;
			}
		else if (strcmp(argv[i], "-memory_dump_at_end") == 0) {
			f_memory_dump_at_end = TRUE;
			memory_dump_at_end_fname = argv[++i];
			cout << "-memory_dump_at_end "
					<< memory_dump_at_end_fname << endl;
			}
		else if (strcmp(argv[i], "-memory_dump_cumulative") == 0) {
			global_mem_object_registry.accumulate_and_ignore_duplicates(
					verbose_level);
			cout << "-memory_dump_cumulative" << endl;
		}
		else if (strcmp(argv[i], "-identify_Sa") == 0) {
			f_identify_Sa = TRUE;
			cout << "-identify_Sa" << endl;
			}
	}



	//f_memory_debug = TRUE;
	//f_memory_debug_verbose = TRUE;

	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}

	if (f_automatic_memory_dump) {
		global_mem_object_registry.set_automatic_dump(
				automatic_dump_interval, automatic_dump_mask,
				verbose_level);
	}

	int f_v = (verbose_level >= 1);
	

	F = NEW_OBJECT(finite_field);
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;
	
	if (is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "surface_classify before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 1);
	
	if (f_v) {
		cout << "surface_classify after LG->init" << endl;
		}


	if (f_v) {
		cout << "surface_classify before Surf->init" << endl;
		}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0 /*verbose_level - 1*/);
	if (f_v) {
		cout << "surface_classify after Surf->init" << endl;
		}


	Surf_A = NEW_OBJECT(surface_with_action);



	if (f_v) {
		cout << "surface_classify before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, f_semilinear, 0 /*verbose_level*/);
	if (f_v) {
		cout << "surface_classify after Surf_A->init" << endl;
		}




	surface_classify_wedge *SCW;

	SCW = NEW_OBJECT(surface_classify_wedge);

	if (f_v) {
		cout << "surface_classify before SCW->init" << endl;
		}
	
	SCW->init(F, LG,
			f_semilinear, Surf_A,
			argc, argv,
			verbose_level - 1);

	if (f_v) {
		cout << "surface_classify after SCW->init" << endl;
		}

	if (f_read_double_sixes) {


		{
		char fname[1000];
	
		sprintf(fname, "Double_sixes_q%d.data", q);
		cout << "Reading file " << fname << " of size "
				<< file_size(fname) << endl;
		{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->Classify_"
					"double_sixes->read_file" << endl;
			}
		SCW->Classify_double_sixes->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_"
					"double_sixes->read_file" << endl;
			}
		}
		}



		}

	else {
	
		if (f_v) {
			cout << "surface_classify before SCW->Classify_"
					"double_sixes->classify_partial_ovoids" << endl;
			}
		SCW->Classify_double_sixes->classify_partial_ovoids(
			f_draw_poset,
			f_draw_poset_full, 
			f_report_5p1,
			verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_"
					"double_sixes->classify_partial_ovoids" << endl;
			}

		if (f_v) {
			cout << "surface_classify before SCW->Classify_"
					"double_sixes->classify" << endl;
			}
		SCW->Classify_double_sixes->classify(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_"
					"double_sixes->classify" << endl;
			}



		{
		char fname[1000];
	
		sprintf(fname, "Double_sixes_q%d.data", q);
		{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->Classify_"
					"double_sixes->write_file" << endl;
			}
		SCW->Classify_double_sixes->write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_"
					"double_sixes->write_file" << endl;
			}
		}
		cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
		
		}


	if (f_v) {
		cout << "surface_classify writing cheat sheet "
				"on double sixes" << endl;
		}
	{
	char fname[1000];
	char title[1000];
	char author[1000];

	sprintf(title, "Cheat Sheet on Double Sixes over GF(%d) ", q);
	sprintf(author, "");
	sprintf(fname, "Double_sixes_q%d.tex", q);

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


		SCW->Classify_double_sixes->print_five_plus_ones(fp);

		SCW->Classify_double_sixes->Double_sixes->print_latex(fp, 
			"Double Sixes", FALSE /* f_with_stabilizers*/);

		latex_foot(fp);
		}
	cout << "Written file " << fname << " of size "
			<< file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify writing cheat sheet on "
				"double sixes done" << endl;
		}


	if (f_double_sixes_only) {
		cout << "f_double_sixes_only is true so we terminate now." << endl;
		exit(0);
		}


	if (f_read_surfaces) {


		{
		char fname[1000];
	
		sprintf(fname, "Surfaces_q%d.data", q);
		cout << "Reading file " << fname << " of size "
				<< file_size(fname) << endl;
		{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->read_file" << endl;
			}
		SCW->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->read_file" << endl;
			}
		}
		}


		}
	else {

		cout << "surface_classify classifying surfaces" << endl;

		if (f_v) {
			cout << "surface_classify before SCW->classify_surfaces_"
					"from_double_sixes" << endl;
			}
		SCW->classify_surfaces_from_double_sixes(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->classify_surfaces_"
					"from_double_sixes" << endl;
			}

		{
		char fname[1000];
	
		sprintf(fname, "Surfaces_q%d.data", q);
		{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->write_file" << endl;
			}
		SCW->write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->write_file" << endl;
			}
		}
		cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		}


		}



	if (f_report) {
		{
		char fname[1000];
		char title[1000];
		char author[1000];
		int f_with_stabilizers = TRUE;

		sprintf(title, "Cubic Surfaces with 27 Lines over GF(%d) ", q);
		sprintf(author, "");
		sprintf(fname, "Surfaces_q%d.tex", q);

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


			SCW->latex_surfaces(fp, f_with_stabilizers);

			latex_foot(fp);
			}
		cout << "Written file " << fname << " of size "
				<< file_size(fname) << endl;
		}
		}

#if 1
	if (SCW->nb_identify) {
		if (f_v) {
			cout << "surface_classify before SCW->"
					"identify_surfaces" << endl;
			}
		SCW->identify_surfaces(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->"
					"identify_surfaces" << endl;
			}
		}
#endif

#if 1
	SCW->generate_source_code(verbose_level);
#endif


#if 0
	cout << "classify_surfaces, we are done but all data is "
			"still in memory, before doing a dump" << endl;
	registry_dump();
	cout << "classify_surfaces, we are done but all data is "
			"still in memory, after doing a dump" << endl;
#endif

	if (f_identify_Sa) {
		SCW->identify_Sa_and_print_table(verbose_level);
	}
	if (f_isomorph) {
		cout << "isomorph" << endl;

		surface_create *SC1;
		surface_create *SC2;
		SC1 = NEW_OBJECT(surface_create);
		SC2 = NEW_OBJECT(surface_create);

		cout << "before SC1->init" << endl;
		SC1->init(surface_descr_isomorph1, Surf_A, verbose_level);
		cout << "after SC1->init" << endl;

		cout << "before SC2->init" << endl;
		SC2->init(surface_descr_isomorph2, Surf_A, verbose_level);
		cout << "after SC2->init" << endl;

		int isomorphic_to1;
		int isomorphic_to2;
		int *Elt_isomorphism_1to2;

		Elt_isomorphism_1to2 = NEW_int(SCW->A->elt_size_in_int);
		if (SCW->isomorphism_test_pairwise(
				SC1, SC2,
				isomorphic_to1, isomorphic_to2,
				Elt_isomorphism_1to2,
				verbose_level)) {
			cout << "The surfaces are isomorphic, "
					"an isomorphism is given by" << endl;
			SCW->A->element_print(Elt_isomorphism_1to2, cout);
			cout << "The surfaces belongs to iso type "
					<< isomorphic_to1 << endl;
		} else {
			cout << "The surfaces are NOT isomorphic." << endl;
			cout << "surface 1 belongs to iso type "
					<< isomorphic_to1 << endl;
			cout << "surface 2 belongs to iso type "
					<< isomorphic_to2 << endl;
		}
	}
	if (f_recognize) {
		cout << "recognize" << endl;

		surface_create *SC;
		strong_generators *SG;
		strong_generators *SG0;

		SC = NEW_OBJECT(surface_create);

		cout << "before SC->init" << endl;
		SC->init(surface_descr_recognize, Surf_A, verbose_level);
		cout << "after SC->init" << endl;

		int isomorphic_to;
		int *Elt_isomorphism;

		Elt_isomorphism = NEW_int(SCW->A->elt_size_in_int);
		SCW->identify_surface(
			SC->coeffs,
			isomorphic_to, Elt_isomorphism,
			verbose_level - 1);
		cout << "surface belongs to iso type "
				<< isomorphic_to << endl;
		SG = NEW_OBJECT(strong_generators);
		SG0 = NEW_OBJECT(strong_generators);
		if (f_v) {
			cout << "before SG->generators_"
					"for_the_stabilizer_of_the_cubic_surface" << endl;
			}
		SG->generators_for_the_stabilizer_of_the_cubic_surface(
			Surf_A->A,
			F, isomorphic_to,
			verbose_level);
		SG0->init_generators_for_the_conjugate_group_aGav(
				SG, Elt_isomorphism, verbose_level);
		longinteger_object go;

		SG0->group_order(go);
		cout << "The full stabilizer has order " << go << endl;
		cout << "And is generated by" << endl;
		SG0->print_generators_tex(cout);
	}


#if 0
	if (f_v) {
		cout << "surface_classify before SCW->print_surfaces" << endl;
		}
	SCW->print_surfaces();
	if (f_v) {
		cout << "surface_classify after SCW->print_surfaces" << endl;
		}


	if (f_v) {
		cout << "surface_classify before SCW->derived_arcs" << endl;
		}
	SCW->derived_arcs(verbose_level - 1);
	if (f_v) {
		cout << "surface_classify after SCW->derived_arcs" << endl;
		}
#endif



	//registry_dump_sorted_by_size();
	//global_mem_object_registry.dump();

	if (f_memory_dump_at_peak) {
		global_mem_object_registry.manual_dump_with_file_name(
				memory_dump_at_peak_fname);
	}



	}

	cout << "classify_surfaces, before the_end" << endl;
	the_end(t0);
	//the_end_quietly(t0);
	cout << "classify_surfaces, after the_end" << endl;
	if (f_memory_dump_at_end) {
		global_mem_object_registry.manual_dump_with_file_name(
				memory_dump_at_end_fname);
	}
}




