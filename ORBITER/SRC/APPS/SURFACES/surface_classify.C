// surface_classify.C
// 
// Anton Betten
// September 1, 2016
//
//
// 
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

int main(int argc, const char **argv);
INT callback_check_surface(INT len, INT *S, void *data, INT verbose_level);

int main(int argc, const char **argv)
{
	t0 = os_ticks();
	

	//start_memory_debug();
	
	{
	finite_field *F;
	linear_group_description *Descr;
	linear_group *LG;


	INT verbose_level = 0;
	INT f_linear = FALSE;
	INT f_report = FALSE;
	INT f_read_double_sixes = FALSE;
	INT f_double_sixes_only = FALSE;
	INT f_read_surfaces = FALSE;
	INT q;
	


	INT i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = new linear_group_description;
			i += Descr->read_arguments(argc - (i - 1), argv + i, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report" << endl;
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
		}

	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}

	INT f_v = (verbose_level >= 1);
	

	F = new finite_field;
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;
	


	LG = new linear_group;
	if (f_v) {
		cout << "surface_classify before LG->init, creating the group" << endl;
		}

	LG->init(Descr, verbose_level - 1);
	
	if (f_v) {
		cout << "surface_classify after LG->init" << endl;
		}

	surface_classify_wedge *SCW;

	SCW = new surface_classify_wedge;

	if (f_v) {
		cout << "surface_classify before SCW->init" << endl;
		}
	
	SCW->init(F, LG, argc, argv, verbose_level - 1);

	if (f_v) {
		cout << "surface_classify after SCW->init" << endl;
		}

	if (f_read_double_sixes) {


		{
		BYTE fname[1000];
	
		sprintf(fname, "Double_sixes_q%ld.data", q);
		cout << "Reading file " << fname << " of size " << file_size(fname) << endl;
		{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->Classify_double_sixes->read_file" << endl;
			}
		SCW->Classify_double_sixes->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_double_sixes->read_file" << endl;
			}
		}
		}



		}

	else {
	
		if (f_v) {
			cout << "surface_classify before SCW->Classify_double_sixes->classify_partial_ovoids" << endl;
			}
		SCW->Classify_double_sixes->classify_partial_ovoids(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_double_sixes->classify_partial_ovoids" << endl;
			}

		if (f_v) {
			cout << "surface_classify before SCW->Classify_double_sixes->classify" << endl;
			}
		SCW->Classify_double_sixes->classify(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_double_sixes->classify" << endl;
			}



		{
		BYTE fname[1000];
	
		sprintf(fname, "Double_sixes_q%ld.data", q);
		{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before SCW->Classify_double_sixes->write_file" << endl;
			}
		SCW->Classify_double_sixes->write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->Classify_double_sixes->write_file" << endl;
			}
		}
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
		
		}


	if (f_v) {
		cout << "surface_classify writing cheat sheet on double sixes" << endl;
		}
	{
	BYTE fname[1000];
	BYTE title[1000];
	BYTE author[1000];

	sprintf(title, "Cheat Sheet on Double Sixes over GF(%ld) ", q);
	sprintf(author, "");
	sprintf(fname, "Double_sixes_q%ld.tex", q);

		{
		ofstream fp(fname);
		
		//latex_head_easy(fp);
		latex_head(fp, FALSE /* f_book */, TRUE /* f_title */, 
			title, author, 
			FALSE /*f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */, 
			TRUE /*f_enlarged_page */, TRUE /* f_pagenumbers*/, 
			NULL /* extra_praeamble */);


		SCW->Classify_double_sixes->print_five_plus_ones(fp);

		SCW->Classify_double_sixes->Double_sixes->print_latex(fp, 
			"Double Sixes", FALSE /* f_with_stabilizers*/);

		latex_foot(fp);
		}
	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify writing cheat sheet on double sixes done" << endl;
		}

	if (f_double_sixes_only) {
		exit(0);
		}

#if 0
	INT coeff_in[20] = 
		{ 0, 0, 0, 0, 12, 0, 0, 12, 0, 0, 0, 0, 9, 0, 0, 10, 10, 6, 0, 0};
		//{0, 0, 0, 0, 12, 0, 0, 12, 0, 0, 0, 0, 9, 0, 0, 10, 10, 6, 0, 0};//= 12X^2Y + 12XY^2 + 9Z^2W + 10ZW^2 + 10YZW + 6XZW
		//{1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0};
	INT coeff_target[20] = 
		{  0, 0, 0, 0, 9, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 9, 9, 6, 11, 9 };
	
	INT coeff_out[20];

#if 0
	INT Mtx[16] = 
		{
0,2,0,0,
1,0,0,0,
0,0,1,0,
0,0,0,1
#if 0
 8,  7,  1,  2, 
12,  2,  4,  3, 
 4,  2,  3,  1, 
 2,  8,  5,  4 
#endif
		};
#endif

	INT Iso1[16] = {
 6, 11, 11, 12, 
 2, 11,  2,  0, 
 1,  4, 12,  0, 
 8,  0,  8,  4, 
		};
	INT Iso2[16] = {
12,  9, 10, 12, 
11,  2, 11,  0, 
 6,  3,  5,  0, 
 9, 10,  9,  3, 
		};
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	INT *Elt_v;

	Elt1 = NEW_INT(SCW->A->elt_size_in_INT);
	Elt2 = NEW_INT(SCW->A->elt_size_in_INT);
	Elt3 = NEW_INT(SCW->A->elt_size_in_INT);
	Elt4 = NEW_INT(SCW->A->elt_size_in_INT);
	Elt_v = NEW_INT(SCW->A->elt_size_in_INT);

	PG_element_normalize_from_front(*SCW->F, coeff_target, 1, SCW->Surf->nb_monomials);


	SCW->A->make_element(Elt1, Iso1, 0);
	SCW->A->make_element(Elt2, Iso2, 0);
	SCW->A->element_invert(Elt2, Elt3, 0);
	SCW->A->element_mult(Elt1, Elt3, Elt4, 0);
	SCW->A->element_invert(Elt4, Elt_v, 0);
	SCW->Surf->substitute_linear(coeff_in, coeff_out, Elt_v, 0 /* verbose_level */);
	cout << "Elt4:" << endl;
	INT_matrix_print(Elt4, 4, 4);
	cout << "Elt_v:" << endl;
	SCW->A->element_print_quick(Elt_v, cout);
	
	cout << "coeff_in :" << endl;
	INT_vec_print(cout, coeff_in, 20);
	cout << endl;

	PG_element_normalize_from_front(*SCW->F, coeff_out, 1, SCW->Surf->nb_monomials);

	cout << "coeff_out :" << endl;
	INT_vec_print(cout, coeff_out, 20);
	cout << endl;
	cout << "coeff_target :" << endl;
	INT_vec_print(cout, coeff_out, 20);
	cout << endl;
	exit(1);
#endif
	


	if (f_read_surfaces) {


		{
		BYTE fname[1000];
	
		sprintf(fname, "Surfaces_q%ld.data", q);
		cout << "Reading file " << fname << " of size " << file_size(fname) << endl;
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

		if (f_v) {
			cout << "surface_classify before SCW->classify_surfaces_from_double_sixes" << endl;
			}
		SCW->classify_surfaces_from_double_sixes(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->classify_surfaces_from_double_sixes" << endl;
			}

		{
		BYTE fname[1000];
	
		sprintf(fname, "Surfaces_q%ld.data", q);
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
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}


		}



	if (f_report) {
		{
		BYTE fname[1000];
		BYTE title[1000];
		BYTE author[1000];
		INT f_with_stabilizers = TRUE;

		sprintf(title, "Cubic Surfaces with 27 Lines over GF(%ld) ", q);
		sprintf(author, "");
		sprintf(fname, "Surfaces_q%ld.tex", q);

			{
			ofstream fp(fname);
		
			//latex_head_easy(fp);
			latex_head(fp, FALSE /* f_book */, TRUE /* f_title */, 
				title, author, 
				FALSE /*f_toc */, FALSE /* f_landscape */, FALSE /* f_12pt */, 
				TRUE /*f_enlarged_page */, TRUE /* f_pagenumbers*/, 
				NULL /* extra_praeamble */);


			SCW->latex_surfaces(fp, f_with_stabilizers);

			latex_foot(fp);
			}
		cout << "Written file " << fname << " of size " << file_size(fname) << endl;
		}
		}

#if 1
	if (SCW->nb_identify) {
		if (f_v) {
			cout << "surface_classify before SCW->identify_surfaces" << endl;
			}
		SCW->identify_surfaces(verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after SCW->identify_surfaces" << endl;
			}
		}
#endif

#if 1
	SCW->generate_source_code(verbose_level);
#endif


#if 0
	cout << "classify_surfaces, we are done but all data is still in memory, before doing a dump" << endl;
	registry_dump();
	cout << "classify_surfaces, we are done but all data is still in memory, after doing a dump" << endl;
#endif
	//SCW->identify_Sa_and_print_table(verbose_level);




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



	}

	cout << "classify_surfaces, before the_end" << endl;
	the_end(t0);
	//the_end_quietly(t0);
	cout << "classify_surfaces, after the_end" << endl;
}




