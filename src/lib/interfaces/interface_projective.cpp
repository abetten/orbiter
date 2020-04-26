/*
 * interface_projective.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace interfaces {


static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_on_parabola_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level);
static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level);
void polynomial_orbits_callback_print_function(
		stringstream &ost, void *data, void *callback_data);
void polynomial_orbits_callback_print_function2(
		stringstream &ost, void *data, void *callback_data);


interface_projective::interface_projective()
{
	argc = 0;
	argv = NULL;

	f_cheat_sheet_PG = FALSE;
	n = 0;
	q = 0;

	f_canonical_form_PG = FALSE;

	f_input = FALSE;
	Data_input_stream = NULL;

	f_all_k_subsets = FALSE;
	k = 0;

	f_save_incma_in_and_out = FALSE;

	f_prefix = FALSE;
	prefix = "";

	f_save = FALSE;
	output_prefix = "";

	fixed_structure_order_list_sz = 0;
	//fixed_structure_order_list[1000];

	f_report = FALSE;

	f_max_TDO_depth = FALSE;
	max_TDO_depth = INT_MAX;

	f_classify_cubic_curves = FALSE;

	f_create_points_on_quartic = FALSE;
	desired_distance = 0;

	f_create_points_on_parabola = FALSE;

	f_smooth_curve = FALSE;

	parabola_N = 0;
	parabola_a = 0;
	parabola_b = 0.;
	parabola_c = 0.;

	smooth_curve_N = 0;
	FP_descr = NULL;
	smooth_curve_t_min = 0;
	smooth_curve_t_max = 0;
	smooth_curve_boundary = 0;
	smooth_curve_Polish = NULL;
	smooth_curve_label = NULL;

	f_create_BLT_set = FALSE;
	BLT_set_descr = NULL;
	nb_transform = 0;
	//const char *transform_coeffs[1000];
	//int f_inverse_transform[1000];

	f_create_surface = FALSE;
	surface_description = NULL;
	f_surface_quartic = FALSE;
	f_surface_clebsch = FALSE;
	f_surface_codes = FALSE;

}


void interface_projective::print_help(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-cheat_sheet_PG") == 0) {
		cout << "-cheat_sheet_PG" << endl;
	}
	else if (strcmp(argv[i], "-canonical_form_PG") == 0) {
		cout << "-canonical_form_PG" << endl;
	}
	else if (strcmp(argv[i], "-classify_cubic_curves") == 0) {
		cout << "-classify_cubic_curves" << endl;
	}
	else if (strcmp(argv[i], "-create_points_on_quartic") == 0) {
		cout << "-create_points_on_quartic <double : desired_distance>" << endl;
	}
	else if (strcmp(argv[i], "-create_points_on_parabola") == 0) {
		cout << "-create_points_on_parabola <double : desired_distance> <double : a> <double : b> <double : c>" << endl;
	}
	else if (strcmp(argv[i], "-smooth_curve") == 0) {
		cout << "-smooth_curve <string : label> <double : desired_distance> <int : N> <double : boundary> <double : t_min> <double : t_max> <function>" << endl;
	}
	else if (strcmp(argv[i], "-create_BLT_set") == 0) {
		cout << "-create_BLT_set <description>" << endl;
	}
	else if (strcmp(argv[i], "-create_surface") == 0) {
		cout << "-create_surface <description>" << endl;
	}
}

int interface_projective::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-cheat_sheet_PG <int : n> < int : q >") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-canonical_form_PG <int : n> < int : q >") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-classify_cubic_curves < int : q >") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_points_on_quartic") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_points_on_parabola") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-smooth_curve") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_BLT_set") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-create_surface") == 0) {
		return true;
	}
	return false;
}

void interface_projective::read_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_projective::read_arguments" << endl;
	//return 0;

	interface_projective::argc = argc;
	interface_projective::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-cheat_sheet_PG") == 0) {
			f_cheat_sheet_PG = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-cheat_sheet_PG " << n << " " <<  q << endl;
			i++;
		}
		else if (strcmp(argv[i], "-canonical_form_PG") == 0) {
			f_canonical_form_PG = TRUE;
			n = atoi(argv[++i]);
			q = atoi(argv[++i]);
			cout << "-canonical_form_PG " << n << " " <<  q << endl;
			i++;
			i = read_canonical_form_arguments(argc, argv, i, verbose_level);

		}
		else if (strcmp(argv[i], "-classify_cubic_curves") == 0) {
			f_classify_cubic_curves = TRUE;
			q = atoi(argv[++i]);
			cout << "-classify_cubic_curves " <<  q << endl;
			i++;
			i = read_canonical_form_arguments(argc, argv, i, verbose_level);

		}
		else if (strcmp(argv[i], "-create_points_on_quartic") == 0) {
			f_create_points_on_quartic = TRUE;
			desired_distance = atof(argv[++i]);
			cout << "-create_points_on_quartic " << desired_distance << endl;
			i++;
		}
		else if (strcmp(argv[i], "-create_points_on_parabola") == 0) {
			f_create_points_on_parabola = TRUE;
			desired_distance = atof(argv[++i]);
			parabola_N = atoi(argv[++i]);
			parabola_a = atof(argv[++i]);
			parabola_b = atof(argv[++i]);
			parabola_c = atof(argv[++i]);
			cout << "-create_points_on_parabola " << desired_distance << " "
					<< parabola_N << " " << parabola_a << " "
					<< parabola_b << " " << parabola_c << endl;
			i++;
		}
		else if (strcmp(argv[i], "-smooth_curve") == 0) {
			f_smooth_curve = TRUE;
			smooth_curve_label = argv[++i];
			desired_distance = atof(argv[++i]);
			smooth_curve_N = atoi(argv[++i]);
			smooth_curve_boundary = atof(argv[++i]);
			smooth_curve_t_min = atof(argv[++i]);
			smooth_curve_t_max = atof(argv[++i]);

			FP_descr = NEW_OBJECT(function_polish_description);

			i += FP_descr->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-smooth_curve "
					<< smooth_curve_label << " "
					<< desired_distance << " "
					<< smooth_curve_N << " "
					<< smooth_curve_boundary << " "
					<< smooth_curve_t_min << " "
					<< smooth_curve_t_max << " "
					<< endl;
			i++;
		}
		else if (strcmp(argv[i], "-create_BLT_set") == 0) {
			f_create_BLT_set = TRUE;
			BLT_set_descr = NEW_OBJECT(BLT_set_create_description);
			i += BLT_set_descr->read_arguments(
					argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-create_BLT_set" << endl;
		}
		else if (strcmp(argv[i], "-create_surface") == 0) {
			f_create_surface = TRUE;
			surface_description = NEW_OBJECT(surface_create_description);
			i += surface_description->read_arguments(
					argc - (i - 1), argv + i,
					verbose_level) - 1;

			cout << "-create_surface" << endl;
		}
		else if (strcmp(argv[i], "-transform") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = FALSE;
			cout << "-transform " << transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
		else if (strcmp(argv[i], "-transform_inverse") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = TRUE;
			cout << "-transform_inverse "
					<< transform_coeffs[nb_transform] << endl;
			nb_transform++;
		}
		else if (strcmp(argv[i], "-surface_quartic") == 0) {
			f_surface_quartic = TRUE;
			cout << "-surface_quartic" << endl;
		}
		else if (strcmp(argv[i], "-surface_clebsch") == 0) {
			f_surface_clebsch = TRUE;
			cout << "=surface_clebsch" << endl;
		}
		else if (strcmp(argv[i], "-surface_codes") == 0) {
			f_surface_codes = TRUE;
			cout << "-surface_codes" << endl;
		}
	}
	cout << "interface_projective::read_arguments done" << endl;
}

int interface_projective::read_canonical_form_arguments(int argc,
		const char **argv, int i0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_projective::read_canonical_form_arguments" << endl;
	}
	for (i = i0; i < argc; i++) {


		if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-input" << endl;
		}


		else if (strcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = atoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
		}
		else if (strcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			cout << "-save_incma_in_and_out" << endl;
		}
		else if (strcmp(argv[i], "-prefix") == 0) {
			f_prefix = TRUE;
			prefix = argv[++i];
			cout << "-prefix " << prefix << endl;
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			output_prefix = argv[++i];
			cout << "-save " << output_prefix << endl;
		}
		else if (strcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
		}
		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}
		else if (strcmp(argv[i], "-canonical_form_PG_end") == 0) {
			cout << "-canonical_form_PG_end " << endl;
			i++;
			break;
		}
		else {
			cout << "-canonical_form_PG: unrecognized option " << argv[i] << endl;
			exit(1);
		}
	}
	if (f_v) {
		cout << "interface_projective::read_canonical_form_arguments done" << endl;
	}
	return i;
}

void interface_projective::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::worker" << endl;
	}

	if (f_cheat_sheet_PG) {
		do_cheat_sheet_PG(n, q, verbose_level);
	}
	else if (f_canonical_form_PG) {
		do_canonical_form_PG(n, q, verbose_level);
	}
	else if (f_classify_cubic_curves) {
		do_classify_cubic_curves(q, verbose_level);
	}
	else if (f_create_points_on_quartic) {
		do_create_points_on_quartic(desired_distance, verbose_level);
	}
	else if (f_create_points_on_parabola) {
		do_create_points_on_parabola(desired_distance,
				parabola_N, parabola_a, parabola_b, parabola_c, verbose_level);
	}
	else if (f_smooth_curve) {
		do_smooth_curve(smooth_curve_label,
				desired_distance, smooth_curve_N,
				smooth_curve_t_min, smooth_curve_t_max, smooth_curve_boundary,
				FP_descr, verbose_level);
	}
	else if (f_create_BLT_set) {
		do_create_BLT_set(BLT_set_descr, verbose_level);
	}
	else if (f_create_surface) {
		do_create_surface(surface_description, verbose_level);
	}
	if (f_v) {
		cout << "interface_projective::worker done" << endl;
	}
}


void interface_projective::do_cheat_sheet_PG(int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG" << endl;
	}
	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	//F->init_override_polynomial(q, my_override_poly, 0);

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

	if (n == 3) {
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

	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG done" << endl;
	}

}

void interface_projective::do_canonical_form_PG(int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_projective::do_canonical_form_PG" << endl;
	}

	if (!f_input) {
		cout << "please use option -input ... -end" << endl;
		exit(1);
	}

	finite_field *F;

	F = NEW_OBJECT(finite_field);

	F->init(q, 0);
	//F->init_override_polynomial(q, poly, 0);

	int f_semilinear;
	number_theory_domain NT;


	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}


	projective_space_with_action *PA;

	PA = NEW_OBJECT(projective_space_with_action);

	PA->init(F, n,
		f_semilinear,
		TRUE /*f_init_incidence_structure*/,
		0 /* verbose_level */);





	classify_bitvectors *CB;

	CB = NEW_OBJECT(classify_bitvectors);


	cout << "canonical_form.cpp before PA->classify_objects_using_nauty" << endl;
	PA->classify_objects_using_nauty(Data_input_stream,
		CB,
		f_save_incma_in_and_out, prefix,
		verbose_level - 1);
	cout << "canonical_form.cpp after PA->classify_objects_using_nauty" << endl;



	cout << "canonical_form.cpp We found " << CB->nb_types << " types" << endl;


	compute_and_print_ago_distribution_with_classes(cout,
			CB, verbose_level);


	cout << "canonical_form.cpp In the ordering of canonical forms, they are" << endl;
	CB->print_reps();
	cout << "We found " << CB->nb_types << " types:" << endl;
	for (i = 0; i < CB->nb_types; i++) {

		object_in_projective_space_with_action *OiPA;
		object_in_projective_space *OiP;

		cout << i << " / " << CB->nb_types << " is "
			<< CB->Type_rep[i] << " : " << CB->Type_mult[i] << " : ";
		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[i];
		OiP = OiPA->OiP;
		if (OiP->type != t_PAC) {
			OiP->print(cout);
		}

#if 0
		for (j = 0; j < rep_len; j++) {
			cout << (int) Type_data[i][j];
			if (j < rep_len - 1) {
				cout << ", ";
				}
			}
#endif
		cout << endl;
	}



	if (f_save) {
		cout << "Saving the classification with output prefix "
				<< output_prefix << endl;
		PA->save(output_prefix, CB, verbose_level);
		CB->save(output_prefix,
			OiPA_encode, OiPA_group_order,
			NULL /* void *global_data */,
			verbose_level);

#if 0
		void save(const char *prefix,
			void (*encode_function)(void *extra_data,
				int *&encoding, int &encoding_sz, void *global_data),
			void (*get_group_order_or_NULL)(void *extra_data,
				longinteger_object &go, void *global_data),
			void *global_data,
			int verbose_level);
#endif
	}




	if (f_report) {

		cout << "Producing a latex report:" << endl;

		char fname[1000];

		if (prefix == NULL) {
			cout << "please use option -prefix <prefix> to set the "
					"prefix for the tex file" << endl;
			exit(1);
			}
		sprintf(fname, "%s_classification.tex", prefix);


		PA->latex_report(fname,
				output_prefix,
				CB,
				f_save_incma_in_and_out,
				fixed_structure_order_list_sz,
				fixed_structure_order_list,
				max_TDO_depth,
				verbose_level);

	}// f_report


	if (f_v) {
		cout << "interface_projective::do_canonical_form_PG done" << endl;
	}
}

void interface_projective::do_classify_cubic_curves(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_classify_cubic_curves" << endl;
	}

	const char *starter_directory_name = "";
	char base_fname[1000];

	sprintf(base_fname, "cubic_curves_%d", q);


	int f_semilinear = FALSE;
	number_theory_domain NT;

	if (!NT.is_prime(q)) {
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


	char fname[1000];
	char title[1000];
	char author[1000];
	sprintf(title, "Cubic Curves in PG$(2,%d)$", q);
	sprintf(author, "");
	sprintf(fname, "Cubic_curves_q%d.tex", q);

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

		fp << "\\subsection*{" << title << "}" << endl;

		CCC->report(fp, verbose_level);

		L.foot(fp);
	}

	file_io Fio;

	cout << "Written file " << fname << " of size "
		<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "classify_cubic_curves writing cheat sheet on "
				"cubic curves done" << endl;
	}


	if (f_v) {
		cout << "interface_projective::do_classify_cubic_curves done" << endl;
	}
}


void interface_projective::do_create_points_on_quartic(double desired_distance, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_points_on_quartic" << endl;
	}

	double amin, amid, amax;
	//double epsilon = 0.001;
	int N = 200;
	int i;

	//a0 = 16. / 25.;
	//b0 = 16. / 25.;

	amin = 0;
	amid = 16. / 25.;
	amax = 100;

	int nb;

	{
		parametric_curve C1;
		parametric_curve C2;

		C1.init(2 /* nb_dimensions */,
				desired_distance,
				amin, amid,
				do_create_points_on_quartic_compute_point_function,
				this /* extra_data */,
				100. /* boundary */,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C1.Pts.size()=" << C1.Pts.size() << endl;


		C2.init(2 /* nb_dimensions */,
				desired_distance,
				amid, amax,
				do_create_points_on_quartic_compute_point_function,
				this /* extra_data */,
				100. /* boundary */,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C2.Pts.size()=" << C2.Pts.size() << endl;


		for (i = 0; i < C1.Pts.size(); i++) {
			cout << C1.Pts[i].t << " : " << C1.Pts[i].coords[0] << ", " << C1.Pts[i].coords[1] << endl;
		}

		double *Pts;
		int nb_pts;

		nb_pts = 4 * (C1.Pts.size() + C2.Pts.size());
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C2.Pts[i].coords[1];
			nb++;
		}
		file_io Fio;

		Fio.double_matrix_write_csv("points.csv", Pts, nb, 2);

		cout << "created curve 1 with " << C1.Pts.size() << " many points" << endl;
		cout << "created curve 2 with " << C2.Pts.size() << " many points" << endl;
	}
	cout << "created 4  curves with " << nb << " many points" << endl;



	if (f_v) {
		cout << "interface_projective::do_create_points_on_quartic done" << endl;
	}
}

void interface_projective::do_create_points_on_parabola(
		double desired_distance, int N,
		double a, double b, double c, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_points_on_parabola" << endl;
	}

	double amin, amax;
	double boundary;
	int i;

	amin = -10;
	amax = 3.08;
	boundary = 10;

	int nb;

	{
		parametric_curve C;

		C.init(2 /* nb_dimensions */,
				desired_distance,
				amin, amax,
				do_create_points_on_parabola_compute_point_function,
				this /* extra_data */,
				boundary,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C.Pts.size()=" << C.Pts.size() << endl;




		for (i = 0; i < C.Pts.size(); i++) {
			cout << C.Pts[i].t << " : " << C.Pts[i].coords[0] << ", " << C.Pts[i].coords[1] << endl;
		}

		{
		double *Pts;
		int nb_pts;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C.Pts[i].coords[1];
			nb++;
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "parabola_N%d_%lf_%lf_%lf_points.csv", N, parabola_a, parabola_b, parabola_c);

		Fio.double_matrix_write_csv(fname, Pts, nb, 2);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 6];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			Pts[nb * 6 + 0] = C.Pts[i].coords[0];
			Pts[nb * 6 + 1] = C.Pts[i].coords[1];
			Pts[nb * 6 + 2] = 0.;
			Pts[nb * 6 + 3] = 0.;
			Pts[nb * 6 + 4] = 0.;
			Pts[nb * 6 + 5] = 1.;
			nb++;
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "parabola_N%d_%lf_%lf_%lf_projection_from_center.csv", N, parabola_a, parabola_b, parabola_c);

		Fio.double_matrix_write_csv(fname, Pts, nb, 6);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		double x, y, H, f;
		double h = 1.;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 6];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			x = C.Pts[i].coords[0];
			y = C.Pts[i].coords[1];
			Pts[nb * 6 + 0] = x;
			Pts[nb * 6 + 1] = y;
			Pts[nb * 6 + 2] = 0.;

			H = sqrt(h * h + x * x + y * y);
			f = h / H;

			Pts[nb * 6 + 3] = x * f;
			Pts[nb * 6 + 4] = y * f;
			Pts[nb * 6 + 5] = 1. - f;
			nb++;
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "parabola_N%d_%lf_%lf_%lf_projection_from_sphere.csv",
				N, parabola_a, parabola_b, parabola_c);

		Fio.double_matrix_write_csv(fname, Pts, nb, 6);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		double x, y, H, f;
		double h = 1.;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 3];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			x = C.Pts[i].coords[0];
			y = C.Pts[i].coords[1];

			H = sqrt(h * h + x * x + y * y);
			f = h / H;

			Pts[nb * 3 + 0] = x * f;
			Pts[nb * 3 + 1] = y * f;
			Pts[nb * 3 + 2] = 1. - f;
			nb++;
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "parabola_N%d_%lf_%lf_%lf_points_projected.csv",
				N, parabola_a, parabola_b, parabola_c);

		Fio.double_matrix_write_csv(fname, Pts, nb, 3);

		cout << "created family of lines 1 with " << C.Pts.size() << " many lines" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}


	}
	cout << "created curve with " << nb << " many points" << endl;



	if (f_v) {
		cout << "interface_projective::do_create_points_on_parabola done" << endl;
	}
}

void interface_projective::do_smooth_curve(const char *curve_label,
		double desired_distance, int N,
		double t_min, double t_max, double boundary,
		function_polish_description *FP_descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int nb_dimensions;

	if (f_v) {
		cout << "interface_projective::do_smooth_curve" << endl;
	}

	smooth_curve_Polish = NEW_OBJECT(function_polish);

	if (f_v) {
		cout << "interface_projective::do_smooth_curve before smooth_curve_Polish->init_from_description" << endl;
	}
	smooth_curve_Polish->init_from_description(FP_descr, verbose_level);
	if (f_v) {
		cout << "interface_projective::do_smooth_curve after smooth_curve_Polish->init_from_description" << endl;
	}
#if 0
	if (smooth_curve_Polish->Variables.size() != 1) {
		cout << "interface_projective::do_smooth_curve number of variables should be 1, is "
				<< smooth_curve_Polish->Variables.size() << endl;
		exit(1);
	}
#endif
	nb_dimensions = smooth_curve_Polish->Entry.size();
	if (f_v) {
		cout << "interface_projective::do_smooth_curve nb_dimensions = " << nb_dimensions << endl;
	}


	{
		parametric_curve C;

		C.init(nb_dimensions,
				desired_distance,
				t_min, t_max,
				do_create_points_smooth_curve_compute_point_function,
				this /* extra_data */,
				boundary,
				N,
				verbose_level);

		cout << "after parametric_curve::init, C.Pts.size()=" << C.Pts.size() << endl;

		{
		double *Pts;
		int nb_pts;
		int i, j, nb;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * nb_dimensions];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			if (C.Pts[i].f_is_valid) {
				for (j = 0; j < nb_dimensions; j++) {
					Pts[nb * nb_dimensions + j] = C.Pts[i].coords[j];
				}
				nb++;
			}
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "function_%s_N%d_points.csv", curve_label, N);

		Fio.double_matrix_write_csv(fname, Pts, nb, nb_dimensions);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

		{
		double *Pts;
		int nb_pts;
		int i, j, nb, n;
		double d; // euclidean distance to the previous point
		numerics Num;

		nb_pts = C.Pts.size();
		n = 1 + nb_dimensions + 1;
		Pts = new double[nb_pts * n];
		nb = 0;
		for (i = 0; i < C.Pts.size(); i++) {
			if (C.Pts[i].f_is_valid) {
				Pts[nb * n + 0] = C.Pts[i].t;
				for (j = 0; j < nb_dimensions; j++) {
					Pts[nb * n + 1 + j] = C.Pts[i].coords[j];
				}
				if (nb) {
					d = Num.distance_euclidean(Pts + (nb - 1) * n + 1, Pts + nb * n + 1, 3);
				}
				else {
					d = 0;
				}
				Pts[nb * n + 1 + 4 + 0] = d;
				nb++;
			}
		}
		file_io Fio;
		char fname[1000];
		sprintf(fname, "function_%s_N%d_points_plus.csv", curve_label, N);

		Fio.double_matrix_write_csv(fname, Pts, nb, n);

		cout << "created curve 1 with " << C.Pts.size() << " many points" << endl;
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		delete [] Pts;
		}

	}

	if (f_v) {
		cout << "interface_projective::do_smooth_curve done" << endl;
	}
}


void interface_projective::do_create_BLT_set(BLT_set_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_BLT_set" << endl;
	}


	BLT_set_create *BC;
	//int j;

	BC = NEW_OBJECT(BLT_set_create);

	cout << "before BC->init" << endl;
	BC->init(Descr, verbose_level);
	cout << "after BC->init" << endl;


	if (nb_transform) {
		cout << "before BC->apply_transformations" << endl;
		BC->apply_transformations(transform_coeffs,
				f_inverse_transform, nb_transform, verbose_level);
		cout << "after BC->apply_transformations" << endl;
	}

	action *A;
	//int *Elt1;
	int *Elt2;

	A = BC->A;

	Elt2 = NEW_int(A->elt_size_in_int);



#if 0
	if (BC->f_has_group) {
		for (i = 0; i < BC->Sg->gens->len; i++) {
			cout << "Testing generator " << i << " / "
					<< BC->Sg->gens->len << endl;
			A->element_invert(BC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);


			cout << "Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
			}
		}
	else {
		cout << "We do not have information about the "
				"automorphism group" << endl;
		}
#endif

	latex_interface L;


	cout << "We have created the following BLT-set:" << endl;
	cout << "$$" << endl;
	L.int_set_print_tex(cout, BC->set, BC->q + 1);
	cout << endl;
	cout << "$$" << endl;

	if (BC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		BC->Sg->print_generators_tex(cout);
	}





	FREE_int(Elt2);

	FREE_OBJECT(BC);

	if (f_v) {
		cout << "interface_projective::do_create_BLT_set done" << endl;
	}

}

void interface_projective::do_create_surface(
		surface_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_surface" << endl;
	}

	int q;
	int i;
	int f_semilinear;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;

	q = Descr->get_q();
	cout << "q=" << q << endl;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	F = NEW_OBJECT(finite_field);
	F->init(q, 0);


	if (f_v) {
		cout << "interface_projective::do_create_surface before Surf->init" << endl;
		}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, 0/*verbose_level - 1*/);
	if (f_v) {
		cout << "interface_projective::do_create_surface after Surf->init" << endl;
		}

	Surf_A = NEW_OBJECT(surface_with_action);

	if (f_v) {
		cout << "interface_projective::do_create_surface before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, f_semilinear, 0 /*verbose_level*/);
	if (f_v) {
		cout << "interface_projective::do_create_surface after Surf_A->init" << endl;
		}


	surface_create *SC;
	SC = NEW_OBJECT(surface_create);

	cout << "before SC->init" << endl;
	SC->init(Descr, Surf_A, verbose_level);
	cout << "after SC->init" << endl;

	if (nb_transform) {
		cout << "interface_projective::do_create_surface "
				"before SC->apply_transformations" << endl;
		SC->apply_transformations(transform_coeffs,
				f_inverse_transform, nb_transform, verbose_level);
		cout << "interface_projective::do_create_surface "
				"after SC->apply_transformations" << endl;
		}

	int coeffs_out[20];
	action *A;
	//int *Elt1;
	int *Elt2;

	A = SC->Surf_A->A;

	Elt2 = NEW_int(A->elt_size_in_int);

	SC->F->init_symbol_for_print("\\omega");

	if (SC->F->e == 1) {
		SC->F->f_print_as_exponentials = FALSE;
	}

	SC->F->PG_element_normalize(SC->coeffs, 1, 20);

	cout << "interface_projective::do_create_surface "
			"We have created the following surface:" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;


	if (SC->f_has_group) {
		for (i = 0; i < SC->Sg->gens->len; i++) {
			cout << "Testing generator " << i << " / "
					<< SC->Sg->gens->len << endl;
			A->element_invert(SC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);



			matrix_group *M;

			M = A->G.matrix_grp;
			M->substitute_surface_equation(Elt2,
					SC->coeffs, coeffs_out, SC->Surf,
					verbose_level - 1);


			if (int_vec_compare(SC->coeffs, coeffs_out, 20)) {
				cout << "error, the transformation does not preserve "
						"the equation of the surface" << endl;
				exit(1);
			}
			cout << "Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
		}
	}
	else {
		cout << "We do not have information about "
				"the automorphism group" << endl;
	}


	cout << "We have created the surface " << SC->label_txt << ":" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);

		if (SC->f_has_nice_gens) {
			cout << "The stabilizer is generated by the following nice generators:" << endl;
			SC->nice_gens->print_tex(cout);

		}
	}

	if (SC->f_has_lines) {
		cout << "The lines are:" << endl;
		SC->Surf->Gr->print_set_tex(cout, SC->Lines, 27);


		surface_object *SO;

		SO = NEW_OBJECT(surface_object);
		if (f_v) {
			cout << "before SO->init" << endl;
			}
		SO->init(SC->Surf, SC->Lines, SC->coeffs,
				FALSE /*f_find_double_six_and_rearrange_lines */, verbose_level);
		if (f_v) {
			cout << "after SO->init" << endl;
			}

		char fname_points[1000];

		sprintf(fname_points, "surface_%s_points.txt", SC->label_txt);
		Fio.write_set_to_file(fname_points,
				SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
		cout << "Written file " << fname_points << " of size "
				<< Fio.file_size(fname_points) << endl;
	}
	else {
		cout << "The surface " << SC->label_txt
				<< " does not come with lines" << endl;
	}




	if (SC->f_has_group) {

		cout << "creating surface_object_with_action object" << endl;

		surface_object_with_action *SoA;

		SoA = NEW_OBJECT(surface_object_with_action);

		if (SC->f_has_lines) {
			cout << "creating surface using the known lines (which are "
					"arranged with respect to a double six):" << endl;
			SoA->init(SC->Surf_A,
				SC->Lines,
				SC->coeffs,
				SC->Sg,
				FALSE /*f_find_double_six_and_rearrange_lines*/,
				SC->f_has_nice_gens, SC->nice_gens,
				verbose_level);
			}
		else {
			cout << "creating surface from equation only "
					"(no lines):" << endl;
			SoA->init_equation(SC->Surf_A,
				SC->coeffs,
				SC->Sg,
				verbose_level);
			}
		cout << "The surface has been created." << endl;




		six_arcs_not_on_a_conic *Six_arcs;
		int *transporter;

		Six_arcs = NEW_OBJECT(six_arcs_not_on_a_conic);


		// classify six arcs not on a conic:

		cout << "Classifying six-arcs not on a conic:" << endl;

		action *A;

		A = NEW_OBJECT(action);


		int f_semilinear = TRUE;
		number_theory_domain NT;

		if (NT.is_prime(F->q)) {
			f_semilinear = FALSE;
			}

		{
			vector_ge *nice_gens;
			A->init_projective_group(3, F,
					f_semilinear, TRUE /*f_basis*/, TRUE /* f_init_sims */,
					nice_gens,
					0 /*verbose_level*/);
			FREE_OBJECT(nice_gens);
		}
		Six_arcs->init(SC->F,
				A,
			SC->Surf->P2,
			argc, argv,
			verbose_level);
		transporter = NEW_int(Six_arcs->Gen->A->elt_size_in_int);




		char fname[1000];
		char fname_mask[1000];
		char label[1000];
		char label_tex[1000];

		sprintf(fname, "surface_%s.tex", SC->prefix);
		sprintf(label, "surface_%s", SC->label_txt);
		sprintf(label_tex, "surface %s", SC->label_tex);
		sprintf(fname_mask, "surface_%s_orbit_%%d", SC->prefix);
		{
			ofstream fp(fname);
			latex_interface L;

			L.head_easy(fp);


			fp << "\\section{The Finite Field $\\mathbb F_{" << q << "}$}" << endl;
			SC->F->cheat_sheet(fp, verbose_level);

			fp << "\\bigskip" << endl;

			SoA->cheat_sheet(fp,
				label,
				label_tex,
				TRUE /* f_print_orbits */,
				fname_mask /* const char *fname_mask*/,
				verbose_level);

			fp << "\\setlength{\\parindent}{0pt}" << endl;

			if (f_surface_clebsch) {

				surface_object *SO;
				SO = SoA->SO;

				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;
				fp << "\\section{Points on the surface}" << endl;
				fp << endl;

				SO->print_affine_points_in_source_code(fp);


				fp << endl;
				fp << "\\bigskip" << endl;
				fp << endl;

				fp << "\\section{Clebsch maps}" << endl;

				SC->Surf->latex_table_of_clebsch_maps(fp);


				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;



				fp << "\\section{Six-arcs not on a conic}" << endl;
				fp << endl;


				//fp << "The six-arcs not on a conic are:\\\\" << endl;
				Six_arcs->report_latex(fp);


				if (f_surface_codes) {

					homogeneous_polynomial_domain *HPD;

					HPD = NEW_OBJECT(homogeneous_polynomial_domain);

					HPD->init(F, 3, 2 /* degree */,
							TRUE /* f_init_incidence_structure */,
							verbose_level);

					action *A_on_poly;

					A_on_poly = NEW_OBJECT(action);
					A_on_poly->induced_action_on_homogeneous_polynomials(A,
						HPD,
						FALSE /* f_induce_action */, NULL,
						verbose_level);

					cout << "created action A_on_poly" << endl;
					A_on_poly->print_info();

					schreier *Sch;
					longinteger_object full_go;

					//Sch = new schreier;
					//A2->all_point_orbits(*Sch, verbose_level);

					cout << "computing orbits:" << endl;

					Sch = A->Strong_gens->orbits_on_points_schreier(A_on_poly, verbose_level);

					//SC->Sg->
					//Sch = SC->Sg->orbits_on_points_schreier(A_on_poly, verbose_level);

					orbit_transversal *T;

					A->group_order(full_go);
					T = NEW_OBJECT(orbit_transversal);

					cout << "before T->init_from_schreier" << endl;

					T->init_from_schreier(
							Sch,
							A,
							full_go,
							verbose_level);

					cout << "after T->init_from_schreier" << endl;

					Sch->print_orbit_reps(cout);

					cout << "orbit reps:" << endl;

					fp << "\\section{Orbits on conics}" << endl;
					fp << endl;

					T->print_table_latex(
							fp,
							TRUE /* f_has_callback */,
							polynomial_orbits_callback_print_function2,
							HPD /* callback_data */,
							TRUE /* f_has_callback */,
							polynomial_orbits_callback_print_function,
							HPD /* callback_data */,
							verbose_level);


				}


#if 0

				int *Arc_iso; // [72]
				int *Clebsch_map; // [nb_pts]
				int *Clebsch_coeff; // [nb_pts * 4]
				//int line_a, line_b;
				//int transversal_line;
				int tritangent_plane_rk;
				int plane_rk_global;
				int ds, ds_row;

				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;

				fp << "\\section{Clebsch maps in detail}" << endl;
				fp << endl;




				Arc_iso = NEW_int(72);
				Clebsch_map = NEW_int(SO->nb_pts);
				Clebsch_coeff = NEW_int(SO->nb_pts * 4);

				for (ds = 0; ds < 36; ds++) {
					for (ds_row = 0; ds_row < 2; ds_row++) {
						SC->Surf->prepare_clebsch_map(
								ds, ds_row,
								line_a, line_b,
								transversal_line,
								0 /*verbose_level */);


						fp << endl;
						fp << "\\bigskip" << endl;
						fp << endl;
						fp << "\\subsection{Clebsch map for double six "
								<< ds << ", row " << ds_row << "}" << endl;
						fp << endl;



						cout << "computing clebsch map:" << endl;
						SO->compute_clebsch_map(line_a, line_b,
							transversal_line,
							tritangent_plane_rk,
							Clebsch_map,
							Clebsch_coeff,
							verbose_level);


						plane_rk_global = SO->Tritangent_planes[
							SO->Eckardt_to_Tritangent_plane[
								tritangent_plane_rk]];

						int Arc[6];
						int Arc2[6];
						int Blown_up_lines[6];
						int perm[6];

						SO->clebsch_map_find_arc_and_lines(
								Clebsch_map,
								Arc,
								Blown_up_lines,
								0 /* verbose_level */);

						for (j = 0; j < 6; j++) {
							perm[j] = j;
							}

						int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
						for (j = 0; j < 6; j++) {
							Arc2[j] = Arc[perm[j]];
							}


						fp << endl;
						fp << "\\bigskip" << endl;
						fp << endl;
						//fp << "\\section{Clebsch map}" << endl;
						//fp << endl;
						fp << "Line 1 = $";
						fp << SC->Surf->Line_label_tex[line_a];
						fp << "$\\\\" << endl;
						fp << "Line 2 = $";
						fp << SC->Surf->Line_label_tex[line_b];
						fp << "$\\\\" << endl;
						fp << "Transversal line $";
						fp << SC->Surf->Line_label_tex[transversal_line];
						fp << "$\\\\" << endl;
						fp << "Image plane $\\pi_{" << tritangent_plane_rk
								<< "}=" << plane_rk_global << "=$\\\\" << endl;
						fp << "$$" << endl;

						fp << "\\left[" << endl;
						SC->Surf->Gr3->print_single_generator_matrix_tex(
								fp, plane_rk_global);
						fp << "\\right]," << endl;

						fp << "$$" << endl;
						fp << "Arc $";
						int_set_print_tex(fp, Arc2, 6);
						fp << "$\\\\" << endl;
						fp << "Half double six: $";
						int_set_print_tex(fp, Blown_up_lines, 6);
						fp << "=\\{";
						for (j = 0; j < 6; j++) {
							fp << SC->Surf->Line_label_tex[Blown_up_lines[j]];
							fp << ", ";
							}
						fp << "\\}$\\\\" << endl;

						fp << "The arc consists of the following "
								"points:\\\\" << endl;
						display_table_of_projective_points(fp,
								SC->F, Arc2, 6, 3);

						int orbit_at_level, idx;
						Six_arcs->Gen->gen->identify(Arc2, 6,
								transporter, orbit_at_level,
								0 /*verbose_level */);


						if (!int_vec_search(Six_arcs->Not_on_conic_idx,
							Six_arcs->nb_arcs_not_on_conic,
							orbit_at_level,
							idx)) {
							cout << "could not find orbit" << endl;
							exit(1);
							}

						fp << "The arc is isomorphic to arc " << orbit_at_level
								<< " in the original classification.\\\\" << endl;
						fp << "The arc is isomorphic to arc " << idx
								<< " in the list.\\\\" << endl;
						Arc_iso[2 * ds + ds_row] = idx;



						SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

						//SO->clebsch_map_print_fibers(Clebsch_map);
						}
					}



				fp << "The isomorphism type of arc associated with "
						"each half-double six is:" << endl;
				fp << "$$" << endl;
				print_integer_matrix_with_standard_labels(fp,
						Arc_iso, 36, 2, TRUE);
				fp << "$$" << endl;

				FREE_int(Arc_iso);
				FREE_int(Clebsch_map);
				FREE_int(Clebsch_coeff);
#endif


#if 0
				fp << endl;
				fp << "\\clearpage" << endl;
				fp << endl;


				fp << "\\section{Clebsch maps in detail by orbits "
						"on half-double sixes}" << endl;
				fp << endl;



				fp << "There are " << SoA->Orbits_on_single_sixes->nb_orbits
						<< "orbits on half double sixes\\\\" << endl;

				Arc_iso = NEW_int(SoA->Orbits_on_single_sixes->nb_orbits);
				Clebsch_map = NEW_int(SO->nb_pts);
				Clebsch_coeff = NEW_int(SO->nb_pts * 4);

				int j, f, l, k;

				for (j = 0; j < SoA->Orbits_on_single_sixes->nb_orbits; j++) {

					int line1, line2, transversal_line;

					if (f_v) {
						cout << "surface_with_action::arc_lifting_and_classify "
							"orbit on single sixes " << j << " / "
							<< SoA->Orbits_on_single_sixes->nb_orbits << ":" << endl;
					}

					fp << "\\subsection*{Orbit on single sixes " << j << " / "
						<< SoA->Orbits_on_single_sixes->nb_orbits << "}" << endl;

					f = SoA->Orbits_on_single_sixes->orbit_first[j];
					l = SoA->Orbits_on_single_sixes->orbit_len[j];
					if (f_v) {
						cout << "orbit f=" << f <<  " l=" << l << endl;
						}
					k = SoA->Orbits_on_single_sixes->orbit[f];

					if (f_v) {
						cout << "The half double six is no " << k << " : ";
						int_vec_print(cout, SoA->Surf->Half_double_sixes + k * 6, 6);
						cout << endl;
						}

					int h;

					fp << "The half double six is no " << k << "$ = "
							<< Surf->Half_double_six_label_tex[k] << "$ : $";
					int_vec_print(fp, Surf->Half_double_sixes + k * 6, 6);
					fp << " = \\{" << endl;
					for (h = 0; h < 6; h++) {
						fp << Surf->Line_label_tex[
								Surf->Half_double_sixes[k * 6 + h]];
						if (h < 6 - 1) {
							fp << ", ";
							}
						}
					fp << "\\}$\\\\" << endl;

					ds = k / 2;
					ds_row = k % 2;

					SC->Surf->prepare_clebsch_map(
							ds, ds_row,
							line1, line2,
							transversal_line,
							0 /*verbose_level */);

					fp << endl;
					fp << "\\bigskip" << endl;
					fp << endl;
					fp << "\\subsection{Clebsch map for double six "
							<< ds << ", row " << ds_row << "}" << endl;
					fp << endl;



					cout << "computing clebsch map:" << endl;
					SO->compute_clebsch_map(line1, line2,
						transversal_line,
						tritangent_plane_rk,
						Clebsch_map,
						Clebsch_coeff,
						verbose_level);


					plane_rk_global = SO->Tritangent_planes[
						SO->Eckardt_to_Tritangent_plane[
							tritangent_plane_rk]];

					int Arc[6];
					int Arc2[6];
					int Blown_up_lines[6];
					int perm[6];

					SO->clebsch_map_find_arc_and_lines(
							Clebsch_map,
							Arc,
							Blown_up_lines,
							0 /* verbose_level */);

					for (h = 0; h < 6; h++) {
						perm[h] = h;
						}

					Sorting.int_vec_heapsort_with_log(Blown_up_lines, perm, 6);
					for (h = 0; h < 6; h++) {
						Arc2[h] = Arc[perm[h]];
						}


					fp << endl;
					fp << "\\bigskip" << endl;
					fp << endl;
					//fp << "\\section{Clebsch map}" << endl;
					//fp << endl;
					fp << "Line 1 = $";
					fp << SC->Surf->Line_label_tex[line1];
					fp << "$\\\\" << endl;
					fp << "Line 2 = $";
					fp << SC->Surf->Line_label_tex[line2];
					fp << "$\\\\" << endl;
					fp << "Transversal line $";
					fp << SC->Surf->Line_label_tex[transversal_line];
					fp << "$\\\\" << endl;
					fp << "Image plane $\\pi_{" << tritangent_plane_rk
							<< "}=" << plane_rk_global << "=$\\\\" << endl;
					fp << "$$" << endl;

					fp << "\\left[" << endl;
					SC->Surf->Gr3->print_single_generator_matrix_tex(
							fp, plane_rk_global);
					fp << "\\right]," << endl;

					fp << "$$" << endl;
					fp << "Arc $";
					int_set_print_tex(fp, Arc2, 6);
					fp << "$\\\\" << endl;
					fp << "Half double six: $";
					int_set_print_tex(fp, Blown_up_lines, 6);
					fp << "=\\{";
					for (h = 0; h < 6; h++) {
						fp << SC->Surf->Line_label_tex[Blown_up_lines[h]];
						fp << ", ";
						}
					fp << "\\}$\\\\" << endl;

					fp << "The arc consists of the following "
							"points:\\\\" << endl;
					SC->F->display_table_of_projective_points(fp,
							Arc2, 6, 3);

					int orbit_at_level, idx;
					Six_arcs->Gen->gen->identify(Arc2, 6,
							transporter, orbit_at_level,
							0 /*verbose_level */);


					if (!Sorting.int_vec_search(Six_arcs->Not_on_conic_idx,
						Six_arcs->nb_arcs_not_on_conic,
						orbit_at_level,
						idx)) {
						cout << "could not find orbit" << endl;
						exit(1);
						}

					fp << "The arc is isomorphic to arc " << orbit_at_level
							<< " in the original classification.\\\\" << endl;
					fp << "The arc is isomorphic to arc " << idx
							<< " in the list.\\\\" << endl;
					Arc_iso[j] = idx;



					SO->clebsch_map_latex(fp, Clebsch_map, Clebsch_coeff);

				} // next j

				fp << "The isomorphism type of arc associated with "
						"each half-double six is:" << endl;
				fp << "$$" << endl;
				int_vec_print(fp,
						Arc_iso, SoA->Orbits_on_single_sixes->nb_orbits);
				fp << "$$" << endl;



				FREE_int(Arc_iso);
				FREE_int(Clebsch_map);
				FREE_int(Clebsch_coeff);

#endif



				if (f_surface_quartic) {
					SoA->quartic(fp, verbose_level);
					}


			}


			L.foot(fp);
		}
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;



		FREE_OBJECT(SoA);
		FREE_OBJECT(Six_arcs);
		FREE_int(transporter);


		}



	FREE_int(Elt2);

	FREE_OBJECT(SC);


	if (f_v) {
		cout << "interface_projective::do_create_surface done" << endl;
	}
}


static int do_create_points_on_quartic_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level)
{
	double num, denom, b;
	double epsilon = 0.00001;

	num = 4. - 4. * t;
	denom = 4. - 25. * t * 0.25;
	if (ABS(denom) < epsilon) {
		return FALSE;
	}
	else {
		b = num / denom;
		if (b < 0) {
			return FALSE;
		}
		else {
			pt[0] = sqrt(t);
			pt[1] = sqrt(b);
		}
	}
	cout << "created point " << pt[0] << ", " << pt[1] << endl;
	return TRUE;
}

static int do_create_points_on_parabola_compute_point_function(double t,
		double *pt, void *extra_data, int verbose_level)
{
	interface_projective *I = (interface_projective *) extra_data;
	double a = I->parabola_a;
	double b = I->parabola_b;
	double c = I->parabola_c;

	pt[0] = t;
	pt[1] = a * t * t + b * t + c;
	//cout << "created point " << pt[0] << ", " << pt[1] << endl;
	return TRUE;
}


static int do_create_points_smooth_curve_compute_point_function(double t,
		double *output, void *extra_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	interface_projective *I = (interface_projective *) extra_data;
	int ret = FALSE;
	double epsilon = 0.0001;
	double *input; // to store the input variable and all local variables during evaluate


	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function t = " << t << endl;
	}
	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function before evaluate" << endl;
	}
	input = new double[I->smooth_curve_Polish->Variables.size()];
	input[0] = t;
	I->smooth_curve_Polish->evaluate(
			input /* variable_values */,
			output,
			verbose_level);
	delete [] input;

	if (I->smooth_curve_Polish->Entry.size() == 4) {
		if (ABS(output[3]) < epsilon) {
			ret = FALSE;
		}
		else {
			double av = 1. / output[3];
			output[0] *= av;
			output[1] *= av;
			output[2] *= av;
			output[3] *= av;
			ret = TRUE;
		}
	}
	else {
		ret = TRUE;
	}
	if (f_v) {
		cout << "do_create_points_smooth_curve_compute_point_function after evaluate t = " << t << endl;
	}
	return ret;
}


void polynomial_orbits_callback_print_function(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;

	coeff = NEW_int(HPD->nb_monomials);
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
}

void polynomial_orbits_callback_print_function2(
		stringstream &ost, void *data, void *callback_data)
{
	homogeneous_polynomial_domain *HPD =
			(homogeneous_polynomial_domain *) callback_data;

	int *coeff;
	int *i_data = (int *) data;
	long int *Pts;
	int nb_pts;

	Pts = NEW_lint(HPD->P->N_points);
	coeff = NEW_int(HPD->nb_monomials);
	HPD->unrank_coeff_vector(coeff, i_data[0]);
	HPD->enumerate_points(coeff, Pts, nb_pts,  0 /*verbose_level*/);
	ost << nb_pts;
	//int_vec_print(cout, coeff, HPD->nb_monomials);
	//cout << " = ";
	//HPD->print_equation_str(ost, coeff);
	//ost << endl;
	FREE_int(coeff);
	FREE_lint(Pts);
}






}}

