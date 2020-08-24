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


interface_projective::interface_projective()
{

	f_cheat_sheet_PG = FALSE;
	n = 0;
	q = 0;

	f_canonical_form_PG = FALSE;

	f_input = FALSE;
	Data_input_stream = NULL;

	f_all_k_subsets = FALSE;
	k = 0;






	f_save_incma_in_and_out = FALSE;
	//save_incma_in_and_out_prefix

	f_classification_prefix = FALSE;
	//classification_prefix

	f_save = FALSE;
	//std::string save_prefix;


	fixed_structure_order_list_sz = 0;
	//fixed_structure_order_list[1000];

	f_report = FALSE;
	// report_prefix

	f_max_TDO_depth = FALSE;
	max_TDO_depth = INT_MAX;

	f_classify_cubic_curves = FALSE;
	f_has_control_six_arcs = FALSE;
	Control_six_arcs = NULL;;

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

	f_create_spread = FALSE;
	Spread_create_description = FALSE;

	f_study_surface = FALSE;
	study_surface_q = 0;
	study_surface_nb = 0;

	f_move_two_lines_in_hyperplane_stabilizer = FALSE;
	line1_from = 0;
	line2_from = 0;
	line1_to = 0;
	line2_to = 0;

	f_make_table_of_surfaces = FALSE;
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
	else if (strcmp(argv[i], "-create_spread") == 0) {
		cout << "-create_spread <description>" << endl;
	}
	else if (strcmp(argv[i], "-study_surface") == 0) {
		cout << "-study_surface <int : q> <int : nb>" << endl;
	}
	else if (strcmp(argv[i], "-prefix") == 0) {
		cout << "-prefix <string : prefix>" << endl;
	}
	else if (strcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer") == 0) {
		cout << "-move_two_lines_in_hyperplane_stabilizer <int : q>  <int : line1_from> <int : line2_from> <int : line1_to> <int : line2_to> " << endl;
	}
	else if (strcmp(argv[i], "-make_table_of_surfaces") == 0) {
		cout << "-make_table_of_surfaces " << endl;
	}
}



int interface_projective::recognize_keyword(int argc,
		const char **argv, int i, int verbose_level)
{
	if (i >= argc) {
		return false;
	}
	if (strcmp(argv[i], "-cheat_sheet_PG") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-canonical_form_PG") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-classify_cubic_curves") == 0) {
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
	else if (strcmp(argv[i], "-create_spread") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-study_surface") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-make_table_of_surfaces") == 0) {
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

	//interface_projective::argc = argc;
	//interface_projective::argv = argv;

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
			cout << "-canonical_form_PG " << n << " " <<  q << ", reading extra arguments" << endl;
			i += read_canonical_form_arguments(argc - (i + 1), argv + i + 1, 0, verbose_level);
			cout << "done reading -canonical_form_PG " << n << " " <<  q << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (strcmp(argv[i], "-classify_cubic_curves") == 0) {
			f_classify_cubic_curves = TRUE;
			q = atoi(argv[++i]);
			cout << "-classify_cubic_curves " <<  q << endl;
			i++;
		}
		else if (strcmp(argv[i], "-control_six_arcs") == 0) {
			f_has_control_six_arcs = TRUE;
			Control_six_arcs = NEW_OBJECT(poset_classification_control);
			i += Control_six_arcs->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "done reading -control_six_arcs " << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
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
		else if (strcmp(argv[i], "-create_spread") == 0) {
			f_create_spread = TRUE;
			Spread_create_description = NEW_OBJECT(spread_create_description);
			i += Spread_create_description->read_arguments(
					argc - (i - 1),
					argv + i, verbose_level) - 1;

			cout << "-create_spread" << endl;
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
		else if (strcmp(argv[i], "-study_surface") == 0) {
			f_study_surface = TRUE;
			study_surface_q = atoi(argv[++i]);
			study_surface_nb = atoi(argv[++i]);
			cout << "-study_surface" << study_surface_q << " " << study_surface_nb << endl;
		}
		else if (strcmp(argv[i], "-move_two_lines_in_hyperplane_stabilizer") == 0) {
			f_move_two_lines_in_hyperplane_stabilizer = TRUE;
			q = atoi(argv[++i]);
			line1_from = atoi(argv[++i]);
			line2_from = atoi(argv[++i]);
			line1_to = atoi(argv[++i]);
			line2_to = atoi(argv[++i]);
			cout << "-move_two_lines_in_hyperplane_stabilizer" << q
					<< " " << line1_from << " " << line1_from
					<< " " << line1_to << " " << line2_to
					<< endl;
		}
		else if (strcmp(argv[i], "-make_table_of_surfaces") == 0) {
			f_make_table_of_surfaces = TRUE;
			cout << "-make_table_of_surfaces" << endl;
		}
		else {
			cout << "interface_projective::read_arguments: unrecognized option "
					<< argv[i] << ", skipping" << endl;
			//exit(1);
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
			cout << "-input" << endl;
			Data_input_stream = NEW_OBJECT(data_input_stream);
			i += Data_input_stream->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "-input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}





		else if (strcmp(argv[i], "-save_incma_in_and_out") == 0) {
			f_save_incma_in_and_out = TRUE;
			save_incma_in_and_out_prefix.assign(argv[++i]);
			cout << "-save_incma_in_and_out" << save_incma_in_and_out_prefix << endl;
		}

		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			save_prefix.assign(argv[++i]);
			cout << "-save " << save_prefix << endl;
		}
		else if (strcmp(argv[i], "-classification_prefix") == 0) {
			f_classification_prefix = TRUE;
			classification_prefix.assign(argv[++i]);
			cout << "-classification_prefix " << classification_prefix << endl;
		}



		else if (strcmp(argv[i], "-all_k_subsets") == 0) {
			f_all_k_subsets = TRUE;
			k = atoi(argv[++i]);
			cout << "-all_k_subsets " << k << endl;
		}
		else if (strcmp(argv[i], "-fixed_structure_of_element_of_order") == 0) {
			fixed_structure_order_list[fixed_structure_order_list_sz] = atoi(argv[++i]);
			cout << "-fixed_structure_of_element_of_order "
					<< fixed_structure_order_list[fixed_structure_order_list_sz] << endl;
			fixed_structure_order_list_sz++;
		}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			report_prefix.assign(argv[++i]);
			cout << "-report " << report_prefix << endl;
		}

		else if (strcmp(argv[i], "-max_TDO_depth") == 0) {
			f_max_TDO_depth = TRUE;
			max_TDO_depth = atoi(argv[++i]);
			cout << "-max_TDO_depth " << max_TDO_depth << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end " << endl;
			break;
		}
		else {
			cout << "-canonical_form_PG: unrecognized option " << argv[i] << ", skipping" << endl;
			//exit(1);
		}
	}
	if (f_v) {
		cout << "interface_projective::read_canonical_form_arguments done" << endl;
	}
	return i + 1;
}

void interface_projective::worker(orbiter_session *Session, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::worker" << endl;
	}

	if (f_cheat_sheet_PG) {
		do_cheat_sheet_PG(Session, n, q, verbose_level);
	}
	else if (f_canonical_form_PG) {
		do_canonical_form_PG(Session, n, q, verbose_level);
	}
	else if (f_classify_cubic_curves) {
		if (!f_has_control_six_arcs) {
			cout << "please use -control_six_arcs <description> -end" << endl;
			exit(1);
		}
		do_classify_cubic_curves(q, Control_six_arcs, verbose_level);
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
	else if (f_create_spread) {
		do_create_spread(Spread_create_description, verbose_level);
	}
	else if (f_study_surface) {
		do_study_surface(study_surface_q, study_surface_nb, verbose_level);
	}
	else if (f_move_two_lines_in_hyperplane_stabilizer) {
		do_move_two_lines_in_hyperplane_stabilizer(
				q,
				line1_from, line2_from,
				line1_to, line2_to, verbose_level);
	}
	else if (f_make_table_of_surfaces) {

		geometry_global GG;

		GG.make_table_of_surfaces(verbose_level);
	}

	if (f_v) {
		cout << "interface_projective::worker done" << endl;
	}
}


void interface_projective::do_cheat_sheet_PG(orbiter_session *Session,
		int n, int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG" << endl;
	}
	finite_field *F;

	F = NEW_OBJECT(finite_field);


	if (Session->f_override_polynomial) {
		F->init_override_polynomial(q, Session->override_polynomial, 0);
	}
	else {
		F->init(q, 0);
	}

	F->cheat_sheet_PG(n, verbose_level);

	FREE_OBJECT(F);

	if (f_v) {
		cout << "interface_projective::do_cheat_sheet_PG done" << endl;
	}

}

void interface_projective::do_canonical_form_PG(orbiter_session *Session,
		int n, int q, int verbose_level)
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

	if (Session->f_override_polynomial) {
		F->init_override_polynomial(q, Session->override_polynomial, 0);
	}
	else {
		F->init(q, 0);
	}
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





	cout << "interface_projective::do_canonical_form_PG "
			"before PA->classify_objects_using_nauty" << endl;
	PA->classify_objects_using_nauty(Data_input_stream,
		CB,
		f_save_incma_in_and_out, classification_prefix,
		verbose_level - 1);
	cout << "interface_projective::do_canonical_form_PG "
			"after PA->classify_objects_using_nauty" << endl;



	cout << "canonical_form.cpp We found " << CB->nb_types << " types" << endl;


	compute_and_print_ago_distribution_with_classes(cout,
			CB, verbose_level);


	cout << "interface_projective::do_canonical_form_PG "
			"In the ordering of canonical forms, they are" << endl;
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
		cout << "Saving the classification with save_prefix " << save_prefix << endl;
		PA->save(save_prefix, CB, verbose_level);
		CB->save(save_prefix,
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

		cout << "interface_projective::do_canonical_form_PG Producing a latex report:" << endl;

		char fname[1000];

		if (f_classification_prefix == FALSE) {
			cout << "please use option -classification_prefix <prefix> to set the "
					"prefix for the output file" << endl;
			exit(1);
			}
		snprintf(fname, 1000, "%s_classification.tex", classification_prefix.c_str());


		PA->latex_report(fname,
				report_prefix,
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

void interface_projective::do_classify_cubic_curves(int q,
		poset_classification_control *Control_six_arcs, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_classify_cubic_curves" << endl;
	}

	//const char *starter_directory_name = "";
	//char base_fname[1000];

	//snprintf(base_fname, 1000, "cubic_curves_%d", q);


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

	group_theoretic_activity *GTA;

	GTA = NEW_OBJECT(group_theoretic_activity);

	classify_cubic_curves *CCC;

	CCC = NEW_OBJECT(classify_cubic_curves);


	CCC->init(
			GTA,
			CCA,
			//starter_directory_name,
			//base_fname,
			Control_six_arcs,
			verbose_level);

	CCC->compute_starter(verbose_level);

	CCC->test_orbits(verbose_level);

	CCC->do_classify(verbose_level);


	char fname[1000];
	char title[1000];
	char author[1000];
	snprintf(title, 1000, "Cubic Curves in PG$(2,%d)$", q);
	strcpy(author, "");
	snprintf(fname, 1000, "Cubic_curves_q%d.tex", q);

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


		for (i = 0; i < (int) C1.Pts.size(); i++) {
			cout << C1.Pts[i].t << " : " << C1.Pts[i].coords[0] << ", " << C1.Pts[i].coords[1] << endl;
		}

		double *Pts;
		int nb_pts;

		nb_pts = 4 * (C1.Pts.size() + C2.Pts.size());
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C1.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C1.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C1.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = -1 * C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C2.Pts[i].coords[0];
			Pts[nb * 2 + 1] = -1 * C2.Pts[i].coords[1];
			nb++;
		}
		for (i = 0; i < (int) C2.Pts.size(); i++) {
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




		for (i = 0; i < (int) C.Pts.size(); i++) {
			cout << C.Pts[i].t << " : " << C.Pts[i].coords[0] << ", " << C.Pts[i].coords[1] << endl;
		}

		{
		double *Pts;
		int nb_pts;

		nb_pts = C.Pts.size();
		Pts = new double[nb_pts * 2];
		nb = 0;
		for (i = 0; i < (int) C.Pts.size(); i++) {
			Pts[nb * 2 + 0] = C.Pts[i].coords[0];
			Pts[nb * 2 + 1] = C.Pts[i].coords[1];
			nb++;
		}
		file_io Fio;
		char fname[1000];
		snprintf(fname, 1000, "parabola_N%d_%lf_%lf_%lf_points.csv", N, parabola_a, parabola_b, parabola_c);

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
		for (i = 0; i < (int) C.Pts.size(); i++) {
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
		snprintf(fname, 1000, "parabola_N%d_%lf_%lf_%lf_projection_from_center.csv", N, parabola_a, parabola_b, parabola_c);

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
		for (i = 0; i < (int) C.Pts.size(); i++) {
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
		snprintf(fname, 1000, "parabola_N%d_%lf_%lf_%lf_projection_from_sphere.csv",
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
		for (i = 0; i < (int) C.Pts.size(); i++) {
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
		snprintf(fname, 1000, "parabola_N%d_%lf_%lf_%lf_points_projected.csv",
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
		for (i = 0; i < (int) C.Pts.size(); i++) {
			if (C.Pts[i].f_is_valid) {
				for (j = 0; j < nb_dimensions; j++) {
					Pts[nb * nb_dimensions + j] = C.Pts[i].coords[j];
				}
				nb++;
			}
		}
		file_io Fio;
		char fname[1000];
		snprintf(fname, 1000, "function_%s_N%d_points.csv", curve_label, N);

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
		for (i = 0; i < (int) C.Pts.size(); i++) {
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
		snprintf(fname, 1000, "function_%s_N%d_points_plus.csv", curve_label, N);

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

void interface_projective::do_create_spread(spread_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_create_spread" << endl;
	}


	spread_create *SC;

	SC = NEW_OBJECT(spread_create);

	cout << "before SC->init" << endl;
	SC->init(Descr, verbose_level);
	cout << "after SC->init" << endl;


	if (nb_transform) {
		cout << "before SC->apply_transformations" << endl;
		SC->apply_transformations(transform_coeffs,
				f_inverse_transform, nb_transform, verbose_level);
		cout << "after SC->apply_transformations" << endl;
		}

	action *A;
	//int *Elt1;
	int *Elt2;

	A = SC->A;

	Elt2 = NEW_int(A->elt_size_in_int);

	latex_interface L;

	cout << "We have created the following spread set:" << endl;
	cout << "$$" << endl;
	L.lint_set_print_tex(cout, SC->set, SC->sz);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);
		}





	FREE_int(Elt2);

	FREE_OBJECT(SC);

	if (f_v) {
		cout << "interface_projective::do_create_spread done" << endl;
	}

}


void interface_projective::do_study_surface(int q, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_study_surface" << endl;
	}

	surface_study *study;

	study = NEW_OBJECT(surface_study);

	cout << "before study->init" << endl;
	study->init(q, nb, verbose_level);
	cout << "after study->init" << endl;

	cout << "before study->study_intersection_points" << endl;
	study->study_intersection_points(verbose_level);
	cout << "after study->study_intersection_points" << endl;

	cout << "before study->study_line_orbits" << endl;
	study->study_line_orbits(verbose_level);
	cout << "after study->study_line_orbits" << endl;

	cout << "before study->study_group" << endl;
	study->study_group(verbose_level);
	cout << "after study->study_group" << endl;

	cout << "before study->study_orbits_on_lines" << endl;
	study->study_orbits_on_lines(verbose_level);
	cout << "after study->study_orbits_on_lines" << endl;

	cout << "before study->study_find_eckardt_points" << endl;
	study->study_find_eckardt_points(verbose_level);
	cout << "after study->study_find_eckardt_points" << endl;

#if 0
	if (study->nb_Eckardt_pts == 6) {
		cout << "before study->study_surface_with_6_eckardt_points" << endl;
		study->study_surface_with_6_eckardt_points(verbose_level);
		cout << "after study->study_surface_with_6_eckardt_points" << endl;
		}
#endif

	if (f_v) {
		cout << "interface_projective::do_study_surface done" << endl;
	}
}


void interface_projective::do_move_two_lines_in_hyperplane_stabilizer(
		int q,
		long int line1_from, long int line2_from,
		long int line1_to, long int line2_to, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::do_move_two_lines_in_hyperplane_stabilizer" << endl;
	}

	finite_field *F;
	projective_space *P;
	int A4[16];

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	P = NEW_OBJECT(projective_space);
	P->init(3, F,
			FALSE /* f_init_incidence_structure */,
			0 /*verbose_level*/);
	P->hyperplane_lifting_with_two_lines_moved(
			line1_from, line1_to,
			line2_from, line2_to,
			A4,
			verbose_level);

	cout << "interface_projective::do_move_two_lines_in_hyperplane_stabilizer A4=" << endl;
	int_matrix_print(A4, 4, 4);

	if (f_v) {
		cout << "interface_projective::do_move_two_lines_in_hyperplane_stabilizer done" << endl;
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






}}

