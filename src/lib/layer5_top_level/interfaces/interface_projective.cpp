/*
 * interface_projective.cpp
 *
 *  Created on: Apr 14, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace user_interface {




interface_projective::interface_projective()
{

	f_create_points_on_quartic = FALSE;
	desired_distance = 0;

	f_create_points_on_parabola = FALSE;
	parabola_N = 0;
	parabola_a = 0;
	parabola_b = 0.;
	parabola_c = 0.;

	f_smooth_curve = FALSE;
	//smooth_curve_label;
	smooth_curve_N = 0;
	smooth_curve_t_min = 0;
	smooth_curve_t_max = 0;
	smooth_curve_boundary = 0;
	//smooth_curve_Polish = NULL;
	FP_descr = NULL;

	f_make_table_of_surfaces = FALSE;

	f_create_surface_reports = FALSE;
	//std::string create_surface_reports_field_orders_text;

	f_create_surface_atlas = FALSE;
	create_surface_atlas_q_max = 0;

	f_create_dickson_atlas = FALSE;



}


void interface_projective::print_help(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (ST.stringcmp(argv[i], "-create_points_on_quartic") == 0) {
		cout << "-create_points_on_quartic <double : desired_distance>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_points_on_parabola") == 0) {
		cout << "-create_points_on_parabola <double : desired_distance> <double : a> <double : b> <double : c>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-smooth_curve") == 0) {
		cout << "-smooth_curve <string : label> <double : desired_distance> <int : N> <double : boundary> <double : t_min> <double : t_max> <function>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
		cout << "-make_table_of_surfaces " << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_surface_reports") == 0) {
		cout << "-create_surface_reports <string : field orders>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_surface_atlas") == 0) {
		cout << "-create_surface_atlas <int : q_max>" << endl;
	}
	else if (ST.stringcmp(argv[i], "-create_dickson_atlas") == 0) {
		cout << "-create_dickson_atlas" << endl;
	}
}



int interface_projective::recognize_keyword(int argc,
		std::string *argv, int i, int verbose_level)
{
	data_structures::string_tools ST;

	if (i >= argc) {
		return false;
	}
	if (ST.stringcmp(argv[i], "-create_points_on_quartic") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_points_on_parabola") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-smooth_curve") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_surface_reports") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_surface_atlas") == 0) {
		return true;
	}
	else if (ST.stringcmp(argv[i], "-create_dickson_atlas") == 0) {
		return true;
	}
	return false;
}

void interface_projective::read_arguments(int argc,
		std::string *argv, int &i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "interface_projective::read_arguments" << endl;
	}
	if (f_v) {
		cout << "interface_projective::read_arguments the next argument is " << argv[i] << endl;
	}
	if (ST.stringcmp(argv[i], "-create_points_on_quartic") == 0) {
		f_create_points_on_quartic = TRUE;
		desired_distance = ST.strtof(argv[++i]);
		if (f_v) {
			cout << "-create_points_on_quartic " << desired_distance << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_points_on_parabola") == 0) {
		f_create_points_on_parabola = TRUE;
		desired_distance = ST.strtof(argv[++i]);
		parabola_N = ST.strtoi(argv[++i]);
		parabola_a = ST.strtof(argv[++i]);
		parabola_b = ST.strtof(argv[++i]);
		parabola_c = ST.strtof(argv[++i]);
		if (f_v) {
			cout << "-create_points_on_parabola " << desired_distance << " "
				<< parabola_N << " " << parabola_a << " "
				<< parabola_b << " " << parabola_c << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-smooth_curve") == 0) {
		f_smooth_curve = TRUE;
		smooth_curve_label.assign(argv[++i]);
		desired_distance = ST.strtof(argv[++i]);
		smooth_curve_N = ST.strtoi(argv[++i]);
		smooth_curve_boundary = ST.strtof(argv[++i]);
		smooth_curve_t_min = ST.strtof(argv[++i]);
		smooth_curve_t_max = ST.strtof(argv[++i]);

		FP_descr = NEW_OBJECT(function_polish_description);

		i += FP_descr->read_arguments(argc - (i + 1),
			argv + i + 1, verbose_level);

		if (f_v) {
			cout << "-smooth_curve "
				<< smooth_curve_label << " "
				<< desired_distance << " "
				<< smooth_curve_N << " "
				<< smooth_curve_boundary << " "
				<< smooth_curve_t_min << " "
				<< smooth_curve_t_max << " "
				<< endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-make_table_of_surfaces") == 0) {
		f_make_table_of_surfaces = TRUE;
		if (f_v) {
			cout << "-make_table_of_surfaces" << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_surface_atlas") == 0) {
		f_create_surface_atlas = TRUE;
		create_surface_atlas_q_max = ST.strtoi(argv[++i]);
		if (f_v) {
			cout << "-create_surface_atlas " << create_surface_atlas_q_max << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_surface_reports") == 0) {
		f_create_surface_reports = TRUE;
		create_surface_reports_field_orders_text.assign(argv[++i]);
		if (f_v) {
			cout << "-create_surface_reports " << create_surface_reports_field_orders_text << endl;
		}
	}
	else if (ST.stringcmp(argv[i], "-create_dickson_atlas") == 0) {
		f_create_dickson_atlas = TRUE;
		if (f_v) {
			cout << "-create_dickson_atlas " << endl;
		}
	}
	if (f_v) {
		cout << "interface_projective::read_arguments done" << endl;
	}
}

void interface_projective::print()
{
	if (f_create_points_on_quartic) {
		cout << "-create_points_on_quartic " << desired_distance << endl;
	}
	if (f_create_points_on_parabola) {
		cout << "-create_points_on_parabola " << desired_distance << " "
				<< parabola_N << " " << parabola_a << " "
				<< parabola_b << " " << parabola_c << endl;
	}
	if (f_smooth_curve) {
		cout << "-smooth_curve "
				<< smooth_curve_label << " "
				<< desired_distance << " "
				<< smooth_curve_N << " "
				<< smooth_curve_boundary << " "
				<< smooth_curve_t_min << " "
				<< smooth_curve_t_max << " "
				<< endl;
		FP_descr->print();
	}
	if (f_make_table_of_surfaces) {
		cout << "-make_table_of_surfaces" << endl;
	}
	if (f_create_surface_atlas) {
		cout << "-create_surface_atlas " << create_surface_atlas_q_max << endl;
	}
	if (f_create_surface_reports) {
		cout << "-create_surface_reports " << create_surface_reports_field_orders_text << endl;
	}
	if (f_create_dickson_atlas) {
		cout << "-create_dickson_atlas " << endl;
	}
}



void interface_projective::worker(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_projective::worker" << endl;
	}

	if (f_create_points_on_quartic) {

		graphics::graphical_output GO;

		GO.do_create_points_on_quartic(desired_distance, verbose_level);
	}
	else if (f_create_points_on_parabola) {

		graphics::graphical_output GO;

		GO.do_create_points_on_parabola(desired_distance,
				parabola_N, parabola_a, parabola_b, parabola_c, verbose_level);
	}
	else if (f_smooth_curve) {

		graphics::graphical_output GO;

		GO.do_smooth_curve(smooth_curve_label,
				desired_distance, smooth_curve_N,
				smooth_curve_t_min, smooth_curve_t_max, smooth_curve_boundary,
				FP_descr, verbose_level);
	}

	else if (f_make_table_of_surfaces) {

		algebraic_geometry::surface_domain Surf;

		Surf.make_table_of_surfaces(verbose_level);
	}

	else if (f_create_surface_reports) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_create_surface_reports(create_surface_reports_field_orders_text, verbose_level);

	}

	else if (f_create_surface_atlas) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_create_surface_atlas(create_surface_atlas_q_max, verbose_level);

	}

	else if (f_create_dickson_atlas) {

		applications_in_algebraic_geometry::cubic_surfaces_in_general::surface_domain_high_level SH;

		SH.do_create_dickson_atlas(verbose_level);

	}




	if (f_v) {
		cout << "interface_projective::worker done" << endl;
	}
}






}}}


