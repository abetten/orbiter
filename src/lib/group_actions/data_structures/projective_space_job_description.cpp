/*
 * projective_space_job_description.cpp
 *
 *  Created on: Apr 28, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


projective_space_job_description::projective_space_job_description()
{

	f_input = FALSE;
	Data = NULL;

	f_fname_base_out = FALSE;
	//fname_base_out;

	f_q = FALSE;
	q = 0;

	f_n = FALSE;
	n = 0;

	f_poly = FALSE;
	//poly = NULL;

	f_embed = FALSE;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	f_andre = FALSE;
		// follow up option for f_andre:
		f_Q = FALSE;
		Q = 0;
		f_poly_Q = FALSE;
		//poly_Q = NULL;


	f_print = FALSE;
		// follow up option for f_print:
		f_lines_in_PG = FALSE;
		f_points_in_PG = FALSE;
		f_points_on_grassmannian = FALSE;
		points_on_grassmannian_k = 0;
		f_orthogonal = FALSE;
		orthogonal_epsilon = 0;
		f_homogeneous_polynomials_LEX = FALSE;
		f_homogeneous_polynomials_PART = FALSE;
		homogeneous_polynomials_degree = 0;


	f_list_group_elements = FALSE;
	f_line_type = FALSE;
	f_plane_type = FALSE;
	f_plane_type_failsafe = FALSE;
	f_conic_type = FALSE;
		// follow up option for f_conic_type:
		f_randomized = FALSE;
		nb_times = 0;

	f_hyperplane_type = FALSE;
	// follow up option for f_hyperplane_type:
		f_show = FALSE;


	f_cone_over = FALSE;

	f_bsf3 = FALSE;
	f_test_diagonals = FALSE;
	//test_diagonals_fname = NULL;
	f_klein = FALSE;

	f_draw_points_in_plane = FALSE;
		//draw_points_in_plane_fname_base = NULL;
		// follow up option for f_draw_points_in_plane:

		f_point_labels = FALSE;
		f_embedded = FALSE;
		f_sideways = FALSE;

	f_canonical_form = FALSE;
	//canonical_form_fname_base;

	f_ideal_LEX = FALSE;
	f_ideal_PART = FALSE;
	ideal_degree = 0;

	f_intersect_with_set_from_file = FALSE;
	//intersect_with_set_from_file_fname = NULL;

	f_arc_with_given_set_as_s_lines_after_dualizing = FALSE;
	arc_size = 0;
	arc_d = 0;
	arc_d_low = 0;
	arc_s = 0;

	f_arc_with_two_given_sets_of_lines_after_dualizing = FALSE;
	//int arc_size;
	//int arc_d;
	arc_t = 0;
	//t_lines_string;


	f_arc_with_three_given_sets_of_lines_after_dualizing = FALSE;
	arc_u = 0;
	//u_lines_string;

	f_dualize_hyperplanes_to_points = FALSE;
	f_dualize_points_to_hyperplanes = FALSE;
}

projective_space_job_description::~projective_space_job_description()
{

}

#if 0
void projective_space_job_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "projective_space_job_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "projective_space_job_description::read_arguments_from_string "
				"done" << endl;
	}
}
#endif

int projective_space_job_description::read_arguments(
	int argc, std::string *argv,
	int verbose_level)
{
	int i;

	cout << "projective_space_job_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (stringcmp(argv[i], "-v") == 0) {
			verbose_level = strtoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
		}
		else if (stringcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = strtoi(argv[++i]);
			cout << "-q " << q << endl;
		}
		else if (stringcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = strtoi(argv[++i]);
			cout << "-Q " << Q << endl;
		}
		else if (stringcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = strtoi(argv[++i]);
			cout << "-n " << n << endl;
		}
		else if (stringcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly.assign(argv[++i]);
			cout << "-poly " << poly << endl;
		}
		else if (stringcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q.assign(argv[++i]);
			cout << "-poly_Q " << poly_Q << endl;
		}
		else if (stringcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			cout << "-input" << endl;
			i += Data->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);
			cout << "projective_space_job_description::read_arguments finished reading -input" << endl;
			cout << "i = " << i << endl;
			cout << "argc = " << argc << endl;
			if (i < argc) {
				cout << "next argument is " << argv[i] << endl;
			}
		}
		else if (stringcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out.assign(argv[++i]);
			cout << "-fname_base_out " << fname_base_out << endl;
		}
		else if (stringcmp(argv[i], "-embed") == 0) {
			f_embed = TRUE;
			cout << "projective_space_job_description::read_arguments -embed" << endl;
		}
		else if (stringcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = strtoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
		}
		else if (stringcmp(argv[i], "-homogeneous_polynomials_LEX") == 0) {
			f_homogeneous_polynomials_LEX = TRUE;
			homogeneous_polynomials_degree = strtoi(argv[++i]);
			cout << "-homogeneous_polynomials_LEX " << homogeneous_polynomials_degree << endl;
		}
		else if (stringcmp(argv[i], "-homogeneous_polynomials_PART") == 0) {
			f_homogeneous_polynomials_PART = TRUE;
			homogeneous_polynomials_degree = strtoi(argv[++i]);
			cout << "-homogeneous_polynomials_PART " << homogeneous_polynomials_degree << endl;
		}
		else if (stringcmp(argv[i], "-andre") == 0) {
			f_andre = TRUE;
			cout << "-andre " << endl;
		}
		else if (stringcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print " << endl;
		}
		else if (stringcmp(argv[i], "-lines_in_PG") == 0) {
			f_lines_in_PG = TRUE;
			cout << "-lines_in_PG " << endl;
		}
		else if (stringcmp(argv[i], "-points_in_PG") == 0) {
			f_points_in_PG = TRUE;
			cout << "-points_in_PG " << endl;
		}
		else if (stringcmp(argv[i], "-points_on_grassmannian") == 0) {
			f_points_on_grassmannian = TRUE;
			points_on_grassmannian_k = strtoi(argv[++i]);
			cout << "-points_on_grassmannian " << points_on_grassmannian_k << endl;
		}
		else if (stringcmp(argv[i], "-list_group_elements") == 0) {
			f_list_group_elements = TRUE;
			cout << "-list_group_elements" << endl;
		}
		else if (stringcmp(argv[i], "-line_type") == 0) {
			f_line_type = TRUE;
			cout << "-line_type" << endl;
		}
		else if (stringcmp(argv[i], "-plane_type") == 0) {
			f_plane_type = TRUE;
			cout << "-plane_type" << endl;
		}
		else if (stringcmp(argv[i], "-plane_type_failsafe") == 0) {
			f_plane_type_failsafe = TRUE;
			cout << "-plane_type_failsafe" << endl;
		}
		else if (stringcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			cout << "-conic_type " << endl;
		}
		else if (stringcmp(argv[i], "-randomized") == 0) {
			f_randomized = TRUE;
			nb_times = strtoi(argv[++i]);
			cout << "-randomized " << nb_times << endl;
		}
		else if (stringcmp(argv[i], "-hyperplane_type") == 0) {
			f_hyperplane_type = TRUE;
			cout << "-hyperplane_type" << endl;
		}
		else if (stringcmp(argv[i], "-show") == 0) {
			f_show = TRUE;
			cout << "-show" << endl;
		}
		else if (stringcmp(argv[i], "-cone_over") == 0) {
			f_cone_over = TRUE;
			cout << "-cone_over" << endl;
		}
		else if (stringcmp(argv[i], "-bsf3") == 0) {
			f_bsf3 = TRUE;
			cout << "-bsf3" << endl;
		}
		else if (stringcmp(argv[i], "-test_diagonals") == 0) {
			f_test_diagonals = TRUE;
			test_diagonals_fname.assign(argv[++i]);
			cout << "-test_diagonals " << test_diagonals_fname << endl;
		}
		else if (stringcmp(argv[i], "-klein") == 0) {
			f_klein = TRUE;
			cout << "-klein" << endl;
		}
		else if (stringcmp(argv[i], "-draw_points_in_plane") == 0) {
			f_draw_points_in_plane = TRUE;
			draw_points_in_plane_fname_base.assign(argv[++i]);
			cout << "-draw_points_in_plane" << draw_points_in_plane_fname_base << endl;
		}
		else if (stringcmp(argv[i], "-point_labels") == 0) {
			f_point_labels = TRUE;
			cout << "-point_labels" << endl;
		}
		else if (stringcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			canonical_form_fname_base.assign(argv[++i]);
			cout << "-canonical_form" << canonical_form_fname_base << endl;
		}
		else if (stringcmp(argv[i], "-ideal_LEX") == 0) {
			f_ideal_LEX = TRUE;
			ideal_degree = strtoi(argv[++i]);
			cout << "-ideal_LEX " << ideal_degree << endl;
		}
		else if (stringcmp(argv[i], "-ideal_PART") == 0) {
			f_ideal_PART = TRUE;
			ideal_degree = strtoi(argv[++i]);
			cout << "-ideal_PART " << ideal_degree << endl;
		}
		else if (stringcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded" << endl;
		}
		else if (stringcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways" << endl;
		}
		else if (stringcmp(argv[i], "-intersect_with_set_from_file") == 0) {
			f_intersect_with_set_from_file = TRUE;
			intersect_with_set_from_file_fname.assign(argv[++i]);
			cout << "-intersect_with_set_from_file " << intersect_with_set_from_file_fname << endl;
		}
		else if (stringcmp(argv[i], "-arc_with_given_set_as_s_lines_after_dualizing") == 0) {
			f_arc_with_given_set_as_s_lines_after_dualizing = TRUE;
			arc_size = strtoi(argv[++i]);
			arc_d = strtoi(argv[++i]);
			arc_d_low = strtoi(argv[++i]);
			arc_s = strtoi(argv[++i]);
			cout << "-arc_with_given_set_as_s_lines_after_dualizing "
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << endl;
		}
		else if (stringcmp(argv[i], "-arc_with_two_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_two_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = strtoi(argv[++i]);
			arc_d = strtoi(argv[++i]);
			arc_d_low = strtoi(argv[++i]);
			arc_s = strtoi(argv[++i]);
			arc_t = strtoi(argv[++i]);
			t_lines_string.assign(argv[++i]);
			cout << "-arc_with_two_given_sets_of_lines_after_dualizing src_size="
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " t=" << arc_t << " " << t_lines_string << endl;
		}
		else if (stringcmp(argv[i], "-arc_with_three_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_three_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = strtoi(argv[++i]);
			arc_d = strtoi(argv[++i]);
			arc_d_low = strtoi(argv[++i]);
			arc_s = strtoi(argv[++i]);
			arc_t = strtoi(argv[++i]);
			t_lines_string.assign(argv[++i]);
			arc_u = strtoi(argv[++i]);
			u_lines_string.assign(argv[++i]);
			cout << "-arc_with_three_given_sets_of_lines_after_dualizing "
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << endl;
			cout << "arc_t = " << arc_t << " t_lines_string = " << t_lines_string << endl;
			cout << "arc_u = " << arc_u << " u_lines_string = " << u_lines_string << endl;
		}
		else if (stringcmp(argv[i], "-dualize_hyperplanes_to_points") == 0) {
			f_dualize_hyperplanes_to_points = TRUE;
			cout << "-dualize_hyperplanes_to_points" << endl;
		}
		else if (stringcmp(argv[i], "-dualize_points_to_hyperplanes") == 0) {
			f_dualize_points_to_hyperplanes = TRUE;
			cout << "-dualize_points_to_hyperplanes" << endl;
		}
		else if (stringcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "projective_space_job_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
		cout << "projective_space_job_description::read_arguments looping, i=" << i << endl;
	} // next i
	cout << "projective_space_job_description::read_arguments done" << endl;
	return i + 1;
}



}}

