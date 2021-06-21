/*
 * projective_space_job_description.cpp
 *
 *  Created on: Apr 28, 2019
 *      Author: betten
 */


#if 0

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {



projective_space_job_description::projective_space_job_description()
{

	f_input = FALSE;
	Data = NULL;

	f_fname_base_out = FALSE;
	//fname_base_out;

#if 0
	f_q = FALSE;
	q = 0;

	f_n = FALSE;
	n = 0;

	f_poly = FALSE;
	//poly = NULL;
#endif

	f_embed = FALSE;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	f_andre = FALSE;

#if 0
		// follow up option for f_andre:
		f_Q = FALSE;
		Q = 0;
		f_poly_Q = FALSE;
		//poly_Q = NULL;
#endif

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
		//f_embedded = FALSE;
		//f_sideways = FALSE;

	f_canonical_form = FALSE;
	//canonical_form_fname_base;

	f_ideal_LEX = FALSE;
	f_ideal_PART = FALSE;
	ideal_degree = 0;

	f_intersect_with_set_from_file = FALSE;
	//intersect_with_set_from_file_fname = NULL;

}

projective_space_job_description::~projective_space_job_description()
{

}


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
#if 0
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
#endif
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
		else if (stringcmp(argv[i], "-intersect_with_set_from_file") == 0) {
			f_intersect_with_set_from_file = TRUE;
			intersect_with_set_from_file_fname.assign(argv[++i]);
			cout << "-intersect_with_set_from_file " << intersect_with_set_from_file_fname << endl;
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


void projective_space_job_description::print()
{
#if 0
	if (f_q) {
		cout << "-q " << q << endl;
	}
	else if (f_Q) {
		cout << "-Q " << Q << endl;
	}
	else if (f_n) {
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
#endif
	if (f_input) {
		cout << "-input" << endl;
		Data->print();
	}
	if (f_fname_base_out) {
		cout << "-fname_base_out " << fname_base_out << endl;
	}
	if (f_embed) {
		cout << "-embed" << endl;
	}
	if (f_orthogonal) {
		cout << "-orthogonal " << orthogonal_epsilon << endl;
	}
	if (f_homogeneous_polynomials_LEX) {
		cout << "-homogeneous_polynomials_LEX " << homogeneous_polynomials_degree << endl;
	}
	if (f_homogeneous_polynomials_PART) {
		cout << "-homogeneous_polynomials_PART " << homogeneous_polynomials_degree << endl;
	}
	if (f_andre) {
		cout << "-andre " << endl;
	}
	if (f_print) {
		cout << "-print " << endl;
	}
	if (f_lines_in_PG) {
		cout << "-lines_in_PG " << endl;
	}
	if (f_points_in_PG) {
		cout << "-points_in_PG " << endl;
	}
	if (f_points_on_grassmannian) {
		cout << "-points_on_grassmannian " << points_on_grassmannian_k << endl;
	}
	if (f_list_group_elements) {
		cout << "-list_group_elements" << endl;
	}
	if (f_line_type) {
		cout << "-line_type" << endl;
	}
	if (f_plane_type) {
		cout << "-plane_type" << endl;
	}
	if (f_plane_type_failsafe) {
		cout << "-plane_type_failsafe" << endl;
	}
	if (f_conic_type) {
		cout << "-conic_type " << endl;
	}
	if (f_randomized) {
		cout << "-randomized " << nb_times << endl;
	}
	if (f_hyperplane_type) {
		cout << "-hyperplane_type" << endl;
	}
	if (f_show) {
		cout << "-show" << endl;
	}
	if (f_cone_over) {
		cout << "-cone_over" << endl;
	}
	if (f_bsf3) {
		cout << "-bsf3" << endl;
	}
	if (f_test_diagonals) {
		cout << "-test_diagonals " << test_diagonals_fname << endl;
	}
	if (f_klein) {
		cout << "-klein" << endl;
	}
	if (f_draw_points_in_plane) {
		cout << "-draw_points_in_plane" << draw_points_in_plane_fname_base << endl;
	}



	if (f_point_labels) {
		cout << "-point_labels" << endl;
	}

	if (f_canonical_form) {
		cout << "-canonical_form" << canonical_form_fname_base << endl;
	}

	if (f_canonical_form) {
		cout << "-ideal_LEX " << ideal_degree << endl;
	}
	if (f_ideal_PART) {
		cout << "-ideal_PART " << ideal_degree << endl;
	}
	if (f_intersect_with_set_from_file) {
		cout << "-intersect_with_set_from_file " << intersect_with_set_from_file_fname << endl;
	}
}


}}

#endif

