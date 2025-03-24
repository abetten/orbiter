/*
 * orthogonal_space_activity.cpp
 *
 *  Created on: Jan 12, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


orthogonal_space_activity::orthogonal_space_activity()
{
	Record_birth();
	Descr = NULL;
	OA = NULL;
	//Blt_set_domain = NULL;

}

orthogonal_space_activity::~orthogonal_space_activity()
{
	Record_death();
#if 0
	if (Blt_set_domain) {
		FREE_OBJECT(Blt_set_domain);
	}
	if (OA) {
		FREE_OBJECT(OA);
	}
#endif
}

void orthogonal_space_activity::init(
		orthogonal_space_activity_description *Descr,
		orthogonal_space_with_action *OA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_activity::init" << endl;
	}

	orthogonal_space_activity::Descr = Descr;
	orthogonal_space_activity::OA = OA;



	if (f_v) {
		cout << "orthogonal_space_activity::init done" << endl;
	}
}


void orthogonal_space_activity::perform_activity(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orthogonal_space_activity::perform_activity" << endl;
	}


	if (Descr->f_cheat_sheet_orthogonal) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity before OA->report" << endl;
		}

		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->cheat_sheet_orthogonal_draw_options_label);


		OA->report(
				Draw_options,
				verbose_level);


		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity after OA->report" << endl;
		}

	}

	else if (Descr->f_print_points) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_print_points" << endl;
		}

		long int *Pts;
		int nb_pts;
		string label_txt;

		label_txt.assign(Descr->print_points_label);

		Get_vector_or_set(Descr->print_points_label, Pts, nb_pts);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity nb_pts = " << nb_pts << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity before OA->O->report_point_set" << endl;
		}
		OA->O->report_point_set(
				Pts, nb_pts,
				label_txt,
				verbose_level);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity after OA->O->report_point_set" << endl;
		}

		FREE_lint(Pts);

	}

	else if (Descr->f_print_lines) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_print_lines" << endl;
		}

		long int *Lines;
		int nb_lines;
		string label_txt;

		label_txt.assign(Descr->print_lines_label);

		Get_vector_or_set(Descr->print_lines_label, Lines, nb_lines);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity nb_lines = " << nb_lines << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity before OA->O->report_line_set" << endl;
		}


		OA->O->report_line_set(
				Lines, nb_lines,
				label_txt,
				verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity after OA->O->report_line_set" << endl;
		}

		FREE_lint(Lines);

	}

	else if (Descr->f_unrank_line_through_two_points) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_unrank_line_through_two_points" << endl;
		}

		long int p1;
		long int p2;
		long int rk;
		other::data_structures::string_tools ST;

		p1 = ST.strtolint(Descr->unrank_line_through_two_points_p1);
		p2 = ST.strtolint(Descr->unrank_line_through_two_points_p2);

		if (f_v) {
			cout << "point rank p1 = " << p1 << endl;
			cout << "point rank p2 = " << p2 << endl;
		}

		rk = OA->O->Hyperbolic_pair->rank_line(p1, p2, verbose_level);

		if (true) {
			cout << "line rank = " << rk << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_unrank_line_through_two_points done" << endl;
		}

	}

	else if (Descr->f_lines_on_point) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_lines_on_point" << endl;
		}

		long int *line_pencil_line_ranks;

		line_pencil_line_ranks = NEW_lint(OA->O->Hyperbolic_pair->alpha);

		if (f_v) {
			cout << "point rank = " << Descr->lines_on_point_rank << endl;
		}

		OA->O->lines_on_point_by_line_rank(Descr->lines_on_point_rank,
				line_pencil_line_ranks, verbose_level);

		if (true) {
			cout << "There are " << OA->O->Hyperbolic_pair->alpha << " lines on point = "
					<< Descr->lines_on_point_rank << ". They are: ";
			Lint_vec_print_fully(cout, line_pencil_line_ranks, OA->O->Hyperbolic_pair->alpha);
			cout << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_lines_on_point done" << endl;
		}

	}

	else if (Descr->f_perp) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_perp" << endl;
		}

		long int *pts;
		int nb_pts;

		Get_vector_or_set(Descr->perp_text, pts, nb_pts);

		if (f_v) {
			cout << "Computing the common perp of the set ";
			Lint_vec_print(cout, pts, nb_pts);
			cout << endl;
		}

		long int *Perp;
		int sz;

		if (nb_pts >= 2) {

			if (f_v) {
				cout << "orthogonal_space_activity::perform_activity "
						"before OA->O->perp_of_k_points" << endl;
			}
			OA->O->perp_of_k_points(pts, nb_pts, Perp, sz, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_activity::perform_activity "
						"after OA->O->perp_of_k_points" << endl;
			}

		}
		else if (nb_pts == 1) {


			Perp = NEW_lint(OA->O->Hyperbolic_pair->alpha * (OA->O->Quadratic_form->q + 1));

			if (f_v) {
				cout << "orthogonal_space_activity::perform_activity "
						"before OA->O->perp" << endl;
			}
			OA->O->perp(pts[0], Perp, sz, verbose_level);
			if (f_v) {
				cout << "orthogonal_space_activity::perform_activity "
						"after OA->O->perp" << endl;
			}

		}
		else {
			cout << "orthogonal_space_activity::perform_activity nb_pts = " << nb_pts << endl;
			exit(1);
		}

		cout << "The perp of the set has size " << sz << endl;

		cout << "The perp is the following set:" << endl;
		Lint_vec_print_fully(cout, Perp, sz);
		cout << endl;


		other::orbiter_kernel_system::file_io Fio;

		string fname;


		fname = "perp_of_" + Descr->perp_text + "_q" + std::to_string(OA->O->F->q) + ".csv";

		Fio.Csv_file_support->lint_matrix_write_csv(
				fname, Perp, sz, 1);




		if (f_v) {
			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}


		FREE_lint(Perp);
		FREE_lint(pts);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_perp done" << endl;
		}

	}

	else if (Descr->f_set_stabilizer) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"f_set_stabilizer" << endl;
		}

 		set_stabilizer(OA,
				Descr->set_stabilizer_intermediate_set_size,
				Descr->set_stabilizer_fname_mask,
				Descr->set_stabilizer_nb,
				Descr->set_stabilizer_column_label,
				Descr->set_stabilizer_fname_out,
				verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"f_set_stabilizer done" << endl;
		}

	}
	if (Descr->f_export_point_line_incidence_matrix) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"f_export_point_line_incidence_matrix" << endl;
		}
		OA->O->export_incidence_matrix_to_csv(verbose_level);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"f_export_point_line_incidence_matrix done" << endl;
		}
	}


	if (Descr->f_intersect_with_subspace) {
		cout << "orthogonal_space_activity::perform_activity "
				"intersect_with_subspace " << Descr->intersect_with_subspace_label << endl;

		int *Basis;
		int m, n;

		Get_matrix(Descr->intersect_with_subspace_label, Basis, m, n);

		if (n != OA->O->Quadratic_form->n) {
			cout << "error: the width of the matrix does not match with the orthogonal space" << endl;
			cout << "width = " << n << endl;
			cout << "dimension of orthogonal space = " << OA->O->Quadratic_form->n << endl;
			exit(1);
		}

		cout << "Intersecting with subspace generated by " << endl;
		Int_matrix_print(Basis, m, n);

		long int *the_points;
		int nb_points;

		if (f_v) {
			cout << "before OA->O->intersection_with_subspace" << endl;
		}
		OA->O->intersection_with_subspace(Basis, m,
				the_points, nb_points, verbose_level);

		cout << "We found that the intersection contains " << nb_points << " points. " << endl;
		cout << "Using ranks in PG, the set of points is: " << endl;
		Lint_vec_print(cout, the_points, nb_points);
		cout << endl;

	}
	else if (Descr->f_table_of_blt_sets) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_table_of_blt_sets" << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"before OA->make_table_of_blt_sets" << endl;
		}
		OA->make_table_of_blt_sets(verbose_level);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"after OA->make_table_of_blt_sets" << endl;
		}
	}
	else if (Descr->f_create_orthogonal_reflection) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_create_orthogonal_reflection" << endl;
		}

		long int *pts;
		int nb_pts;

		Get_vector_or_set(Descr->create_orthogonal_reflection_points, pts, nb_pts);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Computing the orthogonal reflections associated with the set of points ";
			Lint_vec_print(cout, pts, nb_pts);
			cout << endl;
		}

		data_structures_groups::vector_ge *vec;

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"before OA->create_orthogonal_reflections" << endl;
		}
		OA->create_orthogonal_reflections(
				pts, nb_pts,
				vec,
				verbose_level);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"after OA->create_orthogonal_reflections" << endl;
		}




		other::orbiter_kernel_system::file_io Fio;
		std::string fname;

		fname = OA->O->label_txt + "_orthogonal_reflections.csv";

		vec->save_csv(
				fname, verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

	}
	else if (Descr->f_create_Siegel_transformation) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_create_Siegel_transformation" << endl;
		}

		int *u;
		int *v;
		int sz1;
		int sz2;

		Int_vec_scan(Descr->create_Siegel_transformation_u, u, sz1);
		Int_vec_scan(Descr->create_Siegel_transformation_v, v, sz2);

		if (sz1 != sz2) {
			cout << "orthogonal_space_activity::perform_activity sz1 != sz2" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Computing the Siegel transformation associated with ";
			cout << "u = ";
			Int_vec_print(cout, u, sz1);
			cout << endl;
			cout << "v = ";
			Int_vec_print(cout, v, sz1);
			cout << endl;
		}

		data_structures_groups::vector_ge *vec;

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"before OA->create_Siegel_transformation" << endl;
		}
		OA->create_Siegel_transformation(
				u, v, sz1,
				vec,
				verbose_level);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"after OA->create_Siegel_transformation" << endl;
		}




		other::orbiter_kernel_system::file_io Fio;
		std::string fname;

		fname = OA->O->label_txt + "_Siegel_transformation.csv";

		vec->save_csv(
				fname, verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

	}
	else if (Descr->f_make_all_Siegel_transformations) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_make_all_Siegel_transformations" << endl;
		}

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"before OA->O->Orthogonal_group->make_all_Siegel_Transformations" << endl;
		}
		OA->O->Orthogonal_group->make_all_Siegel_Transformations(
				verbose_level - 2);
		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"after OA->O->Orthogonal_group->make_all_Siegel_Transformations" << endl;
		}

	}

	else if (Descr->f_create_orthogonal_reflection_6_and_4) {

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity f_create_orthogonal_reflection_6_and_4" << endl;
		}


		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity using group " << Descr->create_orthogonal_reflection_6_and_4_A4 << endl;
		}

		groups::any_group
			*AG;

		AG = Get_any_group(Descr->create_orthogonal_reflection_6_and_4_A4);

		long int *pts;
		int nb_pts;

		Get_vector_or_set(Descr->create_orthogonal_reflection_6_and_4_points, pts, nb_pts);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Computing the orthogonal reflections associated with the set of points ";
			Lint_vec_print(cout, pts, nb_pts);
			cout << endl;
		}

		data_structures_groups::vector_ge *vec6;
		data_structures_groups::vector_ge *vec4;

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"before OA->create_orthogonal_reflections_6x6_and_4x4" << endl;
		}


		OA->create_orthogonal_reflections_6x6_and_4x4(
				pts, nb_pts,
				AG->A /* actions::action *A4 */,
				vec6,
				vec4,
				verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"after OA->create_orthogonal_reflections_6x6_and_4x4" << endl;
		}



		other::orbiter_kernel_system::file_io Fio;
		std::string fname;


		fname = OA->O->label_txt + "_orthogonal_reflections_6x6.csv";

		vec6->save_csv(
				fname, verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}


		fname = OA->O->label_txt + "_orthogonal_reflections_4x4.csv";

		vec4->save_csv(
				fname, verbose_level);

		if (f_v) {
			cout << "orthogonal_space_activity::perform_activity "
					"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
		}

	}




	if (f_v) {
		cout << "orthogonal_space_activity::perform_activity done" << endl;
	}

}


void orthogonal_space_activity::set_stabilizer(
		orthogonal_space_with_action *OA,
		int intermediate_subset_size,
		std::string &fname_mask, int nb, std::string &column_label,
		std::string &fname_out,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "orthogonal_space_activity::set_stabilizer" << endl;
	}


	set_stabilizer::substructure_classifier SubC;

	if (f_v) {
		cout << "orthogonal_space_activity::set_stabilizer "
				"before SubC.set_stabilizer_in_any_space" << endl;
	}
	SubC.set_stabilizer_in_any_space(
			OA->A, OA->A, OA->A->Strong_gens,
			intermediate_subset_size,
			fname_mask, nb, column_label,
			fname_out,
			verbose_level);
	if (f_v) {
		cout << "orthogonal_space_activity::set_stabilizer "
				"after SubC.set_stabilizer_in_any_space" << endl;
	}



#if 0
	top_level_geometry_global T;

	T.set_stabilizer_orthogonal_space(
				OA,
				intermediate_subset_size,
				fname_mask, nb, column_label,
				verbose_level);
#endif

	if (f_v) {
		cout << "orthogonal_space_activity::set_stabilizer done" << endl;
	}

}




}}}

