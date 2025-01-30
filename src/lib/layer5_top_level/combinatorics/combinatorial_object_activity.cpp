/*
 * combinatorial_object_activity.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combinatorial_object_activity::combinatorial_object_activity()
{
	Record_birth();
	Descr = NULL;

	f_has_geometric_object = false;
	GOC = NULL;

	f_has_combo = false;
	Combo = NULL;

	nb_output = 0;
	Output = NULL;

}

combinatorial_object_activity::~combinatorial_object_activity()
{
	Record_death();
}


void combinatorial_object_activity::init(
		combinatorial_object_activity_description *Descr,
		geometry::other_geometry::geometric_object_create *GOC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	f_has_geometric_object = true;
	combinatorial_object_activity::GOC = GOC;

	if (f_v) {
		cout << "combinatorial_object_activity::init done" << endl;
	}
}



void combinatorial_object_activity::init_combo(
		combinatorial_object_activity_description *Descr,
		apps_combinatorics::combinatorial_object_stream *Combo,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream" << endl;
	}

	combinatorial_object_activity::Descr = Descr;
	f_has_combo = true;
	combinatorial_object_activity::Combo = Combo;


	if (f_v) {
		cout << "combinatorial_object_activity::init_input_stream done" << endl;
	}
}

void combinatorial_object_activity::perform_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
		cout << "combinatorial_object_activity::perform_activity verbose_level = " << verbose_level << endl;
	}

	if (f_has_geometric_object) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"before perform_activity_geometric_object" << endl;
		}
		perform_activity_geometric_object(AO, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"after perform_activity_geometric_object" << endl;
		}
	}
	else if (f_has_combo) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"before perform_activity_combo" << endl;
		}
		perform_activity_combo(AO, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"after perform_activity_combo" << endl;
		}
	}



	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_geometric_object(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_geometric_object" << endl;
	}




	if (Descr->f_line_type_old) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_line_type_old" << endl;
		}

		geometry::projective_geometry::projective_space *P;

		P = GOC->Descr->P;


		int *type;

		type = NEW_int(P->Subspaces->N_lines);


		P->Subspaces->line_intersection_type(
				GOC->Pts, GOC->nb_pts, type, verbose_level - 1);
			// type[N_lines]

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object type:" << endl;
			Int_vec_print_fully(cout, type, P->Subspaces->N_lines);
			cout << endl;
		}

		other::data_structures::tally T;

		T.init(type, P->Subspaces->N_lines, false, 0);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object line type:" << endl;
			T.print(true /* f_backwards*/);
			cout << endl;
			T.print_array_tex(cout, false /* f_backwards*/);
		}

	}
	else if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_line_type" << endl;
		}

		geometry::projective_geometry::projective_space *P;

		P = GOC->Descr->P;

		int threshold = 3;


		geometry::other_geometry::intersection_type *Int_type;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"before P->line_intersection_type" << endl;
		}

		P->line_intersection_type(
				GOC->Pts, GOC->nb_pts, threshold,
			Int_type,
			verbose_level - 2);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"after P->line_intersection_type" << endl;
		}

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object: "
					"We found " << Int_type->len << " intersection types." << endl;

			int i;

			for (i = 0; i < Int_type->len; i++) {
				cout << setw(3) << i << " : " << Int_type->R[i]
					<< " : " << setw(5) << Int_type->nb_pts_on_subspace[i] << " : ";
				Lint_vec_print(cout, Int_type->Pts_on_subspace[i], Int_type->nb_pts_on_subspace[i]);
				cout << endl;
			}
		}

		FREE_OBJECT(Int_type);

	}
	else if (Descr->f_conic_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"f_conic_type" << endl;
		}

		geometry::projective_geometry::projective_space *P;

		P = GOC->Descr->P;

		long int **Pts_on_conic;
		int **Conic_eqn;
		int *nb_pts_on_conic;
		int len;
		int i;

		P->Plane->conic_type(
				GOC->Pts, GOC->nb_pts,
				Descr->conic_type_threshold,
				Pts_on_conic, Conic_eqn, nb_pts_on_conic, len,
				verbose_level);


		cout << "We found " << len << " conics" << endl;
		for (i = 0; i < len; i++) {
			cout << i << " : " << nb_pts_on_conic << endl;
		}


	}
	else if (Descr->f_non_conical_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"f_non_conical_type" << endl;
		}

		geometry::projective_geometry::projective_space *P;

		P = GOC->Descr->P;

		std::vector<int> Rk;

		P->Plane->determine_nonconical_six_subsets(
				GOC->Pts, GOC->nb_pts,
				Rk,
				verbose_level);

		cout << "We found " << Rk.size() << " non-conical 6 subsets" << endl;

	}
	else if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"f_ideal" << endl;
		}

		algebra::ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->ideal_ring_label);


		HPD->explore_vanishing_ideal(GOC->Pts, GOC->nb_pts, verbose_level);

#if 0
		int *Kernel;
		int r;

		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC the input set has size " << GOC->nb_pts << endl;
			cout << "combinatorial_object_activity::perform_activity_GOC the input set is: " << endl;
			Lint_vec_print(cout, GOC->Pts, GOC->nb_pts);
			cout << endl;
			//P->print_set_numerical(cout, GOC->Pts, GOC->nb_pts);
		}


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC before HPD->vanishing_ideal" << endl;
		}
		HPD->vanishing_ideal(GOC->Pts, GOC->nb_pts,
				r, Kernel, verbose_level - 1);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_GOC after HPD->vanishing_ideal" << endl;
		}


		int h; //, ns;
		int nb_pts;
		long int *Pts;

		for (h = 0; h < r; h++) {
			cout << "generator " << h << " / " << r << " is ";
			HPD->print_equation_relaxed(cout, Kernel + h * HPD->get_nb_monomials());
			cout << endl;

		}

		//ns = HPD->get_nb_monomials() - r; // dimension of null space

		cout << "looping over all generators of the ideal:" << endl;
		for (h = 0; h < r; h++) {
			cout << "generator " << h << " / " << r << " is ";
			Int_vec_print(cout, Kernel + h * HPD->get_nb_monomials(), HPD->get_nb_monomials());
			cout << " : " << endl;

			vector<long int> Points;
			int i;

			HPD->enumerate_points(Kernel + h * HPD->get_nb_monomials(),
					Points, verbose_level);
			nb_pts = Points.size();

			Pts = NEW_lint(nb_pts);
			for (i = 0; i < nb_pts; i++) {
				Pts[i] = Points[i];
			}


			cout << "We found " << nb_pts << " points on the generator of the ideal" << endl;
			cout << "They are : ";
			Lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			//P->print_set_numerical(cout, Pts, nb_pts);

			FREE_lint(Pts);

		} // next h
#endif

	}
	else if (Descr->f_save) {

		other::orbiter_kernel_system::file_io Fio;
		string fname;

		fname = GOC->label_txt + ".txt";

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, GOC->Pts, GOC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}
#if 0
	else {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object creating stream" << endl;
		}
		Combo = NEW_OBJECT(apps_combinatorics::combinatorial_object_stream);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"before Combo->init_from_geometric_object" << endl;
		}
		Combo->init_from_geometric_object(
				GOC,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object "
					"after Combo->init_from_geometric_object" << endl;
		}

		f_has_combo = true;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"before perform_activity_combo" << endl;
		}
		perform_activity_combo(AO, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"after perform_activity_combo" << endl;
		}

		f_has_geometric_object = false;
		//f_has_combo = false;

	}
#endif

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_geometric_object "
				"done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_combo(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_combo" << endl;
		cout << "combinatorial_object_activity::perform_activity_combo verbose_level=" << verbose_level << endl;
	}

	if (Descr->f_canonical_form_PG) {



		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_canonical_form_PG" << endl;
		}

		projective_geometry::projective_space_with_action *PA;
		geometry::projective_geometry::projective_space *P;


		if (Descr->f_canonical_form_PG_has_PA) {
			PA = Descr->Canonical_form_PG_PA;
		}
		else {

			PA = Get_projective_space(Descr->canonical_form_PG_PG_label);
		}
		P = PA->P;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->do_canonical_form" << endl;
		}
		Combo->do_canonical_form(
				Descr->Canonical_form_PG_Descr,
				true /* f_projective_space */, PA, P,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->do_canonical_form" << endl;
		}

	}
	else if (Descr->f_canonical_form) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_canonical_form" << endl;
		}

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->do_canonical_form" << endl;
		}
		Combo->do_canonical_form(
				Descr->Canonical_form_Descr,
				false /* f_projective_space */, NULL, NULL,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->do_canonical_form" << endl;
		}

	}

	else if (Descr->f_get_combo_with_group) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"f_get_combo_with_group, iso type idx = " << Descr->get_combo_with_group_idx << endl;
		}


		//combinatorics::canonical_form_classification::any_combinatorial_object *OwCF;

		//OwCF = Combo->Objects_after_classification->Classification_of_objects->OWCF_transversal[Descr->get_combo_with_group_idx];

		canonical_form::combinatorial_object_with_properties *OwP;

		OwP = &Combo->Objects_after_classification->OwP[Descr->get_combo_with_group_idx];


		nb_output = 1;
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

		string output_label;



		output_label = Combo->IS->Descr->label_txt + "_object_" + std::to_string(Descr->get_combo_with_group_idx);


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"f_get_combo_with_group, output_label = " << output_label << endl;
		}

		Output->init_combo_with_group(output_label, OwP, verbose_level);


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"f_get_combo_with_group done" << endl;
		}
	}


	if (Descr->f_report) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_report" << endl;
		}

		if (Combo->Objects_after_classification == NULL) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"Combo->Objects_after_classification == NULL" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before classification_report" << endl;
		}
		Combo->Objects_after_classification->classification_report(
				Descr->Objects_report_options,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after classification_report" << endl;
		}
	}
	else if (Descr->f_draw_incidence_matrices) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_draw_incidence_matrices" << endl;
		}

		other::graphics::draw_incidence_structure_description *Draw_incidence_structure_description;

		Draw_incidence_structure_description = Get_draw_incidence_structure_options(Descr->draw_incidence_matrices_options_label);

		Combo->draw_incidence_matrices(
				Descr->draw_incidence_matrices_prefix,
				Draw_incidence_structure_description,
				verbose_level);

	}
	else if (Descr->f_test_distinguishing_property) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_test_distinguishing_property" << endl;
		}

		combinatorics::graph_theory::colored_graph *CG;


		CG = Get_object_of_type_graph(Descr->test_distinguishing_property_graph);


		Combo->do_test_distinguishing_property(
					CG,
					verbose_level);


	}
	else if (Descr->f_covering_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_covering_type" << endl;
		}

		orbits::orbits_create *Orbits;


		Orbits = Get_orbits(Descr->covering_type_orbits);


		Combo->do_covering_type(
				Orbits,
					Descr->covering_type_size,
					Descr->f_filter_by_Steiner_property,
					AO,
					verbose_level);


	}



	else if (Descr->f_compute_frequency) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_compute_frequency" << endl;
		}

		combinatorics::graph_theory::colored_graph *CG;


		CG = Get_object_of_type_graph(Descr->compute_frequency_graph);

		Combo->do_compute_frequency_graph(
				CG,
					verbose_level);



	}
	else if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_ideal" << endl;
		}

		algebra::ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->ideal_ring_label);

		Combo->do_compute_ideal(
				HPD,
					verbose_level);


	}
	else if (Descr->f_save_as) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_save_as " << Descr->save_as_fname << endl;
		}

		Combo->do_save(Descr->save_as_fname,
				false, NULL, 0,
				verbose_level);


	}
	else if (Descr->f_extract_subset) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_extract_subset " << Descr->extract_subset_fname << endl;
		}
		long int *extract_idx_set;
		int extract_size;

		Get_lint_vector_from_label(
				Descr->extract_subset_set,
				extract_idx_set, extract_size,
				0 /* verbose_level */);
		Combo->do_save(
				Descr->extract_subset_fname,
				true /* f_extract */, extract_idx_set, extract_size,
				verbose_level);


	}
	else if (Descr->f_unpack_from_restricted_action) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"unpack_from_restricted_action "
					<< Descr->unpack_from_restricted_action_prefix
					<< " " << Descr->unpack_from_restricted_action_group_label << endl;
		}

		groups::any_group *G;

		G = Get_any_group(Descr->unpack_from_restricted_action_group_label);

		Combo->unpack_from_restricted_action(
				Descr->unpack_from_restricted_action_prefix,
				G,
				verbose_level);

	}

	else if (Descr->f_line_covering_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"line_covering_type "
					<< Descr->line_covering_type_prefix
					<< " " << Descr->line_covering_type_projective_space
					<< " " << Descr->line_covering_type_lines
					<< endl;
		}

		projective_geometry::projective_space_with_action *PA;

		PA = Get_projective_space(Descr->line_covering_type_projective_space);


		Combo->line_covering_type(
				Descr->line_covering_type_prefix,
				PA,
				Descr->line_covering_type_lines,
				verbose_level);

	}

	else if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_line_type is currently disabled" << endl;
		}

		exit(1);
#if 0
		projective_geometry::projective_space_with_action *PA;

		PA = Get_projective_space(Descr->line_type_projective_space_label);

		Combo->line_type(
					Descr->line_type_prefix,
					PA,
					verbose_level);
#endif


	}

	else if (Descr->f_activity) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_activity "
					<< endl;
		}


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->do_activity "
					<< endl;
		}
		Combo->do_activity(
				Descr->Activity_description,
				AO,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->do_activity "
					<< endl;
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->do_activity nb_cols = " << AO->nb_cols
					<< endl;
		}



	}
	else if (Descr->f_algebraic_degree) {



		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_algebraic_degree" << endl;
		}

		projective_geometry::projective_space_with_action *PA;



		PA = Get_projective_space(Descr->algebraic_degree_PG_label);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->do_algebraic_degree" << endl;
		}


		int *Algebraic_degree;

		Combo->do_algebraic_degree(
				PA,
				Algebraic_degree,
				verbose_level);

		cout << "combinatorial_object_activity::perform_activity_combo Algebraic_degree=";
		Int_vec_print(cout, Algebraic_degree, Combo->IS->Objects.size());
		cout << endl;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->do_algebraic_degree" << endl;
		}

	}

}






}}}




