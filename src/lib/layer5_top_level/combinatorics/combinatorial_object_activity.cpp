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
	Descr = NULL;

	f_has_geometric_object = false;
	GOC = NULL;

	f_has_combo = false;
	Combo = NULL;

}

combinatorial_object_activity::~combinatorial_object_activity()
{
}


void combinatorial_object_activity::init(
		combinatorial_object_activity_description *Descr,
		geometry::geometric_object_create *GOC,
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
		apps_combinatorics::combinatorial_object *Combo,
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
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity" << endl;
	}
	if (f_has_geometric_object) {
		perform_activity_geometric_object(verbose_level);
	}
	else if (f_has_combo) {
		perform_activity_combo(verbose_level);
	}
	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_geometric_object(
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

		geometry::projective_space *P;

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

		data_structures::tally T;

		T.init(type, P->Subspaces->N_lines, false, 0);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object line type:" << endl;
			T.print(true /* f_backwards*/);
			cout << endl;
			T.print_array_tex(cout, false /* f_backwards*/);
		}

	}

	if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_line_type" << endl;
		}

		geometry::projective_space *P;

		P = GOC->Descr->P;

		int threshold = 3;


		geometry::intersection_type *Int_type;

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


	if (Descr->f_conic_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_conic_type" << endl;
		}

		geometry::projective_space *P;

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
		//

	}

	if (Descr->f_non_conical_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_conic_type" << endl;
		}

		geometry::projective_space *P;

		P = GOC->Descr->P;

		std::vector<int> Rk;

		P->Plane->determine_nonconical_six_subsets(
				GOC->Pts, GOC->nb_pts,
				Rk,
				verbose_level);

		cout << "We found " << Rk.size() << " non-conical 6 subsets" << endl;

	}


	if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_geometric_object f_ideal" << endl;
		}

		ring_theory::homogeneous_polynomial_domain *HPD;


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


	if (Descr->f_save) {

		orbiter_kernel_system::file_io Fio;
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

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_geometric_object "
				"done" << endl;
	}
}


void combinatorial_object_activity::perform_activity_combo(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::perform_activity_combo" << endl;
	}

	if (Descr->f_canonical_form_PG) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_canonical_form_PG" << endl;
		}

		projective_geometry::projective_space_with_action *PA;


		if (Descr->f_canonical_form_PG_has_PA) {
			PA = Descr->Canonical_form_PG_PA;
		}
		else {

			PA = Get_projective_space(Descr->canonical_form_PG_PG_label);
		}


		Combo->Classification = NEW_OBJECT(combinatorics::classification_of_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before CO->perform_classification" << endl;
		}
		Combo->Classification->perform_classification(
				Descr->Canonical_form_PG_Descr,
				true /* f_projective_space */, PA->P,
				Combo->IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after CO->perform_classification" << endl;
		}



		Combo->Classification_CO = NEW_OBJECT(canonical_form::classification_of_combinatorial_objects);

		if (!Combo->Data_input_stream_description) {
			cout << "please use -label <label_txt> <label_tex> "
					"when defining the input stream for the combinatorial object" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->Classification_CO->init_after_nauty" << endl;
		}
		Combo->Classification_CO->init_after_nauty(
				Combo->Data_input_stream_description->label_txt,
				Combo->Classification,
				true /* f_projective_space */, PA,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->Classification_CO->init_after_nauty" << endl;
		}


#if 0
		object_with_properties *OwP;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before post_process_classification" << endl;
		}
		post_process_classification(
					CO,
					OwP,
					true /* f_projective_space */, PA,
					CO->Descr->label,
					verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after post_process_classification" << endl;
		}
#endif


#if 0
		//FREE_OBJECTS(OwP);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before FREE_OBJECT(C)" << endl;
		}
		FREE_OBJECT(C);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after FREE_OBJECT(C)" << endl;
		}
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before FREE_OBJECT(CO)" << endl;
		}
		FREE_OBJECT(CO);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after FREE_OBJECT(CO)" << endl;
		}
#endif


	}
	if (Descr->f_report) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before classification_report" << endl;
		}
		Combo->Classification_CO->classification_report(
				Descr->Classification_of_objects_report_options,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after classification_report" << endl;
		}
#if 0
		string fname_base;

		fname_base = Descr->Classification_of_objects_report_options->prefix;
		Combo->Classification_CO->classification_write_file(
				fname_base,
				verbose_level);
#endif
	}
	else if (Descr->f_canonical_form) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_canonical_form" << endl;
		}
#if 0
		classification_of_objects::perform_classification(
				classification_of_objects_description *Descr,
				int f_projective_space,
				geometry::projective_space *P,
				data_structures::data_input_stream *IS,
				int verbose_level)
#endif


		Combo->Classification = NEW_OBJECT(combinatorics::classification_of_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before Combo->Classification->perform_classification" << endl;
		}
		Combo->Classification->perform_classification(
				Descr->Canonical_form_Descr,
				false /* f_projective_space */, NULL /* P */,
				Combo->IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after Combo->Classification->perform_classification" << endl;
		}

#if 0
		combinatorics::classification_of_objects *CO;

		CO = NEW_OBJECT(combinatorics::classification_of_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before CO->perform_classification" << endl;
		}
		CO->perform_classification(
				Descr->Canonical_form_Descr,
				false /* f_projective_space */, NULL /* P */,
				IS,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after CO->perform_classification" << endl;
		}
#endif

		//canonical_form::classification_of_combinatorial_objects *C;

		Combo->Classification_CO = NEW_OBJECT(canonical_form::classification_of_combinatorial_objects);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before C->init_after_nauty" << endl;
		}
		Combo->Classification_CO->init_after_nauty(
				Combo->Classification->Descr->label,
				Combo->Classification,
				false /* f_projective_space */, NULL,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after C->init_after_nauty" << endl;
		}

#if 0
		object_with_properties *OwP;

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before post_process_classification" << endl;
		}
		post_process_classification(
					CO,
					OwP,
					false /* f_projective_space */, NULL,
					CO->Descr->label,
					verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after post_process_classification" << endl;
		}


		FREE_OBJECTS(OwP);
#endif


#if 0
		string fname_base;

		fname_base = Descr->Classification_of_objects_report_options->prefix;
		Combo->Classification_CO->classification_write_file(
				fname_base,
				verbose_level);
#endif


#if 0
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before FREE_OBJECT(C)" << endl;
		}
		FREE_OBJECT(C);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after FREE_OBJECT(C)" << endl;
		}
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before FREE_OBJECT(CO)" << endl;
		}
		FREE_OBJECT(CO);
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"after FREE_OBJECT(CO)" << endl;
		}
#endif

	}
	if (Descr->f_report) {
		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"before classification_report" << endl;
		}
		Combo->Classification_CO->classification_report(
				Descr->Classification_of_objects_report_options,
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
		draw_incidence_matrices(
				Descr->draw_incidence_matrices_prefix,
				Combo->IS,
				verbose_level);

	}
	else if (Descr->f_test_distinguishing_property) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_test_distinguishing_property" << endl;
		}

		graph_theory::colored_graph *CG;


		CG = Get_object_of_type_graph(Descr->test_distinguishing_property_graph);

		int input_idx;
		int *F_distinguishing;

		F_distinguishing = NEW_int(Combo->IS->Objects.size());


		for (input_idx = 0; input_idx < Combo->IS->Objects.size(); input_idx++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[input_idx];

			F_distinguishing[input_idx] = CG->test_distinguishing_property(
					OwCF->set, OwCF->sz, verbose_level);
		}

		data_structures::tally T;

		T.init(F_distinguishing, Combo->IS->Objects.size(), false, 0);
		cout << "classification : ";
		T.print_first(true /* f_backwards*/);
		cout << endl;

		cout << "distinguishing sets are:";
		for (input_idx = 0; input_idx < Combo->IS->Objects.size(); input_idx++) {
			if (F_distinguishing[input_idx]) {
				cout << input_idx << ", ";
			}
		}
		cout << endl;

		cout << "distinguishing sets are:";
		for (input_idx = 0; input_idx < Combo->IS->Objects.size(); input_idx++) {
			if (!F_distinguishing[input_idx]) {
				continue;
			}

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[input_idx];

			OwCF->print(cout);

		}
		cout << endl;


		FREE_int(F_distinguishing);

	}
	else if (Descr->f_compute_frequency) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_compute_frequency" << endl;
		}

		graph_theory::colored_graph *CG;


		CG = Get_object_of_type_graph(Descr->compute_frequency_graph);

		int input_idx;
		//int N;
		int *code = NULL;
		//int sz = 0;


		code = NEW_int(CG->nb_points);

		for (input_idx = 0; input_idx < Combo->IS->Objects.size(); input_idx++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[input_idx];

#if 0
			if (input_idx == 0) {

				sz = OwCF->sz;
				N = 1 << OwCF->sz;
				frequency = NEW_int(N);
			}
			else {
				if (OwCF->sz != sz) {
					cout << "the size of the sets must be constant" << endl;
					exit(1);
				}
			}
#endif

			CG->all_distinguishing_codes(
					OwCF->set, OwCF->sz, code, verbose_level);
			//CG->distinguishing_code_frequency(
			//		OwCF->set, OwCF->sz, frequency, N, verbose_level);

			data_structures::tally T;

			T.init(code, CG->nb_points, false, 0);
			cout << "frequency tally : ";
			T.print_first(true /* f_backwards*/);
			cout << endl;
			T.print_types();

		}


		FREE_int(code);


	}
	if (Descr->f_ideal) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo f_ideal" << endl;
		}

		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->ideal_ring_label);


		int input_idx;


		for (input_idx = 0; input_idx < Combo->IS->Objects.size(); input_idx++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[input_idx];

			HPD->explore_vanishing_ideal(OwCF->set, OwCF->sz, verbose_level);

		}

	}
	else if (Descr->f_save_as) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_save_as " << Descr->save_as_fname << endl;
		}

		do_save(Descr->save_as_fname,
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
		do_save(
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

		unpack_from_restricted_action(
				Descr->unpack_from_restricted_action_prefix,
				Descr->unpack_from_restricted_action_group_label,
				Combo->IS,
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

		line_covering_type(
				Descr->line_covering_type_prefix,
				Descr->line_covering_type_projective_space,
				Descr->line_covering_type_lines,
				Combo->IS,
				verbose_level);

	}

	else if (Descr->f_line_type) {

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity_combo "
					"f_line_type" << endl;
		}

		exit(1);
#if 0
		line_type(
					Descr->line_type_prefix,
					Descr->line_type_projective_space_label,
					IS,
					verbose_level);
#endif


	}



}


void combinatorial_object_activity::do_save(
		std::string &save_as_fname,
		int f_extract,
		long int *extract_idx_set, int extract_size,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::do_save" << endl;
	}
	int input_idx;
	int sz;
	int N;

	N = Combo->IS->Objects.size();

	geometry::object_with_canonical_form *OwCF;

	OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[0];

	//OwCF->set;
	sz = OwCF->sz;

	for (input_idx = 0; input_idx < N; input_idx++) {

		if (false) {
			cout << "combinatorial_object_activity::perform_activity_IS "
					"input_idx = " << input_idx
					<< " / " << Combo->IS->Objects.size() << endl;
		}

		geometry::object_with_canonical_form *OwCF;

		OwCF = (geometry::object_with_canonical_form *)
				Combo->IS->Objects[input_idx];

		//OwCF->set;
		if (OwCF->sz != sz) {
			cout << "the objects have different sizes, cannot save" << endl;
			exit(1);
		}


	}

	long int *Sets;

	Sets = NEW_lint(N * sz);

	for (input_idx = 0; input_idx < N; input_idx++) {
		geometry::object_with_canonical_form *OwCF;

		OwCF = (geometry::object_with_canonical_form *) Combo->IS->Objects[input_idx];

		Lint_vec_copy(OwCF->set, Sets + input_idx * sz, sz);
	}

	cout << "The combined number of objects is " << N << endl;


	if (f_extract) {
		if (f_v) {
			cout << "extracting subset of size " << extract_size << endl;
		}
		long int *Sets2;
		int h, i;

		Sets2 = NEW_lint(extract_size * sz);
		for (h = 0; h < extract_size; h++) {
			i = extract_idx_set[h];
			Lint_vec_copy(Sets + i * sz, Sets2 + h * sz, sz);
		}
		FREE_lint(Sets);
		Sets = Sets2;
		if (f_v) {
			cout << "number of sets is reduced from "
					<< N << " to " << extract_size << endl;
		}
		N = extract_size;
	}
	else {

	}
	orbiter_kernel_system::file_io Fio;

	string fname_out;

	fname_out.assign(save_as_fname);

	Fio.Csv_file_support->lint_matrix_write_csv(
			fname_out, Sets, N, sz);

	cout << "Written file " << fname_out
			<< " of size " << Fio.file_size(fname_out) << endl;

	if (f_v) {
		cout << "combinatorial_object_activity::do_save done" << endl;
	}
}

#if 0
void combinatorial_object_activity::post_process_classification(
		combinatorics::classification_of_objects *CO,
		object_with_properties *&OwP,
		int f_projective_space,
		projective_geometry::projective_space_with_action *PA,
		std::string &prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::post_process_classification" << endl;
	}


	OwP = NEW_OBJECTS(object_with_properties, CO->nb_orbits);

	int iso_type;


	for (iso_type = 0; iso_type < CO->nb_orbits; iso_type++) {

		if (f_v) {
			cout << "combinatorial_object_activity::post_process_classification "
					"iso_type = " << iso_type << " / " << CO->nb_orbits << endl;
			cout << "NO=" << endl;
			//CO->NO_transversal[iso_type]->print();
		}

		std::string label;

		label = prefix + "_object" + std::to_string(iso_type);

		if (f_v) {
			cout << "combinatorial_object_activity::post_process_classification "
					"before OwP[iso_type].init" << endl;
		}
		OwP[iso_type].init(
				CO->OWCF_transversal[iso_type],
				CO->NO_transversal[iso_type],
				f_projective_space, PA,
				CO->Descr->max_TDO_depth,
				label,
				verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::post_process_classification "
					"after OwP[iso_type].init" << endl;
		}


	}

	if (f_v) {
		cout << "combinatorial_object_activity::post_process_classification done" << endl;
	}
}
#endif

#if 0
void combinatorial_object_activity::classification_report(
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report" << endl;
	}

	combinatorics::classification_of_objects_report_options *Report_options;

	if (CO->Descr->f_classification_prefix == false) {
		cout << "please use option -classification_prefix <prefix> to set the "
				"prefix for the output file" << endl;
		exit(1);
	}

	Report_options = Descr->Classification_of_objects_report_options;

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report "
				"before latex_report" << endl;
	}
	latex_report(Report_options,
			CO,
			OwP,
			verbose_level);

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report "
				"after latex_report" << endl;
	}

}

void combinatorial_object_activity::latex_report(
		combinatorics::classification_of_objects_report_options
			*Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::latex_report" << endl;
	}



	string fname;

	fname = Report_options->prefix + "_classification.tex";

	if (f_v) {
		cout << "combinatorial_object_activity::classification_report "
				"before latex_report" << endl;
	}



	if (f_v) {
		cout << "combinatorial_object_activity::latex_report, "
				"CB->nb_types=" << CO->CB->nb_types << endl;
	}
	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;

		L.head_easy(ost);


		CO->report_summary_of_orbits(ost, verbose_level);


		ost << "Ago : ";
		CO->T_Ago->print_file_tex(ost, false /* f_backwards*/);
		ost << "\\\\" << endl;

		if (f_v) {
			cout << "combinatorial_object_activity::latex_report before loop" << endl;
		}

		report_all_isomorphism_types(
				ost, Report_options, CO, OwP,
				verbose_level);

		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	//FREE_int(perm);
	//FREE_int(v);
	if (f_v) {
		cout << "combinatorial_object_activity::latex_report done" << endl;
	}
}

void combinatorial_object_activity::report_all_isomorphism_types(
		std::ostream &ost,
		combinatorics::classification_of_objects_report_options
			*Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_all_isomorphism_types" << endl;
	}
	int i;

	l1_interfaces::latex_interface L;

	for (i = 0; i < CO->CB->nb_types; i++) {

		ost << "\\section*{Isomorphism type " << i << " / " << CO->CB->nb_types << "}" << endl;
		ost << "Isomorphism type " << i << " / " << CO->CB->nb_types
			//<<  " stored at " << j
			<< " is original object "
			<< CO->CB->Type_rep[i] << " and appears "
			<< CO->CB->Type_mult[i] << " times: \\\\" << endl;

		{
			data_structures::sorting Sorting;
			int *Input_objects;
			int nb_input_objects;
			CO->CB->C_type_of->get_class_by_value(Input_objects,
					nb_input_objects, i, 0 /*verbose_level */);
			Sorting.int_vec_heapsort(Input_objects, nb_input_objects);

			ost << "This isomorphism type appears " << nb_input_objects
					<< " times, namely for the following "
					<< nb_input_objects << " input objects: " << endl;
			if (nb_input_objects < 10) {
				ost << "$" << endl;
				L.int_set_print_tex(
						ost, Input_objects, nb_input_objects);
				ost << "$\\\\" << endl;
			}
			else {
				ost << "Too big to print. \\\\" << endl;
#if 0
				fp << "$$" << endl;
				L.int_vec_print_as_matrix(fp, Input_objects,
					nb_input_objects, 10 /* width */, true /* f_tex */);
				fp << "$$" << endl;
#endif
			}

			FREE_int(Input_objects);
		}

		if (f_v) {
			cout << "combinatorial_object_activity::report_all_isomorphism_types "
					"before report_isomorphism_type" << endl;
		}
		report_isomorphism_type(
				ost, Report_options, CO, OwP, i, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::report_all_isomorphism_types "
					"after report_isomorphism_type" << endl;
		}


	} // next i
	if (f_v) {
		cout << "combinatorial_object_activity::report_all_isomorphism_types done" << endl;
	}

}


void combinatorial_object_activity::report_isomorphism_type(
		std::ostream &ost,
		combinatorics::classification_of_objects_report_options
			*Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type "
				"i=" << i << endl;
	}
	int j;
	l1_interfaces::latex_interface L;

	//j = CB->perm[i];
	//j = CB->Type_rep[i];
	j = i;

	cout << "###################################################"
			"#############################" << endl;
	cout << "Orbit " << i << " / " << CO->CB->nb_types
			<< " is canonical form no " << j
			<< ", original object no " << CO->CB->Type_rep[i]
			<< ", frequency " << CO->CB->Type_mult[i]
			<< " : " << endl;


	{
		int *Input_objects;
		int nb_input_objects;
		CO->CB->C_type_of->get_class_by_value(Input_objects,
			nb_input_objects, j, 0 /*verbose_level */);

		cout << "This isomorphism type appears " << nb_input_objects
				<< " times, namely for the following "
						"input objects:" << endl;
		if (nb_input_objects < 10) {
			L.int_vec_print_as_matrix(cout, Input_objects,
					nb_input_objects, 10 /* width */,
					false /* f_tex */);
		}
		else {
			cout << "too many to print" << endl;
		}

		FREE_int(Input_objects);
	}



	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type "
				"i=" << i << " before report_object" << endl;
	}
	report_object(ost,
			Report_options,
			CO,
			OwP,
			i /* object_idx */,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type "
				"i=" << i << " after report_object" << endl;
	}




	if (f_v) {
		cout << "combinatorial_object_activity::report_isomorphism_type "
				"i=" << i << " done" << endl;
	}
}

void combinatorial_object_activity::report_object(
		std::ostream &ost,
		combinatorics::classification_of_objects_report_options
			*Report_options,
		combinatorics::classification_of_objects *CO,
		object_with_properties *OwP,
		int object_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combinatorial_object_activity::report_object "
				"object_idx=" << object_idx << endl;
	}


	geometry::object_with_canonical_form *OwCF = CO->OWCF_transversal[object_idx];

	if (f_v) {
		cout << "combinatorial_object_activity::report_object "
				"before OwCF->print_tex_detailed" << endl;
	}
	OwCF->print_tex_detailed(ost,
			Report_options->f_show_incidence_matrices,
			verbose_level);
	if (f_v) {
		cout << "combinatorial_object_activity::report_object "
				"after OwCF->print_tex_detailed" << endl;
	}

	if (false /*CO->f_projective_space*/) {

#if 0
		object_in_projective_space_with_action *OiPA;

		OiPA = (object_in_projective_space_with_action *)
				CB->Type_extra_data[object_idx];

		OiPA->report(fp, PA, max_TDO_depth, verbose_level);
#endif

	}
	else {
		if (f_v) {
			cout << "combinatorial_object_activity::report_object "
					"before OwP[object_idx].latex_report" << endl;
		}
		OwP[object_idx].latex_report(ost,
				Report_options, verbose_level);
		if (f_v) {
			cout << "combinatorial_object_activity::report_object "
					"after OwP[object_idx].latex_report" << endl;
		}
	}



}
#endif

void combinatorial_object_activity::draw_incidence_matrices(
		std::string &prefix,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices" << endl;
	}



	string fname;

	fname = prefix + "_incma.tex";

	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices "
				"before latex_report" << endl;
	}


	{
		ofstream ost(fname);
		l1_interfaces::latex_interface L;

		L.head_easy(ost);


		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::draw_incidence_matrices "
					"before loop" << endl;
		}

		int i;

		ost << "\\noindent" << endl;

		for (i = 0; i < N; i++) {

			if (f_v) {
				cout << "combinatorial_object_activity::draw_incidence_matrices "
						"object " << i << " / " << N << endl;
			}
			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			combinatorics::encoded_combinatorial_object *Enc;

			if (f_v) {
				cout << "combinatorial_object_activity::draw_incidence_matrices "
						"before OwCF->encode_incma" << endl;
			}
			OwCF->encode_incma(Enc, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_activity::draw_incidence_matrices "
						"after OwCF->encode_incma" << endl;
			}

			//Enc->latex_set_system_by_columns(ost, verbose_level);

			//Enc->latex_set_system_by_rows(ost, verbose_level);

			if (f_v) {
				cout << "combinatorial_object_activity::draw_incidence_matrices "
						"before OwCF->latex_incma" << endl;
			}
			Enc->latex_incma(ost, verbose_level);
			if (f_v) {
				cout << "combinatorial_object_activity::draw_incidence_matrices "
						"after OwCF->latex_incma" << endl;
			}

			FREE_OBJECT(Enc);


		}



		L.foot(ost);
	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_activity::draw_incidence_matrices done" << endl;
	}
}

void combinatorial_object_activity::unpack_from_restricted_action(
		std::string &prefix,
		std::string &group_label,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action" << endl;
	}


	apps_algebra::any_group *G;

	G = Get_object_of_type_any_group(group_label);

#if 0
	groups::linear_group *LG;

	LG = G->LG;


	if (!G->f_linear_group) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action "
				"must be a linear group" << endl;
		exit(1);
	}
#endif

	if (G->A->type_G != action_by_restriction_t) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action "
				"must be a restricted action" << endl;
		exit(1);
	}
	induced_actions::action_by_restriction *ABR;
	ABR = G->A->G.ABR;


	string fname;

	fname = prefix + "_unpacked.txt";

	if (f_v) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action "
				"before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::unpack_from_restricted_action "
					"before loop" << endl;
		}

		int i, h;
		long int a, b;


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			//encoded_combinatorial_object *Enc;

			//OwCF->encode_incma(Enc, verbose_level);

			for (h = 0; h < OwCF->sz; h++) {

				a = OwCF->set[h];
				b = ABR->original_point(a);
				OwCF->set[h] = b;
			}

			//Enc->latex_incma(ost, verbose_level);

			//FREE_OBJECT(Enc);

			ost << OwCF->sz;
			for (h = 0; h < OwCF->sz; h++) {
				ost << " " << OwCF->set[h];
			}
			ost << endl;

		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_activity::unpack_from_restricted_action done" << endl;
	}
}


void combinatorial_object_activity::line_covering_type(
		std::string &prefix,
		std::string &projective_space_label,
		std::string &lines,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::line_covering_type" << endl;
	}

	projective_geometry::projective_space_with_action *PA;

	PA = Get_projective_space(projective_space_label);

	geometry::projective_space *P;

	P = PA->P;

	long int *the_lines;
	int nb_lines;

	Get_lint_vector_from_label(lines, the_lines, nb_lines, verbose_level);

	string fname;

	fname = prefix + "_line_covering_type.txt";

	if (f_v) {
		cout << "combinatorial_object_activity::line_covering_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::line_covering_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(nb_lines);


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			P->Subspaces->line_intersection_type_basic_given_a_set_of_lines(
					the_lines, nb_lines,
					OwCF->set, OwCF->sz, type, 0 /*verbose_level */);

			ost << OwCF->sz;
			for (h = 0; h < nb_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_activity::line_covering_type done" << endl;
	}
}

void combinatorial_object_activity::line_type(
		std::string &prefix,
		std::string &projective_space_label,
		data_structures::data_input_stream *IS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;
	l1_interfaces::latex_interface L;

	if (f_v) {
		cout << "combinatorial_object_activity::line_type" << endl;
	}

	projective_geometry::projective_space_with_action *PA;

	PA = Get_projective_space(projective_space_label);

	geometry::projective_space *P;

	P = PA->P;

	string fname;

	fname = prefix + "_line_type.txt";

	if (f_v) {
		cout << "combinatorial_object_activity::line_type before latex_report" << endl;
	}


	{

		ofstream ost(fname);

		int N;

		N = IS->Objects.size();



		if (f_v) {
			cout << "combinatorial_object_activity::line_type before loop" << endl;
		}

		int i, h;

		int *type;

		type = NEW_int(P->Subspaces->N_lines);


		for (i = 0; i < N; i++) {

			geometry::object_with_canonical_form *OwCF;

			OwCF = (geometry::object_with_canonical_form *) IS->Objects[i];


			P->Subspaces->line_intersection_type(
					OwCF->set, OwCF->sz, type, 0 /* verbose_level */);
				// type[N_lines]

			ost << OwCF->sz;
			for (h = 0; h < P->Subspaces->N_lines; h++) {
				ost << " " << type[h];
			}
			ost << endl;

#if 1
			data_structures::tally T;

			T.init(type, P->Subspaces->N_lines, false, 0);

			if (f_v) {
				cout << "combinatorial_object_activity::perform_activity_GOC line type:" << endl;
				T.print(true /* f_backwards*/);
				cout << endl;
			}

			std::string fname_line_type;


			fname_line_type = prefix + "_line_type_set_partition_" + std::to_string(i) + ".csv";

			T.save_classes_individually(fname_line_type);
			if (true) {
				cout << "Written file " << fname_line_type << " of size "
						<< Fio.file_size(fname_line_type) << endl;
			}

#endif



		}

		ost << -1 << endl;


	}

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "combinatorial_object_activity::line_type done" << endl;
	}
}





}}}




