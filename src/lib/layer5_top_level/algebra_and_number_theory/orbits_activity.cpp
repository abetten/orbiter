/*
 * orbits_activity.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbits_activity::orbits_activity()
{
	Descr = NULL;

	OC = NULL;


}

orbits_activity::~orbits_activity()
{

}

void orbits_activity::init(
		orbits_activity_description *Descr,
		apps_algebra::orbits_create *OC,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::init" << endl;
	}

	orbits_activity::Descr = Descr;
	orbits_activity::OC = OC;

	if (f_v) {
		cout << "orbits_activity::init done" << endl;
	}
}


void orbits_activity::perform_activity(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::perform_activity" << endl;
	}

	if (Descr->f_report) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_report" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_report" << endl;
		}
		do_report(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_report" << endl;
		}
	}
	if (Descr->f_export_something) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_export_something" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_export" << endl;
		}
		do_export(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_export" << endl;
		}

	}
	else if (Descr->f_export_trees) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_export_trees" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_export_trees" << endl;
		}
		do_export_trees(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_export_trees" << endl;
		}
	}
	else if (Descr->f_export_source_code) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_export_source_code" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_export_source_code" << endl;
		}
		do_export_source_code(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_export_source_code" << endl;
		}
	}
	else if (Descr->f_export_levels) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_export_levels" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_export_levels" << endl;
		}
		do_export_levels(Descr->export_levels_orbit_idx, verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_export_levels" << endl;
		}
	}
	else if (Descr->f_draw_tree) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_draw_tree" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_draw_tree" << endl;
		}
		do_draw_tree(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_draw_tree" << endl;
		}
	}
	else if (Descr->f_stabilizer) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_stabilizer" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_stabilizer" << endl;
		}
		do_stabilizer(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_stabilizer" << endl;
		}
	}
	else if (Descr->f_stabilizer_of_orbit_rep) {
		if (f_v) {
			cout << "orbits_activity::perform_activity f_stabilizer_of_orbit_rep" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_stabilizer_of_orbit_rep" << endl;
		}
		do_stabilizer_of_orbit_rep(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_stabilizer_of_orbit_rep" << endl;
		}
	}
	else if (Descr->f_Kramer_Mesner_matrix) {

		if (f_v) {
			cout << "orbits_activity::perform_activity f_Kramer_Mesner_matrix" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_Kramer_Mesner_matrix" << endl;
		}
		do_Kramer_Mesner_matrix(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_Kramer_Mesner_matrix" << endl;
		}
	}
	else if (Descr->f_recognize) {

		if (f_v) {
			cout << "orbits_activity::perform_activity f_recognize" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_recognize" << endl;
		}
		do_recognize(verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_recognize" << endl;
		}

	}
	else if (Descr->f_transporter) {

		if (f_v) {
			cout << "orbits_activity::perform_activity f_transporter" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::perform_activity before do_recognize" << endl;
		}
		do_transporter(Descr->transporter_label_of_set, verbose_level);
		if (f_v) {
			cout << "orbits_activity::perform_activity after do_recognize" << endl;
		}

	}




	if (f_v) {
		cout << "orbits_activity::perform_activity done" << endl;
	}
}

void orbits_activity::do_report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_report" << endl;
	}


	if (OC->f_has_Orb) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_Orb" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->Orb->create_latex_report" << endl;
		}
		OC->Orb->create_latex_report(verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->Orb->create_latex_report" << endl;
		}
	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_On_polynomials" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->On_polynomials->report" << endl;
		}
		OC->On_polynomials->report(verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->On_polynomials->report" << endl;
		}


	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_On_subsets" << endl;
		}

		poset_classification::poset_classification_report_options *report_options;

		if (Descr->f_report_options) {
			report_options = Descr->report_options;
		}
		else {
			report_options = NEW_OBJECT(poset_classification::poset_classification_report_options);
		}
		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->On_subsets->report" << endl;
		}
		OC->On_subsets->report(
				report_options,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->On_subsets->report" << endl;
		}
		if (!Descr->f_report_options) {
			FREE_OBJECT(report_options);
		}


	}

	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->On_Subspaces->orbits_on_subspaces_PC->report" << endl;
		}

		poset_classification::poset_classification_report_options *report_options;

		if (Descr->f_report_options) {
			report_options = Descr->report_options;
		}
		else {
			report_options = NEW_OBJECT(poset_classification::poset_classification_report_options);
		}

		OC->On_Subspaces->orbits_on_subspaces_PC->report(
				report_options,
				verbose_level);
		if (!Descr->f_report_options) {
			FREE_OBJECT(report_options);
		}
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->On_Subspaces->orbits_on_subspaces_PC->report" << endl;
		}

	}

	else if (OC->f_has_classification_by_canonical_form) {

		if (f_v) {
			cout << "orbits_activity::do_report "
					"f_has_classification_by_canonical_form" << endl;
		}

		poset_classification::poset_classification_report_options *report_options;

		if (Descr->f_report_options) {
			report_options = Descr->report_options;
		}
		else {
			report_options = NEW_OBJECT(poset_classification::poset_classification_report_options);
		}

		OC->Canonical_form_classifier->Output->report(
				report_options,
				verbose_level);

		if (!Descr->f_report_options) {
			FREE_OBJECT(report_options);
		}
		if (f_v) {
			cout << "orbits_activity::do_report "
					"f_has_classification_by_canonical_form done" << endl;
		}

	}


	else {
		cout << "orbits_activity::do_report no suitable data structure" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_report done" << endl;
	}

}

void orbits_activity::do_export(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export" << endl;
	}


	if (OC->f_has_Orb) {

		string fname;

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->Orb->export_something" << endl;
		}

		OC->Orb->export_something(
				Descr->export_something_what,
				Descr->export_something_data1, fname, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->Orb->export_something" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_export "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

	}

	else if (OC->f_has_On_polynomials) {

		string fname;

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->On_polynomials->export_something" << endl;
		}

		OC->On_polynomials->export_something(
				Descr->export_something_what,
				Descr->export_something_data1, fname, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->On_polynomials->export_something" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_export "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


	}

	else {
		cout << "orbits_activity::do_export no suitable data structure" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_export done" << endl;
	}

}

void orbits_activity::do_export_trees(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_trees" << endl;
	}

	if (OC->f_has_Orb) {
		string fname_tree_mask;
		int orbit_idx;

		fname_tree_mask = "orbit_" + OC->Group->A->label + "_%d.layered_graph";

		for (orbit_idx = 0; orbit_idx < OC->Orb->Sch->nb_orbits; orbit_idx++) {

			cout << "orbit " << orbit_idx << " / " <<  OC->Orb->Sch->nb_orbits
					<< " before Sch->export_tree_as_layered_graph" << endl;

			OC->Orb->Sch->export_tree_as_layered_graph(orbit_idx,
					fname_tree_mask,
					verbose_level - 1);
		}

	}
	else {
		cout << "orbits_activity::do_export_trees no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_export_trees done" << endl;
	}

}


void orbits_activity::do_export_source_code(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_source_code" << endl;
	}

#if 0
	if (OC->f_has_Orb) {
		string fname_tree_mask;
		int orbit_idx;

		fname_tree_mask = "orbit_" + OC->Group->A->label + "_%d.layered_graph";

		for (orbit_idx = 0; orbit_idx < OC->Orb->Sch->nb_orbits; orbit_idx++) {

			cout << "orbit " << orbit_idx << " / " <<  OC->Orb->Sch->nb_orbits
					<< " before Sch->export_tree_as_layered_graph" << endl;

			OC->Orb->Sch->export_tree_as_layered_graph(orbit_idx,
					fname_tree_mask,
					verbose_level - 1);
		}

	}
#endif
	if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_classification_by_canonical_form" << endl;
		}


		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"before generate_source_code" << endl;
		}

		OC->Canonical_form_classifier->Output->generate_source_code(
				OC->Descr->Canonical_form_classifier_description->fname_base_out,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"after generate_source_code" << endl;
		}


	}
	else {
		cout << "orbits_activity::do_export_source_code no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_export_source_code done" << endl;
	}

}


void orbits_activity::do_export_levels(int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_levels" << endl;
	}

	if (OC->f_has_Orb) {
		string fname_tree_mask;
		string fname;

		fname_tree_mask = "orbit_" + OC->Group->A->label + "_orbit_" + std::to_string(orbit_idx);
		data_structures::set_of_sets *SoS;

		OC->Orb->Sch->get_orbit_by_levels(
				orbit_idx,
				SoS,
				verbose_level);

		int i;

		orbiter_kernel_system::file_io Fio;

		for (i = 0; i < SoS->nb_sets; i++) {
			fname = fname_tree_mask + "_level_" + std::to_string(i) + ".csv";

			string label;

			label = "lvl" + std::to_string(i);

			Fio.Csv_file_support->lint_vec_write_csv(
					SoS->Sets[i], SoS->Set_size[i],
					fname, label);

			if (f_v) {
				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}
		}

		int j;
		int *v;

		if (f_v) {
			cout << "low_level_point_size = " << OC->Group->A->low_level_point_size << endl;
		}


		v = NEW_int(OC->Group->A->low_level_point_size);
		i = SoS->nb_sets - 1;
		for (j = 0; j < SoS->Set_size[i]; j++) {
			OC->Group->A->Group_element->unrank_point(SoS->Sets[i][j], v);

			std::vector<int> path;
			std::vector<int> labels;
			int *Labels;
			int h;

			Labels = NEW_int(i);


			OC->Orb->Sch->get_path_and_labels(
					path, labels,
					SoS->Sets[i][j], verbose_level);



			if (path.size() != i) {
				cout << "orbits_activity::do_export_levels path.size() != i" << endl;
				cout << "path.size()=" << path.size() << endl;
				cout << "i=" << i << endl;
				exit(1);
			}

			for (h = 0; h < i; h++) {
				Labels[h] = labels[h];
			}

			cout << j << " : " << SoS->Sets[i][j] << " : ";
			Int_vec_print(cout, Labels, i);
			cout << " : ";
			Int_vec_print(cout, v, OC->Group->A->low_level_point_size);
			cout << endl;

			FREE_int(Labels);

		}
	}
	else {
		cout << "orbits_activity::do_export_levels no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_export_levels done" << endl;
	}

}


void orbits_activity::do_draw_tree(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_draw_tree" << endl;
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree tree index = " << Descr->draw_tree_idx << endl;
	}


	if (OC->f_has_Orb) {
		string fname;

		fname = OC->Orb->prefix + "_orbit_" + std::to_string(Descr->draw_tree_idx) + "_tree";

		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"before OC->Orb->Sch->draw_tree" << endl;
		}
		OC->Orb->Sch->draw_tree(fname,
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->draw_tree_idx,
				false /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"after OC->Orb->Sch->draw_tree" << endl;
		}

		std::vector<int> Orb;

		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"before OC->Orb->Sch->get_orbit_in_order" << endl;
		}
		OC->Orb->Sch->get_orbit_in_order(Orb,
				Descr->draw_tree_idx, verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"after OC->Orb->Sch->get_orbit_in_order" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		string fname_full;

		fname_full = fname + "_orbit_elements.csv";


		Fio.Csv_file_support->vector_write_csv(
				fname_full, Orb);
		if (f_v) {
			cout << "Written file " << fname_full << " of size "
					<< Fio.file_size(fname_full) << endl;
		}

	}
	else if (OC->f_has_On_polynomials) {

		string fname;


		fname = OC->prefix + "_orbit_" + std::to_string(Descr->draw_tree_idx) + "_tree";

		OC->On_polynomials->Sch->draw_tree(fname,
				orbiter_kernel_system::Orbiter->draw_options,
				Descr->draw_tree_idx,
				false /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);


	}
	else {
		cout << "orbits_activity::do_draw_tree no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree done" << endl;
	}
}

void orbits_activity::do_stabilizer(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer" << endl;
	}

	if (OC->f_has_Orb) {

		groups::strong_generators *Stab;

		if (f_v) {
			cout << "orbits_activity::do_stabilizer before Orb->stabilizer_of" << endl;
		}
		//OC->Orb->stabilizer_of(Descr->stabilizer_orbit_idx, verbose_level);

		OC->Orb->stabilizer_any_point(Descr->stabilizer_point,
				Stab, verbose_level);


		if (f_v) {
			cout << "orbits_activity::do_stabilizer after Orb->stabilizer_of" << endl;
		}


		std::string gens_str;
		ring_theory::longinteger_object stab_go;


		gens_str = Stab->stringify_gens_data(0 /*verbose_level*/);
		Stab->group_order(stab_go);
		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"The stabilizer has order " << stab_go << endl;
			cout << "orbits_activity::do_stabilizer "
					"Number of generators " << Stab->gens->len << endl;
			cout << "orbits_activity::do_stabilizer "
					"Generators for the stabilizer in coded form: " << endl;
			cout << gens_str << endl;
		}

		string fname_stab;
		string label_stab;



		fname_stab = OC->prefix + "_stab_pt_" + std::to_string(Descr->stabilizer_point) + ".makefile";

		label_stab = OC->prefix + "_stab_pt_" + std::to_string(Descr->stabilizer_point);

		Stab->report_group(label_stab, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"exporting stabilizer orbit representative "
					"of point " << Descr->stabilizer_point << " to " << fname_stab << endl;
		}
		Stab->export_to_orbiter_as_bsgs(
				Stab->A,
				fname_stab, label_stab, label_stab,
				verbose_level);

		FREE_OBJECT(Stab);

	}
	else {
		cout << "orbits_activity::do_stabilizer no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer done" << endl;
	}

}

void orbits_activity::do_stabilizer_of_orbit_rep(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep" << endl;
	}

	if (OC->f_has_Orb) {
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before OC->Orb->stabilizer_of" << endl;
		}
		OC->Orb->stabilizer_of(Descr->stabilizer_of_orbit_rep_orbit_idx, verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after OC->Orb->stabilizer_of" << endl;
		}
	}
	else {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep "
				"no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep done" << endl;
	}

}

void orbits_activity::do_Kramer_Mesner_matrix(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_Kramer_Mesner_matrix" << endl;
	}







	if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"f_has_On_subsets" << endl;
		}


		poset_classification::poset_classification_activity_description Activity_descr;
		poset_classification::poset_classification_activity Activity;


		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before Activity.init" << endl;
		}
		Activity.init(
				&Activity_descr,
				OC->On_subsets,
				Descr->Kramer_Mesner_k /* actual_size */,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after Activity.init" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before Activity.compute_Kramer_Mesner_matrix" << endl;
		}
		Activity.compute_Kramer_Mesner_matrix(
				Descr->Kramer_Mesner_t,
				Descr->Kramer_Mesner_k,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after Activity.compute_Kramer_Mesner_matrix" << endl;
		}

	}

	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"f_has_On_Subspaces" << endl;
		}

		poset_classification::poset_classification_activity_description Activity_descr;
		poset_classification::poset_classification_activity Activity;


		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before Activity.init" << endl;
		}
		Activity.init(
				&Activity_descr,
				OC->On_Subspaces->orbits_on_subspaces_PC,
				Descr->Kramer_Mesner_k /* actual_size */,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after Activity.init" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before Activity.compute_Kramer_Mesner_matrix" << endl;
		}
		Activity.compute_Kramer_Mesner_matrix(
				Descr->Kramer_Mesner_t,
				Descr->Kramer_Mesner_k,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after Activity.compute_Kramer_Mesner_matrix" << endl;
		}

	}

	else {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep "
				"unknown type of poset" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_Kramer_Mesner_matrix done" << endl;
	}
}

void orbits_activity::do_recognize(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_recognize" << endl;
	}

	if (OC->f_has_On_subsets) {

		int h;

		for (h = 0; h < Descr->recognize.size(); h++) {
			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"h=" << h << " / " << Descr->recognize.size() << endl;
			}
			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"before OC->On_subsets->recognize" << endl;
			}

			OC->On_subsets->recognize(
					Descr->recognize[h],
					h, Descr->recognize.size(),
					verbose_level);
			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"after OC->On_subsets->recognize" << endl;
			}
		}



	}

	else if (OC->f_has_On_Subspaces) {

		int h;

		for (h = 0; h < Descr->recognize.size(); h++) {
			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"h=" << h << " / " << Descr->recognize.size() << endl;
			}
			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"before OC->On_Subspaces->orbits_on_subspaces_PC->recognize" << endl;
			}

			OC->On_Subspaces->orbits_on_subspaces_PC->recognize(
					Descr->recognize[h],
					h, Descr->recognize.size(),
					verbose_level);

			if (f_v) {
				cout << "orbits_activity::do_recognize "
						"after OC->On_Subspaces->orbits_on_subspaces_PC->recognize" << endl;
			}

		}

	}

	else {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep no suitable data structure" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_recognize done" << endl;
	}
}


void orbits_activity::do_transporter(
		std::string &label_of_set, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_transporter" << endl;
	}

	long int *the_set;
	int set_size;

	Get_lint_vector_from_label(label_of_set, the_set, set_size, 0 /* verbose_level */);


	if (OC->f_has_Of_One_polynomial) {

		string fname;

		if (f_v) {
			cout << "orbits_activity::do_transporter "
					"before OC->On_polynomials->export_something" << endl;
		}

		int *transporter;
		int i;
		int Nb;

		actions::action *A;

		A = OC->Of_One_polynomial->A;

		Nb = set_size;

		transporter = NEW_int(A->elt_size_in_int);

		for (i = 0; i < Nb; i++) {

			if (f_v) {
				cout << "orbits_on_polynomials::orbit_of_one_polynomial "
						"before OC->Of_One_polynomial->Orb->get_transporter" << endl;

			}
			OC->Of_One_polynomial->Orb->get_transporter(
					the_set[i],
					transporter, verbose_level);

			if (f_v) {
				cout << "orbits_on_polynomials::orbit_of_one_polynomial "
						"after OC->Of_One_polynomial->Orb->get_transporter" << endl;

			}

			cout << "i=" << i << " / " << Nb << " Idx[i] = " << the_set[i] << endl;
			cout << "transporter=" << endl;
			A->Group_element->element_print(transporter, cout);
			cout << endl;
		}

		FREE_int(transporter);

		if (f_v) {
			cout << "orbits_activity::do_transporter "
					"after OC->On_polynomials->export_something" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_transporter "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


	}


	if (f_v) {
		cout << "orbits_activity::do_transporter done" << endl;
	}
}

}}}




