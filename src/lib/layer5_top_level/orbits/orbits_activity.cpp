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
namespace orbits {


orbits_activity::orbits_activity()
{
	Record_birth();
	Descr = NULL;

	OC = NULL;


}

orbits_activity::~orbits_activity()
{
	Record_death();

}

void orbits_activity::init(
		orbits_activity_description *Descr,
		orbits_create *OC,
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


void orbits_activity::perform_activity(
		int verbose_level)
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
	else if (Descr->f_export_something) {
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

		other::graphics::layered_graph_draw_options *Draw_options;

		Draw_options = Get_draw_options(Descr->draw_tree_draw_options);


		if (f_v) {
			cout << "orbits_activity::perform_activity before do_draw_tree" << endl;
		}
		do_draw_tree(Draw_options, verbose_level);
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
	else {
		cout << "orbits_activity::perform_activity "
				"no activity found" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "orbits_activity::perform_activity done" << endl;
	}
}

void orbits_activity::do_report(
		int verbose_level)
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

	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_On_subsets" << endl;
		}

		poset_classification::poset_classification_report_options *report_options;

		if (Descr->f_report_options) {
			report_options = Get_poset_classification_report_options(Descr->report_options_label);
		}
		else {
			cout << "orbits_activity::do_report please use -report_options" << endl;
			exit(1);
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
			report_options = Get_poset_classification_report_options(Descr->report_options_label);
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

	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_On_tensors" << endl;
		}

	}

	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_Cascade" << endl;
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

	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_Of_One_polynomial" << endl;
		}



	}

	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_on_cubic_curves" << endl;
		}



		if (f_v) {
			cout << "orbits_activity::do_report "
					"creating cheat sheet" << endl;
		}
		string fname, title, author, extra_praeamble;
		int q;

		q = OC->CCC->CCA->F->q;
		title = "Cubic Curves in PG$(2," + std::to_string(q) + ")$";
		author = "";
		fname = "Cubic_curves_q" + std::to_string(q) + ".tex";

		{
			ofstream fp(fname);
			other::l1_interfaces::latex_interface L;

			//latex_head_easy(fp);
			L.head(fp,
				false /* f_book */,
				true /* f_title */,
				title, author,
				false /*f_toc */,
				false /* f_landscape */,
				false /* f_12pt */,
				true /*f_enlarged_page */,
				true /* f_pagenumbers*/,
				extra_praeamble /* extra_praeamble */);

			fp << "\\subsection*{" << title << "}" << endl;

			if (f_v) {
				cout << "orbits_activity::do_report "
						"before OC->CCC->report" << endl;
			}
			OC->CCC->report(fp, verbose_level);
			if (f_v) {
				cout << "orbits_activity::do_report "
						"after OC->CCC->report" << endl;
			}

			L.foot(fp);
		}

	}

	else if (OC->f_has_cubic_surfaces) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_cubic_surfaces" << endl;
		}

		poset_classification::poset_classification_report_options *report_options;

		if (!Descr->f_report_options) {
			cout << "orbits_activity::do_report please use -report_options" << endl;
			exit(1);
		}
		report_options = Get_poset_classification_report_options(Descr->report_options_label);


		int f_with_stabilizers = true;

		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->SCW->create_report" << endl;
			cout << "OC->SCW->Surf->n = " << OC->SCW->Surf->n << endl;
		}
		OC->SCW->create_report(
				f_with_stabilizers,
				report_options,
				verbose_level - 1);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->SCW->create_report" << endl;
		}

	}

	else if (OC->f_has_semifields) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_semifields" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::do_report "
					"before OC->Semifields->latex_report" << endl;
		}
		OC->Semifields->latex_report(verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_report "
					"after OC->Semifields->latex_report" << endl;
		}

	}


	else if (OC->f_has_boolean_functions) {

		if (f_v) {
			cout << "orbits_activity::do_report f_has_boolean_functions" << endl;
		}
		OC->BFC->print();

	}

	else if (OC->f_has_classification_by_canonical_form) {

		if (f_v) {
			cout << "orbits_activity::do_report "
					"f_has_classification_by_canonical_form" << endl;
		}

		poset_classification::poset_classification_report_options *Report_options;

		if (!Descr->f_report_options) {
			cout << "orbits_activity::do_report please use -report_options" << endl;
			exit(1);
		}
		Report_options = Get_poset_classification_report_options(Descr->report_options_label);

		//string fname_base;

		//fname_base = OC->label_txt;

		OC->Canonical_form_classifier->Classification_of_varieties_nauty->report(
				//fname_base,
				Report_options,
				verbose_level);

#if 0
		if (!Descr->f_report_options) {
			FREE_OBJECT(report_options);
		}
#endif

		if (f_v) {
			cout << "orbits_activity::do_report "
					"f_has_classification_by_canonical_form done" << endl;
		}

	}


	else {
		cout << "orbits_activity::do_report "
				"no suitable data structure found" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "orbits_activity::do_report done" << endl;
	}

}

void orbits_activity::do_export(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export" << endl;
	}

	string fname;

	if (OC->f_has_Orb) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_Orb" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->Orb->export_something" << endl;
		}

		OC->Orb->export_something(
				Descr->export_something_what,
				Descr->export_something_data1, fname,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->Orb->export_something" << endl;
		}


	}

	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_On_subsets" << endl;
		}

		//poset_classification::poset_classification *On_subsets;

		OC->On_subsets->export_something_worker(
				Descr->export_something_what,
				Descr->export_something_data1, fname,
				verbose_level);

	}

	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_On_Subspaces" << endl;
		}


	}

	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_On_tensors" << endl;
		}


	}

	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_Cascade" << endl;
		}

	}

	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_export f_has_On_polynomials" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->On_polynomials->export_something" << endl;
		}

		OC->On_polynomials->export_something(
				Descr->export_something_what,
				Descr->export_something_data1, fname,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->On_polynomials->export_something" << endl;
		}



	}

	else if (OC->f_has_Of_One_polynomial) {


		if (f_v) {
			cout << "orbits_activity::do_export f_has_Of_One_polynomial" << endl;
		}

		if (f_v) {
			cout << "orbits_activity::do_export "
					"before OC->Of_One_polynomial->export_something" << endl;
		}

		OC->Of_One_polynomial->export_something(
				Descr->export_something_what,
				Descr->export_something_data1, fname,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export "
					"after OC->Of_One_polynomial->export_something" << endl;
		}



	}

	else if (OC->f_has_on_cubic_curves) {


		if (f_v) {
			cout << "orbits_activity::do_export f_has_on_cubic_curves" << endl;
		}

	}

	else if (OC->f_has_classification_by_canonical_form) {


		if (f_v) {
			cout << "orbits_activity::do_export f_has_classification_by_canonical_form" << endl;
		}

	}

	else {
		cout << "orbits_activity::do_export "
				"no suitable data structure found" << endl;
		exit(1);
	}

	if (fname.length()) {

		other::orbiter_kernel_system::file_io Fio;

		if (f_v) {
			cout << "orbits_activity::do_export "
					"Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}

	}



	if (f_v) {
		cout << "orbits_activity::do_export done" << endl;
	}

}

void orbits_activity::do_export_trees(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_trees" << endl;
	}

	if (OC->f_has_Orb) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_Orb" << endl;
		}

		string fname_tree_mask;
		int orbit_idx;

		fname_tree_mask = "orbit_" + OC->Group->A->label + "_%d.layered_graph";

		for (orbit_idx = 0; orbit_idx < OC->Orb->Sch->Forest->nb_orbits; orbit_idx++) {

			if (f_v) {
				cout << "orbit " << orbit_idx << " / " <<  OC->Orb->Sch->Forest->nb_orbits
						<< " before Sch->export_tree_as_layered_graph_and_save" << endl;
			}

			OC->Orb->Sch->Forest->export_tree_as_layered_graph_and_save(
					orbit_idx,
					fname_tree_mask,
					verbose_level - 1);
			if (f_v) {
				cout << "orbit " << orbit_idx << " / " <<  OC->Orb->Sch->Forest->nb_orbits
						<< " after Sch->export_tree_as_layered_graph_and_save" << endl;
			}
		}

	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_On_subsets" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_On_subsets not yet implemented" << endl;
		exit(1);


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_On_Subspaces" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_On_Subspaces not yet implemented" << endl;
		exit(1);



	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_On_tensors" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_On_tensors not yet implemented" << endl;
		exit(1);



	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_Cascade" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_Cascade not yet implemented" << endl;
		exit(1);



	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_On_polynomials" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_On_polynomials not yet implemented" << endl;
		exit(1);



	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_Of_One_polynomial" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_Of_One_polynomial not yet implemented" << endl;
		exit(1);

		//orbits_on_polynomials *Of_One_polynomial;


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_on_cubic_curves" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_on_cubic_curves not yet implemented" << endl;
		exit(1);



	}
	else if (OC->f_has_classification_by_canonical_form) {

		if (f_v) {
			cout << "orbits_activity::do_export_trees f_has_classification_by_canonical_form" << endl;
		}
		cout << "orbits_activity::do_export_trees f_has_classification_by_canonical_form not yet implemented" << endl;
		exit(1);



	}
	else {
		cout << "orbits_activity::do_export_trees "
				"no suitable data structure found" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "orbits_activity::do_export_trees done" << endl;
	}

}


void orbits_activity::do_export_source_code(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_source_code" << endl;
	}

	if (OC->f_has_Orb) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_Orb" << endl;
		}


	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_On_Subspaces" << endl;
		}


	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_cubic_surfaces) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_cubic_surfaces" << endl;
		}

		//applications_in_algebraic_geometry::cubic_surfaces_and_double_sixes::surface_classify_wedge *SCW;

		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"before generate_source_code" << endl;
		}
		OC->SCW->Surface_repository->generate_source_code(verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"after generate_source_code" << endl;
		}

	}
	else if (OC->f_has_arcs) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_arcs" << endl;
		}

		//apps_geometry::arc_generator_description *Arc_generator_description_for_arcs;
		//apps_geometry::arc_generator *Arc_generator;

	}
	else if (OC->f_has_semifields) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_semifields" << endl;
		}

		//semifields::semifield_classify_with_substructure *Semifields;

	}
	else if (OC->f_has_boolean_functions) {

		if (f_v) {
			cout << "orbits_activity::do_export_source_code f_has_boolean_functions" << endl;
		}
		//combinatorics::special_functions::boolean_function_domain *BF;
		//apps_combinatorics::boolean_function_classify *BFC;


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"f_has_classification_by_canonical_form" << endl;
		}


		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"before generate_source_code" << endl;
		}

		OC->Canonical_form_classifier->Classification_of_varieties_nauty->generate_source_code(
				OC->Descr->Canonical_form_classifier_description->fname_base_out,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export_source_code "
					"after generate_source_code" << endl;
		}


	}
	else {
		cout << "orbits_activity::do_export_source_code "
				"no suitable data structure found" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "orbits_activity::do_export_source_code done" << endl;
	}

}


void orbits_activity::do_export_levels(
		int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_export_levels" << endl;
	}

	if (OC->f_has_Orb) {

		string fname_tree_mask;

		fname_tree_mask = "orbit_" + OC->Group->A->label + "_orbit_" + std::to_string(orbit_idx);
		other::data_structures::set_of_sets *SoS;

		OC->Orb->Sch->Forest->get_orbit_by_levels(
				orbit_idx,
				SoS,
				verbose_level);


		other::orbiter_kernel_system::file_io Fio;

		std::string fname_layers;

		fname_layers = fname_tree_mask + "_level_sets.csv";

		if (f_v) {
			cout << "orbits_activity::do_export_levels "
					"before SoS->save_csv" << endl;
		}

		SoS->save_csv(
				fname_layers,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_export_levels "
					"after SoS->save_csv" << endl;
		}

#if 0
		int i;
		string fname;

		for (i = 0; i < SoS->nb_sets; i++) {
			fname = fname_tree_mask + "_level_" + std::to_string(i) + ".csv";

			string label;

			label = "lvl" + std::to_string(i);

			Fio.Csv_file_support->lint_vec_write_csv(
					SoS->Sets[i], SoS->Set_size[i],
					fname, label);

			if (f_v) {
				cout << "Written file " << fname
						<< " of size " << Fio.file_size(fname) << endl;
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
				cout << "orbits_activity::do_export_levels "
						"path.size() != i" << endl;
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
#endif

	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_On_Subspaces" << endl;
		}


	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_export_levels f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_export_levels "
					"f_has_classification_by_canonical_form" << endl;
		}
	}
	else {
		cout << "orbits_activity::do_export_levels "
				"no suitable data structure found" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_export_levels done" << endl;
	}

}


void orbits_activity::do_draw_tree(
		other::graphics::layered_graph_draw_options *Draw_options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_draw_tree" << endl;
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree "
				"tree index = " << Descr->draw_tree_idx << endl;
	}


	if (OC->f_has_Orb) {
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"f_has_Orb" << endl;
		}
		string fname;

		fname = OC->Orb->prefix + "_orbit_" + std::to_string(Descr->draw_tree_idx) + "_tree";

		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"before OC->Orb->Sch->Forest->draw_tree" << endl;
		}
		OC->Orb->Sch->Forest->draw_tree(
				fname,
				Draw_options,
				Descr->draw_tree_idx,
				false /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"after OC->Orb->Sch->Forest->draw_tree" << endl;
		}

		std::vector<int> Orb;

		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"before OC->Orb->Sch->Forest->get_orbit_in_order" << endl;
		}
		OC->Orb->Sch->Forest->get_orbit_in_order(
				Orb,
				Descr->draw_tree_idx, verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"after OC->Orb->Sch->Forest->get_orbit_in_order" << endl;
		}

		other::orbiter_kernel_system::file_io Fio;

		string fname_full;

		fname_full = fname + "_orbit_elements.csv";


		Fio.Csv_file_support->vector_write_csv(
				fname_full, Orb);
		if (f_v) {
			cout << "Written file " << fname_full << " of size "
					<< Fio.file_size(fname_full) << endl;
		}

	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_On_Subspaces" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::do_draw_tree before draw_poset" << endl;
		}
		OC->On_Subspaces->orbits_on_subspaces_PC->draw_poset(
				OC->On_Subspaces->orbits_on_subspaces_PC->get_problem_label_with_path(),
				OC->On_Subspaces->orbits_on_subspaces_PC->get_control()->depth,
				0 /* data1 */,
				Draw_options,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree after draw_poset" << endl;
		}




	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		string fname;


		fname = OC->prefix + "_orbit_" + std::to_string(Descr->draw_tree_idx) + "_tree";

		OC->On_polynomials->Sch->Forest->draw_tree(
				fname,
				Draw_options,
				Descr->draw_tree_idx,
				false /* f_has_point_labels */, NULL /* long int *point_labels*/,
				verbose_level);


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_draw_tree f_has_on_cubic_curves" << endl;
		}
		if (f_v) {
			cout << "orbits_activity::do_draw_tree before draw_poset" << endl;
		}
		OC->CCC->Arc_gen->gen->draw_poset(
				OC->CCC->Arc_gen->gen->get_problem_label_with_path(),
				OC->Arc_generator_description->target_size,
			0 /* data1 */,
			Draw_options,
			verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_draw_tree after draw_poset" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_draw_tree "
					"f_has_classification_by_canonical_form" << endl;
		}
	}
	else {
		cout << "orbits_activity::do_draw_tree "
				"no suitable data structure found" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_draw_tree done" << endl;
	}
}

void orbits_activity::do_stabilizer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer" << endl;
	}

	if (OC->f_has_Orb) {

		groups::strong_generators *Stab;

		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"before Orb->stabilizer_of" << endl;
		}
		//OC->Orb->stabilizer_of(Descr->stabilizer_orbit_idx, verbose_level);

		OC->Orb->stabilizer_any_point(
				Descr->stabilizer_point,
				Stab, verbose_level);


		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"after Orb->stabilizer_of" << endl;
		}


		std::string gens_str;
		algebra::ring_theory::longinteger_object stab_go;


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

		Stab->report_group(
				label_stab, verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"exporting stabilizer orbit representative "
					"of point " << Descr->stabilizer_point
					<< " to " << fname_stab << endl;
		}
		Stab->export_to_orbiter_as_bsgs(
				Stab->A,
				fname_stab, label_stab, label_stab,
				verbose_level);

		FREE_OBJECT(Stab);

	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_On_Subspaces" << endl;
		}


	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"f_has_classification_by_canonical_form" << endl;
		}

	}
	else {
		cout << "orbits_activity::do_stabilizer "
				"no suitable data structure found" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer done" << endl;
	}

}

void orbits_activity::do_stabilizer_of_orbit_rep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep" << endl;
	}

	if (OC->f_has_Orb) {

		groups::strong_generators *Stab;

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep "
					"before OC->Orb->stabilizer_of" << endl;
		}
		OC->Orb->stabilizer_of(Descr->stabilizer_of_orbit_rep_orbit_idx, Stab, verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep "
					"after OC->Orb->stabilizer_of" << endl;
		}

		FREE_OBJECT(Stab);

	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_On_Subspaces" << endl;
		}


	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_stabilizer_of_orbit_rep f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_stabilizer "
					"f_has_classification_by_canonical_form" << endl;
		}

	}
	else {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep "
				"no suitable data structure found" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "orbits_activity::do_stabilizer_of_orbit_rep done" << endl;
	}

}

void orbits_activity::do_Kramer_Mesner_matrix(
		int verbose_level)
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

		poset_classification::poset_classification_global PCG;

		PCG.init(
				OC->On_subsets,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before PCG.compute_Kramer_Mesner_matrix" << endl;
		}
		PCG.compute_Kramer_Mesner_matrix(
				Descr->Kramer_Mesner_t,
				Descr->Kramer_Mesner_k,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after PCG.compute_Kramer_Mesner_matrix" << endl;
		}

	}

	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"f_has_On_Subspaces" << endl;
		}

		poset_classification::poset_classification_global PCG;

		PCG.init(
				OC->On_Subspaces->orbits_on_subspaces_PC,
				verbose_level);

		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"before PCG.compute_Kramer_Mesner_matrix" << endl;
		}
		PCG.compute_Kramer_Mesner_matrix(
				Descr->Kramer_Mesner_t,
				Descr->Kramer_Mesner_k,
				verbose_level);
		if (f_v) {
			cout << "orbits_activity::do_Kramer_Mesner_matrix "
					"after PCG.compute_Kramer_Mesner_matrix" << endl;
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

void orbits_activity::do_recognize(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_activity::do_recognize" << endl;
	}

	if (OC->f_has_Orb) {
		if (f_v) {
			cout << "orbits_activity::do_recognize "
					"f_has_Orb" << endl;
		}
	}
	if (OC->f_has_On_subsets) {

		int h;

		if (f_v) {
			cout << "orbits_activity::do_recognize "
					"number of objects to recognize = " << Descr->recognize.size() << endl;
		}
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

	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_recognize f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_recognize f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_recognize f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

		if (f_v) {
			cout << "orbits_activity::do_recognize f_has_Of_One_polynomial" << endl;
		}


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_recognize f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_recognize "
					"f_has_classification_by_canonical_form" << endl;
		}

	}

	else {
		cout << "orbits_activity::do_recognize "
				"no suitable data structure found" << endl;
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

	if (OC->f_has_Orb) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_Orb" << endl;
		}


	}
	else if (OC->f_has_On_subsets) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_On_subsets" << endl;
		}


	}
	else if (OC->f_has_On_Subspaces) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_On_Subspaces" << endl;
		}


	}
	else if (OC->f_has_On_tensors) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_On_tensors" << endl;
		}


	}
	else if (OC->f_has_Cascade) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_Cascade" << endl;
		}


	}
	else if (OC->f_has_On_polynomials) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_On_polynomials" << endl;
		}


	}
	else if (OC->f_has_Of_One_polynomial) {

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
				cout << "orbits_activity::do_transporter "
						"before OC->Of_One_polynomial->Orb->get_transporter" << endl;

			}
			OC->Of_One_polynomial->Orb->get_transporter(
					the_set[i],
					transporter, verbose_level);

			if (f_v) {
				cout << "orbits_activity::do_transporter "
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

		other::orbiter_kernel_system::file_io Fio;

		cout << "orbits_activity::do_transporter "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;


	}
	else if (OC->f_has_on_cubic_curves) {

		if (f_v) {
			cout << "orbits_activity::do_transporter f_has_on_cubic_curves" << endl;
		}


	}
	else if (OC->f_has_classification_by_canonical_form) {
		if (f_v) {
			cout << "orbits_activity::do_transporter "
					"f_has_classification_by_canonical_form" << endl;
		}

	}
	else {
		cout << "orbits_activity::do_transporter "
				"no suitable data structure found" << endl;
		exit(1);
	}




	if (f_v) {
		cout << "orbits_activity::do_transporter done" << endl;
	}
}

}}}




