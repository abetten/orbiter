/*
 * surface_classify_wedge_io.cpp
 *
 *  Created on: Feb 17, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_and_double_sixes {




void surface_classify_wedge::write_file(
		std::ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_file" << endl;
	}
	fp.write((char *) &q, sizeof(int));

	Flag_orbits->write_file(fp, verbose_level);

	Surfaces->write_file(fp, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::write_file finished" << endl;
	}
}

void surface_classify_wedge::read_file(
		std::ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int q1;

	if (f_v) {
		cout << "surface_classify_wedge::read_file" << endl;
	}
	fp.read((char *) &q1, sizeof(int));
	if (q1 != q) {
		cout << "surface_classify_wedge::read_file q1 != q" << endl;
		exit(1);
	}

	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->read_file(fp, A, A2, verbose_level);

	Surfaces = NEW_OBJECT(invariant_relations::classification_step);

	algebra::ring_theory::longinteger_object go;

	A->group_order(go);

	Surfaces->read_file(fp, A, A2, go, verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::read_file finished" << endl;
	}
}







void surface_classify_wedge::generate_history(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::generate_history" << endl;
	}

	poset_classification::poset_classification_global PCG;

	PCG.init(
			Five_p1->Five_plus_one,
			verbose_level);


	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"before PCG.generate_history" << endl;
	}
	PCG.generate_history(5, verbose_level - 2);
	if (f_v) {
		cout << "surface_classify_wedge::generate_history "
				"after PCG.generate_history" << endl;
	}

	if (f_v) {
		cout << "surface_classify_wedge::generate_history done" << endl;
	}

}

int surface_classify_wedge::test_if_surfaces_have_been_computed_already()
{
	string fname;
	other::orbiter_kernel_system::file_io Fio;
	int ret;

	fname = "Surfaces_q" + std::to_string(q) + ".data";
	if (Fio.file_size(fname) > 0) {
		//ret = true;
		ret = false; // !!! ToDo don't use data file
	}
	else {
		ret = false;
	}
	return ret;
}

void surface_classify_wedge::write_surfaces(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces" << endl;
	}
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "Surfaces_q" + std::to_string(q) + ".data";
	{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before write_file" << endl;
		}
		write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_surfaces done" << endl;
	}
}

void surface_classify_wedge::read_surfaces(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces" << endl;
	}
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "Surfaces_q" + std::to_string(q) + ".data";
	cout << "Reading file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	{
		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"before read_file" << endl;
			}
		read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"after read_file" << endl;
		}
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"before post_process" << endl;
		}
		post_process(verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::read_surfaces "
					"after post_process" << endl;
		}

	}
	if (f_v) {
		cout << "surface_classify_wedge::read_surfaces done" << endl;
	}
}

int surface_classify_wedge::test_if_double_sixes_have_been_computed_already()
{
	string fname;
	other::orbiter_kernel_system::file_io Fio;
	int ret;

	fname = "Double_sixes_q" + std::to_string(q) + ".data";
	if (Fio.file_size(fname) > 0) {
		//ret = true;
		ret = false; // !!! ToDo
	}
	else {
		ret = false;
	}
	return ret;
}

void surface_classify_wedge::write_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes" << endl;
	}
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "Double_sixes_q" + std::to_string(q) + ".data";
	{

		ofstream fp(fname);

		if (f_v) {
			cout << "surface_classify before "
					"SCW->Classify_double_sixes->write_file" << endl;
		}
		Classify_double_sixes->write_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after "
					"SCW->Classify_double_sixes->write_file" << endl;
		}
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
	if (f_v) {
		cout << "surface_classify_wedge::write_double_sixes done" << endl;
	}
}

void surface_classify_wedge::read_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes" << endl;
	}
	string fname;
	other::orbiter_kernel_system::file_io Fio;

	fname = "Double_sixes_q" + std::to_string(q) + ".data";
	if (f_v) {
		cout << "Reading file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	{

		ifstream fp(fname);

		if (f_v) {
			cout << "surface_classify before "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
		Classify_double_sixes->read_file(fp, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify after "
					"SCW->Classify_double_sixes->read_file" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::read_double_sixes done" << endl;
	}
}


void surface_classify_wedge::create_report(
		std::string &options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report" << endl;
	}
	string fname, title, author, extra_praeamble;
	other::orbiter_kernel_system::file_io Fio;

	title = "Cubic Surfaces with 27 Lines over GF(" + std::to_string(q) + ") ";

	author = "Orbiter";

	fname = "Surfaces_q" + std::to_string(q) + ".tex";



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


		if (f_v) {
			cout << "surface_classify_wedge::create_report before report" << endl;
		}
		report(fp, options, verbose_level - 1);
		if (f_v) {
			cout << "surface_classify_wedge::create_report after report" << endl;
		}


		L.foot(fp);
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void surface_classify_wedge::report(
		std::ostream &ost,
		std::string &options,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::report" << endl;
	}



	std::map<std::string, std::string> symbol_table;

	other::data_structures::string_tools ST;

	int f_5plus1 = false;
	int f_double_six = false;
	int f_flag_orbits = false;
	int f_show_group = false;
	//int f_show_stabilizers = false;
	int f_show_orbits = false;
	int max_nb_elements_printed = 0;
	poset_classification::poset_classification_report_options *Opt = NULL;
	std::string fname_mask;


	if (options.length()) {

		ST.parse_value_pairs(symbol_table,
				options, verbose_level - 1);

		if (f_v) {
			cout << "surface_classify_wedge::report parsing option pairs" << endl;
		}



		{
			std::map<std::string, std::string>::iterator it = symbol_table.begin();


			// Iterate through the map and print the elements
			while (it != symbol_table.end()) {
				string label;
				string val;

				label = it->first;
				val = it->second;
				if (f_v) {
					cout << "surface_classify_wedge::report key = " << label << " value = " << val << endl;
				}
				//std::cout << "Key: " << it->first << ", Value: " << it->second << std::endl;
				//assignment.insert(std::make_pair(label, a));
				if (label == "5plus1") {
					if (val == "on") {
						f_5plus1 = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_5plus1 = true" << endl;
						}
					}
					else if (val == "off" /*ST.stringcmp(val, "on") == 0*/) {
						f_5plus1 = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_5plus1 = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
				else if (label == "double_sixes") {
					if (val == "on") {
						f_double_six = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_double_six = true" << endl;
						}
					}
					else if (val == "off") {
						f_double_six = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_double_six = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
				else if (label == "flag_orbits") {
					if (val == "on") {
						f_flag_orbits = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_flag_orbits = true" << endl;
						}
					}
					else if (val == "off") {
						f_flag_orbits = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_flag_orbits = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
				else if (label == "show_group") {
					if (val == "on") {
						f_show_group = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_group = true" << endl;
						}
					}
					else if (val == "off") {
						f_show_group = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_group = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
				else if (label == "report_options") {

					Opt = Get_poset_classification_report_options(val);

				}
				else if (label == "fname_mask") {

					fname_mask = val;

				}
				else if (label == "max_nb_elements_printed") {

					max_nb_elements_printed = std::stoi(val);

				}
#if 0
				else if (label == "show_stabilizers") {
					if (val == "on") {
						f_show_stabilizers = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_stabilizers = true" << endl;
						}
					}
					else if (val == "off") {
						f_show_stabilizers = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_stabilizers = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
#endif
				else if (label == "show_orbits") {
					if (val == "on") {
						f_show_orbits = true;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_orbits = true" << endl;
						}
					}
					else if (val == "off") {
						f_show_orbits = false;
						if (f_v) {
							cout << "surface_classify_wedge::report f_show_orbits = false" << endl;
						}
					}
					else {
						cout << "surface_classify_wedge::report unknown value of option "
								<< label << " value " << val << endl;
						exit(1);
					}
				}
				else {
					cout << "surface_classify_wedge::report unknown option "
							<< label << " with value " << val << endl;
					exit(1);
				}

				++it;
			}
		}
	}

	if (!Opt) {
		cout << "please use report_options=ro" << endl;
		exit(1);
	}

	if (!Opt->f_draw_options) {
		cout << "for a report of the surfaces, please use -draw_options" << endl;
		exit(1);
	}

	other::graphics::layered_graph_draw_options *Draw_options;

	Draw_options = Get_draw_options(Opt->draw_options_label);





	other::l1_interfaces::latex_interface L;


#if 0
	ost << "\\section{The field of order " << LG->F->q << "}" << endl;
	ost << "\\noindent The field ${\\mathbb F}_{"
			<< LG->F->q
			<< "}$ :\\\\" << endl;
	LG->F->cheat_sheet(ost, verbose_level);
#endif


	if (f_5plus1) {
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"before Five_p1->report" << endl;
		}
		Five_p1->report(ost, Draw_options, Opt, verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"after Five_p1->report" << endl;
		}
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"before Classify_double_sixes->print_five_plus_ones" << endl;
		}
		Classify_double_sixes->print_five_plus_ones(ost);
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"after Classify_double_sixes->print_five_plus_ones" << endl;
		}
	}

	if (f_double_six) {
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"before Classify_double_sixes->Flag_orbits->print_latex" << endl;
		}

		{
			string title;

			title.assign("Flag orbits for double sixes");

			Classify_double_sixes->Flag_orbits->print_latex(ost, title, true);
		}
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"after Classify_double_sixes->Flag_orbits->print_latex" << endl;
		}
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"before Classify_double_sixes->Double_sixes->print_latex" << endl;
		}
		{
			string title;

			title.assign("Double Sixes");
			Classify_double_sixes->Double_sixes->print_latex(ost, title, true,
					false, NULL, NULL);
		}
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"after Classify_double_sixes->Double_sixes->print_latex" << endl;
		}
	}


	if (f_flag_orbits) {


		if (f_v) {
			cout << "surface_classify_wedge::report "
					"before Flag_orbits->print_latex" << endl;
		}
		{
			string title;

			title.assign("Flag orbits for cubic surfaces");

			Flag_orbits->print_latex(ost, title, true);
		}
		if (f_v) {
			cout << "surface_classify_wedge::report "
					"after Flag_orbits->print_latex" << endl;
		}
	}

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before Surfaces->print_latex" << endl;
	}
	{
		string title;

		title = "Surfaces: Summary";
		Surfaces->print_latex(
				ost, title, true,
				false, NULL, NULL);
	}
	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after Surfaces->print_latex" << endl;
	}



	if (f_show_group) {

		ost << "\\subsection*{The Group $\\PGGL(4," << q << ")$}" << endl;

		{
			algebra::ring_theory::longinteger_object go;
			A->Strong_gens->group_order(go);

			ost << "The order of the group is ";
			go.print_not_scientific(ost);
			ost << "\\\\" << endl;

			ost << "\\bigskip" << endl;
		}
	}



	if (f_v) {
		cout << "surface_classify_wedge::report "
				"before latex_surfaces" << endl;
	}

	latex_surfaces(
			ost, f_show_orbits, fname_mask, Draw_options, Opt, max_nb_elements_printed,
			verbose_level);

	if (f_v) {
		cout << "surface_classify_wedge::report "
				"after latex_surfaces" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before latex_table_of_trihedral_pairs" << endl;
	}
	Surf->Schlaefli->Schlaefli_trihedral_pairs->latex_table_of_trihedral_pairs(ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after latex_table_of_trihedral_pairs" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before latex_tritangent_planes" << endl;
	}
	Surf->Schlaefli->Schlaefli_tritangent_planes->latex_tritangent_planes(
			ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after latex_tritangent_planes" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before latex_table_of_double_sixes" << endl;
	}
	Surf->Schlaefli->Schlaefli_double_six->latex_table_of_double_sixes(
			ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after latex_table_of_double_sixes" << endl;
	}

	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"before latex_table_of_half_double_sixes" << endl;
	}
	Surf->Schlaefli->Schlaefli_double_six->latex_table_of_half_double_sixes(
			ost);
	if (f_v) {
		cout << "surface_object_with_group::print_everything "
				"after latex_table_of_half_double_sixes" << endl;
	}


	if (f_v) {
		cout << "surface_classify_wedge::report done" << endl;
	}
}

void surface_classify_wedge::latex_surfaces(
		std::ostream &ost,
		int f_print_orbits, std::string &fname_mask,
		other::graphics::layered_graph_draw_options *draw_options,
		poset_classification::poset_classification_report_options *Opt,
		int max_nb_elements_printed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string title;

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces" << endl;
	}
	title = "Cubic Surfaces with 27 Lines in $\\PG(3," + std::to_string(q) + ")$";



#if 0
	Classify_double_sixes->print_five_plus_ones(ost);


	Classify_double_sixes->Double_sixes->print_latex(ost, title_ds);
#endif

#if 0
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"before Surfaces->print_latex" << endl;
	}
	Surfaces->print_latex(
			ost, title, f_with_stabilizers,
			false, NULL, NULL);
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"after Surfaces->print_latex" << endl;
	}
#endif

#if 1
	int orbit_index;

	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"before loop over all surfaces" << endl;
	}
	for (orbit_index = 0; orbit_index < Surface_repository->nb_surfaces; orbit_index++) {
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"before report_surface, "
					"orbit_index = " << orbit_index << endl;
		}
		Surface_repository->report_surface(
				ost,
				orbit_index,
				f_print_orbits, fname_mask,
				draw_options,
				max_nb_elements_printed,
				verbose_level);
		if (f_v) {
			cout << "surface_classify_wedge::latex_surfaces "
					"after report_surface" << endl;
		}
	}
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces "
				"after loop over all surfaces" << endl;
	}
#endif
	if (f_v) {
		cout << "surface_classify_wedge::latex_surfaces done" << endl;
	}
}


void surface_classify_wedge::create_report_double_sixes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes" << endl;
	}


	string fname, title, author, extra_praeamble;

	title = "Cheat Sheet on Double Sixes over GF(" + std::to_string(q) + ") ";
	fname = "Double_sixes_q" + std::to_string(q) + ".tex";

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


		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"before Classify_double_sixes->print_five_plus_ones" << endl;
		}
		Classify_double_sixes->print_five_plus_ones(fp);
		if (f_v) {
			cout << "surface_classify_wedge::create_report_double_sixes "
					"after Classify_double_sixes->print_five_plus_ones" << endl;
		}

		{
			string title;

			title = "Double Sixes";
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"before Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
			Classify_double_sixes->Double_sixes->print_latex(
					fp,
				title, false /* f_with_stabilizers*/,
				false, NULL, NULL);
			if (f_v) {
				cout << "surface_classify_wedge::create_report_double_sixes "
						"after Classify_double_sixes->Double_sixes->print_latex" << endl;
			}
		}

		L.foot(fp);
	}
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}
	if (f_v) {
		cout << "surface_classify_wedge::create_report_double_sixes done" << endl;
	}
}





}}}}


