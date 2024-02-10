/*
 * poset_classification_report.cpp
 *
 *  Created on: May 5, 2019
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {

void poset_classification::report(
		poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report" << endl;
	}


	string fname_report;
	fname_report = problem_label + "_poset.tex";
	l1_interfaces::latex_interface L;
	orbiter_kernel_system::file_io Fio;

	{
		ofstream ost(fname_report);
		L.head_easy(ost);

		if (f_v) {
			cout << "poset_classification::report "
					"before get_A2()->report" << endl;
		}

		get_A2()->report(ost,
				false /* f_sims */, NULL,
				false /* f_strong_gens */, NULL,
				Control->draw_options,
				verbose_level - 1);

		if (f_v) {
			cout << "poset_classification::report "
					"after get_A2()->report" << endl;
		}

		if (f_v) {
			cout << "poset_classification::report "
					"before report2" << endl;
		}
		report2(ost, Opt, verbose_level);
		if (f_v) {
			cout << "poset_classification::report "
					"after report2" << endl;
		}

		L.foot(ost);
	}
	cout << "Written file " << fname_report << " of size "
			<< Fio.file_size(fname_report) << endl;


	if (f_v) {
		cout << "poset_classification::report done" << endl;
	}
}

void poset_classification::report2(
		std::ostream &ost,
		poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report2" << endl;
	}


	if (f_v) {
		cout << "poset_classification::report2 Opt:" << endl;
		Opt->print();
		cout << "poset_classification::report2 draw_options:" << endl;
		Control->draw_options->print();
	}

#if 1
	if (Opt->f_draw_poset) {
		if (f_v) {
			cout << "poset_classification::report2 "
					"before draw_poset" << endl;
		}
		if (!Control->f_draw_options) {
			cout << "poset_classification::report2 "
					"Control->f_draw_poset && !Control->f_draw_options" << endl;
			exit(1);
		}
		draw_poset(
				get_problem_label_with_path(),
				depth /*actual_size*/,
			0 /* data1 */,
			Control->draw_options,
			verbose_level);
		if (f_v) {
			cout << "poset_classification::report2 "
					"after draw_poset" << endl;
		}
	}


#endif




	if (f_v) {
		cout << "poset_classification::report2 Orbits" << endl;
	}
	ost << "Poset classification up to depth " << depth << "\\\\" << endl;

	ost << endl;
	ost << "\\section*{The Orbits}" << endl;
	ost << endl;


	if (f_v) {
		cout << "poset_classification::report2 "
				"Number of Orbits By Level" << endl;
	}
	if (f_v) {
		cout << "poset_classification::report2 "
				"before section Number of Orbits By Level" << endl;
	}


	ost << "\\subsection*{Number of Orbits By Level}" << endl;

	if (f_v) {
		cout << "poset_classification::report2 "
				"before report_number_of_orbits_at_level" << endl;
	}
	report_number_of_orbits_at_level(ost, Opt, verbose_level);
	if (f_v) {
		cout << "poset_classification::report2 "
				"after report_number_of_orbits_at_level" << endl;
	}



	if (f_v) {
		cout << "poset_classification::report2 "
				"before section Summary of Orbit Representatives" << endl;
	}



	ost << endl;
	ost << "\\subsection*{Summary of Orbit Representatives}" << endl;
	ost << endl;

	if (f_v) {
		cout << "poset_classification::report2 "
				"before report_orbits_summary" << endl;
	}
	report_orbits_summary(ost, Opt, verbose_level);
	if (f_v) {
		cout << "poset_classification::report2 "
				"after report_orbits_summary" << endl;
	}


	ost << endl;



	if (Opt->f_draw_poset) {

		if (f_v) {
			cout << "poset_classification::report2 "
					"before section The Poset of Orbits: Diagram" << endl;
		}
		ost << "\\section*{The Poset of Orbits: Diagram}" << endl;

		report_poset_of_orbits(ost, Opt, verbose_level);

	}
	else {
		cout << "please use option -draw_poset if you want to draw the poset" << endl;
	}


	if (f_v) {
		cout << "poset_classification::report2 "
				"before section Poset of Orbits in Detail" << endl;
	}


	ost << endl;
	ost << "\\section*{Poset of Orbits in Detail}" << endl;
	ost << endl;

	report_orbits_in_detail(ost, Opt, verbose_level);


	if (f_v) {
		cout << "poset_classification::report2 done" << endl;
	}

}

void poset_classification::report_orbits_in_detail(
		std::ostream &ost,
		poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report_orbits_in_detail" << endl;
	}
	int orbit_at_level;
	int level;
	int nb_orbits;

	for (level = 0; level <= depth; level++) {

		if (Opt->f_select_orbits_by_level && level != Opt->select_orbits_by_level_level) {
			continue;
		}

		if (f_v) {
			cout << "poset_classification::report "
					"Orbits at Level " << level << ":" << endl;
		}
		ost << endl;
		ost << "\\subsection*{Orbits at Level " << level << "}" << endl;
		ost << endl;


		nb_orbits = nb_orbits_at_level(level);

		ost << "There are " << nb_orbits
				<< " orbits at level " << level << ".\\\\" << endl;
		ost << "\\bigskip" << endl;

		for (orbit_at_level = 0;
				orbit_at_level < nb_orbits;
				orbit_at_level++) {

			report_orbit(level, orbit_at_level, Opt, ost, verbose_level);

		}
	}

	if (f_v) {
		cout << "poset_classification::report_orbits_in_detail done" << endl;
	}
}

void poset_classification::report_number_of_orbits_at_level(
		std::ostream &ost,
		poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report_number_of_orbits_at_level" << endl;
	}
	int *N;
	int i;

	N = NEW_int(depth + 1);
	for (i = 0; i <= depth; i++) {
		N[i] = nb_orbits_at_level(i);
	}
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Depth} & \\mbox{Nb of orbits} & \\mbox{Ago}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i <= depth; i++) {

		long int *Ago;
		int nb;

		if (f_v) {
			cout << "poset_classification::report_number_of_orbits_at_level "
					"before get_all_stabilizer_orders_at_level" << endl;
		}

		get_all_stabilizer_orders_at_level(i, Ago, nb, verbose_level);

		if (f_v) {
			cout << "poset_classification::report_number_of_orbits_at_level "
					"after get_all_stabilizer_orders_at_level" << endl;
		}



		ost << i << " & " << N[i] << " & ";

		data_structures::tally T;

		T.init_lint(Ago, nb, false, 0);
		T.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards */);



		ost << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << endl;
	FREE_int(N);
	if (f_v) {
		cout << "poset_classification::report_number_of_orbits_at_level done" << endl;
	}

}

void poset_classification::report_orbits_summary(
		std::ostream &ost,
		poset_classification_report_options *Opt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 10);

	if (f_v) {
		cout << "poset_classification::report_orbits_summary" << endl;
	}

	ost << "N = node\\\\" << endl;
	ost << "D = depth or level\\\\" << endl;
	ost << "O = orbit with a level\\\\" << endl;
	ost << "Rep = orbit representative\\\\" << endl;
	ost << "(S,O) = (order of stabilizer, orbit length)\\\\" << endl;
	ost << "L = number of live points\\\\" << endl;
	ost << "F = number of flags\\\\" << endl;
	//ost << "FO = number of flag orbits\\\\" << endl;
	ost << "Gen = number of generators for the stabilizer of the orbit rep.\\\\" << endl;
	ost << "\\begin{center}" << endl;
	ost << "\\begin{longtable}{|r|r|r|p{3cm}|r|r|r|r|}" << endl;
	ost << "\\caption{Orbit Representatives}\\\\" << endl;
	ost << endl;
	ost << "\\hline N & D & O & Rep & (S,O) "
			"& L & F & Gen\\\\ \\hline " << endl;
	ost << "\\endfirsthead" << endl;
	ost << endl;
	ost << "\\multicolumn{8}{c}%" << endl;
	ost << "{{\\bfseries \\tablename\\ \\thetable{} -- continued "
			"from previous page}} \\\\" << endl;
	ost << "\\hline N & D & O & Rep & (S,O) "
			"& L & F & Gen\\\\ \\hline " << endl;
	ost << "\\endhead" << endl;
	ost << endl;
	ost << "\\hline \\multicolumn{8}{|r|}{{Continued on next page}} "
			"\\\\ \\hline" << endl;
	ost << "\\endfoot" << endl;
	ost << endl;
	ost << "\\hline \\hline" << endl;
	ost << "\\endlastfoot" << endl;

	int i;
	int level, nb_orbits, cnt, nb_live_pts, nb_extensions, /*nbo,*/ nbg;
	long int *rep = NULL;
	poset_orbit_node *O;
	ring_theory::longinteger_object stab_order, orbit_length;
	data_structures_groups::schreier_vector *Schreier_vector;
	l1_interfaces::latex_interface L;


	rep = NEW_lint(depth + 1);

	if (f_vv) {
		cout << "poset_classification::report_orbits_summary "
				"printing orbit representative" << endl;
	}

	cnt = 0;
	for (level = 0; level <= depth; level++) {

		if (f_vv) {
			cout << "poset_classification::report_orbits_summary "
					"printing orbit representative at level " << level << endl;
		}


		if (Opt->f_select_orbits_by_level && level != Opt->select_orbits_by_level_level) {
			continue;
		}



		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"printing orbit representative at "
						"level " << level << " orbit " << i << endl;
			}

			get_set_by_level(level, i, rep);

			string s;

			s = Lint_vec_stringify(rep, level);

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"set: '" << s << "'" << endl;
			}

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"before get_orbit_length_and_stabilizer_order" << endl;
			}
			get_orbit_length_and_stabilizer_order(i, level,
				stab_order, orbit_length);

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"after get_orbit_length_and_stabilizer_order" << endl;
			}

			//stab_order.print_to_string(str);

			//orbit_length.print_to_string(str);

			O = get_node_ij(level, i);

			long int so;

			so = O->get_stabilizer_order_lint(this);

			if (!Opt->is_selected_by_group_order(so)) {
				continue;
			}


			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"after get_node_ij" << endl;
			}

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"before O->get_Schreier_vector" << endl;
			}
			Schreier_vector = O->get_Schreier_vector();

			if (level < depth && Schreier_vector) {
				if (Schreier_vector == NULL) {
					cout << "poset_classification::report_orbits_summary "
							"Schreier_vector == NULL" << endl;
					exit(1);
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"level < depth; level=" << level << endl;
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"before O->get_nb_of_live_points" << endl;
				}
				nb_live_pts = O->get_nb_of_live_points();
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"after O->get_nb_of_live_points" << endl;
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"before O->get_nb_of_extensions" << endl;
				}
				nb_extensions = O->get_nb_of_extensions();
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"after O->get_nb_of_extensions" << endl;
				}
				//nbo = O->get_nb_of_orbits_under_stabilizer();
				if (Schreier_vector->f_has_local_generators) {
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary "
								"before Schreier_vector->local_gens->len" << endl;
					}
					nbg = Schreier_vector->local_gens->len;
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary "
								"after Schreier_vector->local_gens->len" << endl;
					}
				}
				else {
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary "
								"before O->get_nb_strong_generators" << endl;
					}
					nbg = O->get_nb_strong_generators();
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary "
								"after O->get_nb_strong_generators" << endl;
					}
				}
			}
			else {
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary "
							"level < depth is false" << endl;
				}
				nb_live_pts = -1;
				nb_extensions = -1;
				//nbo = -1;
				nbg = O->get_nb_strong_generators();
			}
			if (f_vv) {
				cout << "poset_classification::report_orbits_summary "
						"nb_live_pts=" << nb_live_pts
					<< " nb_extensions=" << nb_extensions
					<< " nbg=" << nbg << endl;
			}

			ost << cnt << " & " << level << " & " << i
					<< " & $\\{$ " << s << " $\\}$ & ("
					<< stab_order << ", "
					<< orbit_length << ") & ";

			if (nb_live_pts >= 0) {
				ost << nb_live_pts << " & ";
			}
			else {
				ost << " & ";
			}
			if (nb_extensions >= 0) {
				ost << nb_extensions << " & ";
			}
			else {
				ost << " & ";
			}
#if 0
			if (nbo >= 0) {
				ost << nbo << " & ";
			}
			else {
				ost << " & ";
			}
#endif
			if (nbg >= 0) {
				ost << nbg << "\\\\" << endl;
			}
			else {
				ost << "\\\\" << endl;
			}


			cnt++;
		}
		ost << "\\hline" << endl;
	}

	ost << "\\end{longtable}" << endl;
	ost << "\\end{center}" << endl;
	ost << endl;

	FREE_lint(rep);

	if (f_v) {
		cout << "poset_classification::report_orbits_summary done" << endl;
	}


}


void poset_classification::report_poset_of_orbits(
		std::ostream &ost,
		poset_classification_report_options *Opt,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report_poset_of_orbits "
				"depth=" << depth << endl;
		cout << "poset_classification::report_poset_of_orbits "
				"Opt:" << endl;
		Opt->print();
	}


	string fname_base;
	string fname_poset;
	string fname_out_base;

	//draw_poset_fname_base_poset_lvl(fname_base, depth);

	if (Opt->f_type_aux) {
		draw_poset_fname_base_aux_poset(fname_base, depth);
	}
	else if (Opt->f_type_ordinary) {
		draw_poset_fname_base_poset_lvl(fname_base, depth);
	}
	else if (Opt->f_type_tree) {
		draw_poset_fname_base_tree_lvl(fname_base, depth);
	}
	else if (Opt->f_type_detailed) {
		draw_poset_fname_base_poset_detailed_lvl(fname_base, depth);
	}
	else {
		cout << "please specify the type of drawing of the poset" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "poset_classification::report_poset_of_orbits "
				"fname_base = " << fname_base << endl;
	}
	fname_poset = fname_base + ".layered_graph";
	fname_out_base = fname_base + "_draw";

	if (f_v) {
		cout << "poset_classification::report_poset_of_orbits "
				"fname_poset=" << fname_poset << endl;
		cout << "poset_classification::report_poset_of_orbits "
				"fname_out_base=" << fname_out_base << endl;
	}

	string cmd;


	if (orbiter_kernel_system::Orbiter->f_orbiter_path) {

		cmd.assign(orbiter_kernel_system::Orbiter->orbiter_path);

	}
	else {
		cout << "poset_classification::report_poset_of_orbits "
				"We need -orbiter_path to be set" << endl;
		exit(1);

	}

	cmd += "orbiter.out -v 3 -draw_layered_graph " + fname_poset + " ";

	if (!Control->f_draw_options) {
		cout << "poset_classification::report_poset_of_orbits "
				"We need -draw_options to be set in -poset_classification_control" << endl;
		exit(1);
	}

	cmd += " -xin " + std::to_string(Control->draw_options->xin)
			+ " -yin " + std::to_string(Control->draw_options->yin)
			+ " -xout " + std::to_string(Control->draw_options->xout)
			+ " -yout " + std::to_string(Control->draw_options->yout)
			+ " -radius " + std::to_string(Control->draw_options->rad) + " ";

	if (Control->draw_options->f_y_stretch) {
		cmd += " -y_stretch "
				+ std::to_string(Control->draw_options->y_stretch) + " ";
	}

	if (Control->draw_options->f_line_width) {
		cmd += " -line_width "
				+ std::to_string(Control->draw_options->line_width) + " ";
	}
	if (Control->draw_options->f_spanning_tree) {
		cmd += " -spanning_tree ";
	}

	cout << "poset_classification::report_poset_of_orbits "
			"executing command: " << cmd << endl;
	system(cmd.c_str());

	cmd = "mpost -tex=latex " + fname_out_base + ".mp";
	cout << "executing: " << cmd << endl;
	system(cmd.c_str());


	ost << "\\input " << fname_out_base << ".tex" << endl;
#if 0
	ost << "\\clearpage" << endl;
	ost << "$$" << endl;
	ost << "\\includegraphics[width=120mm]{" << fname_out_base << "}" << endl;
	ost << "$$" << endl;
#endif
	//ost << "\\includegraphics[width=160mm]{" << fname_mp << ".1}\\\\" << endl;

	if (f_v) {
		cout << "poset_classification::report_poset_of_orbits done" << endl;
	}
}


void poset_classification::report_orbit(
		int level, int orbit_at_level,
		poset_classification_report_options *Opt,
		std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report_orbit "
				"level = " << level
				<< " orbit_at_level = " << orbit_at_level << endl;
	}
	int nb_orbits;
	int nb_gens;
	int nb_extensions;
	poset_orbit_node *O;
	ring_theory::longinteger_object stab_order, orbit_length;
	string s;
	long int *rep = NULL;
	data_structures_groups::schreier_vector *Schreier_vector;
	l1_interfaces::latex_interface L;
	long int so;

	rep = NEW_lint(depth + 1);


	nb_orbits = nb_orbits_at_level(level);

	O = get_node_ij(level, orbit_at_level);

	so = O->get_stabilizer_order_lint(this);

	if (!Opt->is_selected_by_group_order(so)) {
		return;
	}

	ost << "\\subsection*{Orbit " << orbit_at_level
			<< " / " << nb_orbits << " at Level " << level << "}" << endl;


	ost << "Node number: " << O->get_node() << "\\\\" << endl;
	ost << "Parent node: " << O->get_prev() << "\\\\" << endl;




	groups::strong_generators *gens;

	get_stabilizer_generators(gens,
			level, orbit_at_level, Control->verbose_level);


#if 0
	groups::strong_generators *projectivity_group_gens = NULL;

	Poset->A->compute_projectivity_subgroup(projectivity_group_gens,
			gens, 0 /*verbose_level*/);


	if (projectivity_group_gens) {
		ring_theory::longinteger_object proj_stab_order;

		projectivity_group_gens->group_order(proj_stab_order);
		proj_stab_order.print_to_string(str2);
	}
#endif

	get_orbit_length_and_stabilizer_order(
			orbit_at_level, level,
		stab_order, orbit_length);



	stab_order.print_to_string(s);

	//orbit_length.print_to_string(str);

	Schreier_vector = O->get_Schreier_vector();


	get_set_by_level(level, orbit_at_level, rep);


	// print the set and stabilizer order:


	ost << "$$" << endl;
	L.lint_set_print_tex(ost, rep, level);
	ost << "_{";
	ost << s;
#if 0
	if (projectivity_group_gens) {
		ost << "," << str2;
		FREE_OBJECT(projectivity_group_gens);
		projectivity_group_gens = NULL;
	}
#endif
	ost << "}";
	ost << "$$" << endl;


	// print strong generators for the stabilizer:

	if (f_v) {
		cout << "poset_classification::report_orbit "
				"before Poset->A2->latex_point_set" << endl;
	}
	Poset->A2->latex_point_set(
			ost, rep, level, 0 /* verbose_level*/);
	if (f_v) {
		cout << "poset_classification::report_orbit "
				"after Poset->A2->latex_point_set" << endl;
	}


	ost << "{\\small\\arraycolsep=2pt" << endl;
	gens->print_generators_tex(ost);
	ost << "}" << endl;

	nb_gens = gens->gens->len;

	nb_extensions = O->get_nb_of_extensions();
	//ost << "There are " << nbo << " orbits\\\\" << endl;
	ost << "There are " << nb_extensions
			<< " extensions\\\\" << endl;
	ost << "Number of generators " << O->get_nb_strong_generators()
			<< "\\\\" << endl;

	if (Schreier_vector) {
		int nb_orbits_sv = Schreier_vector->number_of_orbits;

		if (Schreier_vector->f_has_local_generators) {

			ost << "Generators for the Schreier trees:\\\\" << endl;
			ost << "{\\small\\arraycolsep=2pt" << endl;
			Schreier_vector->local_gens->print_generators_tex(stab_order, ost);
			ost << "}" << endl;

			nb_gens = Schreier_vector->local_gens->len;
		}

		int nb_o, h;
		int *orbit_reps;
		int *orbit_length;
		int *total_depth;
		Schreier_vector->orbit_stats(
				nb_o, orbit_reps, orbit_length, total_depth,
				0 /*verbose_level*/);
		if (nb_o != nb_orbits_sv) {
			cout << "nb_o != nb_orbits_sv" << endl;
			exit(1);
		}

		data_structures::tally T;

		T.init(orbit_length, nb_o, false, 0);
		ost << "Orbit type of flag orbits: \\\\" << endl;
		ost << "$$" << endl;
		T.print_file_tex_we_are_in_math_mode(ost, true /* f_backwards*/);
		ost << "$$" << endl;

		for (h = 0; h < nb_o; h++) {
			ost << "\\noindent Orbit " << h << " / " << nb_o
					<< ": Point " << orbit_reps[h]
					<< " lies in an orbit of length "
					<< orbit_length[h] << " with average word length "
					<< (double) total_depth[h] / (double) orbit_length[h];
			if (nb_gens > 1) {
				ost << " $H_{" << nb_gens << "} = "
					<< (double) log(total_depth[h]) / log(nb_gens) << "$";
			}
			double delta = (double) total_depth[h] / (double) orbit_length[h];
			delta -= ((double) log(total_depth[h]) / log(nb_gens));
			ost << ", $\\Delta = " << delta << "$";
			ost << "\\\\" << endl;
		}


#if 0
		string fname_mask_base;

		create_shallow_schreier_tree_fname_mask_base(
				fname_mask_base, O->get_node());
		//create_schreier_tree_fname_mask_base(
		//fname_mask_base, O->node);

		for (j = 0; j < nb_orbits_sv; j++) {

			//char fname_base[1000];
			char fname_layered_graph[2000];
			char fname_tex[2000];
			char fname_mp[2000];
			char fname_1[2000];

			snprintf(fname_base, 1000, fname_mask_base, j);
			snprintf(fname_layered_graph, 2000, "%s.layered_graph",
					fname_base);
			snprintf(fname_tex, 2000, "%s_draw_tree.tex", fname_base);
			snprintf(fname_mp, 2000, "%s_draw_tree.mp", fname_base);
			snprintf(fname_1, 2000, "%s_draw_tree.1", fname_base);

			if (Control->f_has_tools_path) {
				snprintf(cmd, 10000, "%s/layered_graph_main.out -v 2 "
					"-file %s "
					"-xin 1000000 -yin 1000000 "
					"-xout 1000000 -yout 1000000 "
					"-y_stretch 0.3 "
					"-rad 2000 "
					"-nodes_empty "
					"-corners "
					//"-embedded "
					"-line_width 0.30 "
					"-spanning_tree",
					Control->tools_path, fname_layered_graph);
				cout << "executing: " << cmd << endl;
				system(cmd);

				snprintf(cmd, 10000, "mpost %s", fname_mp);
				cout << "executing: " << cmd << endl;
				system(cmd);

				ost << "\\subsubsection*{Node " << O->node << " at Level "
						<< level << " Orbit " << orbit_at_level
						<< " / " << nb_orbits
						<< " Tree " << j << " / " << nb_orbits_sv << "}" << endl;

				//nbo = Schreier_vector->number_of_orbits;
				if (Schreier_vector->f_has_local_generators) {
					nbg = Schreier_vector->local_gens->len;
				}
				else {
					nbg = O->nb_strong_generators;
				}
				ost << "Number of generators " << nbg
						<< "\\\\" << endl;


				//ost << "\\input " << fname_tex << endl;
				ost << "\\includegraphics[width=160mm]{"
						<< fname_1 << "}\\" << endl;
			}
			else {
				//cout << "please set tools path using "
				//		"-tools_path <tools_path>" << endl;
				//exit(1);
			}

			int e;

			e = O->find_extension_from_point(this, orbit_reps[j],
					0 /* verbose_level */);

			if (e >= 0) {
				ost << endl;
				ost << "\\noindent Extension number " << e << "\\\\" << endl;
				ost << "Orbit representative " << orbit_reps[j] << "\\\\" << endl;
				ost << "Flag orbit length " << O->E[e].orbit_len << "\\\\" << endl;

				if (O->E[e].type == EXTENSION_TYPE_UNPROCESSED) {
					ost << "Flag orbit is unprocessed.\\\\" << endl;
				}
				else if (O->E[e].type == EXTENSION_TYPE_EXTENSION) {
					ost << "Flag orbit is defining new orbit " << O->E[e].data << " at level " << level + 1 << "\\\\" << endl;
				}
				else if (O->E[e].type == EXTENSION_TYPE_FUSION) {
					ost << "Flag orbit is fused to node " << O->E[e].data1 << " extension " << O->E[e].data2 << "\\\\" << endl;
					ost << "Fusion element:\\\\" << endl;
					ost << "$$" << endl;

					Poset->A->element_retrieve(O->E[e].data, Elt1, 0);

					Poset->A->element_print_latex(Elt1, ost);
					ost << "$$" << endl;
					Poset->A->element_print_for_make_element(Elt1, ost);
					ost << "\\\\" << endl;
				}
			}
			else {
				ost << endl;
				ost << "Cannot find an extension for point " << orbit_reps[j] << "\\\\" << endl;
			}
#if 0
			int pt;
			int orbit_len;
			int type;
				// EXTENSION_TYPE_UNPROCESSED = unprocessed
				// EXTENSION_TYPE_EXTENSION = extension node
				// EXTENSION_TYPE_FUSION = fusion node
				// EXTENSION_TYPE_PROCESSING = currently processing
				// EXTENSION_TYPE_NOT_CANONICAL = no extension formed
				// because it is not canonical
			int data;
				// if EXTENSION_TYPE_EXTENSION: a handle to the next
				//  poset_orbit_node
				// if EXTENSION_TYPE_FUSION: a handle to a fusion element
			int data1;
				// if EXTENSION_TYPE_FUSION: node to which we are fusing
			int data2;
				// if EXTENSION_TYPE_FUSION: extension within that
				// node to which we are fusing
#endif

		}
#endif


		FREE_int(orbit_reps);
		FREE_int(orbit_length);
		FREE_int(total_depth);
	}
	FREE_OBJECT(gens);
	FREE_lint(rep);
	if (f_v) {
		cout << "poset_classification::report_orbit "
				"level = " << level
				<< " orbit_at_level = " << orbit_at_level << " done" << endl;
	}
}


}}}

