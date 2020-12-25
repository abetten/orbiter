/*
 * poset_classification_report.cpp
 *
 *  Created on: May 5, 2019
 *      Author: betten
 */


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {


#if 0
void poset_classification::report_schreier_trees(
		ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h, degree, a;
	int N, Nb_gen;
	int *data1;
	int *data2;
	int *data3;
	int level, nb_orbits, cnt, cnt3, elt_size, hdl;
	int *Elt1;
	poset_orbit_node *O;
	longinteger_object stab_order, orbit_length;

	if (f_v) {
		cout << "poset_classification::report_schreier_trees" << endl;
	}
	N = 0;
	for (level = 0; level < depth; level++) {
		N += nb_orbits_at_level(level);
	}


	degree = Poset->A2->degree;
	elt_size= Poset->A->make_element_size;

	Nb_gen = 0;
	for (level = 0; level < depth; level++) {
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {
			O = get_node_ij(level, i);

			Nb_gen += O->nb_strong_generators;

		}
	}


	ost << "# N = number of nodes in the poset classification" << endl;
	ost << N << endl;
	ost << "# degree of permutation action = size of the set "
			"we act on" << endl;
	ost << degree << endl;
	ost << "# size of a group element in int" << endl;
	ost << elt_size << endl;
	ost << "# Nb_gen = total number of generators = number of rows "
			"of data_matrix3" << endl;
	ost << Nb_gen << endl;


	ost << "# number of generators at each node, row in data_matrix3 "
			"where these generators start:" << endl;

	cnt3 = 0;
	for (level = 0; level < depth; level++) {
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {
			O = get_node_ij(level, i);
			ost << O->nb_strong_generators << " " << cnt3 << endl;
			cnt3 += O->nb_strong_generators;
		}
	}
	ost << endl;

	data1 = NEW_int(N * degree);
	data2 = NEW_int(N * degree);
	data3 = NEW_int(Nb_gen * elt_size);
	Elt1 = NEW_int(Poset->A->elt_size_in_int);


	//rep = NEW_int(depth + 1);

	cnt = 0;
	cnt3 = 0;
	for (level = 0; level < depth; level++) {
		//first = first_poset_orbit_node_at_level[level];
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {

			//get_set_by_level(level, i, rep);
			//int_vec_print_to_str_naked(str, rep, level);

			get_orbit_length_and_stabilizer_order(i, level,
				stab_order, orbit_length);
			//stab_order.print_to_string(str);

			//orbit_length.print_to_string(str);

			O = get_node_ij(level, i);
			schreier_vector *Schreier_vector;

			Schreier_vector = O->Schreier_vector;

			//nb_live_pts = O->get_nb_of_live_points();

			int *sv = Schreier_vector->sv;
			int nb_live_pts = sv[0];
			int *points = sv + 1;
			int *parent = points + nb_live_pts;
			int *label = parent + nb_live_pts;
			int f_group_is_trivial;
			if (O->nb_strong_generators == 0) {
				f_group_is_trivial = TRUE;
			}
			else {
				f_group_is_trivial = FALSE;
			}

			cout << "Node " << level << "/" << i << endl;
			cout << "points : parent : label" << endl;
			for (h = 0; h < nb_live_pts; h++) {
				cout << "points[h]" << " : ";
				if (!f_group_is_trivial) {
					cout << parent[h] << " : " << label[h];
				}
				cout << endl;
			}
			for (j = 0; j < degree; j++) {
				data1[cnt * degree + j] = -2;
				data2[cnt * degree + j] = -2;
			}
			cout << "node " << level << "/" << i
					<< " computing data1/data2" << endl;
			for (h = 0; h < nb_live_pts; h++) {
				a = points[h];
				if (f_group_is_trivial) {
					data1[cnt * degree + a] = -1;
					data2[cnt * degree + a] = -1;
				}
				else {
					data1[cnt * degree + a] = parent[h];
					data2[cnt * degree + a] = label[h];
				}
			}
			cnt++;
			cout << "node " << level << "/" << i
					<< " computing data3 cnt3=" << cnt3 << endl;
			if (!f_group_is_trivial) {
				for (h = 0; h < O->nb_strong_generators; h++) {
					cout << "h=" << h << " / "
							<< O->nb_strong_generators << endl;
					hdl = O->hdl_strong_generators[h];
					cout << "before element_retrieve, hdl=" << hdl << endl;
					Poset->A->element_retrieve(hdl, Elt1, 0);
					cout << "after element_retrieve" << endl;
					for (j = 0; j < elt_size; j++) {
						data3[cnt3 * elt_size + j] = Elt1[j];
					}
					cnt3++;
				}
			}
		}
	}

	ost << "# data_matrix1: parent information" << endl;
	for (i = 0; i < N; i++) {
		for (j = 0; j < degree; j++) {
			ost << data1[i * degree + j] << " ";
		}
		ost << endl;
	}
	ost << endl;

	ost << "# data_matrix2: label" << endl;
	for (i = 0; i < N; i++) {
		for (j = 0; j < degree; j++) {
			ost << data2[i * degree + j] << " ";
		}
		ost << endl;
	}
	ost << endl;

	ost << "# data_matrix3: schreier generators" << endl;
	for (i = 0; i < Nb_gen; i++) {
		for (j = 0; j < elt_size; j++) {
			ost << data3[i * elt_size + j] << " ";
		}
		ost << endl;
	}
	ost << endl;

}
#endif

void poset_classification::report(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::report" << endl;
	}

	ost << "Poset classification up to depth " << depth << "\\\\" << endl;

	ost << endl;
	ost << "\\section*{The Orbits}" << endl;
	ost << endl;


	if (f_v) {
		cout << "poset_classification::report before section Number of Orbits By Level" << endl;
	}


	ost << "\\subsection*{Number of Orbits By Level}" << endl;

	report_number_of_orbits_at_level(ost);



	if (f_v) {
		cout << "poset_classification::report before section Summary of Orbit Representatives" << endl;
	}



	ost << endl;
	ost << "\\subsection*{Summary of Orbit Representatives}" << endl;
	ost << endl;

	report_orbits_summary(ost, verbose_level);


	ost << endl;



	if (Control->f_draw_poset) {

		if (f_v) {
			cout << "poset_classification::report before section The Poset of Orbits: Diagram" << endl;
		}
		ost << "\\section*{The Poset of Orbits: Diagram}" << endl;

		report_poset_of_orbits(ost);

	}
	else {
		cout << "please use option -draw_poset if you want to draw the poset" << endl;
	}


	if (f_v) {
		cout << "poset_classification::report before section Poset of Orbits in Detail" << endl;
	}


	ost << endl;
	ost << "\\section*{Poset of Orbits in Detail}" << endl;
	ost << endl;

	int orbit_at_level;
	int level;
	int nb_orbits;

	for (level = 0; level <= depth; level++) {


		ost << endl;
		ost << "\\subsection*{Orbits at Level " << level << "}" << endl;
		ost << endl;


		nb_orbits = nb_orbits_at_level(level);

		ost << "There are " << nb_orbits << " orbits at level " << level << ".\\\\" << endl;
		ost << "\\bigskip" << endl;

		for (orbit_at_level = 0;
				orbit_at_level < nb_orbits;
				orbit_at_level++) {

			report_orbit(level, orbit_at_level, ost);

		}
	}

	if (f_v) {
		cout << "poset_classification::report done" << endl;
	}

}


void poset_classification::report_number_of_orbits_at_level(std::ostream &ost)
{
	int *N;
	int i;

	N = NEW_int(depth + 1);
	for (i = 0; i <= depth; i++) {
		N[i] = nb_orbits_at_level(i);
	}
	ost << "$$" << endl;
	ost << "\\begin{array}{|r|r|}" << endl;
	ost << "\\hline" << endl;
	ost << "\\mbox{Depth} & \\mbox{Nb of orbits}\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i <= depth; i++) {
		ost << i << " & " << N[i] << "\\\\" << endl;
		ost << "\\hline" << endl;
	}
	ost << "\\end{array}" << endl;
	ost << "$$" << endl;

	ost << endl;
	FREE_int(N);

}

void poset_classification::report_orbits_summary(std::ostream &ost, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

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
	char str[1000];
	poset_orbit_node *O;
	longinteger_object stab_order, orbit_length;
	schreier_vector *Schreier_vector;
	latex_interface L;


	rep = NEW_lint(depth + 1);

	if (f_vv) {
		cout << "poset_classification::report_orbits_summary printing orbit representative" << endl;
	}

	cnt = 0;
	for (level = 0; level <= depth; level++) {

		if (f_vv) {
			cout << "poset_classification::report_orbits_summary printing orbit representative at level " << level << endl;
		}

		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary printing orbit representative at level " << level << " orbit " << i << endl;
			}

			get_set_by_level(level, i, rep);

			lint_vec_print_to_str_naked(str, rep, level);

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary set: '" << str << "'" << endl;
			}

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary before get_orbit_length_and_stabilizer_order" << endl;
			}
			get_orbit_length_and_stabilizer_order(i, level,
				stab_order, orbit_length);

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary after get_orbit_length_and_stabilizer_order" << endl;
			}

			//stab_order.print_to_string(str);

			//orbit_length.print_to_string(str);

			O = get_node_ij(level, i);

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary after get_node_ij" << endl;
			}

			if (f_vv) {
				cout << "poset_classification::report_orbits_summary before O->get_Schreier_vector" << endl;
			}
			Schreier_vector = O->get_Schreier_vector();

			if (level < depth) {
				if (Schreier_vector == NULL) {
					cout << "poset_classification::report_orbits_summary Schreier_vector == NULL" << endl;
					exit(1);
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary level < depth; level=" << level << endl;
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary before O->get_nb_of_live_points" << endl;
				}
				nb_live_pts = O->get_nb_of_live_points();
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary after O->get_nb_of_live_points" << endl;
				}
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary before O->get_nb_of_extensions" << endl;
				}
				nb_extensions = O->get_nb_of_extensions();
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary after O->get_nb_of_extensions" << endl;
				}
				//nbo = O->get_nb_of_orbits_under_stabilizer();
				if (Schreier_vector->f_has_local_generators) {
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary before Schreier_vector->local_gens->len" << endl;
					}
					nbg = Schreier_vector->local_gens->len;
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary after Schreier_vector->local_gens->len" << endl;
					}
				}
				else {
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary before O->get_nb_strong_generators" << endl;
					}
					nbg = O->get_nb_strong_generators();
					if (f_vv) {
						cout << "poset_classification::report_orbits_summary after O->get_nb_strong_generators" << endl;
					}
				}
			}
			else {
				if (f_vv) {
					cout << "poset_classification::report_orbits_summary level < depth is false" << endl;
				}
				nb_live_pts = -1;
				nb_extensions = -1;
				//nbo = -1;
				nbg = O->get_nb_strong_generators();
			}
			if (f_vv) {
				cout << "poset_classification::report_orbits_summary nb_live_pts=" << nb_live_pts
					<< " nb_extensions=" << nb_extensions
					<< " nbg=" << nbg << endl;
			}

			ost << cnt << " & " << level << " & " << i
					<< " & $\\{$ " << str << " $\\}$ & ("
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


void poset_classification::report_poset_of_orbits(std::ostream &ost)
{


	string fname_base;
	string fname_poset;
	string fname_out_base;

	draw_poset_fname_base_poset_lvl(fname_base, depth);
	draw_poset_fname_poset(fname_poset, depth);
	draw_poset_fname_base_poset_lvl(fname_out_base, depth);

	fname_out_base.append("_draw");

	string cmd;


	if (The_Orbiter_session->f_orbiter_path) {

		cmd.assign(The_Orbiter_session->orbiter_path);

	}
	else {
		cout << "poset_classification::report_poset_of_orbits "
				"We need -orbiter_path to be set" << endl;
		exit(1);

	}

	cmd.append("/orbiter.out -v 3 -draw_layered_graph ");
	cmd.append(fname_poset);

	char str[1000];

	if (!Control->f_draw_options) {
		cout << "poset_classification::report_poset_of_orbits "
				"We need -draw_options to be set in -poset_classification_control" << endl;
		exit(1);
	}

	sprintf(str, " -xin %d -yin %d -xout %d -yout %d -rad %d",
			Control->draw_options->xin,
			Control->draw_options->yin,
			Control->draw_options->xout,
			Control->draw_options->yout,
			Control->draw_options->rad);
	cmd.append(str);

	if (Control->draw_options->f_y_stretch) {
		sprintf(str, " -y_stretch %lf ", Control->draw_options->y_stretch);
		cmd.append(str);
	}

	if (Control->draw_options->f_line_width) {
		sprintf(str, " -line_width %lf ", Control->draw_options->line_width);
		cmd.append(str);
	}
	if (Control->draw_options->f_spanning_tree) {
		sprintf(str, " -spanning_tree ");
		cmd.append(str);
	}

	cout << "executing: " << cmd << endl;
	system(cmd.c_str());

	cmd.assign("mpost -tex=latex ");
	cmd.append(fname_out_base);
	cmd.append(".mp");
	cout << "executing: " << cmd << endl;
	system(cmd.c_str());


	ost << "\\input " << fname_out_base << ".tex" << endl;
	//ost << "\\includegraphics[width=160mm]{" << fname_mp << ".1}\\\\" << endl;

}


void poset_classification::report_orbit(int level, int orbit_at_level, std::ostream &ost)
{
	int nb_orbits;
	int nb_gens;
	int nb_extensions;
	poset_orbit_node *O;
	longinteger_object stab_order, orbit_length;
	char str[1000];
	long int *rep = NULL;
	schreier_vector *Schreier_vector;
	latex_interface L;

	rep = NEW_lint(depth + 1);


	nb_orbits = nb_orbits_at_level(level);

	O = get_node_ij(level, orbit_at_level);

	ost << "\\subsection*{Orbit " << orbit_at_level
			<< " / " << nb_orbits << " at Level " << level << "}" << endl;


	ost << "Node number: " << O->get_node() << "\\\\" << endl;




	strong_generators *gens;

	get_stabilizer_generators(gens,
			level, orbit_at_level, Control->verbose_level);
	get_orbit_length_and_stabilizer_order(orbit_at_level, level,
		stab_order, orbit_length);

	stab_order.print_to_string(str);

	//orbit_length.print_to_string(str);

	Schreier_vector = O->get_Schreier_vector();


	get_set_by_level(level, orbit_at_level, rep);


	// print the set and stabilizer order:


	ost << "$$" << endl;
	L.lint_set_print_tex(ost, rep, level);
	ost << "_{";
	ost << str;
	ost << "}";
	ost << "$$" << endl;


	// print strong generators for the stabilizer:

	Poset->A2->latex_point_set(ost, rep, level, 0 /* verbose_level*/);


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
}


}}
