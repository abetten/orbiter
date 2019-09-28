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
			} else {
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
				} else {
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

void poset_classification::report(ostream &ost)
{
	int i;
	int *N;



	ost << "Poset classification up to depth " << depth << "\\\\" << endl;

	ost << endl;
	ost << "\\section{The orbits}" << endl;
	ost << endl;


	ost << "\\subsection{Number of orbits at depth}" << endl;
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
	ost << "\\subsection{Orbit representatives: overview}" << endl;
	ost << endl;

	ost << "N = node\\\\" << endl;
	ost << "D = depth or level\\\\" << endl;
	ost << "O = orbit with a level\\\\" << endl;
	ost << "Rep = orbit representative\\\\" << endl;
	ost << "SO = (order of stabilizer, orbit length)\\\\" << endl;
	ost << "L = number of live points\\\\" << endl;
	ost << "F = number of flags\\\\" << endl;
	//ost << "FO = number of flag orbits\\\\" << endl;
	ost << "Gen = number of generators for the stabilizer of the orbit rep.\\\\" << endl;
	ost << "\\begin{center}" << endl;
	ost << "\\begin{longtable}{|r|r|r|p{3cm}|r|r|r|r|}" << endl;
	ost << "\\caption{Orbit Representatives}\\\\" << endl;
	ost << endl;
	ost << "\\hline N & D & O & Rep & SO "
			"& L & F & Gen\\\\ \\hline " << endl;
	ost << "\\endfirsthead" << endl;
	ost << endl;
	ost << "\\multicolumn{8}{c}%" << endl;
	ost << "{{\\bfseries \\tablename\\ \\thetable{} -- continued "
			"from previous page}} \\\\" << endl;
	ost << "\\hline N & D & O & Rep & SO "
			"& L & F & Gen\\\\ \\hline " << endl;
	ost << "\\endhead" << endl;
	ost << endl;
	ost << "\\hline \\multicolumn{8}{|r|}{{Continued on next page}} "
			"\\\\ \\hline" << endl;
	ost << "\\endfoot" << endl;
	ost << endl;
	ost << "\\hline \\hline" << endl;
	ost << "\\endlastfoot" << endl;

	int level, nb_orbits, cnt, nb_live_pts, nb_extensions, nbo, nbg;
	int *rep = NULL;
	char str[1000];
	poset_orbit_node *O;
	longinteger_object stab_order, orbit_length;
	schreier_vector *Schreier_vector;
	latex_interface L;


	rep = NEW_int(depth + 1);

	cout << "printing orbit representative" << endl;

	cnt = 0;
	for (level = 0; level <= depth; level++) {

		cout << "printing orbit representative at level " << level << endl;

		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {

			cout << "printing orbit representative at level " << level << " orbit " << i << endl;

			get_set_by_level(level, i, rep);

			int_vec_print_to_str_naked(str, rep, level);

			cout << "set: '" << str << "'" << endl;

			get_orbit_length_and_stabilizer_order(i, level,
				stab_order, orbit_length);

			cout << "after get_orbit_length_and_stabilizer_order" << endl;

			//stab_order.print_to_string(str);

			//orbit_length.print_to_string(str);

			O = get_node_ij(level, i);

			cout << "after get_node_ij" << endl;

			Schreier_vector = O->Schreier_vector;

			if (level < depth) {
				cout << "level < depth" << endl;
				nb_live_pts = O->get_nb_of_live_points();
				nb_extensions = O->nb_extensions;
				//nbo = O->get_nb_of_orbits_under_stabilizer();
				if (Schreier_vector->f_has_local_generators) {
					nbg = Schreier_vector->local_gens->len;
				}
				else {
					nbg = O->nb_strong_generators;
				}
			}
			else {
				cout << "level < depth is false" << endl;
				nb_live_pts = -1;
				nb_extensions = -1;
				//nbo = -1;
				nbg = O->nb_strong_generators;
			}
			cout << "nb_live_pts=" << nb_live_pts
					<< " nb_extensions=" << nb_extensions
					<< " nbg=" << nbg << endl;

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
	ost << "\\section{The poset of orbits}" << endl;
	ost << endl;


	char fname_base[1000];
	char cmd[10000];

	draw_poset_fname_base_poset_lvl(fname_base, depth);

	sprintf(cmd, "cp %s.layered_graph ./poset.layered_graph", fname_base);
	cout << "executing: " << cmd << endl;
	system(cmd);




	sprintf(cmd, "%s/layered_graph_main.out -v 2 "
		"-file poset.layered_graph "
		"-xin 1000000 -yin 1000000 "
		"-xout 1000000 -yout 1000000 "
		"-y_stretch 0.75 "
		"-rad 20000 "
		//"-nodes_empty "
		"-corners "
		//"-embedded "
		"-line_width 0.30 "
		"-spanning_tree",
		tools_path);
	cout << "executing: " << cmd << endl;
	system(cmd);

	sprintf(cmd, "mpost -tex=latex poset_draw_tree.mp");
	cout << "executing: " << cmd << endl;
	system(cmd);


	//ost << "\\input " << fname_tex << endl;
	ost << "\\includegraphics[width=160mm]{poset_draw_tree.1}\\" << endl;





	ost << endl;
	ost << "\\section{Stabilizers and Schreier trees}" << endl;
	ost << endl;

	int orbit_at_level, j, nb_gens;

	cnt = 0;
	for (level = 0; level <= depth; level++) {


		ost << endl;
		ost << "\\subsection{Stabilizers and Schreier trees "
				"at level " << level << "}" << endl;
		ost << endl;


		nb_orbits = nb_orbits_at_level(level);
		for (orbit_at_level = 0;
				orbit_at_level < nb_orbits;
				orbit_at_level++) {

			char fname_mask_base[1000];
			O = get_node_ij(level, orbit_at_level);

			create_shallow_schreier_tree_fname_mask_base(
					fname_mask_base, O->node);
			//create_schreier_tree_fname_mask_base(
			//fname_mask_base, O->node);

			strong_generators *gens;

			get_stabilizer_generators(gens,
					level, orbit_at_level, verbose_level);
			get_orbit_length_and_stabilizer_order(orbit_at_level, level,
				stab_order, orbit_length);

			stab_order.print_to_string(str);

			//orbit_length.print_to_string(str);

			Schreier_vector = O->Schreier_vector;


			ost << "\\subsection*{Node " << O->node << " at Level "
					<< level << " Orbit " << orbit_at_level
					<< " / " << nb_orbits << "}" << endl;

			get_set_by_level(level, orbit_at_level, rep);

			ost << "$$" << endl;
			L.int_set_print_tex(ost, rep, level);
			ost << "_{";
			ost << str;
			ost << "}";
			ost << "$$" << endl;

			ost << "{\\small\\arraycolsep=2pt" << endl;
			gens->print_generators_tex(ost);
			ost << "}" << endl;

			nb_gens = gens->gens->len;

			nb_extensions = O->nb_extensions;
			//ost << "There are " << nbo << " orbits\\\\" << endl;
			ost << "There are " << nb_extensions
					<< " extensions\\\\" << endl;
			ost << "Number of generators " << O->nb_strong_generators
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


				for (j = 0; j < nb_orbits_sv; j++) {

					//char fname_base[1000];
					char fname_layered_graph[1000];
					char fname_tex[1000];
					char fname_mp[1000];
					char fname_1[1000];

					sprintf(fname_base, fname_mask_base, j);
					sprintf(fname_layered_graph, "%s.layered_graph",
							fname_base);
					sprintf(fname_tex, "%s_draw_tree.tex", fname_base);
					sprintf(fname_mp, "%s_draw_tree.mp", fname_base);
					sprintf(fname_1, "%s_draw_tree.1", fname_base);

					if (!f_has_tools_path) {
						cout << "please set tools path using "
								"-tools_path <tools_path>" << endl;
						exit(1);
					}
					sprintf(cmd, "%s/layered_graph_main.out -v 2 "
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
						tools_path, fname_layered_graph);
					cout << "executing: " << cmd << endl;
					system(cmd);

					sprintf(cmd, "mpost %s", fname_mp);
					cout << "executing: " << cmd << endl;
					system(cmd);

					ost << "\\subsubsection*{Node " << O->node << " at Level "
							<< level << " Orbit " << orbit_at_level
							<< " / " << nb_orbits
							<< " Tree " << j << " / " << nb_orbits_sv << "}" << endl;

					nbo = Schreier_vector->number_of_orbits;
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
				FREE_int(orbit_reps);
				FREE_int(orbit_length);
				FREE_int(total_depth);
			}
			FREE_OBJECT(gens);
		}
	}
	FREE_int(rep);


}



}}
