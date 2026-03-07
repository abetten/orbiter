/*
 * pc_convert_data_structure.cpp
 *
 *  Created on: Feb 27, 2026
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



pc_convert_data_structure::pc_convert_data_structure()
{
	Record_birth();

	PC = NULL;


}

pc_convert_data_structure::~pc_convert_data_structure()
{
	Record_death();
}

void pc_convert_data_structure::init(
		poset_classification *PC,
		int verbose_level)
{
	pc_convert_data_structure::PC = PC;

}


void pc_convert_data_structure::make_flag_orbits_on_relations(
		int depth,
		std::string &fname_prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v5 = (verbose_level >= 5);
	int nb_layers;
	int *Nb_elements;
	int *Fst;
	int *Nb_orbits;
	int **Fst_element_per_orbit;
	int **Orbit_len;
	int i, j, lvl, po, po2, so, n1, n2, ol1, ol2, el1, el2, h;
	long int *set;
	long int *set1;
	long int *set2;
	int f_contained;
	//longinteger_domain D;
	other::data_structures::sorting Sorting;

	if (f_v) {
		cout << "pc_convert_data_structure::make_flag_orbits_on_relations" << endl;
	}
	set = NEW_lint(depth + 1);
	set1 = NEW_lint(depth + 1);
	set2 = NEW_lint(depth + 1);
	nb_layers = depth + 1;
	Nb_elements = NEW_int(nb_layers);
	Nb_orbits = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers + 1);
	Fst_element_per_orbit = NEW_pint(nb_layers);
	Orbit_len = NEW_pint(nb_layers);

	Fst[0] = 0;
	for (i = 0; i <= depth; i++) {
		Nb_orbits[i] = PC->get_Poo()->nb_orbits_at_level(i);
		Fst_element_per_orbit[i] = NEW_int(Nb_orbits[i] + 1);
		Orbit_len[i] = NEW_int(Nb_orbits[i]);
		Nb_elements[i] = 0;

		Fst_element_per_orbit[i][0] = 0;
		for (j = 0; j < Nb_orbits[i]; j++) {
			Orbit_len[i][j] = PC->get_Poo()->orbit_length_as_int(j, i);
			Nb_elements[i] += Orbit_len[i][j];
			Fst_element_per_orbit[i][j + 1] =
					Fst_element_per_orbit[i][j] + Orbit_len[i][j];
		}
		Fst[i + 1] = Fst[i] + Nb_elements[i];
	}

	for (lvl = 0; lvl <= depth; lvl++) {
		string fname;
		other::orbiter_kernel_system::file_io Fio;

		fname = fname_prefix + "_depth_" + std::to_string(lvl) + "_orbit_lengths.csv";

		string label;

		label.assign("Orbit_length");
		Fio.Csv_file_support->int_vec_write_csv(
				Orbit_len[lvl], Nb_orbits[lvl],
			fname, label);

		cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
					"adding edges lvl=" << lvl << " / " << depth << endl;
			}
		//f = 0;

		int *F;
		int flag_orbit_idx;
		string fname;

		if (f_vv) {
			cout << "pc_convert_data_structure::make_flag_orbits_on_relations allocating F" << endl;
		}
		F = NEW_int(Nb_elements[lvl] * Nb_elements[lvl + 1]);
		Int_vec_zero(F, Nb_elements[lvl] * Nb_elements[lvl + 1]);

		fname = fname_prefix + "_depth_" + std::to_string(lvl) + ".csv";

		flag_orbit_idx = 1;
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {

			if (f_vv) {
				cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
						"adding edges lvl=" << lvl
						<< " po=" << po << " / " << PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " Fst_element_per_orbit[lvl][po]="
						<< Fst_element_per_orbit[lvl][po] << endl;
			}

			ol1 = Orbit_len[lvl][po];
			//
			n1 = PC->get_Poo()->first_node_at_level(lvl) + po;


			int *Down_orbits;
			int nb_down_orbits;

			Down_orbits = NEW_int(PC->get_Poo()->node_get_nb_of_extensions(n1));
			nb_down_orbits = 0;

			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n1); so++) {

				if (f_vv) {
					cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
							"adding edges lvl=" << lvl
							<< " po=" << po << " / " << PC->get_Poo()->nb_orbits_at_level(lvl)
							<< " so=" << so << " / " << PC->get_Poo()->node_get_nb_of_extensions(n1)
							<< endl;
				}


				extension *E = PC->get_Poo()->get_node(n1)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n2 = E->get_data();

					Down_orbits[nb_down_orbits++] = n2;
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n1 << "/" << so << ") "
					//"-> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						PC->print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n2 = E0->get_data();
					Down_orbits[nb_down_orbits++] = n2;
				}

			} // next so


			if (f_vv) {
				cout << "pc_convert_data_structure::make_flag_orbits_on_relations adding edges "
						"lvl=" << lvl
						<< " po=" << po << " / " << PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " so=" << so << " / " << PC->get_Poo()->node_get_nb_of_extensions(n1)
						<< " downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			Sorting.int_vec_sort_and_remove_duplicates(Down_orbits, nb_down_orbits);
			if (f_vv) {
				cout << "pc_convert_data_structure::make_flag_orbits_on_relations adding edges "
						"lvl=" << lvl << " po=" << po
						<< " so=" << so << " unique downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			for (h = 0; h < nb_down_orbits; h++, flag_orbit_idx++) {
				n2 = Down_orbits[h];
				po2 = n2 - PC->get_Poo()->first_node_at_level(lvl + 1);
				ol2 = Orbit_len[lvl + 1][po2];
				if (f_v5) {
					cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
							"adding edges lvl=" << lvl
							<< " po=" << po << " / " << PC->get_Poo()->nb_orbits_at_level(lvl)
							<< " so=" << so << " / " << PC->get_Poo()->node_get_nb_of_extensions(n1)
							<< " downorbit = " << h << " / " << nb_down_orbits
							<< " n1=" << n1 << " n2=" << n2
							<< " po2=" << po2
							<< " ol1=" << ol1 << " ol2=" << ol2
							<< " Fst_element_per_orbit[lvl][po]="
							<< Fst_element_per_orbit[lvl][po]
							<< " Fst_element_per_orbit[lvl + 1][po2]="
							<< Fst_element_per_orbit[lvl + 1][po2] << endl;
				}
				for (el1 = 0; el1 < ol1; el1++) {
					if (f_v5) {
						cout << "unrank " << lvl << ", " << po
								<< ", " << el1 << endl;
					}
					PC->get_Poo()->orbit_element_unrank(
							lvl, po, el1, set1,
							0 /* verbose_level */);
					if (f_v5) {
						cout << "set1=";
						Lint_vec_print(cout, set1, lvl);
						cout << endl;
					}


					for (el2 = 0; el2 < ol2; el2++) {
						if (f_v5) {
							cout << "unrank " << lvl + 1 << ", "
									<< po2 << ", " << el2 << endl;
						}
						PC->get_Poo()->orbit_element_unrank(
								lvl + 1, po2, el2, set2,
								0 /* verbose_level */);
						if (f_v5) {
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}

						if (f_v5) {
							cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
									"adding edges lvl=" << lvl
									<< " po=" << po << " so=" << so
									<< " downorbit = " << h << " / "
									<< nb_down_orbits << " n1=" << n1
									<< " n2=" << n2 << " po2=" << po2
									<< " ol1=" << ol1 << " ol2=" << ol2
									<< " el1=" << el1 << " el2=" << el2
									<< endl;
							cout << "set1=";
							Lint_vec_print(cout, set1, lvl);
							cout << endl;
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}


						Lint_vec_copy(set1, set, lvl);

						//f_contained = int_vec_sort_and_test_if_contained(
						// set, lvl, set2, lvl + 1);
						f_contained = PC->get_poset()->is_contained(
								set, lvl, set2, lvl + 1,
								0 /* verbose_level*/);


						if (f_contained) {
							if (f_v5) {
								cout << "is contained" << endl;
							}

#if 0
							LG->add_edge(lvl,
								Fst_element_per_orbit[lvl][po] + el1,
								lvl + 1,
								Fst_element_per_orbit[lvl + 1][po2] + el2,
								0 /*verbose_level*/);
#else
							F[(Fst_element_per_orbit[lvl][po] + el1) * Nb_elements[lvl + 1] + Fst_element_per_orbit[lvl + 1][po2] + el2] = flag_orbit_idx;
#endif
						}
						else {
							if (f_v5) {
								cout << "is NOT contained" << endl;
							}
						}

					} // next el2
				} // next el1
			} // next h


			FREE_int(Down_orbits);

		} // po

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->int_matrix_write_csv(
				fname,
				F, Nb_elements[lvl], Nb_elements[lvl + 1]);
		FREE_int(F);

		cout << "pc_convert_data_structure::make_flag_orbits_on_relations "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;

	} // lvl






	FREE_lint(set);
	FREE_lint(set1);
	FREE_lint(set2);
	FREE_int(Nb_elements);
	FREE_int(Nb_orbits);
	FREE_int(Fst);
	for (i = 0; i <= depth; i++) {
		FREE_int(Fst_element_per_orbit[i]);
	}
	FREE_pint(Fst_element_per_orbit);
	for (i = 0; i <= depth; i++) {
		FREE_int(Orbit_len[i]);
	}
	FREE_pint(Orbit_len);
	if (f_v) {
		cout << "pc_convert_data_structure::make_flag_orbits_on_relations done" << endl;
	}
}


void pc_convert_data_structure::make_factor_poset(
		int depth, //combinatorics::graph_theory::layered_graph *&LG,
		int data1, double x_stretch,
		layer1_foundations::combinatorics::graph_theory::factor_poset *&Factor_poset,
		int type_of_poset,
		int f_poset_with_horizontal_lines,
		int verbose_level)
// Draws the full poset: each element of each orbit is drawn.
// The orbits are indicated by grouping the elements closer together.
// Uses int_vec_sort_and_test_if_contained to test containment relation.
// This is only good for actions on sets, not for actions on subspaces
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset" << endl;
	}


	int i, j;


	int nb_layers;
	int *Nb_orbits;
	int **Orbit_len;


	nb_layers = depth + 1;
	Nb_orbits = NEW_int(nb_layers);
	Orbit_len = NEW_pint(nb_layers);


	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"depth = " << depth << endl;
	}

	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"computing Nb_orbits" << endl;
	}

	for (i = 0; i <= depth; i++) {
		Nb_orbits[i] = PC->get_Poo()->nb_orbits_at_level(i);
	}


	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"Nb_orbits = ";
		Int_vec_print(cout, Nb_orbits, nb_layers);
		cout << endl;
	}

	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"computing Orbit_len" << endl;
	}
	for (i = 0; i <= depth; i++) {
		Orbit_len[i] = NEW_int(Nb_orbits[i]);
		for (j = 0; j < Nb_orbits[i]; j++) {
			Orbit_len[i][j] = PC->get_Poo()->orbit_length_as_int(j, i);
		}
		if (f_v) {
			cout << "pc_convert_data_structure::make_factor_poset "
					"Orbit_len[" << i << "] = ";
			Int_vec_print(cout, Orbit_len[i], Nb_orbits[i]);
			cout << endl;
		}
	}

	other::data_structures::set_of_sets *All_orbits;

	{
		int *Nb_orbits2;
		int nb_orbits_total;

		if (f_v) {
			cout << "pc_convert_data_structure::make_factor_poset "
					"before get_all_orbits" << endl;
		}
		PC->get_Poo()->get_all_orbits(
				All_orbits,
				Nb_orbits2,
				nb_orbits_total,
				verbose_level);
		if (f_v) {
			cout << "pc_convert_data_structure::make_factor_poset "
					"after get_all_orbits" << endl;
		}

		FREE_int(Nb_orbits2);
	}


	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset All_orbits=" << endl;
		All_orbits->print_table();
	}



	//int lvl, po, po2, so, n1, n2, ol1, ol2, el1, el2, h;

	Factor_poset = NEW_OBJECT(layer1_foundations::combinatorics::graph_theory::factor_poset);


	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"before Factor_poset->init" << endl;
	}
	Factor_poset->init(
			depth,
			Nb_orbits,
			Orbit_len,
			data1,
			x_stretch,
			verbose_level);
	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"after Factor_poset->init" << endl;
	}



	if (f_vv) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"before make_full_poset_graph_edges" << endl;
	}
	make_full_poset_graph_edges(
			Factor_poset,
			All_orbits,
			type_of_poset,
			f_poset_with_horizontal_lines,
			verbose_level - 2);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"after make_full_poset_graph_edges" << endl;
	}





	if (f_vv) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"before make_full_poset_graph_vertex_labels" << endl;
	}
	make_full_poset_graph_vertex_labels(
			Factor_poset,
			All_orbits,
			verbose_level - 2);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_factor_poset "
				"after make_full_poset_graph_vertex_labels" << endl;
	}


	FREE_OBJECT(All_orbits);

	if (f_v) {
		cout << "pc_convert_data_structure::make_factor_poset done" << endl;
	}
}


void pc_convert_data_structure::make_full_poset_graph_edges(
		layer1_foundations::combinatorics::graph_theory::factor_poset *Factor_poset,
		other::data_structures::set_of_sets *All_orbits,
		int type_of_poset,
		int f_poset_with_horizontal_lines,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "pc_convert_data_structure::make_full_poset_graph_edges" << endl;
	}


	int depth;

	depth = Factor_poset->nb_layers - 1;



	long int *set;
	long int *set1;
	long int *set2;
	//long int *set3;
	int f_contained;
	other::data_structures::sorting Sorting;


	// preparing the data structure:


	set = NEW_lint(depth + 1);
	//set1 = NEW_lint(depth + 1);
	//set2 = NEW_lint(depth + 1);
	//set3 = NEW_lint(depth + 1);

	// making edges:


	int lvl, po, po2, so, n1, n2, ol1, ol2, el1, el2, h;


	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_full_poset_graph_edges "
					"adding edges lvl=" << lvl << " / " << depth << endl;
		}
		//f = 0;
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {

			if (f_vv) {
				cout << "pc_convert_data_structure::make_full_poset_graph_edges "
						"adding edges lvl=" << lvl
						<< " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " Fst_element_per_orbit[lvl][po]="
						<< Factor_poset->Fst_element_per_orbit[lvl][po] << endl;
			}

			ol1 = Factor_poset->Orbit_len[lvl][po];
			//
			n1 = PC->get_Poo()->first_node_at_level(lvl) + po;


			if (f_poset_with_horizontal_lines) {

				Factor_poset->LG->add_edge(lvl,
					Factor_poset->Fst_element_per_orbit[lvl][po] + 0,
					lvl,
					Factor_poset->Fst_element_per_orbit[lvl][po] + ol1 - 1,
					0, // edge_color
					0 /*verbose_level*/);

			}


			int *Down_orbits;
			int nb_down_orbits;

			Down_orbits = NEW_int(PC->get_Poo()->node_get_nb_of_extensions(n1));
			nb_down_orbits = 0;

			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n1); so++) {

				if (f_vv) {
					cout << "poset_classification::make_full_poset_graph_edges "
							"adding edges lvl=" << lvl
							<< " po=" << po << " so=" << so << endl;
				}


				extension *E = PC->get_Poo()->get_node(n1)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n2 = E->get_data();

					Down_orbits[nb_down_orbits++] = n2;
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n1 << "/" << so << ") "
					//"-> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						PC->print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n2 = E0->get_data();
					Down_orbits[nb_down_orbits++] = n2;
				}

			} // next so


			if (f_vv) {
				cout << "pc_convert_data_structure::make_full_poset_graph_edges adding edges "
						"lvl=" << lvl << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " so=" << so << " downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			Sorting.int_vec_sort_and_remove_duplicates(Down_orbits, nb_down_orbits);
			if (f_vv) {
				cout << "pc_convert_data_structure::make_full_poset_graph_edges adding edges "
						"lvl=" << lvl << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " so=" << so << " unique downorbits = ";
				Int_vec_print(cout, Down_orbits, nb_down_orbits);
				cout << endl;
			}

			for (h = 0; h < nb_down_orbits; h++) {
				n2 = Down_orbits[h];
				po2 = n2 - PC->get_Poo()->first_node_at_level(lvl + 1);
				ol2 = Factor_poset->Orbit_len[lvl + 1][po2];
				if (f_vv) {
					cout << "pc_convert_data_structure::make_full_poset_graph_edges "
							"adding edges lvl=" << lvl << " po=" << po << " / "
							<< PC->get_Poo()->nb_orbits_at_level(lvl)
							<< " so=" << so << " downorbit = " << h
							<< " / " << nb_down_orbits << " n1=" << n1
							<< " n2=" << n2 << " po2=" << po2
							<< " ol1=" << ol1 << " ol2=" << ol2
							<< " Fst_element_per_orbit[lvl][po]="
							<< Factor_poset->Fst_element_per_orbit[lvl][po]
							<< " Fst_element_per_orbit[lvl + 1][po2]="
							<< Factor_poset->Fst_element_per_orbit[lvl + 1][po2] << endl;
				}
				for (el1 = 0; el1 < ol1; el1++) {
					if (f_vvv) {
						cout << "pc_convert_data_structure::make_full_poset_graph_edges unrank " << lvl << ", " << po
								<< ", " << el1 << endl;
					}

					if (type_of_poset == 1) {

						// Asup

						if (el1 != 0) {
							continue;
						}
					}


					set1 = All_orbits->Sets[n1] + el1 * lvl;

#if 0
					orbit_element_unrank(lvl, po, el1, set1,
							0 /* verbose_level */);
#endif


					if (f_vvv) {
						cout << "pc_convert_data_structure::make_full_poset_graph_edges set1=";
						Lint_vec_print(cout, set1, lvl);
						cout << endl;
					}


					for (el2 = 0; el2 < ol2; el2++) {
						if (f_vvv) {
							cout << "pc_convert_data_structure::make_full_poset_graph_edges unrank " << lvl + 1 << ", "
									<< po2 << ", " << el2 << endl;
						}

						if (type_of_poset == 2) {

							// Ainf

							if (el2 != 0) {
								continue;
							}
						}

						set2 = All_orbits->Sets[n2] + el2 * (lvl + 1);

#if 0
						orbit_element_unrank(lvl + 1, po2, el2, set3,
								0 /* verbose_level */);

						if (Lint_vec_compare(set2, set3, lvl + 1)) {
							cout << "pc_convert_data_structure::make_full_poset_graph_edges unrank not the same set" << endl;
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
							Lint_vec_print(cout, set3, lvl + 1);
							cout << endl;
							cout << "poset_classification::make_full_poset_graph "
									"adding edges lvl=" << lvl
									<< " po=" << po << " / "
									<< nb_orbits_at_level(lvl)
									<< " so=" << so
									<< " downorbit = " << h << " / "
									<< nb_down_orbits << " n1=" << n1
									<< " n2=" << n2 << " po2=" << po2
									<< " ol1=" << ol1 << " ol2=" << ol2
									<< " el1=" << el1 << " el2=" << el2
									<< endl;
							exit(1);
						}
#endif

						if (f_vvv) {
							cout << "pc_convert_data_structure::make_full_poset_graph_edges set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}

						if (f_vvv) {
							cout << "pc_convert_data_structure::make_full_poset_graph_edges "
									"adding edges lvl=" << lvl
									<< " po=" << po << " / "
									<< PC->get_Poo()->nb_orbits_at_level(lvl)
									<< " so=" << so
									<< " downorbit = " << h << " / "
									<< nb_down_orbits << " n1=" << n1
									<< " n2=" << n2 << " po2=" << po2
									<< " ol1=" << ol1 << " ol2=" << ol2
									<< " el1=" << el1 << " el2=" << el2
									<< endl;
							cout << "set1=";
							Lint_vec_print(cout, set1, lvl);
							cout << endl;
							cout << "set2=";
							Lint_vec_print(cout, set2, lvl + 1);
							cout << endl;
						}


						Lint_vec_copy(set1, set, lvl);

						//f_contained = int_vec_sort_and_test_if_contained(
						// set, lvl, set2, lvl + 1);
						f_contained = PC->get_poset()->is_contained(
								set, lvl, set2, lvl + 1,
								0 /* verbose_level*/);


						if (f_contained) {
							if (f_vvv) {
								cout << "pc_convert_data_structure::make_full_poset_graph_edges is contained" << endl;
							}
							Factor_poset->LG->add_edge(lvl,
									Factor_poset->Fst_element_per_orbit[lvl][po] + el1,
								lvl + 1,
								Factor_poset->Fst_element_per_orbit[lvl + 1][po2] + el2,
								1, // edge_color
								0 /*verbose_level*/);
						}
						else {
							if (f_vvv) {
								cout << "pc_convert_data_structure::make_full_poset_graph_edges is NOT contained" << endl;
							}
						}

					} // next el2
				} // next el1
			} // next h


			FREE_int(Down_orbits);

		} // po
	} // lvl

	FREE_lint(set);
	//FREE_lint(set1);
	//FREE_lint(set2);
	//FREE_lint(set3);



	if (f_v) {
		cout << "pc_convert_data_structure::make_full_poset_graph_edges done" << endl;
	}

}

void pc_convert_data_structure::make_full_poset_graph_vertex_labels(
		layer1_foundations::combinatorics::graph_theory::factor_poset *Factor_poset,
		other::data_structures::set_of_sets *All_orbits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "pc_convert_data_structure::make_full_poset_graph_vertex_labels" << endl;
	}


	// making vertex labels:

	int depth;

	depth = Factor_poset->nb_layers - 1;

	long int *set1;

	int lvl, po;
	int ol1, n1;
	int el1;

	//set1 = NEW_lint(depth + 1);

	if (f_vv) {
		cout << "pc_convert_data_structure::make_full_poset_graph_vertex_labels "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_full_poset_graph_vertex_labels "
					"now making vertex labels lvl " << lvl
					<< " / " << depth << endl;
		}
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {

			ol1 = Factor_poset->Orbit_len[lvl][po];
			//
			n1 = PC->get_Poo()->first_node_at_level(lvl) + po;

			if (f_vv) {
				cout << "pc_convert_data_structure::make_full_poset_graph_vertex_labels "
						"now making vertex labels lvl " << lvl
						<< " / " << depth << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl)
						<< " ol1=" << ol1 << endl;
			}

			for (el1 = 0; el1 < ol1; el1++) {

				set1 = All_orbits->Sets[n1] + el1 * lvl;

				if (f_vv) {
					cout << "unrank " << lvl << ", "
							<< po << ", " << el1 << endl;
				}
#if 0
				orbit_element_unrank(lvl, po, el1, set1,
						0 /* verbose_level */);
#endif
				if (f_vv) {
					cout << "set1=";
					Lint_vec_print(cout, set1, lvl);
					cout << endl;
				}

				Factor_poset->LG->add_node_vec_data(
						lvl,
						Factor_poset->Fst_element_per_orbit[lvl][po] + el1,
						set1, lvl,
						0 /* verbose_level */);
			}


		}
	}

	//FREE_lint(set1);


	if (f_v) {
		cout << "pc_convert_data_structure::make_full_poset_graph_vertex_labels done" << endl;
	}
}

void pc_convert_data_structure::make_auxiliary_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG, int data1, int verbose_level)
// makes a graph of the poset of orbits with 2 * depth + 1 layers.
// The middle layers represent the flag orbits.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, lvl, po, so, n, n1, f;
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "pc_convert_data_structure::make_auxiliary_graph" << endl;
	}

	//print_fusion_nodes(depth);



	nb_layers = 2 * depth + 1;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers);

	Fst[0] = 0;
	for (i = 0; i < depth; i++) {
		Nb[2 * i] = PC->get_Poo()->nb_orbits_at_level(i);
		Fst[2 * i + 1] = Fst[2 * i] + Nb[2 * i];
		PC->get_Poo()->count_extension_nodes_at_level(i);
		Nb[2 * i + 1] = PC->get_Poo()->get_nb_extension_nodes_at_level_total(i);
		Fst[2 * i + 2] = Fst[2 * i + 1] + Nb[2 * i + 1];
	}
	Nb[2 * depth] = PC->get_Poo()->nb_orbits_at_level(depth);

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_auxiliary_graph "
				"before LG->init" << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level - 1);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_auxiliary_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level - 1);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_auxiliary_graph "
				"after LG->place" << endl;
	}
	for (lvl = 0; lvl < depth; lvl++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_auxiliary_graph "
					"adding edges "
					"lvl=" << lvl << " / " << depth << endl;
		}
		f = 0;
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {

			if (f_v3) {
				cout << "pc_convert_data_structure::make_auxiliary_graph "
						"adding edges lvl=" << lvl << " po=" << po
						<< " / " << PC->get_Poo()->nb_orbits_at_level(lvl) << endl;
			}

			//
			n = PC->get_Poo()->first_node_at_level(lvl) + po;
			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {

				if (f_v4) {
					cout << "pc_convert_data_structure::make_auxiliary_graph "
							"adding edges "
							"lvl=" << lvl << " po=" << po
							<< " so=" << so << endl;
				}
				LG->add_edge(
						2 * lvl, po, 2 * lvl + 1, f + so,
						1, // edge_color
						verbose_level - 4);

				extension *E = PC->get_Poo()->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					if (f_v4) {
						cout << "extension node" << endl;
					}
					n1 = E->get_data();
					if (f_v4) {
						cout << "n1=" << n1 << endl;
					}
					LG->add_edge(
							2 * lvl + 1, f + so, 2 * lvl + 2,
							n1 - PC->get_Poo()->first_node_at_level(lvl + 1),
							1, // edge_color
							verbose_level - 4);
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					if (f_v4) {
						cout << "fusion node" << endl;
					}
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					if (f_v4) {
						cout << "fusion (" << n << "/" << so << ") -> ("
								<< n0 << "/" << so0 << ")" << endl;
					}
					extension *E0;
					E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point "
								"to extension node" << endl;
						cout << "type = ";
						PC->print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n1 = E0->get_data();
					if (f_v4) {
						cout << "n1=" << n1
								<< " first_poset_orbit_node_node_at_level[lvl + 1] = "
								<< PC->get_Poo()->first_node_at_level(lvl + 1) << endl;
					}
					LG->add_edge(
							2 * lvl + 1, f + so, 2 * lvl + 2,
							n1 - PC->get_Poo()->first_node_at_level(lvl + 1),
							1, // edge_color
							verbose_level - 4);
				}
			}

			f += PC->get_Poo()->node_get_nb_of_extensions(n);
		}
		if (f_vv) {
			cout << "pc_convert_data_structure::make_auxiliary_graph "
					"after LG->add_edge (1)" << endl;
		}
	}


	if (f_vv) {
		cout << "pc_convert_data_structure::make_auxiliary_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		f = 0;
		if (f_vv) {
			cout << "pc_convert_data_structure::make_auxiliary_graph "
					"now making vertex "
					"labels lvl " << lvl << " / " << depth << endl;
		}
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {


			if (f_v3) {
				cout << "pc_convert_data_structure::make_auxiliary_graph "
						"now making vertex labels lvl " << lvl << " / "
						<< depth << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl) << endl;
			}


			string text1;
			string text2;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;

			n = PC->get_Poo()->first_node_at_level(lvl) + po;
			PC->get_Poo()->get_stabilizer_order(lvl, po, go);
			go.print_to_string(text1);
			if (lvl) {
				text2 = "$" + std::to_string(PC->get_Poo()->get_node(n)->get_pt()) + "_{" + text1 + "}$";
			}
			else {
				text2 = "$\\emptyset_{" + text1 + "}$";
			}

			// set label to be the automorphism group order:
			//LG->add_text(2 * lvl + 0, po, text1, 0/*verbose_level*/);

			// set label to be the pt:

			string text3;

			text3.assign(text2);
			LG->add_text(2 * lvl + 0, po, text3, 0/*verbose_level*/);


			LG->add_node_data1(
					2 * lvl + 0, po, PC->get_Poo()->get_node(n)->get_pt(),
					0/*verbose_level*/);
			if (lvl) {
				LG->add_node_data2(
						2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(
						2 * lvl + 0, po,
						PC->get_Poo()->get_node(n)->get_prev() - PC->get_Poo()->first_node_at_level(lvl - 1),
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(
						2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(
						2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {
				extension *E = PC->get_Poo()->get_node(n)->get_E(so);
				len = E->get_orbit_len();
				D.integral_division_by_int(go, len, go1, r);

				go1.print_to_string(text1);
				text2 = "$" + std::to_string(E->get_pt()) + "_{" + text1 + "}$";

				// set label to be the automorphism group order:
				//LG->add_text(2 * lvl + 1, f + so, text1, 0/*verbose_level*/);
				// set label to be the point:

				string text3;

				text3.assign(text2);
				LG->add_text(2 * lvl + 1, f + so, text3, 0/*verbose_level*/);


			}
			f += PC->get_Poo()->node_get_nb_of_extensions(n);
		}
	}
	FREE_int(Nb);
	FREE_int(Fst);

	if (f_v) {
		cout << "pc_convert_data_structure::make_auxiliary_graph done" << endl;
	}
}

void pc_convert_data_structure::make_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int f_tree, int verbose_level)
// makes a graph  of the poset of orbits with depth + 1 layers.
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int nb_layers;
	int *Nb;
	int *Fst;
	int i, lvl, po, so, n, n1;
	long int *the_set;
	//longinteger_domain D;

	if (f_v) {
		cout << "pc_convert_data_structure::make_graph f_tree=" << f_tree << endl;
	}

	//print_fusion_nodes(depth);



	nb_layers = depth + 1;
	Nb = NEW_int(nb_layers);
	Fst = NEW_int(nb_layers);

	Fst[0] = 0;
	for (i = 0; i < depth; i++) {
		Nb[i] = PC->get_Poo()->nb_orbits_at_level(i);
		Fst[i + 1] = Fst[i] + Nb[i];
	}
	Nb[depth] = PC->get_Poo()->nb_orbits_at_level(depth);

	the_set = NEW_lint(depth);


	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_graph "
				"before LG->init" << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_graph "
				"after LG->place" << endl;
	}


	// make edges:
	for (lvl = 0; lvl < depth; lvl++) {
		if (f_v) {
			cout << "pc_convert_data_structure::make_graph "
					"adding edges "
					"lvl=" << lvl << " / " << depth << endl;
		}
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {

			if (false /*f_v*/) {
				cout << "pc_convert_data_structure::make_graph "
						"adding edges "
						"lvl=" << lvl << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl) << endl;
			}

			//
			n = PC->get_Poo()->first_node_at_level(lvl) + po;
			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {

				if (false /*f_v*/) {
					cout << "pc_convert_data_structure::make_graph "
							"adding edges "
							"lvl=" << lvl << " po=" << po
							<< " so=" << so << endl;
				}
				//LG->add_edge(2 * lvl, po, 2 * lvl + 1,
				//f + so, 0 /*verbose_level*/);
				extension *E = PC->get_Poo()->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n1 = E->get_data();
					//cout << "n1=" << n1 << endl;
					LG->add_edge(lvl, po, lvl + 1,
							n1 - PC->get_Poo()->first_node_at_level(lvl + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}

				if (!f_tree) {
					if (E->get_type() == EXTENSION_TYPE_FUSION) {
						//cout << "fusion node" << endl;
						// po = data1
						// so = data2
						int n0, so0;
						n0 = E->get_data1();
						so0 = E->get_data2();
						//cout << "fusion (" << n << "/" << so << ") -> ("
						//<< n0 << "/" << so0 << ")" << endl;
						extension *E0;
						E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
						if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
							cout << "warning: fusion node does not point to "
									"extension node" << endl;
							cout << "type = ";
							PC->print_extension_type(cout, E0->get_type());
							cout << endl;
							exit(1);
						}
						n1 = E0->get_data();
						//cout << "n1=" << n1
						//<< " first_poset_orbit_node_at_level[lvl + 1] = "
						//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
						LG->add_edge(lvl, po, lvl + 1,
								n1 - PC->get_Poo()->first_node_at_level(lvl + 1),
								1, // edge_color
								0 /*verbose_level*/);
					}
				}
			}
		}
		if (f_vv) {
			cout << "pc_convert_data_structure::make_graph "
					"after LG->add_edge (1)" << endl;
		}
	}


	// create vertex labels:
	if (f_vv) {
		cout << "pc_convert_data_structure::make_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = 0; lvl <= depth; lvl++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_graph "
					"now making vertex labels "
					"lvl " << lvl << " / " << depth << endl;
		}
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {


			if (f_vv) {
				cout << "pc_convert_data_structure::make_graph "
						"now making vertex "
						"labels lvl " << lvl << " / " << depth << " po="
						<< po << " / " << PC->get_Poo()->nb_orbits_at_level(lvl) << endl;
			}


			string text;
			string text2;
			algebra::ring_theory::longinteger_object go, go1;
			int n;

			n = PC->get_Poo()->first_node_at_level(lvl) + po;


			PC->get_Poo()->get_set_by_level(lvl, po, the_set);


			PC->get_Poo()->get_stabilizer_order(lvl, po, go);
			go.print_to_string(text);
			if (lvl) {
				text2 = std::to_string(the_set[lvl - 1]);
			}
			else {
				text2 = "$\\emptyset$";
			}

			string text3;

			text3.assign(text2);
			LG->add_text(lvl, po, text3, 0/*verbose_level*/);

			// if no vector data, the text will be printed:
			//LG->add_node_vec_data(lvl, po, the_set, lvl, 0 /* verbose_level */);


			// ToDo:
			if (false /* Control->f_node_label_is_group_order */) {
				// label the node with the group order:
				LG->add_node_data1(lvl, po, go.as_int(), 0/*verbose_level*/);
			}
			// ToDo:
			else if (true /* Control->f_node_label_is_element*/) {
				// label the node with the point:
				if (lvl) {
					LG->add_node_data1(
							lvl, po, PC->get_Poo()->get_node(n)->get_pt(),
							0/*verbose_level*/);
				}
				else {
					// root node has no element
				}
			}
			else {
				LG->add_node_data1(lvl, po, n, 0/*verbose_level*/);
			}

			if (lvl) {
				LG->add_node_data2(lvl, po, lvl - 1, 0/*verbose_level*/);
				LG->add_node_data3(lvl, po,
						PC->get_Poo()->get_node(n)->get_prev() - PC->get_Poo()->first_node_at_level(lvl - 1),
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(lvl, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(lvl, po, -1, 0/*verbose_level*/);
			}
		}
	}
	FREE_int(Nb);
	FREE_int(Fst);
	FREE_lint(the_set);

	if (f_v) {
		cout << "pc_convert_data_structure::make_graph done" << endl;
	}
}

void pc_convert_data_structure::make_level_graph(
		int depth,
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int level, int verbose_level)
// makes a graph with 4 levels showing the relation between
// orbits at level 'level' and orbits at level 'level' + 1
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Nb;
	int *Fst;
	long int nb_middle;
	int i, lvl, po, so, n, n1, f, l;
	algebra::ring_theory::longinteger_domain D;
	int nb_layers = 4;
	long int *the_set;
	long int *the_set2;

	if (f_v) {
		cout << "pc_convert_data_structure::make_level_graph "
				"verbose_level=" << verbose_level << endl;
	}

	//print_fusion_nodes(depth);


	the_set = NEW_lint(depth);
	the_set2 = NEW_lint(depth);
	Nb = NEW_int(4);
	Fst = NEW_int(2);

	Fst[0] = 0;
	for (i = 0; i < level; i++) {
		Fst[0] += PC->get_Poo()->nb_orbits_at_level(i);
	}
	nb_middle = PC->get_Poo()->count_extension_nodes_at_level(level);
	Fst[1] = Fst[0] + PC->get_Poo()->nb_orbits_at_level(level);

	Nb[0] = PC->get_Poo()->nb_orbits_at_level(level);
	Nb[1] = nb_middle;
	Nb[2] = nb_middle;
	Nb[3] = PC->get_Poo()->nb_orbits_at_level(level + 1);

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_level_graph "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		cout << "Nb=";
		Int_vec_print(cout, Nb, 4);
		cout << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_level_graph "
				"after LG->init" << endl;
	}
	LG->place(verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_level_graph "
				"after LG->place" << endl;
	}
	f = 0;
	for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(level); po++) {

		if (f_vv) {
			cout << "pc_convert_data_structure::make_level_graph "
					"adding edges "
					"level=" << level << " po=" << po << " / "
					<< PC->get_Poo()->nb_orbits_at_level(level) << endl;
		}

		//
		n = PC->get_Poo()->first_node_at_level(level) + po;
		for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {

			if (false /*f_v*/) {
				cout << "pc_convert_data_structure::make_level_graph "
						"adding edges lvl=" << lvl << " po="
						<< po << " so=" << so << endl;
			}
			LG->add_edge(0, po, 1, f + so,
					1, // edge_color
					0 /*verbose_level*/);
			LG->add_edge(1, f + so, 2, f + so,
					1, // edge_color
					0 /*verbose_level*/);
			extension *E = PC->get_Poo()->get_node(n)->get_E(so);
			if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
				//cout << "extension node" << endl;
				n1 = E->get_data();
				//cout << "n1=" << n1 << endl;
				LG->add_edge(
						2, f + so, 3,
						n1 - PC->get_Poo()->first_node_at_level(level + 1),
						1, // edge_color
						0 /*verbose_level*/);
			}
			else if (E->get_type() == EXTENSION_TYPE_FUSION) {
				//cout << "fusion node" << endl;
				// po = data1
				// so = data2
				int n0, so0;
				n0 = E->get_data1();
				so0 = E->get_data2();
				//cout << "fusion (" << n << "/" << so
				//<< ") -> (" << n0 << "/" << so0 << ")" << endl;
				extension *E0;
				E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
				if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
					cout << "warning: fusion node does not point to "
							"extension node" << endl;
					cout << "type = ";
					PC->print_extension_type(cout, E0->get_type());
					cout << endl;
					exit(1);
				}
				n1 = E0->get_data();
				//cout << "n1=" << n1
				//<< " first_poset_orbit_node_at_level[lvl + 1] = "
				//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
				LG->add_edge(
						2, f + so, 3,
						n1 - PC->get_Poo()->first_node_at_level(level + 1),
						1, // edge_color
						0 /*verbose_level*/);
			}
		}

		f += PC->get_Poo()->node_get_nb_of_extensions(n);
	}
	if (f_vv) {
		cout << "pc_convert_data_structure::make_level_graph "
				"after LG->add_edge" << endl;
	}


	// creates vertex labels for orbits at level 'level' and 'level' + 1:
	if (f_vv) {
		cout << "pc_convert_data_structure::make_level_graph "
				"now making vertex labels" << endl;
	}
	for (lvl = level; lvl <= level + 1; lvl++) {
		f = 0;
		if (f_vv) {
			cout << "pc_convert_data_structure::make_level_graph "
					"now making vertex labels lvl " << lvl
					<< " / " << depth << endl;
		}

		if (lvl == level) {
			l = 0;
		}
		else {
			l = 3;
		}

		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(lvl); po++) {


			if (f_vv) {
				cout << "pc_convert_data_structure::make_level_graph "
						"now making vertex labels lvl " << lvl
						<< " / " << depth << " po=" << po << " / "
						<< PC->get_Poo()->nb_orbits_at_level(lvl) << endl;
			}


			string text;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;

			n = PC->get_Poo()->first_node_at_level(lvl) + po;
			PC->get_Poo()->get_stabilizer_order(lvl, po, go);
			go.print_to_string(text);


			string text3;

			text3.assign(text);

			LG->add_text(l, po, text3, 0/*verbose_level*/);
			LG->add_node_data1(
					l, po, PC->get_Poo()->get_node(n)->get_pt(), 0/*verbose_level*/);

			PC->get_Poo()->get_set_by_level(lvl, po, the_set);
			LG->add_node_vec_data(
					l, po, the_set, lvl, 0 /* verbose_level */);
#if 0
			if (lvl) {
				LG->add_node_data2(2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po,
						root[n].prev - first_poset_orbit_node_at_level[lvl - 1],
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
#endif

			if (lvl == level) {
				for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {
					extension *E = PC->get_Poo()->get_node(n)->get_E(so);
					len = E->get_orbit_len();
					D.integral_division_by_int(go, len, go1, r);
					go1.print_to_string(text);

					string text3;

					text3.assign(text);

					LG->add_text(1, f + so, text3, 0/*verbose_level*/);
					LG->add_text(2, f + so, text3, 0/*verbose_level*/);

					//get_set_by_level(lvl, po, the_set);
					the_set[lvl] = E->get_pt();
					LG->add_node_vec_data(
							l + 1, f + so, the_set, lvl + 1,
							0 /* verbose_level */);
					LG->set_distinguished_element_index(
							l + 1, f + so, lvl,
							0 /* verbose_level */);


					if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
						the_set[lvl] = E->get_pt();
						LG->add_node_vec_data(
								l + 2, f + so, the_set, lvl + 1,
								0 /* verbose_level */);
						LG->set_distinguished_element_index(
								l + 2, f + so, lvl,
								0 /* verbose_level */);
					}
					else if (E->get_type() == EXTENSION_TYPE_FUSION) {

						//Poset->A->element_retrieve(E->get_data(), Elt1, 0);

						PC->get_poset()->A2->map_a_set_based_on_hdl(
								the_set, the_set2, lvl + 1, PC->get_poset()->A, E->get_data(), 0);

						LG->add_node_vec_data(
								l + 2, f + so, the_set2, lvl + 1,
								0 /* verbose_level */);
						LG->set_distinguished_element_index(
								l + 2, f + so, lvl,
								0 /* verbose_level */);
					}
				}
				f += PC->get_Poo()->node_get_nb_of_extensions(n);
			}
		}
	}
	FREE_lint(the_set);
	FREE_lint(the_set2);
	FREE_int(Nb);
	FREE_int(Fst);

	if (f_v) {
		cout << "pc_convert_data_structure::make_level_graph done" << endl;
	}
}

void pc_convert_data_structure::make_poset_graph_detailed(
		combinatorics::graph_theory::layered_graph *&LG,
		int data1, int max_depth,
		int verbose_level)
// creates the poset graph, with two middle layers at each level.
// In total, the graph that is created will have 3 * depth + 1 layers.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Nb;
	int *Nb_middle;
	int i, po, so, n, n1, f, L;
	algebra::ring_theory::longinteger_domain D;
	int nb_layers = 3 * max_depth + 1;
	long int *the_set;
	long int *the_set2;

	if (f_v) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"verbose_level=" << verbose_level << endl;
		cout << "max_depth=" << max_depth << endl;
		cout << "nb_layers=" << nb_layers << endl;
	}

	//print_fusion_nodes(depth);


	the_set = NEW_lint(max_depth);
	the_set2 = NEW_lint(max_depth);
	Nb = NEW_int(nb_layers);
	Nb_middle = NEW_int(max_depth);

	for (i = 0; i < max_depth; i++) {
		Nb_middle[i] = PC->get_Poo()->count_extension_nodes_at_level(i);
	}

	for (i = 0; i < max_depth; i++) {

		Nb[i * 3 + 0] = PC->get_Poo()->nb_orbits_at_level(i);
		Nb[i * 3 + 1] = Nb_middle[i];
		Nb[i * 3 + 2] = Nb_middle[i];
	}
	Nb[max_depth * 3 + 0] = PC->get_Poo()->nb_orbits_at_level(max_depth);

	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"before LG->init" << endl;
		cout << "nb_layers=" << nb_layers << endl;
		cout << "Nb=";
		Int_vec_print(cout, Nb, nb_layers);
		cout << endl;
	}
	LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"after LG->init" << endl;
	}
	for (i = 0; i < nb_layers; i++) {
		if ((i % 3) == 0) {
			LG->set_radius_factor_for_all_nodes_at_level(
					i, .9 /* radius_factor */, 0 /* verbose_level */);
		}
		else {
			// .9 means we don't draw a label at that node
			//LG->set_radius_factor_for_all_nodes_at_level(
			// i, .9 /* radius_factor */, 0 /* verbose_level */);
			LG->set_radius_factor_for_all_nodes_at_level(
					i, 0.9 /* radius_factor */, 0 /* verbose_level */);
		}
	}

	LG->place(verbose_level);
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"after LG->place" << endl;
	}


	// adding edges:
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"adding edges" << endl;
	}
	for (L = 0; L < max_depth; L++) {
		if (f_vv) {
			cout << "pc_convert_data_structure::make_poset_graph_detailed "
					"adding edges at level " << L << endl;
		}
		f = 0;
		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(L); po++) {

			if (f_vv) {
				cout << "pc_convert_data_structure::make_poset_graph_detailed "
						"adding edges level=" << L << " po=" << po
						<< " / " << PC->get_Poo()->nb_orbits_at_level(L) << endl;
			}

			//
			n = PC->get_Poo()->first_node_at_level(L) + po;
			for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {

				if (false /*f_v*/) {
					cout << "pc_convert_data_structure::make_poset_graph_detailed "
							"adding edges level=" << L << " po=" << po
							<< " so=" << so << endl;
				}
				LG->add_edge(L * 3 + 0, po,
						L * 3 + 1, f + so,
						1, // edge_color
						0 /*verbose_level*/);
				LG->add_edge(L * 3 + 1, f + so,
						L * 3 + 2, f + so,
						1, // edge_color
						0 /*verbose_level*/);
				extension *E = PC->get_Poo()->get_node(n)->get_E(so);
				if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
					//cout << "extension node" << endl;
					n1 = E->get_data();
					//cout << "n1=" << n1 << endl;
					LG->add_edge(L * 3 + 2, f + so, L * 3 + 3,
							n1 - PC->get_Poo()->first_node_at_level(L + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}
				else if (E->get_type() == EXTENSION_TYPE_FUSION) {
					//cout << "fusion node" << endl;
					// po = data1
					// so = data2
					int n0, so0;
					n0 = E->get_data1();
					so0 = E->get_data2();
					//cout << "fusion (" << n << "/" << so
					//<< ") -> (" << n0 << "/" << so0 << ")" << endl;
					extension *E0;
					E0 = PC->get_Poo()->get_node(n0)->get_E(so0);
					if (E0->get_type() != EXTENSION_TYPE_EXTENSION) {
						cout << "warning: fusion node does not point to "
								"extension node" << endl;
						cout << "type = ";
						PC->print_extension_type(cout, E0->get_type());
						cout << endl;
						exit(1);
					}
					n1 = E0->get_data();
					//cout << "n1=" << n1
					//<< " first_poset_orbit_node_at_level[lvl + 1] = "
					//<< first_poset_orbit_node_at_level[lvl + 1] << endl;
					LG->add_edge(L * 3 + 2, f + so, L * 3 + 3,
							n1 - PC->get_Poo()->first_node_at_level(L + 1),
							1, // edge_color
							0 /*verbose_level*/);
				}
			}

			f += PC->get_Poo()->node_get_nb_of_extensions(n);
		}
		if (f_vv) {
			cout << "pc_convert_data_structure::make_poset_graph_detailed "
					"after LG->add_edge" << endl;
		}
	} // next L
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"adding edges done" << endl;
	}


	// adding vertex labels:
	if (f_vv) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed "
				"now making vertex labels" << endl;
	}
	for (L = 0; L <= max_depth; L++) {
		f = 0;
		if (f_vv) {
			cout << "pc_convert_data_structure::make_poset_graph_detailed "
					"now making vertex labels level " << L
					<< " / " << max_depth << endl;
		}

		for (po = 0; po < PC->get_Poo()->nb_orbits_at_level(L); po++) {


			if (f_vv) {
				cout << "pc_convert_data_structure::make_poset_graph_detailed "
						"now making vertex labels level " << L
						<< " / " << max_depth << " po=" << po
						<< " / " << PC->get_Poo()->nb_orbits_at_level(L) << endl;
			}


			string text;
			algebra::ring_theory::longinteger_object go, go1;
			int n, so, len, r;

			n = PC->get_Poo()->first_node_at_level(L) + po;
			PC->get_Poo()->get_stabilizer_order(L, po, go);
			go.print_to_string(text);

			string text3;

			text3.assign(text);

			LG->add_text(3 * L, po, text3, 0/*verbose_level*/);
			if (L) {
				LG->add_node_data1(
						3 * L, po,
						PC->get_Poo()->get_node(n)->get_pt(),
						0/*verbose_level*/);
			}

			PC->get_Poo()->get_set_by_level(L, po, the_set);
			LG->add_node_vec_data(3 * L, po, the_set, L, 0 /* verbose_level */);
#if 0
			if (lvl) {
				LG->add_node_data2(2 * lvl + 0, po, 2 * (lvl - 1),
						0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po,
						root[n].prev - first_poset_orbit_node_at_level[lvl - 1],
						0/*verbose_level*/);
			}
			else {
				LG->add_node_data2(2 * lvl + 0, po, -1, 0/*verbose_level*/);
				LG->add_node_data3(2 * lvl + 0, po, -1, 0/*verbose_level*/);
			}
#endif

			if (L < max_depth) {
				for (so = 0; so < PC->get_Poo()->node_get_nb_of_extensions(n); so++) {
					if (f_vv) {
						cout << "pc_convert_data_structure::make_poset_graph_detailed "
								"now making vertex labels level " << L
								<< " / " << max_depth << " po=" << po
								<< " / " << PC->get_Poo()->nb_orbits_at_level(L)
								<< " so=" << so << endl;
					}
					extension *E = PC->get_Poo()->get_node(n)->get_E(so);
					len = E->get_orbit_len();
					D.integral_division_by_int(go, len, go1, r);
					go1.print_to_string(text);

					string text3;

					text3.assign(text);

					LG->add_text(3 * L + 1, f + so, text3, 0/*verbose_level*/);
					LG->add_text(3 * L + 2, f + so, text3, 0/*verbose_level*/);

					//get_set_by_level(lvl, po, the_set);
					the_set[L] = E->get_pt();
					LG->add_node_vec_data(
							3 * L + 1, f + so, the_set,
							L + 1, 0 /* verbose_level */);
					LG->set_distinguished_element_index(3 * L + 1,
							f + so, L, 0 /* verbose_level */);


					if (E->get_type() == EXTENSION_TYPE_EXTENSION) {
						the_set[L] = E->get_pt();
						LG->add_node_vec_data(3 * L + 2, f + so,
								the_set, L + 1, 0 /* verbose_level */);
						LG->set_distinguished_element_index(3 * L + 2,
								f + so, L, 0 /* verbose_level */);
					}
					else if (E->get_type() == EXTENSION_TYPE_FUSION) {

						//Poset->A->element_retrieve(E->get_data(), Elt1, 0);
						PC->get_poset()->A2->map_a_set_based_on_hdl(
								the_set, the_set2, L + 1,
								PC->get_poset()->A, E->get_data(), 0);
						LG->add_node_vec_data(
								3 * L + 2, f + so,
								the_set2, L + 1, 0 /* verbose_level */);
						LG->set_distinguished_element_index(
								3 * L + 2,
								f + so, L, 0 /* verbose_level */);
					}
				}
				f += PC->get_Poo()->node_get_nb_of_extensions(n);
			} // if (L < max_depth)
		} // next po
	} // next L
	FREE_lint(the_set);
	FREE_lint(the_set2);
	FREE_int(Nb);
	FREE_int(Nb_middle);

	if (f_v) {
		cout << "pc_convert_data_structure::make_poset_graph_detailed done" << endl;
	}
}


void pc_convert_data_structure::make_spreadsheet_of_level_info(
		other::data_structures::spreadsheet *&Sp, int max_depth,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = false; //(verbose_level >= 1);

	if (f_v) {
		cout << "pc_convert_data_structure::make_spreadsheet_of_level_info" << endl;
	}

	int nb_rows, Nb_orbits, nb_orbits, i, level;
	string *Text_label;
	string *Text_nb_orbits;
	string *Text_orbit_length_sum;
	string *Text_schreier_vector_length_sum;
	string *Text_binomial;
	algebra::ring_theory::longinteger_object stab_order, orbit_length,
		orbit_length_sum, orbit_length_total;
	algebra::ring_theory::longinteger_object a, a_total;
	algebra::ring_theory::longinteger_domain D;
	combinatorics::other_combinatorics::combinatorics_domain C;
	int schreier_vector_length_int;
	algebra::ring_theory::longinteger_object schreier_vector_length,
		schreier_vector_length_sum, schreier_vector_length_total;
	int *rep;
	poset_orbit_node *O;

	nb_rows = max_depth + 2; // one extra row for totals
	rep = NEW_int(max_depth);
	Text_label = new string [nb_rows];
	Text_nb_orbits = new string [nb_rows];
	Text_orbit_length_sum = new string [nb_rows];
	Text_schreier_vector_length_sum = new string [nb_rows];
	Text_binomial = new string [nb_rows];

	Nb_orbits = 0;
	orbit_length_total.create(0);
	schreier_vector_length_total.create(0);
	a_total.create(0);

	for (level = 0; level <= max_depth; level++) {

		if (f_v) {
			cout << "pc_convert_data_structure::make_spreadsheet_of_level_info "
					"level = " << level << " / " << max_depth << endl;
		}
		nb_orbits = PC->get_Poo()->nb_orbits_at_level(level);


		Text_label[level] = std::to_string(level);

		Text_nb_orbits[level] = std::to_string(nb_orbits);

		orbit_length_sum.create(0);
		schreier_vector_length_sum.create(0);

		for (i = 0; i < nb_orbits; i++) {

			if (false) {
				cout << "pc_convert_data_structure::make_spreadsheet_of_level_info "
						"level = " << level << " / " << max_depth
						<< " orbit " << i << " / " << nb_orbits << endl;
			}
			PC->get_Poo()->get_orbit_length_and_stabilizer_order(i, level,
				stab_order, orbit_length);

			D.add_in_place(orbit_length_sum, orbit_length);


			O = PC->get_Poo()->get_node_ij(level, i);

			if (O->has_Schreier_vector()) {
				schreier_vector_length_int = O->get_nb_of_live_points();


			}
			else {
				//cout << "node " << level << " / " << i
				//		<< " does not have a Schreier vector" << endl;
				schreier_vector_length_int = 1;
			}
			if (schreier_vector_length_int <= 0) {
				schreier_vector_length_int = 1;
			}
			schreier_vector_length.create(schreier_vector_length_int);

			if (schreier_vector_length_int >= 0) {
				D.add_in_place(schreier_vector_length_sum,
						schreier_vector_length);
			}

		}

		//cout << "poset_classification::make_spreadsheet_of_level_info
		// computing binomial coeffcient" << endl;
		C.binomial(a, PC->get_poset()->A2->degree, level, false);

		Nb_orbits += nb_orbits;
		D.add_in_place(orbit_length_total, orbit_length_sum);
		D.add_in_place(schreier_vector_length_total,
				schreier_vector_length_sum);
		D.add_in_place(a_total, a);

		orbit_length_sum.print_to_string(Text_orbit_length_sum[level]);

		schreier_vector_length_sum.print_to_string(Text_schreier_vector_length_sum[level]);

		a.print_to_string(Text_binomial[level]);

	}

	level = max_depth + 1;
	Text_label[level] = "total";

	Text_nb_orbits[level] = std::to_string(Nb_orbits);

	orbit_length_total.print_to_string(Text_orbit_length_sum[level]);

	schreier_vector_length_total.print_to_string(Text_schreier_vector_length_sum[level]);

	a_total.print_to_string(Text_binomial[level]);


	Sp = NEW_OBJECT(other::data_structures::spreadsheet);
	Sp->init_empty_table(nb_rows + 1, 6);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, Text_label, "Level");
	Sp->fill_column_with_text(2, Text_nb_orbits, "Nb_orbits");
	Sp->fill_column_with_text(3, Text_orbit_length_sum, "Orbit_length_sum");
	Sp->fill_column_with_text(4, Text_schreier_vector_length_sum, "Schreier_vector_length_sum");
	Sp->fill_column_with_text(5, Text_binomial, "Binomial");



#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_int(rep);
	delete [] Text_label;
	delete [] Text_nb_orbits;
	delete [] Text_orbit_length_sum;
	delete [] Text_schreier_vector_length_sum;
	delete [] Text_binomial;

}



}}}

