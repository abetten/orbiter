/*
 * graph_theory_apps.cpp
 *
 *  Created on: Jan 10, 2023
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_graph_theory {


graph_theory_apps::graph_theory_apps()
{
	Record_birth();

}

graph_theory_apps::~graph_theory_apps()
{
	Record_death();

}


void graph_theory_apps::automorphism_group(
		combinatorics::graph_theory::colored_graph *CG,
		std::vector<std::string> &feedback,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = CG->label + ".colored_graph";


	interfaces::nauty_interface_for_graphs Nauty;
	actions::action *Aut;

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"before Nauty.create_automorphism_group_of_colored_graph_object" << endl;
	}
	Aut = Nauty.create_automorphism_group_of_colored_graph_object(
			CG, verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"after Nauty.create_automorphism_group_of_colored_graph_object" << endl;
	}


	algebra::ring_theory::longinteger_object go;

	Aut->Strong_gens->group_order(go);

	string s;

	s = go.stringify();
	feedback.push_back(s);


	string title;

	title = "Automorphism Group of Colored Graph";

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"before report_automorphism_group" << endl;
	}
	report_automorphism_group(CG, Aut, title, verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"after report_automorphism_group" << endl;
	}


	string fname_group;

	fname_group = CG->label + "_group.makefile";

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"before Aut->export_to_orbiter_as_bsgs "
				"label = " << CG->label << endl;
	}
	Aut->degree--;
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"before Aut->export_to_orbiter_as_bsgs "
				"degree = " << Aut->degree << endl;
	}
	Aut->export_to_orbiter_as_bsgs(
			fname_group,
			CG->label, CG->label_tex, Aut->Strong_gens,
			verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group "
				"after Aut->export_to_orbiter_as_bsgs" << endl;
	}
	//file_io Fio;

	cout << "written file " << fname_group << " of size "
			<< Fio.file_size(fname_group) << endl;

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group done" << endl;
	}

}


void graph_theory_apps::automorphism_group_bw(
		combinatorics::graph_theory::colored_graph *CG,
		std::vector<std::string> &feedback,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;

	fname = CG->label + ".colored_graph";


	interfaces::nauty_interface_for_graphs Nauty;
	actions::action *Aut;

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"before Nauty.create_automorphism_group_of_colored_graph_ignoring_colors" << endl;
	}
	Aut = Nauty.create_automorphism_group_of_colored_graph_ignoring_colors(
			CG, verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"after Nauty.create_automorphism_group_of_colored_graph_ignoring_colors" << endl;
	}

	string fname_report;
	string fname_orbits;

	fname_report = CG->label + "_report.tex";
	fname_orbits = CG->label + "_orbits.csv";


	algebra::ring_theory::longinteger_object go;

	Aut->Strong_gens->group_order(go);

	string s;

	s = go.stringify();
	feedback.push_back(s);


	string title;

	title = "Automorphism Group of Graph (b/w)";

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"before report_automorphism_group" << endl;
	}
	report_automorphism_group(
			CG, Aut, title, verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"after report_automorphism_group" << endl;
	}



	string fname_group;

	fname_group = CG->label + "_group.makefile";

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"label = " << CG->label << endl;
	}

	// no need to reduce the degree since we run the automorphism group of the graph in back and white,
	// so no color information is added

	//Aut->degree--;

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"degree = " << Aut->degree << endl;
	}

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"before Aut->export_to_orbiter_as_bsgs" << endl;
	}
	Aut->export_to_orbiter_as_bsgs(
			fname_group,
			CG->label, CG->label_tex, Aut->Strong_gens,
			verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"after Aut->export_to_orbiter_as_bsgs" << endl;
	}
	//file_io Fio;

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw "
				"written file " << fname_group << " of size "
			<< Fio.file_size(fname_group) << endl;
	}

	if (f_v) {
		cout << "graph_theory_apps::automorphism_group_bw done" << endl;
	}

}

void graph_theory_apps::report_automorphism_group(
		combinatorics::graph_theory::colored_graph *CG,
		actions::action *Aut,
		std::string &title,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "graph_theory_apps::report_automorphism_group" << endl;
	}

	string fname_report;
	string fname_orbits;

	fname_report = CG->label + "_report.tex";
	fname_orbits = CG->label + "_orbits.csv";


	algebra::ring_theory::longinteger_object go;

	Aut->Strong_gens->group_order(go);

	string s;

	s = go.stringify();


	string author, extra_praeamble;


	{
		ofstream ost(fname_report);
		other::l1_interfaces::latex_interface L;

		L.head(ost,
				false /* f_book*/,
				true /* f_title */,
				title, author,
				false /* f_toc */,
				false /* f_landscape */,
				true /* f_12pt */,
				true /* f_enlarged_page */,
				true /* f_pagenumbers */,
				extra_praeamble /* extra_praeamble */);



		ost << "\\noindent The automorphism group of \\verb'" << CG->label_tex << "' "
				"has order " << go
				<< " and is generated by:\\\\" << endl;
		Aut->Strong_gens->print_generators_tex(ost);

		actions::action_global GL;
		int *orbit_no;

		GL.get_orbits_on_points_as_characteristic_vector(
				Aut,
				orbit_no,
				verbose_level);

		ost << endl;
		ost << "\\bigskip" << endl;
		ost << endl;


		ost << "Orbits on points: ";
		Int_vec_print_fully(ost, orbit_no, Aut->degree);
		ost << "\\\\" << endl;

		other::orbiter_kernel_system::file_io Fio;

		Fio.Csv_file_support->int_matrix_write_csv(
				fname_orbits, orbit_no, Aut->degree - 1, 1);


		if (f_v) {
			cout << "graph_theory_apps::automorphism_group after report" << endl;
		}


		L.foot(ost);

	}
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "written file " << fname_report << " of size "
				<< Fio.file_size(fname_report) << endl;
	}

	if (f_v) {
		cout << "graph_theory_apps::report_automorphism_group done" << endl;
	}
}



void graph_theory_apps::expander_graph(
		int p, int q,
		int f_special,
		algebra::field_theory::finite_field *F,
		actions::action *A,
		int *&Adj, int &N,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "graph_theory_apps::expander_graph" << endl;
	}

	groups::sims *Sims;

	Sims = A->Sims;


	//longinteger_object go;
	long int goi;

	goi = Sims->group_order_lint();

	if (f_v) {
		cout << "graph_theory_apps::expander_graph "
				"found a group of order " << goi << endl;
	}



	int i, j;
	int a0, a1, a2, a3;
	int sqrt_p;

	int *sqrt_mod_q;
	int I;
	int *A4;
	int nb_A4 = 0;

	A4 = NEW_int((p + 1) * 4);
	sqrt_mod_q = NEW_int(q);
	for (i = 0; i < q; i++) {
		sqrt_mod_q[i] = -1;
	}
	for (i = 0; i < q; i++) {
		j = F->mult(i, i);
		sqrt_mod_q[j] = i;
	}
	if (f_v) {
		cout << "graph_theory_apps::expander_graph sqrt_mod_q:" << endl;
		Int_vec_print(cout, sqrt_mod_q, q);
		cout << endl;
	}

	sqrt_p = 0;
	for (i = 1; i < p; i++) {
		if (i * i > p) {
			sqrt_p = i - 1;
			break;
		}
	}
	if (f_v) {
		cout << "graph_theory_apps::expander_graph p=" << p << endl;
		cout << "graph_theory_apps::expander_graph sqrt_p = " << sqrt_p << endl;
	}


	for (I = 0; I < q; I++) {
		if (F->add(F->mult(I, I), 1) == 0) {
			break;
		}
	}
	if (I == q) {
		cout << "graph_theory_apps::expander_graph did not find I" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "graph_theory_apps::expander_graph I=" << I << endl;
	}

	for (a0 = 1; a0 <= sqrt_p; a0++) {
		if (EVEN(a0)) {
			continue;
		}
		for (a1 = -sqrt_p; a1 <= sqrt_p; a1++) {
			if (ODD(a1)) {
				continue;
			}
			for (a2 = -sqrt_p; a2 <= sqrt_p; a2++) {
				if (ODD(a2)) {
					continue;
				}
				for (a3 = -sqrt_p; a3 <= sqrt_p; a3++) {
					if (ODD(a3)) {
						continue;
					}
					if (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 == p) {
						if (f_v) {
							cout << "graph_theory_apps::expander_graph "
									"solution " << nb_A4 << " : " << a0
									<< ", " << a1 << ", " << a2 << ", "
									<< a3 << ", " << endl;
						}
						if (nb_A4 == p + 1) {
							cout << "graph_theory_apps::expander_graph "
									"too many solutions" << endl;
							exit(1);
						}
						A4[nb_A4 * 4 + 0] = a0;
						A4[nb_A4 * 4 + 1] = a1;
						A4[nb_A4 * 4 + 2] = a2;
						A4[nb_A4 * 4 + 3] = a3;
						nb_A4++;
					}
				}
			}
		}
	}

	if (f_v) {
		cout << "graph_theory_apps::expander_graph nb_A4=" << nb_A4 << endl;
	}
	if (nb_A4 != p + 1) {
		cout << "graph_theory_apps::expander_graph nb_A4 != p + 1" << endl;
		exit(1);
	}

	if (f_v) {
		Int_matrix_print(A4, nb_A4, 4);
	}

	data_structures_groups::vector_ge *gens;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int M4[4];
	int det; //, s, sv;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	gens->init(A, verbose_level - 2);
	gens->allocate(nb_A4, verbose_level - 2);

	if (f_v) {
		cout << "graph_theory_apps::expander_graph making connection set:" << endl;
	}
	for (i = 0; i < nb_A4; i++) {

		if (f_vv) {
			cout << "graph_theory_apps::expander_graph "
					"making generator " << i << ":" << endl;
		}
		a0 = A4[i * 4 + 0];
		a1 = A4[i * 4 + 1];
		a2 = A4[i * 4 + 2];
		a3 = A4[i * 4 + 3];
		while (a0 < 0) {
			a0 += q;
		}
		while (a1 < 0) {
			a1 += q;
		}
		while (a2 < 0) {
			a2 += q;
		}
		while (a3 < 0) {
			a3 += q;
		}
		a0 = a0 % q;
		a1 = a1 % q;
		a2 = a2 % q;
		a3 = a3 % q;
		if (f_vv) {
			cout << "graph_theory_apps::expander_graph "
					"making generator " << i << ": a0=" << a0
					<< " a1=" << a1 << " a2=" << a2
					<< " a3=" << a3 << endl;
		}
		M4[0] = F->add(a0, F->mult(I, a1));
		M4[1] = F->add(a2, F->mult(I, a3));
		M4[2] = F->add(F->negate(a2), F->mult(I, a3));
		M4[3] = F->add(a0, F->negate(F->mult(I, a1)));

		if (f_vv) {
			cout << "M4=";
			Int_vec_print(cout, M4, 4);
			cout << endl;
		}

		if (f_special) {
			det = F->add(F->mult(M4[0], M4[3]),
					F->negate(F->mult(M4[1], M4[2])));

			if (f_vv) {
				cout << "det=" << det << endl;
			}

#if 0
			s = sqrt_mod_q[det];
			if (s == -1) {
				cout << "graph_theory_apps::expander_graph determinant is not a square" << endl;
				exit(1);
			}
			sv = F->inverse(s);
			if (f_vv) {
				cout << "graph_theory_apps::expander_graph det=" << det << " sqrt=" << s
						<< " mutiplying by " << sv << endl;
			}
			for (j = 0; j < 4; j++) {
				M4[j] = F->mult(sv, M4[j]);
			}
			if (f_vv) {
				cout << "graph_theory_apps::expander_graph M4=";
				int_vec_print(cout, M4, 4);
				cout << endl;
			}
#endif
		}

		A->Group_element->make_element(
				Elt1, M4, verbose_level - 1);

		if (f_v) {
			cout << "graph_theory_apps::expander_graph "
					"s_" << i << "=" << endl;
			A->Group_element->element_print_quick(Elt1, cout);
		}

		A->Group_element->element_move(Elt1, gens->ith(i), 0);
	}

	if (f_v) {
		cout << "graph_theory_apps::expander_graph "
				"before Sims->Cayley_graph" << endl;
	}
	Sims->Cayley_graph(Adj, N, gens, verbose_level);
	if (f_v) {
		cout << "graph_theory_apps::expander_graph "
				"after Sims->Cayley_graph" << endl;
	}


	if (f_v) {
		cout << "graph_theory_apps::expander_graph "
				"The adjacency matrix of a graph with " << goi
				<< " vertices has been computed" << endl;
		//int_matrix_print(Adj, goi, goi);
	}

	int k;
	k = 0;
	for (i = 0; i < N; i++) {
		if (Adj[0 * N + i]) {
			k++;
		}
	}
	if (f_v) {
		cout << "graph_theory_apps::expander_graph "
				"the graph is regular of degree " << k << endl;
	}


	//N = goi;


	FREE_OBJECT(gens);
	//FREE_OBJECT(A);
	FREE_int(A4);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	//FREE_OBJECT(F);

	if (f_v) {
		cout << "graph_theory_apps::expander_graph done" << endl;
	}
}


void graph_theory_apps::common_neighbors(
		int nb, combinatorics::graph_theory::colored_graph **CG,
		std::string &common_neighbors_set, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theory_apps::common_neighbors" << endl;
		cout << "graph_theory_apps::test_automorphism_property_of_group nb = " << nb << endl;
		cout << "graph_theory_apps::test_automorphism_property_of_group common_neighbors_set = " << common_neighbors_set << endl;
	}

	if (nb != 1) {
		cout << "graph_theory_apps::test_automorphism_property_of_group nb != 1" << endl;
		exit(1);
	}

	combinatorics::graph_theory::colored_graph *Gamma;

	Gamma = CG[0];


	int *Pts;
	int nb_pts;

	Get_int_vector_from_label(common_neighbors_set,
			Pts, nb_pts, 0 /* verbose_level */);

	if (f_v) {
		cout << "graph_theory_apps::common_neighbors Pts=";
		Int_vec_print(cout, Pts, nb_pts);
		cout << endl;
	}

	other::data_structures::fancy_set *Neighbors;

	Gamma->common_neighbors(
		Pts, nb_pts,
		Neighbors,
		verbose_level);

	if (f_v) {
		cout << "graph_theory_apps::common_neighbors number of common neighbors is " << Neighbors->k << endl;
		cout << "graph_theory_apps::common_neighbors =";
		Neighbors->print();
		cout << endl;
	}


	if (f_v) {
		cout << "graph_theory_apps::common_neighbors done" << endl;
	}
}


void graph_theory_apps::test_automorphism_property_of_group(
		int nb, combinatorics::graph_theory::colored_graph **CG,
		std::string &group_label, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	other::data_structures::string_tools ST;

	if (f_v) {
		cout << "graph_theory_apps::test_automorphism_property_of_group" << endl;
		cout << "graph_theory_apps::test_automorphism_property_of_group nb = " << nb << endl;
		cout << "graph_theory_apps::test_automorphism_property_of_group group_label = " << group_label << endl;
	}

	if (nb != 1) {
		cout << "graph_theory_apps::test_automorphism_property_of_group nb != 1" << endl;
		exit(1);
	}

	combinatorics::graph_theory::colored_graph *Gamma;

	Gamma = CG[0];

	layer3_group_actions::groups::any_group *AG;


	AG = Get_any_group(group_label);


	if (!AG->f_modified_group) {
		cout << "graph_theory_apps::test_automorphism_property_of_group "
				"the group is not of modified group type" << endl;
		exit(1);
	}

	int f_aut = true;

	if (AG->f_modified_group) {
		//group_constructions::modified_group_create *MGC;

		//MGC = AG->MGC;

		groups::strong_generators *Subgroup_gens;

		Subgroup_gens = AG->Subgroup_gens;

		int h;
		int *perm;


		perm = NEW_int(AG->A->degree);

		for (h = 0; h < Subgroup_gens->gens->len; h++) {

			if (nb != 1) {
				cout << "graph_theory_apps::test_automorphism_property_of_group "
						"testing generator " << h << " / " << Subgroup_gens->gens->len << endl;
			}

			int i, j;
			int *Elt1;

			Elt1 = Subgroup_gens->gens->ith(h);

			for (i = 0; i < AG->A->degree; i++) {
				j = AG->A->Group_element->element_image_of(i, Elt1, 0);
				perm[i] = j;
			}


			f_aut = Gamma->test_automorphism_property(
					perm, AG->A->degree,
					verbose_level - 2);

			if (!f_aut) {
				if (f_v) {
					cout << "graph_theory_apps::test_automorphism_property_of_group "
							"generator " << h << " / " << Subgroup_gens->gens->len << " is not an automorphism" << endl;
				}
				break;
			}

		}

		if (h < Subgroup_gens->gens->len) {
			if (f_v) {
				cout << "graph_theory_apps::test_automorphism_property_of_group "
						"fail" << endl;
			}
		}

	}

	if (f_aut) {
		cout << "graph_theory_apps::test_automorphism_property_of_group "
				"the group has the automorphism property" << endl;
	}
	else {
		cout << "graph_theory_apps::test_automorphism_property_of_group "
				"the group does not have the automorphism property" << endl;
	}


	if (f_v) {
		cout << "graph_theory_apps::test_automorphism_property_of_group done" << endl;
	}
}





}}}

