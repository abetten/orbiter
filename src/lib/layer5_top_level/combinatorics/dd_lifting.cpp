/*
 * dd_lifting.cpp
 *
 *  Created on: May 18, 2024
 *      Author: betten
 */







#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {






dd_lifting::dd_lifting()
{
	DD = NULL;
	target_depth = 0;
	level = 0;

	//std::string starter_file;

	Nb_sol = NULL;
	Nb_nodes = NULL;
	Orbit_idx = NULL;
	nb_orbits_not_ruled_out = 0;

	nb_sol_total = 0;
	nb_nodes_total = 0;

	ODF = NULL;
}


dd_lifting::~dd_lifting()
{
	if (ODF) {
		FREE_OBJECT(ODF);
	}
	if (Orbit_idx) {
		FREE_int(Orbit_idx);
	}
	if (Nb_sol) {
		FREE_lint(Nb_sol);
	}
	if (Nb_nodes) {
		FREE_lint(Nb_nodes);
	}
}


void dd_lifting::perform_lifting(
		delandtsheer_doyen *DD,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "dd_lifting::perform_lifting" << endl;
	}

	dd_lifting::DD = DD;

	//int target_depth;
	target_depth = DD->Descr->K - DD->Descr->singletons_starter_size;
	if (f_v) {
		cout << "dd_lifting::perform_lifting target_depth=" << target_depth << endl;
	}

	//orbiter_kernel_system::orbiter_data_file *ODF;


	//string starter_file;


	level = DD->Descr->singletons_starter_size;

	starter_file = DD->Search_control->problem_label + "_lvl_" + std::to_string(level);

	if (f_v) {
		cout << "dd_lifting::perform_lifting starter_file = " << starter_file << endl;
	}

	ODF = NEW_OBJECT(orbiter_kernel_system::orbiter_data_file);

	if (f_v) {
		cout << "dd_lifting::perform_lifting "
				"before ODF->load" << endl;
	}
	ODF->load(starter_file, verbose_level);
	if (f_v) {
		cout << "dd_lifting::perform_lifting "
				"after ODF->load" << endl;
	}

	if (f_v) {
		cout << "dd_lifting::perform_lifting "
				"found " << ODF->nb_cases << " orbits at level " << level << endl;
	}

#if 0
	long int *Nb_sol;
	long int *Nb_nodes;
	int *Orbit_idx;
	int nb_orbits_not_ruled_out;
#endif
	int orbit_idx;
	int nb_cases = 0;
	int nb_cases_eliminated = 0;

	Orbit_idx = NEW_int(ODF->nb_cases);
	Nb_sol = NEW_lint(ODF->nb_cases);
	Nb_nodes = NEW_lint(ODF->nb_cases);
	nb_orbits_not_ruled_out = 0;

	Lint_vec_zero(Nb_sol, ODF->nb_cases);
	Lint_vec_zero(Nb_nodes, ODF->nb_cases);

	for (orbit_idx = 0; orbit_idx < ODF->nb_cases; orbit_idx++) {

	#if 0
		if (f_split) {
			if ((orbit_idx % split_m) == split_r) {
				continue;
			}
		}
	#endif

		if ((orbit_idx % 100)== 0) {
			f_vv = true;
		}
		else {
			f_vv = false;
		}
		if (f_vv) {
			cout << "dd_lifting::perform_lifting " << orbit_idx << " / " << ODF->nb_cases << " : ";
			Lint_vec_print(cout, ODF->sets[orbit_idx],
					ODF->set_sizes[orbit_idx]);
			cout << " : " << ODF->Ago_ascii[orbit_idx] << " : "
					<< ODF->Aut_ascii[orbit_idx] << endl;
		}

		long int *line0;

		line0 = ODF->sets[orbit_idx];
		if (ODF->set_sizes[orbit_idx] != level) {
			cout << "ODF->set_sizes[orbit_idx] != level" << endl;
			exit(1);
		}

		DD->compute_live_points_for_singleton_search(
				line0, level, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "dd_lifting::perform_lifting case " << orbit_idx << " / " << ODF->nb_cases
					<< " we found " << DD->nb_live_points << " live points" << endl;
		}
		if (DD->nb_live_points < target_depth) {
			if (f_vv) {
				cout << "dd_lifting::perform_lifting eliminated!" << endl;
			}
			Nb_sol[orbit_idx] = 0;
			Nb_nodes[orbit_idx] = 0;
			nb_cases_eliminated++;
		}
		else {
			Orbit_idx[nb_orbits_not_ruled_out++] = orbit_idx;
			nb_cases++;
		}
		if (f_vv) {
			cout << "dd_lifting::perform_lifting nb_cases=" << nb_cases << " vs ";
			cout << "dd_lifting::perform_lifting nb_cases_eliminated=" << nb_cases_eliminated << endl;
		}
	} // orbit_idx
	cout << "dd_lifting::perform_lifting nb_cases=" << nb_cases << endl;
	cout << "dd_lifting::perform_lifting nb_cases_eliminated=" << nb_cases_eliminated << endl;

	int orbit_not_ruled_out;
	nb_sol_total = 0;
	nb_nodes_total = 0;

	for (orbit_not_ruled_out = 0;
			orbit_not_ruled_out < nb_orbits_not_ruled_out;
			orbit_not_ruled_out++) {

		orbit_idx = Orbit_idx[orbit_not_ruled_out];

#if 0
		if ((orbit_not_ruled_out % 100)== 0) {
			f_vv = true;
		}
		else {
			f_vv = false;
		}
#endif

		if (f_vv) {
			cout << "dd_lifting::perform_lifting orbit_not_ruled_out=" << orbit_not_ruled_out
					<< " / " << nb_orbits_not_ruled_out
					<< " is orbit_idx " << orbit_idx << endl;
		}
		long int nb_sol, nb_nodes;

		nb_sol = 0;
		nb_nodes = 0;


		search_case_singletons_and_count(
				orbit_idx, nb_sol, nb_nodes, verbose_level - 2);

		Nb_sol[orbit_idx] = nb_sol;
		Nb_nodes[orbit_idx] = nb_nodes;
		nb_sol_total += nb_sol;
		nb_nodes_total += nb_nodes;

		if (f_vv) {
			cout << "dd_lifting::perform_lifting orbit_idx=" << orbit_idx
					<< " / " << ODF->nb_cases
					<< " nb_sol_total " << nb_sol_total << endl;
		}


#if 0
		long int *line0;

		line0 = ODF->sets[orbit_idx];
		if (ODF->set_sizes[orbit_idx] != level) {
			cout << "ODF->set_sizes[orbit_idx] != level" << endl;
			exit(1);
		}

		compute_live_points(
				line0, level, 0 /*verbose_level*/);

		if (f_vv) {
			cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
					<< nb_orbits_not_ruled_out << " is orbit_idx"
					<< orbit_idx << " / " << ODF->nb_cases
					<< " we found " << nb_live_points
					<< " live points" << endl;
		}



#if 0
		if (nb_live_points == target_depth) {
			Lint_vec_copy(line0, line, level);
			Lint_vec_copy(live_points, line + level, target_depth);
			if (check_orbit_covering(line, Descr->K, 0 /* verbose_level */)) {
				if (f_vv) {
					cout << "found a solution in orbit " << orbit_idx << endl;
				}
				nb_sol_total++;
				Nb_sol[orbit_idx] = 1;
			}



		}
		else {
#endif

			if (f_vv) {
				cout << "orbit_not_ruled_out=" << orbit_not_ruled_out << " / "
						<< nb_orbits_not_ruled_out << " is orbit_idx"
						<< orbit_idx << " / " << ODF->nb_cases
						<< " we found " << nb_live_points
						<< " live points, doing a search; " << endl;
			}

#if 0
			int *subset;
			int nCk, l;
			combinatorics::combinatorics_domain Combi;

			subset = NEW_int(target_depth);
			nCk = Combi.int_n_choose_k(
					nb_live_points, target_depth);

			if (f_vv) {
				cout << "nb_live_points = " << nb_live_points
						<< " target_depth = " << target_depth << " nCk = " << nCk << endl;
			}
			for (l = 0; l < nCk; l++) {

				Combi.unrank_k_subset(
						l, subset, nb_live_points, target_depth);

				Lint_vec_copy(
						line0, line, level);

				Int_vec_apply_lint(
						subset, live_points, line + level, target_depth);

				if (check_orbit_covering(
						line, Descr->K, 0 /* verbose_level */)) {
					cout << "found a solution, subset " << l
							<< " / " << nCk << " in orbit "
							<< orbit_idx << endl;
					nb_sol++;
				}
			} // next l

			FREE_int(subset);
#else
			long int nb_sol;

			nb_sol = 0;
			search_case_singletons(
					ODF,
					orbit_idx, nb_sol, verbose_level);

			Nb_sol[orbit_idx] = nb_sol;
			nb_sol_total += nb_sol;
#endif

#if 0
		} // else
#endif
#endif


	} // next orbit_not_ruled_out

	cout << "dd_lifting::perform_lifting nb_sol_total=" << nb_sol_total << endl;
	cout << "dd_lifting::perform_lifting nb_nodes_total=" << nb_nodes_total << endl;




	if (f_v) {
		cout << "dd_lifting::perform_lifting done" << endl;
	}
}

void dd_lifting::search_case_singletons_and_count(
		int orbit_idx, long int &nb_sol, long int &nb_nodes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);


	if (f_v) {
		cout << "dd_lifting::search_case_singletons_and_count "
				<< orbit_idx << " / " << ODF->nb_cases << endl;
	}



	int level = DD->Descr->singletons_starter_size;
	long int *line0;

	line0 = ODF->sets[orbit_idx];
	if (ODF->set_sizes[orbit_idx] != level) {
		cout << "dd_lifting::search_case_singletons_and_count "
				"ODF->set_sizes[orbit_idx] != level" << endl;
		exit(1);
	}

	DD->compute_live_points_for_singleton_search(
			line0, level, 0 /*verbose_level*/);

	if (f_vv) {
		cout << "dd_lifting::search_case_singletons_and_count "
				"orbit_idx = " << orbit_idx << " / " << ODF->nb_cases
				<< " we found " << DD->nb_live_points
				<< " live points" << endl;
	}


#if 0
	if (nb_live_points == target_depth) {
		Lint_vec_copy(line0, line, level);
		Lint_vec_copy(live_points, line + level, target_depth);
		if (check_orbit_covering(line, Descr->K, 0 /* verbose_level */)) {
			if (f_vv) {
				cout << "found a solution in orbit " << orbit_idx << endl;
			}
			nb_sol_total++;
			Nb_sol[orbit_idx] = 1;
		}



	}
	else {
#endif

		if (f_vv) {
			cout << "dd_lifting::search_case_singletons_and_count "
					"orbit_idx = " << orbit_idx << " / " << ODF->nb_cases
					<< " we found " << DD->nb_live_points
					<< " live points, doing a search; " << endl;
		}

#if 0
		int *subset;
		int nCk, l;
		combinatorics::combinatorics_domain Combi;

		subset = NEW_int(target_depth);
		nCk = Combi.int_n_choose_k(
				nb_live_points, target_depth);

		if (f_vv) {
			cout << "nb_live_points = " << nb_live_points
					<< " target_depth = " << target_depth << " nCk = " << nCk << endl;
		}
		for (l = 0; l < nCk; l++) {

			Combi.unrank_k_subset(
					l, subset, nb_live_points, target_depth);

			Lint_vec_copy(
					line0, line, level);

			Int_vec_apply_lint(
					subset, live_points, line + level, target_depth);

			if (check_orbit_covering(
					line, Descr->K, 0 /* verbose_level */)) {
				cout << "found a solution, subset " << l
						<< " / " << nCk << " in orbit "
						<< orbit_idx << endl;
				nb_sol++;
			}
		} // next l

		FREE_int(subset);
#else

		nb_sol = 0;
		search_case_singletons(
				orbit_idx, nb_sol, nb_nodes, verbose_level - 2);


		if (f_vv) {
			cout << "dd_lifting::search_case_singletons_and_count "
					"orbit_idx = " << orbit_idx << " / " << ODF->nb_cases
					<< " we found " << nb_sol
					<< " solutions." << endl;
		}
#endif

#if 0
	} // else
#endif

	if (f_v) {
		cout << "dd_lifting::search_case_singletons_and_count "
				<< orbit_idx << " / " << ODF->nb_cases << " done" << endl;
	}

}

void dd_lifting::search_case_singletons(
		int orbit_idx, long int &nb_sol, long int &nb_nodes,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "dd_lifting::search_case_singletons "
				<< orbit_idx << " / " << ODF->nb_cases << endl;
	}

	dd_search_singletons *DD_ss;

	DD_ss = NEW_OBJECT(dd_search_singletons);

	DD_ss->search_case_singletons(
			this,
			orbit_idx, verbose_level);

	nb_sol = DD_ss->Solutions.size();
	nb_nodes = DD_ss->nb_nodes;

	FREE_OBJECT(DD_ss);


#if 0
	int target_depth;

	target_depth = Descr->K - Descr->singletons_starter_size;

	if (f_v) {
		cout << "target_depth=" << target_depth << endl;
		cout << "nb_live_points=" << nb_live_points << endl;
	}

	data_structures::set_of_sets *Live_points;

	Live_points = NEW_OBJECT(data_structures::set_of_sets);

	Live_points->init_basic_constant_size(
			nb_live_points,
			target_depth + 1, nb_live_points,
			0 /*verbose_level*/);

	Lint_vec_copy(live_points, Live_points->Sets[0], nb_live_points);
	Live_points->Set_size[0] = nb_live_points;



	long int *chosen_set;
	int *index;
	long int nb_nodes = 0;


	chosen_set = NEW_lint(target_depth);
	index = NEW_int(target_depth);


	nb_sol = 0;

	search_case_singletons_recursion(
			ODF, orbit_idx, target_depth, 0 /* level */,
			Live_points,
			chosen_set, index, nb_nodes, nb_sol,
			verbose_level);

	FREE_lint(chosen_set);
	FREE_int(index);
	FREE_OBJECT(Live_points);
#endif

	if (f_v) {
		cout << "dd_lifting::search_case_singletons "
				<< orbit_idx << " / " << ODF->nb_cases << " done" << endl;
	}
}




}}}


