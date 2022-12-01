/*
 * orbit_cascade.cpp
 *
 *  Created on: May 22, 2022
 *      Author: betten
 */



#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbit_cascade::orbit_cascade()
{
	N = 0;
	k = 0;

	G = NULL;

	Control = NULL;
	Primary_poset = NULL;
	Orbits_on_primary_poset = NULL;

	number_primary_orbits = 0;
	stabilizer_gens = NULL;
	Reps_and_complements = NULL;

	A_restricted = NULL;
	Secondary_poset = NULL;
	orbits_secondary_poset = NULL;

	nb_orbits_secondary = NULL;
	flag_orbit_first = NULL;
	nb_orbits_secondary_total = 0;

	Flag_orbits = NULL;
	nb_orbits_reduced = 0;

	Partition_orbits = NULL;

}

orbit_cascade::~orbit_cascade()
{
}

void orbit_cascade::init(int N, int k, any_group *G,
		std::string &Control_label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_cascade::init N=" << N << endl;
		cout << "orbit_cascade::init k=" << k << endl;
	}
	orbit_cascade::N = N; // total size of the set
	orbit_cascade::k = k; // size of the subsets

	if (N % k) {
		cout << "orbit_cascade::init N must be divisible by k" << endl;
		exit(1);
	}

	orbit_cascade::G = G;

	Control = Get_object_of_type_poset_classification_control(Control_label);


	Primary_poset = NEW_OBJECT(poset_classification::poset_with_group_action);

	if (f_v) {
		cout << "orbit_cascade::init control=" << endl;
		Control->print();
	}
	if (f_v) {
		cout << "orbit_cascade::init A_base=" << endl;
		G->A_base->print_info();
	}
	if (f_v) {
		cout << "orbit_cascade::init A=" << endl;
		G->A->print_info();
	}
	if (f_v) {
		cout << "orbit_cascade::init group order" << endl;

		ring_theory::longinteger_object go;

		G->Subgroup_gens->group_order(go);

		cout << go << endl;
	}


	if (f_v) {
		cout << "orbit_cascade::init "
				"before Poset->init_subset_lattice" << endl;
	}
	Primary_poset->init_subset_lattice(G->A_base, G->A,
			G->Subgroup_gens,
			verbose_level);

	if (f_v) {
		cout << "orbit_cascade::init "
				"before Primary_poset->orbits_on_k_sets_compute" << endl;
	}
	Orbits_on_primary_poset = Primary_poset->orbits_on_k_sets_compute(
			Control,
			k /* subset_size */,
			verbose_level);
	if (f_v) {
		cout << "orbit_cascade::init "
				"after Primary_poset->orbits_on_k_sets_compute" << endl;
	}

	number_primary_orbits = Orbits_on_primary_poset->nb_orbits_at_level(k);

	stabilizer_gens = (groups::strong_generators **) NEW_pvoid(number_primary_orbits);

	int i;
	long int *set;
	int size;

	if (f_v) {
		cout << "orbit_cascade::init "
				"getting stabilizer generators" << endl;
	}
	for (i = 0; i < number_primary_orbits; i++) {

		Orbits_on_primary_poset->get_stabilizer_generators(stabilizer_gens[i],
				k, i, 0 /* verbose_level */);

	}
	if (f_v) {
		cout << "orbit_cascade::init "
				"after getting stabilizer generators" << endl;
	}


	set = NEW_lint(k);

	if (N != G->A->degree) {
		cout << "orbit_cascade::init N != G->A->degree" << endl;
		exit(1);
	}

	Reps_and_complements = NEW_lint(number_primary_orbits * N);


	if (f_v) {
		cout << "orbit_cascade::init "
				"getting orbit representatives" << endl;
	}
	for (i = 0; i < number_primary_orbits; i++) {

		Orbits_on_primary_poset->get_set(
			k, i, set, size);

		if (size != k) {
			cout << "orbit_cascade::init size != k" << endl;
			exit(1);
		}
		Lint_vec_copy(set, Reps_and_complements + i * N, k);

		Lint_vec_complement_to(Reps_and_complements + i * N, Reps_and_complements + i * N + k, N, k);
	}
	if (f_v) {
		cout << "orbit_cascade::init "
				"getting orbit representatives done" << endl;
	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"Reps_and_complements:" << endl;
		Lint_matrix_print(Reps_and_complements, number_primary_orbits, N);
	}


	A_restricted = (actions::action **) NEW_pvoid(number_primary_orbits);


	if (f_v) {
		cout << "orbit_cascade::init "
				"computing restricted action" << endl;
	}

	for (i = 0; i < number_primary_orbits; i++) {

		A_restricted[i] = G->A->restricted_action(
				Reps_and_complements + i * N + k, N - k,
				0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"computing restricted action done" << endl;
	}


	if (f_v) {
		cout << "orbit_cascade::init "
				"setting up Secondary_poset" << endl;
	}

	Secondary_poset = (poset_classification::poset_with_group_action **) NEW_pvoid(number_primary_orbits);

	for (i = 0; i < number_primary_orbits; i++) {
		Secondary_poset[i] = NEW_OBJECT(poset_classification::poset_with_group_action);

		Secondary_poset[i]->init_subset_lattice(
				G->A_base, A_restricted[i],
				stabilizer_gens[i],
				verbose_level);

	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"setting up Secondary_poset done" << endl;
	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"computing orbits on secondary posets" << endl;
	}

	orbits_secondary_poset = (poset_classification::poset_classification **) NEW_pvoid(number_primary_orbits);

	for (i = 0; i < number_primary_orbits; i++) {
		orbits_secondary_poset[i] = Secondary_poset[i]->orbits_on_k_sets_compute(
				Control,
				k /* subset_size */,
				verbose_level);

	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"computing orbits on secondary posets done" << endl;
	}


	nb_orbits_secondary = NEW_int(number_primary_orbits);
	flag_orbit_first = NEW_int(number_primary_orbits);
	nb_orbits_secondary_total = 0;


	if (f_v) {
		cout << "orbit_cascade::init "
				"counting orbits on secondary posets" << endl;
	}

	for (i = 0; i < number_primary_orbits; i++) {

		nb_orbits_secondary[i] = orbits_secondary_poset[i]->nb_orbits_at_level(k);

		if (i == 0) {
			flag_orbit_first[0] = 0;
		}
		else {
			flag_orbit_first[i] = flag_orbit_first[i - 1] + nb_orbits_secondary[i - 1];
		}
		nb_orbits_secondary_total += nb_orbits_secondary[i];
	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"counting orbits on secondary posets done" << endl;
		cout << "orbit_cascade::init "
				"nb_orbits_secondary_total = " << nb_orbits_secondary_total << endl;
	}



	// downstep:
	if (f_v) {
		cout << "orbit_cascade::init "
				"before downstep" << endl;
	}
	downstep(verbose_level - 2);
	if (f_v) {

		string title;

		title.assign("Flag orbits");

		cout << "orbit_cascade::init "
				"after downstep" << endl;

		Flag_orbits->print_latex(cout,
			title, TRUE /* f_print_stabilizer_gens*/);
	}


	std::vector<long int> Ago;
	// upstep:
	if (f_v) {
		cout << "orbit_cascade::init "
				"before upstep" << endl;
	}

	upstep(Ago, verbose_level - 2);

	if (f_v) {
		cout << "orbit_cascade::init "
				"after upstep" << endl;
	}

	if (f_v) {
		cout << "orbit_cascade::init "
				"We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits" << endl;

		string prefix;

		prefix.assign("partition_orbit");
		Partition_orbits->generate_source_code(cout, prefix, verbose_level);
	}



	data_structures::tally_lint T;

	T.init_vector_lint(Ago,
			FALSE /* f_second */, 0 /* verbose_level */);
	cout << "Orbit length statistic:" << endl;
	T.print_first(FALSE /* f_backwards */);
	cout << endl;


	FREE_lint(set);

	if (f_v) {
		cout << "orbit_cascade::init done" << endl;
	}

}


void orbit_cascade::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int po, so, f;
	int i, a, b;


	if (f_v) {
		cout << "orbit_cascade::downstep" << endl;
	}
	Flag_orbits = NEW_OBJECT(invariant_relations::flag_orbits);
	Flag_orbits->init(G->A_base, G->A,
			number_primary_orbits /* nb_primary_orbits_lower */,
		N /* pt_representation_sz */,
		nb_orbits_secondary_total /* nb_flag_orbits */,
		1 /* upper_bound_for_number_of_traces */, // ToDo
		NULL /* void (*func_to_free_received_trace)(void *trace_result, void *data, int verbose_level) */,
		NULL /* void (*func_latex_report_trace)(std::ostream &ost, void *trace_result, void *data, int verbose_level)*/,
		NULL /* void *free_received_trace_data */,
		verbose_level);

	long int *primary_data1; // [N]
	long int *primary_data2; // [N]
	long int *secondary_data1; // [N]


	primary_data1 = NEW_lint(N);
	primary_data2 = NEW_lint(N);

	secondary_data1 = NEW_lint(N);

	if (f_v) {
		cout << "orbit_cascade::downstep "
				"initializing flag orbits" << endl;
	}
	if (f_v) {
		f = 0;
		for (po = 0; po < number_primary_orbits; po++) {

			Lint_vec_copy(Reps_and_complements + po * N, primary_data1, N);

			for (so = 0; so < nb_orbits_secondary[po]; so++, f++) {
				data_structures_groups::set_and_stabilizer *R;
				ring_theory::longinteger_object ol;
				ring_theory::longinteger_object go;

				R = orbits_secondary_poset[po]->get_set_and_stabilizer(
						k /* level */,
						so /* orbit_at_level */,
						0 /* verbose_level */);

				orbits_secondary_poset[po]->orbit_length(
						so /* node */,
						k /* level */,
						ol);

				R->Strong_gens->group_order(go);

				Lint_vec_copy(R->data, secondary_data1, k);
				Lint_vec_complement_to(secondary_data1, secondary_data1 + k, N - k, k);
				for (i = 0; i < k; i++) {
					primary_data2[i] = primary_data1[i];
				}
				for (i = 0; i < N - k; i++) {
					a = secondary_data1[i];
					b = primary_data1[k + a];
					primary_data2[k + i] = b;
				}


				cout << f << " = (" << po << "," << so << ")";
				Lint_vec_print(cout, primary_data2, N);
				cout << " : ";
				cout << go;
				cout << endl;
				FREE_OBJECT(R);
			}
		}
	}
	f = 0;
	for (po = 0; po < number_primary_orbits; po++) {
		if (f_v) {
			cout << "orbit_cascade::downstep "
					"initializing po " << po << " / "
					<< number_primary_orbits << endl;
		}

		Lint_vec_copy(Reps_and_complements + po * N, primary_data1, N);


		for (so = 0; so < nb_orbits_secondary[po]; so++, f++) {

			data_structures_groups::set_and_stabilizer *R;
			ring_theory::longinteger_object ol;
			ring_theory::longinteger_object go;

			R = orbits_secondary_poset[po]->get_set_and_stabilizer(
					k /* level */,
					so /* orbit_at_level */,
					0 /* verbose_level */);

			orbits_secondary_poset[po]->orbit_length(
					so /* node */,
					k /* level */,
					ol);

			R->Strong_gens->group_order(go);


			// prepare the data set to use the original numbers:

			Lint_vec_copy(R->data, secondary_data1, k);
			Lint_vec_complement_to(secondary_data1, secondary_data1 + k, N - k, k);
			for (i = 0; i < k; i++) {
				primary_data2[i] = primary_data1[i];
			}
			for (i = 0; i < N - k; i++) {
				a = secondary_data1[i];
				b = primary_data1[k + a];
				primary_data2[k + i] = b;
			}

			Flag_orbits->Flag_orbit_node[f].init(
				Flag_orbits, f /* flag_orbit_index */,
				0 /* downstep_primary_orbit */,
				so /* downstep_secondary_orbit */,
				ol.as_int() /* downstep_orbit_len */,
				FALSE /* f_long_orbit */,
				primary_data2 /* int *pt_representation */,
				R->Strong_gens,
				verbose_level - 2);
			R->Strong_gens = NULL;
			FREE_OBJECT(R);
			if (f_v) {
				cout << "flag orbit " << f << " / "
						<< nb_orbits_secondary_total
						<< " is secondary orbit "
						<< so << " / " << nb_orbits_secondary[po]
						<< " stab order " << go << endl;
			}
		}
		if (f_v) {
			cout << "orbit_cascade::downstep "
					"initializing po done" << endl;
		}
	}

	if (f_v) {
		cout << "orbit_cascade::downstep "
				"initializing flag orbits done" << endl;
	}


	FREE_lint(primary_data1);
	FREE_lint(primary_data2);

	FREE_lint(secondary_data1);


	if (f_v) {
		cout << "orbit_cascade::downstep done" << endl;
	}
}

void orbit_cascade::upstep(std::vector<long int> &Ago, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_cascade::upstep" << endl;
	}

	int *f_processed;
	int nb_processed, po, so, f;
	long int *partition1; // [N]
	long int *partition2; // [N]
	long int *partition3; // [N]
	long int *primary_data; // [N]
	long int *primary_data_inv; // [N]
	int *Elt1;
	int *Elt2;
	int *Elt3;

	int po1, so1, f1;

	int i, j;


	f_processed = NEW_int(nb_orbits_secondary_total);
	Int_vec_zero(f_processed, nb_orbits_secondary_total);
	nb_processed = 0;

	partition1 = NEW_lint(N);
	partition2 = NEW_lint(N);
	partition3 = NEW_lint(N);
	primary_data = NEW_lint(N);
	primary_data_inv = NEW_lint(N);

	Elt1 = NEW_int(G->A_base->elt_size_in_int);
	Elt2 = NEW_int(G->A_base->elt_size_in_int);
	Elt3 = NEW_int(G->A_base->elt_size_in_int);

	Partition_orbits = NEW_OBJECT(invariant_relations::classification_step);

	ring_theory::longinteger_object go;

	G->Subgroup_gens->group_order(go);

	//G->A_base->group_order(go);

	Partition_orbits->init(G->A_base, G->A,
			nb_orbits_secondary_total,
			N /* representation_sz */,
			go, verbose_level);


	// process all flag orbits:

	for (f = 0; f < nb_orbits_secondary_total; f++) {

		double progress;

		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) / (double) nb_orbits_secondary_total;

		if (f_v) {
			cout << "Defining n e w orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " from flag orbit " << f << " / "
					<< nb_orbits_secondary_total
					<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;


		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		Lint_vec_copy(Flag_orbits->Pt + f * N, partition1, N);

		if (f_v) {
			cout << "orbit_cascade::upstep f=" << f << " initializing partition: ";
			Lint_vec_print(cout, partition1, N);
			cout << endl;
		}


		// coset reps:

		data_structures_groups::vector_ge *coset_reps;
		int nb_coset_reps;
		int max_nb_cosets = 6;

		coset_reps = NEW_OBJECT(data_structures_groups::vector_ge);
		coset_reps->init(G->A_base, verbose_level - 2);
		coset_reps->allocate(max_nb_cosets, verbose_level - 2);



		groups::strong_generators *S;
		ring_theory::longinteger_object go;

		if (f_v) {
			cout << "orbit_cascade::upstep f=" << f << " before gens->create_copy" << endl;
		}


		if (Flag_orbits->Flag_orbit_node[f].gens == NULL) {
			cout << "orbit_cascade::upstep NULL pointer" << endl;
			exit(1);
		}

		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy(verbose_level - 2);
		S->group_order(go);


		if (f_v) {
			cout << "orbit_cascade::upstep flag orbit "
							<< f << "=(" << po << "," << so << ") go=" << go << endl;
		}

		nb_coset_reps = 0;
		int coset;
		int a, b, c;

		int coset_perm[] = {
				0,1,2,
				0,2,1,
				1,0,2,
				1,2,0,
				2,0,1,
				2,1,0,
		};

		for (coset = 0; coset < max_nb_cosets; coset++) {


			// perform one coset test:

			if (f_v) {
				cout << "orbit_cascade::upstep flag orbit " << f << " / " << nb_orbits_secondary_total << ", coset " << coset << " / " << max_nb_cosets << endl;
			}

			if (f_v) {
				cout << "orbit_cascade::upstep partition1: ";
				Lint_vec_print(cout, partition1, N);
				cout << endl;
			}


			// map the partition and reorder:

			a = coset_perm[coset * 3 + 0];
			b = coset_perm[coset * 3 + 1];
			c = coset_perm[coset * 3 + 2];


			if (f_v) {
				cout << "orbit_cascade::upstep flag orbit " << f << " / " << nb_orbits_secondary_total << ", coset " << coset << " / " << max_nb_cosets << " a=" << a << " b=" << b << " c=" << c << endl;
			}


			Orbits_on_primary_poset->get_Orbit_tracer()->identify(
					partition1 + a * k, k,
					Elt1, po1 /* orbit_at_level */,
					verbose_level - 3);

			G->A->map_a_set_and_reorder(
					partition1 + a * k, partition2, k,
					Elt1, 0 /* verbose_level */);
			G->A->map_a_set_and_reorder(
					partition1 + b * k, partition2 + k, k,
					Elt1, 0 /* verbose_level */);
			G->A->map_a_set_and_reorder(
					partition1 + c * k, partition2 + 2 * k, N - 2 * k,
					Elt1, 0 /* verbose_level */);

			Lint_vec_copy(Reps_and_complements + po1 * N, primary_data, N);

			if (f_v) {
				cout << "orbit_cascade::upstep primary_data: ";
				Lint_vec_print(cout, primary_data, N);
				cout << endl;
			}


			for (i = 0; i < N; i++) {
				j = primary_data[i];
				primary_data_inv[j] = i;
			}

			if (f_v) {
				cout << "orbit_cascade::upstep primary_data_inv: ";
				Lint_vec_print(cout, primary_data_inv, N);
				cout << endl;
			}


			if (f_v) {
				cout << "orbit_cascade::upstep partition2: ";
				Lint_vec_print(cout, partition2, N);
				cout << endl;
			}

			for (i = 0; i < N - k; i++) {
				j = partition2[k + i];
				partition3[i] = primary_data_inv[j] - k;
			}

			if (f_v) {
				cout << "orbit_cascade::upstep partition3: ";
				Lint_vec_print(cout, partition3, N - k);
				cout << endl;
			}

			orbits_secondary_poset[po1]->get_Orbit_tracer()->identify(
					partition3, k,
					Elt2, so1 /* orbit_at_level */,
					verbose_level - 3);

			if (f_v) {
				cout << "orbit_cascade::upstep coset " << coset << " / " << max_nb_cosets << " flag orbit (" << po1 << "," << so1 << ")" << endl;
			}


			f1 = flag_orbit_first[po1] + so1;

			if (f_v) {
				cout << "orbit_cascade::upstep coset " << coset << " / " << max_nb_cosets << " flag orbit " << f1 << "=(" << po1 << "," << so1 << ")" << endl;
			}



			G->A_base->element_mult(Elt1, Elt2, Elt3, 0);





			if (f1 == f) {
				if (f_v) {
					cout << "orbit_cascade::upstep We found an automorphism "
							"of the partition:" << endl;
					G->A_base->element_print_quick(Elt3, cout);
					cout << endl;
				}

				G->A_base->element_move(Elt3, coset_reps->ith(nb_coset_reps), 0);
				nb_coset_reps++;

				//S->add_single_generator(Elt3,
				//		2 /* group_index */, verbose_level - 2);
			}
			else {
				if (f_v) {
					cout << "orbit_cascade::upstep We are identifying with flag orbit "
							<< f1 << "=(" << po1 << "," << so1 << ")" << endl;
				}

				if (!f_processed[f1]) {
					if (f_v) {
						cout << "orbit_cascade::upstep We are identifying with flag orbit "
							<< f1 << "=(" << po1 << "," << so1 << ")" << endl;
					}
					Flag_orbits->Flag_orbit_node[f1].f_fusion_node = TRUE;
					Flag_orbits->Flag_orbit_node[f1].fusion_with = f;
					Flag_orbits->Flag_orbit_node[f1].fusion_elt =
							NEW_int(G->A_base->elt_size_in_int);
					G->A_base->element_invert(Elt3,
							Flag_orbits->Flag_orbit_node[f1].fusion_elt, 0);
					f_processed[f1] = TRUE;
					nb_processed++;
				}
				else {
					if (f_v) {
						cout << "orbit_cascade::upstep flag orbit "
							<< f1 << "=(" << po1 << "," << so1 << ") "
									"has already been fused" << endl;
					}
				}
			}

		}



		coset_reps->reallocate(nb_coset_reps, verbose_level - 2);

		groups::strong_generators *Aut_gens;
		ring_theory::longinteger_object ago;

		{

			if (f_v) {
				cout << "orbit_cascade::upstep "
						"Extending the group by a factor of "
						<< nb_coset_reps << endl;
			}
			Aut_gens = NEW_OBJECT(groups::strong_generators);
			Aut_gens->init_group_extension(S,
					coset_reps, nb_coset_reps,
					verbose_level - 2);
			if (f_v) {
				cout << "orbit_cascade::upstep "
						"Aut_gens tl = ";
				Int_vec_print(cout, Aut_gens->tl, Aut_gens->A->base_len());
				cout << endl;
			}

			Aut_gens->group_order(ago);


		}


		Ago.push_back(ago.as_lint());

		if (f_v) {
			cout << "orbit_cascade::upstep the partition has a stabilizer of order "
					<< ago << endl;
			cout << "orbit_cascade::upstep The partition stabilizer is:" << endl;
			Aut_gens->print_generators_tex(cout);
		}


		Partition_orbits->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
				Partition_orbits,
				Flag_orbits->nb_primary_orbits_upper,
				Aut_gens,
				partition1 /*Rep*/,
				NULL /* extra_data */, verbose_level);

		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;

	} // next f



	if (nb_processed != nb_orbits_secondary_total) {
		cout << "orbit_cascade::upstep nb_processed != nb_orbits_secondary_total" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "nb_orbits_secondary_total = "
				<< nb_orbits_secondary_total << endl;
		exit(1);
	}

	Partition_orbits->nb_orbits = Flag_orbits->nb_primary_orbits_upper;

	if (f_v) {
		cout << "orbit_cascade::upstep We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of partitions. The stabilizer orders are:" << endl;

		for (i = 0; i < Partition_orbits->nb_orbits; i++) {
			cout << i << " : " << Ago[i] << endl;
		}

		data_structures::tally_lint T;

		T.init_vector_lint(Ago,
				FALSE /* f_second */, 0 /* verbose_level */);
		cout << "Orbit length statistic:" << endl;
		T.print_first(FALSE /* f_backwards */);
		cout << endl;
	}


	FREE_lint(partition1);
	FREE_lint(partition2);
	FREE_lint(partition3);

	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	FREE_int(f_processed);

	if (f_v) {
		cout << "orbit_cascade::upstep done" << endl;
	}
}


}}}

