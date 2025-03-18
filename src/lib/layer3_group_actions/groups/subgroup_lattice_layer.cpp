/*
 * subgroup_lattice_layer.cpp
 *
 *  Created on: Aug 31, 2024
 *      Author: betten
 */



#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace groups {



subgroup_lattice_layer::subgroup_lattice_layer()
{
	Record_birth();
	Subgroup_lattice = NULL;

	layer_idx = 0;

	//std::vector<long int> Divisors;

	Hash_table_subgroups = NULL;

	A_on_groups = NULL;

	Sch_on_groups = NULL;


}


subgroup_lattice_layer::~subgroup_lattice_layer()
{
	Record_death();
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::~subgroup_lattice_layer layer_idx = " << layer_idx << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice_layer::~subgroup_lattice_layer done" << endl;
	}
}

void subgroup_lattice_layer::init(
		subgroup_lattice *Subgroup_lattice,
		int layer_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::init" << endl;
	}

	subgroup_lattice_layer::Subgroup_lattice = Subgroup_lattice;
	subgroup_lattice_layer::layer_idx = layer_idx;


	Hash_table_subgroups = NEW_OBJECT(data_structures_groups::hash_table_subgroups);

	if (f_v) {
		cout << "subgroup_lattice_layer::init done" << endl;
	}
}

groups::subgroup *subgroup_lattice_layer::get_subgroup(
		int group_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = Hash_table_subgroups->get_subgroup(group_idx);
	//Subgroup = (groups::subgroup *) Subgroups[group_idx];

	return Subgroup;
}

groups::subgroup *subgroup_lattice_layer::get_subgroup_by_orbit(
		int orbit_idx, int group_in_orbit_idx)
{
	groups::subgroup *Subgroup;
	int group_idx;


	group_idx = Sch_on_groups->orbit[Sch_on_groups->orbit_first[orbit_idx] + group_in_orbit_idx];
	Subgroup = Hash_table_subgroups->get_subgroup(group_idx);

	return Subgroup;
}

int subgroup_lattice_layer::get_orbit_length(
		int orbit_idx)
{
	int len;


	len = Sch_on_groups->orbit_len[orbit_idx];

	return len;
}


void subgroup_lattice_layer::print(
		std::ostream &ost)
{
	long int *Go;
	int i;

	Go = NEW_lint(nb_subgroups());
	for (i = 0; i < nb_subgroups(); i++) {

		Go[i] = Hash_table_subgroups->get_subgroup(i)->group_order;

	}

	other::data_structures::tally_lint T;

	T.init(
			Go,
			nb_subgroups(), false /* f_second */,
			0 /* verbose_level */);

	cout << "layer " << layer_idx << " : ";
	Lint_vec_stl_print(cout, Divisors);
	cout << " : " << nb_subgroups();
	cout << " : ";
	T.print_bare(true /*f_backwards*/);
	cout << endl;

	FREE_lint(Go);
}

int subgroup_lattice_layer::add_subgroup(
		groups::subgroup *Subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::add_subgroup" << endl;
	}
	int f_new_group;

	f_new_group = Hash_table_subgroups->add_subgroup(Subgroup, verbose_level);



	if (f_v) {
		cout << "subgroup_lattice_layer::add_subgroup done" << endl;
	}
	return f_new_group;
}

int subgroup_lattice_layer::find_subgroup(
		groups::subgroup *Subgroup,
		int &pos, uint32_t &hash, int verbose_level)
{
	int f_found;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup" << endl;
	}
	f_found = Hash_table_subgroups->find_subgroup(Subgroup, pos, hash, verbose_level - 1);
	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup done" << endl;
	}

	return f_found;
}


int subgroup_lattice_layer::find_subgroup_direct(
		int *Elements, int group_order,
		int &pos, uint32_t &hash, int verbose_level)
{
	int f_found;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup_direct" << endl;
	}
	f_found = Hash_table_subgroups->find_subgroup_direct(
			Elements, group_order, pos, hash, verbose_level - 1);
	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup_direct done" << endl;
	}
	return f_found;
}

void subgroup_lattice_layer::group_global_to_orbit_and_group_local(
		int group_idx_global, int &orb, int &group_idx_local,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::group_global_to_orbit_and_group_local" << endl;
	}

	int coset;

	coset = Sch_on_groups->orbit_inv[group_idx_global];
	for (orb = 0; orb < Sch_on_groups->nb_orbits; orb++) {
		if (Sch_on_groups->orbit_first[orb] <= coset &&
				Sch_on_groups->orbit_first[orb] + Sch_on_groups->orbit_len[orb] > coset) {
			break;
		}
	}
	if (orb == Sch_on_groups->nb_orbits) {
		cout << "subgroup_lattice_layer::group_global_to_orbit_and_group_local not found" << endl;
		exit(1);
	}

	group_idx_local = coset - Sch_on_groups->orbit_first[orb];

	if (f_v) {
		cout << "subgroup_lattice_layer::group_global_to_orbit_and_group_local done" << endl;
	}
}


int subgroup_lattice_layer::nb_subgroups()
{
	return Hash_table_subgroups->Subgroups.size();
}

int subgroup_lattice_layer::nb_orbits()
{
	return Sch_on_groups->nb_orbits;
}

void subgroup_lattice_layer::orbits_under_conjugation(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << endl;
	}

	if (f_vv) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << " SG=" << endl;
		Subgroup_lattice->SG->print_generators(cout, verbose_level);
	}

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " before creating A_on_groups" << endl;
	}


	A_on_groups = Subgroup_lattice->A->Induced_action->create_induced_action_on_subgroups(
			Subgroup_lattice->Sims,
		Hash_table_subgroups,
		verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " after creating A_on_groups" << endl;
	}

	int print_interval = 10000;

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " before SG->compute_all_point_orbits_schreier" << endl;
	}
	Sch_on_groups = Subgroup_lattice->SG->compute_all_point_orbits_schreier(
			A_on_groups, print_interval, verbose_level - 2);

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " after SG->compute_all_point_orbits_schreier" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
				<< " The conjugacy classes of groups have the following lengths: ";
		Sch_on_groups->print_orbit_length_distribution(cout);
	}

	if (f_v) {
		cout << "subgroup_lattice_layer::orbits_under_conjugation "
				"layer = " << layer_idx << " done" << endl;
	}
}


int subgroup_lattice_layer::extend_layer(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::extend_layer "
				"layer = " << layer_idx << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx
				<< " before orbits_under_conjugation" << endl;
	}
	orbits_under_conjugation(
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::extend_layer layer = " << layer_idx
				<< " after orbits_under_conjugation" << endl;
	}

	int group_idx;
	int *Nb_new_groups;
	int nb_new_groups_total = 0;

	Nb_new_groups = NEW_int(nb_subgroups());

	for (group_idx = 0; group_idx < nb_subgroups(); group_idx++) {
		if (f_v) {
			cout << "subgroup_lattice::extend_layer layer = " << layer_idx
					<< " before extend_group " << group_idx << " / " << nb_subgroups() << endl;
		}
		Nb_new_groups[group_idx] = extend_group(group_idx, verbose_level - 2);
		nb_new_groups_total += Nb_new_groups[group_idx];
		if (f_v) {
			cout << "subgroup_lattice::extend_layer layer = " << layer_idx
					<< " after extend_group " << group_idx << " / " << nb_subgroups()
					<< " : nb_new_groups = " << Nb_new_groups[group_idx]
					<< " : " << nb_new_groups_total << endl;
			Subgroup_lattice->print();
		}

		std::string fname;

		fname = Subgroup_lattice->label_txt + "_subgroup_lattice_work.csv";

		if (f_v) {
			cout << "subgroup_lattice::extend_layer "
					"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
					<< " before save_csv" << endl;
		}
		Subgroup_lattice->save_csv(
				fname,
				verbose_level - 2);
		if (f_v) {
			cout << "subgroup_lattice::extend_layer "
					"layer " << layer_idx << " / " << Subgroup_lattice->nb_layers
					<< " after save_csv" << endl;
		}

	}

	if (f_v) {
		cout << "subgroup_lattice::extend_layer "
				"layer = " << layer_idx << " done" << endl;
	}
	return nb_new_groups_total;
}

int subgroup_lattice_layer::extend_group(
		int group_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::extend_group" << endl;
	}

	groups::subgroup *Subgroup1;

	Subgroup1 = get_subgroup(group_idx);

	int z;
	int *gens;
	int nb_gens;
	int *cosets;
	int *group;
	int group_sz;
	int nb_new_groups;

	cosets = NEW_int(Subgroup_lattice->group_order);
	group = NEW_int(Subgroup_lattice->group_order);
	gens = NEW_int(Subgroup_lattice->group_order);

	nb_new_groups = 0;

	for (z = 0; z < Subgroup_lattice->Zuppos.size(); z++) {

		int f_new_group;

		nb_gens = Subgroup1->nb_gens;
		Int_vec_copy(Subgroup1->gens, gens, Subgroup1->nb_gens);


		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size() << ":" << endl;
		}
		Subgroup_lattice->Sims->dimino(
				Subgroup1->Elements,
				Subgroup1->group_order,
				gens, nb_gens,
			cosets,
			Subgroup_lattice->Zuppos[z] /* new_gen*/,
			group, group_sz,
			0 /* verbose_level */);

		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": found a group of order " << group_sz << " : generators: ";
			Int_vec_print(cout, gens, nb_gens);
			cout << endl;
		}

		groups::subgroup *Subgroup;

		Subgroup = NEW_OBJECT(groups::subgroup);

		Subgroup->init(
				Subgroup_lattice,
				group, group_sz, gens, nb_gens,
				verbose_level - 2);

		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": before Subgroup_lattice->add_subgroup" << endl;
		}
		f_new_group = Subgroup_lattice->add_subgroup(
				Subgroup,
				verbose_level - 2);
		if (f_v) {
			cout << "subgroup_lattice::extend_group layer " << layer_idx
					<< " group " << group_idx << " / " << nb_subgroups() <<
					" zuppo " << z << " / " << Subgroup_lattice->Zuppos.size()
					<< ": after Subgroup_lattice->add_subgroup f_new_group=" << f_new_group << endl;
		}
		if (!f_new_group) {
			FREE_OBJECT(Subgroup);
		}
		else {
			nb_new_groups++;
		}

	}

	FREE_int(cosets);
	FREE_int(group);
	FREE_int(gens);


	if (f_v) {
		cout << "subgroup_lattice::extend_group done" << endl;
	}
	return nb_new_groups;
}

void subgroup_lattice_layer::do_export_to_string(
		std::string *&Table, int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::do_export_to_string" << endl;
	}

	nb_rows = Sch_on_groups->nb_orbits;
	nb_cols = 4;

	int i;

	Table = new std::string [nb_rows * nb_cols];
	for (i = 0; i < nb_rows; i++) {

		Table[i * nb_cols + 0] = std::to_string(i);

		groups::subgroup *Subgroup;

		int fst, len, idx;

		fst = Sch_on_groups->orbit_first[i];
		len = Sch_on_groups->orbit_len[i];
		idx = Sch_on_groups->orbit[fst];

		Subgroup = Hash_table_subgroups->get_subgroup(idx);

		Table[i * nb_cols + 1] = std::to_string(Subgroup->group_order);
		Table[i * nb_cols + 2] = std::to_string(len);
		Table[i * nb_cols + 3] = "\"" + Int_vec_stringify(Sch_on_groups->orbit + fst, len) + "\"";;
	}

	if (f_v) {
		cout << "subgroup_lattice::do_export_to_string done" << endl;
	}
}



}}}



