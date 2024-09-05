/*
 * subgroup_lattice.cpp
 *
 *  Created on: Aug 31, 2024
 *      Author: betten
 */




#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



subgroup_lattice::subgroup_lattice()
{

	A = NULL;

	Sims = NULL;

	//std::string label_txt;
	//std::string label_tex;


	SG = NULL;
	group_order = 0;

	gens = NULL;
	nb_gens = 0;

	//std::vector<long int> Zuppos;

	nb_layers = 0;

	//std::vector<long int> Divisors;

	Subgroup_lattice_layer = NULL;

}



subgroup_lattice::~subgroup_lattice()
{
	int i;

	if (Subgroup_lattice_layer) {
		for (i = 0; i < nb_layers; i++) {

			FREE_OBJECT(Subgroup_lattice_layer[i]);
		}
		FREE_pvoid((void **) Subgroup_lattice_layer);
	}
}

void subgroup_lattice::init(
		actions::action *A,
		sims *Sims,
		std::string &label_txt,
		std::string &label_tex,
		strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::init" << endl;
	}

	subgroup_lattice::A = A;
	subgroup_lattice::Sims = Sims;

	subgroup_lattice::label_txt = label_txt;
	subgroup_lattice::label_tex = label_tex;

	subgroup_lattice::SG = SG;

	group_order = SG->group_order_as_lint();

	if (f_v) {
		cout << "subgroup_lattice::init label_txt = " << label_txt << endl;
		cout << "subgroup_lattice::init group_order = " << group_order << endl;
	}

	number_theory::number_theory_domain NT;

	nb_layers = NT.nb_prime_factors_counting_multiplicities(
			group_order) + 1;

	if (f_v) {
		cout << "subgroup_lattice::init nb_layers = " << nb_layers << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::init "
				"before NT.all_divisors" << endl;
	}
	NT.all_divisors(group_order, Divisors);
	if (f_v) {
		cout << "subgroup_lattice::init "
				"after NT.all_divisors" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice::init Divisors:" << endl;
		Lint_vec_stl_print(cout, Divisors);
		cout << endl;
	}

	Subgroup_lattice_layer = (subgroup_lattice_layer **) NEW_pvoid(nb_layers);

	int i;

	if (f_v) {
		cout << "subgroup_lattice::init "
				"before Subgroup_lattice_layer[i]->init" << endl;
	}
	for (i = 0; i < nb_layers; i++) {
		Subgroup_lattice_layer[i] = NEW_OBJECT(subgroup_lattice_layer);

		Subgroup_lattice_layer[i]->init(this, i, verbose_level - 1);
	}
	if (f_v) {
		cout << "subgroup_lattice::init "
				"after Subgroup_lattice_layer[i]->init" << endl;
	}

	long int d;
	int idx;

	for (i = 0; i < Divisors.size(); i++) {
		d = Divisors[i];
		idx = NT.nb_prime_factors_counting_multiplicities(
				d);

		Subgroup_lattice_layer[idx]->Divisors.push_back(d);

	}

	if (f_v) {
		cout << "subgroup_lattice::init lattice:" << endl;
		print();
	}


	if (f_v) {
		cout << "subgroup_lattice::init "
				"before Sims->zuppo_list" << endl;
	}

	Sims->zuppo_list(
			Zuppos, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::init "
				"after Sims->zuppo_list" << endl;
	}

	if (f_v) {
		print_zuppos(verbose_level);
	}


	if (f_v) {
		cout << "subgroup_lattice::init "
				"before adding trivial subgroup" << endl;
	}

	groups::subgroup *Subgroup;

	Subgroup = NEW_OBJECT(groups::subgroup);

	Subgroup->init_trivial_subgroup(
			this);

	add_subgroup(
			Subgroup,
			verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::init "
				"after adding trivial subgroup" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::init lattice:" << endl;
		print();
	}


	if (f_v) {
		cout << "subgroup_lattice::init "
				"before extend_all_layers" << endl;
	}

	extend_all_layers(
			verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::init "
				"after extend_all_layers" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::init done" << endl;
	}
}

groups::subgroup *subgroup_lattice::get_subgroup(
		int layer_idx, int group_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = (groups::subgroup *) Subgroup_lattice_layer[layer_idx]->get_subgroup(group_idx);

	return Subgroup;
}


void subgroup_lattice::extend_all_layers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::extend_all_layers" << endl;
	}

	int layer_idx;
	int *Nb_new_groups;

	Nb_new_groups = NEW_int(nb_layers);
	Int_vec_zero(Nb_new_groups, nb_layers);

	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {
		if (f_v) {
			cout << "subgroup_lattice::extend_all_layers "
					"layer " << layer_idx << " / " << nb_layers << endl;
		}




		Nb_new_groups[layer_idx] = Subgroup_lattice_layer[layer_idx]->extend_layer(verbose_level);
		if (f_v) {
			cout << "subgroup_lattice::extend_all_layers "
					"layer " << layer_idx << " / " << nb_layers
					<< " done with " << Nb_new_groups[layer_idx]
					<< " new groups" << endl;
		}

		if (f_v) {
			cout << "subgroup_lattice::extend_all_layers "
					"layer " << layer_idx << " / " << nb_layers
					<< " before save_csv" << endl;
		}
		save_csv(
				verbose_level);
		if (f_v) {
			cout << "subgroup_lattice::extend_all_layers "
					"layer " << layer_idx << " / " << nb_layers
					<< " after save_csv" << endl;
		}


	}



	if (f_v) {
		cout << "subgroup_lattice::extend_all_layers Nb_new_groups=";
		Int_vec_print(cout, Nb_new_groups, nb_layers);
		cout << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::extend_all_layers done" << endl;
	}
}

void subgroup_lattice::print()
{
	int i;
	int nb_groups;

	nb_groups = 0;
	cout << "subgroup_lattice::init layers" << endl;
	for (i = 0; i < nb_layers; i++) {
		cout << "layer " << i << " : ";
		Subgroup_lattice_layer[i]->print(cout);
		nb_groups += Subgroup_lattice_layer[i]->nb_subgroups();
	}
	cout << "subgroup_lattice::init total number of groups = " << nb_groups << endl;

}

int subgroup_lattice::number_of_groups_total()
{
	int i;
	int nb_groups;

	nb_groups = 0;
	for (i = 0; i < nb_layers; i++) {
		nb_groups += Subgroup_lattice_layer[i]->nb_subgroups();
	}
	return nb_groups;

}

void subgroup_lattice::save_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::save_csv" << endl;
	}
	int layer_idx, group_idx;
	int nb_g, cnt;
	int nb_r, nb_c;
	int nb_groups;
	std::string *Headers;
	std::string *Table;

	nb_groups = number_of_groups_total();
	nb_r = nb_groups;
	nb_c = 8;
	Headers = new std::string[nb_c];
	Table = new std::string[nb_r * nb_c];

	Headers[0] = "subgroup_idx";
	Headers[1] = "layer";
	Headers[2] = "idx_in_layer";
	Headers[3] = "group_order";
	Headers[4] = "prev";
	Headers[5] = "label";
	Headers[6] = "generators";
	Headers[7] = "elements";


	cnt = 0;
	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {

		if (f_v) {
			cout << "subgroup_lattice::save_csv layer " << layer_idx << " / " << nb_layers << endl;
		}

		nb_g = Subgroup_lattice_layer[layer_idx]->nb_subgroups();
		for (group_idx = 0; group_idx < nb_g; group_idx++, cnt++) {
			Table[cnt * nb_c + 0] = std::to_string(cnt);
			Table[cnt * nb_c + 1] = std::to_string(layer_idx);
			Table[cnt * nb_c + 2] = std::to_string(group_idx);

			groups::subgroup *Subgroup;

			Subgroup = get_subgroup(layer_idx, group_idx);


			Table[cnt * nb_c + 3] = std::to_string(Subgroup->group_order);


			int coset, previous, label;
			if (Subgroup_lattice_layer[layer_idx]->Sch_on_groups) {

				coset = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->prev[group_idx];
				label = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->label[group_idx];
				previous = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_inv[coset];
			}
			else {
				coset = -1;
				previous = -1;
				label = -1;
			}

			Table[cnt * nb_c + 4] = std::to_string(previous);
			Table[cnt * nb_c + 5] = std::to_string(label);
			Table[cnt * nb_c + 6] = "\"" + Int_vec_stringify(Subgroup->gens, Subgroup->nb_gens) + "\"";
			Table[cnt * nb_c + 7] = "\"" + Int_vec_stringify(Subgroup->Elements, Subgroup->group_order) + "\"";

		}
	}

	orbiter_kernel_system::file_io Fio;

	std::string fname;

	fname = label_txt + "_subgroup_lattice.csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_c, Table,
			Headers,
			verbose_level - 1);

	if (f_v) {
		cout << "subgroup_lattice::save_csv "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	delete [] Headers;
	delete [] Table;


	if (f_v) {
		cout << "subgroup_lattice::save_csv done" << endl;
	}

}


int subgroup_lattice::add_subgroup(
		groups::subgroup *Subgroup,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::add_subgroup" << endl;
	}

	long int order;
	int layer_idx;
	int f_new_group;

	order = Subgroup->group_order;

	number_theory::number_theory_domain NT;

	layer_idx = NT.nb_prime_factors_counting_multiplicities(order);

	f_new_group = Subgroup_lattice_layer[layer_idx]->add_subgroup(Subgroup, verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::add_subgroup done" << endl;
	}
	return f_new_group;
}

void subgroup_lattice::print_zuppos(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::print_zuppos" << endl;
	}

	int i, ord;
	int *Elt1;
	int *data;

	Elt1 = NEW_int(A->elt_size_in_int);
	data = NEW_int(A->make_element_size);

	cout << "Number of zuppos: " << Zuppos.size() << endl;

	cout << "index : zuppo : order : element coding" << endl;

	for (i = 0; i < Zuppos.size(); i++) {

		Sims->element_unrank_lint(Zuppos[i], Elt1, 0 /*verbose_level*/);

		A->Group_element->element_code_for_make_element(
				Elt1, data);

		ord = A->Group_element->element_order(Elt1);

		cout << i << " / " << Zuppos.size() << " : " << Zuppos[i] << " : " << ord << " : ";
		Int_vec_print(cout, data, A->make_element_size);
		cout << endl;
	}

	FREE_int(Elt1);
	FREE_int(data);

	if (f_v) {
		cout << "subgroup_lattice::print_zuppos done" << endl;
	}
}

}}}



