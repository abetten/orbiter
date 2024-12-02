/*
 * subgroup_lattice.cpp
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



subgroup_lattice::subgroup_lattice()
{
	Record_birth();

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
	Record_death();
	int verbose_level = 0;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::~subgroup_lattice" << endl;
	}
	int i;

	if (Subgroup_lattice_layer) {
		for (i = 0; i < nb_layers; i++) {

			FREE_OBJECT(Subgroup_lattice_layer[i]);
		}
		FREE_pvoid((void **) Subgroup_lattice_layer);
	}
	if (f_v) {
		cout << "subgroup_lattice::~subgroup_lattice done" << endl;
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


	if (f_v) {
		cout << "subgroup_lattice::init "
				"before init_basic" << endl;
	}
	init_basic(A, Sims, label_txt, label_tex, SG, verbose_level);
	if (f_v) {
		cout << "subgroup_lattice::init "
				"after init_basic" << endl;
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
		cout << "subgroup_lattice::init done" << endl;
	}
}


void subgroup_lattice::init_basic(
		actions::action *A,
		sims *Sims,
		std::string &label_txt,
		std::string &label_tex,
		strong_generators *SG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::init_basic" << endl;
	}

	subgroup_lattice::A = A;
	subgroup_lattice::Sims = Sims;

	subgroup_lattice::label_txt = label_txt;
	subgroup_lattice::label_tex = label_tex;

	subgroup_lattice::SG = SG;

	group_order = SG->group_order_as_lint();

	if (f_v) {
		cout << "subgroup_lattice::init_basic label_txt = " << label_txt << endl;
		cout << "subgroup_lattice::init_basic group_order = " << group_order << endl;
	}

	algebra::number_theory::number_theory_domain NT;

	nb_layers = NT.nb_prime_factors_counting_multiplicities(
			group_order) + 1;

	if (f_v) {
		cout << "subgroup_lattice::init_basic nb_layers = " << nb_layers << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::init_basic "
				"before NT.all_divisors" << endl;
	}
	NT.all_divisors(group_order, Divisors);
	if (f_v) {
		cout << "subgroup_lattice::init_basic "
				"after NT.all_divisors" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice::init_basic Divisors:" << endl;
		Lint_vec_stl_print(cout, Divisors);
		cout << endl;
	}

	Subgroup_lattice_layer = (subgroup_lattice_layer **) NEW_pvoid(nb_layers);

	int i;

	if (f_v) {
		cout << "subgroup_lattice::init_basic "
				"before Subgroup_lattice_layer[i]->init" << endl;
	}
	for (i = 0; i < nb_layers; i++) {
		Subgroup_lattice_layer[i] = NEW_OBJECT(subgroup_lattice_layer);

		Subgroup_lattice_layer[i]->init(this, i, verbose_level - 1);
	}
	if (f_v) {
		cout << "subgroup_lattice::init_basic "
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
		cout << "subgroup_lattice::init_basic lattice:" << endl;
		print();
	}


	if (f_v) {
		cout << "subgroup_lattice::init_basic done" << endl;
	}
}




void subgroup_lattice::compute(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::compute" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::compute "
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
		cout << "subgroup_lattice::compute "
				"after adding trivial subgroup" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::compute lattice:" << endl;
		print();
	}


	if (f_v) {
		cout << "subgroup_lattice::compute "
				"before extend_all_layers" << endl;
	}

	extend_all_layers(
			verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::compute "
				"after extend_all_layers" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::compute done" << endl;
	}
}

groups::subgroup *subgroup_lattice::get_subgroup(
		int layer_idx, int group_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = (groups::subgroup *) Subgroup_lattice_layer[layer_idx]->get_subgroup(group_idx);

	return Subgroup;
}

groups::subgroup *subgroup_lattice::get_subgroup_by_orbit(
		int layer_idx, int orbit_idx, int group_in_orbit_idx)
{
	groups::subgroup *Subgroup;

	Subgroup = (groups::subgroup *) Subgroup_lattice_layer[layer_idx]->get_subgroup_by_orbit(orbit_idx, group_in_orbit_idx);

	return Subgroup;
}

void subgroup_lattice::conjugacy_classes(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::conjugacy_classes" << endl;
	}


	int layer_idx;

	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {
		if (f_v) {
			cout << "subgroup_lattice::conjugacy_classes "
					"layer " << layer_idx << " / " << nb_layers << endl;
		}
		if (f_v) {
			cout << "subgroup_lattice::conjugacy_classes layer = " << layer_idx
					<< " before orbits_under_conjugation" << endl;
		}
		Subgroup_lattice_layer[layer_idx]->orbits_under_conjugation(
				verbose_level);
		if (f_v) {
			cout << "subgroup_lattice::conjugacy_classes layer = " << layer_idx
					<< " after orbits_under_conjugation" << endl;
		}
	}

	if (f_v) {
		cout << "subgroup_lattice::conjugacy_classes done" << endl;
	}
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

		std::string fname;

		fname = label_txt + "_subgroup_lattice_after_layer_" + std::to_string(layer_idx) + ".csv";

		if (f_v) {
			cout << "subgroup_lattice::extend_all_layers "
					"layer " << layer_idx << " / " << nb_layers
					<< " before save_csv" << endl;
		}
		save_csv(
				fname,
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
	int nb_groups, nb_orbits;
	int f_has_orbits;

	nb_groups = 0;
	nb_orbits = 0;
	cout << "subgroup_lattice::print layers" << endl;


	for (i = 0; i < nb_layers; i++) {
		cout << "layer " << i << " : ";
		Subgroup_lattice_layer[i]->print(cout);
		nb_groups += Subgroup_lattice_layer[i]->nb_subgroups();

		if (Subgroup_lattice_layer[i]->Sch_on_groups) {
			f_has_orbits = true;
		}
		else {
			f_has_orbits = false;
		}

		if (f_has_orbits) {
			nb_orbits += Subgroup_lattice_layer[i]->nb_orbits();
		}
	}

	cout << "subgroup_lattice::print total number of groups = " << nb_groups << endl;
	cout << "subgroup_lattice::print total number of orbits = " << nb_orbits << endl;

}

void subgroup_lattice::make_partition_by_layers(
		int *&first, int *&length, int &nb_parts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_layers" << endl;
	}


	nb_parts = nb_layers;
	first = NEW_int(nb_layers + 1);
	length = NEW_int(nb_layers);

	int i;
	int nb_groups;

	first[0] = 0;
	for (i = 0; i < nb_layers; i++) {
		nb_groups = Subgroup_lattice_layer[i]->nb_subgroups();
		length[i] = nb_groups;
		first[i + 1] = first[i] + nb_groups;
	}

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_layers first=" << endl;
		Int_vec_print(cout, first, nb_layers + 1);
		cout << endl;
		cout << "subgroup_lattice::make_partition_by_layers length=" << endl;
		Int_vec_print(cout, length, nb_layers);
		cout << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_layers done" << endl;
	}
}

void subgroup_lattice::make_partition_by_orbits(
		int *&first, int *&length, int &nb_parts, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_orbits" << endl;
	}


	nb_parts = nb_layers;
	first = NEW_int(nb_layers + 1);
	length = NEW_int(nb_layers);

	int i;
	int nb_orbits;

	first[0] = 0;
	for (i = 0; i < nb_layers; i++) {
		nb_orbits = Subgroup_lattice_layer[i]->nb_orbits();
		length[i] = nb_orbits;
		first[i + 1] = first[i] + nb_orbits;
	}

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_orbits first=" << endl;
		Int_vec_print(cout, first, nb_layers + 1);
		cout << endl;
		cout << "subgroup_lattice::make_partition_by_orbits length=" << endl;
		Int_vec_print(cout, length, nb_layers);
		cout << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::make_partition_by_orbits done" << endl;
	}
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

int subgroup_lattice::number_of_orbits_total()
{
	int i;
	int nb_orbits;

	nb_orbits = 0;
	for (i = 0; i < nb_layers; i++) {
		nb_orbits += Subgroup_lattice_layer[i]->nb_orbits();
	}
	return nb_orbits;

}

void subgroup_lattice::save_csv(
		std::string &fname,
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
			cout << "subgroup_lattice::save_csv "
					"layer " << layer_idx << " / " << nb_layers << endl;
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

				coset = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_inv[group_idx];
				previous = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->prev[coset];
				label = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->label[coset];

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

	other::orbiter_kernel_system::file_io Fio;

	//std::string fname;

	//fname = label_txt + "_subgroup_lattice.csv";

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

void subgroup_lattice::save_rearranged_by_orbits_csv(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::save_rearranged_by_orbits_csv" << endl;
	}
	int nb_r, nb_c;
	int nb_groups;
	std::string *Headers;
	std::string *Table;

	nb_groups = number_of_groups_total();
	nb_r = nb_groups;
	nb_c = 10;
	Headers = new std::string[nb_c];
	Table = new std::string[nb_r * nb_c];

	Headers[0] = "group_idx_global";
	Headers[1] = "orbit_idx_global";
	Headers[2] = "layer";
	Headers[3] = "orbit";
	Headers[4] = "group_in_layer_idx";
	Headers[5] = "group_order";
	Headers[6] = "prev";
	Headers[7] = "label";
	Headers[8] = "generators";
	Headers[9] = "elements";

	int *first;
	int *length;
	int nb_parts;
	//int nb_orbits_total;
	int nb_orbits;
	int layer_idx;
	int orb_idx;
	int orbit_length;
	int group_idx;
	int cnt_orbit;
	int cnt_group;

	if (f_v) {
		cout << "subgroup_lattice::save_rearranged_by_orbits_csv "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::save_rearranged_by_orbits_csv "
				"after make_partition_by_orbits" << endl;
	}

	//nb_orbits_total = first[nb_parts];


	cnt_orbit = 0;
	cnt_group = 0;
	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {

		if (f_v) {
			cout << "subgroup_lattice::save_rearranged_by_orbits_csv "
					"layer " << layer_idx << " / " << nb_layers << endl;
		}

		nb_orbits = Subgroup_lattice_layer[layer_idx]->nb_orbits();

		int group_in_layer_idx = 0;

		for (orb_idx = 0; orb_idx < nb_orbits; orb_idx++, cnt_orbit++) {


			orbit_length = Subgroup_lattice_layer[layer_idx]->get_orbit_length(orb_idx);

			for (group_idx = 0; group_idx < orbit_length; group_idx++, cnt_group++, group_in_layer_idx++) {

				groups::subgroup *Subgroup;

				Subgroup = get_subgroup_by_orbit(layer_idx, orb_idx, group_idx);

				Table[cnt_group * nb_c + 0] = std::to_string(cnt_group);
				Table[cnt_group * nb_c + 1] = std::to_string(cnt_orbit);
				Table[cnt_group * nb_c + 2] = std::to_string(layer_idx);
				Table[cnt_group * nb_c + 3] = std::to_string(orb_idx);

				Table[cnt_group * nb_c + 4] = std::to_string(group_in_layer_idx);

				Table[cnt_group * nb_c + 5] = std::to_string(Subgroup->group_order);


				int coset, previous, label;

				if (Subgroup_lattice_layer[layer_idx]->Sch_on_groups) {

					coset = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_first[orb_idx] + group_idx;
					//coset = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_inv[group_idx];
					previous = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->prev[coset];
					if (previous >= 0) {
						previous = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_inv[previous];
							//- Subgroup_lattice_layer[layer_idx]->Sch_on_groups->orbit_first[orb_idx];
					}
					label = Subgroup_lattice_layer[layer_idx]->Sch_on_groups->label[coset];

				}
				else {
					coset = -1;
					previous = -1;
					label = -1;
				}

				Table[cnt_group * nb_c + 6] = std::to_string(previous);
				Table[cnt_group * nb_c + 7] = std::to_string(label);
				Table[cnt_group * nb_c + 8] = "\"" + Int_vec_stringify(Subgroup->gens, Subgroup->nb_gens) + "\"";
				Table[cnt_group * nb_c + 9] = "\"" + Int_vec_stringify(Subgroup->Elements, Subgroup->group_order) + "\"";
			}
		}
	}

	other::orbiter_kernel_system::file_io Fio;

	//std::string fname;

	//fname = label_txt + "_subgroup_lattice.csv";

	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_r, nb_c, Table,
			Headers,
			verbose_level - 1);

	if (f_v) {
		cout << "subgroup_lattice::save_rearranged_by_orbits_csv "
				"Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	FREE_int(first);
	FREE_int(length);
	delete [] Headers;
	delete [] Table;


	if (f_v) {
		cout << "subgroup_lattice::save_rearranged_by_orbits_csv done" << endl;
	}

}


void subgroup_lattice::load_csv(
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);

	if (f_v) {
		cout << "subgroup_lattice::load_csv" << endl;
	}

	other::orbiter_kernel_system::file_io Fio;

	other::data_structures::spreadsheet S;

	S.read_spreadsheet(fname, verbose_level);


	int nb_rows;

	nb_rows = S.nb_rows;
	if (f_v) {
		cout << "subgroup_lattice::load_csv "
				"nb_rows=" << nb_rows << endl;
	}




	std::string *Header_rows;
	std::string *Header_cols;
	std::string *T;
	int nb_r, nb_c;

	if (f_vv) {
		cout << "subgroup_lattice::load_csv "
				"before S.stringify" << endl;
	}
	S.stringify(
			Header_rows, Header_cols, T,
			nb_r, nb_c,
			verbose_level - 1);
	if (f_vv) {
		cout << "subgroup_lattice::load_csv "
				"after S.stringify" << endl;
	}

	int i, j;

	if (f_vv) {
		cout << "Header_cols" << endl;
		for (j = 0; j < nb_c; j++) {
			cout << j << " : " << Header_cols[j] << endl;
		}
		cout << "Header_rows" << endl;
		for (i = 0; i < nb_r; i++) {
			cout << i << " : " << Header_rows[i] << endl;
		}
		cout << "T" << endl;
		for (i = 0; i < nb_r; i++) {
			for (j = 0; j < nb_c; j++) {
				cout << i << "," << j << " : " << T[i * nb_c + j] << endl;

			}
		}
	}

	other::data_structures::string_tools String;
	int layer_col_idx;
	int idx_in_layer_col_idx;
	int go_col_idx;
	int prev_col_idx;
	int label_col_idx;
	int generators_col_idx;
	int elements_col_idx;
	std::string header;


	header = "layer";
	if (f_vv) {
		cout << "subgroup_lattice::load_csv "
				"before String.find_string_in_array" << endl;
	}
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, layer_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	if (f_vv) {
		cout << "subgroup_lattice::load_csv "
				"after String.find_string_in_array" << endl;
	}
	header = "idx_in_layer";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, idx_in_layer_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	header = "group_order";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, go_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	header = "prev";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, prev_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	header = "label";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, label_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	header = "generators";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, generators_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}
	header = "elements";
	if (!String.find_string_in_array(
			Header_cols, nb_c,
			header, elements_col_idx)) {
		cout << "subgroup_lattice::load_csv Cannot find column with label " << header << endl;
		exit(1);

	}

	long int *Layer;
	long int *Idx_in_layer;
	long int *Go;
	long int *Prev;
	long int *Label;
	int **Generators;
	int *Nb_generators;
	int **Elements;
	int *Nb_elements;
	int go;

	Layer = NEW_lint(nb_r);
	Idx_in_layer = NEW_lint(nb_r);
	Go = NEW_lint(nb_r);
	Prev = NEW_lint(nb_r);
	Label = NEW_lint(nb_r);
	Generators = NEW_pint(nb_r);
	Nb_generators = NEW_int(nb_r);
	Elements = NEW_pint(nb_r);
	Nb_elements = NEW_int(nb_r);

	for (i = 0; i < nb_r; i++) {
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << endl;
		}
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Layer" << endl;
		}
		Layer[i] = stoi(T[i * nb_c + layer_col_idx]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Idx_in_layer" << endl;
		}
		Idx_in_layer[i] = stoi(T[i * nb_c + idx_in_layer_col_idx]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Go" << endl;
		}
		Go[i] = stoi(T[i * nb_c + go_col_idx]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Go=" << Go[i] << endl;
		}
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Prev" << endl;
		}
		Prev[i] = stoi(T[i * nb_c + prev_col_idx]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Label" << endl;
		}
		Label[i] = stoi(T[i * nb_c + label_col_idx]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Int_vec_scan" << endl;
		}

		string s1, s2;

		s1 = T[i * nb_c + generators_col_idx];
		String.drop_quotes(s1, s2);
		Int_vec_scan(s2, Generators[i], Nb_generators[i]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Int_vec_scan" << endl;
		}
		s1 = T[i * nb_c + elements_col_idx];
		String.drop_quotes(s1, s2);
		Int_vec_scan(s2, Elements[i], Nb_elements[i]);
		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"reading row " << i << " / " << nb_r << " Nb_elements[i]=" << Nb_elements[i] << endl;
		}
	}
	go = Go[nb_r - 1];
	if (f_v) {
		cout << "subgroup_lattice::load_csv go = " << go << endl;
	}

	if (go != group_order) {
		cout << "subgroup_lattice::load_csv go != group_order" << endl;
		exit(1);
	}


	int nb_new_groups;

	nb_new_groups = 0;

	for (i = 0; i < nb_r; i++) {

		int f_new_group;
		groups::subgroup *Subgroup;

		Subgroup = NEW_OBJECT(groups::subgroup);

		Subgroup->init(
				this,
				Elements[i], Nb_elements[i], Generators[i], Nb_generators[i],
				verbose_level - 2);

		if (f_vv) {
			cout << "subgroup_lattice::load_csv "
					"before add_subgroup, i=" << i << " go=" << Subgroup->group_order << endl;
		}
		f_new_group = add_subgroup(
				Subgroup,
				verbose_level - 3);
		if (f_vv) {
			if (f_vv) {
				cout << "subgroup_lattice::load_csv "
						"after add_subgroup" << endl;
			}
		}
		if (!f_new_group) {
			cout << "subgroup_lattice::load_csv !f_new_group" << endl;
			exit(1);
			//FREE_OBJECT(Subgroup);
		}
		else {
			nb_new_groups++;
		}

	}
	if (f_v) {
		cout << "subgroup_lattice::load_csv nb_new_groups = " << nb_new_groups << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::load_csv done" << endl;
	}
}

void subgroup_lattice::create_drawing(
		combinatorics::graph_theory::layered_graph *&LG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_drawing" << endl;
	}
	int *incma;
	int *maximals;
	int nb_groups;

	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"before create_incidence_matrix" << endl;
	}
	create_incidence_matrix(
			incma, nb_groups, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"after create_incidence_matrix" << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"before reduce_to_maximals" << endl;
	}
	reduce_to_maximals(
			incma, nb_groups,
			maximals,
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"after reduce_to_maximals" << endl;
	}

	int *Nb;
	int *Fst;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"before make_partition_by_layers" << endl;
	}
	make_partition_by_layers(
				Fst, Nb, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"after make_partition_by_layers" << endl;
	}



	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"before LG->init" << endl;
	}
	//LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"after LG->init" << endl;
	}
	LG->place(verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing "
				"after LG->place" << endl;
	}

	int l1, nb_groups1, g1;
	int l2, nb_groups2, g2;
	int i, j;

	// create edges:

	for (l1 = 0; l1 < nb_layers; l1++) {
		nb_groups1 = Subgroup_lattice_layer[l1]->nb_subgroups();
		for (g1 = 0; g1 < nb_groups1; g1++) {

			for (l2 = l1 + 1; l2 < nb_layers; l2++) {
				nb_groups2 = Subgroup_lattice_layer[l2]->nb_subgroups();
				for (g2 = 0; g2 < nb_groups2; g2++) {

					i = Fst[l1] + g1;
					j = Fst[l2] + g2;

					if (incma[i * nb_groups + j]) {
						LG->add_edge(l1, g1, l2, g2,
							1, // edge_color
							0 /*verbose_level*/);
					}
				}
			}
		}
	}

	groups::subgroup *Subgroup1;

	for (l1 = 0; l1 < nb_layers; l1++) {
		nb_groups1 = Subgroup_lattice_layer[l1]->nb_subgroups();
		for (g1 = 0; g1 < nb_groups1; g1++) {
			Subgroup1 = get_subgroup(l1, g1);
			string text3;

			text3 = std::to_string(Subgroup1->group_order);
			LG->add_text(l1, g1, text3, 0/*verbose_level*/);
		}
	}


	FREE_int(Nb);
	FREE_int(Fst);

	FREE_int(incma);
	FREE_int(maximals);

	if (f_v) {
		cout << "subgroup_lattice::create_drawing done" << endl;
	}
}


void subgroup_lattice::create_drawing_by_orbits(
		combinatorics::graph_theory::layered_graph *&LG,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits" << endl;
	}
	int *incma;
	int *maximals;
	int nb_orbits;

	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"before create_incidence_matrix_for_orbits_Asup" << endl;
	}
	create_incidence_matrix_for_orbits_Asup(
			incma, nb_orbits, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"after create_incidence_matrix_for_orbits_Asup" << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"before reduce_to_maximals_for_orbits" << endl;
	}
	reduce_to_maximals_for_orbits(
			incma, nb_orbits,
			maximals,
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"after reduce_to_maximals_for_orbits" << endl;
	}

	int *Nb;
	int *Fst;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				Fst, Nb, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"after make_partition_by_orbits" << endl;
	}



	LG = NEW_OBJECT(combinatorics::graph_theory::layered_graph);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"before LG->init" << endl;
	}
	//LG->add_data1(data1, 0/*verbose_level*/);

	string dummy;
	dummy.assign("");

	LG->init(nb_layers, Nb, dummy, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"after LG->init" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"before LG->place_upside_down" << endl;
	}
	LG->place_upside_down(verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits "
				"after LG->place_upside_down" << endl;
	}

	int l1, nb_orbits1, orb1;
	int l2, nb_orbits2, orb2;
	groups::subgroup *Subgroup1;
	int i, j;

	// create edges:

	for (l1 = 0; l1 < nb_layers; l1++) {

		nb_orbits1 = Subgroup_lattice_layer[l1]->nb_orbits();

		for (orb1 = 0; orb1 < nb_orbits1; orb1++) {

			for (l2 = l1 + 1; l2 < nb_layers; l2++) {

				nb_orbits2 = Subgroup_lattice_layer[l2]->nb_orbits();

				for (orb2 = 0; orb2 < nb_orbits2; orb2++) {

					i = Fst[l1] + orb1;
					j = Fst[l2] + orb2;

					if (maximals[i * nb_orbits + j]) {
						LG->add_edge(l1, orb1, l2, orb2,
								1, // edge_color
							0 /*verbose_level*/);
					}
				}
			}
		}
	}


	for (l1 = 0; l1 < nb_layers; l1++) {

		nb_orbits1 = Subgroup_lattice_layer[l1]->nb_orbits();

		for (orb1 = 0; orb1 < nb_orbits1; orb1++) {

			Subgroup1 = get_subgroup_by_orbit(l1, orb1, 0);

			string text3;

#if 0
			text3 = "${" + std::to_string(Subgroup1->group_order) + " \\atop "
					+ std::to_string(Subgroup_lattice_layer[l1]->get_orbit_length(orb1))
					+ "}$";
			text3 = "{\\tiny $" + std::to_string(Fst[l1] + orb1) + "^{" + std::to_string(Subgroup1->group_order) + "}_{"
					+ std::to_string(Subgroup_lattice_layer[l1]->get_orbit_length(orb1))
					+ "}$}";
#endif

			text3 = "{\\tiny $" + std::to_string(Subgroup1->group_order) + "$}";
			LG->add_text(l1, orb1, text3, 0/*verbose_level*/);
		}
	}


	FREE_int(Nb);
	FREE_int(Fst);

	FREE_int(incma);
	FREE_int(maximals);

	if (f_v) {
		cout << "subgroup_lattice::create_drawing_by_orbits done" << endl;
	}
}


void subgroup_lattice::create_incidence_matrix(
		int *&incma, int &nb_groups,
		int verbose_level)
// incma[nb_groups * nb_groups]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix" << endl;
	}

	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix "
				"before make_partition_by_layers" << endl;
	}
	make_partition_by_layers(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix "
				"after make_partition_by_layers" << endl;
	}

	nb_groups = first[nb_parts];

	incma = NEW_int(nb_groups * nb_groups);
	Int_vec_zero(incma, nb_groups * nb_groups);

	int l1, g1, l2, g2;
	int nb_groups1, nb_groups2;
	groups::subgroup *Subgroup1;
	groups::subgroup *Subgroup2;
	int i, j;

	other::data_structures::sorting Sorting;


	for (l1 = 0; l1 < nb_layers; l1++) {
		nb_groups1 = Subgroup_lattice_layer[l1]->nb_subgroups();
		for (g1 = 0; g1 < nb_groups1; g1++) {
			Subgroup1 = get_subgroup(l1, g1);

			for (l2 = l1 + 1; l2 < nb_layers; l2++) {
				nb_groups2 = Subgroup_lattice_layer[l2]->nb_subgroups();
				for (g2 = 0; g2 < nb_groups2; g2++) {
					Subgroup2 = get_subgroup(l2, g2);

					if (Subgroup1->is_subgroup_of(Subgroup2)) {

						i = first[l1] + g1;
						j = first[l2] + g2;
						incma[i * nb_groups + j] = 1;
					}

				}
			}
		}
	}

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix "
				"incma=" << endl;
		Int_matrix_print(incma, nb_groups, nb_groups);

		cout << "subgroup_lattice::create_incidence_matrix "
				"incma=" << endl;
		Int_matrix_print_nonzero_entries(incma, nb_groups, nb_groups);
}

	FREE_int(first);
	FREE_int(length);

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix done" << endl;
	}
}

void subgroup_lattice::create_incidence_matrix_for_orbits_Asup(
		int *&incma, int &nb_orbits,
		int verbose_level)
// incma[nb_orbits * nb_orbits]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup" << endl;
	}

	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"after make_partition_by_orbits" << endl;
	}

	nb_orbits = first[nb_parts];

	incma = NEW_int(nb_orbits * nb_orbits);
	Int_vec_zero(incma, nb_orbits * nb_orbits);

	int l1, orb1, /*len1,*/ l2, orb2, len2;
	int nb_orbits1, nb_orbits2;
	groups::subgroup *Subgroup1;
	groups::subgroup *Subgroup2;
	int i, j, g2, cnt;

	other::data_structures::sorting Sorting;


	for (l1 = 0; l1 < nb_layers; l1++) {

		nb_orbits1 = Subgroup_lattice_layer[l1]->nb_orbits();

		for (orb1 = 0; orb1 < nb_orbits1; orb1++) {

			//len1 = Subgroup_lattice_layer[l1]->get_orbit_length(orb1);

			Subgroup1 = get_subgroup_by_orbit(l1, orb1, 0);

			for (l2 = l1 + 1; l2 < nb_layers; l2++) {

				nb_orbits2 = Subgroup_lattice_layer[l2]->nb_orbits();

				for (orb2 = 0; orb2 < nb_orbits2; orb2++) {

					len2 = Subgroup_lattice_layer[l2]->get_orbit_length(orb2);

					cnt = 0;

					for (g2 = 0; g2 < len2; g2++) {

						Subgroup2 = get_subgroup_by_orbit(l2, orb2, g2);

						if (Subgroup1->is_subgroup_of(Subgroup2)) {
							cnt++;
						}
					}
					i = first[l1] + orb1;
					j = first[l2] + orb2;
					incma[i * nb_orbits + j] = cnt;
				}
			}
		}
	}
	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"nb_orbits=" << nb_orbits << endl;
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"incma=" << endl;
		Int_matrix_print(incma, nb_orbits, nb_orbits);

		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"incma=" << endl;
		Int_matrix_print_nonzero_entries(incma, nb_orbits, nb_orbits);
	}


	FREE_int(first);
	FREE_int(length);

	if (f_v) {
		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup done" << endl;
	}
}


void subgroup_lattice::reduce_to_maximals(
		int *incma, int nb_groups,
		int *&maximals,
		int verbose_level)
// incma[nb_groups * nb_groups]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals" << endl;
		cout << "subgroup_lattice::reduce_to_maximals "
				"nb_groups=" << nb_groups << endl;
	}

	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals "
				"before make_partition_by_layers" << endl;
	}
	make_partition_by_layers(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals "
				"after make_partition_by_layers" << endl;
	}



	maximals = NEW_int(nb_groups * nb_groups);
	Int_vec_copy(incma, maximals, nb_groups * nb_groups);


	int l1, g1, l2, g2, l3, g3;
	int nb_groups1, nb_groups2, nb_groups3;
	int i, j, k;

	for (l1 = 0; l1 < nb_layers; l1++) {
		nb_groups1 = Subgroup_lattice_layer[l1]->nb_subgroups();
		for (g1 = 0; g1 < nb_groups1; g1++) {
			i = first[l1] + g1;
			for (l2 = l1 + 1; l2 < nb_layers; l2++) {
				nb_groups2 = Subgroup_lattice_layer[l2]->nb_subgroups();
				for (g2 = 0; g2 < nb_groups2; g2++) {
					j = first[l2] + g2;
					for (l3 = l2 + 1; l3 < nb_layers; l3++) {
						nb_groups3 = Subgroup_lattice_layer[l3]->nb_subgroups();
						for (g3 = 0; g3 < nb_groups3; g3++) {
							k = first[l3] + g3;
							if (incma[i * nb_groups + j] && incma[j * nb_groups + k] && incma[i * nb_groups + k]) {
								maximals[i * nb_groups + k] = 0;
							}
						}
					}
				}
			}
		}
	}
	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals "
				"maximals=" << endl;
		Int_matrix_print(maximals, nb_groups, nb_groups);

		cout << "subgroup_lattice::reduce_to_maximals "
				"maximals=" << endl;
		Int_matrix_print_nonzero_entries(maximals, nb_groups, nb_groups);
	}

	FREE_int(first);
	FREE_int(length);


	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals done" << endl;
	}
}

void subgroup_lattice::reduce_to_maximals_for_orbits(
		int *incma, int nb_orbits,
		int *&maximals,
		int verbose_level)
// incma[nb_orbits * nb_orbits]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits" << endl;
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits nb_orbits=" << nb_orbits << endl;
	}

	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits "
				"after make_partition_by_orbits" << endl;
	}



	maximals = NEW_int(nb_orbits * nb_orbits);
	Int_vec_copy(incma, maximals, nb_orbits * nb_orbits);


	int l1, orb1, l2, orb2, l3, orb3;
	int nb_orbits1, nb_orbits2, nb_orbits3;
	int i, j, k;

	for (l1 = 0; l1 < nb_layers; l1++) {

		nb_orbits1 = Subgroup_lattice_layer[l1]->nb_orbits();

		for (orb1 = 0; orb1 < nb_orbits1; orb1++) {

			i = first[l1] + orb1;

			for (l2 = l1 + 1; l2 < nb_layers; l2++) {

				nb_orbits2 = Subgroup_lattice_layer[l2]->nb_orbits();

				for (orb2 = 0; orb2 < nb_orbits2; orb2++) {

					j = first[l2] + orb2;


					for (l3 = l2 + 1; l3 < nb_layers; l3++) {

						nb_orbits3 = Subgroup_lattice_layer[l3]->nb_orbits();

						for (orb3 = 0; orb3 < nb_orbits3; orb3++) {

							k = first[l3] + orb3;

							if (incma[i * nb_orbits + j] && incma[j * nb_orbits + k] && incma[i * nb_orbits + k]) {
								if (f_v) {
									cout << "subgroup_lattice::reduce_to_maximals_for_orbits entry (i,k) = (" << i << ", " << k << ") made to zero because entries (i,j) = (" << i << ", " << j << ") and (j,k) = (" << j << ", " << k << ") both present" << endl;
								}
								maximals[i * nb_orbits + k] = 0;
							}
						}
					}
				}
			}
		}
	}
	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits "
				"maximals=" << endl;
		Int_matrix_print(maximals, nb_orbits, nb_orbits);

		cout << "subgroup_lattice::create_incidence_matrix_for_orbits_Asup "
				"maximals=" << endl;
		Int_matrix_print_nonzero_entries(maximals, nb_orbits, nb_orbits);
	}

	FREE_int(first);
	FREE_int(length);


	if (f_v) {
		cout << "subgroup_lattice::reduce_to_maximals_for_orbits done" << endl;
	}
}


void subgroup_lattice::find_overgroup_in_orbit(
		int layer1, int orb1, int group1,
		int layer2, int orb2, int &group2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::find_overgroup_in_orbit" << endl;
	}

	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::find_overgroup_in_orbit "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::find_overgroup_in_orbit "
				"after make_partition_by_orbits" << endl;
	}

	//nb_orbits = first[nb_parts];

	groups::subgroup *Subgroup1;
	groups::subgroup *Subgroup2;

	Subgroup1 = get_subgroup_by_orbit(layer1, orb1, group1);

	int orbit_len;

	orbit_len = Subgroup_lattice_layer[layer2]->get_orbit_length(orb2);

	for (group2 = 0; group2 < orbit_len; group2++) {

		Subgroup2 = get_subgroup_by_orbit(layer2, orb2, group2);

		if (Subgroup1->is_subgroup_of(Subgroup2)) {
			break;
		}
	}
	if (group2 == orbit_len) {
		cout << "subgroup_lattice::find_overgroup_in_orbit overgroup not found" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "subgroup_lattice::find_overgroup_in_orbit done" << endl;
	}
}



void subgroup_lattice::create_flag_transitive_geometry_with_partition(
		int P_layer, int P_orb_local,
		int Q_layer, int Q_orb_local,
		int R_layer, int R_orb_local, int R_group,
		int intersection_size,
		int *&intersection_matrix,
		int &nb_r, int &nb_c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition P_layer=" << P_layer << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition P_orb_local=" << P_orb_local << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition Q_layer=" << Q_layer << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition Q_orb_local=" << Q_orb_local << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition R_layer=" << R_layer << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition R_orb_local=" << R_orb_local << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition R_group=" << R_group << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition intersection_size=" << intersection_size << endl;
	}

	int G_layer;

	G_layer = nb_layers - 1;

	groups::subgroup *Subgroup_P;
	groups::subgroup *Subgroup_Q;
	groups::subgroup *Subgroup_R;
	groups::subgroup *Subgroup_G;


	Subgroup_Q = get_subgroup_by_orbit(
			Q_layer, Q_orb_local, 0 /* group_in_orbit_idx */);

	Subgroup_R = get_subgroup_by_orbit(
			R_layer, R_orb_local, R_group);

	Subgroup_G = get_subgroup_by_orbit(
			G_layer, 0, 0);



	int *cosets1; // right cosets for Q in R
	int nb_cosets1;

	int *cosets2; // right cosets for R in G
	int nb_cosets2;

	int *Q_image1;
	int *Q_image2;
	int *group;
	int *gens;
	int group_sz;

	cosets1 = NEW_int(group_order);
	cosets2 = NEW_int(group_order);
	Q_image1 = NEW_int(group_order);
	Q_image2 = NEW_int(group_order);
	group = NEW_int(group_order);
	gens = NEW_int(group_order);


	nb_gens = Subgroup_Q->nb_gens;
	Int_vec_copy(Subgroup_Q->gens, gens, Subgroup_Q->nb_gens);

	Sims->dimino_with_multiple_generators(
			Subgroup_Q->Elements,
			Subgroup_Q->group_order,
			gens, nb_gens,
		cosets1, nb_cosets1,
		Subgroup_R->gens /* new_gens*/, Subgroup_R->nb_gens /* nb_new_gens */,
		group, group_sz,
		0 /* verbose_level */);


	nb_gens = Subgroup_R->nb_gens;
	Int_vec_copy(Subgroup_R->gens, gens, Subgroup_R->nb_gens);

	Sims->dimino_with_multiple_generators(
			Subgroup_R->Elements,
			Subgroup_R->group_order,
			gens, nb_gens,
		cosets2, nb_cosets2,
		Subgroup_G->gens /* new_gens*/, Subgroup_G->nb_gens /* nb_new_gens */,
		group, group_sz,
		0 /* verbose_level */);

	if (f_v) {
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition nb_cosets1=" << nb_cosets1 << endl;
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition nb_cosets2=" << nb_cosets2 << endl;
	}

	int j1, j2;
	int *Elt1;
	int *Elt2;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);


	int P_orbit_len;
	nb_c = nb_cosets1 * nb_cosets2;

	P_orbit_len = Subgroup_lattice_layer[P_layer]->get_orbit_length(
			P_orb_local);

	nb_r = P_orbit_len;

	intersection_matrix = NEW_int(nb_r * nb_c);



	for (j2 = 0; j2 < nb_cosets2; j2++) {

		Sims->element_unrank_lint(cosets2[j2], Elt2);

		for (j1 = 0; j1 < nb_cosets1; j1++) {

			Sims->element_unrank_lint(cosets1[j1], Elt1);

			if (f_v) {
				cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
						"before Sims->conjugate_numerical_set (1)" << endl;
			}

			Sims->conjugate_numerical_set(
					Subgroup_Q->Elements, Subgroup_Q->group_order,
					Elt1, Q_image1,
					verbose_level);

			if (f_v) {
				cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
						"after Sims->conjugate_numerical_set (1)" << endl;
			}

			if (f_v) {
				cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
						"before Sims->conjugate_numerical_set (2)" << endl;
			}

			Sims->conjugate_numerical_set(
					Q_image1, Subgroup_Q->group_order,
					Elt2, Q_image2,
					verbose_level);

			if (f_v) {
				cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
						"after Sims->conjugate_numerical_set (2)" << endl;
			}

			int group_idx;
			int S_layer_idx, S_orb_idx, S_group_idx;

			for (group_idx = 0; group_idx < P_orbit_len; group_idx++) {

				Subgroup_P = get_subgroup_by_orbit(
						P_layer, P_orb_local, group_idx);


				groups::subgroup Subgroup2;
				groups::subgroup *S_Subgroup;

				Subgroup2.Elements = Q_image2;
				Subgroup2.group_order = Subgroup_Q->group_order;

				if (f_v) {
					cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
							"before intersect_subgroups" << endl;
				}
				intersect_subgroups(
						Subgroup_P,
						&Subgroup2,
						S_layer_idx, S_orb_idx, S_group_idx,
						verbose_level - 2);
				if (f_v) {
					cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition "
							"after intersect_subgroups" << endl;
				}

				Subgroup2.Elements = NULL;
				Subgroup2.group_order = 0;

				S_Subgroup = get_subgroup_by_orbit(
						S_layer_idx, S_orb_idx, S_group_idx);


				int entry;

				if (S_Subgroup->group_order == 4) {
					entry = 1;
				}
				else {
					entry = 0;
				}
				intersection_matrix[group_idx * nb_c + j2 * nb_cosets1 + j1] = entry;


			}
		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(cosets1);
	FREE_int(cosets2);
	FREE_int(Q_image1);
	FREE_int(Q_image2);
	FREE_int(group);
	FREE_int(gens);

	if (f_v) {
		cout << "subgroup_lattice::create_flag_transitive_geometry_with_partition done" << endl;
	}
}

void subgroup_lattice::create_coset_geometry(
		int P_orb_global, int P_group,
		int Q_orb_global, int Q_group,
		int intersection_size,
		int *&intersection_matrix,
		int &nb_r, int &nb_c,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry" << endl;
	}
	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry P_orb_global=" << P_orb_global << endl;
		cout << "subgroup_lattice::create_coset_geometry P_group=" << P_group << endl;
		cout << "subgroup_lattice::create_coset_geometry Q_orb_global=" << Q_orb_global << endl;
		cout << "subgroup_lattice::create_coset_geometry Q_group=" << Q_group << endl;
	}

	groups::subgroup *Subgroup_P;
	groups::subgroup *Subgroup_Q;

	int P_layer, P_orb_local;
	int Q_layer, Q_orb_local;

	orb_global_to_orb_local(
			P_orb_global, P_layer, P_orb_local,
				verbose_level - 2);

	orb_global_to_orb_local(
			Q_orb_global, Q_layer, Q_orb_local,
				verbose_level - 2);

	Subgroup_P = get_subgroup_by_orbit(
			P_layer, P_orb_local, P_group);

	Subgroup_Q = get_subgroup_by_orbit(
			Q_layer, Q_orb_local, Q_group);


	int *cosets1;
	int nb_cosets1;
	int *cosets2;
	int nb_cosets2;

	int nb_orbits;
	int G_orb_global;
	int G_group;

	nb_orbits = number_of_orbits_total();
	G_orb_global = nb_orbits - 1;
	G_group = 0;

	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry G_orb_global=" << G_orb_global << endl;
		cout << "subgroup_lattice::create_coset_geometry G_group=" << G_group << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry "
				"before right_transversal (1)" << endl;
	}
	right_transversal(
			P_orb_global, P_group,
			G_orb_global, G_group,
			cosets1, nb_cosets1,
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry "
				"after right_transversal (1)" << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry "
				"before right_transversal (2)" << endl;
	}
	right_transversal(
			Q_orb_global, Q_group,
			G_orb_global, G_group,
			cosets2, nb_cosets2,
			verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry "
				"after right_transversal (2)" << endl;
	}


	other::data_structures::sorting Sorting;
	int i, j;
	int *Elt1;
	int *Elt2;
	int *P_image;
	int *Q_image;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	P_image = NEW_int(group_order);
	Q_image = NEW_int(group_order);

	nb_r = nb_cosets1;
	nb_c = nb_cosets2;

	intersection_matrix = NEW_int(nb_r * nb_c);

	for (i = 0; i < nb_cosets1; i++) {

		Sims->element_unrank_lint(cosets1[i], Elt1);

		if (f_v) {
			cout << "subgroup_lattice::create_coset_geometry "
					"before Sims->right_translate_numerical_set (P_image)" << endl;
		}

		Sims->right_translate_numerical_set(
				Subgroup_P->Elements, Subgroup_P->group_order,
				Elt1, P_image,
				verbose_level - 2);

#if 0
		Sims->conjugate_numerical_set(
				Subgroup_P->Elements, Subgroup_P->group_order,
				Elt1, P_image,
				verbose_level);
#endif

		if (f_v) {
			cout << "subgroup_lattice::create_coset_geometry "
					"after Sims->right_translate_numerical_set (P_image)" << endl;
		}

		for (j = 0; j < nb_cosets2; j++) {

			Sims->element_unrank_lint(cosets2[j], Elt2);

			if (f_v) {
				cout << "subgroup_lattice::create_coset_geometry "
						"before Sims->right_translate_numerical_set (Q_image)" << endl;
			}

			Sims->right_translate_numerical_set(
					Subgroup_Q->Elements, Subgroup_Q->group_order,
					Elt2, Q_image,
					verbose_level - 2);
#if 0
			Sims->conjugate_numerical_set(
					Subgroup_Q->Elements, Subgroup_Q->group_order,
					Elt2, Q_image,
					verbose_level);
#endif
			if (f_v) {
				cout << "subgroup_lattice::create_coset_geometry "
						"after Sims->right_translate_numerical_set (Q_image)" << endl;
			}

			int *intersection;
			int intersection_sz;

			Sorting.int_vec_intersect(
					P_image, Subgroup_P->group_order,
					Q_image, Subgroup_Q->group_order,
					intersection, intersection_sz);

#if 0
			int entry;

			if (intersection_sz == intersection_size) {
				entry = 1;
			}
			else {
				entry = 0;
			}
#endif

			intersection_matrix[i * nb_c + j] = intersection_sz;

			FREE_int(intersection);

		}
	}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(P_image);
	FREE_int(Q_image);
	FREE_int(cosets1);
	FREE_int(cosets2);

	if (f_v) {
		cout << "subgroup_lattice::create_coset_geometry done" << endl;
	}

}



void subgroup_lattice::right_transversal(
		int P_orb_global, int P_group,
		int Q_orb_global, int Q_group,
		int *&cosets, int &nb_cosets,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::right_transversal" << endl;
	}

	groups::subgroup *Subgroup_P;
	groups::subgroup *Subgroup_Q;

	int P_layer, P_orb_local;
	int Q_layer, Q_orb_local;

	orb_global_to_orb_local(
			P_orb_global, P_layer, P_orb_local,
				verbose_level - 2);

	orb_global_to_orb_local(
			Q_orb_global, Q_layer, Q_orb_local,
				verbose_level - 2);

	Subgroup_P = get_subgroup_by_orbit(
			P_layer, P_orb_local, P_group);

	Subgroup_Q = get_subgroup_by_orbit(
			Q_layer, Q_orb_local, Q_group);



	int *group;
	int *gens;
	int group_sz;

	cosets = NEW_int(group_order);
	group = NEW_int(group_order);
	gens = NEW_int(group_order);


	nb_gens = Subgroup_P->nb_gens;
	Int_vec_copy(Subgroup_P->gens, gens, Subgroup_P->nb_gens);

	if (f_v) {
		cout << "subgroup_lattice::right_transversal "
				"before Sims->dimino_with_multiple_generators" << endl;
	}
	Sims->dimino_with_multiple_generators(
			Subgroup_P->Elements,
			Subgroup_P->group_order,
			gens, nb_gens,
		cosets, nb_cosets,
		Subgroup_Q->gens /* new_gens*/,
		Subgroup_Q->nb_gens /* nb_new_gens */,
		group, group_sz,
		0 /* verbose_level */);
	if (f_v) {
		cout << "subgroup_lattice::right_transversal "
				"after Sims->dimino_with_multiple_generators" << endl;
	}



	if (f_v) {
		cout << "subgroup_lattice::right_transversal nb_cosets=" << nb_cosets << endl;
	}

	FREE_int(group);
	FREE_int(gens);

	if (f_v) {
		cout << "subgroup_lattice::right_transversal done" << endl;
	}
}



void subgroup_lattice::two_step_transversal(
		int P_layer, int P_orb_local, int P_group,
		int Q_layer, int Q_orb_local, int Q_group,
		int R_layer, int R_orb_local, int R_group,
		int *&cosets1, int &nb_cosets1,
		int *&cosets2, int &nb_cosets2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal" << endl;
	}

	groups::subgroup *Subgroup_P;
	groups::subgroup *Subgroup_Q;
	groups::subgroup *Subgroup_R;


	Subgroup_P = get_subgroup_by_orbit(
			P_layer, P_orb_local, P_group);

	Subgroup_Q = get_subgroup_by_orbit(
			Q_layer, Q_orb_local, Q_group);

	Subgroup_R = get_subgroup_by_orbit(
			R_layer, R_orb_local, R_group);


	int *group;
	int *gens;
	int group_sz;

	cosets1 = NEW_int(group_order);
	cosets2 = NEW_int(group_order);
	group = NEW_int(group_order);
	gens = NEW_int(group_order);


	nb_gens = Subgroup_P->nb_gens;
	Int_vec_copy(Subgroup_P->gens, gens, Subgroup_P->nb_gens);

	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal "
				"before Sims->dimino_with_multiple_generators (1)" << endl;
	}
	Sims->dimino_with_multiple_generators(
			Subgroup_P->Elements,
			Subgroup_P->group_order,
			gens, nb_gens,
		cosets1, nb_cosets1,
		Subgroup_Q->gens /* new_gens*/,
		Subgroup_Q->nb_gens /* nb_new_gens */,
		group, group_sz,
		0 /* verbose_level */);
	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal "
				"after Sims->dimino_with_multiple_generators (1)" << endl;
	}


	nb_gens = Subgroup_Q->nb_gens;
	Int_vec_copy(Subgroup_Q->gens, gens, Subgroup_Q->nb_gens);

	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal "
				"before Sims->dimino_with_multiple_generators (2)" << endl;
	}
	Sims->dimino_with_multiple_generators(
			Subgroup_Q->Elements,
			Subgroup_Q->group_order,
			gens, nb_gens,
		cosets2, nb_cosets2,
		Subgroup_R->gens /* new_gens*/, Subgroup_R->nb_gens /* nb_new_gens */,
		group, group_sz,
		0 /* verbose_level */);
	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal "
				"after Sims->dimino_with_multiple_generators (2)" << endl;
	}

	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal nb_cosets1=" << nb_cosets1 << endl;
		cout << "subgroup_lattice::two_step_transversal nb_cosets2=" << nb_cosets2 << endl;
	}

	FREE_int(group);
	FREE_int(gens);

	if (f_v) {
		cout << "subgroup_lattice::two_step_transversal done" << endl;
	}
}

void subgroup_lattice::orb_global_to_orb_local(
		int orb_global, int &layer, int &orb_local,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::orb_global_to_orb_local" << endl;
	}
	int *first;
	int *length;
	int nb_parts;

	if (f_v) {
		cout << "subgroup_lattice::orb_global_to_orb_local "
				"before make_partition_by_orbits" << endl;
	}
	make_partition_by_orbits(
				first, length, nb_parts, verbose_level - 2);
	if (f_v) {
		cout << "subgroup_lattice::orb_global_to_orb_local "
				"after make_partition_by_orbits" << endl;
	}

	for (layer = 0; layer < nb_parts; layer++) {
		if (first[layer] <= orb_global && first[layer + 1] > orb_global) {
			break;
		}
	}

	if (layer == nb_parts) {
		cout << "subgroup_lattice::orb_global_to_orb_local not found" << endl;
		exit(1);
	}

	orb_local = orb_global - first[layer];



	FREE_int(first);
	FREE_int(length);

	if (f_v) {
		cout << "subgroup_lattice::orb_global_to_orb_local" << endl;
	}
}

void subgroup_lattice::intersection_orbit_orbit(
		int orb1, int orb2,
		int *&intersection_matrix,
		int &len1, int &len2,
		int verbose_level)
// intersection_matrix[len1 * len2]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::intersection_orbit_orbit" << endl;
	}

	int l1, o1;
	int l2, o2;

	orb_global_to_orb_local(orb1, l1, o1, verbose_level - 2);
	orb_global_to_orb_local(orb2, l2, o2, verbose_level - 2);

	len1 = Subgroup_lattice_layer[l1]->get_orbit_length(o1);
	len2 = Subgroup_lattice_layer[l2]->get_orbit_length(o2);

	if (f_v) {
		cout << "subgroup_lattice::intersection_orbit_orbit "
				"l1 = " << l1 << " o1 = " << o1 << " len1 = " << len1 << endl;
		cout << "subgroup_lattice::intersection_orbit_orbit "
				"l2 = " << l2 << " o2 = " << o2 << " len2 = " << len2 << endl;
	}

	intersection_matrix = NEW_int(len1 * len2);
	Int_vec_zero(intersection_matrix, len1 * len2);

	int g1, g2;
	groups::subgroup *Subgroup1;
	groups::subgroup *Subgroup2;
	groups::subgroup *Subgroup3;
	int layer_idx, orb_idx, group_idx;


	for (g1 = 0; g1 < len1; g1++) {

		Subgroup1 = get_subgroup_by_orbit(l1, o1, g1);

		for (g2 = 0; g2 < len2; g2++) {

			Subgroup2 = get_subgroup_by_orbit(l2, o2, g2);


			intersect_subgroups(
					Subgroup1,
					Subgroup2,
					layer_idx, orb_idx, group_idx,
					verbose_level - 2);

			Subgroup3 = get_subgroup_by_orbit(
					layer_idx, orb_idx, group_idx);

			intersection_matrix[g1 * len2 + g2] = Subgroup3->group_order;

		}

	}
	if (f_v) {
		cout << "subgroup_lattice::intersection_orbit_orbit done" << endl;
	}
}

void subgroup_lattice::intersect_subgroups(
		groups::subgroup *Subgroup1,
		groups::subgroup *Subgroup2,
		int &layer_idx, int &orb_idx, int &group_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "subgroup_lattice::intersect_subgroups" << endl;
	}

	other::data_structures::sorting Sorting;
	int *elements3;
	int group_order3;
	int f_found;
	int pos;
	uint32_t hash;

	Sorting.int_vec_intersect(
			Subgroup1->Elements, Subgroup1->group_order,
						Subgroup2->Elements, Subgroup2->group_order,
						elements3, group_order3);

	f_found = find_subgroup_direct(
			elements3, group_order3,
			layer_idx, pos, hash, verbose_level - 2);

	if (!f_found) {
		cout << "subgroup_lattice::intersect_subgroups did not find the intersection" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "subgroup_lattice::intersect_subgroups "
				"intersection is (layer,pos) = "
				"(" << layer_idx << "," << pos << ")" << endl;
	}

	Subgroup_lattice_layer[layer_idx]->group_global_to_orbit_and_group_local(
			pos /* group_idx_global */, orb_idx, group_idx,
			verbose_level - 2);

	if (f_vv) {
		cout << "subgroup_lattice::intersect_subgroups "
				"intersection is (layer,orb,grp) = "
				"(" << layer_idx << "," << orb_idx << "," << group_idx << ")" << endl;
	}

	FREE_int(elements3);

	if (f_v) {
		cout << "subgroup_lattice::intersect_subgroups done" << endl;
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

	algebra::number_theory::number_theory_domain NT;

	layer_idx = NT.nb_prime_factors_counting_multiplicities(order);

	f_new_group = Subgroup_lattice_layer[layer_idx]->add_subgroup(Subgroup, verbose_level - 2);

	if (f_v) {
		cout << "subgroup_lattice::add_subgroup done" << endl;
	}
	return f_new_group;
}

int subgroup_lattice::find_subgroup_direct(
		int *elements, int group_order,
		int &layer_idx, int &pos, uint32_t &hash, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup_direct" << endl;
	}

	int f_found;

	algebra::number_theory::number_theory_domain NT;

	layer_idx = NT.nb_prime_factors_counting_multiplicities(group_order);

	f_found = Subgroup_lattice_layer[layer_idx]->find_subgroup_direct(
			elements, group_order, pos, hash, verbose_level - 2);

	if (f_v) {
		cout << "subgroup_lattice_layer::find_subgroup_direct done" << endl;
	}
	return f_found;
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

		cout << i << " / " << Zuppos.size()
				<< " : " << Zuppos[i] << " : " << ord << " : ";
		Int_vec_print(cout, data, A->make_element_size);
		cout << endl;
	}

	FREE_int(Elt1);
	FREE_int(data);

	if (f_v) {
		cout << "subgroup_lattice::print_zuppos done" << endl;
	}
}

void subgroup_lattice::identify_subgroup(
		groups::strong_generators *Strong_gens,
		int &go, int &layer_idx, int &orb_idx, int &group_idx,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup" << endl;
	}

	long int *rank_vector;
	int *rank_vector_int;
	int len;

	Strong_gens->compute_rank_vector(
			rank_vector, len, Sims,
			verbose_level);

	rank_vector_int = NEW_int(len);
	Lint_vec_copy_to_int(rank_vector, rank_vector_int, len);

	go = Strong_gens->group_order_as_lint();

	algebra::number_theory::number_theory_domain NT;


	layer_idx = NT.nb_prime_factors_counting_multiplicities(go);


	int *subgroup;
	int subgroup_sz = 1;

	int *gens;
	int nb_gens = 0;

	int *cosets;
	int nb_cosets;

	int *group;
	int group_sz;


	subgroup = NEW_int(Sims->group_order_lint());
	gens = NEW_int(Sims->group_order_lint());
	cosets = NEW_int(Sims->group_order_lint());
	group = NEW_int(Sims->group_order_lint());

	subgroup[0] = 0;

	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup "
				"before Sims->dimino_with_multiple_generators" << endl;
	}
	Sims->dimino_with_multiple_generators(
		subgroup, subgroup_sz, gens, nb_gens,
		cosets, nb_cosets,
		rank_vector_int /* new_gens */, len /* nb_new_gens */,
		group, group_sz,
		verbose_level);
	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup "
				"after Sims->dimino_with_multiple_generators" << endl;
	}

	if (group_sz != go) {
		cout << "subgroup_lattice::identify_subgroup "
				"group_sz != go" << endl;
		exit(1);
	}

	groups::subgroup *Subgroup;

	Subgroup = NEW_OBJECT(groups::subgroup);

	Subgroup->init(
			this,
			group, group_sz, gens, nb_gens,
			verbose_level - 2);


	int f_found;
	int pos;
	uint32_t hash;

	f_found = Subgroup_lattice_layer[layer_idx]->find_subgroup(
			Subgroup,
			pos, hash, verbose_level - 2);

	if (!f_found) {
		cout << "subgroup_lattice::identify_subgroup "
				"did not find subgroup" << endl;
		exit(1);

	}
	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup "
				"found subgroup at layer " << layer_idx << " in position " << pos << endl;
	}



	Subgroup_lattice_layer[layer_idx]->group_global_to_orbit_and_group_local(
			pos /* group_idx_global */, orb_idx, group_idx,
			verbose_level - 2);

	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup "
				"found subgroup in layer " << layer_idx
				<< " in orbit " << orb_idx << " at position " << group_idx << endl;
	}

	FREE_OBJECT(Subgroup);
	FREE_int(subgroup);
	FREE_int(gens);
	FREE_int(cosets);
	FREE_int(group);


	FREE_lint(rank_vector);
	FREE_int(rank_vector_int);

	if (f_v) {
		cout << "subgroup_lattice::identify_subgroup done" << endl;
	}
}

void subgroup_lattice::do_export_csv(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "subgroup_lattice::do_export_csv" << endl;
	}

	std::string *Table;
	int nb_rows;
	int nb_cols;
	int layer_idx;

	nb_cols = 6;

	nb_rows = 0;
	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {
		nb_rows += Subgroup_lattice_layer[layer_idx]->nb_orbits();
	}

	Table = new std::string [nb_rows * nb_cols];

	int cur_row;
	int i;

	cur_row = 0;

	for (layer_idx = 0; layer_idx < nb_layers; layer_idx++) {

		std::string *Table1;
		int nb_rows1, nb_cols1;

		Subgroup_lattice_layer[layer_idx]->do_export_to_string(
				Table1, nb_rows1, nb_cols1,
				verbose_level);

		if (nb_cols1 != 4) {
			cout << "subgroup_lattice::do_export_csv nb_cols1 != 4" << endl;
			exit(1);
		}

		for (i = 0; i < nb_rows1; i++, cur_row++) {
			Table[cur_row * nb_cols + 0] = std::to_string(cur_row);
			Table[cur_row * nb_cols + 1] = std::to_string(layer_idx);
			//Table[cur_row * nb_cols + 2] = std::to_string(cur_row);
			Table[cur_row * nb_cols + 2] = Table1[i * nb_cols1 + 0];
			Table[cur_row * nb_cols + 3] = Table1[i * nb_cols1 + 1];
			Table[cur_row * nb_cols + 4] = Table1[i * nb_cols1 + 2];
			Table[cur_row * nb_cols + 5] = Table1[i * nb_cols1 + 3];
		}

		delete [] Table1;


	}

	std::string fname;

	fname = label_txt + "_classes.csv";

	std::string headings;

	headings = "orbit,layer,idxlocal,go,length,groupidx";

	other::orbiter_kernel_system::file_io Fio;

	Fio.Csv_file_support->write_table_of_strings(
			fname,
			nb_rows, nb_cols, Table,
			headings,
			verbose_level);

	if (f_v) {
		cout << "subgroup_lattice::do_export_csv Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}


	if (f_v) {
		cout << "subgroup_lattice::do_export_csv done" << endl;
	}
}


}}}



