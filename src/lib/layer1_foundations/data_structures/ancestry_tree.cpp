/*
 * ancestry_tree.cpp
 *
 *  Created on: Feb 1, 2024
 *      Author: betten
 */







#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {




ancestry_tree::ancestry_tree()
{
	//std::string fname_gedcom;
	//std::string fname_base;

	//std::vector<std::vector<std::string> > Data;
	//std::vector<std::vector<int> > Indi;
	//std::vector<std::vector<int> > Fam;

	nb_indi = 0;
	nb_fam = 0;
	Family = NULL;
	Individual = NULL;

}

ancestry_tree::~ancestry_tree()
{

}

void ancestry_tree::read_gedcom(
		std::string &fname_gedcom, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_tree::read_gedcom" << endl;
	}
	orbiter_kernel_system::file_io Fio;

	ancestry_tree::fname_gedcom = fname_gedcom;

	Fio.read_gedcom_file(
			fname_gedcom,
			Data,
			verbose_level);

	int l, i;

	l = Data.size();
	if (f_v) {
		cout << "ancestry_tree::read_gedcom Read file " << fname_gedcom << " with " << l << " lines" << endl;
	}
#if 0
	for (i = 0; i < l; i++) {
		cout << Data[i][0] << " : " << Data[i][1] << " : " << Data[i][2] << endl;
	}
#endif

	data_structures::string_tools ST;
	//string fname_base;
	string fname_out;

	fname_base = fname_gedcom;

	ST.chop_off_extension(fname_base);

	fname_out = fname_base + ".csv";

	Fio.Csv_file_support->write_gedcom_file_as_csv(
			fname_out,
			Data,
		verbose_level);

	int len;

	len = Data.size();

	// scan INDI
	for (i = 0; i < len; i++) {
		if (ST.stringcmp(Data[i][0], "0") == 0 && ST.stringcmp(Data[i][2], "INDI") == 0) {
			int start = i;
			//cout << "INDI start in line " << i << endl;
			for (i++; i < len; i++) {
				if (ST.stringcmp(Data[i][0], "0") == 0) {
					int length = i - start;
					//cout << "INDI end in line " << i << endl;
					vector<int> entry;

					entry.push_back(start);
					entry.push_back(length);
					Indi.push_back(entry);
					i--;
					break;
				}
			}
		}
	}

	// scan FAM
	for (i = 0; i < len; i++) {
		if (ST.stringcmp(Data[i][0], "0") == 0 && ST.stringcmp(Data[i][2], "FAM") == 0) {
			int start = i;
			for (i++; i < len; i++) {
				if (ST.stringcmp(Data[i][0], "0") == 0) {
					int length = i - start;
					vector<int> entry;

					entry.push_back(start);
					entry.push_back(length);
					Fam.push_back(entry);
					i--;
					break;
				}
			}
		}
	}


	nb_indi = Indi.size();
	nb_fam = Fam.size();

	if (f_v) {
		cout << "# of INDI = " << nb_indi << endl;
		cout << "# of FAM = " << nb_fam << endl;
	}


	Family = (data_structures::ancestry_family **) NEW_pvoid(nb_fam);
	Individual = (data_structures::ancestry_indi **) NEW_pvoid(nb_indi);

	int start, length;

	if (f_v) {
		cout << "setting up Individual[]" << endl;
	}
	for (i = 0; i < nb_indi; i++) {
		start = Indi[i][0];
		length = Indi[i][1];
		Individual[i] = NEW_OBJECT(data_structures::ancestry_indi);
		Individual[i]->init(this, i, start, length, Data, 0/*verbose_level*/);

	}
	if (f_v) {
		cout << "setting up Individual[] done" << endl;
	}


	if (f_v) {
		for (i = 0; i < nb_indi; i++) {
			cout << i
					<< " : " << Individual[i]->id
					<< " : " << Individual[i]->sex
					<< " : " << Individual[i]->name
					<< " : " << Individual[i]->famc
					<< " : " << Individual[i]->fams
					<< " : " << Individual[i]->birth_date
					<< " : " << Individual[i]->death_date
					<< endl;
		}
	}

	string fname_indi;

	fname_indi = fname_base + "_indi.csv";

	Fio.Csv_file_support->write_ancestry_indi(
			fname_indi,
			Data,
			nb_indi,
			Individual,
		verbose_level);

	if (f_v) {
		cout << "setting up Family[]" << endl;
	}
	for (i = 0; i < nb_fam; i++) {
		start = Fam[i][0];
		length = Fam[i][1];
		Family[i] = NEW_OBJECT(data_structures::ancestry_family);
		Family[i]->init(this, i, start, length, Data, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "setting up Family[] done" << endl;
	}



	if (f_v) {
		cout << "ancestry_tree::read_gedcom "
				"before get_connnections" << endl;
	}
	get_connnections(verbose_level);
	if (f_v) {
		cout << "ancestry_tree::read_gedcom "
				"after get_connnections" << endl;
	}

	graph_theory::layered_graph *L;

	if (f_v) {
		cout << "ancestry_tree::read_gedcom "
				"before create_poset" << endl;
	}
	create_poset(L, verbose_level);
	if (f_v) {
		cout << "ancestry_tree::read_gedcom "
				"after create_poset" << endl;
	}


	string fname_family;

	fname_indi = fname_base + "_family.csv";

	Fio.Csv_file_support->write_ancestry_family(
			fname_indi,
			Data,
			nb_indi,
			nb_fam,
			Individual,
			Family,
		verbose_level);


}

void ancestry_tree::create_poset(
		graph_theory::layered_graph *&L, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_tree::create_poset" << endl;
	}

	int *topo_rank;
	int rk_max;

	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"before topo_sort" << endl;
	}
	topo_sort(topo_rank, rk_max, verbose_level);
	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"after topo_sort" << endl;
	}

	int *topo_rank_copy;
	int *perm;
	int *perm_inv;

	{
		int i;


		topo_rank_copy = NEW_int(nb_fam);
		for (i = 0; i < nb_fam; i++) {
			topo_rank_copy[i] = topo_rank[i];
		}
		perm = NEW_int(nb_fam);
		for (i = 0; i < nb_fam; i++) {
			perm[i] = i;
		}
		perm_inv = NEW_int(nb_fam);
		for (i = 0; i < nb_fam; i++) {
			perm_inv[i] = i;
		}
	}

	data_structures::sorting Sorting;

	Sorting.int_vec_sorting_permutation(
			topo_rank_copy, nb_fam,
		perm, perm_inv, true /*f_increasingly*/);


	if (f_v) {
		int a, i;

		cout << "families ordered by topo rank:" << endl;
		for (i = 0; i < nb_fam; i++) {
			a = perm_inv[i];
			cout << i << " : " << topo_rank[a] << " : " << a << endl;
		}
	}


	L = NEW_OBJECT(graph_theory::layered_graph);

	int nb_layers;
	int *Nb_nodes_layer;
	int *First_node_of_layer;
	int r;

	nb_layers = rk_max + 1;


	Nb_nodes_layer = NEW_int(nb_layers);
	Int_vec_zero(Nb_nodes_layer, nb_layers);

	{
		int i, a;
		for (i = 0; i < nb_fam; i++) {
			a = perm_inv[i];
			r = topo_rank[a];
			Nb_nodes_layer[r]++;
		}
	}

	First_node_of_layer = NEW_int(nb_layers);
	{
		int i, r_prev, a;
		First_node_of_layer[0] = 0;
		r_prev = 0;
		for (i = 0; i < nb_fam; i++) {
			a = perm_inv[i];
			r = topo_rank[a];
			if (r > r_prev) {
				First_node_of_layer[r] = i;
				r_prev = r;
			}
		}
	}

	if (f_v) {
		cout << "r : First_node_of_layer[r]" << endl;
		for (r = 0; r < nb_layers; r++) {
			cout << r << " : " << First_node_of_layer[r] << endl;
		}
	}

	std::string fname_topo;


	fname_topo = fname_base + "_poset";

	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"before L->init" << endl;
	}

	L->init(
			nb_layers, Nb_nodes_layer,
			fname_topo, verbose_level);
	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"after L->init" << endl;
	}

	{
		int f, a, r, l, n;

		for (f = 0; f < nb_fam; f++) {
			a = perm_inv[f];
			r = topo_rank[a];
			l = r;
			n = f - First_node_of_layer[r];

			string text;

			text = Family[a]->get_initials(verbose_level);

			if (f_v) {
				cout << "ancestry_tree::create_poset "
						"before L->add_text text=" << text << endl;
			}
			L->add_text(
					l, n,
					text,
					verbose_level);
			if (f_v) {
				cout << "ancestry_tree::create_poset "
						"before L->add_text" << endl;
			}
		}
	}


	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"after L->init" << endl;
	}

	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"computing connections" << endl;
	}
	int f1, f2;
	int l1, n1, l2, n2, a;

	for (f1 = 0; f1 < nb_fam; f1++) {

		a = perm_inv[f1];
		r = topo_rank[a];

		l1 = r;
		n1 = f1 - First_node_of_layer[r];

		int i, j, b, r2;

		for (i = 0; i < Family[a]->child_index.size(); i++) {
			for (j = 0; j < Family[a]->child_family_index[i].size(); j++) {
				b = Family[a]->child_family_index[i][j];
				f2 = perm[b];
				r2 = topo_rank[b];


				l2 = r2;
				n2 = f2 - First_node_of_layer[r2];

				L->add_edge(
						l1, n1, l2, n2,
						0 /*verbose_level*/);

			}
		}




	}
	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"computing connections done" << endl;
	}

	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"before L->place" << endl;
	}
	L->place(verbose_level);
	if (f_v) {
		cout << "ancestry_tree::create_poset "
				"after L->place" << endl;
	}

	std::string fname_poset;


	fname_poset = fname_topo + ".layered_graph";

	L->write_file(
			fname_poset, verbose_level);

	if (f_v) {
		cout << "ancestry_tree::create_poset done" << endl;
	}
}


void ancestry_tree::topo_sort(
		int *&topo_rank, int &rk_max, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_tree::topo_sort" << endl;
	}

	int i;

	if (f_v) {
		cout << "ancestry_tree::topo_sort before topo_sort_prepare" << endl;
	}
	topo_sort_prepare(verbose_level);
	if (f_v) {
		cout << "ancestry_tree::topo_sort after topo_sort_prepare" << endl;
	}


	if (f_v) {
		int j;

		for (i = 0; i < nb_fam; i++) {
			cout << "family " << i << " : " << endl;
			for (j = 0; j < Family[i]->topo_downlink.size(); j++) {
				cout << Family[i]->topo_downlink[j] << " ";
			}
			cout << endl;
		}

	}


	if (f_v) {
		cout << "ancestry_tree::topo_sort after topo_sort_prepare" << endl;
	}

	int *degree;
	int *Q;
	int Q_len;
	int a, b, rk, i0, i1, u, j, h, parent_family_idx;

	degree = NEW_int(nb_fam);
	topo_rank = NEW_int(nb_fam);
	Q = NEW_int(nb_fam);

	for (i = 0; i < nb_fam; i++) {
		degree[i] = Family[i]->topo_downlink.size();
	}
	Int_vec_zero(topo_rank, nb_fam);
	for (i = 0; i < nb_fam; i++) {
		Q[i] = i;
	}
	Q_len = nb_fam;


	rk = 0;
	i0 = 0;
	while (i0 < nb_fam) {

		if (f_v) {
			cout << "ancestry_tree::topo_sort rk = " << rk << " i0 = " << i0 << endl;
		}
		if (f_v) {
			cout << "i : Q[i] : degree[Q[i]]" << endl;
			for (i = i0; i < Q_len; i++) {
				cout << i << " : " << Q[i] << " : " << degree[Q[i]] << endl;
			}
		}

		if (rk > nb_fam) {
			cout << "rk > nb_fam" << endl;
			exit(1);
		}

		i1 = i0;

		// get the nodes with degree = 0:
		// Sort them up front in Q[];

		for (i = i0; i < Q_len; i++) {
			a = Q[i];
			if (degree[a] == 0) {
				if (i1 != i) {
					b = Q[i1];
					Q[i1] = Q[i];
					Q[i] = b;
				}
				i1++;
				topo_rank[a] = rk;
				if (f_v) {
					cout << "ancestry_tree::topo_sort topo_rank[" << a << "] = " << rk << endl;
				}
			}
		}

		if (f_v) {
			cout << "ancestry_tree::topo_sort rk = " << rk << " i0 = " << i0 << " i1 = " << i1 << endl;
		}

		// remove the degrees of parent nodes:

		if (f_v) {
			cout << "ancestry_tree::topo_sort reduce degrees of parent nodes" << endl;
		}
		for (i = i0; i < i1; i++) {

			a = Q[i];
			if (f_v) {
				cout << "ancestry_tree::topo_sort i = " << i << " Q[i] = " << a << endl;
			}

			for (u = 0; u < 2; u++) {
				if (u == 0) {
					parent_family_idx = Family[a]->husband_family_index;
				}
				else {
					parent_family_idx = Family[a]->wife_family_index;
				}
				if (parent_family_idx == -1) {
					continue;
				}
				if (f_v) {
					cout << "ancestry_tree::topo_sort i = " << i << " Q[i] = " << a << " parent_family_idx=" << parent_family_idx << endl;
				}
				for (j = 0; j < Family[parent_family_idx]->topo_downlink.size(); j++) {
					if (Family[parent_family_idx]->topo_downlink[j] == a) {
						break;
					}
				} // j
				if (j == Family[parent_family_idx]->topo_downlink.size()) {
					cout << "did not find downlink, parent_family_idx = " << parent_family_idx << " a=" << a << endl;
					exit(1);
				}
				else {

					// remove downlink and reduce the degree:

					for (h = j + 1; h < Family[parent_family_idx]->topo_downlink.size(); h++) {
						Family[parent_family_idx]->topo_downlink[h - 1] = Family[parent_family_idx]->topo_downlink[h];
					}
					Family[parent_family_idx]->topo_downlink.pop_back();
					degree[parent_family_idx]--;
					if (degree[parent_family_idx] < 0) {
						cout << "degree[parent_family_idx] < 0" << endl;
						exit(1);
					}
					if (f_v) {
						cout << "ancestry_tree::topo_sort degree[" << parent_family_idx << "] = " << degree[parent_family_idx] << endl;
					}
				}
			} // u
		} // i

		if (i0 == i1) {
			for (i = i1; i < nb_fam; i++) {
				a = Q[i];
				topo_rank[a] = rk;
			}
			break;
		}
		i0 = i1;
		rk++;

	}

	rk_max = rk;


	if (f_v) {
		cout << "ancestry_tree::topo_sort finished" << endl;
		cout << "ancestry_tree::topo_sort rk_max = " << rk_max << endl;
		cout << "i : topo_rank[i]" << endl;
		for (i = 0; i < nb_fam; i++) {
			cout << i << " : " << topo_rank[i] << endl;
		}
		cout << "ancestry_tree::topo_sort finished" << endl;
		cout << "ancestry_tree::topo_sort rk_max = " << rk_max << endl;
		//Int_vec_print(cout, topo_rank, nb_fam);
		//cout << endl;
	}


	// adjust:

	topo_sort_prepare(verbose_level);

	for (i = 0; i < nb_fam; i++) {
		degree[i] = Family[i]->topo_downlink.size();
	}

	int r;

	for (i = 0; i < nb_fam; i++) {
		if (degree[i] == 0) {
			r = Family[i]->topo_rank_of_parents(
					topo_rank, 0 /* verbose_level*/);
			if (r > 0) {
				topo_rank[i] = r - 1;
			}
		}
	}


	FREE_int(degree);
	FREE_int(Q);

	if (f_v) {
		cout << "ancestry_tree::topo_sort done" << endl;
	}
}


void ancestry_tree::topo_sort_prepare(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_tree::topo_sort_prepare" << endl;
	}

	int i;

	for (i = 0; i < nb_fam; i++) {
		Family[i]->topo_sort_prepare(0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "ancestry_tree::topo_sort_prepare done" << endl;
	}
}

void ancestry_tree::get_connnections(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_tree::get_connnections" << endl;
	}

	int i;

	for (i = 0; i < nb_fam; i++) {
		Family[i]->get_connnections(0 /*verbose_level*/);
	}

	if (f_v) {
		cout << "ancestry_tree::get_connnections done" << endl;
	}
}

int ancestry_tree::find_individual(
		std::string &id, int verbose_level)
{
	int f_v = false;//(verbose_level >= 1);
	data_structures::string_tools ST;
	int i;

	if (f_v) {
		cout << "ancestry_tree::find_individual id=" << id << endl;
	}
	for (i = 0; i < nb_indi; i++) {
		if (ST.compare_string_string(Individual[i]->id, id) == 0) {
			if (f_v) {
				cout << "ancestry_tree::find_individual "
						"found id=" << id << " at i=" << i << endl;
			}
			return i;
		}
	}
	return -1;
}

int ancestry_tree::find_in_family_as_child(
		int indi_idx)
{
	data_structures::string_tools ST;
	int i, j;

	if (indi_idx == -1) {
		return -1;
	}

	for (i = 0; i < nb_fam; i++) {
		for (j = 0; j < Family[i]->child_index.size(); j++) {
			if (Family[i]->child_index[j] == indi_idx) {
				return i;
			}
		}
	}
	return -1;
}

std::vector<int> ancestry_tree::find_in_family_as_parent(
		int indi_idx)
{
	data_structures::string_tools ST;
	int i;
	std::vector<int> parent;

	if (indi_idx == -1) {
		return parent;
	}

	for (i = 0; i < nb_fam; i++) {
		if (Family[i]->husband_index == indi_idx) {
			parent.push_back(i);
		}
		if (Family[i]->wife_index == indi_idx) {
			parent.push_back(i);
		}
	}
	return parent;
}



void ancestry_tree::register_individual(
		int individual_index, int family_idx,
		int verbose_level)
{
	Individual[individual_index]->family_index.push_back(family_idx);
}



}}}



