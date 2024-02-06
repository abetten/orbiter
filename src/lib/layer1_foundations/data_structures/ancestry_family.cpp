/*
 * ancestry_family.cpp
 *
 *  Created on: Jan 29, 2024
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace data_structures {




ancestry_family::ancestry_family()
{
	Tree = NULL;
	idx = -1;

	start = 0;
	length = 0;
	// id

	//std::string husband;
	husband_index = -1;
	husband_family_index = -1;

	//std::string wife;
	wife_index = -1;
	wife_family_index = -1;

	//std::vector<int> child;
	//std::vector<int> child_index;
	//std::vector<int> child_family_index;
	//std::vector<int> topo_downlink;

}

ancestry_family::~ancestry_family()
{

}

void ancestry_family::init(
		ancestry_tree *Tree,
		int idx,
		int start, int length,
		std::vector<std::vector<std::string> > &Data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_family::init idx = " << idx << endl;
	}
	data_structures::string_tools ST;

	ancestry_family::Tree = Tree;
	ancestry_family::idx = idx;
	ancestry_family::start = start;
	ancestry_family::length = length;
	id = Data[start][1];


	int i;

	// get husband:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "HUSB") == 0) {
			husband = Data[start + i][2];
			husband_index = Tree->find_individual(husband, verbose_level - 1);
			if (husband_index != -1) {
				Tree->register_individual(husband_index, idx, verbose_level);
			}
			else {
				if (f_v) {
					cout << "ancestry_family::init idx = " << idx << " husband_index = " << husband_index << endl;
				}
			}
			break;
		}
	}

	// get wife:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "WIFE") == 0) {
			wife = Data[start + i][2];
			wife_index = Tree->find_individual(wife, verbose_level - 1);
			if (wife_index != -1) {
				Tree->register_individual(wife_index, idx, verbose_level);
			}
			break;
		}
	}

	// get children:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "CHIL") == 0) {
			string s;
			int child_idx;

			s = Data[start + i][2];
			child.push_back(s);
			child_idx = Tree->find_individual(s, verbose_level - 1);
			child_index.push_back(child_idx);
			if (child_idx != -1) {
				Tree->register_individual(child_idx, idx, verbose_level);
			}
		}
	}



	if (f_v) {
		cout << "ancestry_family::init done" << endl;
	}
}

void ancestry_family::get_connnections(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_family::get_connnections" << endl;
	}
	int i, index;

	if (f_v) {
		cout << "ancestry_family::get_connnections family idx = " << idx << " number of children = " << child_index.size() << endl;
	}
	husband_family_index = Tree->find_in_family_as_child(husband_index);

	if (f_v) {
		cout << "ancestry_family::get_connnections husband_family_index = " << husband_family_index << endl;
	}

	wife_family_index = Tree->find_in_family_as_child(wife_index);

	if (f_v) {
		cout << "ancestry_family::get_connnections wife_family_index = " << wife_family_index << endl;
	}

	for (i = 0; i < child_index.size(); i++) {
		vector<int> parenting;
		parenting = Tree->find_in_family_as_parent(child_index[i]);
		child_family_index.push_back(parenting);
		if (f_v) {
			cout << "ancestry_family::get_connnections idx = " << idx << " child_family_index = " << index << endl;
		}
	}
}

std::string ancestry_family::get_initials(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_family::get_initials" << endl;
	}

	string s;

	if (f_v) {
		cout << "ancestry_family::get_initials husband_index = " << husband_index << endl;
	}
	if (husband_index >= 0) {
		s += Tree->Individual[husband_index]->initials(verbose_level);
	}
	s += ":" + std::to_string(idx);
	if (f_v) {
		cout << "ancestry_family::get_initials s = " << s << endl;
	}
	return s;
}

void ancestry_family::topo_sort_prepare(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_family::topo_sort_prepare" << endl;
	}
	int i, j, a;

	for (i = 0; i < child_index.size(); i++) {
		if (child_family_index[i].size()) {
			for (j = 0; j < child_family_index[i].size(); j++) {
				a = child_family_index[i][j];
				topo_downlink.push_back(a);
			}
		}
	}
	if (f_v) {
		cout << "ancestry_family::topo_sort_prepare topo_downlink.size() = " << topo_downlink.size() << endl;
	}
	if (f_v) {
		cout << "ancestry_family::topo_sort_prepare done" << endl;
	}
}

int ancestry_family::topo_rank_of_parents(
		int *topo_rank, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_family::topo_rank_of_parents" << endl;
	}
	int r;

	r = INT_MAX;
	if (husband_family_index >= 0) {
		r = MINIMUM(r, topo_rank[husband_family_index]);
	}
	if (wife_family_index >= 0) {
		r = MINIMUM(r, topo_rank[wife_family_index]);
	}

	if (r == INT_MAX) {
		r = 0;
	}

	if (f_v) {
		cout << "ancestry_family::topo_rank_of_parents done" << endl;
	}
	return r;
}

}}}


