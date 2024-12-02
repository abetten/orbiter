/*
 * ancestry_indi.cpp
 *
 *  Created on: Jan 29, 2024
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {




ancestry_indi::ancestry_indi()
{
	Record_birth();
	Tree = NULL;
	idx = -1;

	start = 0;
	length = 0;
	//std::string id;
	//std::string name;
	//std::string given_name;
	//std::string sur_name;
	//std::string sex;
	//std::string famc;
	//std::string fams;
	//std::string birth_date;
	//std::string death_date;

	//std::vector<int> family_index;

}

ancestry_indi::~ancestry_indi()
{
	Record_death();

}

void ancestry_indi::init(
		ancestry_tree *Tree,
		int idx,
		int start, int length,
		std::vector<std::vector<std::string> > &Data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_indi::init" << endl;
	}
	data_structures::string_tools ST;

	ancestry_indi::Tree = Tree;
	ancestry_indi::idx = idx;
	ancestry_indi::start = start;
	ancestry_indi::length = length;
	id = Data[start][1];

	int i;

	// get name:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "NAME") == 0) {
			name = Data[start + i][2];
			break;
		}
	}

	// get given name:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "2") == 0 && ST.stringcmp(Data[start + i][1], "GIVN") == 0) {
			given_name = Data[start + i][2];
			break;
		}
	}

	// get sur name:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "2") == 0 && ST.stringcmp(Data[start + i][1], "SURN") == 0) {
			sur_name = Data[start + i][2];
			break;
		}
	}

	// get sex:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "SEX") == 0) {
			sex = Data[start + i][2];
			break;
		}
	}

	// get famc:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "FAMC") == 0) {
			famc = Data[start + i][2];
			break;
		}
	}

	// get fams:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "FAMS") == 0) {
			fams = Data[start + i][2];
			break;
		}
	}

	// get birth_date:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "BIRT") == 0 && ST.stringcmp(Data[start + i + 1][1], "DATE") == 0) {
			birth_date = Data[start + i + 1][2];
			break;
		}
	}

	// get death_date:
	for (i = 0; i < length; i++) {
		if (ST.stringcmp(Data[start + i][0], "1") == 0 && ST.stringcmp(Data[start + i][1], "DEAT") == 0 && ST.stringcmp(Data[start + i + 1][1], "DATE") == 0) {
			death_date = Data[start + i + 1][2];
			break;
		}
	}



	if (f_v) {
		cout << "ancestry_indi::init done" << endl;
	}
}

std::string ancestry_indi::initials(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "ancestry_indi::initials" << endl;
	}
	string s;

	if (given_name.length()) {
		s += given_name.substr(0, 1);
	}

	if (sur_name.length()) {
		s += sur_name.substr(0, 1);
	}
	return s;
}



}}}}



