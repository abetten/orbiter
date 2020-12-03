/*
 * orbiter_symbol_table.cpp
 *
 *  Created on: Dec 2, 2020
 *      Author: betten
 */




#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {




orbiter_symbol_table::orbiter_symbol_table()
{
	// Table;
}

orbiter_symbol_table::~orbiter_symbol_table()
{

}

int orbiter_symbol_table::find_symbol(std::string &str)
{
	int i;

	for (i = 0; i < Table.size(); i++) {
		if (stringcmp(str, Table[i].label.c_str()) == 0) {
			return i;
		}
	}
	return -1;
}

void orbiter_symbol_table::add_symbol_table_entry(std::string &str,
		orbiter_symbol_table_entry *Symb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx;

	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry" << endl;
	}
	idx = find_symbol(str);

	if (idx >= 0) {
		cout << "orbiter_symbol_table::add_symbol_table_entry Overriding symbol " << idx << endl;
		Symb[idx].freeself();
		Symb[idx] = *Symb;
	}
	else {
		Table.push_back(*Symb);
		Symb->freeself();
	}
	if (f_v) {
		cout << "orbiter_symbol_table::add_symbol_table_entry done" << endl;
	}
}

void orbiter_symbol_table::print_symbol_table()
{
	int i;

	if (Table.size()) {
		for (i = 0; i < Table.size(); i++) {
			cout << i << " : " << Table[i].label << " : ";
			Table[i].print();
			cout << endl;
		}
	}
	else {
		cout << "orbiter_symbol_table::print_symbol_table symbol table is empty" << endl;
	}
}

void *orbiter_symbol_table::get_object(int idx)
{
	if (idx >= Table.size()) {
		cout << "orbiter_symbol_table::get_object out of bounds" << endl;
		exit(1);
	}
	return Table[idx].ptr;
}



}}

