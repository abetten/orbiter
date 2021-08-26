/*
 * iso_grid.cpp
 *
 *  Created on: Aug 16, 2021
 *      Author: betten
 */

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {


iso_grid::iso_grid()
{
	m = 0;
	n = 0;


	//cperm q;
	//cperm qv; /* q^-1 */


	//int type[MAX_VB][MAX_TYPE];
	G_max = 0;
	//int first[MAX_GRID];
	//int len[MAX_GRID];
	//int type_idx[MAX_GRID];
	//int grid_entry[MAX_GRID];

}

iso_grid::~iso_grid()
{

}

void iso_grid::print()
{
	int i, j, f, i1;

	for (i = 0; i < G_max; i++) {
		cout << "at " << first[i] << " " << len[i] << " x (";
		f = first[i];
		i1 = type_idx[f];
		for (j = 0; j < n; j++) {
			cout << type[i1][j];
			if (j < n) {
				cout << " ";
			}
		}
		cout << ")" << endl;
	}
	cout << "total: " << first[G_max] << endl;
}



}}


