/*
 * grid.cpp
 *
 *  Created on: Aug 18, 2021
 *      Author: betten
 */




#include <stdlib.h>
#include "geo.h"

using namespace std;


grid::grid()
{
	f_points = FALSE;
	m = 0;
	n = 0;
	G_max = 0;
	//int first[MAX_GRID + 1];
	//int len[MAX_GRID];
	//int type_idx[MAX_GRID];
	//int grid_entry[MAX_GRID];
	//int type[MAX_GRID][MAX_GRID];

}

grid::~grid()
{

}

void grid::print()
{
	int i, j, f, i1;

	if (f_points)
		cout << "the points:" << endl;
	else
		cout << "the blocks:" << endl;
	cout << "grid:" << endl;
	cout << "m=" << m << endl;
	cout << "n=" << n << endl;
	cout << "G_max=" << G_max << endl;
	for (i = 0; i < G_max; i++) {
		cout << "at " << first[i] << ", " << len[i] << " x (";
		f = first[i];
		i1 = type_idx[f];
		for (j = 0; j < n; j++) {
			cout << type[i1][j];
			if (j < n - 1)
				cout << ", ";
			}
		cout << ")" << endl;
		}
	cout << "total:" << first[G_max] << endl;
}

void grid::init_derived_i_first(grid *G_old, int derive_at_i)
/* Erstes Element des grid-entry i
 * von G_old auszeichnen.
 * G->G_max wird zu G_old->G_max + 1.
 * Setzt auch type_idx[] und type[][]. */
{
	int i, i0, j, k, first, len, old_type_idx, ti;

	f_points = G_old->f_points;
	m = G_old->m;
	n = G_old->n;
	for (i = 0, i0 = 0; i < G_old->G_max;
		i++, i0++) {
		first = G_old->first[i];
		len = G_old->len[i];
		old_type_idx = G_old->type_idx[first];
		ti = first;
		for (k = 0; k < G_old->n; k++) {
			type[ti][k] =
			G_old->type[old_type_idx][k];
		}
		if (i != derive_at_i) {
			grid::first[i0] = first;
			grid::len[i0] = len;
		}
		else {
			grid::first[i0] = first;
			grid::len[i0] = 1;
			type_idx[first] = first;
			grid_entry[first] = i0;
			len--;
			if (len > 0) {
				i0++;
				first++;
				grid::first[i0] = first;
				grid::len[i0] = len;
			}
		}
		for (j = 0; j < len; j++) {
			type_idx[first + j] = ti;
			grid_entry[first + j] = i0;
		}
	}
	grid::first[i0] = G_old->first[i];
	G_max = i0;
}

void grid::init_derived_ij_first(grid *G_old, int I, int J)
/* Erste Elemente der grid-entrys
 * I und J von G_old auszeichnen.
 * G->G_max wird zu G_old->G_max + 1 (bzw. + 2).
 * Setzt auch type_idx[] und type[][]. */
{
	int i, i0, j, k, first, len;
	int old_type_idx, ti;

	f_points = G_old->f_points;
	m = G_old->m;
	n = G_old->n;
	i0 = 0; /* aktueller Index nach G->first[] etc. */
	first = 0;
	for (i = 0; i < G_old->G_max; i++) {
		if (first != G_old->first[i]) {
			cout << "grid_init_derived_ij_first first != G_old->first[i]" << endl;
			exit(1);
		}
		/* first = G_old->first[i]; */
		len = G_old->len[i];
		old_type_idx = G_old->type_idx[first];

		if (i == I && i == J) {
			if (len < 2) {
				cout << "grid_init_derived_ij_first i == I && i == J && len < 2" << endl;
				exit(1);
			}
			/* Ein Paar desselben
			 * Bereichs auszeichnen:
			 * gemeinsamer Zweierbereich,
			 * NICHT einzeln !
			 * Grund dafuer:
			 * Bei einzelner Unterteilung
			 * waeren die Punkte (Bloecke)
			 * unterschiedlich behandelt,
			 * je nachdem wer zuerst liegt.
			 * Dies muss vermieden werden,
			 * da die TDO Invariante
			 * fuer jede Ausgangslage der Inzidenz
			 * gleich sein muss. */
			grid::first[i0] = first;
			grid::len[i0] = 2;
			ti = first;
			for (k = 0; k < G_old->n; k++) {
				type[ti][k] =
				G_old->type[old_type_idx][k];
			}
			type_idx[first + 0] = ti;
			type_idx[first + 1] = ti;
			grid_entry[first + 0] = i0;
			grid_entry[first + 1] = i0;
			first += 2;
			len -= 2;
			i0++;
		}
		else if (i == I || i == J) {
			grid::first[i0] = first;
			grid::len[i0] = 1;
			ti = first;
			for (k = 0; k < G_old->n; k++) {
				type[ti][k] =
				G_old->type[old_type_idx][k];
			}
			type_idx[first + 0] = ti;
			grid_entry[first + 0] = i0;
			i0++;
			first++;
			len--;
		}

		/* Eventuellen Rest verarbeiten
		 * bzw. i disjunkt von I, J: */
		if (len) {
			grid::first[i0] = first;
			grid::len[i0] = len;
			ti = first;
			for (k = 0; k < G_old->n; k++) {
				type[ti][k] =
				G_old->type[old_type_idx][k];
			}
			for (j = 0; j < len; j++) {
				type_idx[first + j] = ti;
				grid_entry[first + j] = i0;
			}
			i0++;
			first += len;
		}
	}
	grid::first[i0] = G_old->first[i];
	G_max = i0;
}

void grid::copy_frame_to(grid *G_to)
{
	int i, j, ti;

	G_to->f_points = f_points;
	G_to->m = m;
	G_to->n = n;
	for (i = 0; i < G_max; i++) {
		G_to->first[i] = first[i];
		G_to->len[i] = len[i];
		ti = type_idx[first[i]];
		for (j = 0; j < G_to->n; j++) {
			G_to->type[ti][j] = type[ti][j];
		}
	}
	G_to->first[i] = first[i];
	G_to->G_max = i;
	for (i = 0; i < G_to->m; i++) {
		G_to->type_idx[i] = type_idx[i];
		G_to->grid_entry[i] = grid_entry[i];
	}
}

int grid::insert_idx(int f, int l, int radix, int search_this, int *idx)
{
	int i, st1, cur, cur1, res;
	int f_found;

	st1 = type_idx[search_this];
	f_found = FALSE;
	for (i = 0; i < l; i++) {
		cur = f + i;
		cur1 = type_idx[cur];
		res = type[cur1][radix] - type[st1][radix];
		if (res == 0) {
			f_found = TRUE;
		}
		if (res < 0) {
			*idx = cur;
			return f_found;
			}
		}
	*idx = f + l;
	return f_found;
}

