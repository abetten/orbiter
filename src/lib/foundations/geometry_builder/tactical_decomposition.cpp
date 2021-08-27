/*
 * tactical_decomposition.cpp
 *
 *  Created on: Aug 15, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {

tactical_decomposition::tactical_decomposition()
{
	Encoding = NULL;

	f_TDO_multiple = FALSE;
	f_TDO_d_multiple = FALSE;
	//cperm p;
		/* row permutation;
		 * degree tdo->v */
	//cperm q;
		/* column permutation;
		 * degree tdo->inc->B */
	//cperm pv; /* p^-1 */
	//cperm qv; /* q^-1 */
		/* given theX, applying p to the rows,
		 * q to the columns, You get the
		 * matrix of the TDO */
	G_last = NULL;
	G_current = NULL;
	G_next = NULL;
	tdos = NULL;
	tdos2 = NULL;

}

tactical_decomposition::~tactical_decomposition()
{
	if (G_last) {
		FREE_OBJECT(G_last);
	}
	if (G_current) {
		FREE_OBJECT(G_current);
	}
	if (G_next) {
		FREE_OBJECT(G_next);
	}
	if (tdos) {
		FREE_OBJECT(tdos);
	}
	if (tdos2) {
		FREE_OBJECT(tdos2);
	}
}

void tactical_decomposition::init(inc_encoding *Encoding,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::init" << endl;
	}

	tactical_decomposition::Encoding = Encoding;

	f_TDO_multiple = FALSE;
	f_TDO_d_multiple = FALSE;


	if (f_v) {
		cout << "tactical_decomposition::init before allocating G_last" << endl;
	}
	G_last = NEW_OBJECT(grid);
	if (f_v) {
		cout << "tactical_decomposition::init before allocating G_current" << endl;
	}
	G_current = NEW_OBJECT(grid);
	if (f_v) {
		cout << "tactical_decomposition::init before allocating G_next" << endl;
	}
	G_next = NEW_OBJECT(grid);

	if (f_v) {
		cout << "tactical_decomposition::init before cp_int" << endl;
	}

	p.init_and_identity(Encoding->v);
	pv.init_and_identity(Encoding->v);
	q.init_and_identity(Encoding->b);
	qv.init_and_identity(Encoding->b);


	if (f_v) {
		cout << "p=";
		p.print();
		cout << endl;
		cout << "pv=";
		pv.print();
		cout << endl;

		cout << "q=";
		q.print();
		cout << endl;
		cout << "qv=";
		qv.print();
		cout << endl;
	}

	if (f_v) {
		cout << "tactical_decomposition::init done" << endl;
	}
}


void tactical_decomposition::tdo_calc(inc_encoding *Encoding, incidence *inc, int v,
	int f_second_tactical_decomposition, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::tdo_calc" << endl;
	}

	if (f_v) {
		cout << "tactical_decomposition::tdo_calc before init" << endl;
	}
	init(Encoding, verbose_level);

	if (f_v) {
		cout << "tactical_decomposition::tdo_calc after init" << endl;
	}


	if (f_v) {
		cout << "tactical_decomposition::tdo_calc before init_partition" << endl;
	}
	init_partition(G_current, G_last, inc, v, verbose_level);
	if (f_v) {
		cout << "tactical_decomposition::tdo_calc after init_partition" << endl;
		cout << "G_current=" << endl;
		G_current->print();
		cout << "G_last=" << endl;
		G_last->print();
	}

	if (f_v) {
		cout << "tactical_decomposition::tdo_calc before calc2" << endl;
	}
	calc2(v, verbose_level);
	if (f_v) {
		cout << "tactical_decomposition::tdo_calc after calc2" << endl;
		cout << "G_current=" << endl;
		G_current->print();
		cout << "G_next=" << endl;
		G_next->print();
	}


	if (f_v) {
		cout << "tactical_decomposition::tdo_calc before tdos_int" << endl;
	}
	tdos = get_tdos(G_current, G_next, FALSE, verbose_level);
	if (f_v) {
		cout << "tactical_decomposition::tdo_calc after tdos_int" << endl;
		//tdos->print();
	}

	if (f_second_tactical_decomposition) {
		if (f_v) {
			cout << "tdo_calc before second_order_tdo" << endl;
		}
		second_order_tdo(v, verbose_level);
		if (f_v) {
			cout << "tdo_calc after second_order_tdo" << endl;
		}
		FREE_OBJECT(tdos);
		tdos = tdos2;
		tdos2 = NULL;
	}

}

void tactical_decomposition::make_point_and_block_partition(grid *Gpoints, grid *Gblocks)
{
	int i, j, first, last_p1;

	/* init Gpoints (only up to tdo->v): */
	Gpoints->f_points = TRUE;
	Gpoints->m = Encoding->v;
	Gpoints->n = 1 /* tdo->inc->nb_i_vbar */;
	Gpoints->G_max = 0;
	Gpoints->first[0] = 0;
	first = 0 /* tdo->inc->i_hbar[I] */;
	last_p1 = Encoding->v;
	/* first[0] ist bereits gesetzt */
	Gpoints->len[0] = last_p1 - first;
	Gpoints->first[0 + 1] = last_p1;
	for (j = 0; j < Gpoints->n; j++) {
		Gpoints->type[first][j] = 0;
	}
	for (i = first; i < last_p1; i++) {
		Gpoints->type_idx[i] = first;
		Gpoints->grid_entry[i] = 0;
	}
	Gpoints->G_max++;

	/* init Gblocks: */
	Gblocks->f_points = FALSE;
	Gblocks->m = Encoding->b /* tdo->inc->B */;
	Gblocks->n = Gpoints->G_max;
		/* tdo->inc->nb_i_hbar */
	Gblocks->G_max = 0;
	Gblocks->first[0] = 0;
	first = 0 /* tdo->inc->i_vbar[I] */;
	last_p1 = Encoding->b;
	Gblocks->len[0] = last_p1 - first;
	Gblocks->first[0 + 1] = last_p1;
	for (j = 0; j < Gblocks->n; j++) {
		Gblocks->type[first][j] = 0;
	}
	for (i = first; i < last_p1; i++) {
		Gblocks->type_idx[i] = first;
		Gblocks->grid_entry[i] = 0;
	}
	Gblocks->G_max++;
}

void tactical_decomposition::init_partition(grid *Gpoints, grid *Gblocks,
	incidence *inc, int v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::init_partition" << endl;
	}
	int i, j, I, first, last_p1, f_break;

	// init Gpoints (only up to tdo->v):
	Gpoints->f_points = TRUE;
	Gpoints->m = v;
	Gpoints->n = inc->nb_i_vbar;
	Gpoints->G_max = 0;
	Gpoints->first[0] = 0;
	f_break = FALSE;
	for (I = 0; I < inc->nb_i_hbar; I++) {
		first = inc->i_hbar[I];
		if (I < inc->nb_i_hbar - 1) {
			last_p1 = inc->i_hbar[I + 1];
		}
		else {
			last_p1 = inc->Encoding->v;
		}
		if (first >= v) {
			break;
		}
		if (last_p1 > v) {
			f_break = TRUE;
			last_p1 = v;
		}
		// first[I] ist bereits gesetzt
		Gpoints->len[I] = last_p1 - first;
		Gpoints->first[I + 1] = last_p1;
		for (j = 0; j < Gpoints->n; j++) {
			Gpoints->type[first][j] = 0;
		}
		for (i = first; i < last_p1; i++) {
			Gpoints->type_idx[i] = first;
			Gpoints->grid_entry[i] = I;
		}
		Gpoints->G_max++;
		if (f_break) {
			break;
		}
	}

	// init Gblocks:
	Gblocks->f_points = FALSE;
	Gblocks->m = inc->Encoding->b;
	Gblocks->n = Gpoints->G_max;
	Gblocks->G_max = 0;
	Gblocks->first[0] = 0;
	for (I = 0; I < inc->nb_i_vbar; I++) {
		first = inc->i_vbar[I];
		if (I < inc->nb_i_vbar - 1) {
			last_p1 = inc->i_vbar[I + 1];
		}
		else {
			last_p1 = inc->Encoding->b;
		}
		// first[I] ist bereits gesetzt
		Gblocks->len[I] = last_p1 - first;
		Gblocks->first[I + 1] = last_p1;
		for (j = 0; j < Gblocks->n; j++) {
			Gblocks->type[first][j] = 0;
		}
		for (i = first; i < last_p1; i++) {
			Gblocks->type_idx[i] = first;
			Gblocks->grid_entry[i] = I;
		}
		Gblocks->G_max++;
	}
	if (f_v) {
		cout << "tactical_decomposition::init_partition done" << endl;
	}
}


void tactical_decomposition::print()
{
	cout << "tactical_decomposition:" << endl;
	tdos->print();
	printf("%d / %d\n", f_TDO_multiple, f_TDO_d_multiple);
	/* printf("tactical_decomposition mehrfach: %d / "
	"tactical_decomposition_d mehrfach: %d\n",
		tdo->f_tactical_decomposition_multiple,
		tdo->f_tactical_decomposition_d_multiple); */
}

void tactical_decomposition::radix_sort(grid *G, int radix, int first, int last)
/* radix sort of
 * G->type[first...last][radix] */
/* Es wird nicht die type Tabelle sortiert,
 * alle Permutationen werden in
 * tdo->p, pv, q, qv notiert.
 * Ausserdem wird type_idx mitvertauscht. */
{
	int f_found, idx, k, l, t;
	int first0, first1, res, k1;
	cperm *perm, *perm_inv;

	if (first == last || radix == G->n) {
		/* Bereich first .. last
		 * als neuen grid entry
		 * nach G eintragen.
		 * grid_entry[first..last] setzen. */
		k = G->G_max;
		if (G->first[k] != first) {
			cout << "radix_sort G->first[k] != first" << endl;
			exit(1);
		}
		for (l = first; l <= last; l++) {
			G->grid_entry[l] = k;
		}
		G->len[k] = last - first + 1;
		G->first[k + 1] = last + 1;
		G->G_max++;
		/* printf("radix_sort()|new entry: "
		"first = %d len = %d : ",
			(int)first, (int)G->len[k]);
		i1 = G->type_idx[first];
		for (j = 0; j < G->n; j++) {
			printf("%d ", (int)G->type[i1][j]);
			}
		printf("\n"); */
		return;
	}
	for (k = first; k <= last; k++) {
		f_found = G->insert_idx(first, k - first, radix, k, &idx);
		if (idx != k) {
			/* s = (idx idx+1 ... k)
			 * auf den aktuellen Stand
			 * der Matrix anwenden.
			 * if (G->f_points)
			 *   p := p * s, pv := s^-1 * pv
			 * else
			 *   q := q * s, qv := s^-1 * qv */
			if (G->f_points) {
				perm = &p;
				perm_inv = &pv;
			}
			else {
				perm = &q;
				perm_inv = &qv;
			}
			perm->mult_apply_forwc_r(idx, k - idx + 1);
			perm_inv->mult_apply_backwc_l(idx, k - idx + 1);
			t = G->type_idx[k];
			for (l = k; l > idx; l--) {
				G->type_idx[l] = G->type_idx[l - 1];
			}
			G->type_idx[idx] = t;
			/* grid_entry ist noch nicht gesetzt,
			 * braucht nicht mitgeschoben zu werden. */
		}
	}
	first0 = first;
	first1 = G->type_idx[first0];
	for (k = first + 1; k <= last; k++) {
		k1 = G->type_idx[k];
		res = G->type[k1][radix] - G->type[first1][radix];
		if (res > 0) {
			cout << "radix_sort not descending" << endl;
			exit(1);
		}
		if (res < 0) {
			radix_sort(G, radix + 1, first0, k - 1);
			first0 = k;
			first1 = G->type_idx[first0];
		}
		if (k == last) {
			radix_sort(G, radix + 1, first0, k);
		}
	}
}

void tactical_decomposition::refine_types(grid *Gm1, grid *G1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::refine_types" << endl;
	}
	int old_k, old_first, old_len;

	for (old_k = 0; old_k < Gm1->G_max; old_k++) {
		old_first = Gm1->first[old_k];
		old_len = Gm1->len[old_k];
		/* verfeinere jeden Bereich
		 * des vorletzten grids einzeln: */
		radix_sort(G1, 0, old_first, old_first + old_len - 1);
	} /* next old_k */
	if (f_v) {
		cout << "tactical_decomposition::refine_types done" << endl;
	}
}

void tactical_decomposition::recollect_types(int v, grid *G0, grid *G1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::recollect_types" << endl;
		cout << "tactical_decomposition::recollect_types "
				"G1->f_points=" << G1->f_points << " G1->m=" << G1->m << " G1->n=" << G1->n << endl;
		//cout << "tactical_decomposition::recollect_types dim_n=" << dim_n << endl;
	}

	if (f_v) {
		cout << "p=";
		p.print();
		cout << endl;

		cout << "q=";
		q.print();
		cout << endl;
	}

	int i, j, x, i1, x1, ge;

	for (i = 0; i < G1->m; i++) {
		for (j = 0; j < G1->n; j++) {
			G1->type[i][j] = 0;
		}
	}
	for (i = 0; i < v; i++) {
		if (f_v) {
			cout << "tactical_decomposition::recollect_types i=" << i << endl;
		}
		for (j = 0; j < Encoding->R[i]; j++) {
			x = Encoding->theX[i * Encoding->dim_n + j];
			i1 = p.data[i];
			x1 = q.data[x];
			if (f_v) {
				cout << "tactical_decomposition::recollect_types i=" << i << " j=" << j << " x=" << x << " i1=" << i1 << " x1=" << x1 << endl;
			}
			/* Kreuz (i, x) liegt momentan in (i1, x1)
			 * in der Inzidenzmatrix. */
			if (!G1->f_points) {
				ge = G0->grid_entry[i1];
				G1->type[x1][ge]++;
			}
			else {
				ge = G0->grid_entry[x1];
				G1->type[i1][ge]++;
			}
			if (f_v) {
				cout << "tactical_decomposition::recollect_types ge=" << ge << endl;
			}
		}
	}
	if (f_v) {
		cout << "tactical_decomposition::recollect_types done" << endl;
	}
}

void tactical_decomposition::collect_types(int v, grid *G0, grid *G1, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::collect_types" << endl;
		cout << "tactical_decomposition::collect_types G1->m = " << G1->m << endl;
	}
	int i, j;

	recollect_types(v, G0, G1, verbose_level);

	for (i = 0; i < G1->m; i++) {
		G1->type_idx[i] = i;
	}
	G1->G_max = 0;
	G1->first[0] = 0;

	if (f_v) {
		cout << "tactical_decomposition::collect_types" << endl;
		for (i = 0; i < G1->m; i++) {
			cout << "( ";
			for (j = 0; j < G1->n; j++) {
				cout << G1->type[i][j];
				if (j < G1->n - 1) {
					cout << ", ";
				}
			}
			cout << ")" << endl;
		}
	}

	if (f_v) {
		cout << "tactical_decomposition::collect_types done" << endl;
	}
}

void tactical_decomposition::next(int v, int verbose_level)
/* berechnet aus tdo->G_last
 * und tdo->G tdo->G_next.
 * Die G_last Einteilung wird
 * dabei weiter verfeinert. */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::next" << endl;
	}
	grid *G0 = G_current;
	grid *G1 = G_next;

	G1->f_points = !G0->f_points;
	if (G1->f_points) {
		G1->m = v;
	}
	else {
		G1->m = Encoding->b;
	}
	G1->n = G0->G_max;

	collect_types(v, G_current, G_next, verbose_level);

	refine_types(G_last, G_next, verbose_level);

	if (f_v) {
		cout << "tactical_decomposition::next after refine_types, tdo has size " << G_next->G_max << " x " << G_last->G_max << endl;
	}


	if (f_v) {
		cout << "tactical_decomposition::next done" << endl;
	}
}

void tactical_decomposition::calc2(int v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::calc2" << endl;
	}
	grid *G;
	int steps = 0;

	/* printf("G_last:\n");
	print_grid(tdo->G_last);
	printf("G:\n");
	print_grid(tdo->G); */
	while (TRUE) {
		if (f_v) {
			cout << "tactical_decomposition::calc2 before next" << endl;
		}
		next(v, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::calc2 after next" << endl;
		}
		steps++;
		if (f_v) {
			cout << "tactical_decomposition::calc2 steps=" << steps << endl;
			cout << "G_next=" << endl;
			G_next->print();
			cout << "G_next has m=" << G_next->m << endl;
			cout << "G_current has m=" << G_current->m << endl;
			cout << "G_last has m=" << G_last->m << endl;
		}
		/* print_grid(G_next); */
		if (G_next->G_max == G_last->G_max && steps >= 2) {
			break; /* this is a tactical_decomposition */
		}
		G = G_last;
		G_last = G_current;
		G_current = G_next;
		G_next = G;
	}
	if (f_v) {
		cout << "tactical_decomposition::calc2 steps=" << steps << endl;
		cout << "G_next has m=" << G_next->m << endl;
		cout << "G_current has m=" << G_current->m << endl;
		cout << "G_last has m=" << G_last->m << endl;
	}
	if (f_v) {
		cout << "tactical_decomposition::calc2 done" << endl;
	}
}


tdo_scheme *tactical_decomposition::get_tdos(grid *G0, grid *G1, int f_derived, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::get_tdos" << endl;
	}
	tdo_scheme *tdos;
	grid *Gpoints, *Gblocks;
	int i, j;
	int first, i1;

	tdos = NEW_OBJECT(tdo_scheme);

	if (G0->f_points) {
		Gpoints = G0;
		Gblocks = G1;
	}
	else {
		Gpoints = G1;
		Gblocks = G0;
	}

	if (f_v) {
		cout << "Gpoints->G_max=" << Gpoints->G_max << endl;
		cout << "Gblocks->G_max=" << Gblocks->G_max << endl;
		cout << "tactical_decomposition::get_tdos before tdos->allocate" << endl;
	}
	tdos->allocate(Gpoints->G_max, Gblocks->G_max);
	if (f_v) {
		cout << "tactical_decomposition::get_tdos after tdos->allocate" << endl;
	}


	for (i = 0; i < tdos->m; i++) {
		first = Gpoints->first[i];
		i1 = Gpoints->type_idx[first];
		for (j = 0; j < tdos->n; j++) {
			tdos->aij(i, j) = Gpoints->type[i1][j];
		}
	}
	for (i = 0; i < tdos->m; i++) {
		tdos->Vi(i) = Gpoints->len[i];
	}
	for (j = 0; j < tdos->n; j++) {
		tdos->Bj(j) = Gblocks->len[j];
	}
	return tdos;
}

void tactical_decomposition::dd_work(int v, int f_points,
	short *&dd, int &N, short *&dd_mult, int verbose_level)
/* Allociert dd auf einen N Vector von SHORTs.
 * dd enthaelt fuer alle Paare
 * (in der aktuellen Inzidenzlage)
 * den Index des Ableitungs TDOs.
 * Allociert dd_mult auf einen
 * Multiplizitaetenvektor;
 * dd_mult[0] = Anzahl der auftretenden
 *   Ableitungs TDOs.
 * dd_mult ist dann ein Vector
 * der Laenge (dd_mult[0] + 1) sizeof(SHORT).
 * setzt N := n * (n - 1) >> 1,
 * wobei n = tdo->v if f_points,
 * = tdo->inc->B sonst. */
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::dd_work" << endl;
	}

	tdo_scheme *tdos;
	grid *G_l = NULL;
	grid *G_c = NULL;
	grid *G_n = NULL;
	grid *G0, *G1;
	cperm p_, *P = &p_;
	cperm pv_, *Pv = &pv_;
	cperm q_, *Q = &q_;
	cperm qv_, *Qv = &qv_;

	cperm *perm, *perm_inv;
	cperm *tdo_perm, *tdo_perm_inv;
	int I, I_len, i, i0;
	int i1, J, J_len, j, j0, j1;
	int k;
	short *dd1 = NULL;
	short *dd_mult1 = NULL;


	dd = NULL;
	dd_mult = NULL;

	p.init_and_identity(Encoding->v);
	pv.init_and_identity(Encoding->v);
	q.init_and_identity(Encoding->b);
	qv.init_and_identity(Encoding->b);

	// save p, pv, q, qv:
	p.move_to(P);
	pv.move_to(Pv);
	q.move_to(Q);
	qv.move_to(Qv);

	G_l = G_last;
	G_c = G_current;
	G_n = G_next;
	G_last = NEW_OBJECT(grid);
	G_current = NEW_OBJECT(grid);
	G_next = NEW_OBJECT(grid);

	if (G_c->f_points == f_points) {
		G0 = G_n;
		G1 = G_c;
	}
	else {
		G0 = G_c;
		G1 = G_n;
	}
	/* G1 ist die TDO - Einteilung,
	 * G0 die Gegenrichtungs
	 * TDO - Einteilung. */

	if (f_points) {
		perm = P;
		perm_inv = Pv;
		tdo_perm = &p;
		tdo_perm_inv = &pv;
	}
	else {
		perm = Q;
		perm_inv = Qv;
		tdo_perm = &q;
		tdo_perm_inv = &qv;
	}

	int n;

	if (f_points)
		n = Encoding->v;
	else
		n = Encoding->b;
	N = (n * (n - 1)) >> 1;


	tdo_gradient *tdog;
	tdog = NEW_OBJECT(tdo_gradient);
	tdog->allocate(N);


	/* ueber alle Paare (i1, j1)
	 * mit i1 < j1 d.h. geeignete
	 * (I, i), (J, j) Kombinationen: */
	for (I = 0; I < G1->G_max; I++) {
		i0 = G1->first[I];
		I_len = G1->len[I];
		for (i = 0; i < I_len; i++) {
			i1 = i0 + i;
			for (J = I; J < G1->G_max; J++) {
				j0 = G1->first[J];
				J_len = G1->len[J];
				if (I == J) {
					j = i + 1;
				}
				else {
					j = 0;
				}
				for (; j < J_len; j++) {
					j1 = j0 + j;
					G0->copy_frame_to(G_last);
					/* Gegenrichtung aus G0
					 * nach tdo->G_last uebertragen. */

					G_current->init_derived_ij_first(G1 /* G_old */, I, J);
					if (I == J) {
						if (i != 0) {
							tdo_perm->mult_apply_tau_r(i0, i1);
							tdo_perm_inv->mult_apply_tau_l(i0, i1);
							/* dies hat j1 nicht
							 * bewegt, da i1 < j1 */
						}
						if (j != 1) {
							tdo_perm->mult_apply_tau_r(j0 + 1, j1);
							tdo_perm_inv->mult_apply_tau_l(j0 + 1, j1);
						}
					}
					else {
						if (i != 0) {
							tdo_perm->mult_apply_tau_r(i0, i1);
							tdo_perm_inv->mult_apply_tau_l(i0, i1);
						}
						if (j != 0) {
							tdo_perm->mult_apply_tau_r(j0, j1);
							tdo_perm_inv->mult_apply_tau_l(j0, j1);
						}
					}

					if (f_v) {
						cout << "tactical_decomposition::dd_work before calc2" << endl;
					}
					calc2(v, verbose_level);
					if (f_v) {
						cout << "tactical_decomposition::dd_work after calc2" << endl;
					}

					if (f_v) {
						cout << "tactical_decomposition::dd_work before get_tdos" << endl;
					}
					tdos = get_tdos(G_current, G_next, TRUE, verbose_level);
					if (f_v) {
						cout << "tactical_decomposition::dd_work after get_tdos" << endl;
					}

					k = ij2k(i1, j1, n);

					tdog->add_tdos(tdos, k, verbose_level);

					// restore p, pv, q, qv:
					P->move_to(&p);
					Pv->move_to(&pv);
					Q->move_to(&q);
					Qv->move_to(&qv);

				} // next j
			} // next J
		} // next i
	} // next I

	dd1 = new short [N];

	dd_mult1 = new short [tdog->nb_tdos + 1];

	for (i = 0; i < tdog->N; i++) {
		dd1[i] = (short) tdog->type[i];
	}
	dd_mult1[0] = tdog->nb_tdos;
	for (i = 0; i < tdog->nb_tdos; i++) {
		dd_mult1[1 + i] = (short) tdog->mult[i];
	}

	dd = dd1;
	dd_mult = dd_mult1;

	FREE_OBJECT(tdog);

	if (G_last) {
		FREE_OBJECT(G_last);
	}
	if (G_current) {
		FREE_OBJECT(G_current);
	}
	if (G_n) {
		FREE_OBJECT(G_n);
	}

	G_last = G_l;
	G_current = G_c;
	G_next = G_n;

	if (f_v) {
		cout << "tactical_decomposition::dd_work done" << endl;
	}
}

void tactical_decomposition::tdo_dd(int v, int f_points, int f_blocks,
	short *&ddp, int &Np, short *&ddp_mult,
	short *&ddb, int &Nb, short *&ddb_mult, int verbose_level)
{
	if (f_points) {
		dd_work(
				v,
			TRUE /* f_points */,
			ddp, Np, ddp_mult, verbose_level);
		}
	if (f_blocks) {
		dd_work(
				v,
			FALSE /* f_points */,
			ddb, Nb, ddb_mult, verbose_level);
		}
}


void tactical_decomposition::refine(int v,
		grid *G, grid *G_next,
		int f_points, geo_frame *frame,
		cperm *P, cperm *Pv, cperm *Q, cperm *Qv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::refine f_points=" << f_points << endl;
	}

	grid *G0, *G1;
	cperm *perm, *perm_inv;
	cperm *tdo_perm, *tdo_perm_inv;
	int first, len, first1, len1;
	int i, j, j1, k, l, t;

	if (G->f_points == f_points) {
		G0 = G_next;
		G1 = G;
	}
	else {
		G0 = G;
		G1 = G_next;
	}
	/* G1 ist die unverfeinerte
	 * TDO - Einteilung,
	 * G0 die Gegenrichtungs
	 * TDO - Einteilung (auch die alte). */
	if (f_points) {
		perm = P;
		perm_inv = Pv;
		tdo_perm = &p;
		tdo_perm_inv = &pv;
	}
	else {
		perm = Q;
		perm_inv = Qv;
		tdo_perm = &q;
		tdo_perm_inv = &qv;
	}
	frame->G_max = 0;
	frame->first[0] = 0;


	for (i = 0; i < G1->G_max; i++) {
		first = G1->first[i];
		len = G1->len[i];

		if (f_v) {
			cout << "tactical_decomposition::refine class " << i << " / " << G1->G_max << " of size " << len << endl;
		}


		int N;
		N = len;

		tdo_gradient *tdog;

		tdog = NEW_OBJECT(tdo_gradient);
		tdog->allocate(N);



		for (j = 0; j < len; j++) {

			if (f_v) {
				cout << "tactical_decomposition::refine class " << i << " / " << G1->G_max << " j=" << j << " / " << len << endl;
			}

			P->move_to(&p);
			Pv->move_to(&pv);
			Q->move_to(&q);
			Qv->move_to(&qv);

			G0->copy_frame_to(G_last);

			/* Gegenrichtung aus G0 nach
			 * tdo->G_last uebertragen. */
			if (j != 0) {
				tdo_perm->mult_apply_tau_r(first, first + j);
				tdo_perm_inv->mult_apply_tau_l(first, first + j);
			}

			G_current->init_derived_i_first(G1 /* G_old */, i);

			/* In tdo->G ist jetzt die erste Zeile des
			 * i-ten Bereichs ausgezeichnet. */

			if (f_v) {
				cout << "tactical_decomposition::refine before calc2" << endl;
			}
			calc2(v, verbose_level);
			if (f_v) {
				cout << "tactical_decomposition::refine after calc2" << endl;
			}

			tdo_scheme *tdos;

			if (f_v) {
				cout << "tactical_decomposition::refine before get_tdos" << endl;
			}
			tdos = get_tdos(G_current, tactical_decomposition::G_next, TRUE, verbose_level);
			if (f_v) {
				cout << "tactical_decomposition::refine after get_tdos" << endl;
			}

			if (f_v) {
				cout << "tactical_decomposition::refine before add_tdos" << endl;
			}
			tdog->add_tdos(tdos, j, verbose_level);
			if (f_v) {
				cout << "tactical_decomposition::refine after add_tdos" << endl;
			}

		} /* next j */


		if (f_v) {
			cout << "tactical_decomposition::refine computing refinement in class i=" << i << endl;
		}
		/* Verfeinerung des Bereichs (first, len)
		 * nach Typen der Ableitungen
		 * (jetzt in tdog), sortieren
		 * und nach frame schreiben.
		 * Die erfolgten Permutationen werden nach
		 * p/pv bzw. q/qv geschrieben: */
		j = 0;
		for (k = 0; k < tdog->nb_tdos; k++) {
			first1 = j;
			for (j1 = j; j1 < len; j1++) {
				if (tdog->type[j1] == k) {
					if (j1 != j) {
						/* (j j+1 ... j1) anwenden: */
						t = tdog->type[j1];
						for (l = j1; l > j; l--)
							tdog->type[l] = tdog->type[l - 1];
						tdog->type[j] = t;

						perm->mult_apply_forwc_r(first + j, j1 - j + 1);
						perm_inv->mult_apply_backwc_l(first + j, j1 - j + 1);
					}
					j++;
				}
			} /* next j1 */

			len1 = j - first1;
			/* Eintrag (first + first1, len1): */
			for (l = 0; l < len1; l++) {
				frame->grid_entry[first + first1 + l] = frame->G_max;
			}
			frame->len[frame->G_max] = len1;
			/* printf("at %d len = %d\n",
				frame->first[frame->G_max],
				frame->len[frame->G_max]); */
			frame->G_max++;
			frame->first[frame->G_max] = first + first1 + len1;
		}
		FREE_OBJECT(tdog);

	} /* next i */
	if (f_v) {
		cout << "tactical_decomposition::refine done" << endl;
	}
}

void tactical_decomposition::second_order_tdo(int v, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "tactical_decomposition::second_order_tdo" << endl;
	}
	//tdo_gradient *tdog;
	geo_frame *frame = NULL;
	grid *G_l, *G_c, *G_n, *Gtmp;
	cperm p, q, pv, qv;
	int f_points, m, n;

	p.init_and_identity(Encoding->v);
	pv.init_and_identity(Encoding->v);
	q.init_and_identity(Encoding->b);
	qv.init_and_identity(Encoding->b);

	frame = NEW_OBJECT(geo_frame);

	G_l = G_last;
	G_c = G_current;
	G_n = G_next;
	G_last = NEW_OBJECT(grid);
	G_current = NEW_OBJECT(grid);
	G_next = NEW_OBJECT(grid);

	//tdog = new tdo_gradient;
	//tdog->allocate(v + Encoding->b);


	while (TRUE) {
		/* aktuelles TDO jetzt in G, G_next;
		 * aktuelle Permutation
		 * in p, pv, q, qv. */
		f_points = TRUE;
		m = v;
		n = Encoding->b;

		/* tdo->p etc. sichern: */
		p.move_to(&p);
		pv.move_to(&pv);
		q.move_to(&q);
		qv.move_to(&qv);

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before refine" << endl;
		}
		refine(v, G_c, G_n, f_points, frame, &p, &pv, &q, &qv, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after refine" << endl;
		}
		/* Das (zeilenweise) verfeinerte
		 * TDO wird durch
		 * frame beschrieben (nur die Zeilen). */
		/* Die (Zeilen-) Permutationen
		 * auf das verfeinerte TDO
		 * befinden sich jetzt in p, pv */
		if (G_c->f_points == f_points) {
			G_n->copy_frame_to(G_last);
				/* Alte Block-TDO-Einteilung von G_n
				 * nach tdo->G_last kopieren. */
				/* setzt f_points, m, n */
		}
		else {
			G_c->copy_frame_to(G_last);
				/* Alte Block-TDO-Einteilung von G_c
				 * nach tdo->G_last kopieren. */
		}
		frame2grid(frame, G_current);
			/* Neue Punkt-TDO-Einteilung
			 * nach tdo->G. */

		G_current->f_points = f_points;
		G_current->m = m;
		G_current->n = n;

		recollect_types(v, G_last, G_current, 0 /*verbose_level*/);
		p.move_to(&p);
		pv.move_to(&pv);
		q.move_to(&q);
		qv.move_to(&qv);

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before calc2" << endl;
		}
		calc2(v, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after calc2" << endl;
		}

		/* Die Permutationen auf das (zeilenweise)
		 * verfeinerte TDO befinden sich jetzt in
		 * tdo->p, tdo->pv, tdo->q, tdo->qv. */

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before get_tdos" << endl;
		}
		tdos2 = get_tdos(G_current, G_next, TRUE, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after get_tdos" << endl;
		}

		Gtmp = G_c;
		G_c = G_current;
		G_current = Gtmp;
		Gtmp = G_n;
		G_n = G_next;
		G_next = Gtmp;

		// save p, pv, q, qv:
		p.move_to(&p);
		pv.move_to(&pv);
		q.move_to(&q);
		qv.move_to(&qv);

		f_points = FALSE;
		m = Encoding->b;
		n = v;
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before refine" << endl;
		}

		refine(v, G_c, G_n, f_points, frame, &p, &pv, &q, &qv, verbose_level);

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after refine" << endl;
		}
		/* Das (spaltenweise) verfeinerte
		 * TDO wird durch
		 * frame beschrieben. */
		/* Die (Spalten-) Permutationen
		 * auf das verfeinerte TDO
		 * befinden sich jetzt in q, qv */
		if (G_c->f_points == f_points) {
			G_n->copy_frame_to(G_last);
				/* Alte Punkt-TDO-Einteilung von G_n
				 * nach tdo->G_last kopieren. */
				/* setzt f_points, m, n */
		}
		else {
			G_c->copy_frame_to(G_last);
				/* Alte Punkt-TDO-Einteilung von G_c
				 * nach tdo->G_last kopieren. */
		}
		frame2grid(frame, G_current);
			/* Neue Block-TDO-Einteilung
			 * nach tdo->G. */
		G_current->f_points = f_points;
		G_current->m = m;
		G_current->n = n;
		recollect_types(v, G_last, G_current, 0 /*verbose_level*/);
		p.move_to(&p);
		pv.move_to(&pv);
		q.move_to(&q);
		qv.move_to(&qv);

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before calc2" << endl;
		}
		calc2(v, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after calc2" << endl;
		}

		/* Die Permutationen auf das (spaltenweise)
		 * verfeinerte TDO befinden sich jetzt in
		 * tdo->p, tdo->pv, tdo->q, tdo->qv. */

		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo before get_tdos" << endl;
		}
		tdos2 = get_tdos(G_current, G_next, TRUE, verbose_level);
		if (f_v) {
			cout << "tactical_decomposition::second_order_tdo after get_tdos" << endl;
		}

		break;
	}

	//delete tdog;

	if (frame) {
		FREE_OBJECT(frame);
	}
	if (G_last) {
		FREE_OBJECT(G_last);
	}
	if (G_current) {
		FREE_OBJECT(G_current);
	}
	if (G_next) {
		FREE_OBJECT(G_next);
	}
	G_last = G_l;
	G_current = G_c;
	G_next = G_n;

	if (f_v) {
		cout << "tactical_decomposition::second_order_tdo done" << endl;
	}
}


}}



