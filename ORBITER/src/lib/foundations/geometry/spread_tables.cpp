/*
 * spread_tables.cpp
 *
 *  Created on: Feb 24, 2019
 *      Author: betten
 */




#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


spread_tables::spread_tables()
{
	q = 0;
	d = 4; // = 4
	F = NULL;
	P = NULL; // PG(3,q)
	Gr = NULL; // Gr_{4,2}
	nb_lines = 0;
	spread_size = 0;
	nb_iso_types_of_spreads = 0;

	prefix[0] = 0;

	fname_dual_line_idx[0] = 0;
	fname_self_dual_lines[0] = 0;
	fname_spreads[0] = 0;
	fname_isomorphism_type_of_spreads[0] = 0;
	fname_dual_spread[0] = 0;
	fname_self_dual_spreads[0] = 0;

	dual_line_idx = NULL;
	self_dual_lines = NULL;
	nb_self_dual_lines = 0;

	nb_spreads = 0;
	spread_table = NULL;
	spread_iso_type = NULL;
	dual_spread_idx = NULL;
	self_dual_spreads = NULL;
	nb_self_dual_spreads = 0;

	//null();
}

spread_tables::~spread_tables()
{
	if (P) {
		FREE_OBJECT(P);
	}
	if (Gr) {
		FREE_OBJECT(Gr);
	}
	if (dual_line_idx) {
		FREE_int(dual_line_idx);
	}
	if (self_dual_lines) {
		FREE_int(self_dual_lines);
	}
	if (spread_table) {
		FREE_lint(spread_table);
	}
	if (spread_iso_type) {
		FREE_int(spread_iso_type);
	}
	if (dual_spread_idx) {
		FREE_int(dual_spread_idx);
	}
	//freeself();
}

void spread_tables::init(finite_field *F,
		int f_load,
		int nb_iso_types_of_spreads,
		const char *prefix,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "spread_tables::init" << endl;
	}

	spread_tables::F = F;
	q = F->q;
	d = 4; // = 4
	P = NEW_OBJECT(projective_space);
	P->init(3, F,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);


	Gr = NEW_OBJECT(grassmann);
	Gr->init(d, 2, F, 0 /* verbose_level */);
	nb_lines = Gr->nCkq.as_int();
	spread_size = q * q + 1;
	spread_tables::nb_iso_types_of_spreads = nb_iso_types_of_spreads;

	if (f_v) {
		cout << "spread_tables::init nb_lines=" << nb_lines << endl;
		cout << "spread_tables::init spread_size=" << spread_size << endl;
		cout << "spread_tables::init nb_iso_types_of_spreads="
				<< nb_iso_types_of_spreads << endl;
	}


	sprintf(spread_tables::prefix, "%sspread_%d", prefix, NT.i_power_j(q, 2));
	if (f_v) {
		cout << "spread_tables::init prefix=" << spread_tables::prefix << endl;
	}

	sprintf(fname_dual_line_idx, "%s_dual_line_idx.csv", spread_tables::prefix);
	sprintf(fname_self_dual_lines, "%s_self_dual_lines.csv", spread_tables::prefix);
	sprintf(fname_spreads, "%s_spreads.csv", spread_tables::prefix);
	sprintf(fname_isomorphism_type_of_spreads, "%s_spreads_iso.csv", spread_tables::prefix);
	sprintf(fname_dual_spread, "%s_dual_spread_idx.csv", spread_tables::prefix);
	sprintf(fname_self_dual_spreads, "%s_self_dual_spreads.csv", spread_tables::prefix);


	if (f_v) {
		cout << "spread_tables::init before Gr->compute_dual_line_idx" << endl;
	}
	Gr->compute_dual_line_idx(dual_line_idx,
			self_dual_lines, nb_self_dual_lines,
			0 /*verbose_level*/);
	if (f_v) {
		cout << "spread_tables::init after Gr->compute_dual_line_idx" << endl;
	}

	if (f_load) {
		if (f_v) {
			cout << "spread_tables::init before load" << endl;
		}
		load(verbose_level);
		if (f_v) {
			cout << "spread_tables::init after load" << endl;
		}
	}


	if (f_v) {
		cout << "spread_tables::init done" << endl;
	}
}

void spread_tables::init_spread_table(int nb_spreads,
		long int *spread_table, int *spread_iso_type,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::init_spread_table" << endl;
	}
	spread_tables::nb_spreads = nb_spreads;
	spread_tables::spread_table = spread_table;
	spread_tables::spread_iso_type = spread_iso_type;
	if (f_v) {
		cout << "spread_tables::init_spread_table done" << endl;
	}
}

void spread_tables::init_tables(int nb_spreads,
		long int *spread_table, int *spread_iso_type,
		int *dual_spread_idx,
		int *self_dual_spreads, int nb_self_dual_spreads,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_tables::init_tables" << endl;
	}
	spread_tables::nb_spreads = nb_spreads;
	spread_tables::spread_table = spread_table;
	spread_tables::spread_iso_type = spread_iso_type;
	spread_tables::dual_spread_idx = dual_spread_idx;
	spread_tables::self_dual_spreads = self_dual_spreads;
	spread_tables::nb_self_dual_spreads = nb_self_dual_spreads;
	if (f_v) {
		cout << "spread_tables::init_tables done" << endl;
	}
}

void spread_tables::init_reduced(
		int nb_select, int *select,
		spread_tables *old_spread_table,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "spread_tables::init_reduced" << endl;
	}

	F = old_spread_table->F;
	q = F->q;
	d = 4; // = 4
	P = NEW_OBJECT(projective_space);
	P->init(3, F,
		TRUE /* f_init_incidence_structure */,
		0 /* verbose_level - 2 */);


	Gr = NEW_OBJECT(grassmann);
	Gr->init(d, 2, F, 0 /* verbose_level */);
	nb_lines = Gr->nCkq.as_int();
	spread_size = old_spread_table->spread_size;
	nb_iso_types_of_spreads = old_spread_table->nb_iso_types_of_spreads;


	nb_spreads = nb_select;
	if (f_v) {
		cout << "spread_tables::init_reduced allocating spread_table" << endl;
	}
	spread_table = NEW_lint(nb_spreads * spread_size);
	if (f_v) {
		cout << "spread_tables::init_reduced allocating spread_iso_type" << endl;
	}
	spread_iso_type = NEW_int(nb_spreads);
	for (i = 0; i < nb_spreads; i++) {
		a = select[i];
		lint_vec_copy(old_spread_table->spread_table + a * spread_size,
				spread_table + i * spread_size, spread_size);
		spread_iso_type[i] = old_spread_table->spread_iso_type[a];
	}
#if 0
	spread_tables::dual_spread_idx = dual_spread_idx;
	spread_tables::self_dual_spreads = self_dual_spreads;
	spread_tables::nb_self_dual_spreads = nb_self_dual_spreads;
#endif
	if (f_v) {
		cout << "spread_tables::init_reduced done" << endl;
	}
}

void spread_tables::classify_self_dual_spreads(int *&type,
		set_of_sets *&SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a;

	if (f_v) {
		cout << "spread_tables::classify_self_dual_spreads" << endl;
	}
	type = NEW_int(nb_iso_types_of_spreads);
	int_vec_zero(type, nb_iso_types_of_spreads);
	for (i = 0; i < nb_self_dual_spreads; i++) {
		a = spread_iso_type[i];
		type[a]++;
	}
	SoS = NEW_OBJECT(set_of_sets);
	SoS->init_basic(
			nb_self_dual_spreads /* underlying_set_size */,
			nb_iso_types_of_spreads /* nb_sets */,
			type, 0 /* verbose_level */);
	for (a = 0; a < nb_iso_types_of_spreads; a++) {
		SoS->Set_size[a] = 0;
	}
	for (i = 0; i < nb_self_dual_spreads; i++) {
		a = spread_iso_type[i];
		SoS->Sets[a][SoS->Set_size[a]++] = i;
	}

	if (f_v) {
		cout << "spread_tables::classify_self_dual_spreads done" << endl;
	}
}

void spread_tables::save(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "spread_tables::save" << endl;
	}

	if (f_v) {
		cout << "spread_tables::save "
				"writing file " << fname_spreads << endl;
		}

	Fio.lint_matrix_write_csv(fname_spreads,
			spread_table, nb_spreads, spread_size);
	if (f_v) {
		cout << "spread_tables::save "
				"written file " << fname_spreads << endl;
		}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_isomorphism_type_of_spreads
				<< endl;
		}
	Fio.int_vec_write_csv(
			spread_iso_type, nb_spreads,
			fname_isomorphism_type_of_spreads,
			"isomorphism_type_of_spread");
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_isomorphism_type_of_spreads
				<< endl;
		}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_dual_spread
				<< endl;
		}
	Fio.int_vec_write_csv(
			dual_spread_idx, nb_spreads,
			fname_dual_spread,
			"dual_spread_idx");
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_dual_spread
				<< endl;
		}

	if (f_v) {
		cout << "spread_tables::save, "
				"writing file " << fname_self_dual_spreads
				<< endl;
		}
	Fio.int_vec_write_csv(
			self_dual_spreads, nb_self_dual_spreads,
			fname_self_dual_spreads,
			"self_dual_spreads");
	if (f_v) {
		cout << "spread_tables::save, "
				"written file " << fname_self_dual_spreads
				<< endl;
		}



	if (f_v) {
		cout << "spread_tables::save done" << endl;
	}
}

void spread_tables::load(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b;
	file_io Fio;

	if (f_v) {
		cout << "spread_tables::load" << endl;
	}

	if (f_v) {
		cout << "spread_tables::load "
				"reading file " << fname_spreads << endl;
		}

	Fio.lint_matrix_read_csv(fname_spreads,
			spread_table, nb_spreads, b,
			0 /* verbose_level */);
	if (b != spread_size) {
		cout << "spread_tables::load b != spread_size" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load "
				"read file " << fname_spreads << endl;
		}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_isomorphism_type_of_spreads
				<< endl;
		}
	Fio.int_matrix_read_csv(fname_isomorphism_type_of_spreads,
			spread_iso_type, a, b,
			0 /* verbose_level */);
	if (a != nb_spreads) {
		cout << "spread_tables::load a != nb_spreads" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_isomorphism_type_of_spreads
				<< endl;
		}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_dual_spread
				<< endl;
		}
	Fio.int_matrix_read_csv(fname_dual_spread,
			dual_spread_idx, a, b,
			0 /* verbose_level */);
	if (a != nb_spreads) {
		cout << "spread_tables::load a != nb_spreads" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_dual_spread
				<< endl;
		}

	if (f_v) {
		cout << "spread_tables::load, "
				"reading file " << fname_self_dual_spreads
				<< endl;
		}
	Fio.int_matrix_read_csv(fname_self_dual_spreads,
			self_dual_spreads, nb_self_dual_spreads, b,
			0 /* verbose_level */);
	if (f_v) {
		cout << "spread_tables::load, "
				"read file " << fname_self_dual_spreads
				<< endl;
		}



	if (f_v) {
		cout << "spread_tables::load done" << endl;
	}
}


void spread_tables::compute_adjacency_matrix(
		uchar *&bitvector_adjacency,
		long int &bitvector_length,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j, k, N2, cnt;

	if (f_v) {
		cout << "spread_tables::compute_adjacency_matrix" << endl;
		}

	N2 = ((long int) nb_spreads * (long int) nb_spreads) >> 1;

	bitvector_length = (N2 + 7) >> 3;

	bitvector_adjacency = NEW_uchar(bitvector_length);

	if (f_v) {
		cout << "after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
		}
	k = 0;
	cnt = 0;
	for (i = 0; i < nb_spreads; i++) {
		for (j = i + 1; j < nb_spreads; j++) {

			long int *p, *q;
			int u, v;

			p = spread_table + i * spread_size;
			q = spread_table + j * spread_size;
			u = v = 0;
			while (u + v < 2 * spread_size) {
				if (p[u] == q[v]) {
					break;
					}
				if (u == spread_size) {
					v++;
					}
				else if (v == spread_size) {
					u++;
					}
				else if (p[u] < q[v]) {
					u++;
					}
				else {
					v++;
					}
				}
			if (u + v < 2 * spread_size) {
				bitvector_m_ii(bitvector_adjacency, k, 0);

				}
			else {
				bitvector_m_ii(bitvector_adjacency, k, 1);
				cnt++;
				}

			k++;
			if ((k & ((1 << 21) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k
						<< " / " << N2 << endl;
				}
			}
		}





	{
	colored_graph *CG;
	char fname[1000];
	file_io Fio;

	CG = NEW_OBJECT(colored_graph);
	int *color;

	color = NEW_int(nb_spreads);
	int_vec_zero(color, nb_spreads);

	CG->init(nb_spreads, 1, 1,
			color, bitvector_adjacency,
			FALSE, verbose_level);

	sprintf(fname, "%s_disjoint_spreads.colored_graph", prefix);

	CG->save(fname, verbose_level);

	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	FREE_int(color);
	FREE_OBJECT(CG);
	}


	if (f_v) {
		cout << "spread_tables::compute_adjacency_matrix done" << endl;
		}
}

int spread_tables::test_if_spreads_are_disjoint(int a, int b)
{
	long int *p, *q;
	int u, v;

	p = spread_table + a * spread_size;
	q = spread_table + b * spread_size;
	u = v = 0;
	while (u + v < 2 * spread_size) {
		if (p[u] == q[v]) {
			break;
			}
		if (u == spread_size) {
			v++;
			}
		else if (v == spread_size) {
			u++;
			}
		else if (p[u] < q[v]) {
			u++;
			}
		else {
			v++;
			}
		}
	if (u + v < 2 * spread_size) {
		return FALSE;
		}
	else {
		return TRUE;
		}

}




}}

