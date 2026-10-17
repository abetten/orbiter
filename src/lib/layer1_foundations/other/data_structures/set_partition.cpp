/*
 * set_partition.cpp
 *
 *  Created on: Oct 4, 2026
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace data_structures {



set_partition::set_partition()
{
	Record_birth();


	underlying_set_size = 0;

	nb_classes = 0;
	Class_size = NULL;

	perm = NULL;

	perm_inv = NULL;

	class_index = NULL;

	Fst = NULL;
}



set_partition::~set_partition()
{
	Record_death();

	if (Class_size) {
		FREE_int(Class_size);
	}
	if (perm) {
		FREE_int(perm);
	}
	if (perm_inv) {
		FREE_int(perm_inv);
	}
	if (class_index) {
		FREE_int(class_index);
	}
	if (Fst) {
		FREE_int(Fst);
	}
}

void set_partition::init_SoS(
		other::data_structures::set_of_sets *SoS,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_partition::init_SoS" << endl;
	}

	int i;

	nb_classes = SoS->nb_sets;
	underlying_set_size = SoS->underlying_set_size;

	Class_size = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {
		Class_size[i] = SoS->Set_size[i];
	}


	Fst = NEW_int(nb_classes);

	perm = NEW_int(underlying_set_size);
	perm_inv = NEW_int(underlying_set_size);
	class_index = NEW_int(underlying_set_size);

	int fst, l, n, a;

	fst = 0;
	for (l = 0; l < nb_classes; l++) {
		Fst[l] = fst;
		for (n = 0; n < Class_size[l]; n++) {
			a = SoS->Sets[l][n];
			perm[fst + n] = a;
			perm_inv[a] = fst + n;
			class_index[fst + n] = l;
		}
		fst += Class_size[l];
	}


	if (f_v) {
		cout << "set_partition::init_SoS done" << endl;
	}
}

void set_partition::print()
{
	cout << "Set partition of a set of size " << underlying_set_size << " with " << nb_classes << " classes" << endl;
	cout << "underlying_set_size = " << underlying_set_size << endl;
	cout << "class size partition: ";

	int i;

	for (i = 0; i < nb_classes; i++) {
		if (i) {
			cout << " + ";
		}
		cout << Class_size[i];
	}
	cout << endl;
	for (i = 0; i < nb_classes; i++) {
		cout << setw(3) << i << " : ";
		cout << setw(3) << Fst[i] << " : ";
		cout << setw(3) << Class_size[i] << endl;
	}
}

void set_partition::init_layered_graph(
		layer1_foundations::combinatorics::graph_theory::layered_graph *Layered_graph,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_partition::init_layered_graph" << endl;
	}

	int i;


	underlying_set_size = Layered_graph->nb_nodes_total;
	nb_classes = Layered_graph->nb_layers;

	Class_size = NEW_int(nb_classes);

	for (i = 0; i < nb_classes; i++) {
		Class_size[i] = Layered_graph->L[i].nb_nodes;
	}


	Fst = NEW_int(nb_classes);
	perm = NEW_int(underlying_set_size);
	perm_inv = NEW_int(underlying_set_size);
	class_index = NEW_int(underlying_set_size);

	int fst, l, n, a;

	fst = 0;
	for (l = 0; l < nb_classes; l++) {

		Fst[l] = fst;

		for (n = 0; n < Class_size[l]; n++) {
			a = Layered_graph->L[l].Nodes[n].id;
			perm[fst + n] = a;
			perm_inv[a] = fst + n;
			class_index[fst + n] = l;
		}
		fst += Class_size[l];
	}


	if (f_v) {
		cout << "set_partition::init_layered_graph done" << endl;
	}
}

void set_partition::split_last_class(
		other::data_structures::tally_vector_data *T,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_partition::split_last_class" << endl;
	}

	if (T->data_length != Class_size[nb_classes - 1]) {
		cout << "set_partition::split_last_class "
				"T->data_length != Class_size[nb_classes - 1]" << endl;
		exit(1);
	}

	data_structures::set_of_sets *SoS_last;
	data_structures::set_of_sets *SoS_relabeled;

	SoS_last = T->get_set_partition(verbose_level);

	int nb_subclasses;

	nb_subclasses = SoS_last->nb_sets;

	SoS_relabeled = SoS_last->copy();


	if (f_v) {
		cout << "set_partition::split_last_class computing "
				"SoS_relabeled" << endl;
	}


	int i, j, l;
	long int a, b;
	int fst;

	fst = Fst[nb_classes - 1];

	for (i = 0; i < nb_subclasses; i++) {

		l = SoS_last->Set_size[i];

		for (j = 0; j < l; j++) {

			a = SoS_last->Sets[i][j];
			b = perm[fst + a];
			SoS_relabeled->Sets[i][j] = b;
		}
	}

	// Next, we will split the partition.
	// Only the last class is split.

	if (f_v) {
		cout << "set_partition::split_last_class "
				"refining the partition" << endl;
	}

	int nb_classes_new;
	int *Class_size_new; // [nb_classes_new]
	int *Fst_new; // [nb_classes_new]


	nb_classes_new = nb_classes - 1 + nb_subclasses;
	Class_size_new = NEW_int(nb_classes_new);
	Fst_new = NEW_int(nb_classes_new);

	Int_vec_copy(Class_size, Class_size_new, nb_classes);
	Int_vec_copy(Fst, Fst_new, nb_classes);

	for (i = 0; i < nb_subclasses; i++) {
		l = SoS_last->Set_size[i];
		Class_size_new[nb_classes - 1 + i] = l;
		if (i) {
			Fst_new[nb_classes - 1 + i] =
					Fst_new[nb_classes - 1 + i - 1] + SoS_last->Set_size[i - 1];
		}
		fst = Fst_new[nb_classes - 1 + i];
		for (j = 0; j < l; j++) {
			b = SoS_relabeled->Sets[i][j];
			perm[fst + j] = b;
			perm_inv[b] = fst + j;
			class_index[fst + j] = nb_classes - 1 + i;
		}
	}

	FREE_int(Class_size);
	FREE_int(Fst);

	Class_size = Class_size_new;
	Fst = Fst_new;
	nb_classes = nb_classes_new;


	FREE_OBJECT(SoS_last);
	FREE_OBJECT(SoS_relabeled);


	if (f_v) {
		cout << "set_partition::split_last_class done" << endl;
	}
}



}}}}


