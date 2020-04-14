/*
 * interface_combinatorics.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace interfaces {


interface_combinatorics::interface_combinatorics()
{
	argc = 0;
	argv = NULL;

	f_create_combinatorial_object = FALSE;
	Descr = NULL;
	f_save = FALSE;
	fname_prefix = FALSE;
	f_process_combinatorial_objects = FALSE;
	Job = NULL;
	f_bent = FALSE;
	bent_n = 0;
	f_random_permutation = FALSE;
	random_permutation_degree = 0;
	random_permutation_fname_csv = NULL;
}


void interface_combinatorics::print_help(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
		cout << "-create_combinatorial_object " << endl;
	}
	else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
		cout << "-process_combinatorial_objects " << endl;
	}
	else if (strcmp(argv[i], "-bent") == 0) {
		cout << "-bent <int : n>" << endl;
	}
	else if (strcmp(argv[i], "-random_permutation") == 0) {
		cout << "-random_permutation <ind : degree> <string : <fname_csv>" << endl;
	}
}

int interface_combinatorics::recognize_keyword(int argc, const char **argv, int i, int verbose_level)
{
	if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-bent") == 0) {
		return true;
	}
	else if (strcmp(argv[i], "-random_permutation") == 0) {
		return true;
	}
	return false;
}

void interface_combinatorics::read_arguments(int argc, const char **argv, int i0, int verbose_level)
{
	int i;

	cout << "interface_combinatorics::read_arguments" << endl;

	interface_combinatorics::argc = argc;
	interface_combinatorics::argv = argv;

	for (i = i0; i < argc; i++) {
		if (strcmp(argv[i], "-create_combinatorial_object") == 0) {
			f_create_combinatorial_object = TRUE;
			cout << "-create_combinatorial_object " << endl;
			Descr = NEW_OBJECT(combinatorial_object_description);
			i += Descr->read_arguments(argc - (i - 1),
					argv + i, verbose_level) - 1;
		}
		else if (strcmp(argv[i], "-save") == 0) {
			f_save = TRUE;
			fname_prefix = argv[++i];
			cout << "-save " << fname_prefix << endl;
		}
		else if (strcmp(argv[i], "-process_combinatorial_objects") == 0) {
			f_process_combinatorial_objects = TRUE;

			cout << "-process_combinatorial_objects " << endl;

			Job = NEW_OBJECT(projective_space_job_description);

			i += Job->read_arguments(argc - i,
				argv + i + 1, verbose_level) + 1;
		}
		else if (strcmp(argv[i], "-bent") == 0) {
			f_bent = TRUE;
			bent_n = atoi(argv[++i]);
			cout << "-bent " << bent_n << endl;
		}
		else if (strcmp(argv[i], "-random_permutation") == 0) {
			f_random_permutation = TRUE;
			random_permutation_degree = atoi(argv[++i]);
			random_permutation_fname_csv = argv[++i];
			cout << "-random_permutation " << random_permutation_degree << endl;
		}
	}
	cout << "interface_combinatorics::read_arguments done" << endl;
}


void interface_combinatorics::worker(int verbose_level)
{
	if (f_create_combinatorial_object) {
		do_create_combinatorial_object(verbose_level);
	}
	else if (f_process_combinatorial_objects) {
		do_process_combinatorial_object(verbose_level);
	}
	else if (f_bent) {
		do_bent(bent_n, verbose_level);
	}
	else if (f_random_permutation) {
		do_random_permutation(random_permutation_degree, random_permutation_fname_csv, verbose_level);
	}
}

void interface_combinatorics::do_create_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "interface_combinatorics::do_create_combinatorial_object" << endl;
	}

	combinatorial_object_create *COC;

	COC = NEW_OBJECT(combinatorial_object_create);

	if (f_v) {
		cout << "before COC->init" << endl;
	}
	COC->init(Descr, verbose_level);
	if (f_v) {
		cout << "after COC->init" << endl;
	}



	if (f_v) {
		cout << "we created a set of " << COC->nb_pts
				<< " points, called " << COC->fname << endl;
		cout << "list of points:" << endl;

		cout << COC->nb_pts << endl;
		for (i = 0; i < COC->nb_pts; i++) {
			cout << COC->Pts[i] << " ";
			}
		cout << endl;
	}





	if (f_save) {
		file_io Fio;
		char fname[1000];

		sprintf(fname, "%s%s", fname_prefix, COC->fname);

		if (f_v) {
			cout << "We will write to the file " << fname << endl;
		}
		Fio.write_set_to_file(fname, COC->Pts, COC->nb_pts, verbose_level);
		if (f_v) {
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;
		}
	}

	FREE_OBJECT(COC);

	if (f_v) {
		cout << "interface_combinatorics::do_create_combinatorial_object done" << endl;
	}
}

void interface_combinatorics::do_process_combinatorial_object(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_process_combinatorial_object" << endl;
	}

	if (!Job->f_q) {
		cout << "please use option -q <q> within the job description" << endl;
		exit(1);
	}
	if (!Job->f_n) {
		cout << "please use option -n <n> to specify the projective dimension  within the job description" << endl;
		exit(1);
	}
	if (!Job->f_fname_base_out) {
		cout << "please use option -fname_base_out <fname_base_out> within the job description" << endl;
		exit(1);
	}

	Job->perform_job(verbose_level);

	if (f_v) {
		cout << "interface_combinatorics::do_process_combinatorial_object done" << endl;
	}
}

void interface_combinatorics::do_bent(int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_bent" << endl;
	}

	{
		boolean_function *BF;

		BF = NEW_OBJECT(boolean_function);

		BF->init(n, verbose_level);

		BF->search_for_bent_functions(verbose_level);

		FREE_OBJECT(BF);
	}

	if (f_v) {
		cout << "interface_combinatorics::do_bent done" << endl;
	}
}

void interface_combinatorics::do_random_permutation(int deg, const char *fname_csv, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_combinatorics::do_random_permutation" << endl;
	}

	{
		combinatorics_domain Combi;
		file_io Fio;


		int *P;

		P = NEW_int(deg);
		Combi.random_permutation(P, deg);

		Fio.int_vec_write_csv(P, deg, fname_csv, "perm");
	}

	if (f_v) {
		cout << "interface_combinatorics::do_random_permutation done" << endl;
	}
}





}}
