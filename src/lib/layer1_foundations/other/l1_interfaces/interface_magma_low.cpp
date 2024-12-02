/*
 * interface_magma_low.cpp
 *
 *  Created on: Jan 27, 2023
 *      Author: betten
 */



#include "foundations.h"


using namespace std;


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace l1_interfaces {


interface_magma_low::interface_magma_low()
{
	Record_birth();
}

interface_magma_low::~interface_magma_low()
{
	Record_death();
}

void interface_magma_low::magma_set_stabilizer_in_collineation_group(
		algebra::field_theory::finite_field *F,
		int d, long int *Pts, int nb_pts,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_magma_low::magma_set_stabilizer_in_collineation_group" << endl;
	}

	string fname2;
	int *v;
	int h, i, a, b;
	data_structures::string_tools ST;

	v = NEW_int(d);
	fname2.assign(fname);
	ST.replace_extension_with(fname2, ".magma");

	{
		ofstream fp(fname2);

		fp << "G,I:=PGammaL(" << d << "," << F->q
				<< ");F:=GF(" << F->q << ");" << endl;
		fp << "S:={};" << endl;
		fp << "a := F.1;" << endl;
		for (h = 0; h < nb_pts; h++) {
			F->Projective_space_basic->PG_element_unrank_modified_lint(v, 1, d, Pts[h]);

			F->Projective_space_basic->PG_element_normalize_from_front(v, 1, d);

			fp << "Include(~S,Index(I,[";
			for (i = 0; i < d; i++) {
				a = v[i];
				if (a == 0) {
					fp << "0";
				}
				else if (a == 1) {
					fp << "1";
				}
				else {
					b = F->log_alpha(a);
					fp << "a^" << b;
				}
				if (i < d - 1) {
					fp << ",";
				}
			}
			fp << "]));" << endl;
		}
		fp << "Stab := Stabilizer(G,S);" << endl;
		fp << "Size(Stab);" << endl;
		fp << endl;
	}
	orbiter_kernel_system::file_io Fio;

	cout << "Written file " << fname2 << " of size "
			<< Fio.file_size(fname2) << endl;

	FREE_int(v);
	if (f_v) {
		cout << "interface_magma_low::magma_set_stabilizer_in_collineation_group done" << endl;
	}
}

void interface_magma_low::export_colored_graph_to_magma(
		combinatorics::graph_theory::colored_graph *Gamma,
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int *neighbors;
	int nb_neighbors;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "interface_magma_low::export_colored_graph_to_magma" << endl;
	}
	{
		ofstream fp(fname);

		neighbors = NEW_int(Gamma->nb_points);
		fp << "G := Graph< " << Gamma->nb_points << " | [" << endl;
		for (i = 0; i < Gamma->nb_points; i++) {


			nb_neighbors = 0;
			for (j = 0; j < Gamma->nb_points; j++) {
				if (j == i) {
					continue;
				}
				if (Gamma->is_adjacent(i, j)) {
					neighbors[nb_neighbors++] = j;
				}
			}

			fp << "{";
			for (j = 0; j < nb_neighbors; j++) {
				fp << neighbors[j] + 1;
				if (j < nb_neighbors - 1) {
					fp << ",";
				}
			}
			fp << "}";
			if (i < Gamma->nb_points - 1) {
				fp << ", " << endl;
			}
		}

		FREE_int(neighbors);

		fp << "]>;" << endl;

//> G := Graph< 9 | [ {4,5,6,7,8,9}, {4,5,6,7,8,9}, {4,5,6,7,8,9},
//>                   {1,2,3,7,8,9}, {1,2,3,7,8,9}, {1,2,3,7,8,9},
//>                   {1,2,3,4,5,6}, {1,2,3,4,5,6}, {1,2,3,4,5,6} ]>;


	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "interface_magma_low::export_colored_graph_to_magma" << endl;
	}
}

void interface_magma_low::export_linear_code(
		std::string &fname,
		algebra::field_theory::finite_field *F,
		int *genma, int n, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "interface_magma_low::export_linear_code" << endl;
	}

	ofstream ost(fname);
	int i, j, a;

	ost << "K<w> := GF(" << F->q << ");" << endl;
	ost << "V := VectorSpace(K, " << n << ");" << endl;
	ost << "C := LinearCode(sub<V |" << endl;
	for (i = 0; i < k; i++) {
		ost << "[";
		for (j = 0; j < n; j++) {
			a = genma[i * n + j];
			if (F->e == 1) {
				ost << a;
			}
			else {
				if (a <= 1) {
					ost << a;
				}
				else {
					ost << "w^" << F->log_alpha(a);
				}
			}
			if (j < n - 1) {
				ost << ",";
			}
		}
		ost << "]";
		if (i < k - 1) {
			ost << "," << endl;
		}
		else {
			ost << ">);" << endl;
		}
	}
	if (f_v) {
		cout << "interface_magma_low::export_linear_code done" << endl;
	}
}

void interface_magma_low::read_permutation_group(
		std::string &fname,
	int degree, int *&gens, int &nb_gens, int &go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "interface_magma_low::read_permutation_group" << endl;
	}
	{
	ifstream fp(fname);
	int i, j, a;


	fp >> go;
	fp >> nb_gens;
	if (f_v) {
		cout << "interface_magma_low::read_permutation_group "
				"go = " << go << " nb_gens = " << nb_gens << endl;
	}
	gens = NEW_int(nb_gens * degree);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < degree; j++) {
			fp >> a;
			a--;
			gens[i * degree + j] = a;
			}
		}
	}
	if (f_v) {
		cout << "interface_magma_low::read_permutation_group done" << endl;
	}
}

void interface_magma_low::run_magma_file(
		std::string &fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	string cmd;

	cmd = orbiter_kernel_system::Orbiter->magma_path + "magma " + fname;

	if (f_v) {
		cout << "executing: " << cmd << endl;
	}
	system(cmd.c_str());
}



}}}}




