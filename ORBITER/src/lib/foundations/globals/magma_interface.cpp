// magma_interface.C
//
// Anton Betten
//
// started: April 27, 2017
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


magma_interface::magma_interface()
{
}

magma_interface::~magma_interface()
{
}

void magma_interface::write_permutation_group(const char *fname_base,
	int group_order, int *Table, int *gens, int nb_gens,
	int verbose_level)
{
	char fname[1000];
	int i;
	combinatorics_domain Combi;
	file_io Fio;

	sprintf(fname, "%s.magma", fname_base);
	{
	ofstream fp(fname);
	
	fp << "G := PermutationGroup< " << group_order << " | " << endl;
	for (i = 0; i < nb_gens; i++) {
		Combi.perm_print_counting_from_one(fp,
				Table + gens[i] * group_order,
				group_order);
		if (i < nb_gens - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;
	}
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;
}

void magma_interface::normalizer_in_Sym_n(
	const char *fname_base,
	int group_order, int *Table, int *gens, int nb_gens,
	int *&N_gens, int &N_nb_gens, int &N_go,
	int verbose_level)
{
	char fname_magma[1000];
	char fname_output[1000];
	//char cmd[1000];
	int i;
	combinatorics_domain Combi;
	file_io Fio;

	sprintf(fname_magma, "%s.magma", fname_base);
	sprintf(fname_output, "%s.txt", fname_base);

	{
	ofstream fp(fname_magma);
	
	fp << "S := Sym(" << group_order << ");" << endl;


	fp << "G := PermutationGroup< " << group_order << " | " << endl;
	for (i = 0; i < nb_gens; i++) {
		Combi.perm_print_counting_from_one(fp,
			Table + gens[i] * group_order, group_order);
		if (i < nb_gens - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;
	
	fp << "N := Normalizer(S, G);" << endl;
	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #N; printf \"\\n\";" << endl;
	fp << "printf \"%o\", #Generators(N); printf \"\\n\";" << endl;
	fp << "for h := 1 to #Generators(N) do for i := 1 to "
		<< group_order << " do printf \"%o\", i^N.h; printf \" \"; "
		"if i mod 25 eq 0 then printf \"\n\"; end if; end for; "
		"printf \"\\n\"; end for;" << endl;
	fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << "/usr/local/magma/magma " << fname_magma << endl;
		exit(1);
		}
#if 0	
	sprintf(cmd, "/usr/local/magma/magma %s", fname_magma);
	cout << "executing normalizer command in MAGMA" << endl;
	system(cmd);
#endif

	cout << "normalizer command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;
	

	read_permutation_group(fname_output,
			group_order, N_gens, N_nb_gens, N_go, verbose_level);

#if 0
	{
	ifstream fp(fname_output);


	fp >> N_go;
	fp >> N_nb_gens;
	cout << "N_go = " << N_go << " nb_gens = " << N_nb_gens << endl;
	N_gens = NEW_int(N_nb_gens * group_order);
	for (i = 0; i < N_nb_gens; i++) {
		for (j = 0; j < group_order; j++) {
			fp >> a;
			a--;
			N_gens[i * group_order + j] = a;
			}
		}
	}
#endif

}

void magma_interface::read_permutation_group(const char *fname,
	int degree, int *&gens, int &nb_gens, int &go,
	int verbose_level)
{
	{
	ifstream fp(fname);
	int i, j, a;


	fp >> go;
	fp >> nb_gens;
	cout << "go = " << go << " nb_gens = " << nb_gens << endl;
	gens = NEW_int(nb_gens * degree);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < degree; j++) {
			fp >> a;
			a--;
			gens[i * degree + j] = a;
			}
		}
	}
}

void magma_interface::orbit_of_matrix_group_on_vector(
	const char *fname_base,
	int d, int q,
	int *initial_vector, int **gens, int nb_gens,
	int &orbit_length,
	int verbose_level)
{
	char fname_magma[1000];
	char fname_output[1000];
	//char cmd[1000];
	int i, j;
	combinatorics_domain Combi;
	file_io Fio;

	sprintf(fname_magma, "%s.magma", fname_base);
	sprintf(fname_output, "%s.txt", fname_base);

	{
	ofstream fp(fname_magma);


	fp << "G := MatrixGroup< " << d << ", GF(" << q << ") | " << endl;
	for (i = 0; i < nb_gens; i++) {
		fp << "[";
		for (j = 0; j < d * d; j++) {
			fp << gens[i][j];
			if (j < d * d - 1) {
				fp << ",";
				}
		}
		fp << "]";
		if (i < nb_gens - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;

	fp << "V := RSpace(G);" << endl;
	fp << "u := V![";
	for (j = 0; j < d; j++) {
		fp << initial_vector[j];
		if (j < d - 1) {
			fp << ",";
			}
	}

	fp << "];" << endl;
	fp << "O := Orbit(G,u);" << endl;


	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #O; printf \"\\n\";" << endl;
	fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << "/usr/local/magma/magma " << fname_magma << endl;
		exit(1);
		}
#if 0
	sprintf(cmd, "/usr/local/magma/magma %s", fname_magma);
	cout << "executing normalizer command in MAGMA" << endl;
	system(cmd);
#endif

	cout << "orbit command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;



	{
	ifstream fp(fname_output);


	fp >> orbit_length;
	}

}


void magma_interface::orbit_of_matrix_group_on_subspaces(
	const char *fname_base,
	int d, int q, int k,
	int *initial_subspace, int **gens, int nb_gens,
	int &orbit_length,
	int verbose_level)
{
	char fname_magma[1000];
	char fname_output[1000];
	//char cmd[1000];
	int i, j;
	combinatorics_domain Combi;
	file_io Fio;

	sprintf(fname_magma, "%s.magma", fname_base);
	sprintf(fname_output, "%s.txt", fname_base);

	{
	ofstream fp(fname_magma);


	fp << "G := MatrixGroup< " << d << ", GF(" << q << ") | " << endl;
	for (i = 0; i < nb_gens; i++) {
		fp << "[";
		for (j = 0; j < d * d; j++) {
			fp << gens[i][j];
			if (j < d * d - 1) {
				fp << ",";
				}
		}
		fp << "]";
		if (i < nb_gens - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;

	fp << "V := RSpace(G);" << endl;
	for (i = 0; i < k; i++) {
		fp << "u" << i << " := V![";
		for (j = 0; j < d; j++) {
			fp << initial_subspace[i * d + j];
			if (j < d - 1) {
				fp << ",";
				}
		}
		fp << "];" << endl;
	}

	fp << "W := sub< V | ";
	for (i = 0; i < k; i++) {
		fp << "u" << i;
		if (i < k - 1) {
			fp << ", ";
		}
	}
	fp << " >;" << endl;
	fp << "O := Orbit(G,W);" << endl;


	fp << "SetOutputFile(\"" << fname_output << "\");" << endl;
	fp << "printf \"%o\", #O; printf \"\\n\";" << endl;
	fp << "UnsetOutputFile();" << endl;
	}


	if (Fio.file_size(fname_output) == 0) {
		cout << "please run magma on the file " << fname_magma << endl;
		cout << "for instance, try" << endl;
		cout << "/usr/local/magma/magma " << fname_magma << endl;
		exit(1);
		}
#if 0
	sprintf(cmd, "/usr/local/magma/magma %s", fname_magma);
	cout << "executing normalizer command in MAGMA" << endl;
	system(cmd);
#endif

	cout << "orbit command in MAGMA has finished, written file "
		<< fname_output << " of size " << Fio.file_size(fname_output) << endl;



	{
	ifstream fp(fname_output);


	fp >> orbit_length;
	}

}



}}

