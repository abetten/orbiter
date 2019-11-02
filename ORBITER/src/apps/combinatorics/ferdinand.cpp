// ferdinand.cpp
// 
// Anton Betten
// May 2, 2017
//
//
// 
//
//

#include "orbiter.h"


using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;





void ferdinand(int level, int group, int subgroup, 
	int f_create_graph, int create_graph_level,
	int create_graph_index, int connection_element,
	int verbose_level);
int ferdinand_incremental_check_func(int len, int *S,
		void *data, int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int f_level = FALSE;
	int level = 0;
	int f_group = FALSE;
	int group = 0;
	int f_subgroup = FALSE;
	int subgroup = 0;
	int f_create_graph = FALSE;
	int create_graph_level = 0;
	int create_graph_index = 0;
	int connection_element = 0;
	int i;
	
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-level") == 0) {
			f_level = TRUE;
			level = atoi(argv[++i]);
			cout << "-level " << level << endl;
			}
		else if (strcmp(argv[i], "-group") == 0) {
			f_group = TRUE;
			group = atoi(argv[++i]);
			cout << "-group " << group << endl;
			}
		else if (strcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = TRUE;
			subgroup = atoi(argv[++i]);
			cout << "-subgroup " << subgroup << endl;
			}
		else if (strcmp(argv[i], "-create_graph") == 0) {
			f_create_graph = TRUE;
			create_graph_level = atoi(argv[++i]);
			create_graph_index = atoi(argv[++i]);
			connection_element = atoi(argv[++i]);
			cout << "-create_graph " << create_graph_level
					<< " " << create_graph_index << " "
					<< connection_element << endl;
			}
		}
	if (!f_level) {
		cout << "please use option -level <level>" << endl;
		exit(1);
		}
	if (!f_subgroup) {
		cout << "please use option -subgroup <subgroup>" << endl;
		exit(1);
		}
	if (!f_group) {
		cout << "please use option -group <group>" << endl;
		exit(1);
		}
	ferdinand(level, group, subgroup, 
		f_create_graph, create_graph_level,
		create_graph_index, connection_element,
		verbose_level);
	cout << "end" << endl;
}




void ferdinand(int level, int group, int subgroup, 
	int f_create_graph, int create_graph_level,
	int create_graph_index, int connection_element,
	int verbose_level)
{

	cayley_graph_search *Cayley;

	Cayley = NEW_OBJECT(cayley_graph_search);



	cout << "ferdinand level = " << level << " group = " << group
			<< " subgroup = " << subgroup << endl;

	cout << "before init" << endl;
	Cayley->init(level, group, subgroup, verbose_level);
	cout << "after init" << endl;


	cout << "before classify_subsets" << endl;
	Cayley->classify_subsets(verbose_level);
	cout << "after classify_subsets" << endl;


	cout << "before write_file" << endl;
	Cayley->write_file(verbose_level);
	cout << "after write_file" << endl;
	

	if (f_create_graph) {
		cout << "creating graph level=" << create_graph_level
				<< " index = " << create_graph_index << endl;

		set_and_stabilizer *SaS;

		SaS = Cayley->gen->get_set_and_stabilizer(create_graph_level, 
			create_graph_index, verbose_level);
		cout << "the orbit representative is:" << endl;
		SaS->print_set_tex(cout);
		cout << endl;

		int go;
		int *Adj;
		int *Adj_list;
		int *Additional_neighbor;
		int *Additional_neighbor_sz;

		go = Cayley->go;
		Additional_neighbor = NEW_int(go);
		Additional_neighbor_sz = NEW_int(go);
		Adj_list = NEW_int(go * SaS->sz);
		Adj = NEW_int(go * go);
		Cayley->create_Adjacency_list(Adj_list, 
			SaS->data, SaS->sz, 
			verbose_level);

		cout << "The adjacency list is:" << endl;
		int_matrix_print(Adj_list, go, SaS->sz);

		Cayley->create_additional_edges(Additional_neighbor, 
			Additional_neighbor_sz, 
			connection_element, 
			verbose_level);

		cout << "additional neighbors have been computed" << endl;


		int_vec_zero(Adj, go * go);

		int i, j, ii, jj, h;
		for (i = 0; i < go; i++) {
			for (h = 0; h < SaS->sz; h++) {
				j = Adj_list[i * SaS->sz + h];
				ii = Cayley->list_of_elements_inverse[i];
				jj = Cayley->list_of_elements_inverse[j];
				Adj[ii * go + jj] = 1;
				Adj[jj * go + ii] = 1;
				}
			if (Additional_neighbor_sz[i]) {
				j = Additional_neighbor[i];
				ii = Cayley->list_of_elements_inverse[i];
				jj = Cayley->list_of_elements_inverse[j];
				Adj[ii * go + jj] = 1;
				Adj[jj * go + ii] = 1;
				}
			}
		cout << "The adjacency matrix is:" << endl;
		for (i = 0; i < go; i++) {
			for (j = 0; j < go; j++) {
				cout << Adj[i * go + j];
				}
			cout << endl;
			}

		cout << "Maple output:" << endl;
		cout << "A := Matrix([" << endl;
		for (i = 0; i < go; i++) {
			cout << "[";
			for (j = 0; j < go; j++) {
				cout << Adj[i * go + j];
				if (j < go - 1) {
					cout << ",";
					}
				}
			cout << "]";
			if (i < go - 1) {
				cout << ",";
				}
			cout << endl;
			}
		cout << "]);" << endl;
		cout << "Eigenvalues(A);" << endl;


		colored_graph *CG;

		CG = NEW_OBJECT(colored_graph);

		CG->init_adjacency(go /* nb_points*/, 2 /* nb_colors */, 1,
			Cayley->f_subgroup /*colors*/, Adj, 0 /* verbose_level*/);

		char fname[1000];

		sprintf(fname, "F_%d_%d_%d.bin",
				go, create_graph_level, connection_element);
		CG->save(fname, verbose_level);

		FREE_OBJECT(CG);
		FREE_int(Adj_list);
		FREE_int(Adj);
		FREE_int(Additional_neighbor);
		FREE_int(Additional_neighbor_sz);

		}



}


int ferdinand_incremental_check_func(int len, int *S,
		void *data, int verbose_level)
{
	cayley_graph_search *Cayley = (cayley_graph_search *) data;

	return Cayley->incremental_check_func(len, S, verbose_level);

}




