// graph_layer.cpp
// 
// Anton Betten
// December 30, 2013
//
//
// 
//
//

#include "foundations.h"


using namespace std;


namespace orbiter {
namespace foundations {


graph_layer::graph_layer()
{
	null();
}

graph_layer::~graph_layer()
{
	freeself();
}

void graph_layer::null()
{
	Nodes = NULL;
}

void graph_layer::freeself()
{
	if (Nodes) {
		FREE_OBJECTS(Nodes);
		}
	null();
}

void graph_layer::init(int nb_nodes,
		int id_of_first_node, int verbose_level)
{
	int i, id;

	graph_layer::id_of_first_node = id_of_first_node;
	graph_layer::nb_nodes = nb_nodes;
	Nodes = NEW_OBJECTS(graph_node, nb_nodes);
	id = id_of_first_node;
	for (i = 0; i < nb_nodes; i++) {
		Nodes[i].id =id;
		Nodes[i].layer = i;
		Nodes[i].label = NULL;
		id++;
		}
}

void graph_layer::place(int verbose_level)
{
	double dx, dx2;
	int i;

	dx = 1. / nb_nodes;
	dx2 = dx * .5;
	for (i = 0; i < nb_nodes; i++) {
		Nodes[i].x_coordinate = i * dx + dx2;
		}
	
}

void graph_layer::place_with_grouping(
		int *group_size, int nb_groups,
		double x_stretch, int verbose_level)
// x_stretch is less than 1.
{
	int f_v = (verbose_level >= 1);
	int *group_start;
	double *group_x;
	double *group_dx;
	int i, j, nb_elements;

	if (f_v) {
		cout << "graph_layer::place_with_grouping "
				"nb_groups=" << nb_groups << endl;
		}
	group_start = NEW_int(nb_groups + 1);
	group_dx = new double[nb_groups + 1];
	group_x = new double[nb_groups + 1];
	group_start[0] = 0;
	for (j = 0; j < nb_groups; j++) {
		group_start[j + 1] = group_start[j] + group_size[j];
		}
	nb_elements = group_start[nb_groups];
	for (j = 0; j < nb_groups; j++) {
		group_dx[j] = group_size[j] / (double) nb_elements;
		}
	for (j = 0; j < nb_groups; j++) {
		group_x[j] = (double) group_start[j] / (double) nb_elements
				+ (double) group_dx[j] * .5;
		}
	for (j = 0; j < nb_groups; j++) {
		if (f_v) {
			cout << "j=" << j << " / " << nb_groups
					<< " group_size[j]=" << group_size[j] << endl;
			}
		for (i = 0; i < group_size[j]; i++) {
			if (FALSE) {
				cout << "i=" << i << " / " << group_size[j] << endl;
				}
			Nodes[group_start[j] + i].x_coordinate = 
			group_x[j] - 
				((double) group_dx[j] * .5) * x_stretch + 
	(((double)i + .5) * group_dx[j] / (double) group_size[j]) * x_stretch;
			}
		}
	FREE_int(group_start);
	delete [] group_dx;
	delete [] group_x;
	if (f_v) {
		cout << "graph_layer::place_with_grouping done" << endl;
		}
}

void graph_layer::write_memory_object(
		memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "graph_layer::write_memory_object "
				<< nb_nodes << " nodes" << endl;
		}
	m->write_int(id_of_first_node);
	m->write_int(nb_nodes);
	m->write_int(id_of_first_node);
	for (i = 0; i < nb_nodes; i++) {
		Nodes[i].write_memory_object(m, verbose_level - 1);
		}
	m->write_double(y_coordinate);
	if (f_v) {
		cout << "graph_layer::write_memory_object finished, "
				"data size (in chars) = " << m->used_length << endl;
		}
}

void graph_layer::read_memory_object(
		memory_object *m, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "graph_layer::read_memory_object" << endl;
		}
	freeself();

	m->read_int(&id_of_first_node);
	m->read_int(&nb_nodes);
	m->read_int(&id_of_first_node);

	Nodes = NEW_OBJECTS(graph_node, nb_nodes);

	for (i = 0; i < nb_nodes; i++) {
		Nodes[i].read_memory_object(m, verbose_level - 1);
		}
	m->read_double(&y_coordinate);
	if (f_v) {
		cout << "graph_layer::read_memory_object finished" << endl;
		}
}

}
}




