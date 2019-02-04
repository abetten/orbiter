// poset_orbit_node_io.C
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.C
// December 27, 2008
// renamed from io.C into oracle_io.C Aug 24, 2011


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

namespace orbiter {
namespace classification {

void poset_orbit_node::read_memory_object(
		poset_classification *PC,
		action *A, memory_object *m, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt;
	
	Elt = PC->Elt6;
	m->read_int(&node);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object "
				"node " << node << endl;
		cout << "cur_pointer=" << m->cur_pointer << endl;
		}
	m->read_int(&prev);
	m->read_int(&pt);
	m->read_int(&nb_strong_generators);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object "
				"nb_strong_generators " << nb_strong_generators << endl;
		}
	if (nb_strong_generators) {
		hdl_strong_generators = NEW_int(nb_strong_generators);
		tl = NEW_int(A->base_len);
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_from_memory_object(Elt, m, verbose_level - 2);
			hdl_strong_generators[i] = A->element_store(Elt, FALSE);
			nb_group_elements++;
			}
		for (i = 0; i < A->base_len; i++) {
			m->read_int(&tl[i]);
			}
		}
	else {
		hdl_strong_generators = NULL;
		tl = NULL;
		}
	m->read_int(&nb_extensions);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object nb_extensions "
				<< nb_extensions << endl;
		cout << "cur_pointer=" << m->cur_pointer << endl;
		}
	E = NEW_OBJECTS(extension, nb_extensions);
	if (f_v) {
		cout << "E allocated" << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"extension " << i << endl;
			}
		m->read_int(&E[i].pt);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"pt = " << E[i].pt << endl;
			}
		m->read_int(&E[i].orbit_len);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"pt = " << E[i].orbit_len << endl;
			}
		m->read_int(&E[i].type);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"type = " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->read_int(&E[i].data); // next poset_orbit_node
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			A->element_read_from_memory_object(Elt, m, verbose_level - 2);
			E[i].data = A->element_store(Elt, FALSE);
			m->read_int(&E[i].data1);
			m->read_int(&E[i].data2);
			nb_group_elements++;
			}
		else {
			cout << "poset_orbit_node::read_memory_object type "
					<< E[i].type << " is illegal" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::write_memory_object(
		poset_classification *PC,
		action *A, memory_object *m, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt;
	
	Elt = PC->Elt6;

	if (f_v) {
		cout << "poset_orbit_node::write_memory_object "
				"node " << node << endl;
		cout << "used_length=" << m->used_length << endl;
		}
	m->write_int(node);
	m->write_int(prev);
	m->write_int(pt);
	m->write_int(nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " "
				<< nb_strong_generators << endl;
		}
	for (i = 0; i < nb_strong_generators; i++) {
		A->element_retrieve(hdl_strong_generators[i], Elt, FALSE);
		A->element_write_to_memory_object(Elt, m, verbose_level);
		nb_group_elements++;
		}
	if (nb_strong_generators) {
		if (f_v) {
			cout << "writing tl" << endl;
			}
		for (i = 0; i < A->base_len; i++) {
			m->write_int(tl[i]);
			if (f_v) {
				cout << tl[i] << " ";
				}
			}
		if (f_v) {
			cout << endl;
			}
		}
	m->write_int(nb_extensions);
	if (f_v) {
		cout << "nb_extensions=" << nb_extensions << endl;
		cout << "used_length=" << m->used_length << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		m->write_int(E[i].pt);
		m->write_int(E[i].orbit_len);
		m->write_int(E[i].type);
		if (f_v) {
			cout << i << " : " << E[i].pt << " : "
					<< E[i].orbit_len << " : " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->write_int(E[i].data); // next poset_orbit_node
			if (f_v) {
				cout << "extension node, data=" << E[i].data << endl;
				}
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			if (f_v) {
				cout << "fusion node, hdl=" << E[i].data << endl;
				}
			A->element_retrieve(E[i].data, Elt, FALSE);
			A->element_write_to_memory_object(Elt, m, verbose_level);
			m->write_int(E[i].data1);
			m->write_int(E[i].data2);
			nb_group_elements++;
			}
		else {
			cout << "poset_orbit_node::write_memory: type "
						<< E[i].type << " is illegal" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "poset_orbit_node::write_memory_object node "
				<< node << " finished" << endl;
		}
}

int poset_orbit_node::calc_size_on_file(action *A, int verbose_level)
{
	int i, s = 0;
	s += 4 * sizeof(int); // node, prev, pt, nb_strong_generators
	//m->write_int(node);
	//m->write_int(prev);
	//m->write_int(pt);
	//m->write_int(nb_strong_generators);

	s += nb_strong_generators * A->coded_elt_size_in_char;
	if (nb_strong_generators) {
		s += A->base_len * sizeof(int);
		// tl[]
		}

	s += sizeof(int); // nb_extensions
	//m->write_int(nb_extensions);

	for (i = 0; i < nb_extensions; i++) {
		s += 3 * sizeof(int); // pt, orbit_len, type
		//m->write_int(E[i].pt);
		//m->write_int(E[i].orbit_len);
		//m->write_int(E[i].type);

		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			s += sizeof(int); // data
			//m->write_int(E[i].data); // next poset_orbit_node
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			s += A->coded_elt_size_in_char; // group element
			//A->element_retrieve(E[i].data, Elt, FALSE);
			//A->element_write_to_memory_object(Elt, m, verbose_level);

			s += 2 * sizeof(int); // data1, data2
			//m->write_int(E[i].data1);
			//m->write_int(E[i].data2);
			}
		}
	return s;
}


void poset_orbit_node::sv_read_file(poset_classification *PC,
		FILE *fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node << endl;
		}
	int hdl;

	if (nb_strong_generators) {
		hdl = hdl_strong_generators[0];
	} else {
		hdl = -1;
	}
	Schreier_vector = PC->Schreier_vector_handler->sv_read_file(
			hdl, nb_strong_generators,
			fp, verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node
				<< " finished" << endl;
		}
}

void poset_orbit_node::sv_write_file(poset_classification *PC,
		FILE *fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_write_file node " << node << endl;
		}

	PC->Schreier_vector_handler->sv_write_file(Schreier_vector,
			fp, verbose_level);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_write_file node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::read_file(action *A,
		FILE *fp, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;
	
	Elt = NEW_int(A->elt_size_in_int);
	node = fread_int4(fp);
	if (f_v) {
		cout << "poset_orbit_node::read_file node " << node << endl;
		}
	prev = fread_int4(fp);
	pt = fread_int4(fp);
	nb_strong_generators = fread_int4(fp);
	if (f_vv) {
		cout << "prev=" << prev << endl;
		cout << "pt=" << pt << endl;
		cout << "nb_strong_generators " << nb_strong_generators << endl;
		}
	if (nb_strong_generators) {
		hdl_strong_generators = NEW_int(nb_strong_generators);
		tl = NEW_int(A->base_len);
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_file_fp(Elt, fp, verbose_level);
			if (f_vv) {
				cout << "read element" << endl;
				A->element_print_quick(Elt, cout);
				}
			hdl_strong_generators[i] = A->element_store(Elt, FALSE);
			nb_group_elements++;
			}
		for (i = 0; i < A->base_len; i++) {
			tl[i] = fread_int4(fp);
			if (f_vv) {
				cout << "read tl[" << i << "]=" << tl[i] << endl;
				}
			}
		}
	else {
		hdl_strong_generators = NULL;
		tl = NULL;
		}
	nb_extensions = fread_int4(fp);
	if (f_vv) {
		cout << "nb_extensions " << nb_extensions << endl;
		}
	E = NEW_OBJECTS(extension, nb_extensions);
	if (f_vv) {
		cout << "E allocated" << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		if (f_vv) {
			cout << "poset_orbit_node::read_file extension " << i << endl;
			}
		E[i].pt = fread_int4(fp);
		if (f_vv) {
			cout << "pt = " << E[i].pt << endl;
			}
		E[i].orbit_len = fread_int4(fp);
		if (f_vv) {
			cout << "orbit_len = " << E[i].orbit_len << endl;
			}
		E[i].type = fread_int4(fp);
		if (f_vv) {
			cout << "type = " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			E[i].data = fread_int4(fp);
			// next poset_orbit_node
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			A->element_read_file_fp(Elt, fp, verbose_level);
			if (f_vv) {
				cout << "read element" << endl;
				A->element_print_quick(Elt, cout);
				}
			//element_read_file(A, Elt, elt, fp, verbose_level);
			E[i].data = A->element_store(Elt, FALSE);
			nb_group_elements++;
			}
		else if (E[i].type == EXTENSION_TYPE_PROCESSING) {
			cout << "poset_orbit_node::read_file: "
					"type EXTENSION_TYPE_PROCESSING is illegal" << endl;
			exit(1);
			}
		}
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_orbit_node::read_file node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::write_file(action *A,
		FILE *fp, int &nb_group_elements,
		int verbose_level)
{
	int i;
	int *Elt;
	int f_v = FALSE;//(verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	
	Elt = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "poset_orbit_node::write_file node " << node << endl;
		}
	fwrite_int4(fp, node);
	fwrite_int4(fp, prev);
	fwrite_int4(fp, pt);
	fwrite_int4(fp, nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " "
				<< nb_strong_generators << endl;
		}
	for (i = 0; i < nb_strong_generators; i++) {
		A->element_retrieve(hdl_strong_generators[i], Elt, FALSE);
		A->element_write_file_fp(Elt, fp, 0);
		nb_group_elements++;
		}
	if (nb_strong_generators) {
		if (f_vv) {
			cout << "writing tl" << endl;
			}
		for (i = 0; i < A->base_len; i++) {
			fwrite_int4(fp, tl[i]);
			if (f_vv) {
				cout << tl[i] << " ";
				}
			}
		if (f_vv) {
			cout << endl;
			}
		}
	fwrite_int4(fp, nb_extensions);
	if (f_vv) {
		cout << "nb_extensions=" << nb_extensions << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		fwrite_int4(fp, E[i].pt);
		fwrite_int4(fp, E[i].orbit_len);
		fwrite_int4(fp, E[i].type);
		if (f_vv) {
			cout << i << " : " << E[i].pt << " : "
					<< E[i].orbit_len << " : " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			fwrite_int4(fp, E[i].data);
			if (f_vv) {
				cout << "extension node, data=" << E[i].data << endl;
				}
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			if (f_vv) {
				cout << "fusion node, hdl=" << E[i].data << endl;
				}
			A->element_retrieve(E[i].data, Elt, FALSE);
			A->element_write_file_fp(Elt, fp, 0);
			nb_group_elements++;
			}
		else if (E[i].type == EXTENSION_TYPE_PROCESSING) {
			cout << "poset_orbit_node::write_file: "
					"type EXTENSION_TYPE_PROCESSING is illegal" << endl;
			exit(1);
			}
		}
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_orbit_node::write_file node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::save_schreier_forest(
	poset_classification *PC,
	schreier *Schreier,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::save_schreier_forest"
				<< endl;
	}
	if (PC->f_export_schreier_trees) {
		int orbit_no, nb_orbits;

		nb_orbits = Schreier->nb_orbits;
		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			char fname_mask_base[1000];
			char fname_mask[1000];

			PC->create_schreier_tree_fname_mask_base(
					fname_mask_base, node);

			sprintf(fname_mask, "%s.layered_graph", fname_mask_base);

			Schreier->export_tree_as_layered_graph(orbit_no,
					fname_mask,
					verbose_level);
		}
	}
	if (f_v) {
		cout << "poset_orbit_node::save_schreier_forest "
				"done" << endl;
	}
}

void poset_orbit_node::save_shallow_schreier_forest(
	poset_classification *PC,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::save_shallow_schreier_forest"
				<< endl;
	}
	if (PC->f_export_schreier_trees) {
		int orbit_no, nb_orbits;
		int *orbit_reps;

		if (Schreier_vector == NULL) {
			cout << "poset_orbit_node::save_shallow_schreier_forest "
					"Schreier_vector == NULL" << endl;
			exit(1);
		}
		Schreier_vector->count_number_of_orbits_and_get_orbit_reps(
					orbit_reps, nb_orbits);
		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			char fname_mask_base[1000];
			char fname_mask[1000];

			PC->create_shallow_schreier_tree_fname_mask_base(
					fname_mask_base, node);

			sprintf(fname_mask, "%s.layered_graph", fname_mask_base);

			Schreier_vector->export_tree_as_layered_graph(
					orbit_no, orbit_reps[orbit_no],
					fname_mask,
					verbose_level);
		}
	}
	if (f_v) {
		cout << "poset_orbit_node::save_shallow_schreier_forest "
				"done" << endl;
	}
}

void poset_orbit_node::draw_schreier_forest(
	poset_classification *PC,
	schreier *Schreier,
	int f_using_invariant_subset, action *AR,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::draw_schreier_forest"
				<< endl;
	}
	if (PC->f_draw_schreier_trees) {
		int orbit_no, nb_orbits;

		nb_orbits = Schreier->nb_orbits;
		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			char label[1000];
			int xmax = PC->schreier_tree_xmax;
			int ymax =  PC->schreier_tree_ymax;
			int f_circletext = PC->schreier_tree_f_circletext;
			int rad = PC->schreier_tree_rad;
			int f_embedded = PC->schreier_tree_f_embedded;
			int f_sideways = PC->schreier_tree_f_sideways;
			double scale = PC->schreier_tree_scale;
			double line_width = PC->schreier_tree_line_width;
			int f_has_point_labels = FALSE;
			int *point_labels = NULL;

			sprintf(label, "%sschreier_tree_node_%d_%d",
					PC->schreier_tree_prefix, node, orbit_no);

			if (f_using_invariant_subset) {
				f_has_point_labels = TRUE;
				point_labels = AR->G.ABR->points;
				}

			if (f_v) {
				cout << "poset_orbit_node::draw_schreier_forest"
						<< endl;
				cout << "Node " << node << " " << orbit_no
						<< " drawing schreier tree" << endl;
			}
			Schreier->draw_tree(label, orbit_no, xmax, ymax,
				f_circletext, rad,
				f_embedded, f_sideways,
				scale, line_width,
				f_has_point_labels, point_labels,
				verbose_level - 1);
			}

		char label_data[1000];
		sprintf(label_data, "%sschreier_data_node_%d.tex",
				PC->schreier_tree_prefix, node);
		Schreier->latex(label_data);
		}

	if (f_v) {
		cout << "poset_orbit_node::draw_schreier_forest"
				" done" << endl;
	}
}

}}




