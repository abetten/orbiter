// poset_orbit_node_io.cpp
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.cpp
// December 27, 2008
// renamed from io.cpp into oracle_io.cpp Aug 24, 2011


#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

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
	
	Elt = NEW_int(A->elt_size_in_int);

	//Elt = PC->Elt6;
	m->read_int(&node);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object "
				"node " << node << endl;
		cout << "cur_pointer=" << m->cur_pointer << endl;
	}
	m->read_int(&prev);
	m->read_lint(&pt);
	m->read_int(&nb_strong_generators);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object "
				"nb_strong_generators " << nb_strong_generators << endl;
	}
	if (nb_strong_generators) {
#if 0
		hdl_strong_generators = NEW_int(nb_strong_generators);
		tl = NEW_int(A->base_len());
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_from_memory_object(Elt, m, verbose_level - 2);
			hdl_strong_generators[i] = A->element_store(Elt, FALSE);
			nb_group_elements++;
		}
#else
		first_strong_generator_handle = -1;
		tl = NEW_int(A->base_len());
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_from_memory_object(Elt, m, verbose_level - 2);
			if (i == 0) {
				first_strong_generator_handle = A->element_store(Elt, FALSE);
			}
			else {
				A->element_store(Elt, FALSE);
			}
			nb_group_elements++;
		}
#endif
		for (i = 0; i < A->base_len(); i++) {
			m->read_int(&tl[i]);
		}
	}
	else {
		//hdl_strong_generators = NULL;
		first_strong_generator_handle = -1;
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
		long int a;
		int b;

		m->read_lint(&a);
		E[i].set_pt(a);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"pt = " << E[i].get_pt() << endl;
		}
		m->read_int(&b);
		E[i].set_orbit_len(b);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"pt = " << E[i].get_orbit_len() << endl;
		}
		m->read_int(&b);
		E[i].set_type(b);
		if (f_v) {
			cout << "poset_orbit_node::read_memory_object "
					"type = " << E[i].get_type() << endl;
		}
		if (b == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->read_int(&b); // next poset_orbit_node
			E[i].set_data(b);
		}
		else if (b == EXTENSION_TYPE_FUSION) {
			// fusion node
			A->element_read_from_memory_object(Elt, m, verbose_level - 2);
			E[i].set_data(A->element_store(Elt, FALSE));
			m->read_int(&b);
			E[i].set_data1(b);
			m->read_int(&b);
			E[i].set_data2(b);
			nb_group_elements++;
		}
		else {
			cout << "poset_orbit_node::read_memory_object type "
					<< E[i].get_type() << " is illegal" << endl;
			exit(1);
		}
	}
	FREE_int(Elt);
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
	
	Elt = NEW_int(A->elt_size_in_int);

	//Elt = PC->Elt6;

	if (f_v) {
		cout << "poset_orbit_node::write_memory_object "
				"node " << node << endl;
		cout << "used_length=" << m->used_length << endl;
	}
	m->write_int(node);
	m->write_int(prev);
	m->write_lint(pt);
	m->write_int(nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " "
				<< nb_strong_generators << endl;
	}
	for (i = 0; i < nb_strong_generators; i++) {
		//A->element_retrieve(hdl_strong_generators[i], Elt, FALSE);
		A->element_retrieve(first_strong_generator_handle + i, Elt, FALSE);
		A->element_write_to_memory_object(Elt, m, verbose_level);
		nb_group_elements++;
	}
	if (nb_strong_generators) {
		if (f_v) {
			cout << "writing tl" << endl;
		}
		for (i = 0; i < A->base_len(); i++) {
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
		m->write_lint(E[i].get_pt());
		m->write_int(E[i].get_orbit_len());
		m->write_int(E[i].get_type());
		if (f_v) {
			cout << i << " : " << E[i].get_pt() << " : "
					<< E[i].get_orbit_len() << " : " << E[i].get_type() << endl;
		}
		if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->write_int(E[i].get_data()); // next poset_orbit_node
			if (f_v) {
				cout << "extension node, data=" << E[i].get_data() << endl;
			}
		}
		else if (E[i].get_type() == EXTENSION_TYPE_FUSION) {
			// fusion node
			if (f_v) {
				cout << "fusion node, hdl=" << E[i].get_data() << endl;
			}
			A->element_retrieve(E[i].get_data(), Elt, FALSE);
			A->element_write_to_memory_object(Elt, m, verbose_level);
			m->write_int(E[i].get_data1());
			m->write_int(E[i].get_data2());
			nb_group_elements++;
		}
		else {
			cout << "poset_orbit_node::write_memory: type "
						<< E[i].get_type() << " is illegal" << endl;
			exit(1);
		}
	}
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_orbit_node::write_memory_object node "
				<< node << " finished" << endl;
	}
}

long int poset_orbit_node::calc_size_on_file(action *A, int verbose_level)
{
	int i;
	long int s = 0;

	s += 2 * sizeof(int); // node, prev
	s += sizeof(long int); // pt
	s += sizeof(int); // nb_strong_generators
	//m->write_int(node);
	//m->write_int(prev);
	//m->write_int(pt);
	//m->write_int(nb_strong_generators);

	s += nb_strong_generators * A->coded_elt_size_in_char;
	if (nb_strong_generators) {
		s += A->base_len() * sizeof(int);
		// tl[]
		}

	s += sizeof(int); // nb_extensions
	//m->write_int(nb_extensions);

	for (i = 0; i < nb_extensions; i++) {
		s += sizeof(long int); // pt
		s += 2 * sizeof(int); // orbit_len, type
		//m->write_int(E[i].pt);
		//m->write_int(E[i].orbit_len);
		//m->write_int(E[i].type);

		if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			// extension node
			s += sizeof(int); // data
			//m->write_int(E[i].data); // next poset_orbit_node
			}
		else if (E[i].get_type() == EXTENSION_TYPE_FUSION) {
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
		ifstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node << endl;
		}
	int hdl;

	if (nb_strong_generators) {
		//hdl = hdl_strong_generators[0];
		hdl = first_strong_generator_handle;
	}
	else {
		hdl = -1;
	}
	Schreier_vector = PC->get_schreier_vector_handler()->sv_read_file(
			hdl, nb_strong_generators,
			fp, verbose_level);
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node
				<< " finished" << endl;
		}
}

void poset_orbit_node::sv_write_file(poset_classification *PC,
		ofstream &fp, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_write_file node " << node << endl;
		}

	PC->get_schreier_vector_handler()->sv_write_file(Schreier_vector,
			fp, verbose_level);
	
	if (f_v) {
		cout << "poset_orbit_node::sv_write_file node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::read_file(action *A,
		ifstream &fp, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int *Elt;
	file_io Fio;

	Elt = NEW_int(A->elt_size_in_int);
	//node = Fio.fread_int4(fp);
	fp.read((char *) &node, sizeof(int));
	if (f_v) {
		cout << "poset_orbit_node::read_file node " << node << endl;
	}
	fp.read((char *) &prev, sizeof(int));
	fp.read((char *) &pt, sizeof(long int));
	fp.read((char *) &nb_strong_generators, sizeof(int));
	//prev = Fio.fread_int4(fp);
	//pt = Fio.fread_int4(fp);
	//nb_strong_generators = Fio.fread_int4(fp);
	if (f_vv) {
		cout << "prev=" << prev << endl;
		cout << "pt=" << pt << endl;
		cout << "nb_strong_generators " << nb_strong_generators << endl;
	}
	if (nb_strong_generators) {
#if 0
		hdl_strong_generators = NEW_int(nb_strong_generators);
		tl = NEW_int(A->base_len());
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_file_fp(Elt, fp, verbose_level);
			if (f_vv) {
				cout << "read element" << endl;
				A->element_print_quick(Elt, cout);
				}
			hdl_strong_generators[i] = A->element_store(Elt, FALSE);
			nb_group_elements++;
		}
#else
		tl = NEW_int(A->base_len());
		first_strong_generator_handle = -1;
		for (i = 0; i < nb_strong_generators; i++) {
			A->element_read_file_fp(Elt, fp, verbose_level);
			if (f_vv) {
				cout << "read element" << endl;
				A->element_print_quick(Elt, cout);
				}
			if (i == 0) {
				first_strong_generator_handle = A->element_store(Elt, FALSE);
			}
			else {
				A->element_store(Elt, FALSE);
			}
			nb_group_elements++;
		}
#endif
		for (i = 0; i < A->base_len(); i++) {
			fp.read((char *) &tl[i], sizeof(int));
			//tl[i] = Fio.fread_int4(fp);
			if (f_vv) {
				cout << "read tl[" << i << "]=" << tl[i] << endl;
			}
		}
	}
	else {
		//hdl_strong_generators = NULL;
		first_strong_generator_handle = -1;
		tl = NULL;
	}
	fp.read((char *) &nb_extensions, sizeof(int));
	//nb_extensions = Fio.fread_int4(fp);
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
		long int a;
		int b;

		fp.read((char *) &a, sizeof(long int));
		E[i].set_pt(a);
		//E[i].pt = Fio.fread_int4(fp);
		if (f_vv) {
			cout << "pt = " << E[i].get_pt() << endl;
		}
		fp.read((char *) &b, sizeof(int));
		E[i].set_orbit_len(b);
		//E[i].orbit_len = Fio.fread_int4(fp);
		if (f_vv) {
			cout << "orbit_len = " << E[i].get_orbit_len() << endl;
		}
		fp.read((char *) &b, sizeof(int));
		E[i].set_type(b);
		//E[i].type = Fio.fread_int4(fp);
		if (f_vv) {
			cout << "type = " << E[i].get_type() << endl;
		}
		if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			// extension node
			fp.read((char *) &b, sizeof(int));
			E[i].set_data(b);
			//E[i].data = Fio.fread_int4(fp);
			// next poset_orbit_node
		}
		else if (E[i].get_type() == EXTENSION_TYPE_FUSION) {
			// fusion node
			A->element_read_file_fp(Elt, fp, verbose_level);
			if (f_vv) {
				cout << "read element" << endl;
				A->element_print_quick(Elt, cout);
			}
			//element_read_file(A, Elt, elt, fp, verbose_level);
			E[i].set_data(A->element_store(Elt, FALSE));
			fp.read((char *) &b, sizeof(int));
			E[i].set_data1(b);
			fp.read((char *) &b, sizeof(int));
			E[i].set_data2(b);
			nb_group_elements++;
		}
		else if (E[i].get_type() == EXTENSION_TYPE_PROCESSING) {
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
		ofstream &fp, int &nb_group_elements,
		int verbose_level)
{
	int i;
	int *Elt;
	int f_v = FALSE;//(verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	file_io Fio;

	Elt = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "poset_orbit_node::write_file node " << node << endl;
	}
	fp.write((char *) &node, sizeof(int));
	fp.write((char *) &prev, sizeof(int));
	fp.write((char *) &pt, sizeof(long int));
	fp.write((char *) &nb_strong_generators, sizeof(int));
	//Fio.fwrite_int4(fp, node);
	//Fio.fwrite_int4(fp, prev);
	//Fio.fwrite_int4(fp, pt);
	//Fio.fwrite_int4(fp, nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " "
				<< nb_strong_generators << endl;
	}
	for (i = 0; i < nb_strong_generators; i++) {
		A->element_retrieve(first_strong_generator_handle + i, Elt, FALSE);
		A->element_write_file_fp(Elt, fp, 0);
		nb_group_elements++;
	}
	if (nb_strong_generators) {
		if (f_vv) {
			cout << "writing tl" << endl;
		}
		for (i = 0; i < A->base_len(); i++) {
			fp.write((char *) &tl[i], sizeof(int));
			//Fio.fwrite_int4(fp, tl[i]);
			if (f_vv) {
				cout << tl[i] << " ";
			}
		}
		if (f_vv) {
			cout << endl;
		}
	}
	fp.write((char *) &nb_extensions, sizeof(int));
	//Fio.fwrite_int4(fp, nb_extensions);
	if (f_vv) {
		cout << "nb_extensions=" << nb_extensions << endl;
	}
	for (i = 0; i < nb_extensions; i++) {
		long int a;
		int b;

		a = E[i].get_pt();
		fp.write((char *) &a, sizeof(long int));
		b = E[i].get_orbit_len();
		fp.write((char *) &b, sizeof(int));
		b = E[i].get_type();
		fp.write((char *) &b, sizeof(int));
		//Fio.fwrite_int4(fp, E[i].pt);
		//Fio.fwrite_int4(fp, E[i].orbit_len);
		//Fio.fwrite_int4(fp, E[i].type);
		if (f_vv) {
			cout << i << " : " << E[i].get_pt() << " : "
					<< E[i].get_orbit_len() << " : " << E[i].get_type() << endl;
		}
		if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			// extension node
			b = E[i].get_data();
			fp.write((char *) &b, sizeof(int));
			//Fio.fwrite_int4(fp, E[i].data);
			if (f_vv) {
				cout << "extension node, data=" << E[i].get_data() << endl;
			}
		}
		else if (E[i].get_type() == EXTENSION_TYPE_FUSION) {
			// fusion node
			if (f_vv) {
				cout << "fusion node, hdl=" << E[i].get_data() << endl;
			}
			A->element_retrieve(E[i].get_data(), Elt, FALSE);
			A->element_write_file_fp(Elt, fp, 0);
			b = E[i].get_data1();
			fp.write((char *) &b, sizeof(int));
			b = E[i].get_data2();
			fp.write((char *) &b, sizeof(int));
			nb_group_elements++;
		}
		else if (E[i].get_type() == EXTENSION_TYPE_PROCESSING) {
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
	if (PC->get_control()->f_export_schreier_trees) {
		int orbit_no, nb_orbits;

		nb_orbits = Schreier->nb_orbits;
		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			string fname_mask_base;
			string fname_mask;

			PC->create_schreier_tree_fname_mask_base(
					fname_mask_base, node);

			fname_mask.assign(fname_mask_base);
			fname_mask.append(".layered_graph");

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
	if (PC->get_control()->f_export_schreier_trees) {
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
			string fname_mask_base;
			string fname_mask;

			PC->create_shallow_schreier_tree_fname_mask_base(
					fname_mask_base, node);


			fname_mask.assign(fname_mask_base);
			fname_mask.append(".layered_graph");

			Schreier_vector->export_tree_as_layered_graph(
					orbit_no, orbit_reps[orbit_no],
					fname_mask.c_str(),
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
	if (PC->get_control()->f_draw_schreier_trees) {
		int orbit_no, nb_orbits;

		nb_orbits = Schreier->nb_orbits;
		for (orbit_no = 0; orbit_no < nb_orbits; orbit_no++) {
			char label[2000];
			int xmax = PC->get_control()->schreier_tree_xmax;
			int ymax =  PC->get_control()->schreier_tree_ymax;
			int f_circletext = PC->get_control()->schreier_tree_f_circletext;
			int rad = PC->get_control()->schreier_tree_rad;
			int f_embedded = PC->get_control()->schreier_tree_f_embedded;
			int f_sideways = PC->get_control()->schreier_tree_f_sideways;
			double scale = PC->get_control()->schreier_tree_scale;
			double line_width = PC->get_control()->schreier_tree_line_width;
			int f_has_point_labels = FALSE;
			long int *point_labels = NULL;

			snprintf(label, 2000, "%sschreier_tree_node_%d_%d",
					PC->get_control()->schreier_tree_prefix, node, orbit_no);

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

		char label_data[2000];
		snprintf(label_data, 2000, "%sschreier_data_node_%d.tex",
				PC->get_control()->schreier_tree_prefix, node);
		Schreier->latex(label_data);
		}

	if (f_v) {
		cout << "poset_orbit_node::draw_schreier_forest"
				" done" << endl;
	}
}

}}




