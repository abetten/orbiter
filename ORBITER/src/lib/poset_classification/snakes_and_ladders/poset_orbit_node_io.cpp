// poset_orbit_node_io.C
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.C
// December 27, 2008
// renamed from io.C into oracle_io.C Aug 24, 2011


#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

void poset_orbit_node::read_memory_object(
		action *A, memory_object *m, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt;
	
	Elt = NEW_int(A->elt_size_in_int);
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
			A->element_read_from_memory_object(Elt, m, verbose_level);
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
			A->element_read_from_memory_object(Elt, m, verbose_level);
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
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_orbit_node::read_memory_object node "
				<< node << " finished" << endl;
		}
}

void poset_orbit_node::write_memory_object(
		action *A, memory_object *m, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Elt;
	
	Elt = NEW_int(A->elt_size_in_int);
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
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_orbit_node::write_memory_object node "
				<< node << " finished" << endl;
		}
}


void poset_orbit_node::sv_read_file(poset_classification *PC,
		FILE *fp, int verbose_level)
{
	//int i, n, len;
	//int4 I;
	int f_v = (verbose_level >= 1);
	//int f_trivial_group;
	
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node << endl;
		}
#if 0
	I = fread_int4(fp);
	if (I == 0) {
		sv = NULL;
		cout << "poset_orbit_node::sv_read_file node " << node
				<< ", sv = NULL, no schreier vector" << endl;
		return;
		}
	f_trivial_group = fread_int4(fp);
	n = fread_int4(fp);
	
	
	int *osv;
	if (f_trivial_group) {
		osv = NEW_int(n + 1);
		len = n;
		}
	else {
		osv = NEW_int(3 * n + 1);
		len = 3 * n;
		}
	osv[0] = n;
	for (i = 0; i < len; i++) {
		osv[1 + i] = fread_int4(fp);
		}
	sv = osv;
	cout << "poset_orbit_node::sv_read_file node " << node
			<< " read sv with " << n << " live points" << endl;
#else
	int hdl;

	if (nb_strong_generators) {
		hdl = hdl_strong_generators[0];
	} else {
		hdl = -1;
	}
	Schreier_vector = PC->Schreier_vector_handler->sv_read_file(
			hdl, nb_strong_generators,
			fp, verbose_level);
#endif
	if (f_v) {
		cout << "poset_orbit_node::sv_read_file node " << node
				<< " finished" << endl;
		}
}

void poset_orbit_node::sv_write_file(poset_classification *PC,
		FILE *fp, int verbose_level)
{
	//int i, len;
	int f_v = (verbose_level >= 1);
	//int f_trivial_group;
	
	if (f_v) {
		cout << "poset_orbit_node::sv_write_file node " << node << endl;
		}

#if 0
	if (sv == NULL) {
		fwrite_int4(fp, 0);
		}
	else {
		fwrite_int4(fp, 1);
		if (nb_strong_generators == 0) {
			f_trivial_group = TRUE;
			}
		else {
			f_trivial_group = FALSE;
			}
		fwrite_int4(fp, f_trivial_group);
		int *osv = sv;
		int n = osv[0];
		fwrite_int4(fp, n);
		if (f_trivial_group) {
			len = n;
			}
		else {
			len = 3 * n;
			}
		for (i = 0; i < len; i++) {
			fwrite_int4(fp, osv[1 + i]);
			}
		}
#else
	PC->Schreier_vector_handler->sv_write_file(Schreier_vector,
			fp, verbose_level);
#endif
	
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

int poset_orbit_node::calc_size_on_file(action *A, int verbose_level)
{
	int i, s = 0;
	s += 4 * 4; // node, prev, pt, nb_strong_generators
	s += nb_strong_generators * A->coded_elt_size_in_char;
	if (nb_strong_generators) {
		s += A->base_len * 4; 
		// tl[]
		}
	s += 4; // nb_extensions
	for (i = 0; i < nb_extensions; i++) {
		s += 3 * 4; // pt, orbit_len, type
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			s += 4; // data
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			// fusion node
			s += A->coded_elt_size_in_char; // group element
			}
		}
	return s;
}




