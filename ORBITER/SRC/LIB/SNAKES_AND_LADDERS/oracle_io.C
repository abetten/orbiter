// oracle_io.C
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.C
// December 27, 2008
// renamed from io.C into oracle_io.C Aug 24, 2011


#include "orbiter.h"

void oracle::read_memory_object(action *A, memory_object *m, INT &nb_group_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	INT *Elt;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	m->read_int(&node);
	if (f_v) {
		cout << "oracle::read_memory_object node " << node << endl;
		cout << "cur_pointer=" << m->cur_pointer << endl;
		}
	m->read_int(&prev);
	m->read_int(&pt);
	m->read_int(&nb_strong_generators);
	if (f_v) {
		cout << "oracle::read_memory_object nb_strong_generators " << nb_strong_generators << endl;
		}
	if (nb_strong_generators) {
		hdl_strong_generators = NEW_INT(nb_strong_generators);
		tl = NEW_INT(A->base_len);
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
		cout << "oracle::read_memory_object nb_extensions " << nb_extensions << endl;
		cout << "cur_pointer=" << m->cur_pointer << endl;
		}
	E = new extension[nb_extensions];
	if (f_v) {
		cout << "E allocated" << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		if (f_v) {
			cout << "oracle::read_memory_object extension " << i << endl;
			}
		m->read_int(&E[i].pt);
		if (f_v) {
			cout << "oracle::read_memory_object pt = " << E[i].pt << endl;
			}
		m->read_int(&E[i].orbit_len);
		if (f_v) {
			cout << "oracle::read_memory_object pt = " << E[i].orbit_len << endl;
			}
		m->read_int(&E[i].type);
		if (f_v) {
			cout << "oracle::read_memory_object type = " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->read_int(&E[i].data); // next oracle node
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
			cout << "oracle::read_memory_object type " << E[i].type << " is illegal" << endl;
			exit(1);
			}
		}
	FREE_INT(Elt);
	if (f_v) {
		cout << "oracle::read_memory_object node " << node << " finished" << endl;
		}
}

void oracle::write_memory_object(action *A, memory_object *m, INT &nb_group_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	INT *Elt;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	if (f_v) {
		cout << "oracle::write_memory_object node " << node << endl;
		cout << "used_length=" << m->used_length << endl;
		}
	m->write_int(node);
	m->write_int(prev);
	m->write_int(pt);
	m->write_int(nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " " << nb_strong_generators << endl;
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
			cout << i << " : " << E[i].pt << " : " << E[i].orbit_len << " : " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			m->write_int(E[i].data); // next oracle node
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
			cout << "oracle_write_memory: type " << E[i].type << " is illegal" << endl;
			exit(1);
			}
		}
	FREE_INT(Elt);
	if (f_v) {
		cout << "oracle::write_memory_object node " << node << " finished" << endl;
		}
}


void oracle::sv_read_file(FILE *fp, INT verbose_level)
{
	INT i, n, len;
	INT4 I;
	INT f_v = (verbose_level >= 1);
	INT f_trivial_group;
	
	if (f_v) {
		cout << "oracle::sv_read_file node " << node << endl;
		}
	I = fread_INT4(fp);
	if (I == 0) {
		sv = NULL;
		cout << "oracle::sv_read_file node " << node << ", sv = NULL, no schreier vector" << endl;
		return;
		}
	f_trivial_group = fread_INT4(fp);
	n = fread_INT4(fp);
	
	
	INT *osv;
	if (f_trivial_group) {
		osv = NEW_INT(n + 1);
		len = n;
		}
	else {
		osv = NEW_INT(3 * n + 1);
		len = 3 * n;
		}
	osv[0] = n;
	for (i = 0; i < len; i++) {
		osv[1 + i] = fread_INT4(fp);
		}
	sv = osv;
	cout << "oracle::sv_read_file node " << node << " read sv with " << n << " live points" << endl;
	
	if (f_v) {
		cout << "oracle::sv_read_file node " << node << " finished" << endl;
		}
}

void oracle::sv_write_file(FILE *fp, INT verbose_level)
{
	INT i, len;
	INT f_v = (verbose_level >= 1);
	INT f_trivial_group;
	
	if (f_v) {
		cout << "oracle::sv_write_file node " << node << endl;
		}
	if (sv == NULL) {
		fwrite_INT4(fp, 0);
		}
	else {
		fwrite_INT4(fp, 1);
		if (nb_strong_generators == 0) {
			f_trivial_group = TRUE;
			}
		else {
			f_trivial_group = FALSE;
			}
		fwrite_INT4(fp, f_trivial_group);
		INT *osv = sv;
		INT n = osv[0];
		fwrite_INT4(fp, n);
		if (f_trivial_group) {
			len = n;
			}
		else {
			len = 3 * n;
			}
		for (i = 0; i < len; i++) {
			fwrite_INT4(fp, osv[1 + i]);
			}
		}
	
	if (f_v) {
		cout << "oracle::sv_write_file node " << node << " finished" << endl;
		}
}

void oracle::read_file(action *A, FILE *fp, INT &nb_group_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	INT *Elt;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	node = fread_INT4(fp);
	if (f_v) {
		cout << "oracle_read_file node " << node << endl;
		}
	prev = fread_INT4(fp);
	pt = fread_INT4(fp);
	nb_strong_generators = fread_INT4(fp);
	if (f_vv) {
		cout << "prev=" << prev << endl;
		cout << "pt=" << pt << endl;
		cout << "nb_strong_generators " << nb_strong_generators << endl;
		}
	if (nb_strong_generators) {
		hdl_strong_generators = NEW_INT(nb_strong_generators);
		tl = NEW_INT(A->base_len);
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
			tl[i] = fread_INT4(fp);
			if (f_vv) {
				cout << "read tl[" << i << "]=" << tl[i] << endl;
				}
			}
		}
	else {
		hdl_strong_generators = NULL;
		tl = NULL;
		}
	nb_extensions = fread_INT4(fp);
	if (f_vv) {
		cout << "nb_extensions " << nb_extensions << endl;
		}
	E = new extension[nb_extensions];
	if (f_vv) {
		cout << "E allocated" << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		if (f_vv) {
			cout << "oracle_read_file extension " << i << endl;
			}
		E[i].pt = fread_INT4(fp);
		if (f_vv) {
			cout << "pt = " << E[i].pt << endl;
			}
		E[i].orbit_len = fread_INT4(fp);
		if (f_vv) {
			cout << "orbit_len = " << E[i].orbit_len << endl;
			}
		E[i].type = fread_INT4(fp);
		if (f_vv) {
			cout << "type = " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			E[i].data = fread_INT4(fp);
			// next oracle node
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
			cout << "oracle_read_file: type EXTENSION_TYPE_PROCESSING is illegal" << endl;
			exit(1);
			}
		}
	FREE_INT(Elt);
	if (f_v) {
		cout << "oracle_read_file node " << node << " finished" << endl;
		}
}

void oracle::write_file(action *A, FILE *fp, INT &nb_group_elements, INT verbose_level)
{
	INT i;
	INT *Elt;
	INT f_v = FALSE;//(verbose_level >= 1);
	INT f_vv = FALSE;//(verbose_level >= 2);
	
	Elt = NEW_INT(A->elt_size_in_INT);
	if (f_v) {
		cout << "oracle_write_file node " << node << endl;
		}
	fwrite_INT4(fp, node);
	fwrite_INT4(fp, prev);
	fwrite_INT4(fp, pt);
	fwrite_INT4(fp, nb_strong_generators);
	if (f_v) {
		cout << node << " " << prev << " " << pt << " " << nb_strong_generators << endl;
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
			fwrite_INT4(fp, tl[i]);
			if (f_vv) {
				cout << tl[i] << " ";
				}
			}
		if (f_vv) {
			cout << endl;
			}
		}
	fwrite_INT4(fp, nb_extensions);
	if (f_vv) {
		cout << "nb_extensions=" << nb_extensions << endl;
		}
	for (i = 0; i < nb_extensions; i++) {
		fwrite_INT4(fp, E[i].pt);
		fwrite_INT4(fp, E[i].orbit_len);
		fwrite_INT4(fp, E[i].type);
		if (f_vv) {
			cout << i << " : " << E[i].pt << " : " << E[i].orbit_len << " : " << E[i].type << endl;
			}
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			// extension node
			fwrite_INT4(fp, E[i].data);
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
			cout << "oracle_write_file: type EXTENSION_TYPE_PROCESSING is illegal" << endl;
			exit(1);
			}
		}
	FREE_INT(Elt);
	if (f_v) {
		cout << "oracle_write_file node " << node << " finished" << endl;
		}
}

INT oracle::calc_size_on_file(action *A, INT verbose_level)
{
	INT i, s = 0;
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




