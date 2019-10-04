// poset_orbit_node.cpp
//
// Anton Betten
// December 27, 2004

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace classification {

poset_orbit_node::poset_orbit_node()
{
	null();
}

poset_orbit_node::~poset_orbit_node()
{
	freeself();
}

void poset_orbit_node::null()
{
	nb_strong_generators = 0;
	hdl_strong_generators = NULL;
	tl = NULL;
	nb_extensions = 0;
	E = NULL;
	//sv = NULL;
	Schreier_vector = NULL;
	A_on_upset = NULL;
}

void poset_orbit_node::freeself()
{
	if (hdl_strong_generators) {
#if 0
		cout << "poset_orbit_node::freeself "
				"deleting hdl_strong_generators: ";
		int_vec_print(cout, hdl_strong_generators, nb_strong_generators);
		cout << endl;
		cout << "pointer = ";
		print_pointer_hex(cout, hdl_strong_generators);
		cout << endl;
#endif
		FREE_int(hdl_strong_generators);
		//cout << "poset_orbit_node::freeself() "
		//"deleting hdl_strong_generators done" << endl;
		}
	if (tl) {
		//cout << "poset_orbit_node::freeself deleting tl" << endl;
		FREE_int(tl);
		}
	if (E) {
		//cout << "poset_orbit_node::freeself deleting E" << endl;
		FREE_OBJECTS(E);
		}
#if 0
	if (sv) {
		//cout << "poset_orbit_node::freeself deleting sv" << endl;
		FREE_int(sv);
		}
#endif
	if (Schreier_vector) {
		FREE_OBJECT(Schreier_vector);
	}
	if (A_on_upset) {
		FREE_OBJECT(A_on_upset);
	}
	null();
	//cout << "poset_orbit_node::freeself finished" << endl;
}


void poset_orbit_node::init_root_node(
	poset_classification *gen, int verbose_level)
// copies gen->SG0 and gen->transversal_length
// into the poset_orbit_node structure using store_strong_generators
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::init_root_node "
				"initializing root node" << endl;
		}
	
	freeself();
	
	node = 0;
	prev = -1;
	//sv = NULL;
	Schreier_vector = NULL;
	
	longinteger_object go;

	gen->Poset->Strong_gens->group_order(go);

	if (f_v) {
		cout << "poset_orbit_node::init_root_node "
				"storing strong generators "
				"for a group of order " << go << endl;
		}
	store_strong_generators(gen, gen->Poset->Strong_gens);
		// stores the strong generators into
		// the poset_orbit_node structure,
		// copies transversal_length into tl
	if (f_v) {
		cout << "poset_orbit_node::init_root_node done" << endl;
		}
	
}

int poset_orbit_node::get_level(poset_classification *gen)
{
	int l;

	l = depth_of_node(gen);
	return l;
}

int poset_orbit_node::get_node_in_level(poset_classification *gen)
{
	int l, n;

	l = depth_of_node(gen);
	n = node - gen->first_poset_orbit_node_at_level[l];
	return n;
}

int *poset_orbit_node::live_points()
{
	if (Schreier_vector == NULL) {
		cout << "poset_orbit_node::live_points "
				"Schreier_vector == NULL" << endl;
		exit(1);
	} else {
		return Schreier_vector->points();
	}
}

int poset_orbit_node::get_nb_of_live_points()
{
	if (Schreier_vector == NULL) {
		//cout << "poset_orbit_node::get_nb_of_live_points "
		//		"Schreier_vector == NULL" << endl;
		return 0;
	} else {
		return Schreier_vector->get_number_of_points();
	}
}

int poset_orbit_node::get_nb_of_orbits_under_stabilizer()
{
	if (Schreier_vector == NULL) {
		//cout << "poset_orbit_node::get_nb_of_orbits_under_stabilizer "
		//		"Schreier_vector == NULL" << endl;
		return 0;
	} else {
		return Schreier_vector->get_number_of_orbits();
	}
}

void poset_orbit_node::poset_orbit_node_depth_breadth_perm_and_inverse(
	poset_classification *gen, int max_depth,
	int &idx, int hdl, int cur_depth,
	int *perm, int *perm_inv)
{
	int i, nxt;
	
	perm[idx] = hdl;
	perm_inv[hdl] = idx;
	idx++;
	if (cur_depth == max_depth)
		return;
	for (i = 0; i < nb_extensions; i++) {
		if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			nxt = E[i].data;
			if (nxt >= 0) {
				gen->root[nxt].
					poset_orbit_node_depth_breadth_perm_and_inverse(gen,
					max_depth, idx, nxt, cur_depth + 1, perm, perm_inv);
				}
			}
		}
}

int poset_orbit_node::find_extension_from_point(
		poset_classification *gen,
		int pt, int verbose_level)
// a -1 means not found
{
	int i;
	
	for (i = 0; i < nb_extensions; i++) {
		if (E[i].pt == pt)
			break;
		}
	if (i == nb_extensions) {
		return -1;
		}
	return i;
}

void poset_orbit_node::print_extensions(ostream &ost)
{
	int i;
	
	ost << "Node " << node << ", the extensions are" << endl;
	if (nb_extensions >= 10) {
		ost << "too many to print" << endl;
		return;
		}
	ost << "i : pt : orbit_len : type : to where" << endl;
	for (i = 0; i < nb_extensions; i++) {
		ost << setw(5) << i << " : " 
			<< setw(7) << E[i].pt << " : " 
			<< setw(5) << E[i].orbit_len << " : ";

		print_extension_type(ost, E[i].type);
		if (E[i].type == EXTENSION_TYPE_FUSION) {
			ost << " -> (" << E[i].data1 << ","
					<< E[i].data2 << ") hdl=" << E[i].data << endl;
			}
		else if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			ost << " -> " << E[i].data << endl;
			}
		else {
			ost << setw(5) << E[i].data << endl;
			}
		if (E[i].type >= NB_EXTENSION_TYPES) {
			ost << "E[i].type >= NB_EXTENSION_TYPES" << endl;
			exit(1);
			}
		}
	cout << "done with node " << node << endl;
}

void poset_orbit_node::log_current_node_without_group(
		poset_classification *gen,
		int s, ostream &f, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::log_current_node_without_group" << endl;
		}
	store_set_to(gen, s - 1, gen->set0);
	
	if (f_v) {
		f << "# ***** orbit ***** " <<
				node - gen->first_poset_orbit_node_at_level[s] << " "<< endl;
		}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << gen->set0[i] << " ";
		}
	f << endl;

#if 0
	if (f_v) {
		f << "# BEGINCOMMENT" << endl;
		if (gen->f_print_function) {
			(*gen->print_function)(f, s,
					gen->set0, gen->print_function_data);
			}
		
		f << "# ENDCOMMENT" << endl;
		}
#endif
}

void poset_orbit_node::log_current_node(poset_classification *gen,
		int s, ostream &f, int f_with_stabilizer_generators,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go, rk;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::log_current_node node="
				<< node << " s=" << s << endl;
		}
	store_set_to(gen, s - 1, gen->set0);
	if (f_v) {
		cout << "poset_orbit_node::log_current_node node="
				<< node << " after store_set_to" << endl;
		}
	
	if (f_v) {
		f << "# ***** orbit ***** "
				<< node - gen->first_poset_orbit_node_at_level[s]
				<< " " << endl;
		}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << gen->set0[i] << " ";
		}
		
	if (nb_strong_generators == 0) {
		f << " 1" << endl;
		if (f_v) {
			cout << "poset_orbit_node::log_current_node "
					"node=" << node << " done" << endl;
			}
		return;
		}


	if (f_v) {
		cout << "poset_orbit_node::log_current_node "
				"node=" << node << " creating group" << endl;
		}

	group G;

	G.init(gen->Poset->A, verbose_level - 2);
	G.init_strong_generators_by_hdl(
			nb_strong_generators, hdl_strong_generators, tl, FALSE);

	if (f_v) {
		cout << "poset_orbit_node::log_current_node "
				"node=" << node << " before schreier_sims" << endl;
		}
	G.schreier_sims(0);
	if (f_v) {
		cout << "poset_orbit_node::log_current_node "
				"node=" << node << " after schreier_sims" << endl;
		}
	G.group_order(go);
	if (f_v) {
		cout << "poset_orbit_node::log_current_node "
				"node=" << node << " group order = " << go << endl;
		}
	//if (f_v) {
		//cout << "poset_orbit_node::log_current_node() "
		//"stabilizer of order " << go << " reconstructed" << endl;
		//}
	if (go.is_one()) {
		go.print_not_scientific(f);
		f << endl;
		//f << go << endl;
		}
	else {
		G.code_ascii(FALSE);
		go.print_not_scientific(f);
		f << " " << G.ascii_coding << endl;
		//f << go << " " << G.ascii_coding << endl;
		}


	if (f_with_stabilizer_generators) {
		strong_generators *Strong_gens;
		longinteger_object go1;

		get_stabilizer_generators(gen, Strong_gens, verbose_level);
		Strong_gens->group_order(go1);
		cout << "The stabilizer is a group of order " << go1 << endl;
		cout << "With the following generators:" << endl;
		Strong_gens->print_generators_ost(cout);
		delete Strong_gens;
		}

#if 1
	if (gen->f_print_function) {
		f << "# BEGINCOMMENT" << endl;
		if (gen->f_print_function) {
			(*gen->print_function)(f, s, gen->set0,
					gen->print_function_data);
			}
		
		if (!go.is_one()) {
			if (f_v) {
				cout << "poset_orbit_node::log_current_node "
						"node=" << node << " printing generators" << endl;
				}
			G.require_strong_generators();
			f << "tl: ";
			for (i = 0; i < G.A->base_len(); i++)
				f << G.tl[i] << " ";
			f << endl;
			f << G.SG->len << " strong generators by rank: " << endl;
			for (i = 0; i < G.SG->len; i++) {
				f << i << " : " << endl;
				
				G.A->element_print(G.SG->ith(i), f);
				f << endl;
				//G.A->element_print_as_permutation(G.SG->ith(i), f);
				//f << endl;

#if 0
				G.A->element_rank(rk, G.SG->ith(i), 0);
				f << "\"" << rk << "\", ";
				f << endl;
#endif
				}
			//for (i = 0; i < G.SG->len; i++) {
				//}
			}
		f << "# ENDCOMMENT" << endl;
		}
#endif
}

void poset_orbit_node::log_current_node_after_applying_group_element(
		poset_classification *gen, int s, ostream &f, int hdl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go;
	int i;
	int *S;
	int *Elt;
	int *Elt_inv;
	int *Elt1;
	int *Elt2;
	
	S = NEW_int(s);
	Elt = NEW_int(gen->Poset->A->elt_size_in_int);
	Elt_inv = NEW_int(gen->Poset->A->elt_size_in_int);
	Elt1 = NEW_int(gen->Poset->A->elt_size_in_int);
	Elt2 = NEW_int(gen->Poset->A->elt_size_in_int);
	
	store_set_to(gen, s - 1, gen->set0);
	gen->Poset->A->element_retrieve(hdl, Elt, 0);
	//gen->A->element_print(Elt, cout);
	gen->Poset->A->element_invert(Elt, Elt_inv, 0);
	for (i = 0; i < s; i++) {
		S[i] = Elt[gen->set0[i]];
		}
	
	if (f_v) {
		f << "# ***** orbit ***** "
				<< node - gen->first_poset_orbit_node_at_level[s]
				<< " " << endl;
		}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << S[i] << " ";
		}
	group G;

	G.init(gen->Poset->A, verbose_level - 2);
	G.init_strong_generators_by_hdl(
			nb_strong_generators,
			hdl_strong_generators, tl,
			FALSE);
	G.schreier_sims(0);
	G.group_order(go);
	//if (f_v) {
		//cout << "poset_orbit_node::log_current_node() "
		//"stabilizer of order " << go << " reconstructed" << endl;
		//}
	if (go.is_one()) {
		f << go << endl;
		}
	else {
		G.code_ascii(FALSE);
		f << go << " " << G.ascii_coding << endl;
		}

	if (f_v) {
#if 0
		if (gen->f_print_function) {
			(*gen->print_function)(f, s, S,
					gen->print_function_data);
			}
#endif
		if (!go.is_one()) {
			G.require_strong_generators();
			f << "# ";
			for (i = 0; i < G.A->base_len(); i++)
				f << G.tl[i] << " ";
			f << endl;
			for (i = 0; i < G.SG->len; i++) {
				f << "# ";
				//G.A->element_print(G.SG->ith(i), f);
				G.A->element_mult(Elt_inv, G.SG->ith(i), Elt1, FALSE);
				G.A->element_mult(Elt1, Elt, Elt2, FALSE);
				G.A->element_print(Elt2, f);
				//f << endl;
				}
			}
		}
	FREE_int(S);
	FREE_int(Elt);
	FREE_int(Elt_inv);
	FREE_int(Elt1);
	FREE_int(Elt2);
}

void poset_orbit_node::log_current_node_with_candidates(
		poset_classification *gen, int lvl, ostream &f,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int i;

	store_set_to(gen, lvl - 1, gen->set0);
	
	f << lvl << " ";
	for (i = 0; i < lvl; i++) {
		f << gen->set0[i] << " ";
		}
	f << -1 << " ";
	
	int n;
	int *subset;
	int *candidates = NULL;
	int nb_candidates = 0;
	int f_subset_is_allocated;

	if (!downstep_get_invariant_subset(
		gen, 
		lvl, 
		n, subset, f_subset_is_allocated, 
		verbose_level)) {
		cout << "poset_orbit_node::log_current_node_with_candidates "
				"downstep_get_invariant_subset returns FALSE" << endl;
		exit(1);
		}
	candidates = NEW_int(n);
		
	downstep_apply_early_test(gen, lvl, 
		n, subset, 
		candidates, nb_candidates, 
		verbose_level - 2);
	f << nb_candidates << " ";
	for (i = 0; i < nb_candidates; i++) {
		f << candidates[i] << " ";
		}
	f << -1 << endl;
	if (f_subset_is_allocated) {
		FREE_int(subset);
		}
	FREE_int(candidates);
}


int poset_orbit_node::depth_of_node(poset_classification *gen)
{
	if (prev == -1) {
		return 0;
		}
	else {
		return gen->root[prev].depth_of_node(gen) + 1;
		}
}

void poset_orbit_node::store_set(poset_classification *gen, int i)
// stores a set of size i + 1 to gen->S[]
{
	if (i < 0)
		return;
	gen->S[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set prev == -1" << endl;
			exit(1);
			}
		gen->root[prev].store_set(gen, i - 1);
		}
}

void poset_orbit_node::store_set_with_verbose_level(
		poset_classification *gen, int i, int verbose_level)
// stores a set of size i + 1 to gen->S[]
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::store_set_with_verbose_level "
				"node=" << node << " prev=" << prev
				<< " pt=" << pt << " i=" << i << endl;
		}
	if (i < 0)
		return;
	gen->S[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set prev == -1" << endl;
			exit(1);
			}
		gen->root[prev].store_set(gen, i - 1);
		}
}

void poset_orbit_node::store_set_to(
		poset_classification *gen, int i, int *to)
// stores a set of size i + 1 to 'to'
{
	if (i < 0)
		return;
	to[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set_to prev == -1" << endl;
			exit(1);
			}
		gen->root[prev].store_set_to(gen, i - 1, to);
		}
}

void poset_orbit_node::store_set_to(
		poset_classification *gen, int *to)
{
	store_set_to(gen, depth_of_node(gen), to);
}

int poset_orbit_node::check_node_and_set_consistency(
		poset_classification *gen, int i, int *set)
{
	if (i < 0)
		return TRUE;
	if (set[i] != pt) {
		cout << "check_node_and_set_consistency inconsistent" << endl;
		return FALSE;
		}
	if (i >= 0) {
		if (prev == -1) {
			cout << "check_node_and_set_consistency prev == -1" << endl;
			exit(1);
			}
		gen->root[prev].check_node_and_set_consistency(
				gen, i - 1, set);
		}
	return TRUE;
}

void poset_orbit_node::print_set_verbose(poset_classification *gen)
{
	int depth;
	int *set;

	//cout << "poset_orbit_node::print_set_verbose" << endl;
	depth = depth_of_node(gen);
	print_set(gen);
	cout << endl;


	set = NEW_int(depth);
	store_set_to(gen, depth - 1, set /* gen->S0 */);
	if (gen->f_print_function) {
		(*gen->print_function)(cout, depth,
				set /* gen->S0 */, gen->print_function_data);
		}
	FREE_int(set);
	//cout << "poset_orbit_node::print_set_verbose done" << endl;
}

void poset_orbit_node::print_set(poset_classification *gen)
{
	int depth, size, i;
	longinteger_object go;
	longinteger_domain D;
	int *set;
	
	depth = depth_of_node(gen);
	//cout << "poset_orbit_node::print_set depth = " << depth << endl;
	size = depth;
	set = NEW_int(size);
	store_set_to(gen, depth - 1, set /*gen->S0*/);
	int_set_print(cout, set /*gen->S0*/, size);
	if (nb_strong_generators == 0) {
		cout << "_1";
		}
	else {
		D.multiply_up(go, tl, gen->Poset->A->base_len(), 0 /* verbose_level */);
		cout << "_{";
		for (i = 0; i < gen->Poset->A->base_len(); i++) {
			cout << tl[i];
			if (i < gen->Poset->A->base_len() - 1)
				cout << " * ";
			}
		cout << " = " << go << "}";
		cout << " in action ";
		cout << gen->Poset->A->label << endl;
		}

	//gen->print_lex_rank(set, size);
	
	FREE_int(set);
}

void poset_orbit_node::print_node(poset_classification *gen)
{
	int depth;
	int *set;
	//int i, depth, node2, len;
	//int *orbit;
	
	//orbit = NEW_int(gen->A->degree);
	depth = depth_of_node(gen);
	cout << "Node " << node << " at depth "
			<< depth << ", prev=" << prev << endl;
	print_set(gen);
	cout << endl;
	//cout << "pt=" << pt << endl;
	cout << "nb_strong_generators=" << nb_strong_generators << endl;
	cout << "nb_extensions=" << nb_extensions << endl;
	
	set = NEW_int(depth);
	store_set_to(gen, depth - 1, set /*gen->S0*/);

	if (gen->f_print_function) {
		(*gen->print_function)(cout, depth,
				set /* gen->S0 */, gen->print_function_data);
		}

	FREE_int(set);
	print_extensions(gen);
	
#if 0
	for (i = 0; i < nb_extensions; i++) {
		cout << setw(3) << i << " : " << setw(7)
				<< E[i].pt << " : " << setw(5) << E[i].orbit_len << " : ";
		len = gen->A->compute_orbit_of_point_generators_by_handle(
			nb_strong_generators, hdl_strong_generators, E[i].pt, orbit, 0);
		if (len != E[i].orbit_len) {
			cout << "poset_orbit_node::print_node "
					"len != E[i].orbit_len" << endl;
			cout << "len = " << len << endl;
			cout << "E[i].orbit_len = " << E[i].orbit_len << endl;
			}
		int_vec_heapsort(orbit, len); // int_vec_sort(len, orbit);
		if (E[i].type == EXTENSION_TYPE_UNPROCESSED) {
			cout << "unprocessed";
			}
		else if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			cout << "extension to node " << E[i].data;
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			//cout << "fusion node from ";
			gen->A->element_retrieve(E[i].data, gen->Elt1, FALSE);
			store_set(gen, depth - 1);
			gen->S[depth] = E[i].pt;
			//int_vec_print(cout, gen->S, depth + 1);
			//cout << " to ";
			gen->A->map_a_set(gen->S, gen->set[0], depth + 1, gen->Elt1, 0);
			//int_vec_print(cout, gen->set[0], depth + 1);
			int_vec_heapsort(gen->set[0], depth + 1);
			// int_vec_sort(depth + 1, gen->set[0]);
			//cout << " = ";
			//int_vec_print(cout, gen->set[0], depth + 1);
			node2 = gen->find_poset_orbit_node_for_set(
					depth + 1, gen->set[0], 0 /* f_tolerant */, 0);
			//cout << node2;
			cout << "fusion to node " << node2;
			}
		else if (E[i].type == EXTENSION_TYPE_PROCESSING) {
			cout << "currently processing";
			}
		cout << " : ";
		int_vec_print(cout, orbit, len);
		cout << endl;
		}
	FREE_int(orbit);
#endif	
}

void poset_orbit_node::print_extensions(poset_classification *gen)
{
	//int i, depth, /*node2,*/ len;
	int depth;
	int *orbit;
	
	depth = depth_of_node(gen);
	cout << "poset_orbit_node::print_extensions node=" << node
			<< " at depth " << depth
			<< " degree=" << gen->Poset->A2->degree << endl;
	print_extensions(cout);
	orbit = NEW_int(gen->Poset->A2->degree);

#if 0
	if (nb_extensions >= 10) {
		cout << "too many to print "
				"(nb_extensions=" << nb_extensions << ")" << endl;
		goto the_end;
	}
#endif

	int i;

	cout << "flag orbits:" << endl;
	cout << "i : point : orbit length" << endl;
	for (i = 0; i < nb_extensions; i++) {
		cout << setw(3) << i << " : " << setw(7) << E[i].pt
				<< " : " << setw(5) << E[i].orbit_len << endl;
	}

#if 0
	for (i = 0; i < nb_extensions; i++) {
		cout << setw(3) << i << " : " << setw(7) << E[i].pt
				<< " : " << setw(5) << E[i].orbit_len << " : ";

#if 0
		cout << "before gen->A->compute_orbit_of_point_generators_"
				"by_handle nb_strong_generators="
				<< nb_strong_generators << endl;
#endif

		if (FALSE) {
			len = gen->A2->compute_orbit_of_point_generators_by_handle(
				nb_strong_generators,
				hdl_strong_generators,
				E[i].pt, orbit, 0);
			cout << "orbit of length " << len << endl;
			if (len != E[i].orbit_len) {
				cout << "poset_orbit_node::print_extensions "
						"len != E[i].orbit_len" << endl;
				cout << "len = " << len << endl;
				cout << "E[i].orbit_len = " << E[i].orbit_len << endl;
				}
			int_vec_heapsort(orbit, len);
			}
		if (E[i].type == EXTENSION_TYPE_UNPROCESSED) {
			cout << "unprocessed";
			}
		else if (E[i].type == EXTENSION_TYPE_EXTENSION) {
			cout << "extension to node " << E[i].data;
			}
		else if (E[i].type == EXTENSION_TYPE_FUSION) {
			cout << "fusion node from " << endl;
			store_set_with_verbose_level(gen, depth - 1, 1);
			gen->S[depth] = E[i].pt;
			int_vec_print(cout, gen->S, depth + 1);

			cout << "fusion handle=" << E[i].data << endl;
			gen->A->element_retrieve(E[i].data, gen->Elt1, FALSE);
			cout << "fusion element:" << endl;
			gen->A2->element_print_quick(gen->Elt1, cout);

			cout << " to " << E[i].data1 << "/" << E[i].data2 << endl;
#if 0
			gen->A2->map_a_set(gen->S, gen->set[0], depth + 1, gen->Elt1, 0);
			int_vec_print(cout, gen->set[0], depth + 1);
			cout << endl;
			int_vec_heapsort(gen->set[0], depth + 1);
			cout << " = " << endl;
			int_vec_print(cout, gen->set[0], depth + 1);
			cout << endl;
			node2 = gen->find_poset_orbit_node_for_set(
					depth + 1, gen->set[0], 0 /* f_tolerant */, 0);
			cout << "Which is node " << node2 << endl;
			cout << "fusion to node " << node2 << endl;
#endif

			}
		else if (E[i].type == EXTENSION_TYPE_PROCESSING) {
			cout << "currently processing";
			}
		cout << " : ";


		//int_vec_print(cout, orbit, len);
		cout << endl;
		}
#endif

the_end:
	FREE_int(orbit);	
}



void poset_orbit_node::reconstruct_extensions_from_sv(
		poset_classification *gen, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int n, nb, i, j, a, idx;
	int *pts;
	int *prev;
	int *ancestor;
	int *depth;
	int *orbit_reps;
	sorting Sorting;


	if (f_v) {
		cout << "poset_orbit_node::reconstruct_extensions_from_sv" << endl;
		}
	n = get_nb_of_live_points();
	nb = get_nb_of_orbits_under_stabilizer();
	if (f_v) {
		cout << "n=" << n << " nb=" << nb << endl;
		}
	pts = live_points();
	prev = Schreier_vector->prev();

	ancestor = NEW_int(n);
	depth = NEW_int(n);
	orbit_reps = NEW_int(nb);
	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
		}
	for (i = 0; i < n; i++) {
		Schreier_vector->determine_depth_recursion(
				n, pts, prev, depth, ancestor, i);
		}
	
	nb_extensions = nb;
	E = NEW_OBJECTS(extension, nb);
	for (i = 0; i < nb; i++) {
		E[i].orbit_len = 0;
		E[i].type = EXTENSION_TYPE_UNPROCESSED;
		}
	j = 0;
	for (i = 0; i < n; i++) {
		if (prev[i] == -1) {
			E[j].pt = pts[i];
			orbit_reps[j] = pts[i];
			j++;
			}
		}
	for (i = 0; i < n; i++) {
		a = ancestor[i];
		if (!Sorting.int_vec_search(orbit_reps, nb, a, idx)) {
			cout << "poset_orbit_node::reconstruct_extensions_from_sv "
					"did not find orbit rep" << endl;
			exit(1);
			}
		E[idx].orbit_len++;
		}

	FREE_int(ancestor);
	FREE_int(depth);
	FREE_int(orbit_reps);
}

int poset_orbit_node::nb_extension_points()
// sums up the lengths of orbits in all extensions
{
	int i, n;

	n = 0;
	for (i = 0; i < nb_extensions; i++) {
		n += E[i].orbit_len;
		}
	return n;

}


}}


