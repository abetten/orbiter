// poset_orbit_node.cpp
//
// Anton Betten
// December 27, 2004

#include "foundations/foundations.h"
#include "discreta/discreta.h"
#include "group_actions/group_actions.h"
#include "classification/classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace poset_classification {


poset_orbit_node::poset_orbit_node()
{
	pt = -1;
	prev = -1;
	node = -1;
	nb_strong_generators = 0;
	first_strong_generator_handle = -1;
	//hdl_strong_generators = NULL;
	tl = NULL;
	nb_extensions = 0;
	E = NULL;
	Schreier_vector = NULL;
	A_on_upset = NULL;
	//null();
}

poset_orbit_node::~poset_orbit_node()
{
	freeself();
}

void poset_orbit_node::null()
{
	pt = -1;
	prev = -1;
	node = -1;
	nb_strong_generators = 0;
	first_strong_generator_handle = -1;
	//hdl_strong_generators = NULL;
	tl = NULL;
	nb_extensions = 0;
	E = NULL;
	Schreier_vector = NULL;
	A_on_upset = NULL;
}

void poset_orbit_node::freeself()
{
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_orbit_node::freeself node=" << node << endl;
	}
#if 0
	if (hdl_strong_generators) {
		if (f_v) {
			cout << "poset_orbit_node::freeself "
					"deleting hdl_strong_generators: ";
			int_vec_print(cout, hdl_strong_generators, nb_strong_generators);
			cout << endl;
			cout << "pointer = ";
			print_pointer_hex(cout, hdl_strong_generators);
			cout << endl;
		}
		FREE_int(hdl_strong_generators);
		//cout << "poset_orbit_node::freeself() "
		//"deleting hdl_strong_generators done" << endl;
		}
#endif
	if (tl) {
		if (f_v) {
			cout << "poset_orbit_node::freeself deleting tl" << endl;
		}
		FREE_int(tl);
		}
	if (E) {
		if (f_v) {
			cout << "poset_orbit_node::freeself deleting E" << endl;
		}
		FREE_OBJECTS(E);
	}
	if (Schreier_vector) {
		if (f_v) {
			cout << "poset_orbit_node::freeself deleting Schreier_vector" << endl;
		}
		FREE_OBJECT(Schreier_vector);
	}
	if (A_on_upset) {
		if (f_v) {
			cout << "poset_orbit_node::freeself deleting A_on_upset" << endl;
		}
		FREE_OBJECT(A_on_upset);
	}
	null();
	if (f_v) {
		cout << "poset_orbit_node::freeself done" << endl;
	}
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
	
	ring_theory::longinteger_object go;

	if (f_v) {
		cout << "poset_orbit_node::init_root_node "
				"poset " << gen->get_problem_label()
				<< ", computing the group order of G" << endl;
	}
	gen->get_poset()->Strong_gens->group_order(go);

	if (f_v) {
		cout << "poset_orbit_node::init_root_node "
				"storing strong generators "
				"for a group of order " << go << endl;
	}
	store_strong_generators(gen, gen->get_poset()->Strong_gens);
		// stores the strong generators into
		// the poset_orbit_node structure,
		// copies transversal_length into tl
	if (f_v) {
		cout << "poset_orbit_node::init_root_node done" << endl;
	}
	
}

void poset_orbit_node::init_node(int node, int prev, long int pt, int verbose_level)
{
	//freeself();
	poset_orbit_node::node = node;
	poset_orbit_node::prev = prev;
	poset_orbit_node::pt = pt;
	nb_strong_generators = 0;
	first_strong_generator_handle = -1;
	//hdl_strong_generators = NULL;
	tl = NULL;
	E = NULL;
	Schreier_vector = NULL;
}


int poset_orbit_node::get_node()
{
	return node;
}

void poset_orbit_node::set_node(int node)
{
	poset_orbit_node::node = node;
}

void poset_orbit_node::delete_Schreier_vector()
{
	FREE_OBJECT(Schreier_vector);
	Schreier_vector = NULL;
}


void poset_orbit_node::allocate_E(int nb_extensions, int verbose_level)
{
	E = NEW_OBJECTS(extension, 1);
	nb_extensions = 1;
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
	n = node - gen->first_node_at_level(l);
	return n;
}


void poset_orbit_node::get_strong_generators_handle(std::vector<int> &gen_hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_orbit_node::get_strong_generators_handle" << endl;
	}
	for (i = 0; i < nb_strong_generators; i++) {
		//gen_hdl.push_back(hdl_strong_generators[i]);
		gen_hdl.push_back(first_strong_generator_handle + i);
	}

	if (f_v) {
		cout << "poset_orbit_node::get_strong_generators_handle done" << endl;
	}
}

void poset_orbit_node::get_tl(std::vector<int> &tl, poset_classification *PC, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_orbit_node::get_tl" << endl;
	}
	if (nb_strong_generators) {
		for (i = 0; i < PC->get_poset()->A->base_len(); i++) {
			tl.push_back(poset_orbit_node::tl[i]);
		}
	}
	else {
		for (i = 0; i < PC->get_poset()->A->base_len(); i++) {
			tl.push_back(1);
		}
	}

	if (f_v) {
		cout << "poset_orbit_node::get_tl done" << endl;
	}
}

int poset_orbit_node::get_tl(int i)
{
	return tl[i];
}


int poset_orbit_node::has_Schreier_vector()
{
	if (Schreier_vector == NULL) {
		return FALSE;
	}
	else {
		return TRUE;
	}
}


data_structures_groups::schreier_vector *poset_orbit_node::get_Schreier_vector()
{
	return Schreier_vector;
}

int poset_orbit_node::get_nb_strong_generators()
{
	return nb_strong_generators;
}

int *poset_orbit_node::live_points()
{
	if (Schreier_vector == NULL) {
		cout << "poset_orbit_node::live_points "
				"Schreier_vector == NULL" << endl;
		exit(1);
	}
	else {
		return Schreier_vector->points();
	}
}

int poset_orbit_node::get_nb_of_live_points()
{
	if (Schreier_vector == NULL) {
		//cout << "poset_orbit_node::get_nb_of_live_points "
		//		"Schreier_vector == NULL" << endl;
		return 0;
	}
	else {
		return Schreier_vector->get_number_of_points();
	}
}

int poset_orbit_node::get_nb_of_orbits_under_stabilizer()
{
	if (Schreier_vector == NULL) {
		//cout << "poset_orbit_node::get_nb_of_orbits_under_stabilizer "
		//		"Schreier_vector == NULL" << endl;
		return 0;
	}
	else {
		return Schreier_vector->get_number_of_orbits();
	}
}

int poset_orbit_node::get_nb_of_extensions()
{
	return nb_extensions;
}


extension *poset_orbit_node::get_E(int idx)
{
	return E + idx;
}

long int poset_orbit_node::get_pt()
{
	return pt;
}

void poset_orbit_node::set_pt(long int pt)
{
	poset_orbit_node::pt = pt;
}

int poset_orbit_node::get_prev()
{
	return prev;
}

void poset_orbit_node::set_prev(int prev)
{
	poset_orbit_node::prev = prev;
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
	if (cur_depth == max_depth) {
		return;
	}
	for (i = 0; i < nb_extensions; i++) {
		if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			nxt = E[i].get_data();
			if (nxt >= 0) {
				gen->get_node(nxt)->poset_orbit_node_depth_breadth_perm_and_inverse(gen,
					max_depth, idx, nxt, cur_depth + 1, perm, perm_inv);
			}
		}
	}
}

int poset_orbit_node::find_extension_from_point(
		poset_classification *gen,
		long int pt, int verbose_level)
// a -1 means not found
{
	int i;
	
	for (i = 0; i < nb_extensions; i++) {
		if (E[i].get_pt() == pt) {
			break;
		}
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
			<< setw(7) << E[i].get_pt() << " : "
			<< setw(5) << E[i].get_orbit_len() << " : ";

		print_extension_type(ost, E[i].get_type());
		if (E[i].get_type() == EXTENSION_TYPE_FUSION) {
			ost << " -> (" << E[i].get_data1() << ","
					<< E[i].get_data2() << ") hdl=" << E[i].get_data() << endl;
		}
		else if (E[i].get_type() == EXTENSION_TYPE_EXTENSION) {
			ost << " -> " << E[i].get_data() << endl;
		}
		else {
			ost << setw(5) << E[i].get_data() << endl;
		}
		if (E[i].get_type() >= NB_EXTENSION_TYPES) {
			ost << "E[i].get_type() >= NB_EXTENSION_TYPES" << endl;
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
	ring_theory::longinteger_object go;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::log_current_node_without_group" << endl;
		}
	store_set_to(gen, s - 1, gen->get_set0());
	
	if (f_v) {
		f << "# ***** orbit ***** " <<
				node - gen->first_node_at_level(s) << " "<< endl;
		}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << gen->get_set0()[i] << " ";
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
	ring_theory::longinteger_object go, rk;
	int i;

	if (f_v) {
		cout << "poset_orbit_node::log_current_node node="
				<< node << " s=" << s << endl;
	}
	store_set_to(gen, s - 1, gen->get_set0());
	if (f_v) {
		cout << "poset_orbit_node::log_current_node node="
				<< node << " after store_set_to" << endl;
	}
	
	if (f_v) {
		f << "# ***** orbit ***** "
				<< node - gen->first_node_at_level(s)
				<< " " << endl;
	}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << gen->get_set0()[i] << " ";
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

	data_structures_groups::group_container G;

	G.init(gen->get_poset()->A, verbose_level - 2);


#if 0
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

#else

	get_stabilizer(
		gen,
		G, go,
		verbose_level);

#endif
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
		groups::strong_generators *Strong_gens;
		ring_theory::longinteger_object go1;

		get_stabilizer_generators(gen, Strong_gens, verbose_level);
		Strong_gens->group_order(go1);
		cout << "The stabilizer is a group of order " << go1 << endl;
		cout << "With the following generators:" << endl;
		Strong_gens->print_generators(cout);
		FREE_OBJECT(Strong_gens);
	}

#if 1
	if (gen->get_poset()->f_print_function) {
		f << "# BEGINCOMMENT" << endl;
		if (gen->get_poset()->f_print_function) {
			gen->get_poset()->invoke_print_function(f, s, gen->get_set0());
		}
		
		if (!go.is_one()) {
			if (f_v) {
				cout << "poset_orbit_node::log_current_node "
						"node=" << node << " printing generators" << endl;
			}
			G.require_strong_generators();
			f << "tl: ";
			for (i = 0; i < G.A->base_len(); i++) {
				f << G.tl[i] << " ";
			}
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
	ring_theory::longinteger_object go;
	int i;
	int *S;
	int *Elt;
	int *Elt_inv;
	int *Elt1;
	int *Elt2;
	
	S = NEW_int(s);
	Elt = NEW_int(gen->get_poset()->A->elt_size_in_int);
	Elt_inv = NEW_int(gen->get_poset()->A->elt_size_in_int);
	Elt1 = NEW_int(gen->get_poset()->A->elt_size_in_int);
	Elt2 = NEW_int(gen->get_poset()->A->elt_size_in_int);
	
	store_set_to(gen, s - 1, gen->get_set0());
	gen->get_poset()->A->element_retrieve(hdl, Elt, 0);
	//gen->A->element_print(Elt, cout);
	gen->get_poset()->A->element_invert(Elt, Elt_inv, 0);
	for (i = 0; i < s; i++) {
		S[i] = Elt[gen->get_set0()[i]];
		}
	
	if (f_v) {
		f << "# ***** orbit ***** "
				<< node - gen->first_node_at_level(s)
				<< " " << endl;
		}
	f << s << " ";
	for (i = 0; i < s; i++) {
		f << S[i] << " ";
		}
	data_structures_groups::group_container G;

	G.init(gen->get_poset()->A, verbose_level - 2);

#if 0
	G.init_strong_generators_by_hdl(
			nb_strong_generators,
			hdl_strong_generators, tl,
			FALSE);
	G.schreier_sims(0);
	G.group_order(go);
#else

	get_stabilizer(
		gen,
		G, go,
		verbose_level);


	#endif

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

	store_set_to(gen, lvl - 1, gen->get_set0());
	
	f << lvl << " ";
	for (i = 0; i < lvl; i++) {
		f << gen->get_set0()[i] << " ";
		}
	f << -1 << " ";
	
#if 0
	// ToDo
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
#endif
}


int poset_orbit_node::depth_of_node(poset_classification *gen)
{
	if (prev == -1) {
		return 0;
		}
	else {
		return gen->get_node(prev)->depth_of_node(gen) + 1;
		}
}

void poset_orbit_node::store_set(poset_classification *gen, int i)
// stores a set of size i + 1 to gen->S[]
{
	if (i < 0) {
		return;
	}
	gen->get_S()[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set prev == -1" << endl;
			exit(1);
		}
		gen->get_node(prev)->store_set(gen, i - 1);
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
	if (i < 0) {
		return;
	}
	gen->get_S()[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set prev == -1" << endl;
			exit(1);
			}
		gen->get_node(prev)->store_set(gen, i - 1);
	}
}

void poset_orbit_node::store_set_to(
		poset_classification *gen, int i, long int *to)
// stores a set of size i + 1 to 'to'
{
	if (i < 0) {
		return;
	}
	to[i] = pt;
	if (i >= 0) {
		if (prev == -1) {
			cout << "store_set_to prev == -1" << endl;
			exit(1);
		}
		gen->get_node(prev)->store_set_to(gen, i - 1, to);
	}
}

void poset_orbit_node::store_set_to(
		poset_classification *gen, long int *to)
{
	store_set_to(gen, depth_of_node(gen), to);
}

int poset_orbit_node::check_node_and_set_consistency(
		poset_classification *gen, int i, long int *set)
{
	if (i < 0) {
		return TRUE;
	}
	if (set[i] != pt) {
		cout << "check_node_and_set_consistency inconsistent" << endl;
		return FALSE;
	}
	if (i >= 0) {
		if (prev == -1) {
			cout << "check_node_and_set_consistency prev == -1" << endl;
			exit(1);
			}
		gen->get_node(prev)->check_node_and_set_consistency(
				gen, i - 1, set);
	}
	return TRUE;
}

void poset_orbit_node::print_set_verbose(poset_classification *gen)
{
	int depth;
	long int *set;

	//cout << "poset_orbit_node::print_set_verbose" << endl;
	depth = depth_of_node(gen);
	print_set(gen);
	cout << endl;


	set = NEW_lint(depth);
	store_set_to(gen, depth - 1, set /* gen->S0 */);
	if (gen->get_poset()->f_print_function) {
		gen->get_poset()->invoke_print_function(cout, depth, set /* gen->S0 */);
		}
	FREE_lint(set);
	//cout << "poset_orbit_node::print_set_verbose done" << endl;
}

void poset_orbit_node::print_set(poset_classification *gen)
{
	int depth, size, i;
	ring_theory::longinteger_object go;
	ring_theory::longinteger_domain D;
	long int *set;
	
	depth = depth_of_node(gen);
	//cout << "poset_orbit_node::print_set depth = " << depth << endl;
	size = depth;
	set = NEW_lint(size);
	store_set_to(gen, depth - 1, set /*gen->S0*/);
	Orbiter->Lint_vec->print(cout, set /*gen->S0*/, size);
	if (nb_strong_generators == 0) {
		cout << "_1";
	}
	else {
		D.multiply_up(go, tl, gen->get_A()->base_len(), 0 /* verbose_level */);
		cout << "_{";
		for (i = 0; i < gen->get_A()->base_len(); i++) {
			cout << tl[i];
			if (i < gen->get_A()->base_len() - 1)
				cout << " * ";
			}
		cout << " = " << go << "}";
		cout << " in action ";
		cout << gen->get_A2()->label << endl;
	}

	//gen->print_lex_rank(set, size);
	
	FREE_lint(set);
}

void poset_orbit_node::print_node(poset_classification *gen)
{
	int depth;
	long int *set;
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
	
	set = NEW_lint(depth);
	store_set_to(gen, depth - 1, set /*gen->S0*/);

	if (gen->get_poset()->f_print_function) {
		gen->get_poset()->invoke_print_function(cout, depth, set /* gen->S0 */);
		}

	FREE_lint(set);
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
			<< " degree=" << gen->get_A2()->degree << endl;
	print_extensions(cout);
	orbit = NEW_int(gen->get_A2()->degree);

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
		cout << setw(3) << i << " : " << setw(7) << E[i].get_pt()
				<< " : " << setw(5) << E[i].get_orbit_len() << endl;
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

//the_end:
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
	data_structures::sorting Sorting;


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
		E[i].set_orbit_len(0);
		E[i].set_type(EXTENSION_TYPE_UNPROCESSED);
	}
	j = 0;
	for (i = 0; i < n; i++) {
		if (prev[i] == -1) {
			E[j].set_pt(pts[i]);
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
		E[idx].set_orbit_len(E[idx].get_orbit_len() + 1);
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
		n += E[i].get_orbit_len();
	}
	return n;

}


}}}


