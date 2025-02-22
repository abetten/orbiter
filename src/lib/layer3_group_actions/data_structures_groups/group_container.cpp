// group_container.cpp
//
// Anton Betten
// December 24, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


group_container::group_container()
{
	Record_birth();
	A = NULL;
	f_has_ascii_coding = false;
	//std::string ascii_coding;

	f_has_strong_generators = false;
	SG = NULL;
	tl = NULL;

	f_has_sims = false;
	S = NULL;
}

group_container::~group_container()
{
	Record_death();
	delete_ascii_coding();
	delete_strong_generators();
	delete_sims();
}

void group_container::init(
		actions::action *A,
		int verbose_level)
{
	//null();
	group_container::A = A;
}

void group_container::init_ascii_coding_to_sims(
		std::string &ascii_coding, int verbose_level)
{
	if (ascii_coding.length()) {
		init_ascii_coding(ascii_coding, verbose_level);
		
		decode_ascii(0);
		
		// now strong generators are available
		
	}
	else {
		//cout << "trivial group" << endl;
		init_strong_generators_empty_set(verbose_level);
	}
	
	schreier_sims(0);
}

void group_container::init_ascii_coding(
		std::string &ascii_coding, int verbose_level)
{
	delete_ascii_coding();
	
	group_container::ascii_coding.assign(ascii_coding);
	f_has_ascii_coding = true;
}

void group_container::delete_ascii_coding()
{
}

void group_container::init_strong_generators_empty_set(
		int verbose_level)
{
	int i;
	
	delete_strong_generators();
	
	group_container::SG = NEW_OBJECT(vector_ge);
	group_container::SG->init(A, verbose_level - 2);
	group_container::SG->allocate(0, verbose_level - 2);
	group_container::tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		group_container::tl[i] = 1;
	}
	f_has_strong_generators = true;
}

void group_container::init_strong_generators(
		vector_ge &SG,
		int *tl, int verbose_level)
{
	int i;
	
	delete_strong_generators();
	
	group_container::SG = NEW_OBJECT(vector_ge);
	group_container::SG->init(A, verbose_level - 2);
	group_container::SG->allocate(SG.len, verbose_level - 2);
	for (i = 0; i < SG.len; i++) {
		group_container::SG->copy_in(i, SG.ith(i));
	}
	group_container::tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		group_container::tl[i] = tl[i];
	}
	f_has_strong_generators = true;
}

void group_container::init_strong_generators_by_handle_and_with_tl(
		std::vector<int> &gen_handle,
		std::vector<int> &tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *Gen_hdl;
	int *Tl;

	if (f_v) {
		cout << "group_container::init_strong_generators_by_handle" << endl;
	}
	Gen_hdl = NEW_int(gen_handle.size());
	for (i = 0; i < gen_handle.size(); i++) {
		Gen_hdl[i] = gen_handle[i];
	}

	Tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		Tl[i] = tl[i];
	}

	init_strong_generators_by_hdl(gen_handle.size(),
			Gen_hdl, Tl, verbose_level);

	FREE_int(Gen_hdl);
	FREE_int(Tl);

	if (f_v) {
		cout << "group_container::init_strong_generators_by_handle done" << endl;
	}
}

void group_container::init_strong_generators_by_hdl(
		int nb_gen, int *gen_hdl,
		int *tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl" << endl;
		}
	if (f_v) {
		cout << "gen_hdl=";
		Int_vec_print(cout, gen_hdl, nb_gen);
		cout << endl;
		if (nb_gen) {
			cout << "tl=";
			Int_vec_print(cout, tl, A->base_len());
			cout << endl;
		}
	}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before delete_strong_generators" << endl;
	}
	delete_strong_generators();
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"after delete_strong_generators" << endl;
	}
	
	SG = NEW_OBJECT(vector_ge);
	SG->init(A, verbose_level - 2);
	SG->allocate(nb_gen, verbose_level - 2);
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before A->element_retrieve" << endl;
	}
	for (i = 0; i < nb_gen; i++) {
		A->Group_element->element_retrieve(gen_hdl[i], SG->ith(i), 0/*verbose_level*/);
	}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"after A->element_retrieve" << endl;
	}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before allocating tl of size " << A->base_len() << endl;
	}
	group_container::tl = NEW_int(A->base_len());
	if (nb_gen) {
		for (i = 0; i < A->base_len(); i++) {
			group_container::tl[i] = tl[i];
		}
	}
	else {
		for (i = 0; i < A->base_len(); i++) {
			group_container::tl[i] = 1;
		}
	}
	f_has_strong_generators = true;
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl done" << endl;
	}
}

void group_container::delete_strong_generators()
{
	if (f_has_strong_generators) {
		FREE_OBJECT(SG);
		FREE_int(tl);
		SG = NULL;
		tl = NULL;
		f_has_strong_generators = false;
	}
}

void group_container::delete_sims()
{
	if (f_has_sims) {
		if (S) {
			FREE_OBJECT(S);
			S = NULL;
		}
		f_has_sims = false;
	}
}

void group_container::require_ascii_coding()
{
	if (!f_has_ascii_coding) {
		cout << "group_container::require_ascii_coding !f_has_ascii_coding" << endl;
		exit(1);
	}
}

void group_container::require_strong_generators()
{
	if (!f_has_strong_generators) {
		cout << "group_container::require_strong_generators !f_has_strong_generators" << endl;
		exit(1);
	}
}

void group_container::require_sims()
{
	if (!f_has_sims) {
		cout << "group_container::require_sims !f_has_sims" << endl;
		exit(1);
	}
}

void group_container::group_order(
		algebra::ring_theory::longinteger_object &go)
{
	algebra::ring_theory::longinteger_domain D;
	
	if (f_has_sims) {
		S->group_order(go);
		//D.multiply_up(go, S->orbit_len, A->base_len());
	}
	else if (f_has_strong_generators) {
		D.multiply_up(go, tl, A->base_len(), 0 /* verbose_level */);
	}
	else {
		cout << "group::group_order need sims or strong_generators" << endl;
		exit(1);
	}
}

void group_container::print_group_order(
		std::ostream &ost)
{
	algebra::ring_theory::longinteger_object go;
	group_order(go);
	ost << go;
}

void group_container::print_tl()
{
	int i;
	
	if (f_has_strong_generators) {
		for (i = 0; i < A->base_len(); i++) {
			cout << tl[i] << " ";
		}
		cout << endl;
	}
}

void group_container::code_ascii(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int sz, i, j;
	char *p;
	char *p0;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "group_container::code_ascii action " << A->label
				<< " base_len=" << A->base_len() << endl;
	}
	require_strong_generators();
	sz = 2 * ((2 + A->base_len() + A->base_len()) * sizeof(int_4)
			+ A->coded_elt_size_in_char * SG->len) + 1;
	p = NEW_char(sz);
	p0 = p;

	//cout << "group::code_ascii action A->base_len=" << A->base_len << endl;
	Os.code_int4(p, (int_4) A->base_len());

	//cout << "group::code_ascii action SG->len=" << SG->len << endl;
	Os.code_int4(p, (int_4) SG->len);
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) A->base_i(i));
	}
	for (i = 0; i < A->base_len(); i++) {
		Os.code_int4(p, (int_4) tl[i]);
	}
	for (i = 0; i < SG->len; i++) {
		A->Group_element->element_pack(SG->ith(i), A->Group_element->elt1, false);
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			Os.code_uchar(p, A->Group_element->elt1[j]);
		}
	}
	*p++ = 0;
	if (p - p0 != sz) {
		cout << "group_container::code_ascii p - p0 != sz" << endl;
		exit(1);
	}

	ascii_coding.assign(p0);
	f_has_ascii_coding = true;
	if (f_v) {
		cout << "group_container::code_ascii " << ascii_coding << endl;
	}
}

void group_container::decode_ascii(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int len, nbsg;
	int *base1;
	const char *p, *p0;
	int str_len;
	other::orbiter_kernel_system::os_interface Os;

	require_ascii_coding();
	//cout << "group_container::decode_ascii ascii_coding=" << ascii_coding << endl;
	p = ascii_coding.c_str();
	p0 = p;
	str_len = ascii_coding.length();
	len = Os.decode_int4(p);
	nbsg = Os.decode_int4(p);
	if (len != A->base_len()) {
		cout << "group_container::decode_ascii len != A->base_len" << endl;
		cout << "len=" << len << " (from file)" << endl;
		cout << "A->base_len=" << A->base_len() << endl;
		cout << "action A is " << A->label << endl;
		exit(1);
	}
	delete_strong_generators();
	SG = NEW_OBJECT(vector_ge);
	SG->init(A, verbose_level - 2);
	SG->allocate(nbsg, verbose_level - 2);
	base1 = NEW_int(A->base_len());
	tl = NEW_int(A->base_len());
	for (i = 0; i < A->base_len(); i++) {
		base1[i] = Os.decode_int4(p);
	}
	for (i = 0; i < A->base_len(); i++) {
		if (base1[i] != A->base_i(i)) {
			cout << "group_container::decode_ascii base mismatch" << endl;
			exit(1);
		}
	}
	for (i = 0; i < A->base_len(); i++) {
		tl[i] = Os.decode_int4(p);
	}
	for (i = 0; i < nbsg; i++) {
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			Os.decode_uchar(p, A->Group_element->elt1[j]);
		}
		A->Group_element->element_unpack(A->Group_element->elt1, SG->ith(i), false);
	}
	FREE_int(base1);
	if (p - p0 != str_len) {
		cout << "group_container::decode_ascii p - p0 != str_len" << endl;
		cout << "p - p0 = " << p - p0 << endl;
		cout << "str_len = " << str_len << endl;
		exit(1);
	}
	f_has_strong_generators = true;
	if (f_v) {
		if (SG->len < 10) {
			SG->print(cout);
		}
		cout << "found a group with " << SG->len
				<< " strong generators" << endl;
	}
}

void group_container::schreier_sims(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "group_container::schreier_sims" << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	require_strong_generators();
	if (f_v) {
		cout << "group_container::schreier_sims before delete_sims" << endl;
	}
	delete_sims();
	if (f_v) {
		cout << "group_container::schreier_sims after delete_sims" << endl;
	}
	S = NEW_OBJECT(groups::sims);
	if (false) {
		cout << "group_container::schreier_sims calling S->init(A)" << endl;
	}
	S->init(A, verbose_level - 2);
	if (false) {
		cout << "group_container::schreier_sims calling S->init_generators" << endl;
	}
	if (false) {
		cout << "generators" << endl;
		SG->print(cout);
	}
	S->init_generators(*SG, verbose_level - 2);
	if (f_v) {
		cout << "group_container::schreier_sims after S->init_generators" << endl;
		cout << "tl: ";
		Int_vec_print(cout, tl, A->base_len());
		cout << endl;
	}
	if (f_v) {
		cout << "group_container::schreier_sims before "
				"compute_base_orbits_known_length" << endl;
	}
	S->compute_base_orbits_known_length(tl, verbose_level - 2);
	if (f_v) {
		cout << "group_container::schreier_sims after "
				"compute_base_orbits_known_length" << endl;
	}
	
	if (f_v) {
		cout << "group_container::schreier_sims done. Found a group of order ";
		S->print_group_order(cout);
		cout << endl;
	}
	f_has_sims = true;
}

void group_container::get_strong_generators(
		int verbose_level)
{
	require_sims();
	delete_strong_generators();
	SG = NEW_OBJECT(vector_ge);
	SG->init(A, verbose_level - 2);
	tl = NEW_int(A->base_len());
	S->extract_strong_generators_in_order(*SG, tl, verbose_level - 1);
}

void group_container::point_stabilizer(
		group_container &stab, int pt,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	require_sims();

	vector_ge stab_gens;
	int *tl;
	
	if (f_v) {
		cout << "group_container::point_stabilizer "
				"computing stabilizer of point " << pt << endl;
	}
	
	
	tl = NEW_int(A->base_len());
	S->point_stabilizer(stab_gens, tl, pt, verbose_level - 1);
	
#if 0
	if (f_v) {
		if (f_vv) {
			stab_gens.print(cout);
			}
		cout << stab_gens.len << " strong generators computed" << endl;
		}
#endif
	stab.init(A, verbose_level - 2);
	stab.init_strong_generators(stab_gens, tl, verbose_level - 2);
	FREE_int(tl);
	if (f_v) {
		cout << "stabilizer of point " << pt << " has order ";
		stab.print_group_order(cout);
		cout << " ";
		Int_vec_print(cout, stab.tl, A->base_len());
		cout << " with " << stab_gens.len << " strong generators" << endl;
		if (f_vv) {
			stab_gens.print(cout);
		}
	}
}

void group_container::point_stabilizer_with_action(
		actions::action *A2,
		group_container &stab, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	require_sims();

	vector_ge stab_gens;
	int *tl;
	
	if (f_v) {
		cout << "group_container::point_stabilizer_with_action ";
		cout << "computing stabilizer of point " << pt 
			<< " in action " << A2->label 
			<< " internal action is " << stab.A->label << endl;
		cout << "verbose_level = " << verbose_level << endl;
	}
	
	
	tl = NEW_int(A->base_len());
	if (f_v) {
		cout << "group_container::point_stabilizer_with_action "
				"calling S->point_stabilizer_with_action" << endl;
	}
	S->point_stabilizer_with_action(A2, stab_gens, tl, pt, verbose_level - 1);
	if (f_v) {
		cout << "group_container::point_stabilizer_with_action "
				"after S->point_stabilizer_with_action" << endl;
	}
	
#if 0
	if (f_v) {
		if (f_vv) {
			stab_gens.print(cout);
		}
		cout << stab_gens.len << " strong generators computed" << endl;
	}
#endif
	stab.init(A, verbose_level - 2);
	if (f_v) {
		cout << "group_container::point_stabilizer_with_action "
				"before stab.init_strong_generators" << endl;
	}
	stab.init_strong_generators(stab_gens, tl, verbose_level - 2);
	if (f_v) {
		cout << "group_container::point_stabilizer_with_action "
				"after stab.init_strong_generators" << endl;
	}
	FREE_int(tl);
	if (f_v) {
		cout << "stabilizer of point " << pt << " has order ";
		stab.print_group_order(cout);
		cout << " ";
		Int_vec_print(cout, stab.tl, A->base_len());
		cout << " with " << stab_gens.len << " strong generators" << endl;
		if (f_vv) {
			stab_gens.print(cout);
		}
	}
}

void group_container::induced_action(
		actions::action &induced_action,
		group_container &H, group_container &K,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "group_container::induced_action" << endl;
		}
	{
		vector_ge H_SG, K_SG;
		int *H_tl, *K_tl;
		int n = 0;
		{
			groups::sims HH, KK;
			algebra::ring_theory::longinteger_object go, H_order, K_order, HK_order, quo, rem;
			int drop_out_level, image;
			algebra::ring_theory::longinteger_domain D;

			require_sims();

			group_order(go);

			HH.init(&induced_action, verbose_level - 2);
			HH.init_trivial_group(verbose_level - 1);
			HH.group_order(H_order);

			KK.init(A, verbose_level - 2);
			KK.init_trivial_group(verbose_level - 1);
			KK.group_order(K_order);

			D.mult(H_order, K_order, HK_order);
			if (f_v) {
				cout << "step " << n << " H_order " << H_order
					<< " K_order = " << K_order
					<< " HK_order " << HK_order << " of " << go << endl;
			}

			while (D.compare_unsigned(HK_order, go) != 0) {

				if (f_v) {
					cout << "step " << n << ":" << endl;
				}
				S->random_element(A->Group_element->Elt1, verbose_level - 1);
				if (f_v) {
					cout << "random group element:" << endl;
					A->Group_element->element_print(A->Group_element->Elt1, cout);
				}

				if (HH.strip(A->Group_element->Elt1, A->Group_element->Elt2 /* residue */,
						drop_out_level, image, verbose_level - 1)) {
					if (f_vv) {
						cout << "element strips through H" << endl;
					}
					if (KK.strip(A->Group_element->Elt2, A->Group_element->Elt3 /* residue */,
							drop_out_level, image, verbose_level - 1)) {
						if (f_vv) {
							cout << "element strips through K" << endl;
						}
					}
					else {
						KK.add_generator_at_level(A->Group_element->Elt3,
								drop_out_level, verbose_level - 1);
					}
				}
				else {
					HH.add_generator_at_level(A->Group_element->Elt2,
							drop_out_level, verbose_level - 1);
				}

				HH.group_order(H_order);
				KK.group_order(K_order);
				D.mult(H_order, K_order, HK_order);

				if (f_v) {
					cout << "step " << n << " H_order " << H_order
						<< " K_order = " << K_order
						<< " HK_order " << HK_order << " of " << go << endl;
					D.integral_division(go, HK_order, quo, rem, 0);
					cout << "remaining factor: " << quo
						<< " remainder " << rem << endl;
				}
				n++;
			}

#if 0
			if (f_v) {
				cout << "group::induced_action "
						"finished after " << n << " steps" << endl;
				cout << "H_order " << H_order << " K_order = " << K_order << endl;
				cout << "# generators for H = " << HH.gens.len
						<< ", # generators for K = " << KK.gens.len << endl;
				cout << "H:" << endl;
				HH.print(f_vv);
				cout << "K:" << endl;
				KK.print(f_vv);
			}
#endif

			H_tl = NEW_int(induced_action.base_len());
			K_tl = NEW_int(A->base_len());
	
			HH.extract_strong_generators_in_order(H_SG, H_tl, verbose_level - 2);
			KK.extract_strong_generators_in_order(K_SG, K_tl, verbose_level - 2);
	

			//cout << "group::induced_action deleting HH,KK" << endl;
		}

		H.init(&induced_action, verbose_level - 2);
		K.init(A, verbose_level - 2);
		H.init_strong_generators(H_SG, H_tl, verbose_level - 2);
		K.init_strong_generators(K_SG, K_tl, verbose_level - 2);
		if (f_v) {
			cout << "group_container::induced_action finished after "
					<< n << " iterations" << endl;
			cout << "order of the induced group  = ";
			H.print_group_order(cout);
			cout << endl;
			cout << "order of the kernel = ";
			K.print_group_order(cout);
			cout << endl;
		}
#if 0
		if (f_vv) {
			cout << "induced group:" << endl;
			HH.print(false);
			cout << endl;
			cout << "kernel:" << endl;
			KK.print(false);
			cout << endl;
			cout << H.SG->len << " strong generators for induced group:" << endl;
			H.SG->print(cout);
			cout << endl;
			cout << K.SG->len << " strong generators for kernel:" << endl;
			K.SG->print(cout);
			cout << endl;
		}
#endif
		FREE_int(H_tl);
		FREE_int(K_tl);
		//cout << "group::induced_action deleting SG" << endl;
	}
	if (f_v) {
		cout << "group_container::induced_action finished" << endl;
	}
}

void group_container::extension(
		group_container &N,
		group_container &H, int verbose_level)
	// N needs to have strong generators, 
	// H needs to have sims
	// N and H may have different actions, 
	// the action of N is taken for the extension.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	actions::action *A = N.A;
	groups::sims G;
	algebra::ring_theory::longinteger_object go_N, go_H, go_G, cur_go, quo, rem;
	algebra::ring_theory::longinteger_domain D;
	int n = 0, drop_out_level, image;
	int *p_gen;
	int *Elt;
	
	if (f_v) {
		cout << "group_container::extension" << endl;
	}
	N.require_strong_generators();
	N.group_order(go_N);
	H.group_order(go_H);
	D.mult(go_N, go_H, go_G);
	
	Elt = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "group_container::extension |N| = " << go_N << " |H| = "
			<< go_H << " |G| = |N|*|H| = " << go_G << endl;
	}
	H.require_sims();
	
	init(N.A, verbose_level - 2);
	G.init(N.A, verbose_level - 2);
	G.init_generators(*N.SG, f_v);
	G.compute_base_orbits(verbose_level - 1);
	G.group_order(cur_go);

	while (D.compare_unsigned(go_G, cur_go) != 0) {
		
		if (f_v) {
			cout << "step " << n << ":" << endl;
		}
		if (n % 2 || G.nb_gen[0] == 0) {
			H.S->random_element(A->Group_element->Elt1, verbose_level - 1);
			p_gen = A->Group_element->Elt1;
			if (f_v) {
				cout << "random group element:" << endl;
				A->Group_element->element_print(p_gen, cout);
			}
		}
		else {
			G.random_schreier_generator(Elt, verbose_level - 1);
			//p_gen = G.schreier_gen;
			if (f_v) {
				cout << "random schreier generator:" << endl;
				A->Group_element->element_print(Elt, cout);
			}
		}
		
		
		if (G.strip(Elt, A->Group_element->Elt2 /* residue */,
				drop_out_level, image, verbose_level - 1)) {
			if (f_vv) {
				cout << "element strips through" << endl;
			}
		}
		else {
			G.add_generator_at_level(A->Group_element->Elt2,
					drop_out_level, verbose_level - 1);
		}

		G.group_order(cur_go);

		if (f_v) {
			cout << "step " << n 
				<< " cur_go " << cur_go << " of " << go_G << endl;
			D.integral_division(go_G, cur_go, quo, rem, 0);
			cout << "remaining factor: " << quo
					<< " remainder " << rem << endl;
		}
		n++;
	}
	
	vector_ge SG;
	int *tl;
	
	tl = NEW_int(A->base_len());
	
	G.extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	
	init(A, verbose_level - 2);
	init_strong_generators(SG, tl, verbose_level - 2);
	
	if (f_v) {
		cout << "group_container::extension finished after "
				<< n << " iterations" << endl;
		cout << "order of the extension = ";
		print_group_order(cout);
		cout << endl;
	}
	FREE_int(Elt);
	FREE_int(tl);
}

void group_container::print_strong_generators(
		std::ostream &ost,
		int f_print_as_permutation)
{
	int i, l;
	
	if (!f_has_strong_generators) {
		cout << "group_container::print_strong_generators "
				"no strong generators" << endl;
		exit(1);
	}
	ost << "group::print_strong_generators a group with tl=";
	Int_vec_print(ost, tl, A->base_len());
	l = SG->len;
	ost << " and with " << l << " strong generators" << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->Group_element->element_print_quick(SG->ith(i), ost);
		ost << endl;
		if (f_print_as_permutation) {
			A->Group_element->element_print_as_permutation(SG->ith(i), ost);
			ost << endl;
		}
	}
}

void group_container::print_strong_generators_with_different_action(
		std::ostream &ost, actions::action *A2)
{
	print_strong_generators_with_different_action_verbose(
			ost, A2, 0);
}

void group_container::print_strong_generators_with_different_action_verbose(
		std::ostream &ost,
		actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, l;
	int *Elt;
	
	if (f_v) {
		cout << "group_container::print_strong_generators_with_different_action_verbose" << endl;
	}
	if (!f_has_strong_generators) {
		cout << "group_container::print_strong_generators_with_different_"
				"action no strong generators" << endl;
		exit(1);
	}
	ost << "group_container::print_strong_generators_with_different_"
			"action_verbose a group with tl=";
	Int_vec_print(ost, tl, A->base_len());
	l = SG->len;
	ost << " and with " << l << " strong generators" << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->Group_element->element_print_quick(SG->ith(i), ost);
		ost << endl;
		Elt = SG->ith(i);
		if (f_vv) {
			if (f_v) {
				cout << "group_container::print_strong_generators_with_"
						"different_action_verbose computing images "
						"individually" << endl;
			}
			int j; //, k;
			for (j = 0; j < A2->degree; j++) {
				//cout << "group::print_strong_generators_with_"
				//"different_action_verbose  computing image of "
				// << j << endl;
				/*k =*/ A2->Group_element->element_image_of(j, Elt,
						0 /*verbose_level - 2*/);
				//cout << "group::print_strong_generators_with_"
				//"different_action_verbose  image of "
				// << j << " is " << k << endl;
			}
		}
		ost << "as permutation in action " << A2->label
				<< " of degree " << A2->degree << ":" << endl;
		A2->Group_element->element_print_as_permutation_verbose(Elt,
				ost, 0/*verbose_level - 2*/);
		ost << endl;
	}
}

}}}

