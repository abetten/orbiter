// arcs_orderly.C
//
// Anton Betten
// October 23, 2017

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started

	INT verbose_level = 0;
	INT k;
	finite_field *F;
	projective_space *P;
	action *A_linear;
	INT *Arc;
	INT arc_sz;
	INT target_sz;
	INT *Idx_table; // [target_sz * P->N_points]
	INT *line_type; // [P->N_lines]
	INT *Line_type; // [target_sz * P->N_lines]
	INT *Nb_total; // [target_sz + 1]
	INT *Nb_canonical; // [target_sz + 1]
	INT *Nb_complete; // [target_sz + 1]
	INT *Nb_orbits; // [target_sz + 1]
	INT *Cur_orbit; // [target_sz + 1]
	INT *canonical_set;
	INT cnt;


int main(int argc, char **argv);
void do_arc_lifting(projective_space *P, INT k, 
	INT *arc, INT arc_sz, INT target_sz, INT verbose_level);
void extend(INT arc_size, INT verbose_level);

int main(int argc, char **argv)
{
	INT f_k = FALSE;
	INT f_sz = FALSE;
	INT sz = 0;
	INT f_q = FALSE;
	INT q = 0;
	INT f_poly = FALSE;
	const BYTE *poly = NULL;
	INT f_arc = FALSE;
	const BYTE *arc_text = NULL;
	INT f_fining = FALSE;
	INT i;

	t0 = os_ticks();
	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-sz") == 0) {
			f_sz = TRUE;
			sz = atoi(argv[++i]);
			cout << "-sz " << sz << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-arc") == 0) {
			f_arc = TRUE;
			arc_text = argv[++i];
			cout << "-arc " << arc_text << endl;
			}
		else if (strcmp(argv[i], "-fining") == 0) {
			f_fining = TRUE;
			cout << "-fining " << endl;
			}
		}
	
	if (!f_k) {
		cout << "please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_sz) {
		cout << "please use option -sz <sz>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	INT f_v = (verbose_level >= 1);
	
	cout << "k=" << k << endl;
	cout << "q=" << q << endl;
	cout << "poly=";
	if (f_poly) {
		cout << poly;
		}
	else {
		cout << endl;
		}
	

	if (f_v) {
		cout << "creating finite field:" << endl;
		}
	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0 /*verbose_level*/);


	if (f_v) {
		cout << "creating projective space PG(" << 2 << ", " << q << ")" << endl;
		}


	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "before P->init" << endl;
		}
	P->init(2, F, 
		FALSE /* f_init_incidence_structure */, 
		0 /*verbose_level - 2*/);

	P->init_incidence_structure(0 /*verbose_level*/);
	

	if (f_arc) {

		if (f_fining) {
			INT a;
			INT v[3];
			
			for (a = 1; a <= P->N_points; a++) {
				PG_element_unrank_fining(*P->F, v, 3, a);
				cout << a << " : ";
				INT_vec_print(cout, v, 3);
				cout << endl;
				}
			}

		INT *the_arc;
		INT the_arc_sz;
		
		INT_vec_scan(arc_text, the_arc, the_arc_sz);
		cout << "input arc of size " << the_arc_sz << " = ";
		INT_vec_print(cout, the_arc, the_arc_sz);
		cout << endl;


		if (f_fining) {
			INT a, b;
			INT v[3];
			INT w[3];
			
			cout << "changing from fining to orbiter:" << endl;
			for (i = 0; i < the_arc_sz; i++) {
				a = the_arc[i];
				PG_element_unrank_fining(*P->F, v, 3, a);
				INT_vec_copy(v, w, 3);
				PG_element_rank_modified(*P->F, w, 1, 3, b);
				cout << a << " : ";
				INT_vec_print(cout, v, 3);
				cout << " : " << b << endl;
				the_arc[i] = b;
				}
			cout << "input arc in orbiter labels = ";
			INT_vec_print(cout, the_arc, the_arc_sz);
			cout << endl;
			}

		INT_vec_heapsort(the_arc, the_arc_sz);

		cout << "input arc in orbiter labels sorted= ";
		INT_vec_print(cout, the_arc, the_arc_sz);
		cout << endl;

		
		
		do_arc_lifting(P, k, the_arc, the_arc_sz, sz, verbose_level);

		}
	else {
		do_arc_lifting(P, k, NULL, 0, sz, verbose_level);
		}

	FREE_OBJECT(P);
	FREE_OBJECT(F);

	the_end(t0);
}

void do_arc_lifting(projective_space *P, INT k, 
	INT *arc, INT arc_sz, INT target_sz, INT verbose_level)
{
	INT i;

	
	::target_sz = target_sz;
	::arc_sz = arc_sz;
	Arc = NEW_INT(target_sz);
	INT_vec_copy(arc, Arc, arc_sz);

	cout << "do_arc_lifting" << endl;
	F = P->F;
	
	Nb_total = NEW_INT(target_sz + 1);
	Nb_canonical = NEW_INT(target_sz + 1);
	Nb_complete = NEW_INT(target_sz + 1);
	Nb_orbits = NEW_INT(target_sz + 1);
	Cur_orbit = NEW_INT(target_sz + 1);
	INT_vec_zero(Nb_total, target_sz + 1);
	INT_vec_zero(Nb_canonical, target_sz + 1);
	INT_vec_zero(Nb_complete, target_sz + 1);
	INT_vec_zero(Nb_orbits, target_sz + 1);
	INT_vec_zero(Cur_orbit, target_sz + 1);
	
	INT f_semilinear;

	if (is_prime(F->q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}
	A_linear = NEW_OBJECT(action);
	A_linear->init_projective_group(P->n + 1,
			F, f_semilinear, TRUE /*f_basis */, 0 /*verbose_level*/);
	

	Idx_table = NEW_INT(target_sz * P->N_points);


	Line_type = NEW_INT(target_sz * P->N_lines);
	line_type = NEW_INT(P->N_lines);
	P->line_intersection_type(arc, arc_sz, line_type, 0 /* verbose_level */);
	cout << "line_type: ";
	INT_vec_print_fully(cout, line_type, P->N_lines);
	cout << endl;

#if 0
	cout << "line type:";
	for (i = 0; i < P->N_lines; i++) {
		cout << i << " : " << line_type[i] << endl;
		}
#endif

	canonical_set = NEW_INT(P->N_points);


	cnt = -1;
	extend(arc_sz, verbose_level);

	FREE_INT(canonical_set);

	for (i = 0; i <= target_sz; i++) {
		cout << setw(5) << i
			<< " : " << setw(5) << Nb_canonical[i]
			<< " : " << setw(5) << Nb_total[i]
			<< " : " << setw(5) << Nb_complete[i]
			<< endl;
		}
	cout << "number of search nodes = " << cnt << endl;

}


void extend(INT arc_size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	sims *Stab;
	longinteger_object go;
	INT i, pt, nb, orb, h, idx, line, f;
	INT *line_type_before;
	INT *line_type_after;
	INT *Idx;
	INT canonical_pt;

	cnt++;
	if (f_v) {
		cout << "Node " << cnt << " level " << arc_size << " : ";
		for (i = 0; i < arc_size; i++) {
			cout << Cur_orbit[i] << "/" << Nb_orbits[i] << " ";
		}
		cout << "; Arc=";
		INT_vec_print(cout, Arc, arc_size);
		cout << endl;
	}

	Idx = Idx_table + arc_size * P->N_points;
	
	Nb_total[arc_size]++;

	
	if (f_vv) {
		cout << "computing stabilizer of the arc:" << endl;
	}
	Stab = set_stabilizer_in_projective_space(
		A_linear, P, 
		Arc, arc_size, canonical_pt, canonical_set, 
		FALSE, NULL, 
		verbose_level - 2);
		// in ACTION/action_global.C
	if (f_v) {
		cout << "Node " << cnt << " level " << arc_size
			<< " The stabilizer of the arc has been computed" << endl;
		Stab->group_order(go);
		cout << "It is a group of order " << go << endl;
		cout << "canonical_pt = " << canonical_pt << endl;
	}
	
	//exit(1);
	
	strong_generators *gens;

	gens = NEW_OBJECT(strong_generators);

	
	gens->init_from_sims(Stab, 0 /* verbose_level */);

	schreier *Sch;
	

	Sch = gens->orbits_on_points_schreier(A_linear, 0 /* verbose_level */);

	if (arc_size > arc_sz) {
		if (Sch->orbit_representative(Arc[arc_size - 1]) != Sch->orbit_representative(canonical_pt)) {
			if (f_vv) {
				cout << "The flag orbit is not canonical, reject" << endl;
				cout << "canonical_pt=" << canonical_pt << endl;
				cout << "Arc[arc_size - 1]=" << Arc[arc_size - 1] << endl;
			}
			FREE_OBJECT(Sch);
			FREE_OBJECT(gens);
			FREE_OBJECT(Stab);
			return;
			}
		else {
			if (f_vv) {
				cout << "The flag orbit is canonical, accept" << endl;
			}
			Nb_canonical[arc_size]++;
			}

		}
	if (arc_size == arc_sz) {
		line_type_before = line_type;
		}
	else {
		line_type_before = Line_type + (arc_size - 1) * P->N_lines;
		}
	line_type_after = Line_type + arc_size * P->N_lines;



	nb = 0;
	for (orb = 0; orb < Sch->nb_orbits; orb++) {
		f = Sch->orbit_first[orb];
		pt = Sch->orbit[f];
		if (f_vv) {
			cout << "Node " << cnt << " level " << arc_size << " testing orbit " << orb
					<< " / " << Sch->nb_orbits << " pt=" << pt << " ";
		}
		if (INT_vec_search(Arc, arc_size, pt, idx)) {
			if (f_vv) {
				cout << "fail (already in the set)" << endl;
			}
			continue;
		}

		
		INT_vec_copy(line_type_before, line_type_after, P->N_lines);

		for (i = 0; i < F->q + 1; i++) {
			line = P->Lines_on_point[pt * P->k + i];
			line_type_after[line]++;
			if (line_type_after[line] > k) {
				break;
			}
		}
		if (i == F->q + 1) {
			if (f_vv) {
				cout << "is OK" << endl;
			}
			Idx[nb++] = orb;
		}
		else {
			if (f_vv) {
				cout << "fail (line type)" << endl;
			}
		}


		
	} // next orb

	if (nb == 0) {
		if (f_vv) {
			cout << "The arc is complete" << endl;
		}
		Nb_complete[arc_size]++;
	}

	if (arc_size == target_sz) {
		if (f_vv) {
			cout << "extend, arc_size == target_sz" << endl;
		}
		FREE_OBJECT(Sch);
		FREE_OBJECT(gens);
		FREE_OBJECT(Stab);
		return;
		}




	if (nb) {
		if (arc_size == 0 && P->N_lines == 273 && nb == 273) {
			nb = 1;
			cout << "overriding the value of nb to 1." << endl;
		}
		Nb_orbits[arc_size] = nb;
		for (h = 0; h < nb; h++) {
			Cur_orbit[arc_size] = h;
			if (f_vv) {
				cout << "Node " << cnt << " level "
						<< arc_size << " orbit " << h << " / " << nb << endl;
			}
			orb = Idx[h];
			f = Sch->orbit_first[orb];
			pt = Sch->orbit[f];
			Arc[arc_size] = pt;


			INT_vec_copy(line_type_before, line_type_after, P->N_lines);
			for (i = 0; i < F->q + 1; i++) {
				line = P->Lines_on_point[pt * P->k + i];
				line_type_after[line]++;
				if (line_type_after[line] > k) {
					cout << "fatal: line_type_after[line] > k" << endl;
					exit(1);
				}
			} // next i
			if (f_vv) {
				classify C;

				C.init(line_type_after, P->N_lines, FALSE, 0);
				cout << "Node " << cnt << " level " << arc_size << " orbit " << h << " / " << nb << " : ";
				cout << "line type: ";
				C.print(TRUE /*f_backwards*/);
				cout << endl;
			}


			extend(arc_size + 1, verbose_level);
		} // next h
	}

	if (f_v) {
		cout << "Node " << cnt << " level " << arc_size << " before FREE" << endl;
	}
	FREE_OBJECT(Sch);
	FREE_OBJECT(gens);
	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "Node " << cnt << " level " << arc_size << " after FREE" << endl;
	}
}

