// representatives.cpp
// 
// Anton Betten
// started July 3, 2012
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {


representatives::representatives()
{
	A = NULL;

	//std::string prefix;
	//std::string fname_rep;
	//std::string fname_stabgens;
	//std::string fname_fusion;
	//std::string fname_fusion_ge;

	nb_objects = 0;
	fusion = NULL;
	handle = NULL;

	count = 0;
	rep = NULL;
	stab = NULL;

	Elt1 = NULL;
	tl = NULL;

	nb_open = 0;
	nb_reps = 0;
	nb_fused = 0;
}




representatives::~representatives()
{
	int i;
	int f_v = TRUE;

	if (f_v) {
		cout << "representatives::free" << endl;
		}
	if (rep) {
		FREE_int(rep);
		rep = NULL;
		}
	if (stab) {
		for (i = 0; i < count; i++) {
			if (stab[i]) {
				FREE_OBJECT(stab[i]);
				stab[i] = NULL;
				}
			}
		delete [] stab;
		stab = NULL;
		}
	if (fusion) {
		FREE_int(fusion);
		fusion = NULL;
		}
	if (handle) {
		FREE_int(handle);
		handle = NULL;
		}
	if (Elt1) {
		FREE_int(Elt1);
		Elt1 = NULL;
		}
	if (tl) {
		FREE_int(tl);
		tl = NULL;
		}
}

void representatives::init(actions::action *A,
		int nb_objects, std::string &prefix, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "representatives::init prefix=" << prefix << endl;
		cout << "representatives::init nb_objects=" << nb_objects << endl;
		}
	representatives::A = A;
	representatives::nb_objects = nb_objects;
	
	representatives::prefix.assign(prefix);

	
	if (f_v) {
		cout << "representatives::init before allocating things" << endl;
	}
	rep = NEW_int(nb_objects);
	stab = new groups::psims[nb_objects];
	fusion = NEW_int(nb_objects);
	handle = NEW_int(nb_objects);
	Elt1 = NEW_int(A->elt_size_in_int);
	tl = NEW_int(A->base_len());

	count = 0;
	for (i = 0; i < nb_objects; i++) {
		stab[i] = NULL;
		fusion[i] = -2;
		handle[i] = -1;
		}

	if (f_v) {
		cout << "representatives::init before creating fnames" << endl;
	}
	fname_rep.assign(prefix);
	fname_rep.append("classification_reps.txt");

	//sprintf(fname_rep, "%sclassification_reps.txt", prefix);

	fname_stabgens.assign(prefix);
	fname_stabgens.append("classification_stabgens.bin");

	//sprintf(fname_stabgens, "%sclassification_stabgens.bin", prefix);

	fname_fusion.assign(prefix);
	fname_fusion.append("classification_fusion.txt");

	//sprintf(fname_fusion, "%sclassification_fusion.txt", prefix);

	fname_fusion_ge.assign(prefix);
	fname_fusion_ge.append("classification_fusion_ge.bin");

	//sprintf(fname_fusion_ge, "%sclassification_fusion_ge.bin", prefix);
	if (f_v) {
		cout << "representatives::init done" << endl;
		}
}

void representatives::write_fusion(int verbose_level)
// Writes fusion[] and handle[]
// If the object is a chosen representative for an isomorphism type 
// (i.e., if fusion[i] == i) then the identity element is written.
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "representatives::write_fusion" << endl;
		}
	if (f_v) {
		cout << "representatives::write_fusion fname_fusion=" << fname_fusion << endl;
		}
	{
	ofstream f1(fname_fusion);
	int i;

	ofstream f2(fname_fusion_ge, ios::binary);
	//FILE *f2;
	//f2 = fopen(fname_fusion_ge, "wb");
	
	for (i = 0; i < nb_objects; i++) {
		if (fusion[i] == -2) {
			cout << "representatives::write_fusion "
					"fusion[" << i << "] = -2" << endl;
			exit(1);
			}
		f1 << setw(5) << i << " " << setw(3) << fusion[i] << endl;
		if (fusion[i] == i) {
			//cout << "orbit " << i << " is representative" << endl;
			A->one(Elt1);
			}
		else {
			A->element_retrieve(handle[i], Elt1, FALSE);
			}
		A->element_write_file_fp(Elt1, f2, 0/* verbose_level*/);
		}
	f1 << -1 << endl;
	//fclose(f2);
	}
	if (f_v) {
		cout << "representatives::write_fusion finished" << endl;
		cout << "written file " << fname_fusion << " of size "
				<< Fio.file_size(fname_fusion) << endl;
		cout << "written file " << fname_fusion_ge << " of size "
				<< Fio.file_size(fname_fusion_ge) << endl;
		}
	
}

void representatives::read_fusion(int verbose_level)
// Reads fusion[] and handle[]
{
	int f_v = (verbose_level >= 1);
	int a, b, i;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "representatives::read_fusion nb_objects="
				<< nb_objects << endl;
		}
	if (f_v) {
		cout << "representatives::read_fusion reading file "
				<< fname_fusion << " of size "
				<< Fio.file_size(fname_fusion) << endl;
		}

	if (Fio.file_size(fname_fusion) < 0) {
		cout << "representatives::read_fusion the file "
				<< fname_fusion << " does not exist" << endl;
		exit(1);
	}
	{
		ifstream f1(fname_fusion);
		for (i = 0; i < nb_objects; i++) {
			f1 >> a >> b;
			if (a != i) {
				cout << "representatives::read_fusion "
						"a != i" << endl;
				exit(1);
				}
			fusion[i] = b;
			}
		f1 >> a;
		if (a != -1) {
			cout << "representatives::read_fusion problem with end "
					"of file marker" << endl;
			exit(1);
			}
	}
	if (f_v) {
		cout << "representatives::read_fusion reading file "
				<< fname_fusion_ge << " of size "
				<< Fio.file_size(fname_fusion_ge) << endl;
		}
	{
		ifstream f2(fname_fusion_ge, ios::binary);
		//FILE *f2;
	
		//f2 = fopen(fname_fusion_ge, "rb");
	
		for (i = 0; i < nb_objects; i++) {
			A->element_read_file_fp(Elt1, f2, 0/* verbose_level*/);
			handle[i] = A->element_store(Elt1, FALSE);
			}
	}
	if (f_v) {
		cout << "representatives::read_fusion done" << endl;
		}
}

void representatives::write_representatives_and_stabilizers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "representatives::write_representatives_and_stabilizers" << endl;
		}
	{
	ofstream f1(fname_rep);
	int i, j, cnt = 0;

	ofstream f2(fname_stabgens, ios::binary);
	//FILE *f2;
	//f2 = fopen(fname_stabgens, "wb");
	
	
	f1 << count << " " << setw(3) << A->base_len() << " ";
	for (i = 0; i < A->base_len(); i++) {
		f1 << setw(3) << A->base_i(i) << " ";
		}
	f1 << endl;
	
	for (i = 0; i < count; i++) {
		groups::sims *Stab;
		ring_theory::longinteger_object go;
		data_structures_groups::vector_ge SG;
		
		Stab = stab[i];
		Stab->group_order(go);

		Stab->extract_strong_generators_in_order(
				SG, tl, 0 /* verbose_level */);

		f1 << setw(3) << i << " " 
			<< setw(7) << rep[i] << " " 
			<< setw(5) << cnt << " " 
			<< setw(5) << SG.len << " ";
		go.print_width(f1, 10);
		f1 << " ";
		for (j = 0; j < A->base_len(); j++) {
			f1 << setw(3) << tl[j] << " ";
			}
		f1 << endl;
		
		
		for (j = 0; j < SG.len; j++) {
			A->element_write_file_fp(SG.ith(j), f2,
					0/* verbose_level*/);
			cnt++;
			}
		}
	f1 << -1 << endl;
	//fclose(f2);
	}
	if (f_v) {
		cout << "representatives::write_representatives_and_stabilizers finished" << endl;
		cout << "written file " << fname_rep << " of size "
				<< Fio.file_size(fname_rep) << endl;
		cout << "written file " << fname_stabgens << " of size "
				<< Fio.file_size(fname_stabgens) << endl;
		}
	
}

void representatives::read_representatives_and_stabilizers(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >=2);
	
	if (f_v) {
		cout << "representatives::read_representatives_and_stabilizers" << endl;
		cout << "reading files " << fname_rep << " and "
				<< fname_stabgens << endl;
		}
	{
	ifstream f1(fname_rep);
	int i, j, /*first,*/ len, a, b, c, d, e;
	
	ifstream f2(fname_stabgens, ios::binary);
	//FILE *f2;
	//f2 = fopen(fname_stabgens, "rb");
	
	f1 >> count >> a;
	if (a != A->base_len()) {
		cout << "representatives::read_representatives_and_stabilizers "
				"base_len does not match" << endl;
		exit(1);
		}
	for (j = 0; j < A->base_len(); j++) {
		f1 >> a;
		if (a != A->base_i(j)) {
			cout << "representatives::read_representatives_and_stabilizers "
					"base point does not match" << endl;
			exit(1);
			}
		}
	for (i = 0; i < count; i++) {
		groups::sims *Stab;
		ring_theory::longinteger_object go;
		data_structures_groups::vector_ge gens;
		
		stab[i] = NEW_OBJECT(groups::sims);
		Stab = stab[i];
		f1 >> a >> b >> c >> d >> e;
		if (a != i) {
			cout << "representatives::read_representatives_and_stabilizers "
					"a != i" << endl;
			exit(1);
			}
		rep[i] = b;
		//first = c;
		len = d;
		gens.init(A, verbose_level - 2);
		gens.allocate(len, verbose_level - 2);
		for (j = 0; j < A->base_len(); j++) {
			f1 >> tl[j];
			}
		for (j = 0; j < len; j++) {
			A->element_read_file_fp(gens.ith(j), f2, 0/* verbose_level*/);
			}
		if (f_vv) {
			cout << "representative of orbit " << i << " read" << endl;
			cout << "stabilizer is generated by" << endl;
			for (j = 0; j < len; j++) {
				cout << "generator " << j << ":" << endl;
				A->print(cout, gens.ith(j));
				cout << endl;
				}
			cout << "transversal lengths:" << endl;
			Int_vec_print(cout, tl, A->base_len());
			cout << endl;
			}
		Stab->init(A, verbose_level - 2);
		Stab->init_generators(gens, FALSE);
		Stab->compute_base_orbits(0/*verbose_level - 5*/);
		Stab->group_order(go);
		if (f_v) {
			cout << "representatives::read_representatives_and_stabilizers "
					"stabilizer " << i << " has order " << go << endl;
			}
		}
	f1 >> a;
	if (a != -1) {
		cout << "representatives::read_representatives_and_stabilizers "
				"problems reading end of file marker" << endl;
		exit(1);
		}
	//fclose(f2);
	}
	if (f_v) {
		cout << "representatives::read_representatives_and_stabilizers "
				"finished" << endl;
		}
}

void representatives::save(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "representatives::save" << endl;
		}
	write_fusion(verbose_level - 1);
	write_representatives_and_stabilizers(verbose_level - 1);
	if (f_v) {
		cout << "representatives::save done" << endl;
		}
}

void representatives::load(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "representatives::load" << endl;
		}
	if (f_v) {
		cout << "representatives::load before read_fusion" << endl;
		}
	read_fusion(verbose_level - 1);
	if (f_v) {
		cout << "representatives::load after read_fusion" << endl;
		}
	if (f_v) {
		cout << "representatives::load before read_representatives_and_stabilizers" << endl;
		}
	read_representatives_and_stabilizers(verbose_level - 1);
	if (f_v) {
		cout << "representatives::load after read_representatives_and_stabilizers" << endl;
		}
	if (f_v) {
		cout << "representatives::load done found " << count
				<< " orbit representatives" << endl;
		}
}

void representatives::calc_fusion_statistics()
{
	int i;
	
	nb_open = 0;
	nb_reps = 0;
	nb_fused = 0;
	for (i = 0; i < nb_objects; i++) {
		if (fusion[i] == -2) {
			nb_open++;
			}
		if (fusion[i] == i) {
			nb_reps++;
			}
		else if (fusion[i] >= 0) {
			nb_fused++;
			}
		}
	
}

void representatives::print_fusion_statistics()
{
	cout << "nb_reps = " << nb_reps << endl;
	cout << "nb_fused = " << nb_fused << endl;
	cout << "nb_open = " << nb_open << endl;
}

}}


