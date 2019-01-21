// memory.C
//
// Anton Betten
// 04.04.2000
// moved from D2 to ORBI Nov 15, 2007

#include "orbiter.h"


namespace orbiter {

#undef MEMORY_COPY_VERBOSE
#undef DEBUG_MEM
#undef DEBUG_WRITE_CHAR
#undef DEBUG_WRITE_int

#define MEM_OVERSIZE 32
#define MEM_OVERSIZE1 512

/*
 * int - offset - 3 + 0: alloc_length
 *              - 3 + 1: used_length
 *              - 3 + 2: cur_pointer
 * 
 * alloc_length: allocated length in chars
 * used_length: used length in charS
 * cur_pointer:
 *          0 <= pointer < used length. 
 */



memory::memory()
{
	k = MEMORY;
	self.char_pointer = NULL;
}

memory::memory(const discreta_base &x)
	// copy constructor:    this := x
{
	// cout << "memory::copy constructor for object: " << const_cast<discreta_base &>(x) << "\n";
	clearself();
	const_cast<discreta_base &>(x).copyobject_to(*this);
}

memory& memory::operator = (const discreta_base &x)
	// copy assignment
{
	// cout << "memory::operator = (copy assignment)" << endl;
	copyobject(const_cast<discreta_base &>(x));
	return *this;
}

void memory::settype_memory()
{
	OBJECTSELF s;
	
	s = self;
	new(this) memory;
	self = s;
	k = MEMORY;
}

memory::~memory()
{
	// cout << "memory::~memory()\n";
	freeself_memory();
}

void memory::freeself_memory()
{

	char *pc;
	int *pi;
	
	if (self.char_pointer == NULL) {
		// cout << "returning\n";
		return;
		}
	if (s_kind() != MEMORY) {
		cout << "memory::freeself(): kind != MEM" << endl;
		exit(1);
		}
	// cout << "memory::freeself_memory():"; cout << *this << endl;
	pc = self.char_pointer;
	if (pc) {
		pi = (int *) pc;
		pi -= 3;
		delete [] pi;
		self.char_pointer = NULL;
		}
}

kind memory::s_virtual_kind()
{
	return MEMORY;
}

void memory::copyobject_to(discreta_base &x)
{
	int l;
	
#ifdef MEMORY_COPY_VERBOSE
	cout << "in memory::copyobject_to()\n";
#endif
	x.freeself();
	if (x.s_kind() != MEMORY) {
#ifdef MEMORY_CHANGE_KIND_VERBOSE
		cout << "warning: memory::copyobject_to x not a memory\n";
#endif
		x.c_kind(MEMORY);
		// x.printobjectkindln();
		}
	memory m = x.as_memory();
	
	l = used_length();
	m.init(l, self.char_pointer);
	m.cur_pointer() = cur_pointer();
}

ostream& memory::print(ostream& ost)
{
	if (self.char_pointer == NULL) {
		ost << "memory not allocated";
		}
	cout << "memory, used_length=" << used_length() 
		<< ", alloc_length=" << alloc_length() 
		<< ", cur_pointer=" << cur_pointer() << endl;
	return ost;
}

void memory::init(int length, char *d)
{
	int i;
	
	alloc(length);
	for (i = 0; i < length; i++) {
		s_i(i) = d[i];
		}
}

void memory::alloc(int length)
// sets alloc_length to length + MEM_OVERSIZE, 
// sets used_length to length, 
// sets cur_pointer to 0.
{
	int size, mem_oversize;
	int *pi;
	
#ifdef DEBUG_MEM
	cout << "memory::alloc()" << endl;
#endif
	if (length >= MEM_OVERSIZE) {
		mem_oversize = MEM_OVERSIZE1;
		}
	else {
		mem_oversize = MEM_OVERSIZE;
		}
	freeself_memory();
	size = length + mem_oversize + 3 * sizeof(int);
#ifdef DEBUG_MEM
	cout << "memory::alloc() allocating " << size << " chars" << endl;
#endif

	pi = (int *) new char[size];
	if (pi == NULL) {
		cout << "memoy::alloc() out of memory" << endl;
		exit(1);
		}
	self.char_pointer = (char *) (pi + 3);
#ifdef DEBUG_MEM
	cout << "memory::alloc() setting alloc_length()" << endl;
#endif
	alloc_length() = length + mem_oversize;
	used_length() = length;
	cur_pointer() = 0;
	c_kind(MEMORY);
#ifdef DEBUG_MEM
	cout << "memory::alloc() " << used_length() << " chars allocated." << endl;
#endif
}

void memory::append(int length, char *d)
{
	char *pc;
	int i, old_length, new_length;
	
	old_length = used_length();
	new_length = old_length + length;
	if (new_length > alloc_length()) {
		realloc(new_length);
		}
	else {
		used_length() = new_length;
		}
	pc = self.char_pointer;
	for (i = 0; i < length; i++) {
		pc[old_length + i] = d[i];
		}
}

void memory::realloc(int new_length)
{
	int old_length;
	int old_cur_pointer;
	int i;
	char *old_pc, *pc;
	int *old_pi;
	
	old_pc = self.char_pointer;
	old_pi = (int *)old_pc - 3;
	old_length = used_length();
	old_cur_pointer = cur_pointer();
	if (new_length < old_length) {
		cout << "memory::realloc() warning: new_length < old_length" << endl;
		}
	self.char_pointer = NULL;
	alloc(new_length);
	pc = self.char_pointer;
	for (i = 0; 
		i < MINIMUM(old_length, new_length); 
		i++) {
		pc[i] = old_pc[i];
		}
	for (i = old_length; i < new_length; i++) {
		pc[i] = 0;
		}
	delete [] old_pi;
	cur_pointer() = old_cur_pointer;
#ifdef DEBUG_MEM
	cout << "memory::realloc() to " << used_length() << " chars" << endl;
#endif
}

void memory::write_char(char c)
{	
#ifdef DEBUG_WRITE_CHAR
	cout << "memory::write_char(), at " << used_length() << ", writing char " << (int) c << endl;
#endif
	append(1, &c);
}

void memory::read_char(char *c)
{
	int l1, cur_p, l;
	char *cp;
	
	cur_p = cur_pointer();
	l = used_length();
	l1 = l - cur_p;
	if (l1 < 1) {
		cout << "memory::read_char() error: l1 < 1" << endl;
		exit(1);
		}
	cp = self.char_pointer + cur_p;
	*c = *cp;
#ifdef DEBUG_WRITE_CHAR
	cout << "memory::read_char(), at " << cur_pointer() << ", reading char " << (int) c << endl;
#endif
	cur_pointer()++;
}

void memory::write_int(int i)
{
	int_4 i1 = (int_4) i;
	
#ifdef DEBUG_WRITE_int
	cout << "memory::write_int(), at " << used_length() << ", writing int " << i1 << endl;
#endif
	block_swap_chars((char *) &i1, 4, 1);
	append(4, (char *) &i1);
}

void memory::read_int(int *i)
{
	int_4 i1;
	int l1, j, cur_p, l;
	char *cp, *cp1;
	
	cur_p = cur_pointer();
	l = used_length();
	l1 = l - cur_p;
	if (l1 < 4) {
		cout << "memory::read_int() error: l1 < 4\n";
		exit(1);
		}
	cp = self.char_pointer + cur_p;
	cp1 = (char *) &i1;
	for (j = 0; j < 4; j++) {
		*cp1 = *cp;
		cp1++;
		cp++;
		}
	/* i1 = *(int *) (cp + cur_p); */
	block_swap_chars((char *) &i1, 4, 1);
#ifdef DEBUG_WRITE_int
	cout << "memory::read_int(), at " << cur_pointer() << ", reading " << i1 << endl;
#endif
	cur_pointer() = cur_p + 4;
	*i = (int) i1;
}

#include <cstdio>

void memory::read_file(char *fname, int f_v)
{
	FILE *fp;
	int fsize;
	char *pc;

	fsize = file_size(fname);
	alloc(fsize);
	pc = self.char_pointer;
	fp = fopen(fname, "r");
	if ((int) fread(pc, 1 /* size */, fsize /* nitems */, fp) != fsize) {
		cout << "memory::read_file() error in fread" << endl;
		}
	fclose(fp);
	used_length() = fsize;
	cur_pointer() = 0;
	if (f_v) {
		cout << "memory::read_file() read file " 
			<< fname << " of size " << fsize << endl;
		}
}

void memory::write_file(char *fname, int f_v)
{
	FILE *fp;
	int size;
	char *pc;

	size = used_length();
	pc = self.char_pointer;
	
	fp = fopen(fname, "wb");

	fwrite(pc, 1 /* size */, size /* items */, fp);
	
	fclose(fp);
	if (file_size(fname) != size) {
		cout << "memory::write_file() error: file_size(fname) != size\n";
		exit(1);
		}
	if (f_v) {
		cout << "memory::write_file() wrote file " 
			<< fname << " of size " << size << endl;
		}
}

int memory::multiplicity_of_character(char c)
{
	char *pc;
	int i, l = 0, len;
	
	pc = self.char_pointer;
	len = used_length();
	for (i = 0; i < len; i++)
		if (pc[i] == c)
			l++;
	return l;
}

static void code(uchar *pc, int l, uchar *pc2, uchar code_char);
static int decode(uchar *pc2, int l2, uchar *pc, uchar code_char);

static void code(uchar *pc, int l, uchar *pc2, uchar code_char)
/* Wolfgang Boessenecker 940919 */
{
	uchar cc;
	int pos = 0, pos2 = 0, pos2h = 0, i;

	while (pos < l) {
		pos2++;
		cc = 0;
#if 0
		if ((posf % 100000) == 0) {
			cout << posf << endl;
			}
#endif
		for (i = 0; i < 8; i++) {
			cc <<= 1;
			if (pos < l) {
				if (pc[pos] == code_char)
					cc = cc | 0X1U;
				else {
					pc2[pos2] = pc[pos];
					pos2++;
					}
				pos++;
				}
			}
		pc2[pos2h] = cc;
		pos2h = pos2;
		}
}

static int decode(uchar *pc2, int l2, uchar *pc, uchar code_char)
// returns length of decompressed data 
// pc may be NULL */
{
	uchar cc = 0;
	int pos = 0, pos2 = 0, i = 8;
	
	while (TRUE) {
	/* for (; pos2 < l2; ) { */
		if (pos2 >= l2 && i >= 8)
			break;
		if (i == 8) {
			cc = pc2[pos2];
			pos2++;
			i = 0;
			}
		if (cc & (uchar) 128U) {
			if (pc) {
				pc[pos] = code_char;
				}
			pos++;
			}
		else {
			if (pos2 < l2) {
				if (pc) {
					pc[pos] = pc2[pos2];
					}
				pos2++;
				pos++;
				}
			}
		cc <<= 1;
		i++;
		}
	return pos;
}

void memory::compress(int f_v)
// Wolfgang Boessenecker 9/94 
{
	memory mem2;
	char *pc, *pc2;
	int l, l2, l_c;

	pc = self.char_pointer;
	l = used_length();
	if (f_v) {
		cout << "compressing from " << l << " to ";
		}
	l_c = multiplicity_of_character((char) 0);
	l2 = l - l_c + ((l + 7) >> 3);
	mem2.alloc(l2); // sets used_length to l2 
	pc2 = mem2.self.char_pointer;
	code((uchar *) pc, l, (uchar *) pc2, (uchar) 0);
#if 0
	if (l3 != l2) {
		cout << "memory::compress() warning: l2 = " << l2 << " != l3 = " << l3 << endl;
		}
#endif
	swap(mem2);
	if (f_v) {
		cout << l2 << " chars." << endl;
		print(cout);
		}
}

void memory::decompress(int f_v)
{
	memory mem;
	char *pc, *pc2;
	int l, l2;
	
	pc2 = self.char_pointer;
	l2 = used_length();
	if (f_v) {
		cout << "decompressing from " << l2 << " to ";
		}
	l = decode((uchar *) pc2, l2, NULL, (uchar) 0);
	mem.alloc(l);
	pc = mem.self.char_pointer;
	decode((uchar *) pc2, l2, (uchar *) pc, (uchar) 0);
	swap(mem);
	if (f_v) {
		cout << l << " chars." << endl;
		}
}

int memory::csf()
{
	int l;
	int size = 0;
	
	l = used_length();
	size += 4; /* l */
	size += l;
	return size;
}

void memory::write_mem(memory & M, int debug_depth)
{
	int i, l, a;
	
	l = used_length();
	M.write_int(l);
	for (i = 0; i < l; i++) {
		a = s_i(i);
		M.write_char((char) a);
		}
}

void memory::read_mem(memory & M, int debug_depth)
{
	int i, l;
	char c;
	char *mem;
	
	M.read_int(&l);	
	mem = new char[l];
	
	for (i = 0; i < l; i++) {
		M.read_char(&c);
		mem[i] = c;
		}
	M.init(l, mem);
	delete [] mem;
}

}


