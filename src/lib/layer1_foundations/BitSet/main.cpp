#include "bitset.h"
#include <iostream>
#include <bitset>

using std::cout;
using std::endl;

#define __INDENT__(n) {for (size_t i=0; i<n; ++i) cout << " ";}
#define __PRINT_LINE__(n) {for (size_t i=0; i<n; ++i) cout << "-"; printf("\n");}
#define print_bitset(bs) { \
    std::cout << bs.to_string<char,std::string::traits_type,std::string::allocator_type>() << '\n';\
}

#include "bitset_not_operator_test.h"

void  test_rshift () {
    bitset bs (64), org (64);
    bs.set_value(202335968067588);
    org.set_value(202335968067588);

    size_t N = 3;

    bs.rshift(N);

    org.print();
    bs.print();

    // automatic shift checking
    bool pass = true;
    for (size_t i=N; i<org.size(); ++i) {
        if (org[i] != bs[i-N]){
            cout << org[i] << " != " << bs[i-N] << endl;
            pass = false;
            break;
        }
    }
    if (pass) {
        cout << BRIGHT_GREEN << "right shift passed!" << RESET_COLOR_SCHEME;
    } else {
        cout << BRIGHT_RED << "right shift failed!" << RESET_COLOR_SCHEME;
    } cout << endl;
}

int main (int argc, char** argv) {
    bitset bs (64), org (64);
    auto val = -1;
    bs.set_value(val);
    bs.unset(12);

    if (bs.to_uint64()+1 == (++bs).to_uint64()) {
        cout << "Passed!" << endl;
    } else {
        cout << "Failed!" << endl;
    }

    if (bs.to_uint64()-1 == (--bs).to_uint64()) {
        cout << "Passed!" << endl;
    } else {
        cout << "Failed!" << endl;
    }
}
