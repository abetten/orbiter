#include "bitset.h"
#include "bitset_test_values.h"
#include "ansi_color.h"

#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;

#ifndef _BITSET_NOT_OPERATOR_TEST_H_
#define _BITSET_NOT_OPERATOR_TEST_H_

#define randint(min, max) rand() % max + min
#define NEW_LINE cout << endl;

template <typename T>
void test_bitset_not_operator_helper(T val) {
    size_t nbit = sizeof(T)*8;
    cout << "   Testing with " << std::right << std::setw(3) << nbit << " bits." ;
    cout << std::right << std::setw(5) << "[" ;
    cout << std::right << std::setw(20) << std::to_string(val);
    cout << std::left << std::setw(5) << "]";
    


    bool all_pass = true;
    for (size_t bitSize=1; bitSize<8192; ++bitSize) {
        bitset bs1 (bitSize), bs2 (bitSize);
        
        bs1.set_value(val);
        bs2.set_value(val);
        bs1._NOT_();

        size_t l = ( bitSize < sizeof(T) * 8 ) ? bitSize : sizeof(T) * 8;

        for (size_t i=0; i<l; ++i) {
            if (bs1[i] != !bs2[i]) {
                all_pass = false;
                cout << BRIGHT_RED;
                cout << endl << "\t";
                cout << "Testing with " << bitSize << " bit(s)";
                cout << " gave the following:" << endl;
                cout << "\tExpected: ";
                bs2.print();
                cout << "\tReceived: ";
                bs1.print();
                cout << RESET_COLOR_SCHEME;
                break;
            }
        }

        if (!all_pass) {
            cout << endl ;
            cout << "\tFailure encountered, aborting test." ;
            break;
        }
    }
    
    
    if (all_pass) {
        cout << BRIGHT_GREEN ;
        cout << "[ All tests passed! ]";
    } else {
        cout << BRIGHT_RED ;
        cout << "[ Some tests failed. ]";
    }
    cout << RESET_COLOR_SCHEME << endl;
}


void test_bitset_not_operator() {
    NEW_LINE;

    cout << "Testing bitset NOT logical operator" << endl;
    cout << "---------------------------------------------------------------------------------" << endl;

    test_bitset_not_operator_helper(val_int8_min);
    test_bitset_not_operator_helper(val_int8_max);
    test_bitset_not_operator_helper(val_uint8_min);
    test_bitset_not_operator_helper(val_uint8_max);

    cout << endl;

    test_bitset_not_operator_helper(val_int16_min);
    test_bitset_not_operator_helper(val_int16_max);
    test_bitset_not_operator_helper(val_uint16_min);
    test_bitset_not_operator_helper(val_uint16_max);

    cout << endl;

    test_bitset_not_operator_helper(val_int32_min);
    test_bitset_not_operator_helper(val_int32_max);
    test_bitset_not_operator_helper(val_uint32_min);
    test_bitset_not_operator_helper(val_uint32_max);

    cout << endl;

    test_bitset_not_operator_helper(val_int64_min);
    test_bitset_not_operator_helper(val_int64_max);
    test_bitset_not_operator_helper(val_uint64_min);
    test_bitset_not_operator_helper(val_uint64_max);

    cout << "---------------------------------------------------------------------------------" << endl;

    NEW_LINE;
}


#endif