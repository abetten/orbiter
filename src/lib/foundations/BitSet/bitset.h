/*
 *  This bitset class stores bits in a bit array. Element
 *  zero of the bit array are the least significant bits,
 *  while the last element of the bit array is the most
 *  significant bit.
 *
 *  The print function prints the bits of the elements of
 *  the bit array reading from the last element of the bit
 *  array to the first.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <alloca.h>
#include <math.h>

using std::cout;
using std::endl;

#ifndef _BITSET_H_
#define _BITSET_H_


#define __print__bits__main__(t, l) {\
    size_t i##l=0,k##l = sizeof(t)*8; \
    for (; i##l < k##l ; ++i##l) {\
        printf("%d",bool(t & (1UL << (k##l -1 - i##l))));\
    }\
    printf("\n");\
}
#define _print_bits_(t) __print__bits__main__(t, __LINE__)

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif


class bitset {

    typedef uint32_t TYPE;

    size_t nb_bits_ = 0; /* Number of bits in the bitset */
    size_t nb_bytes_ = 0; /* Number of bytes in the bit array */
    TYPE* bit_array = NULL; /* Array used to store the bits */
    size_t bit_array_size = 0; /* Number of elts. in the bit array */
    const size_t WORD_LEN = sizeof(TYPE) * 8;
    bool stack_storage = false;

    /*------------------------------------------------------------------------*/

    __forceinline__ void increment (void) {
    	for (size_t i=0; i<bit_array_size; ++i) {
			bit_array[i] += 1;
			if (bit_array[i] != 0) return;
		}
    }

    __forceinline__ void increment_uint8_t (const uint8_t rhs) {
    	printf("%s : %ld : Needs Implementation\n", __FILE__, __LINE__);
    	exit (-1);
    }

    __forceinline__ void increment_uint16_t (const uint16_t rhs) {
		printf("%s : %ld : Needs Implementation\n", __FILE__, __LINE__);
		exit (-1);
	}

    __forceinline__ void increment_uint32_t (const uint32_t rhs) {
		printf("%s : %ld : Needs Implementation\n", __FILE__, __LINE__);
		exit (-1);
	}

    __forceinline__ void increment_uint64_t (const uint64_t rhs) {
		printf("%s : %ld : Needs Implementation\n", __FILE__, __LINE__);
		exit (-1);
	}

    __forceinline__ void increment (const int8_t rhs)  { increment_uint8_t(rhs); }
    __forceinline__ void increment (const int16_t rhs) { increment_uint16_t(rhs); }
    __forceinline__ void increment (const int32_t rhs) { increment_uint32_t(rhs); }
    __forceinline__ void increment (const int64_t rhs) { increment_uint64_t(rhs); }

    __forceinline__ void increment (const bitset& rhs) {
    	printf("%s : %ld : Needs Implementation\n", __FILE__, __LINE__);
		exit (-1);
    }

    /*------------------------------------------------------------------------*/

public:

#if defined(__clang__)
    size_t _MAX_STACK_ALLOC_LIMIT = 0; /* bits */
#elif defined(__GNUC__) || defined(__GNUG__)
    size_t _MAX_STACK_ALLOC_LIMIT = 32; /* bits */
#elif defined(_MSC_VER)
    size_t _MAX_STACK_ALLOC_LIMIT = 0; /* bits */
#endif

    __forceinline__ bitset () {};

    __forceinline__ bitset (size_t nb_bits) {
        init(nb_bits);
    }

    __forceinline__ bitset (const bitset& rhs) {
        nb_bits_ = rhs.nb_bits_;
        bit_array_size = rhs.bit_array_size;
        nb_bytes_ = sizeof(TYPE) * bit_array_size;

        bit_array = new TYPE [bit_array_size] ();

        // perform the copy operation
        memcpy (bit_array, rhs.bit_array, nb_bytes_) ;
    }

    __forceinline__ ~bitset () {
        if (bit_array && !stack_storage) {
            delete [] bit_array;
            bit_array = NULL;
        }
    }

    __forceinline__
    inline void init (size_t nb_bits) {
    	nb_bits_ = nb_bits;
		bit_array_size = 1 + nb_bits / WORD_LEN;
		nb_bytes_ = sizeof(TYPE) * bit_array_size;

		if (bit_array_size <= _MAX_STACK_ALLOC_LIMIT/WORD_LEN) {
			bit_array = (TYPE*)alloca(nb_bytes_);
			memset(bit_array, 0, nb_bytes_);
			stack_storage = true;
		} else {
			bit_array = new TYPE [bit_array_size] ();
			stack_storage = false;
		}
    }


    inline void set (size_t i) {
        bit_array[i/WORD_LEN] |=  (1 << (i % WORD_LEN));
    }

    template <typename T>
    inline void set_value (T val) {
        _OR_(val);
    }

    inline void unset (size_t i) {
        bit_array[i/WORD_LEN] &= ~(1 << (i % WORD_LEN));
    }

    inline void reset () {
        memset(bit_array, 0, sizeof(TYPE)*bit_array_size);
    }

    inline void twos_complement (void) {
        _NOT_();
        increment();
    }

    /* Convert to builtin integer types */
    inline int8_t   to_int8   () { return int8_t(bit_array[0]);                        }
    inline uint8_t  to_uint8  () { return uint8_t(bit_array[0]);                       }
    inline int16_t  to_int16  () { return int16_t(bit_array[0]);                       }
    inline uint16_t to_uint16 () { return uint16_t(bit_array[0]);                      }
    inline int32_t  to_int32  () { return int32_t(bit_array[0]);                       }
    inline uint32_t to_uint32 () { return uint32_t(bit_array[0]);                      }
    inline int64_t  to_int64  () { return int64_t(bit_array[1]) << 32 | bit_array[0];  }
    inline uint64_t to_uint64 () { return uint64_t(bit_array[1]) << 32 | bit_array[0]; }


    /* ----------------------------------------------------------------- */
    /*  Overloaded Operators                                             */
    /* ----------------------------------------------------------------- */


    inline bitset& operator>> (const size_t i) {

        return *this;
    }

    inline bitset& operator<< (const size_t i) {
        return *this;
    }

    /*------------------------------------------------------------------------*/
    /* Assignment Operators */
    /*------------------------------------------------------------------------*/
    template <typename T>
    inline bitset& operator= (const T& rhs) {
        reset();
        _OR_(rhs);
        return *this;
    }

    template <typename T>
    inline bitset& operator+= (const T& rhs);

    template <typename T>
    inline bitset& operator-= (const T& rhs);

    template <typename T>
    inline bitset& operator*= (const T& rhs);

    template <typename T>
    inline bitset& operator/= (const T& rhs);

    template <typename T>
    inline bitset& operator%= (const T& rhs);

    template <typename T>
    inline bitset& operator&= (const T& rhs) {
        _AND_(rhs);
        return *this;
    }

    template <typename T>
    inline bitset& operator|= (const T& rhs) {
        _OR_(rhs);
        return *this;
    }

    template <typename T>
    inline bitset& operator^= (const T& rhs);


    inline bitset& operator<<= (const bitset& rhs);
    inline bitset& operator<<= (const size_t rhs);

    inline bitset& operator>>= (const bitset& rhs);
    inline bitset& operator>>= (const size_t N);
    /*------------------------------------------------------------------------*/

    /*------------------------------------------------------------------------*/
    /* Increment/Decrement Operators */
    /*------------------------------------------------------------------------*/
    inline bitset& operator++ () {
        // pre-increment

        // TYPE* x = bit_array;
        // size_t N = bit_array_size;
        // for (uint64_t i=0, z=1UL; i<N; ++i) {
        //     z += uint64_t(x[i]);
        //     x[i] = z;
        //     z >>= 32;
        // }

        this->increment();
        
        return *this;
    }

    inline bitset& operator-- () {
        // pre-decrement
        TYPE* x = bit_array;
        size_t N = bit_array_size;
        for (uint64_t i=0, z=0; i<N; ++i) {
            z += uint64_t(TYPE(-1)) + uint64_t(x[i]);
            x[i] = z;
            z >>= 32;
        }
        return *this;
    }
    
    inline bitset operator++ (int) {
        // post-increment
        return *this;
    }
    
    inline bitset operator-- (int) {
        // post-decrement
        return *this;
    }
    /*------------------------------------------------------------------------*/

    /*------------------------------------------------------------------------*/
    /* Arithmetic Operators */
    /*------------------------------------------------------------------------*/
    inline bitset operator+ () const;
    inline bitset operator- () const;

    template <typename T>
    inline bitset operator+ (const T& rhs) const;

    template <typename T>
    inline bitset operator- (const T& rhs) const;

    template <typename T>
    inline bitset operator* (const T& rhs) const;

    template <typename T>
    inline bitset operator/ (const T& rhs) const;

    template <typename T>
    inline bitset operator% (const T& rhs) const;

    inline bitset operator~ () const;

    template <typename T>
    inline bitset operator& (const T& rhs) const;

    template <typename T>
    inline bitset operator| (const T& rhs) const;

    template <typename T>
    inline bitset operator^ (const T& rhs) const;

    template <typename T>
    inline bitset operator<< (const T& rhs) const;

    template <typename T>
    inline bitset operator>> (const T& rhs) const;
    /*------------------------------------------------------------------------*/

    /*------------------------------------------------------------------------*/
    /* Logical Operators */
    /*------------------------------------------------------------------------*/
    inline bool operator! () const;

    template <typename T>
    inline bool operator&& (const T& rhs) const;
   
    template <typename T>
    inline bool operator|| (const T& rhs) const;
    /*------------------------------------------------------------------------*/

    /*------------------------------------------------------------------------*/
    /* Comparison Operators */
    /*------------------------------------------------------------------------*/

    /* is strictly equal to */
    template <typename T>
    inline bool operator== (const T& rhs) const {
        const size_t NB_BITS_RHS = sizeof(rhs)*8;

        if (NB_BITS_RHS != nb_bits_) return false;

        for (size_t i=0; i<NB_BITS_RHS; ++i) {
            if (at(i) != bool(rhs & (T(1) << i)))
                return false;
        }

        return true;
    }

    template <typename T>
    inline bool operator!= (const T& rhs) const;

    template <typename T>
    inline bool operator< (const T& rhs) const;

    template <typename T>
    inline bool operator> (const T& rhs) const;

    template <typename T>
    inline bool operator<= (const T& rhs) const;

    template <typename T>
    inline bool operator>= (const T& rhs) const;
    /*------------------------------------------------------------------------*/

    /* ----------------------------------------------------------------- */

    /* Check equality upto a certain number of bits */
    template <typename T>
    inline bool isEqualTo (const T& rhs, size_t len /* bits */) {
        for (size_t i=0; i<len; ++i) {
            if (at(i) != bool(rhs & (T(1) << i)))
                return false;
        }
        return true;
    }

    template <typename t>
    inline void _OR_ (const t& rhs) {
        const size_t NB_BITS_RHS = (sizeof(rhs)*8);
        const bool   SIGN_BIT_SET = signbit(rhs);
        size_t EXT_IDX = 0; // keep track of what elt. of the bitarray
                            // to start sign extending from.

        if (NB_BITS_RHS % WORD_LEN == 0) { /* for {32, 64}%32 */

            // k keeps track of the nb_bits of rhs processed
            for (size_t i=0,k=0; i<bit_array_size && k<NB_BITS_RHS; ++i,k+=WORD_LEN) {
                bit_array[i] |= rhs >> i*WORD_LEN;
                EXT_IDX = i+1;
            }

        } else if (NB_BITS_RHS % WORD_LEN < WORD_LEN) {

            bit_array[0] |= TYPE(rhs);
            EXT_IDX = 1;

        }

        if (SIGN_BIT_SET) {
            for (size_t i=EXT_IDX; i<bit_array_size; ++i) {
                bit_array[i] |= TYPE(-1);
            }
        }

        bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
        // bit_array[bit_array_size-1] &= TYPE(pow(2, nb_bits_ % WORD_LEN)-1);

    }

    template <typename t>
    inline void _AND_ (const t& rhs) {
        const size_t NB_BITS_RHS = (sizeof(rhs)*8);
        const bool   SIGN_BIT_SET = signbit(rhs);
        size_t EXT_IDX = 0;

        if (NB_BITS_RHS % WORD_LEN == 0) { /* for {32, 64}%32 */

            // k keeps track of the nb_bits of rhs processed
            for (size_t i=0,k=0; i<bit_array_size && k<NB_BITS_RHS; ++i,k+=WORD_LEN) {
                bit_array[i] &= rhs >> i*WORD_LEN;
                EXT_IDX = i+1;
            }

        } else if (NB_BITS_RHS % WORD_LEN < WORD_LEN) {

            bit_array[0] &= TYPE(rhs);
            EXT_IDX = 1;

        }

        if (!SIGN_BIT_SET) {

            for (size_t i=EXT_IDX; i<bit_array_size; ++i)
                bit_array[i] &= TYPE(0);

        }

        bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
        // bit_array[bit_array_size-1] &= TYPE(pow(2, nb_bits_ % WORD_LEN)-1);

    }

    template <typename t>
    inline void _XOR_ (const t& rhs) {
        cout << __FILE__ << ":" << __LINE__ ;
        cout << "Needs Implementation" << endl;
        exit(-1);
    }

    inline void _NOT_ () {
        for (size_t i=0; i<bit_array_size; ++i)
            bit_array[i] = ~bit_array[i];
    }

    inline void lshift (const size_t N) {
        if (N==0) return;
        TYPE truncated_bits = 0;

        for (size_t i=0; i<bit_array_size; ++i) {
            TYPE result = ( bit_array[i] << N ) | truncated_bits;
            truncated_bits = bit_array[i] >> (WORD_LEN - N);
            bit_array[i] = result;
        }

        bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
    }

    inline void rshift (const size_t N) {
        if (N == 0) return;

        TYPE truncated_bits = 0;
        bit_array[0] >>= N;

        for (size_t i=1; i<bit_array_size; ++i) {
            truncated_bits = (TYPE(-1) >> (WORD_LEN-N)) & bit_array[i];
            truncated_bits <<= (WORD_LEN-N);
            bit_array[i-1] |= truncated_bits;
            bit_array[i] >>= N;
        }

        bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
    }

    /*
     * Accessors
     */

    inline bool operator[] (const size_t i) const {
        return at(i);
    }

    inline bool at (const size_t i) const {
        return bit_array[i/WORD_LEN] & (1 << (i % WORD_LEN));
    }

    void print() const {
    	 for (size_t i=0; i<nb_bits_; ++i) {
			cout << this->at(nb_bits_-1-i) ;
		}
		printf("\n");
    }

    void value_print() const {
    	printf("[ ");
		for (size_t i=0; i<bit_array_size; ++i) {
			cout << bit_array[i] << " " ;
		}
		printf("]\n");
    }

    // Getters
    inline size_t size() const { return nb_bits_; }
    inline size_t len()  const { return size();   }


};

/*
 * Template specializations
 */

template <>
inline void bitset::_OR_ (const bitset& rhs) {
    size_t l = (nb_bits_ <= rhs.nb_bits_) ? bit_array_size : rhs.bit_array_size;

    for (size_t i=0; i<l; ++i) {
        bit_array[i] |= rhs.bit_array[i];
    }

    bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
}

template <>
inline void bitset::_OR_ (const double& rhs) {
    cout << __FILE__ << ":" << __LINE__ << ":function not implemented" << endl;
}

template <>
inline void bitset::_AND_ (const bitset& rhs) {
    size_t l = (nb_bits_ <= rhs.nb_bits_) ? bit_array_size : rhs.bit_array_size;

    for (size_t i=0; i<l; ++i) {
        bit_array[i] &= rhs.bit_array[i];
    }

    bit_array[bit_array_size-1] &= ~(TYPE(-1) << (nb_bits_ % WORD_LEN));
}

/* Check equality upto a certain number of bits for bitset as the RHS */
template <>
inline bool bitset::isEqualTo (const bitset& rhs, size_t len /* bits */) {
    for (size_t i=0; i<len; ++i) {
        if (at(i) != rhs[i])
            return false;
    }
    return true;
}

template <>
inline bool bitset::operator== (const bitset& rhs) const {
    const size_t NB_BITS_RHS = rhs.nb_bits_;

    if (NB_BITS_RHS != nb_bits_) return false;

    for (size_t i=0; i<bit_array_size; ++i) {
        if (bit_array[i] != rhs.bit_array[i])
            return false;
    }

    return true;
}

#endif
