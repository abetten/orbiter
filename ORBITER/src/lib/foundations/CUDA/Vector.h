/*
 * Vector.h
 *
 *  Created on: Oct 25, 2018
 *      Author: sajeeb
 */

#ifndef VECTOR_
#define VECTOR_

#include <stdlib.h>
#include <iostream>
#include <iostream>
#include <ostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

using std::cout;
using std::endl;
using std::ostream;


template <typename T>

class Vector {
public:

	__device__
	__host__
	Vector() {}

	__host__
	Vector(size_t size)
	{
		size_ = size;
		vec_ = new T [size_]();
	}

	__device__
	__host__
	Vector(T v[], size_t size)
	{
		size_ = size;
		vec_ = v;
	//	vec_ = new unsigned int [size_];
	//	memcpy(vec_, v, sizeof(*vec_) * size_);
	}

	__host__
	Vector(const Vector& vec)
	{
		T* new_vec_ = new T [vec.size_];
		memcpy(new_vec_, vec.vec_, sizeof(new_vec_[0]) * vec.size_);
		delete [] vec_;
		vec_ = new_vec_;
	}

	//__device__
	__host__
	~Vector()
	{
		if (vec_gpu_)
		{
			cudaFree(vec_);
			vec_ = vec_gpu_;
			vec_gpu_ = NULL;
		}

		if (vec_ != NULL) {
			delete [] vec_;
			size_ = 0;
			vec_ = NULL;
		}
	}

	__device__
	__host__
	bool operator== (const Vector& V) const {
		if (V.size_ != size_) return false;
		for (size_t i=0; i<size_; ++i) {
			if (V(i) != vec_[i]) return false;
		}
		return true;
	}

	__device__
	__host__
	Vector& operator=(const Vector& v) {
		auto* new_vec_ = new unsigned int [v.size_];
		memcpy(new_vec_, v.vec_, sizeof(*new_vec_) * v.size_);
		delete [] vec_;
		vec_ = new_vec_;
		return *this;
	}

	__device__
	__host__
	bool operator< (const Vector& V) const {
		return num_rep_ < V.num_rep_;
	}

	__device__
	__host__
	const T& operator() (int i) const {
		if (i >= size_ || i < 0) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
		return vec_[i];
	}

	__device__
	__host__
	T& operator[] (size_t i) {
		if (i >= size_) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
		return vec_[i];
	}

	__device__
	__host__
	T& operator() (int i) {
	#ifndef __CUDACC__
		if (i >= size_ || i < 0) {
			printf("%s:%d:index out of range\n", __FILE__, __LINE__);
			exit(-1);
		}
	#else
		return vec_[i];
	#endif
	}

	__device__
	__host__
	size_t size() const { return size_; }

	__host__
	void print()
	{	cout << "[";
		for (size_t i=0; i<size_; ++i) {
			cout << this->operator()(i);
			if (i+1 != size_) cout << ", ";
		}
		cout << "]";
	}

	__host__
	__device__
	inline int num_rep() const {return num_rep_;}

	__host__
	void make_str_rep() {
		str = "[";
		for (size_t i=0; i<size_; ++i) {
			str += std::to_string(this->operator ()(i));
			if (i+1 != size_) str += ", ";
		}
		str += "]";
	}

	__device__
	__host__
	void
	InitializeOnGPU()
	{
		T* tmp = vec_;
		gpuErrchk( cudaMalloc(&vec_, sizeof(T)*size_) );
		gpuErrchk( cudaMemcpy(vec_, tmp, sizeof(T)*size_, cudaMemcpyHostToDevice) );
		vec_gpu_ = tmp;
	}

//	template <typename Mat, typename Vec>
//	__device__ __host__
//	friend Vec* cuda_dot(Mat& M, Vec& V);

//private:

	size_t size_ = 0;

	T* vec_ = NULL;
	T* vec_gpu_ = NULL;
	int num_rep_ = 0;

	std::string str = "";

};

#endif
