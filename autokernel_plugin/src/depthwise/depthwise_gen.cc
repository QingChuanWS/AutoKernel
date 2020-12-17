#include "Halide.h"
#include "HalideBuffer.h"
using namespace Halide;
using Halide::Expr;
using Halide::TailStrategy;
using Halide::Func;
using Halide::Generator;
using Halide::Var;
using Halide::BoundaryConditions::constant_exterior;

class halide_depthwise:public Halide::Generator<halide_depthwise>{
public:
    Input<Buffer<float>> A{"A_", 1};
    Input<Buffer<float>> B{"B_", 2};
    Output<Buffer<float>> result{"r_", 1};

    void generate() {
        // The algorithm.
		Var i("i");
		RDom rv(0, 9);
		Func gemm("gemm");
		gemm(i) += A(rv) * B(i, rv);
		result(i) = gemm(i);

        //if(auto_schedule)
        //{
        //}

        //schedule
		Var ti[2]; 
		gemm.update()
			.split(i, ti[1], ti[0], 9, TailStrategy::GuardWithIf)
			.parallel(ti[1])
			.unroll(rv);
			//.vectorize(ti[0], 8, TailStrategy::GuardWithIf);
	}
};

HALIDE_REGISTER_GENERATOR(halide_depthwise, halide_depthwise)
