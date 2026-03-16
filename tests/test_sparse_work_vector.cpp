#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "mipx/sparse_work_vector.h"

namespace mipx {
namespace {

using Catch::Approx;

TEST_CASE("SparseWorkVector tracks touched entries uniquely", "[core]") {
    SparseWorkVector vec(8);

    vec.add(3, 1.5);
    vec.add(3, 2.0);
    vec.set(5, -4.0);

    CHECK(vec[3] == Approx(3.5));
    CHECK(vec[5] == Approx(-4.0));
    CHECK(vec.touched().size() == 2);
    CHECK(vec.touched()[0] == 3);
    CHECK(vec.touched()[1] == 5);
}

TEST_CASE("SparseWorkVector clear only resets touched entries", "[core]") {
    SparseWorkVector vec(6);

    vec.set(1, 2.0);
    vec.set(4, -3.0);
    vec.clear();

    CHECK(vec[1] == Approx(0.0));
    CHECK(vec[4] == Approx(0.0));
    CHECK(vec.touched().empty());

    vec.add(4, 7.0);
    CHECK(vec[4] == Approx(7.0));
    REQUIRE(vec.touched().size() == 1);
    CHECK(vec.touched()[0] == 4);
}

TEST_CASE("SparseWorkVector clearAll resets dense state", "[core]") {
    SparseWorkVector vec(5);

    vec.set(0, 1.0);
    vec.set(2, 2.0);
    vec.set(4, 3.0);
    vec.clearAll();

    for (Index i = 0; i < vec.size(); ++i) {
        CHECK(vec[i] == Approx(0.0));
    }
    CHECK(vec.touched().empty());
}

}  // namespace
}  // namespace mipx
