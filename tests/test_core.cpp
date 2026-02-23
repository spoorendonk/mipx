#include <catch2/catch_test_macros.hpp>

#include "mipx/core.h"

TEST_CASE("Core type aliases", "[core]") {
    static_assert(std::is_same_v<mipx::Real, double>);
    static_assert(std::is_same_v<mipx::Int, int>);
    static_assert(std::is_same_v<mipx::Index, int>);
}

TEST_CASE("Status enum values are distinct", "[core]") {
    REQUIRE(mipx::Status::Optimal != mipx::Status::Infeasible);
    REQUIRE(mipx::Status::Infeasible != mipx::Status::Unbounded);
    REQUIRE(mipx::Status::Unbounded != mipx::Status::IterLimit);
    REQUIRE(mipx::Status::IterLimit != mipx::Status::TimeLimit);
    REQUIRE(mipx::Status::TimeLimit != mipx::Status::NodeLimit);
    REQUIRE(mipx::Status::NodeLimit != mipx::Status::Error);
}

TEST_CASE("Sense enum", "[core]") {
    REQUIRE(mipx::Sense::Minimize != mipx::Sense::Maximize);
}

TEST_CASE("VarType enum", "[core]") {
    REQUIRE(mipx::VarType::Continuous != mipx::VarType::Integer);
    REQUIRE(mipx::VarType::Integer != mipx::VarType::Binary);
    REQUIRE(mipx::VarType::Continuous != mipx::VarType::Binary);
}

TEST_CASE("ConstraintSense enum", "[core]") {
    REQUIRE(mipx::ConstraintSense::Leq != mipx::ConstraintSense::Geq);
    REQUIRE(mipx::ConstraintSense::Geq != mipx::ConstraintSense::Eq);
    REQUIRE(mipx::ConstraintSense::Eq != mipx::ConstraintSense::Range);
}
