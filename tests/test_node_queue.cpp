#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "mipx/bnb_node.h"

using namespace mipx;

TEST_CASE("NodeQueue basic push/pop/size/empty", "[node_queue]") {
    NodeQueue q;
    REQUIRE(q.empty());
    REQUIRE(q.size() == 0);

    BnbNode n;
    n.lp_bound = 1.0;
    q.push(n);
    REQUIRE(q.size() == 1);
    REQUIRE_FALSE(q.empty());

    auto popped = q.pop();
    REQUIRE(popped.id == 0);
    REQUIRE(q.empty());
}

TEST_CASE("NodeQueue assigns unique ids", "[node_queue]") {
    NodeQueue q;
    BnbNode n;
    n.lp_bound = 0.0;
    q.push(n);
    q.push(n);
    q.push(n);

    auto a = q.pop();
    auto b = q.pop();
    auto c = q.pop();
    // All ids should be distinct.
    REQUIRE(a.id != b.id);
    REQUIRE(b.id != c.id);
    REQUIRE(a.id != c.id);
}

TEST_CASE("BestFirst pops in order of increasing lp_bound", "[node_queue]") {
    NodeQueue q(NodePolicy::BestFirst);

    BnbNode n1, n2, n3;
    n1.lp_bound = 5.0;
    n2.lp_bound = 1.0;
    n3.lp_bound = 3.0;

    q.push(n1);
    q.push(n2);
    q.push(n3);

    REQUIRE(q.pop().lp_bound == 1.0);
    REQUIRE(q.pop().lp_bound == 3.0);
    REQUIRE(q.pop().lp_bound == 5.0);
}

TEST_CASE("BestFirst tie-breaking prefers deeper node", "[node_queue]") {
    NodeQueue q(NodePolicy::BestFirst);

    BnbNode shallow, deep;
    shallow.lp_bound = 2.0;
    shallow.depth = 1;
    deep.lp_bound = 2.0;
    deep.depth = 5;

    q.push(shallow);
    q.push(deep);

    auto first = q.pop();
    REQUIRE(first.depth == 5);
}

TEST_CASE("DepthFirst pops deepest node first", "[node_queue]") {
    NodeQueue q(NodePolicy::DepthFirst);

    BnbNode n1, n2, n3;
    n1.depth = 1;
    n1.lp_bound = 10.0;
    n2.depth = 3;
    n2.lp_bound = 20.0;
    n3.depth = 2;
    n3.lp_bound = 15.0;

    q.push(n1);
    q.push(n2);
    q.push(n3);

    REQUIRE(q.pop().depth == 3);
    REQUIRE(q.pop().depth == 2);
    REQUIRE(q.pop().depth == 1);
}

TEST_CASE("DepthFirst tie-breaking prefers better bound", "[node_queue]") {
    NodeQueue q(NodePolicy::DepthFirst);

    BnbNode a, b;
    a.depth = 3;
    a.lp_bound = 10.0;
    b.depth = 3;
    b.lp_bound = 5.0;

    q.push(a);
    q.push(b);

    REQUIRE(q.pop().lp_bound == 5.0);
}

TEST_CASE("BestEstimate pops by smallest estimate", "[node_queue]") {
    NodeQueue q(NodePolicy::BestEstimate);

    BnbNode a, b, c;
    a.lp_bound = 1.0;
    a.estimate = 10.0;
    a.depth = 1;
    b.lp_bound = 3.0;
    b.estimate = 4.0;
    b.depth = 2;
    c.lp_bound = 2.0;
    c.estimate = 6.0;
    c.depth = 3;

    q.push(a);
    q.push(b);
    q.push(c);

    REQUIRE(q.pop().estimate == 4.0);
    REQUIRE(q.pop().estimate == 6.0);
    REQUIRE(q.pop().estimate == 10.0);
}

TEST_CASE("DepthBiased favors deeper nodes at near-equal bounds", "[node_queue]") {
    NodeQueue q(NodePolicy::DepthBiased);

    BnbNode shallow, deep;
    shallow.lp_bound = 5.0;
    shallow.depth = 1;
    deep.lp_bound = 5.0;
    deep.depth = 9;

    q.push(shallow);
    q.push(deep);

    auto first = q.pop();
    REQUIRE(first.depth == 9);
}

TEST_CASE("bestBound returns minimum bound in queue", "[node_queue]") {
    NodeQueue q;

    BnbNode n1, n2, n3;
    n1.lp_bound = 5.0;
    n2.lp_bound = 1.0;
    n3.lp_bound = 3.0;

    q.push(n1);
    q.push(n2);
    q.push(n3);

    REQUIRE(q.bestBound() == 1.0);
}

TEST_CASE("bestBound returns kInf for empty queue", "[node_queue]") {
    NodeQueue q;
    REQUIRE(q.bestBound() == kInf);
}

TEST_CASE("prune removes nodes with bound >= cutoff", "[node_queue]") {
    NodeQueue q;

    BnbNode n1, n2, n3, n4;
    n1.lp_bound = 1.0;
    n2.lp_bound = 5.0;
    n3.lp_bound = 3.0;
    n4.lp_bound = 7.0;

    q.push(n1);
    q.push(n2);
    q.push(n3);
    q.push(n4);

    q.prune(5.0);
    REQUIRE(q.size() == 2);
    // Remaining nodes should have bound < 5.0.
    REQUIRE(q.bestBound() == 1.0);

    auto a = q.pop();
    auto b = q.pop();
    REQUIRE((a.lp_bound < 5.0 && b.lp_bound < 5.0));
}

TEST_CASE("Single node push/pop roundtrip", "[node_queue]") {
    NodeQueue q;
    BnbNode n;
    n.lp_bound = 42.0;
    n.depth = 7;
    n.parent_id = 3;
    q.push(n);

    auto result = q.pop();
    REQUIRE(result.lp_bound == 42.0);
    REQUIRE(result.depth == 7);
    REQUIRE(result.parent_id == 3);
    REQUIRE(result.id >= 0);
}

TEST_CASE("takeAll and replaceAll preserve queue contents", "[node_queue]") {
    NodeQueue q;
    BnbNode a, b;
    a.lp_bound = 3.0;
    b.lp_bound = 1.0;
    q.push(a);
    q.push(b);

    auto nodes = q.takeAll();
    REQUIRE(q.empty());
    REQUIRE(nodes.size() == 2);

    q.replaceAll(std::move(nodes));
    REQUIRE(q.size() == 2);
    REQUIRE(q.bestBound() == 1.0);
}
