#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include "ArcadiaEngine.h"
#include <algorithm>
using namespace std;
/*
    BY : Folka
*/

// ==========================================
// FACTORY FUNCTIONS (LINKING)
// ==========================================
// These link to the functions at the bottom of your .cpp file
extern "C" {
    PlayerTable* createPlayerTable();
    Leaderboard* createLeaderboard();
    AuctionTree* createAuctionTree();
}

// ==========================================
// TEST UTILITIES
// ==========================================
class StudentTestRunner {
    int count = 0;
    int passed = 0;
    int failed = 0;

public:
    void runTest(string testName, bool condition) {
        count++;
        cout << "TEST: " << left << setw(80) << testName;
        if (condition) {
            cout << "[ PASS ]";
            passed++;
        }
        else {
            cout << "[ FAIL ]";
            failed++;
        }
        cout << endl;
    }

    void printSummary() {
        cout << "\n==========================================" << endl;
        cout << "SUMMARY: Passed: " << passed << " | Failed: " << failed << endl;
        cout << "==========================================" << endl;
        cout << "TOTAL TESTS: " << count << endl;
        if (failed == 0) {
            cout << "Great job! All basic scenarios passed." << endl;
            cout << "Now make sure to handle edge cases (empty inputs, collisions, etc.)!" << endl;
        }
        else {
            cout << "Some basic tests failed. Check your logic against the PDF examples." << endl;
        }
    }
};

StudentTestRunner runner;

// ==========================================
// PART A: DATA STRUCTURES
// ==========================================

void test_PartA_DataStructures() {
    cout << "\n--- Part A: Data Structures ---" << endl;

    // 1. PlayerTable (Double Hashing)
    // Requirement: Basic Insert and Search
    PlayerTable* table = createPlayerTable();
    runner.runTest("PlayerTable: Insert 'Alice' and Search", [&]() {
        table->insert(101, "Alice");
        return table->search(101) == "Alice";
        }());

    runner.runTest("PlayerTable: Insert 100 player and Search", [&]() {
        for (int i = 0; i < 100; i++) {
            table->insert(i, string(i + 1, 'a'));
        }
        bool ok = true;
        for (int i = 0; i < 100; i++) {
            ok &= table->search(i) == string(i + 1, 'a');
        }
        return ok;
        }());

    delete table;

    // 2. Leaderboard (Skip List)
    Leaderboard* board = createLeaderboard();

    // Test A: Basic High Score
    runner.runTest("Leaderboard: Add Scores & Get Top 1", [&]() {
        board->addScore(1, 100);
        board->addScore(2, 200); // 2 is Leader
        vector<int> top = board->getTopN(1);
        return (!top.empty() && top[0] == 2);
        }());

    // Test B: Tie-Breaking Visual Example (Crucial!)
    // PDF Visual Example: Player A (ID 10) 500pts, Player B (ID 20) 500pts.
    // Correct Order: ID 10 then ID 20.
    runner.runTest("Leaderboard: Tie-Break (ID 10 before ID 20)", [&]() {
        board->addScore(10, 500);
        board->addScore(20, 500);
        vector<int> top = board->getTopN(2);
        // We expect {10, 20} NOT {20, 10}
        if (top.size() < 2) return false;
        return (top[0] == 10 && top[1] == 20);
        }());

    runner.runTest("Leaderboard: remove ID 10", [&]() {
        board->removePlayer(10);
        board->addScore(10, 0);
        board->removePlayer(10);
        vector<int> top = board->getTopN(2);
        if (top.size() < 2) return false;
        return (top[0] == 20 && top[1] == 2);
        }());
    runner.runTest("Leaderboard: Top-N more than players", [&]() {
        vector<int> top = board->getTopN(10);
        if (top.size() != 3) return false;
        return (top[0] == 20 && top[1] == 2 && top[2] == 1);
        }());

    runner.runTest("Leaderboard: adding 5 players", [&]() {
        board->addScore(1, 500);
        board->addScore(2, 500);
        board->addScore(3, 500);
        board->addScore(4, 500);
        board->addScore(5, 500);
        return true;
        }());

    runner.runTest("Leaderboard: insert 1e5 players", [&]() {
        int n = 1e3;
        for (int i = 1; i <= n; i++) {
            board->removePlayer(i);
        }
        auto score = [](int i) -> int {
            return i / 2 + 1;
            };
        vector< int > exp;
        for (int i = 1; i <= n; i++) {
            board->addScore(i, score(i));
            exp.push_back(i);
        }
        sort(exp.begin(), exp.end(), [&](int i, int j) {
            if (score(i) != score(j)) {
                return score(i) > score(j);
            }
            return i < j;
            });
        vector< int > top = board->getTopN(n);
        for (int i = 1; i <= n; i++) {
            board->removePlayer(i);
        }
        return top == exp;
        }());

    delete board;

    // 3. AuctionTree (Red-Black Tree)
    // Requirement: Insert items without crashing
    AuctionTree* tree = createAuctionTree();
    runner.runTest("AuctionTree: Insert Items", [&]() {
        tree->insertItem(1, 100);
        tree->insertItem(2, 50);
        tree->insertItem(3, 50);
        tree->insertItem(5, 54);
        tree->deleteItem(1);
        tree->deleteItem(2);
        tree->deleteItem(3);
        tree->deleteItem(4);
        tree->deleteItem(3);
        return true; // Pass if no crash
        }());
    runner.runTest("AuctionTree: Insert 1e6 Items", [&]() {
        int n = 1e6;
        for (int i = 10; i <= n; i++) {
            tree->insertItem(i, i * 5);
        }
        for (int i = 10; i <= n; i++) {
            tree->deleteItem(i);
        }
        return true; // Pass if no crash
        }());
    delete tree;
}

// ==========================================
// PART B: INVENTORY SYSTEM
// ==========================================

void test_PartB_Inventory() {
    cout << "\n--- Part B: Inventory System ---" << endl;

    // 1. Loot Splitting (Partition)
    // PDF Example: coins = {1, 2, 4} -> Best split {4} vs {1,2} -> Diff 1
    runner.runTest("LootSplit: {1, 2, 4} -> Diff 1", [&]() {
        vector<int> coins = { 1, 2, 4 };
        return InventorySystem::optimizeLootSplit(3, coins) == 1;
        }());

    runner.runTest("LootSplit: {2, 2} -> Diff 0", [&]() {
        vector<int> coins = { 2, 2 };
        return InventorySystem::optimizeLootSplit(2, coins) == 0;
        }());

    runner.runTest("LootSplit: {1, 5, 11, 5} -> Diff 0", [&]() {
        vector<int> coins = { 1, 5, 11, 5 };
        return InventorySystem::optimizeLootSplit(4, coins) == 0;
        }());


    // 2. Inventory Packer (Knapsack)
    // PDF Example: Cap=10, Items={{1,10}, {2,20}, {3,30}}. All fit. Value=60.
    runner.runTest("Knapsack: Cap 10, All Fit -> Value 60", [&]() {
        vector<pair<int, int>> items = { {1, 10}, {2, 20}, {3, 30} };
        return InventorySystem::maximizeCarryValue(10, items) == 60;
        }());

    runner.runTest("Knapsack: Cap 3 -> Value 40", [&]() {
        vector<pair<int, int>> items = { {1, 10}, {2, 15}, {3, 40} };
        return InventorySystem::maximizeCarryValue(3, items) == 40;
        }());

    runner.runTest("Knapsack: Cap 5 -> Value 55", [&]() {
        vector<pair<int, int>> items = { {1, 10}, {2, 15}, {3, 40} };
        return InventorySystem::maximizeCarryValue(5, items) == 55;
        }());

    runner.runTest("Knapsack: Cap 10, All Fit -> Value 60", [&]() {
        vector<pair<int, int>> items = { {1, 10}, {2, 20}, {3, 30} };
        return InventorySystem::maximizeCarryValue(10, items) == 60;
        }());


    // 3. Chat Autocorrect (String DP)
    // PDF Example: "uu" -> "uu" or "w" -> 2 possibilities
    runner.runTest("ChatDecorder: 'uu' -> 2 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uu") == 2;
        }());

    runner.runTest("ChatDecorder: '' -> 1 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("") == 1;
        }());

    runner.runTest("ChatDecorder: 'm' -> 0 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("m") == 0;
        }());

    runner.runTest("ChatDecoder: 'uunn' -> 4 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uunn") == 4;
        }());

    runner.runTest("ChatDecoder: 'uuu' -> 3 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uuu") == 3;
        }());

    runner.runTest("ChatDecoder: 'uuuu' -> 5 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uuuu") == 5;
        }());

    runner.runTest("ChatDecoder: 'uuauu' -> 4 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uuauu") == 4;
        }());

    runner.runTest("ChatDecoder: 'uunnuu' -> 8 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uunnuu") == 8;
        }());

    runner.runTest("ChatDecoder: 'uuunnnuuu' -> 27 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uuunnnuuu") == 27;
        }());

    runner.runTest("ChatDecoder: 'uuunnnnuuu' -> 45 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities("uuunnnnuuu") == 45;
        }());

    runner.runTest("ChatDecoder: 1000 'u's -> 107579939 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities(string(1000, 'u')) == 107579939;
        }());

    runner.runTest("ChatDecoder: 1000 'u's + 1000 'n's -> 194229620 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities(string(1000, 'u') + string(1000, 'n')) == 194229620;
        }());

    runner.runTest("ChatDecoder: 1000 'u's + 1000 'n's + 1000 'u's -> 525326710 Possibilities", [&]() {
        return InventorySystem::countStringPossibilities(string(1000, 'u') + string(1000, 'n') + string(1000, 'u')) == 525326710;
        }());

}

// ==========================================
// PART C: WORLD NAVIGATOR
// ==========================================

void test_PartC_Navigator() {
    cout << "\n--- Part C: World Navigator ---" << endl;

    // 1. Safe Passage (Path Exists)
    // PDF Example: 0-1, 1-2. Path 0->2 exists.
    runner.runTest("PathExists: 0->1->2 -> True", [&]() {
        vector<vector<int>> edges = { {0, 1}, {1, 2} };
        return WorldNavigator::pathExists(3, edges, 0, 2) == true;
        }());

    runner.runTest("PathExists: 0->1, 2->3 -> False", [&]() {
        vector<vector<int>> edges = { {0, 1}, {2, 3} };
        return WorldNavigator::pathExists(4, edges, 0, 3) == false;
        }());

    runner.runTest("PathExists: Single Node 0 -> True", [&]() {
        vector<vector<int>> edges = {};
        return WorldNavigator::pathExists(1, edges, 0, 0) == true;
        }());


    // 2. The Bribe (MST)
    // PDF Example: 3 Nodes. Roads: {0,1,10}, {1,2,5}, {0,2,20}. Rate=1.
    // MST should pick 10 and 5. Total 15.
    runner.runTest("MinBribeCost: Triangle Graph -> Cost 15", [&]() {
        vector<vector<int>> roads = {
            {0, 1, 10, 0},
            {1, 2, 5, 0},
            {0, 2, 20, 0}
        };
        // n=3, m=3, goldRate=1, silverRate=1
        return WorldNavigator::minBribeCost(3, 3, 1, 1, roads) == 15;
        }());

    runner.runTest("MinBribeCost: Large Graph -> Cost 4000000000000000000", [&]() {
        vector<vector<int>> roads = {
            {0, 1, 1000000000, 1000000000},
            {1, 2, 1000000000, 1000000000},
            {0, 2, 1000000000, 1000000000}
        };
        // n=3, m=3, goldRate=1e9, silverRate=1e9
        return WorldNavigator::minBribeCost(3, 3, 1000000000, 1000000000, roads) == 4000000000000000000ll;
        }());


    // 3. Teleporter (Binary Sum APSP)
    // PDF Example: 0-1 (1), 1-2 (2). Distances: 1, 2, 3. Sum=6 -> "110"
    runner.runTest("BinarySum: Line Graph -> '110'", [&]() {
        vector<vector<int>> roads = {
            {0, 1, 1},
            {1, 2, 2}
        };
        return WorldNavigator::sumMinDistancesBinary(3, roads) == "110";
        }());

    runner.runTest("BinarySum: Single Edge -> '100'", [&]() {
        vector<vector<int>> roads = {
            {0, 1, 4}
        };
        return WorldNavigator::sumMinDistancesBinary(2, roads) == "100";
        }());

    runner.runTest("BinarySum: Two Paths from 0 -> '10100'", [&]() {
        vector<vector<int>> roads = {
            {0, 1, 2},
            {0, 2, 8}
        };
        return WorldNavigator::sumMinDistancesBinary(3, roads) == "10100";
        }());
}

// ==========================================
// PART D: SERVER KERNEL
// ==========================================

void test_PartD_Kernel() {
    cout << "\n--- Part D: Server Kernel ---" << endl;

    // 1. Task Scheduler
    // PDF Example: Tasks={A, A, B}, n=2.
    // Order: A -> B -> idle -> A. Total intervals: 4.
    runner.runTest("Scheduler: {A, A, B}, n=2 -> 4 Intervals", [&]() {
        vector<char> tasks = { 'A', 'A', 'B' };
        return ServerKernel::minIntervals(tasks, 2) == 4;
        }());

    runner.runTest("Scheduler: {A, A, B}, n=2 -> 4 Intervals", [&]() {
        vector<char> tasks = { 'A', 'A', 'B' };
        return ServerKernel::minIntervals(tasks, 2) == 4;
        }());

    runner.runTest("Scheduler: {A, A, B}, n=2 -> 4 Intervals", [&]() {
        vector<char> tasks = { 'A', 'A', 'B' };
        return ServerKernel::minIntervals(tasks, 2) == 4;
        }());

    runner.runTest("Scheduler: {A, A, A}, n=2 -> 7 Intervals", [&]() {
        vector<char> tasks = { 'A', 'A', 'A' };
        return ServerKernel::minIntervals(tasks, 2) == 7;
        }());

    runner.runTest("Scheduler: {A, B, C}, n=2 -> 3 Intervals", [&]() {
        vector<char> tasks = { 'A', 'B', 'C' };
        return ServerKernel::minIntervals(tasks, 2) == 3;
        }());

    runner.runTest("Scheduler: {A, A, A, B, B, B}, n=2 -> 8 Intervals", [&]() {
        vector<char> tasks = { 'A', 'A', 'A', 'B', 'B', 'B' };
        return ServerKernel::minIntervals(tasks, 2) == 8;
        }());

}

int main() {
    cout << "Arcadia Engine - Student Happy Path Tests" << endl;
    cout << "-----------------------------------------" << endl;

    test_PartA_DataStructures();
    test_PartB_Inventory();
    test_PartC_Navigator();
    test_PartD_Kernel();

    runner.printSummary();

    return 0;
}
