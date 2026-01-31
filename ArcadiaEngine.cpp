// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

#define TableSize 101

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// -------- 1. PlayerTable (Double Hashing) --------
class ConcretePlayerTable : public PlayerTable
{
private:
    struct Player
    {
        int key;
        string name;
        bool isOccupied;

        Player() : key(-1), name(""), isOccupied(false) {}
    };

    vector<Player> table;

    int hash1(int key)
    {
        return (key % TableSize + TableSize) % TableSize; // Ensure non-negative
    }

    int hash2(int key)
    {
        return 1 + ((key % (TableSize - 1) + (TableSize - 1)) % (TableSize - 1));
    }

    void deleteTable()
    {
        table.clear();
    }

public:
    ConcretePlayerTable()
    {
        table.resize(TableSize);
    }

    void insert(int playerID, string name)
    {
        int hashing1 = hash1(playerID);
        int hashing2 = hash2(playerID);

        for (int i = 0; i < TableSize; i++)
        {
            int idx = (hashing1 + i * hashing2) % TableSize;
            if (table[idx].isOccupied && table[idx].key == playerID)
            {
                table[idx].name = name;
                return;
            }

            if (!table[idx].isOccupied)
            {
                table[idx].key = playerID;
                table[idx].name = name;
                table[idx].isOccupied = true;
                return;
            }
        }

        // Table is full
        cout << "Table is Full";
    }

    string search(int playerID)
    {
        int hashing1 = hash1(playerID);
        int hashing2 = hash2(playerID);

        for (int i = 0; i < TableSize; i++)
        {
            int idx = (hashing1 + i * hashing2) % TableSize;
            if (!table[idx].isOccupied)
            {
                return ""; // Not found
            }

            if (table[idx].key == playerID)
            {
                return table[idx].name; // Found
            }
        }

        return "";
    }

    ~ConcretePlayerTable()
    {
        deleteTable();
    }
};

class ConcreteLeaderboard : public Leaderboard
{
private:
    struct Node
    {
        int id;
        int score;
        vector<Node*> forward;

        Node(int _id, int _score, int level) : id(_id), score(_score)
        {
            forward.resize(level + 1, nullptr);
        }
    };

    int MAX_LEVEL;
    float P;
    int level;
    Node* head;

    int randomLevel()
    {
        int lvl = 0;
        while (((double)rand() / RAND_MAX) < P && lvl < MAX_LEVEL)
            lvl++;
        return lvl;
    }

    bool comesBefore(int id1, int score1, int id2, int score2)
    {
        if (score1 != score2)
            return score1 > score2;
        return id1 < id2;
    }

public:
    ConcreteLeaderboard()
    {
        MAX_LEVEL = 64; // Increased for large inserts
        P = 0.5;
        level = 0;
        head = new Node(-1, INT_MAX, MAX_LEVEL);
        srand(1);
    }

    void addScore(int playerID, int score) override
    {
        removePlayer(playerID); // Remove existing player first

        vector<Node*> update(MAX_LEVEL + 1, nullptr);
        Node* curr = head;

        for (int i = level; i >= 0; i--)
        {
            while (curr->forward[i] != nullptr &&
                comesBefore(curr->forward[i]->id, curr->forward[i]->score,
                    playerID, score))
            {
                curr = curr->forward[i];
            }
            update[i] = curr;
        }

        int newLevel = randomLevel();
        if (newLevel > level)
        {
            for (int i = level + 1; i <= newLevel; i++)
                update[i] = head;
            level = newLevel;
        }

        Node* newNode = new Node(playerID, score, newLevel);
        for (int i = 0; i <= newLevel; i++)
        {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }
    }

    void removePlayer(int playerID) override
    {
        vector<Node*> update(MAX_LEVEL + 1, nullptr);
        Node* curr = head;

        // Traverse all levels to find predecessors of target
        for (int i = level; i >= 0; i--)
        {
            Node* temp = curr;
            while (temp->forward[i] != nullptr &&
                temp->forward[i]->id != playerID &&
                comesBefore(temp->forward[i]->id, temp->forward[i]->score, playerID, INT_MAX))
            {
                temp = temp->forward[i];
            }
            update[i] = temp;
        }

        Node* target = curr->forward[0];
        if (!target || target->id != playerID)
            return;

        for (int i = 0; i <= level; i++)
        {
            if (update[i]->forward[i] != target)
                break;
            update[i]->forward[i] = target->forward[i];
        }

        delete target;

        while (level > 0 && head->forward[level] == nullptr)
            level--;
    }

    vector<int> getTopN(int n) override
    {
        vector<int> result;
        Node* curr = head->forward[0];
        while (curr != nullptr && n-- > 0)
        {
            result.push_back(curr->id);
            curr = curr->forward[0];
        }
        return result;
    }

    ~ConcreteLeaderboard()
    {
        Node* curr = head;
        while (curr != nullptr)
        {
            Node* next = curr->forward[0];
            delete curr;
            curr = next;
        }
    }
};


// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree
{
private:
    enum Color { RED, BLACK };

    struct Node
    {
        int itemID;
        int price;
        Color color;
        Node* left, * right, * parent;
    };

    Node* root;
    Node* NIL;

    bool lessThan(Node* a, Node* b)
    {
        return a->price < b->price;
    }

    Color getColor(Node* n)
    {
        return (n == NIL) ? BLACK : n->color;
    }

    bool priceExists(int price)
    {
        Node* curr = root;
        while (curr != NIL)
        {
            if (price == curr->price)
                return true;
            if (price < curr->price)
                curr = curr->left;
            else
                curr = curr->right;
        }
        return false;
    }

    void leftRotate(Node* x)
    {
        Node* y = x->right;
        x->right = y->left;
        if (y->left != NIL)
            y->left->parent = x;

        y->parent = x->parent;
        if (x->parent == NIL)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;

        y->left = x;
        x->parent = y;
    }

    void rightRotate(Node* x)
    {
        Node* y = x->left;
        x->left = y->right;
        if (y->right != NIL)
            y->right->parent = x;

        y->parent = x->parent;
        if (x->parent == NIL)
            root = y;
        else if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;

        y->right = x;
        x->parent = y;
    }

    /* =================== INSERT FIXUP =================== */
    void insertFixup(Node* z)
    {
        while (z->parent->color == RED)
        {
            if (z->parent == z->parent->parent->left)
            {
                Node* y = z->parent->parent->right;

                if (y->color == RED)
                {            // Case 1
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }

                else
                {
                    if (z == z->parent->right)
                    { // Case 2
                        z = z->parent;
                        leftRotate(z);
                    }
                    z->parent->color = BLACK;    // Case 3
                    z->parent->parent->color = RED;
                    rightRotate(z->parent->parent);
                }
            }
            else
            {
                Node* y = z->parent->parent->left;

                if (y->color == RED)
                {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }

                else
                {
                    if (z == z->parent->left)
                    {
                        z = z->parent;
                        rightRotate(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    leftRotate(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }

    void transplant(Node* u, Node* v)
    {
        if (u->parent == NIL)
            root = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;

        v->parent = u->parent;
    }

    Node* treeMinimum(Node* x)
    {
        while (x->left != NIL)
            x = x->left;
        return x;
    }

    void deleteFixup(Node* x)
    {
        while (x != root && getColor(x) == BLACK)
        {
            if (x == x->parent->left)
            {
                Node* w = x->parent->right;

                // ----- Case 4: Sibling is red -----
                if (getColor(w) == RED)
                {
                    w->color = BLACK;
                    x->parent->color = RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }

                // ----- Case 3: Sibling black, both children black -----
                if (getColor(w->left) == BLACK && getColor(w->right) == BLACK)
                {
                    w->color = RED;
                    x = x->parent;
                }

                else
                {
                    // ----- Case 5: Sibling black, near child red, far child black -----
                    if (getColor(w->right) == BLACK) {
                        w->left->color = BLACK;
                        w->color = RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }
                    // ----- Case 6: Sibling black, far child red -----
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    leftRotate(x->parent);
                    x = root;
                }
            }

            else
            {
                Node* w = x->parent->left;

                // Case 4: Sibling red
                if (getColor(w) == RED)
                {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }

                // Case 3: Sibling black, both children black
                if (getColor(w->left) == BLACK && getColor(w->right) == BLACK)
                {
                    w->color = RED;
                    x = x->parent;
                }

                else
                {
                    // Case 5: Sibling black, near child red, far child black
                    if (getColor(w->left) == BLACK)
                    {
                        w->right->color = BLACK;
                        w->color = RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }
                    // Case 6: Sibling black, far child red
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rightRotate(x->parent);
                    x = root; // Done
                }
            }
        }
        // ----- Case 2: DB is root or Case 1: red leaf -----
        x->color = BLACK;
    }

    Node* findByID(Node* node, int id)
    {
        if (node == NIL)
            return NIL;
        if (node->itemID == id)
            return node;

        Node* left = findByID(node->left, id);
        if (left != NIL)
            return left;

        return findByID(node->right, id);
    }

    void inorder(Node* n)
    {
        if (n == NIL)
            return;
        inorder(n->left);
        cout << "Price=" << n->price
            << " Color=" << (n->color == RED ? "R" : "B") << endl;
        inorder(n->right);
    }

public:
    ConcreteAuctionTree()
    {
        NIL = new Node{ 0, 0, BLACK, nullptr, nullptr, nullptr };
        NIL->left = NIL->right = NIL->parent = NIL;
        root = NIL;
    }

    void insertItem(int id, int price) override
    {
        if (priceExists(price))
        {
            cout << "Insertion rejected: price " << price << " already exists.\n";
            return;
        }

        Node* z = new Node{ id, price, RED, NIL, NIL, NIL };
        Node* y = NIL;
        Node* x = root;

        while (x != NIL)
        {
            y = x;
            if (price < x->price)
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (y == NIL)
            root = z;
        else if (price < y->price)
            y->left = z;
        else
            y->right = z;

        insertFixup(z);
    }

    void deleteItem(int id) override
    {
        Node* z = findByID(root, id);
        if (z == NIL)
            return;

        Node* y = z;
        Node* x;
        Color yOriginal = y->color;

        if (z->left == NIL)
        {
            x = z->right;
            transplant(z, z->right);
        }

        else if (z->right == NIL)
        {
            x = z->left;
            transplant(z, z->left);
        }

        else
        {
            y = treeMinimum(z->right);
            yOriginal = y->color;
            x = y->right;

            if (y->parent == z)
                x->parent = y;

            else
            {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }

            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        if (yOriginal == BLACK)
            deleteFixup(x);

        delete z;
    }

};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins)
{
    // TODO: Implement partition problem using DP
    // Goal: Minimize |sum(subset1) - sum(subset2)|
    // Hint: Use subset sum DP to find closest sum to total/2
    int total = 0, target;
    for (int i = 0; i < n; i++)
    {
        total += coins[i];
    }

    target = total / 2;

    vector<bool> dp(target + 1, false);
    dp[0] = true;

    for (int i = 0; i < n; i++)
    {
        int coin = coins[i];
        for (int sum = target; sum >= coin; sum--)
        {
            if (dp[sum - coin] == true)
            {
                dp[sum] = true;
            }
        }
    }

    int best = 0;
    for (int sum = target; sum >= 0; sum--)
    {
        if (dp[sum] == true)
        {
            best = sum;
            break;
        }
    }
    return total - 2 * best;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items)
{
    // TODO: Implement 0/1 Knapsack using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity
    int n = items.size();

    vector<vector<int>> V(n + 1, vector<int>(capacity + 1, 0));
    vector<vector<int>> P(n + 1, vector<int>(capacity + 1, 0));

    // Fill DP Table
    for (int i = 1; i <= n; i++)
    {
        int wi = items[i - 1].first;
        int vi = items[i - 1].second;

        for (int j = 1; j <= capacity; j++)
        {
            if (wi <= j && vi + V[i - 1][j - wi] > V[i - 1][j])
            {
                V[i][j] = vi + V[i - 1][j - wi];
                P[i][j] = j - wi;
            }

            else
            {
                V[i][j] = V[i - 1][j];
                P[i][j] = j;
            }
        }
    }
    int maxValue = V[n][capacity];
    return maxValue;
}

long long InventorySystem::countStringPossibilities(string s)
{
    const int MOD = 1e9 + 7;
    int n = s.size();

    if (n == 0)
        return 1;

    for (char c : s)
    {
        if (c == 'w' || c == 'm')
            return 0;
    }

    vector<long long> dp(n + 1, 0);
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; i++)
    {
        dp[i] = dp[i - 1];

        if (s[i - 1] == s[i - 2] && (s[i - 1] == 'u' || s[i - 1] == 'n'))
        {
            dp[i] = (dp[i] + dp[i - 2]) % MOD;
        }
    }

    return dp[n];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest)
{
    if (source == dest)
        return true;

    vector<vector<int>> adj(n);
    for (auto& e : edges)
    {
        int u = e[0];
        int v = e[1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<bool> visited(n, false);
    queue<int> q;

    visited[source] = true;
    q.push(source);

    while (!q.empty())
    {
        int u = q.front();
        q.pop();

        for (int v : adj[u])
        {
            if (!visited[v])
            {
                if (v == dest)
                    return true;
                visited[v] = true;
                q.push(v);
            }
        }
    }

    return false;
}

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate, vector<vector<int>>& roadData)
{
    // TODO: Implement Minimum Spanning Tree (Kruskal's or Prim's)
    // roadData[i] = {u, v, goldCost, silverCost}
    // Total cost = goldCost * goldRate + silverCost * silverRate
    // Return -1 if graph cannot be fully connected

    // Implementation useing Kruskal's Algorithm
    vector<tuple<long long, int, int>> edges;

    for (int i = 0; i < m; i++)
    {
        int u = roadData[i][0];
        int v = roadData[i][1];
        long long goldCost = roadData[i][2];
        long long silverCost = roadData[i][3];

        long long totalCost = goldCost * goldRate + silverCost * silverRate;
        edges.push_back({ totalCost, u, v });
    }

    sort(edges.begin(), edges.end());

    // Union-Find (DSU)
    vector<int> parent(n), rank(n, 0);
    for (int i = 0; i < n; i++)
        parent[i] = i;

    // Iterative find with path compression
    auto findSet = [&](int x) {
        int root = x;
        while (parent[root] != root)
            root = parent[root];

        // Path compression
        while (parent[x] != x)
        {
            int p = parent[x];
            parent[x] = root;
            x = p;
        }
        return root;
        };

    auto unionSet = [&](int a, int b) {
        a = findSet(a);
        b = findSet(b);

        if (a == b)
            return false;

        if (rank[a] < rank[b])
            swap(a, b);

        parent[b] = a;
        if (rank[a] == rank[b])
            rank[a]++;

        return true;
        };

    long long totalCost = 0;
    int edgesUsed = 0;

    for (auto& e : edges)
    {
        long long cost;
        int u, v;
        tie(cost, u, v) = e;

        if (unionSet(u, v))
        {
            totalCost += cost;
            edgesUsed++;
            if (edgesUsed == n - 1)
                break;
        }
    }
    if (edgesUsed != n - 1)
        return -1;

    return totalCost;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads)
{
    const long long INF = LLONG_MAX / 4;

    vector<vector<long long>> dist(n, vector<long long>(n, INF));

    for (int i = 0; i < n; i++)
        dist[i][i] = 0;

    // Initialize edges
    for (auto& r : roads)
    {
        int u = r[0], v = r[1], w = r[2];
        dist[u][v] = min(dist[u][v], (long long)w);
        dist[v][u] = min(dist[v][u], (long long)w);
    }

    // Floyd–Warshall
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            if (dist[i][k] == INF)
                continue;

            for (int j = 0; j < n; j++)
            {
                if (dist[k][j] == INF)
                    continue;

                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }

    string total = "0";

    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (dist[i][j] == INF)
                continue;

            // convert distance to binary
            long long d = dist[i][j];
            string bin = "";
            while (d > 0)
            {
                bin.push_back(char('0' + (d & 1)));
                d >>= 1;
            }
            reverse(bin.begin(), bin.end());

            // ===== binary addition =====
            string result = "";
            int p = static_cast<int>(total.size()) - 1;
            int q = static_cast<int>(bin.size()) - 1;
            int carry = 0;

            while (p >= 0 || q >= 0 || carry)
            {
                int sum = carry;
                if (p >= 0)
                    sum += total[p--] - '0';

                if (q >= 0)
                    sum += bin[q--] - '0';

                result.push_back(char('0' + (sum & 1)));
                carry = sum >> 1;
            }

            reverse(result.begin(), result.end());
            total = result;
        }
    }
    return total;
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n)
{
    vector<int> freq(26, 0);

    for (char task : tasks)
        freq[task - 'A']++;

    int maxFreq = 0;
    for (int f : freq)
        maxFreq = max(maxFreq, f);

    int countMax = 0;
    for (int f : freq)
        if (f == maxFreq)
            countMax++;

    // Greedy formula
    int intervals = (maxFreq - 1) * (n + 1) + countMax;

    // return max between total tasks and calculated intervals
    return max(static_cast<int>(tasks.size()), intervals);
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable()
    {
        return new ConcretePlayerTable();
    }

    Leaderboard* createLeaderboard()
    {
        return new ConcreteLeaderboard();
    }

    AuctionTree* createAuctionTree()
    {
        return new ConcreteAuctionTree();
    }
}
