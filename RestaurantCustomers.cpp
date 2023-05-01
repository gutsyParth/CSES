/*
    कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।
    मा कर्मफलहेतुर्भुर्मा ते संगोऽस्त्वकर्मणि ॥
अर्थ:- तेरा कर्म करने में अधिकार है इनके फलो में नही. तू कर्म के फल प्रति असक्त न हो या कर्म न करने के प्रति प्रेरित न हो.
*/

#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <climits>
using namespace std;
using namespace __gnu_pbds;

typedef tree<int, null_type, less<int>, rb_tree_tag,
             tree_order_statistics_node_update>
    new_data_set;

#define MOD 1000000007
#define MOD1 998244353
#define inf 1e18

#define int long long int
#define loop(i, a, b) for (int i = a; i < b; i++)
#define loopr(i, a, b) for (int i = a; i >= b; i--)
#define loops(i, a, b, step) for (int i = a; i < b; i += step)
#define looprs(i, a, b, step) for (int i = a; i >= b; i -= step)
#define ll long long
#define F first
#define MP make_pair
#define S second
#define pb push_back
#define ppb pop_back
#define si set<int>
#define usi unordered_set<int>
#define umsi unordered_multiset<int>
#define msi multiset<int>
#define vi vector<int>
#define vvi vector<vector<int>>
#define pii pair<int, int>
#define vpi vector<pii>
#define vpp vector<pair<int, pii>>
#define mii map<int, int>
#define umii unordered_map<int, int>
#define mpi map<pii, int>
#define umpi unordered_map<pii, int>
#define spi set<pii>
#define endl "\n"
#define sz(x) ((int)x.size())
#define all(p) p.begin(), p.end()
#define double long double
#define que_max priority_queue<int>
#define countSetBits(a) __builtin_popcount(a)
#define que_min priority_queue<int, vi, greater<int>>
#define bug(...) __f(#__VA_ARGS__, __VA_ARGS__)
#define print(a)          \
    for (auto x : a)      \
        cout << x << " "; \
    cout << endl
#define print1(a)    \
    for (auto x : a) \
    cout << x.F << " " << x.S << endl
#define print2(a, x, y)         \
    for (int i = x; i < y; i++) \
        cout << a[i] << " ";    \
    cout << endl

template <typename T1, typename T2>
istream &operator>>(istream &istream, pair<T1, T2> &p)
{
    return (istream >> p.first >> p.second);
}

template <typename T>
istream &operator>>(istream &istream, vector<T> &v)
{
    for (auto &it : v)
        cin >> it;
    return istream;
}

template <typename T1, typename T2>
ostream &operator<<(ostream &ostream, const pair<T1, T2> &p)
{
    return (ostream << p.first << " " << p.second);
}
template <typename T>
ostream &operator<<(ostream &ostream, const vector<T> &c)
{
    for (auto &it : c)
        cout << it << " ";
    return ostream;
}

inline int power(int a, int b)
{
    int x = 1;
    while (b)
    {
        if (b & 1)
            x *= a;
        a *= a;
        b >>= 1;
    }
    return x;
}
// for modular multiplicative inverse pass b as mod-2
int expo(int a, int b, int mod)
{
    int res = 1;
    while (b > 0)
    {
        if (b & 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        b = b >> 1;
    }
    return res;
}

void extendgcd(int a, int b, int *v)
{
    if (b == 0)
    {
        v[0] = 1;
        v[1] = 0;
        v[2] = a;
        return;
    }
    extendgcd(b, a % b, v);
    int x = v[1];
    v[1] = v[0] - v[1] * (a / b);
    v[0] = x;
    return;
}

// for non prime b
int mminv(int a, int b)
{
    int arr[3];
    extendgcd(a, b, arr);
    return arr[0];
}

int mminvprime(int a, int b) { return expo(a, b - 2, b); }

int combination(int n, int r, int m, int *fact, int *ifact)
{
    int val1 = fact[n];
    int val2 = ifact[n - r];
    int val3 = ifact[r];
    return (((val1 * val2) % m) * val3) % m;
}

vector<int> sieve(int n)
{
    int *arr = new int[n + 1]();
    vector<int> vect;
    for (int i = 2; i <= n; i++)
        if (arr[i] == 0)
        {
            vect.push_back(i);
            for (int j = 2 * i; j <= n; j += i)
                arr[j] = 1;
        }
    return vect;
}

int mod_add(int a, int b, int m)
{
    a = a % m;
    b = b % m;
    return (((a + b) % m) + m) % m;
}
int mod_mul(int a, int b, int m)
{
    a = a % m;
    b = b % m;
    return (((a * b) % m) + m) % m;
}
int mod_sub(int a, int b, int m)
{
    a = a % m;
    b = b % m;
    return (((a - b) % m) + m) % m;
}
// only for prime m
int mod_div(int a, int b, int m)
{
    a = a % m;
    b = b % m;
    return (mod_mul(a, mminvprime(b, m), m) + m) % m;
}

int isSubstring(string s2, string s1)
{
    if (s2.find(s1) != string::npos)
        return s2.find(s1);
    return -1;
}

void no() { printf("NO\n"); }
void yes() { printf("YES\n"); }

template <typename Arg1>
void __f(const char *name, Arg1 &&arg1) { cout << name << " : " << arg1 << endl; }
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args)
{
    const char *comma = strchr(names + 1, ',');
    cout.write(names, comma - names) << " : " << arg1 << " | ";
    __f(comma + 1, args...);
}

template <class T>
bool ckmin(T &a, const T &b) { return b < a ? a = b, 1 : 0; }
template <class T>
bool ckmax(T &a, const T &b) { return a < b ? a = b, 1 : 0; }

class union_find
{

public:
    int *pr;
    int *sz;

    union_find(int n)
    {
        pr = new int[n + 1];
        sz = new int[n + 1];

        for (int i = 0; i < n; ++i)
            pr[i] = i, sz[i] = 1;
    }

    int root(int i)
    {
        if (pr[i] == i)
            return i;

        return pr[i] = root(pr[pr[i]]);
    }

    int find(int i, int j)
    {
        return (root(i) == root(j));
    }

    int un(int i, int j)
    {
        int u = root(i);
        int v = root(j);

        if (u == v)
            return 0;

        if (sz[u] < sz[v])
            swap(u, v);

        pr[v] = u;
        sz[u] += sz[v];

        return 1;
    }
};

template <class T, class U>
// T -> node, U->update.
struct Lsegtree
{
    vector<T> st;
    vector<U> lazy;
    ll n;
    T identity_element;
    U identity_update;

    /*
        Definition of identity_element: the element I such that combine(x,I) = x
        for all x
        Definition of identity_update: the element I such that apply(x,I) = x
        for all x
    */

    Lsegtree(ll n, T identity_element, U identity_update)
    {
        this->n = n;
        this->identity_element = identity_element;
        this->identity_update = identity_update;
        st.assign(4 * n, identity_element);
        lazy.assign(4 * n, identity_update);
    }

    T combine(T l, T r)
    {
        // if this combine function is summing up two values then the identitiy element becomes 0

        // if this combine function is taking the maximum then the identitiy element becomes -inf

        // if this combine function is taking the minimum then the identitiy element becomes inf

        // change this function as required.
        T ans = (l + r);

        // max: T ans = max(l,r)
        // min : T ans = min(l,r)
        // gcd : T ans = gcd(l,r)
        return ans;
    }

    void buildUtil(ll v, ll tl, ll tr, vector<T> &a)
    {
        if (tl == tr)
        {
            st[v] = a[tl];
            return;
        }
        ll tm = (tl + tr) >> 1;
        buildUtil(2 * v + 1, tl, tm, a);
        buildUtil(2 * v + 2, tm + 1, tr, a);
        st[v] = combine(st[2 * v + 1], st[2 * v + 2]);
    }

    // change the following 2 functions, and you're more or less done.
    T apply(T curr, U upd, ll tl, ll tr)
    {
        // lets say we were assigning a value then the element that we choose as identity update is totally upto us so we can choose it to be as -1, if we have to add an element upto the entire range in that case the identity element would be zero

        T ans = (tr - tl + 1) * upd;
        // increment range by upd:
        // T ans = curr + (tr - tl + 1)*upd
        // query, take max, update, assign a value:
        // T ans = upd;

        return ans;
    }

    // this function combines two updates
    U combineUpdate(U old_upd, U new_upd, ll tl, ll tr)
    {
        // assigning a value to a range
        U ans = old_upd;
        ans = new_upd;

        // adding a value to a range
        // U ans = old_upd + new_upd

        return ans;
    }

    void push_down(ll v, ll tl, ll tr)
    {
        // for the below line to work, make sure the "==" operator is defined for U.
        if (lazy[v] == identity_update)
            return;
        st[v] = apply(st[v], lazy[v], tl, tr);
        if (2 * v + 1 <= 4 * n)
        {
            ll tm = (tl + tr) >> 1;
            lazy[2 * v + 1] = combineUpdate(lazy[2 * v + 1], lazy[v], tl, tm);
            lazy[2 * v + 2] = combineUpdate(lazy[2 * v + 2], lazy[v], tm + 1, tr);
        }
        lazy[v] = identity_update;
    }
    T queryUtil(ll v, ll tl, ll tr, ll l, ll r)
    {
        push_down(v, tl, tr);
        if (l > r)
            return identity_element;
        if (tr < l or tl > r)
        {
            return identity_element;
        }
        if (l <= tl and r >= tr)
        {
            return st[v];
        }
        ll tm = (tl + tr) >> 1;
        return combine(queryUtil(2 * v + 1, tl, tm, l, r), queryUtil(2 * v + 2, tm + 1, tr, l, r));
    }

    void updateUtil(ll v, ll tl, ll tr, ll l, ll r, U upd)
    {
        push_down(v, tl, tr);
        if (tr < l or tl > r)
            return;
        if (tl >= l and tr <= r)
        {
            lazy[v] = combineUpdate(lazy[v], upd, tl, tr);
            push_down(v, tl, tr);
        }
        else
        {
            ll tm = (tl + tr) >> 1;
            updateUtil(2 * v + 1, tl, tm, l, r, upd);
            updateUtil(2 * v + 2, tm + 1, tr, l, r, upd);
            st[v] = combine(st[2 * v + 1], st[2 * v + 2]);
        }
    }

    void build(vector<T> a)
    {
        assert((ll)a.size() == n);
        buildUtil(0, 0, n - 1, a);
    }
    T query(ll l, ll r)
    {
        return queryUtil(0, 0, n - 1, l, r);
    }
    void update(ll l, ll r, U upd)
    {
        updateUtil(0, 0, n - 1, l, r, upd);
    }
};

struct Fenwick
{
    vector<ll> t;
    void reset(int n)
    {
        t.assign(n + 1, 0);
    }
    void update(int p, ll v)
    {
        p++;
        for (; p < (int)t.size(); p += (p & (-p)))
            t[p] += v;
    }
    ll query(int r) // finds [1, r] sum
    {
        r++;
        ll sum = 0;
        for (; r; r -= (r & (-r)))
            sum += t[r];
        return sum;
    }
    ll query(int l, int r) // finds [l, r] sum
    {
        if (l == 0)
            return query(r);
        return query(r) - query(l - 1);
    }
};

struct DifferenceArray
{
    vector<int> D;
    void initializeDiffArray(vector<int> &A)
    {
        int n = A.size();

        // We use one extra space because
        // update(l, r, x) updates D[r+1]
        D.resize(n + 1);
        D[0] = A[0], D[n] = 0;
        for (int i = 1; i < n; i++)
            D[i] = A[i] - A[i - 1];
    }

    // Does range update
    void update(int l, int r, int x)
    {
        D[l] += x;
        D[r + 1] -= x;
    }

    // Prints updated Array
    void printArray(vector<int> &A)
    {
        for (int i = 0; i < A.size(); i++)
        {
            if (i == 0)
                A[i] = D[i];

            // Note that A[0] or D[0] decides
            // values of rest of the elements.
            else
                A[i] = D[i] + A[i - 1];

            cout << A[i] << " ";
        }
        cout << endl;
    }
};

template <typename T>
struct rmq
{
    vector<T> v;
    int n;
    static const int b = 30;
    vector<int> mask, t;

    int op(int x, int y)
    {
        return v[x] < v[y] ? x : y;
    }

    int lsb(int x)
    {
        return x & -x;
    }

    int msb_index(int x)
    {
        return __builtin_clz(1) - __builtin_clz(x);
    }

    int small(int r, int size = b)
    {

        int dist_from_r = msb_index(mask[r] & ((1 << size) - 1));

        return r - dist_from_r;
    }
    rmq(const vector<T> &v_) : v(v_), n(v.size()), mask(n), t(n)
    {
        int curr_mask = 0;
        for (int i = 0; i < n; i++)
        {

            curr_mask = (curr_mask << 1) & ((1 << b) - 1);

            while (curr_mask > 0 and op(i, i - msb_index(lsb(curr_mask))) == i)
            {

                curr_mask ^= lsb(curr_mask);
            }

            curr_mask |= 1;

            mask[i] = curr_mask;
        }

        for (int i = 0; i < n / b; i++)
            t[i] = small(b * i + b - 1);
        for (int j = 1; (1 << j) <= n / b; j++)
            for (int i = 0; i + (1 << j) <= n / b; i++)
                t[n / b * j + i] = op(t[n / b * (j - 1) + i], t[n / b * (j - 1) + i + (1 << (j - 1))]);
    }

    T query(int l, int r)
    {

        if (r - l + 1 <= b)
            return v[small(r, r - l + 1)];

        int ans = op(small(l + b - 1), small(r));

        int x = l / b + 1, y = r / b - 1;

        if (x <= y)
        {
            int j = msb_index(y - x + 1);
            ans = op(ans, op(t[n / b * j + x], t[n / b * j + y - (1 << j) + 1]));
        }

        return v[ans];
    }
};

// void dijkstra(int s, vector<int> &d)
// {
//     d = vector<int>(n, inf);
//     d[s] = 0;
//     set<pair<int, int>> st;
//     st.insert({d[s], s});
//     while (!st.empty())
//     {
//         int v = st.begin()->second;
//         st.erase(st.begin());
//         for (auto [to, w] : g[v])
//         {
//             if (d[to] > d[v] + w)
//             {
//                 auto it = st.find({d[to], to});
//                 if (it != st.end())
//                     st.erase(it);
//                 d[to] = d[v] + w;
//                 st.insert({d[to], to});
//             }
//         }
//     }
// }

// int fact[200001];
// int ifact[200001];

// void factorial()
// {
//     fact[0] = 1;
//     ifact[0] = 1;
//     for (int i = 1; i <= 200000; i++)
//     {
//         fact[i] = (fact[i - 1] * (i)) % MOD;
//         ifact[i] = mminvprime(fact[i], MOD);
//     }
// }

// int N = 200000;
// vvi primes(N);

// void allPrimeDivisors()
// {
//     for (int i = 2; i < N; i++)
//     {
//         if (primes[i].size() == 0)
//         {
//             for (int j = 1; i * j < N; j++)
//                 primes[i * j].push_back(i);
//         }
//     }
// }

struct sparseTableMin
{
    int n, k;
    vector<vector<int>> table;
    vector<int> logs;

    void init(int x)
    {
        n = x;
        logs.resize(n + 1);
        logs[1] = 0;
        for (int i = 2; i <= n; i++)
            logs[i] = logs[i / 2] + 1;
        k = *max_element(logs.begin(), logs.end());
        table.resize(k + 1, vector<int>(n + 1, 1e9));
    }

    int operation(int x, int y)
    {
        return std::min(x, y);
    }

    void build(vector<int> &arr)
    {
        for (int i = 0; i < n; i++)
            table[0][i] = arr[i];

        for (int j = 1; j <= k; j++)
        {
            for (int i = 0; i + (1 << j) <= n; i++)
                table[j][i] = operation(table[j - 1][i], table[j - 1][i + (1 << (j - 1))]);
        }
    }

    int query(int l, int r)
    {
        int j = logs[r - l + 1];
        int answer = operation(table[j][l], table[j][r - (1 << j) + 1]);
        return answer;
    }
};

struct sparseTableMax
{
    int n, k;
    vector<vector<int>> table;
    vector<int> logs;

    void init(int x)
    {
        n = x;
        logs.resize(n + 1);
        logs[1] = 0;
        for (int i = 2; i <= n; i++)
            logs[i] = logs[i / 2] + 1;
        k = *max_element(logs.begin(), logs.end());
        table.resize(k + 1, vector<int>(n + 1, 1e9));
    }

    int operation(int x, int y)
    {
        return std::max(x, y);
    }

    void build(vector<int> &arr)
    {
        for (int i = 0; i < n; i++)
            table[0][i] = arr[i];

        for (int j = 1; j <= k; j++)
        {
            for (int i = 0; i + (1 << j) <= n; i++)
                table[j][i] = operation(table[j - 1][i], table[j - 1][i + (1 << (j - 1))]);
        }
    }

    int query(int l, int r)
    {
        int j = logs[r - l + 1];
        int answer = operation(table[j][l], table[j][r - (1 << j) + 1]);
        return answer;
    }
};

struct Trie
{
    struct node
    {
        node *next[10];
        node()
        {
            for (int i = 0; i < 10; i++)
                next[i] = NULL;
        }
    };

    node root;

    void add(vector<int> &val)
    {
        node *temp = &root;
        for (auto ele : val)
        {
            if (temp->next[ele] == NULL)
                temp->next[ele] = new node();
            temp = temp->next[ele];
        }
    }

    int query(vector<int> &val)
    {
        node *temp = &root;
        int ans = 0;
        for (auto ele : val)
        {
            if (temp->next[ele] == NULL)
                break;
            ans++;
            temp = temp->next[ele];
        }
        return ans;
    }
};

void solve()
{
    vpi v;

    int n;
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        int x, y;
        cin >> x >> y;
        v.pb(MP(x, 0));
        v.pb(MP(y, 1));
    }

    sort(all(v));

    int cnt = 0, ans = 0;

    for (auto x : v)
    {
        if (x.S == 0)
        {
            cnt++;
        }
        else
        {
            cnt--;
        }
        ans = max(ans, cnt);
    }

    cout << ans;
}

int32_t main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif

    clock_t z = clock();

    // factorial();

    int t = 1;
    // cin >> t;
    while (t--)
        solve();

    cerr << "Run Time : " << ((double)(clock() - z) / CLOCKS_PER_SEC);

    return 0;
}