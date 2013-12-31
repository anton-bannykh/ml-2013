#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <set>
#include <cmath>
#include <cstdio>

using namespace std;

const int TESTS = 5;
const int DEPTH = 20;
int maxx, maxy, var, train_size, test_size;

struct person
{
	int x, y;
	int value;
	person() {};
	person(int x, int y, int value): x(x), y(y), value(value) {};
};

bool is_field(int x, int y)
{
	return (0 <= x && x <= maxx && 0 <= y && y <= maxy);
}

template<typename T>
string tostring(const T & x)
{
	stringstream ss;
	ss << x;
	return ss.str();
}

int main()
{
	ofstream out("Result.txt");

	for (int test = 1; test <= TESTS; test++)
	{
        string t = tostring(test);
        cerr << "Test #" + t + "\n";	
        ifstream in("movielensfold" + t + ".txt");
        ifstream ans("movielensfold" + t + "ans.txt");
        	in >> var >> maxx >> maxy >> train_size >> test_size;
        maxx++, maxy++;

        cerr << "Read answer\n";
        vector<int> exact_ans(test_size);
        for (int i = 0; i < test_size; i++)
        	ans >> exact_ans[i];

        cerr << "Train\n";
        vector<vector<vector<int> > > data(vector<vector<vector<int> > >(maxx, vector<vector<int> >(maxy)));
        for (int i = 0; i < train_size; i++)
        {
        	person p;
        	in >> p.x >> p.y >> p.value;
        	data[p.x][p.y].push_back(p.value - 1); 	
        }
        
        cerr << "Testing\n";
        vector<int> my_ans(test_size);
        for (int i = 0; i < test_size; i++)
        {
        	int x, y;
        	in >> x >> y;
        	vector<pair<int, int> > cnt(var);
        	for (int j = 0; j < var; j++)
        		cnt[j].second = j + 1;
            
            for (int cur_x = max(0, x - DEPTH); cur_x < min(maxx, x + DEPTH); cur_x++)
            	for (int cur_y = max(0, y - DEPTH); cur_y < min(maxy, y + DEPTH); cur_y++)
            	{
            		if (!is_field(cur_x, cur_y)) continue;
            		for (size_t k = 0; k < data[cur_x][cur_y].size(); k++)
            			cnt[data[cur_x][cur_y][k]].first++;
            	}


            sort(cnt.begin(), cnt.end());
            
            my_ans[i] = cnt.back().second;
        }

        cerr << "Calc RMSE\n";
        double RMSE = 0.0;
        for (int i = 0; i < test_size; i++)
        	RMSE += fabs(my_ans[i] - exact_ans[i]);
        RMSE /= test_size;
        RMSE = sqrt(RMSE);
        out << "In test :" << test << " RMSE: " << fixed << RMSE << endl;
	}
}
