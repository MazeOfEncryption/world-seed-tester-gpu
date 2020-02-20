#include <stdio.h>
template <typename type>
struct vec2 {
	type x, y;
	bool operator==(const vec2<type> &rhs) {
		return this->x == rhs.x && this->y == rhs.y;
	}
};
int test () {return 1;}
int main () {
	vec2<int> testing = {test(), test()};
	if (testing == vec2<int>{1, 1}) {
		printf("True.\n");
	}
	printf("%d\n", testing.x);
	return 0;
}