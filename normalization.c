#include "normalization.h" 
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


/*	Appies Min-max feature scaling to data
	n: number of items
	m: item size 							*/
void norm_minmax(value_t *data, int n, int m)
{
	value_t *min, *max;
	int i, j, item_size = m * sizeof(value_t);

	min = malloc(item_size);
	max = malloc(item_size);

	memcpy(max, data, item_size);
	memcpy(min, data, item_size);

	for (i = 0; i < m; i++) {
		for (j = m + i; j < n*m; j += m) {
			min[i] = (data[j] < min[i]) ? data[j] : min[i];
			max[i] = (data[j] > max[i]) ? data[j] : max[i];
		}
		max[i] -= min[i];
	}

	for (i = 0; i < m; i++) {
		if (max[i] == 0) {
			for (j = i; j < n*m; j += m)
				data[j] = 0.5;
			continue;
		}
		for (j = i; j < n*m; j += m)
			data[j] = (data[j] - min[i]) / max[i];
	}

	free(min);
	free(max);
}
