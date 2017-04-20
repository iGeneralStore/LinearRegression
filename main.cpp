#include <iostream>
#include <assert.h>

#define ABS(x) (((x) < 0) ? -(x) : (x))
#define TRAINIG_COUNT 100
#define NUM_EXAMPLES 100

using namespace std;

char FILE_PATH[50] = "size_price_examples.txt";

class MyData {
public:
	MyData() :
		count(0),
		size_min((float)1e8),
		size_max(-(float)1e8),
		price_min((float)1e8),
		price_max(-(float)1e8) 
	{}

	bool addData(float s, float p) {

		if(count > NUM_EXAMPLES) {
			cout << "DATA OVERFLOW (current limit: " << NUM_EXAMPLES << ")" << endl;
			return false;
		}

        if(s < size_min) size_min = s;
        if(s > size_max) size_max = s;
        if(p < price_min) price_min = p;
        if(p > price_max) price_max = p;

        size[count] = s;
        price[count] = p;

        count++;

		return true;
	}

	bool dataFeatureScaling(float from, float to) {

		assert(size_max != size_min);
		assert(price_max != price_min);
		assert(from < to);

		if(from >= to)
		    return false;

		float size_scale = to / (size_max - size_min) + from;
		float price_scale = to / (price_max - price_min) + from;

		for(int i=0;i<count;i++) {
		    size[i] -= size_min;
		    size[i] *= size_scale;
		    price[i] -= price_min;
		    price[i] *= price_scale;
		}

		return true;
	}

	const float* getSize() const { return size; }
	const float* getPrice() const { return price; }
	const int getCount() const { return count; }

	void printStatus() {
		for(int i=0;i<count;i++) 
			cout << (i+1) << " : (size, price) : (" << size[i] << ", " << price[i] << ")" << endl;
	}


private:
	int count;
	float size[NUM_EXAMPLES], price[NUM_EXAMPLES];
	float size_min, size_max;
	float price_min, price_max;
};

class Neuron {
public:
	Neuron(float weight, float bias)
		: w(weight), b(bias)
	{}

	float getSigma(float x) const { return w*x+b; }
	float getActFunc(float sigma) const { return sigma; }
	float getY(float f) const { return f; }

	float feedForward(const float* x) const {
		return getY(getActFunc(getSigma(*x)));
	}
	
	void feedForward(float* x, float* y) {
		*y = getY(getActFunc(getSigma(*x)));
	}

	float dE_dw(float* x=NULL, float* y=NULL, float* y_target=NULL) {
		float dSigma_dw = *x;

		// if you want function, then _df_dSigma = df_dSigma() 
		float df_dSigma = 1;
		float dy_df = 1;
		float dE_dy = *y - *y_target;

		return (dSigma_dw*df_dSigma*dy_df*dE_dy);
	}
	float dE_db(float* y=NULL, float* y_target=NULL) {
		float dSigma_db = 1;

		float df_dSigma = 1;
		float dy_df = 1;
		float dE_dy = *y - *y_target;

		return (dSigma_db*df_dSigma*dy_df*dE_dy);
	}

	void updateOneGDStep(const MyData& d, const float alpha=1.0f) {
		
		const float *arr_size = d.getSize();
		const float *arr_price = d.getPrice();

		float x,y,y_target;
		float _dE_dw = 0.0f;
		float _dE_db = 0.0f;
		for(int i=0;i<NUM_EXAMPLES;i++) {
			x = arr_size[i];
			y_target = arr_price[i];

			feedForward(&x,&y);
			
			_dE_dw += dE_dw(&x, &y, &y_target);	
			_dE_db += dE_db(&y, &y_target);	
		}


		w = w - alpha * _dE_dw / (float)NUM_EXAMPLES;
		b = b - alpha * _dE_db / (float)NUM_EXAMPLES;
	}

	float w,b;
	float input, output;
};


bool loadData(MyData& d, char* path) {
	FILE *ifp  = fopen(path, "r");

    if(ifp == NULL) {
    	cout << "DATA FILE NOT FOUND" << endl;
    	return false;
    }

    for(int i=0;i<NUM_EXAMPLES;i++) {
    	float size_input, price_input;
    	fscanf(ifp, "%f %f", &size_input, &price_input);

    	if( !d.addData(size_input, price_input) )
    	    return false;
    }
	
	fclose(ifp);

	return true;
}

float meanSquaredError(const Neuron& n, const MyData& d) {

	float total = 0;
	const float *arr_size = d.getSize();
	const float *arr_price = d.getPrice();
	const int d_count = d.getCount();

	for (int i=0;i<d_count;i++) {
		const float x = arr_size[i];
		const float y_target = arr_price[i];
		const float y = n.feedForward(&x);
		total += (y-y_target) * (y-y_target);
	}

	return total / (2 * d_count);
}

int main() {
	MyData d;
	Neuron n(0.0, 0.0);

	if(!loadData(d, FILE_PATH)) {
		cout << "FILE LOAD FAILED" << endl;
	}

	d.dataFeatureScaling(0.0f, 1.0f);

	d.printStatus();

	float MSE;
	float MSE_prev = meanSquaredError(n, d);
	float w_prev = n.w;
	float b_prev = n.b;

	int i=0;	
	for(;i<TRAINIG_COUNT;i++) {

		n.updateOneGDStep(d, 1.0f); 

		MSE = meanSquaredError(n, d);

		if(ABS(MSE) >= ABS(MSE_prev)) {
			MSE = MSE_prev;
			n.w = w_prev;
			n.b = b_prev;
			cout << "count " << i << " finished" << endl;
			break;
		}
	
		MSE_prev = MSE;	
		w_prev = n.w;
		b_prev = n.b;
	}

	cout << "final n : " << n.w << endl;
	cout << "final b : " << n.b << endl;

	return 0;
}
