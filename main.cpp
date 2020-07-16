/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <complex>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

using namespace std;

//typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m = 1638400;    // DO NOT CHANGE!!
const int K = 100000;    // DO NOT CHANGE!!

double
logDataVSPrior(const double *dat_r, const double *dat_i, const double *pri_r, const double *pri_i, const double *ctf,
               const double *sigRcp, const double disturb0);

int main(int argc, char *argv[]) {
    double *dat_r = new double[m];
    double *dat_i = new double[m];
    double *pri_r = new double[m];
    double *pri_i = new double[m];
    double *ctf = new double[m];
    double *sigRcp = new double[m];
    double *disturb = new double[K];
    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    /***************************
     * Read data from input.dat
     * *************************/
    ifstream fin;

    fin.open("input.dat");
    if (!fin.is_open()) {
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i = 0;
    while (!fin.eof()) {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        dat_r[i] = dat0;
        dat_i[i] = dat1;
        pri_r[i] = pri0;
        pri_i[i] = pri1;
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        i++;
        if (i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if (!fin.is_open()) {
        cout << "Error opening file K.dat" << endl;
        exit(1);
    }
    i = 0;
    while (!fin.eof()) {
        fin >> disturb[i];
        i++;
        if (i == K) break;
    }
    fin.close();

    /***************************
     * main computation is here
     * ************************/
    auto startTime = Clock::now();

    ofstream fout;
    fout.open("result.dat");
    if (!fout.is_open()) {
        cout << "Error opening file for result" << endl;
        exit(1);
    }
    double result[K];
#   pragma omp parallel for schedule(dynamic)
    for (unsigned int t = 0; t < K; t++) {
        result[t] = logDataVSPrior(dat_r,dat_i, pri_r,pri_i ,ctf, sigRcp,disturb[t]);
    }
    for (unsigned int t = 0; t < K; t++)
        fout << t + 1 << ": " << result[t] << endl;
    fout.close();

    auto endTime = Clock::now();

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    delete[] dat_i;
    delete[] dat_r;
    delete[] pri_i;
    delete[] pri_r;

    delete[] ctf;
    delete[] sigRcp;
    delete[] disturb;
    return EXIT_SUCCESS;
}

double
logDataVSPrior(const double *dat_r, const double *dat_i, const double *pri_r, const double *pri_i, const double *ctf,
               const double *sigRcp, const double disturb0) {
    double result = 0.0;
    double *res = new double[8];
    __m512d a, b, c, d, e, f;
    for (int i = 0; i <  m / 8; i++) {
        //一次加载8个double
        a = _mm512_load_pd(dat_r + i * 8);
        b = _mm512_load_pd(dat_i + i * 8);
        c = _mm512_load_pd(pri_r + i * 8);
        d = _mm512_load_pd(pri_i + i * 8);
        e = _mm512_load_pd(ctf + i * 8);
        f = _mm512_load_pd(sigRcp + i * 8);
        //相应的运算操作

        //1.ce,de
        c = _mm512_mul_pd(c, e);
        d = _mm512_mul_pd(d, e);
        //2.a-c,b-d
        a = _mm512_sub_pd(a, c);
        b = _mm512_sub_pd(b, d);
        //3.a2,b2
        a = _mm512_exp2a23_pd(a);
        b = _mm512_exp2a23_pd(b);
        //4.a+b
        a = _mm512_add_pd(a, b);
        //5.a*f
        a = _mm512_mul_pd(a, f);

        //store
        _mm512_storeu_pd(res, a);
        for (int j = 0; j < 8; j++) {
            result += res[j];
        }
    }
    return result * disturb0;
}
