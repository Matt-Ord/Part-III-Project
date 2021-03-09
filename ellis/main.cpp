#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

template <int N>
struct wavefunction
{
    complex<double> wavefunction_vector[N];
};

template <int N>
wavefunction<N> propogate(
    wavefunction<N> wfn,
    int number_of_steps,
    double dt);

main()
{
    complex<double> initial_vector[5] = {
        1, 2, 3, 4, 5};

    wavefunction<5> wfn = {{1, 2, 3, 4, 5}};

    propogate(wfn, 5, 100);
};

template <int N>
wavefunction<N> propogate(wavefunction<N> wfn, int number_of_steps,
                          double dt)
{
    vector<complex<double>> current_vector = wfn.wavefunction_vector.;
    for (int i = 0; i < number_of_steps; i++)
    {
        std::cout << "pass \n";
    }
    // std::cout << wfn.wavefunction_vector;
    return wfn;
}