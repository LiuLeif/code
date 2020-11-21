#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

int main( ) {

    USING_NAMESPACE_ACADO

    DifferentialState        s, v, m      ;
    Control                  u          ;
    Parameter                T          ;
    DifferentialEquation     f( 0.0, T );

    OCP ocp( 0.0, T );
    ocp.minimizeMayerTerm( T );

    f << dot(s) == v;
    f << dot(v) == (u-0.2*v*v)/m;  // 0.2*v*v 是空气摩擦
    f << dot(m) == -0.01*u*u;

    ocp.subjectTo( f);
    ocp.subjectTo( AT_START, s ==  0.0 );
    ocp.subjectTo( AT_START, v ==  0.0 );
    ocp.subjectTo( AT_START, m ==  1.0 );

    ocp.subjectTo( AT_END, s == 10.0 );
    ocp.subjectTo( AT_END, v ==  0.0 );

    ocp.subjectTo( -0.1 <= v <=  1.7   );
    ocp.subjectTo( -1.1 <= u <=  1.1   );
    ocp.subjectTo(  5.0 <= T <= 15.0   );

    GnuplotWindow window;
    window.addSubplot( s, "THE DISTANCE s"      );
    window.addSubplot( v, "THE VELOCITY v"      );
    window.addSubplot( m, "THE MASS m"          );
    window.addSubplot( u, "THE CONTROL INPUT u" );

    OptimizationAlgorithm algorithm(ocp);
    algorithm << window;
    algorithm.solve();

    return 0;
}
