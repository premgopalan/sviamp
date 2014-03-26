Installation
------------

Required libraries: gsl, gslblas, pthread

On Linux/Unix run

 ./configure
 make; make install

On Mac OS, the location of the required gsl, gslblas and pthread
libraries may need to be specified:

 ./configure LDFLAGS="-L/opt/local/lib" CPPFLAGS="-I/opt/local/include"
 make; make install

The binary 'gaprec' will be installed in /usr/local/bin unless a
different prefix is provided to configure. (See INSTALL.)

Citation
--------

P. Gopalan, C. Wang, and D.M. Blei. Modeling Overlapping Communities
with Node Popularities. To appear in NIPS 2013.

(more readme to follow)

[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/premgopalan/sviamp/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

