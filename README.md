# Pulsar Timing Data Analysis
This repository contains several exercises and explanatory Jupyter notebooks that detail basic frequentist and Bayesian statistical techniques, pulsar residual and noise modeling, and Gravitational Wave (GW) detection methods. The repository will be updated as new learning material is made available through student workshops and various busyweeks.

------------

## Materials

Here we host a collection of exercises, notes and, presentations from various schools and workshops dedicated to pulsar timing data analysis. Each page will contain a complete description of it's contents in the README.

* [Basic data analysis methods](https://github.com/nanograv/cit-busyweek/tree/master/materials/nano_studentworkshop): Here we host materials used in several IPTA and NANOGrav schools. In this repository we cover basic frequentist and Bayesian data analysis techniques as well as some basic GW signal modeling theory.

* [Pulsar timing data analysis basics](https://github.com/nanograv/cit-busyweek/tree/master/materials/pulsar_data_analysis): In this repository we host a Jupyter notebook with an extensive explanaion of the noise modeling and Bayesian data analysis formulation that is used in nearly all modern pulsar timing analysis. This is an excellent place to start for new students that are already familiar with linear algebra and basic Bayesian data analysis concepts.

* [Pulsar timing noise modeling and GW detection exercises](https://github.com/nanograv/cit-busyweek/tree/master/materials/cit-busyweek): In this repository we host material from an intensive school held at Caltech that begins with basic Markov Chain Monte-Carlo design and gradually builds a fully functional GW detection pipeline from scratch. If you are already familiar with Bayesian data analysis, are comfortable with the Python programming language, and want to get your hands dirty and learn a lot, then this is the place to go.

* [Pulsar timing code profiling](https://github.com/nanograv/cit-busyweek/tree/master/materials/nano_fall2016_hackathon): In this repository we host material from a NANOGrav hackathon held at NCSA at the University of Illinois at Urbana-Champaign in Fall 2016. The goal was to profile our likelihood and find out where the computational bottlenecks are. This profiling is done through a jupyter notebook, which calls various scripts and data, also within the folder.

------------

## Literature

Here we list several important papers in several different areas of pulsar timing data analysis. While this list is by no means exhaustive it gives a very good introduction to several areas of pulsar data analysis and modeling. The reading lists have been broken into broad topics:

### 1. Ph.D theses on gravitational wave and pulsar data analysis

The Ph.D theses here are all focused on various aspects of pulsar timing 
and GW detection and modeling. While they are, of course, very long and 
detailed they provide a complete basis of nearly all modern pulsar timing 
and GW analysis techniques and theory. 

* [Exploring the cosmos with Gravitational Waves](https://gwic.ligo.org/thesisprize/2014/taylor_thesis.pdf), S. R. Taylor (2014)
* [Searching for Gravitational Waves Using Pulsar Timing Arrays](http://dc.uwm.edu/cgi/viewcontent.cgi?article=1565&context=etd), J. A. Ellis (2014)
* [Gravitational Wave Astrophysics with Pulsar Timing Arrays](https://gwic.ligo.org/thesisprize/2014/mingarelli_thesis.pdf), C. M. F Mingarelli (2014)
* [Gravitational Wave detection & data analysis for Pulsar Timing Arrays](https://gwic.ligo.org/thesisprize/2011/van_haasteren_thesis.pdf), R. van Haasteren (2011)
* [A Comprehensive Bayesian Approach to Gravitational Wave Astronomy](http://scholarworks.montana.edu/xmlui/bitstream/handle/1/1740/LittenbergT0509.pdf?sequence=1), T. Littenberg (2009)
* [Long-Term Timing of Millisecond Pulsars and Gravitational Wave Detection](https://arxiv.org/pdf/0906.4246v1.pdf), J. P. W. Verbiest (2009)
* [Measuring the Gravitational Wave Background using Precision Pulsar Timing](http://www.cv.nrao.edu/~pdemores/thesis.pdf), P. B. Demorest (2007)


### 2. Data analysis and sampling techniques

These papers form an excellent basis for nearly all of the data Bayesian data analysis and samping methods that are currently used in pulsar timing.

* [Bayesian model selection without evidences: application to the dark energy equation-of-state](http://adsabs.harvard.edu/abs/2016MNRAS.455.2461H), S. Hee (2016)
* [New advances in the Gaussian-process approach to pulsar-timing data analysis](http://adsabs.harvard.edu/abs/2014PhRvD..90j4012V), R. van Haasteren & M. Vallisneri (2014)
* [MULTINEST: an efficient and robust Bayesian inference tool for cosmology and particle physics](http://adsabs.harvard.edu/abs/2009MNRAS.398.1601F), F. Feroz (2008)
* [Data analysis recipes: Fitting a model to data](http://adsabs.harvard.edu/abs/2010arXiv1008.4686H), D. W. Hogg (2010)
* [Bayesian approach to the detection problem in gravitational wave astronomy](http://adsabs.harvard.edu/abs/2009PhRvD..80f3007L), T. Litternberg & N. J. Cornish (2009)
* [Comment on "Tainted evidence: cosmological model selection versus fitting"](http://adsabs.harvard.edu/abs/2007astro.ph..3285L), A. R. Liddle et al (2007)


### 3. Pulsar noise modeling and analysis

Noise modeling in pulsar timing data analysis is extremely important and it permeates all other areas of analysis in pulsar timing. These papers are some of the most comprehensive covering physical and mathematical modeling of pulsar noise sources.

* [Transdimensional Bayesian approach to pulsar timing noise analysis](http://adsabs.harvard.edu/abs/2016PhRvD..93h4048E), J. A. Ellis & N. J. Cornish (2016)
* [The NANOGrav Nine-year Data Set: Observations, Arrival Time Measurements, and Analysis of 37 Millisecond Pulsars](http://adsabs.harvard.edu/abs/2015ApJ...813...65T), NANOGrav Collaboration (2015)
* [New advances in the Gaussian-process approach to pulsar-timing data analysis](http://adsabs.harvard.edu/abs/2014PhRvD..90j4012V), R. van Haasteren & M. Vallisneri (2014)
* [A Measurement Model for Precision Pulsar Timing](http://adsabs.harvard.edu/abs/2010arXiv1010.3785C), J. M. Cordes & R. M. Shannon (2010)

### 4. Non-linear Bayesian pulsar timing

Full Bayesian non-linear pulsar timing is a very new field and these two papers form the basis of that work to date.

* [Bayesian inference for pulsar-timing models](http://adsabs.harvard.edu/abs/2014MNRAS.440.1446V), S. J. Vigeland & M. Vallisneri (2014)
* [TEMPONEST: a Bayesian approach to pulsar timing analysis](http://adsabs.harvard.edu/abs/2014MNRAS.437.3004L), L. Lentati et al (2014)

### 5. Stochastic gravitational wave background analysis

* [The NANOGrav Nine-year Data Set: Limits on the Isotropic Stochastic Gravitational Wave Background](http://adsabs.harvard.edu/abs/2016ApJ...821...13A), NANOGrav Collaboration (2016)
* [Time-domain implementation of the optimal cross-correlation statistic for stochastic gravitational-wave background searches in pulsar timing data](http://adsabs.harvard.edu/abs/2015PhRvD..91d4048C), S. J. Chamberlin et al (2015)
* [Searching for anisotropic gravitational-wave backgrounds using pulsar timing arrays](http://adsabs.harvard.edu/abs/2013PhRvD..88h4001T), S. R. Taylor & J. R. Gair (2013)
* [Hyper-efficient model-independent Bayesian method for the analysis of pulsar timing data](http://adsabs.harvard.edu/abs/2013PhRvD..87j4021L), L. Lentati et al (2013)
* [On measuring the gravitational-wave background using Pulsar Timing Arrays](http://adsabs.harvard.edu/abs/2009MNRAS.395.1005V), R. van Haasteren et al (2009)


### 6. Continuous gravitational wave analysis

The papers here and references therein detail the modeling and detection techniques for continuous GW sources in both circular and eccentric orbits.

* [Detecting Eccentric Supermassive Black Hole Binaries with Pulsar Timing Arrays: Resolvable Source Strategies](http://adsabs.harvard.edu/abs/2016ApJ...817...70T), S. R. Taylor (2016)
* [Gravitational Waves from Individual Supermassive Black Hole Binaries in Circular Orbits: Limits from the North American Nanohertz Observatory for Gravitational Waves](http://adsabs.harvard.edu/abs/2014ApJ...794..141A), NANOGrav Collaboration (2014)
* [Optimal Strategies for Continuous Gravitational Wave Detection in Pulsar Timing Arrays](http://adsabs.harvard.edu/abs/2012ApJ...756..175E), J. A. Ellis et al (2012)
* [Response of Doppler spacecraft tracking to gravitational radiation](http://adsabs.harvard.edu/abs/1975GReGr...6..439E), F. B. Estabrook & H. D. Wahlquist (1975)

### 7. Gravitational wave burst analysis

Bursts are likely the least studied form of GWs in the pulsar timing frequency bands. This list forms a great basis for modeling and detection techniques for both un-modeled and bursts with memory (BWM).

* [NANOGrav Constraints on Gravitational Wave Bursts with Memory](http://adsabs.harvard.edu/abs/2015ApJ...810..150A), NANOGrav Collaboration (2015)
* [Searching for gravitational wave bursts via Bayesian nonparametric data analysis with pulsar timing arrays](http://adsabs.harvard.edu/abs/2014PhRvD..90b4020D), X. Deng (2014)
* [Detection, Localization, and Characterization of Gravitational Wave Bursts in a Pulsar Timing Array](http://adsabs.harvard.edu/abs/2010ApJ...718.1400F), L. S. Finn & A. N. Lommen (2010)
* [Gravitational-wave memory and pulsar timing arrays](http://adsabs.harvard.edu/abs/2010MNRAS.401.2372V), R. van Haasteren & Y. Levin (2010)

--------------

## Questions, Comments, Requests

If you have questions, comments or requests then submit an [issue](https://github.com/nanograv/cit-busyweek/issues) and one of us will try to answer it ASAP.

---------------

## Contributions?

If you have material of your own that you think would be useful then fork this repository, add your material and submit a pull request.

