# NANOGrav Fall Meeting 2016 Hackathon

This material is from a NANOGrav hackathon hosted at NCSA, University of Illinois at Urbana-Champaign. The jupyter notebook, assoicated scripts, and data, are meant for profiling the likelihood of our model. Users can see the structure of the likelihood, where the bottlenecks are, and trial some improvments. The contents are as follows:

* [likelihood.py](https://github.com/nanograv/pulsar_timing_school/blob/master/materials/nano_fall2016_hackathon/likelihood.py): A python script that contains the pulsar and model classes. The pulsar class processes the TOAs, and the model class contains the prior and likelihood functions.
* [utils.py](https://github.com/nanograv/pulsar_timing_school/blob/master/materials/nano_fall2016_hackathon/utils.py): A python script that containing various utility functions that aid the analysis.
* [partim](https://github.com/nanograv/pulsar_timing_school/blob/master/materials/nano_fall2016_hackathon/partim): A folder containing some sample data, in the form of ".par" and ".tim" files.
* [likelihood.ipynb](https://github.com/nanograv/pulsar_timing_school/blob/master/materials/nano_fall2016_hackathon/likelihood.ipynb): A jupyter notebook allowing the user to read-in the pulsar data, form the model, and profile (i.e. time) the likelihood function.
* [STaylor_NANOFall2016_Hackathon.pdf](https://github.com/nanograv/pulsar_timing_school/blob/master/materials/nano_fall2016_hackathon/STaylor_NANOFall2016_Hackathon.pdf): A presentation brriefly describing how a pulsar-timing array works, and how we form our likelihood.
