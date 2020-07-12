FROM tensorflow/tensorflow 
COPY ./bin/                  ./particleSHAI/bin/
COPY ./Examples/             ./particleSHAI/Examples/

RUN apt-get update
RUN apt-get -y install apt-utils python3-dev bash libpng-dev g++ git  python3 python3-pip python-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install numpythia matplotlib pyjet tensorflow requests pandas scikit-hep scikit-learn uproot scipy pyhepmc-ng 
RUN git clone https://github.com/lukasheinrich/pylhe && cd pylhe && python3 setup.py install
RUN apt-get -y install vim
WORKDIR ./particleSHAI/Examples
