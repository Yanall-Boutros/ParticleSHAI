FROM tensorflow/tensorflow 
COPY ./bin/                  ./particleSHAI/bin/
COPY ./Examples/             ./particleSHAI/Examples/

RUN apt-get update
RUN apt-get -y install python3-dev bash libpng-dev g++ git  python3 python3-pip python-pip
RUN python3 -m pip uninstall -y pip &&  apt install python3-pip --reinstall
RUN pip install --user --upgrade pip
RUN pip3 install --user --upgrade pip
RUN python3 -m pip install --user numpy
RUN python3 -m pip install --user numpythia matplotlib pyjet tensorflow requests pandas scikit-hep scikit-learn uproot scipy pyhepmc-ng 
RUN git clone https://github.com/lukasheinrich/pylhe && cd pylhe && python3 setup.py install
RUN apt-get -y install vim
WORKDIR ./particleSHAI/Examples
